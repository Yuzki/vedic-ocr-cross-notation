"""Evaluate initial checkpoint: PACLIC 2026 v2 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from scripts.evaluation.prediction_metrics import evaluate_prediction_file
from scripts.evaluation.prediction_io import (
    write_metrics_json,
    write_predictions_csv,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

#: The run-axis label written into every prediction row's ``condition`` column so
#: the study CSV/metrics are self-describing (distinct from the fine-tuned cells).
CONDITION = "initial_checkpoint_zero_shot"

#: (view, domain) pairs. ``view`` names the output file + the study ``view`` axis;
#: ``domain`` selects the ``data.hf.<domain>.<split>`` sub-block. Both views use the
#: ``test`` split.
VIEWS: tuple[tuple[str, str], ...] = (("rv_test", "rv"), ("ms_test", "ms"))
SPLIT = "test"

#: The HF dataset column used as the per-row ``sample_id`` (the RV/MS Sanskrit2003
#: schema carries ``file_name``; study revision-freeze manifest).
SAMPLE_ID_COLUMN = "file_name"

#: Generation-config fields snapshotted into the manifest so the decode policy is
#: auditable from the artifact alone.
DECODE_FIELDS: tuple[str, ...] = (
    "num_beams",
    "length_penalty",
    "no_repeat_ngram_size",
    "early_stopping",
    "max_length",
    "decoder_start_token_id",
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
)


def build_run_config(
    *,
    model_id: str,
    model_revision: Optional[str],
    dataset_id: str,
    dataset_revision: Optional[str],
    image_column: str = "image",
    text_column: str = "text",
    rv_test_csv: Optional[str] = None,
    ms_test_csv: Optional[str] = None,
) -> DictConfig:
    """Assemble the minimal Hydra-shaped config the reused loaders expect.

    ``model.base`` / ``model.revision`` feed
    :func:`~scripts.training.trocr.experiment_accent_ratio.load_model_and_processor`;
    the ``data.hf`` block (with the ``rv``/``ms`` split maps) feeds
    :func:`~scripts.training.trocr.dataset.load_ocr_dataset_from_config` for
    ``domain="rv"|"ms"``, ``split="test"``. ``token=None`` keeps the load
    token-free (offline from the shared cache).
    """
    # A local-CSV corpus has no Hub revision to pin, so the two paths are
    # mutually exclusive rather than layered: pass the CSVs and the loader reads
    # them directly; pass neither and it resolves the Hub pin as before. The v2
    # corpus is local-only, which is why this branch exists.
    if rv_test_csv or ms_test_csv:
        if not (rv_test_csv and ms_test_csv):
            raise ValueError("give both --rv-test-csv and --ms-test-csv, or neither")
        cfg = OmegaConf.create(
            {
                "model": {"base": model_id, "revision": model_revision},
                "data": {
                    "source": "csv",
                    "rv": {"test": rv_test_csv},
                    "ms": {"test": ms_test_csv},
                },
            }
        )
        assert isinstance(cfg, DictConfig)
        return cfg

    cfg = OmegaConf.create(
        {
            "model": {"base": model_id, "revision": model_revision},
            "data": {
                "source": "hf",
                "hf": {
                    "dataset_id": dataset_id,
                    "revision": dataset_revision,
                    "token": None,
                    "image_column": image_column,
                    "text_column": text_column,
                    "rv": {"test": "rv_test"},
                    "ms": {"test": "ms_test"},
                },
            },
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def sample_ids_for(dataset: Any, expected: int) -> tuple[list[str], str]:
    """Per-row sample ids for the study CSV, aligned to dataloader (row) order.

    :func:`evaluate_on_dataset` iterates a ``shuffle=False`` ``DataLoader``, so
    predictions/references come back in HF row order — the same order as the
    backing ``dataset.hf_dataset``. Returns ``(ids, source)`` where ``source`` is
    ``"file_name"`` when the canonical id column is present and length-matches,
    else ``"index"`` (a safe positional fallback that still yields a stable id).
    """
    hf = getattr(dataset, "hf_dataset", None)
    if hf is not None:
        columns = list(getattr(hf, "column_names", []) or [])
        if SAMPLE_ID_COLUMN in columns:
            ids = [str(x) for x in hf[SAMPLE_ID_COLUMN]]
            if len(ids) == expected:
                return ids, SAMPLE_ID_COLUMN
    return [str(i) for i in range(expected)], "index"


def resolved_decode_config(model: Any) -> dict[str, Any]:
    """Snapshot the model's effective ``generation_config`` decode fields.

    Read *after* ``apply_generation_policy`` so it reflects the policy-resolved
    ``decoder_start_token_id`` alongside the preserved saved knobs (``num_beams``
    etc.). ``eos_token_id`` may be a list; JSON handles it as-is.
    """
    gen_cfg = getattr(model, "generation_config", None)
    return {field: getattr(gen_cfg, field, None) for field in DECODE_FIELDS}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _headline_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """The scalar metrics surfaced in the manifest summary (full set in the JSON)."""
    keys = (
        "num_samples",
        "num_scored",
        "cer",
        "wer",
        "chrf",
        "exact_match",
        "exact_match_count",
        "accent_precision",
        "accent_recall",
        "accent_f1",
        "accent_stripped_cer",
    )
    return {k: payload.get(k) for k in keys}


def _sha256_file(path: Path) -> str:
    """Content digest of a corpus CSV, so the manifest pins bytes not a filename."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    model_id: str,
    model_revision: Optional[str],
    dataset_id: Optional[str],
    dataset_revision: Optional[str],
    # Absent for a Hub run, which has no local CSVs to pin — hence a default
    # rather than a required argument.
    local_csvs: Optional[dict[str, str]] = None,
    decode: dict[str, Any],
    start_token_resolution: str,
    eval_settings: dict[str, Any],
    environment: dict[str, Any],
    views: list[dict[str, Any]],
    generated_date: str,
) -> dict[str, Any]:
    """Pure assembly of the provenance manifest from already-computed pieces."""
    return {
        "issue": 146,
        "parent_issue": 122,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "title": (
            "Zero-shot pre-evaluation of the fixed pre-adapted INITIAL checkpoint "
            "on RV/MS test (no fine-tuning)"
        ),
        "generated_date": generated_date,
        "condition": CONDITION,
        "fine_tuning": CONDITION != "initial_checkpoint_zero_shot",
        "note": (
            "Direction differences here are an initialization-bias observation of "
            "the fixed pre-fine-tuning checkpoint; per the study they are NOT "
            "attributed to the pre-adaptation training corpus (out of scope)."
        ),
        "model": {"hub_repo_id": model_id, "hub_revision": model_revision},
        "dataset": {
            # A local-CSV run has no Hub id. Recording only a null there would
            # leave the bundle unable to say WHICH corpus it evaluated — the whole
            # point of a v2 re-run — so the CSV paths and their sha256 are the pin.
            "hub_repo_id": dataset_id,
            "local_csvs": local_csvs,
            "hub_revision": dataset_revision,
            "splits": [view["view"] for view in views],
        },
        "decode": {
            "source": "model generation_config.json (saved), preserve-first policy",
            "start_token_resolution": start_token_resolution,
            "on_policy": True,
            "config": decode,
            "matches_accent_ratio_recipe": (
                "Verified identical to conf/experiment.yaml generation block "
                "(num_beams=4, length_penalty=2.0, no_repeat_ngram_size=3, "
                "early_stopping=true, max_length=256, decoder_start_token_id=2/EOS)"
                " → decode-matched with the fine-tuned endpoints."
            ),
        },
        "eval": eval_settings,
        "environment": environment,
        "views": views,
        "reused_code": {
            "model_loader": "scripts.training.trocr.experiment_accent_ratio.load_model_and_processor",
            "generation_policy": "scripts.training.trocr.generation_policy.apply_generation_policy",
            "dataset_loader": "scripts.training.trocr.dataset.load_ocr_dataset_from_config",
            "eval_fn": "scripts.training.trocr.experiment_accent_ratio.evaluate_on_dataset",
            "prediction_writer": "scripts.evaluation.prediction_io.write_predictions_csv",
            "metrics_evaluator": "scripts.evaluation.prediction_metrics.evaluate_prediction_file (study)",
        },
        "fine_tuned_comparison": {
            "study": "paclic-accent-ratio-multiseed (study)",
            "rv_endpoint": "rv_100_ms_0 (RV-only fine-tuned) on rv_test",
            "ms_endpoint": "rv_0_ms_100 (MS-only fine-tuned) on ms_test",
            "results_ref": (
                "results/"
                "paclic-accent-ratio-multiseed/results.md"
            ),
        },
    }


def evaluate_one_view(
    model: Any,
    processor: Any,
    tokenizer: Any,
    run_cfg: DictConfig,
    *,
    view: str,
    domain: str,
    out_dir: Path,
    model_id: str,
    model_revision: Optional[str],
    dataset_id: str,
    dataset_revision: Optional[str],
    batch_size: int,
    num_workers: int,
    input_scheme: str,
    model_scheme: str,
    max_target_length: int,
    use_chrf: bool,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Load one test view, generate, write the study CSV, recompute metrics.

    Heavy repo functions are imported here (not at module top) so the pure helpers
    + manifest assembly stay importable/testable without torch/transformers.
    ``max_samples`` (smoke only) truncates the split before generation.
    """
    from scripts.training.trocr.dataset import load_ocr_dataset_from_config
    from scripts.training.trocr.experiment_accent_ratio import evaluate_on_dataset

    dataset = load_ocr_dataset_from_config(
        run_cfg.data,
        split=SPLIT,
        domain=domain,
        root_dir=ROOT_DIR,
        processor=processor,
        tokenizer=tokenizer,
        input_scheme=input_scheme,
        model_scheme=model_scheme,
        max_target_length=max_target_length,
        max_samples=max_samples,
        accent_system=domain,
    )

    result = evaluate_on_dataset(
        model,
        processor,
        tokenizer,
        dataset,
        view,
        batch_size=batch_size,
        num_workers=num_workers,
        use_chrf=use_chrf,
        model_scheme=model_scheme,
    )

    predictions = result["predictions"]
    references = result["references"]
    sample_ids, sample_id_source = sample_ids_for(dataset, len(predictions))

    csv_path = out_dir / "predictions" / f"{view}.csv"
    write_predictions_csv(
        csv_path,
        references=references,
        predictions=predictions,
        sample_ids=sample_ids,
        condition=CONDITION,
        view=view,
        split=SPLIT,
    )

    # Re-compute ALL metrics inference-free from the just-written CSV (study), so
    # the committed CSV regenerates the metric JSON exactly (the study cond. 2).
    payload = evaluate_prediction_file(csv_path, use_chrf=use_chrf)
    metrics = {
        "issue": 146,
        "condition": CONDITION,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        **payload,
    }
    metrics_path = out_dir / "metrics" / f"{view}.json"
    write_metrics_json(metrics_path, metrics)

    headline = _headline_metrics(payload)
    print(
        f"[initial-checkpoint-eval] {view}: n={headline['num_samples']} "
        f"CER={headline['cer']:.4f} WER={headline['wer']:.4f} "
        f"chrF={headline['chrf']:.2f} EM={headline['exact_match']:.4f} "
        f"accentF1={headline['accent_f1']:.4f} "
        f"accent_stripped_CER={headline['accent_stripped_cer']:.4f} "
        f"(sample_id={sample_id_source})",
        flush=True,
    )

    return {
        "view": view,
        "domain": domain,
        "split": SPLIT,
        "n": len(predictions),
        "sample_id_source": sample_id_source,
        "predictions_csv": str(csv_path.relative_to(out_dir)),
        "predictions_sha256": _sha256(csv_path),
        "metrics_json": str(metrics_path.relative_to(out_dir)),
        "metrics": headline,
    }


def capture_environment(device: str) -> dict[str, Any]:
    """Record the offline/token posture + runtime versions for the manifest."""
    import torch
    import transformers

    return {
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        # A visible assertion that NO token was in the environment for this load.
        "token_env_present": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python": sys.version.split()[0],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load the pinned checkpoint, evaluate every view zero-shot, write artifacts."""
    import torch

    from scripts.training.trocr.experiment_accent_ratio import (
        load_model_and_processor,
    )
    from scripts.training.trocr.generation_policy import apply_generation_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[initial-checkpoint-eval] device={device}", flush=True)
    print(
        f"[initial-checkpoint-eval] model={args.model_id}@{args.model_revision}",
        flush=True,
    )
    print(
        f"[initial-checkpoint-eval] dataset={args.dataset_id}@{args.dataset_revision}",
        flush=True,
    )

    if not args.dataset_id and not (args.rv_test_csv and args.ms_test_csv):
        parser_error = (
            "give --dataset-id, or both --rv-test-csv and --ms-test-csv for a "
            "corpus with no Hub publication"
        )
        raise SystemExit(f"evaluate_initial_checkpoint: {parser_error}")

    run_cfg = build_run_config(
        model_id=args.model_id,
        model_revision=args.model_revision,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        image_column=args.image_column,
        text_column=args.text_column,
        rv_test_csv=args.rv_test_csv,
        ms_test_csv=args.ms_test_csv,
    )

    model, processor, tokenizer = load_model_and_processor(run_cfg)
    model = model.to(device)
    model.eval()

    # No decode overrides → preserve the checkpoint's SAVED generation_config;
    # resolve the start token to the saved value (EOS=2, on-policy). This is the
    # decode setting the study asks for ("use the saved generation configuration").
    resolution = apply_generation_policy(model, tokenizer)
    print(
        f"[initial-checkpoint-eval] generation-policy: {resolution.describe()}",
        flush=True,
    )
    decode = resolved_decode_config(model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views: list[dict[str, Any]] = []
    for view, domain in VIEWS:
        views.append(
            evaluate_one_view(
                model,
                processor,
                tokenizer,
                run_cfg,
                view=view,
                domain=domain,
                out_dir=out_dir,
                model_id=args.model_id,
                model_revision=args.model_revision,
                dataset_id=args.dataset_id,
                dataset_revision=args.dataset_revision,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                input_scheme=args.input_scheme,
                model_scheme=args.model_scheme,
                max_target_length=args.max_target_length,
                use_chrf=not args.no_chrf,
                max_samples=args.max_samples,
            )
        )

    local_csvs: Optional[dict[str, str]] = None
    if args.rv_test_csv and args.ms_test_csv:
        local_csvs = {
            f"{view}_test": f"{path} sha256:{_sha256_file(Path(path))}"
            for view, path in (("rv", args.rv_test_csv), ("ms", args.ms_test_csv))
        }

    manifest = build_manifest(
        model_id=args.model_id,
        model_revision=args.model_revision,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        local_csvs=local_csvs,
        decode=decode,
        start_token_resolution=resolution.describe(),
        eval_settings={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "input_scheme": args.input_scheme,
            "model_scheme": args.model_scheme,
            "max_target_length": args.max_target_length,
            "use_chrf": not args.no_chrf,
        },
        environment=capture_environment(device),
        views=views,
        generated_date=args.generated_date,
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[initial-checkpoint-eval] manifest → {manifest_path}", flush=True)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-tuned", action="store_true", help="Label evaluation of a trained checkpoint.")
    parser.add_argument("--model-id", default="yzk/trocr-large-printed-sanskrit-iso-preadapt")
    parser.add_argument("--model-revision", default="6dc79fc60cb0f8149b234478b46062c7b8ac732f")
    # Not required when the corpus is read from local CSVs: a local-only corpus
    # has no Hub id, and demanding a placeholder would put a false pin in the
    # manifest. Validated after parsing instead.
    parser.add_argument("--dataset-id", default="yzk/veda-ocr-rv-ms-noto-v2")
    parser.add_argument("--dataset-revision", default="7225f9adb86af6d63316b1e5a60b2b8de46e320c")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--rv-test-csv",
        default=None,
        help=(
            "Local RV test CSV. Give this with --ms-test-csv to read a corpus that "
            "has no Hub publication (the v2 corpus); the Hub pin is then unused."
        ),
    )
    parser.add_argument("--ms-test-csv", default=None)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT_DIR / "outputs" / "eval" / "paclic_initial_checkpoint_eval"),
        help="Where predictions/, metrics/, and manifest.json are written.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    # The pinned model + RV/MS dataset are ISO-15919; convert_text is then a
    # near-no-op NFC normalize (accent metrics require the ISO representation).
    parser.add_argument("--input-scheme", default="iso")
    parser.add_argument("--model-scheme", default="iso")
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--no-chrf", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Truncate each split to N rows (SMOKE only; default: full 578).",
    )
    parser.add_argument(
        "--generated-date",
        default=None,
        help="Manifest date (YYYY-MM-DD); defaults to today (UTC).",
    )
    args = parser.parse_args(argv)
    if args.generated_date is None:
        from datetime import date

        args.generated_date = date.today().isoformat()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    global CONDITION
    args = parse_args(argv)
    if args.fine_tuned:
        CONDITION = "fine_tuned_checkpoint"
    if Path(args.model_id).is_dir():
        args.model_revision = None
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
