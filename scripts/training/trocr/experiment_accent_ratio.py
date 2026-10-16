"""Experiment accent ratio: PACLIC 2026 v2 release."""

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    default_data_collator,
)

import wandb

from scripts.evaluation.accent_metrics import compute_all_accent_aware_metrics
from scripts.utils.transliteration import convert_text

from .dataset import (
    MixedDataset,
    create_parallel_mixed_dataset,
    hash_key_list,
    load_ocr_dataset_from_config,
)
from .generation_policy import apply_generation_policy
from .metrics import OCRMetrics, create_compute_metrics_fn
from .model_freezing import apply_freeze_mode, normalize_freeze_mode
from .seq2seq_loss import install_no_shift_loss
from .train import early_stopping_callback_from_params

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent


def build_model_source(model_cfg) -> dict:
    """Foundation-model provenance for ``results.json`` / WandB config .

    Records the ``model.base`` (HF repo id or local dir) and the pinned
    ``model.revision`` (immutable commit SHA/tag; ``None`` = unpinned/local) so a
    finished run is self-describing about the exact init weights it consumed. The
    same ``revision`` is forwarded to every ``from_pretrained`` in
    :func:`load_model_and_processor`.
    """
    meta = model_cfg or {}
    return {"base": meta.get("base"), "revision": meta.get("revision")}


def load_model_and_processor(cfg: DictConfig):
    """Load model and processor based on configuration.

    ``model.revision`` (an immutable Hub commit/tag; the study) is forwarded to
    every ``from_pretrained`` on the ``model.base`` path so a claim-bearing run
    pins the exact foundation weights. ``None`` (the default) reproduces the prior
    unpinned behaviour and is a no-op for a local ``model.base`` directory. The
    encoder/decoder path loads two *separate* repos, so a single ``model.revision``
    is intentionally not applied there.
    """
    model_cfg = cfg.model
    model_revision = model_cfg.get("revision")

    if model_cfg.get("base"):
        model_name = model_cfg.base
        processor = TrOCRProcessor.from_pretrained(
            model_name, revision=model_revision, use_fast=False
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name, revision=model_revision
        )
        tokenizer = processor.tokenizer  # ty: ignore[unresolved-attribute]
    elif model_cfg.get("encoder") and model_cfg.get("decoder"):
        processor = ViTImageProcessor.from_pretrained(model_cfg.encoder)
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.decoder)
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            model_cfg.encoder, model_cfg.decoder
        )
    else:
        raise ValueError("Invalid model configuration")

    return model, processor, tokenizer


def configure_model_for_generation(model, tokenizer, generation_cfg: DictConfig):
    """Configure model for text generation via the shared start-token policy.

    Uses the same
    :func:`scripts.training.trocr.generation_policy.apply_generation_policy` as
    ``train.py`` / ``test.py`` so the in-training eval and the final eval here use
    the *same* ``decoder_start_token_id`` the model is trained with, rather than
    the old hard-coded BOS . For a freshly loaded TrOCR base that
    resolves to EOS (the upstream convention); an explicit Hydra
    ``generation.decoder_start_token_id`` still overrides it.
    """
    decode_overrides = {
        "max_length": generation_cfg.get("max_length", 256),
        "early_stopping": generation_cfg.get("early_stopping", True),
        "no_repeat_ngram_size": generation_cfg.get("no_repeat_ngram_size", 3),
        "length_penalty": generation_cfg.get("length_penalty", 2.0),
        "num_beams": generation_cfg.get("num_beams", 4),
    }
    resolution = apply_generation_policy(
        model,
        tokenizer,
        decode_overrides=decode_overrides,
        decoder_start_token_id=generation_cfg.get("decoder_start_token_id"),
    )
    print(f"[generation-policy] {resolution.describe()}", flush=True)
    return resolution


def _inject_avg_metrics(metrics: dict, metric_key_prefix: str = "eval") -> dict:
    """Add ``<prefix>_avg_<name>`` keys for metrics present under both
    ``<prefix>_rv_<name>`` and ``<prefix>_ms_<name>``. Mutates and returns
    ``metrics``.
    """
    rv_prefix = f"{metric_key_prefix}_rv_"
    ms_prefix = f"{metric_key_prefix}_ms_"
    for key, rv_val in list(metrics.items()):
        if not key.startswith(rv_prefix):
            continue
        base = key[len(rv_prefix) :]
        ms_key = f"{ms_prefix}{base}"
        if ms_key not in metrics:
            continue
        ms_val = metrics[ms_key]
        if not isinstance(rv_val, (int, float)) or not isinstance(ms_val, (int, float)):
            continue
        metrics[f"{metric_key_prefix}_avg_{base}"] = (rv_val + ms_val) / 2
    return metrics


class MultiEvalSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that injects ``eval_avg_<metric>`` keys whenever both
    the ``rv`` and ``ms`` per-target metrics are present in the result dict.

    This makes ``metric_for_best_model='avg_cer'`` (and the WandB ``eval_avg_cer``
    log) work out of the box when ``eval_dataset`` is a dict like
    ``{"rv": ..., "ms": ...}``.

    For dict eval we drive the per-target evaluations ourselves (rather than
    delegating to HF's recursive dict handling) for two reasons: HF's outer
    dict-eval does not ``self.log`` the merged dict, and its inner per-dataset
    calls fire the callbacks with metrics that never contain ``eval_avg_*`` — so
    ``EarlyStoppingCallback`` (``metric_for_best_model='avg_cer'``) could not find
    its metric and silently disabled itself . We therefore detach
    early stopping during the inner evals, inject ``eval_avg_*`` + ``self.log``
    them, then fire early stopping once on the merged metrics.
    """

    def evaluate(  # type: ignore[override]
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ):
        resolved = eval_dataset if eval_dataset is not None else self.eval_dataset
        if not isinstance(resolved, dict):
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                **gen_kwargs,
            )
            _inject_avg_metrics(metrics, metric_key_prefix=metric_key_prefix)
            return metrics

        # Dict eval (e.g. {"rv": ..., "ms": ...}): HF's recursive per-dataset
        # evaluate fires the callbacks with the *inner* per-target metrics, which
        # never contain ``eval_avg_*``. ``EarlyStoppingCallback`` (whose
        # ``metric_for_best_model`` is ``avg_cer`` for the rv50 mix) then "did not
        # find eval_avg_cer" and silently disabled itself . So we drive
        # the per-target evals ourselves with early stopping detached, inject the
        # averaged metrics, then fire early stopping once on the merged dict.
        removed = self._detach_early_stopping_callbacks()
        try:
            metrics: dict = {}
            for name, dataset in resolved.items():
                metrics.update(
                    super().evaluate(
                        # dict values are OCRDatasets; ty sees the untyped dict as
                        # ``object`` and can't prove the Dataset type.
                        eval_dataset=dataset,  # ty: ignore[invalid-argument-type]
                        ignore_keys=ignore_keys,
                        metric_key_prefix=f"{metric_key_prefix}_{name}",
                        **gen_kwargs,
                    )
                )
        finally:
            for callback in removed:
                self.add_callback(callback)

        before = set(metrics)
        _inject_avg_metrics(metrics, metric_key_prefix=metric_key_prefix)
        new_keys = sorted(set(metrics) - before)
        if new_keys:
            self.log({k: metrics[k] for k in new_keys})

        for callback in removed:
            callback.on_evaluate(self.args, self.state, self.control, metrics=metrics)
        return metrics

    def _detach_early_stopping_callbacks(self) -> list[EarlyStoppingCallback]:
        """Remove and return the ``EarlyStoppingCallback`` instances (preserving
        their patience state) so the per-target inner evals don't trigger them on
        ``eval_avg``-less metrics ."""
        removed = [
            cb
            for cb in self.callback_handler.callbacks
            if isinstance(cb, EarlyStoppingCallback)
        ]
        for callback in removed:
            self.remove_callback(callback)
        return removed


def _resolve_metric_for_best_model(
    eval_targets: list[str],
    rv_ratio: float,
    policy: str,
    tie_breaker: str,
) -> str:
    """Pick the ``metric_for_best_model`` key based on training mix.

    With multi-target eval the Trainer prefixes metrics as ``eval_<target>_<name>``
    (and ``MultiEvalSeq2SeqTrainer`` adds ``eval_avg_<name>``). The HF trainer
    auto-prepends ``eval_`` if missing, so we return the suffix only.
    """
    if not eval_targets:
        raise ValueError("eval_targets must contain at least one of {'rv', 'ms'}")

    if len(eval_targets) == 1:
        # Single-target eval ⇒ Trainer uses the default ``eval_`` prefix only.
        return "cer"

    if {"rv", "ms"} - set(eval_targets):
        raise ValueError(
            "Multi-target eval requires both 'rv' and 'ms' in eval_targets, "
            f"got {eval_targets}"
        )

    policy = policy.lower()
    if policy == "rv":
        return "rv_cer"
    if policy == "ms":
        return "ms_cer"
    if policy == "avg":
        return "avg_cer"
    if policy != "dominant_cer":
        raise ValueError(
            f"Unknown best_model_policy={policy!r}; expected one of "
            "{'dominant_cer', 'rv', 'ms', 'avg'}"
        )

    if rv_ratio > 0.5:
        return "rv_cer"
    if rv_ratio < 0.5:
        return "ms_cer"

    tie_breaker_normalized = tie_breaker.lower()
    if tie_breaker_normalized in {"avg", "avg_cer"}:
        return "avg_cer"
    if tie_breaker_normalized in {"rv", "rv_cer"}:
        return "rv_cer"
    if tie_breaker_normalized in {"ms", "ms_cer"}:
        return "ms_cer"
    raise ValueError(
        f"Unknown best_model_tie_breaker={tie_breaker!r}; expected one of "
        "{'avg_cer', 'rv_cer', 'ms_cer'}"
    )


def build_train_transform(cfg: DictConfig) -> Optional[Callable]:
    """Build a *train-only* PIL-image augmentation pipeline from ``cfg.augmentation``.

    Returns ``None`` when the section is missing or ``enabled=false``, so train-time
    behaviour matches the pre-augmentation code path. When enabled, the pipeline is
    designed for synthetic OCR line images: small affine jitter + mild colour jitter
    + light Gaussian blur. We deliberately avoid crops, flips, hue, and saturation
    because they either truncate end-of-line characters (label/image mismatch) or
    invert the printed-text contrast assumption.

    Raises ``ValueError`` (with all offending fields batched into one message) when
    the user-facing knobs would otherwise produce a confusing torchvision exception
    — typically at *sampling* time, deep in training. Pre-flight validation here
    means a typo in ``conf/experiment.yaml`` fails immediately at run start.
    """
    aug_cfg = cfg.get("augmentation")
    if aug_cfg is None or not aug_cfg.get("enabled", False):
        return None

    affine_degrees = float(aug_cfg.get("affine_degrees", 2))
    affine_translate = float(aug_cfg.get("affine_translate", 0.02))
    affine_scale_min = float(aug_cfg.get("affine_scale_min", 0.95))
    affine_scale_max = float(aug_cfg.get("affine_scale_max", 1.02))
    brightness = float(aug_cfg.get("color_jitter_brightness", 0.15))
    contrast = float(aug_cfg.get("color_jitter_contrast", 0.15))
    blur_kernel = int(aug_cfg.get("blur_kernel", 3))
    blur_sigma_min = float(aug_cfg.get("blur_sigma_min", 0.1))
    blur_sigma_max = float(aug_cfg.get("blur_sigma_max", 1.0))

    errors: list[str] = []
    # ``RandomAffine.translate`` documents fractions in [0, 1].
    if not 0.0 <= affine_translate <= 1.0:
        errors.append(
            f"augmentation.affine_translate must be in [0.0, 1.0], "
            f"got {affine_translate}"
        )
    # ``scale=(lo, hi)`` is sampled via ``uniform_(lo, hi)`` at *runtime*, so a
    # misordered pair only crashes mid-epoch. Validate up front.
    if affine_scale_min > affine_scale_max:
        errors.append(
            f"augmentation.affine_scale_min ({affine_scale_min}) must be "
            f"<= augmentation.affine_scale_max ({affine_scale_max})"
        )
    # ``GaussianBlur.kernel_size`` requires a positive odd int; torchvision's own
    # error message doesn't name the offending YAML field.
    if blur_kernel <= 0 or blur_kernel % 2 == 0:
        errors.append(
            f"augmentation.blur_kernel must be a positive odd integer, "
            f"got {blur_kernel}"
        )
    # ``sigma=(lo, hi)`` is also sampled at runtime — same failure-mode story.
    if blur_sigma_min > blur_sigma_max:
        errors.append(
            f"augmentation.blur_sigma_min ({blur_sigma_min}) must be "
            f"<= augmentation.blur_sigma_max ({blur_sigma_max})"
        )
    if errors:
        raise ValueError("Invalid augmentation config:\n  - " + "\n  - ".join(errors))

    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=affine_degrees,
                translate=(affine_translate, affine_translate),
                scale=(affine_scale_min, affine_scale_max),
                interpolation=InterpolationMode.BILINEAR,
                fill=255,
            ),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
            transforms.GaussianBlur(
                kernel_size=blur_kernel,
                sigma=(blur_sigma_min, blur_sigma_max),
            ),
        ]
    )


def evaluate_on_dataset(
    model,
    processor,
    tokenizer,
    dataset,
    dataset_name: str,
    batch_size: int = 16,
    num_workers: int = 4,
    use_chrf: bool = True,
    model_scheme: str = "slp1_accented",
) -> dict:
    """Evaluate model on a specific dataset.

    In addition to CER/WER/CHRF the returned dict carries the alignment-based
    accent-aware metrics (``accent_precision``/``recall``/``f1`` and
    ``accent_stripped_cer``) so ``run_experiment`` can flatten them into
    ``results.json`` for the representative pre-adaptation comparisons.
    """
    model.eval()
    device = next(model.parameters()).device

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            # Generate predictions
            generated_ids = model.generate(pixel_values)

            # Decode predictions
            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Decode references
            labels[labels == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(pred_texts)
            all_refs.extend(ref_texts)

    # Compute metrics
    metrics_calculator = OCRMetrics(use_chrf=use_chrf)
    metrics = metrics_calculator.compute(all_preds, all_refs)

    # Convert to ISO before computing accent-aware metrics (combining acute /
    # grave only make sense in the ISO representation). If the model already
    # emits ISO, ``convert_text`` is a no-op aside from NFC normalisation.
    iso_preds = [convert_text(p, model_scheme, "iso") for p in all_preds]
    iso_refs = [convert_text(r, model_scheme, "iso") for r in all_refs]
    accent_metrics = compute_all_accent_aware_metrics(iso_preds, iso_refs)

    print(f"\n{dataset_name} Results:")
    print(f"  CER: {metrics['cer']:.4f}")
    print(f"  WER: {metrics['wer']:.4f}")
    print(f"  Exact Match: {metrics['exact_match']:.4f}")
    print(
        f"  Accent F1: {accent_metrics['accent_f1']:.4f} "
        f"(P={accent_metrics['accent_precision']:.4f}, "
        f"R={accent_metrics['accent_recall']:.4f})"
    )
    print(f"  Accent-stripped CER: {accent_metrics['accent_stripped_cer']:.4f}")

    return {
        "dataset": dataset_name,
        "predictions": all_preds,
        "references": all_refs,
        **metrics,
        **accent_metrics,
    }


def build_experiment_name(
    rv_ratio: float,
    ms_ratio: float,
    run_suffix: Optional[str] = None,
    freeze: Optional[str] = None,
) -> str:
    """Build the experiment/run name from the RV/MS mix, suffix, and freeze mode.

    The base name encodes the train mix as ``rv_<pct>_ms_<pct>``. Disambiguation
    precedence:

    1. A non-empty ``run_suffix`` is appended as ``_<suffix>`` and *wins* — the
       user is taking explicit control of the name (e.g. the text scheme in the
       ISO-vs-SLP1 comparison, the study). A non-default ``freeze`` is then *not*
       also appended; the mode is still recorded in WandB config / ``results.json``
       so the run stays auditable.
    2. Otherwise a non-default ``freeze`` mode is appended as ``_freeze_<mode>``
       so encoder/decoder-frozen conditions at the same RV/MS ratio don't
       overwrite the full-fine-tune ``rv_<pct>_ms_<pct>`` outputs .
    3. ``run_suffix`` ``None`` / empty and ``freeze`` ``none`` preserve the
       historical ``rv_<pct>_ms_<pct>`` paths used by the study accent-ratio sweep.
    """
    base = f"rv_{int(rv_ratio * 100)}_ms_{int(ms_ratio * 100)}"
    suffix = None if run_suffix is None else str(run_suffix).strip()
    if suffix:
        return f"{base}_{suffix}"
    freeze_norm = "none" if freeze is None else str(freeze).strip().lower()
    if freeze_norm and freeze_norm != "none":
        return f"{base}_freeze_{freeze_norm}"
    return base


def build_wandb_tags(
    rv_ratio: float,
    ms_ratio: float,
    model_scheme: str,
    freeze_mode: str,
    seed: Optional[int] = None,
    extra_tags: Optional[Iterable[str]] = None,
) -> list[str]:
    """Build the WandB tag list so a run is self-describing from its tags alone.

    The built-in tags encode the train mix (``rv_<pct>`` / ``ms_<pct>``), the text
    scheme, and the freeze mode. ``seed`` adds a ``seed_<n>`` tag — needed for the
    multi-seed studies  where the same ratio runs under several seeds.
    ``extra_tags`` appends study-level labels (e.g. ``paclic``, ``multi-seed``)
    supplied via ``wandb.tags`` in config; blanks and duplicates (including any
    that collide with the built-ins) are dropped so a launcher can pass a coarse
    list without bookkeeping. Order is stable: built-ins, then seed, then extras
    in the given order.
    """
    tags = [
        f"rv_{int(rv_ratio * 100)}",
        f"ms_{int(ms_ratio * 100)}",
        f"text_{model_scheme}",
        f"freeze_{freeze_mode}",
    ]
    if seed is not None:
        tags.append(f"seed_{seed}")
    for tag in extra_tags or []:
        text = str(tag).strip()
        if text and text not in tags:
            tags.append(text)
    return tags


def _parse_checkpoint_step(checkpoint_path: Optional[str]) -> Optional[int]:
    """Extract the integer step from a ``.../checkpoint-<step>`` path (else None)."""
    if not checkpoint_path:
        return None
    name = Path(checkpoint_path).name
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix) :]
    return int(tail) if tail.isdigit() else None


def summarize_training_state(state, metric_for_best_model: str) -> dict:
    """Flatten a finished ``TrainerState`` into the run-ledger provenance study needs.

    Records *how* a run trained — optimizer steps (``global_step``), the final and
    best epoch, and which checkpoint was selected as best — into ``results.json``
    so the record survives even after the launcher purges the checkpoint dir (and
    its ``trainer_state.json``) on success. Every field degrades to ``None``
    instead of raising, so an unusual state (e.g. no best checkpoint selected)
    still yields a valid summary. ``metric_for_best_model`` is the trainer's metric
    *suffix* (e.g. ``avg_cer``); the resolved key logged in ``log_history`` is
    ``eval_<suffix>``.
    """
    best_metric_key = f"eval_{metric_for_best_model}"
    best_metric = getattr(state, "best_metric", None)
    best_ckpt = getattr(state, "best_model_checkpoint", None)
    best_step = _parse_checkpoint_step(best_ckpt)
    log_history = list(getattr(state, "log_history", None) or [])

    best_epoch = None
    # Prefer the log entry at the best checkpoint's step (unambiguous); fall back
    # to the eval entry whose best-metric value matches (older/edge states).
    if best_step is not None:
        for entry in log_history:
            if entry.get("step") == best_step and "epoch" in entry:
                best_epoch = entry.get("epoch")
                break
    if best_epoch is None and best_metric is not None:
        for entry in log_history:
            if entry.get(best_metric_key) == best_metric and "epoch" in entry:
                best_epoch = entry.get("epoch")
                if best_step is None:
                    best_step = entry.get("step")
                break

    return {
        "global_step": getattr(state, "global_step", None),
        "final_epoch": getattr(state, "epoch", None),
        "best_metric_key": best_metric_key,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_model_checkpoint": best_ckpt,
    }


def build_subset_metadata(
    mixed: MixedDataset,
    *,
    experiment_name: str,
    rv_ratio: float,
    ms_ratio: float,
    seed: int,
    parallel_subset_seed_explicit: Optional[int],
    total_train_samples_requested: Optional[int],
) -> dict:
    """Flatten the parallel-subset provenance a :class:`MixedDataset` carries.

    The deterministic subset ladder  is what makes the data-efficiency
    curve auditable, so this records *both* seeds without conflating them: ``seed``
    is the Trainer RNG, ``parallel_subset_seed`` is the resolved seed that actually
    selected the shared texts (``mixed.subset_seed``). ``parallel_subset_seed_explicit``
    keeps the raw config value (``None`` when the knob was absent and the selection
    fell back to ``seed``), so a reader can tell an explicit choice from the default.

    The full ordered selection is written to ``parallel_keys.txt`` (ratio-invariant),
    so only the routed *hashes* and ``n_a`` are recorded here: with the txt file and
    ``n_a`` the RV/MS routing (``keys_a = selection[:n_a]``) is fully reconstructable
    without bloating the JSON with two more copies of ~10k keys.
    """
    keys_a = mixed.keys_a or []
    keys_b = mixed.keys_b or []
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "parallel_subset_seed": mixed.subset_seed,
        "parallel_subset_seed_explicit": parallel_subset_seed_explicit,
        "rv_ratio": rv_ratio,
        "ms_ratio": ms_ratio,
        "total_train_samples_requested": total_train_samples_requested,
        "mix_mode": mixed.mix_mode,
        "key_column": mixed.key_column,
        "selection_algorithm": mixed.selection_algorithm,
        "parallel_keys_total": mixed.parallel_keys_total,
        "parallel_keys_used": mixed.parallel_keys_used,
        "n_a": mixed.n_a,
        "n_b": mixed.n_b,
        "rv_train_samples": mixed.n_a,
        "ms_train_samples": mixed.n_b,
        "key_list_sha256": mixed.key_list_sha256,
        "keys_a_sha256": hash_key_list(keys_a),
        "keys_b_sha256": hash_key_list(keys_b),
        "keys_a_count": len(keys_a),
        "keys_b_count": len(keys_b),
        "routing_note": (
            "keys_a (RV / dataset_a) = parallel_keys.txt[:n_a]; "
            "keys_b (MS / dataset_b) = parallel_keys.txt[n_a:]. The ordered key "
            "list is ratio-invariant for a fixed (parallel_subset_seed, "
            "total_train_samples), so key_list_sha256 is identical across the five "
            "mix ratios; only n_a / the routed hashes change."
        ),
    }


def write_subset_manifest(
    output_dir: Path, mixed: MixedDataset, metadata: dict
) -> None:
    """Persist ``subset_manifest.json`` + ``parallel_keys.txt`` in ``output_dir``.

    Written up front (before training) so the deterministic-subset record survives
    even if training later crashes. ``parallel_keys.txt`` is the full ordered
    (ratio-invariant) selection, one key per line, newline-joined to match
    :func:`~scripts.training.trocr.dataset.hash_key_list`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "subset_manifest.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    keys = mixed.selected_parallel_keys or []
    (output_dir / "parallel_keys.txt").write_text(
        "\n".join(keys) + ("\n" if keys else ""), encoding="utf-8"
    )


def run_experiment(cfg: DictConfig):
    """Run a single experiment with specified RV/MS ratio."""

    rv_ratio = cfg.rv_ratio
    ms_ratio = 1.0 - rv_ratio
    run_suffix = cfg.get("run_suffix")
    # Encoder/decoder freeze control . Validate up front so an unknown
    # mode fails before WandB init / training, and let a non-default mode
    # disambiguate the run name (below) so matched freeze conditions at the same
    # RV/MS ratio don't overwrite each other. The model itself is frozen later,
    # after generation/loss setup and before the Trainer is built.
    model_cfg_meta = cfg.get("model", {}) or {}
    freeze_mode = normalize_freeze_mode(model_cfg_meta.get("freeze"))
    # Record the foundation in results.json so a finished run is self-describing
    # about which ``model.base`` (and pinned ``model.revision``, the study)
    # produced it (the release contract §4 / the study: matched comparisons must hold the
    # foundation fixed, and the freeze factorial aggregator verifies a single base
    # across the 15 cells). ``None`` base for the encoder/decoder-pretrained path,
    # which has no single base.
    model_source_meta = build_model_source(model_cfg_meta)
    model_base = model_source_meta["base"]
    model_revision = model_source_meta["revision"]
    experiment_name = build_experiment_name(
        rv_ratio, ms_ratio, run_suffix, freeze=freeze_mode
    )

    # Read the text scheme once up front so it can flow into the WandB tags /
    # config and into ``results.json`` (the study: matched ISO-vs-SLP1 runs must
    # be self-describing about which scheme produced each number).
    text_cfg = cfg.get("text", {}) or {}
    input_scheme = text_cfg.get("input_scheme", "iso")
    model_scheme = text_cfg.get("model_scheme", "slp1_accented")

    # Parallel-subset selection seed . Decoupled from the Trainer RNG
    # (``seed``) so the data-efficiency subset ladder can be held fixed across
    # seeds (fixed subset ⇒ seed variation is training stochasticity, not a
    # re-drawn low-data sample). When the knob is absent, the mixer falls back to
    # ``seed`` — exact previous behaviour. Resolve the effective value up front so
    # it can flow into the WandB config alongside the (separate) Trainer seed.
    parallel_subset_seed = cfg.get("parallel_subset_seed")
    trainer_seed = cfg.get("seed", 42)
    subset_seed_effective = (
        trainer_seed if parallel_subset_seed is None else parallel_subset_seed
    )

    # Read the data source once up front so it (and, for HF, the dataset
    # id/revision) flows into the WandB config and ``results.json`` — the study:
    # a run must be self-describing about which RV/MS data it consumed. ``csv``
    # (default) keeps the local file_name,text + image-tree compatibility path.
    data_cfg_meta = cfg.get("data", {}) or {}
    data_source = str(data_cfg_meta.get("source", "csv") or "csv").lower()
    hf_meta: Optional[dict] = None
    if data_source == "hf" and data_cfg_meta.get("hf") is not None:
        container = OmegaConf.to_container(data_cfg_meta.get("hf"), resolve=True)
        hf_meta = container if isinstance(container, dict) else {}
    data_source_meta: dict = {"source": data_source}
    if hf_meta is not None:
        data_source_meta.update(
            {
                "hf_dataset_id": hf_meta.get("dataset_id"),
                "hf_revision": hf_meta.get("revision"),
                "hf_image_column": hf_meta.get("image_column", "image"),
                "hf_text_column": hf_meta.get("text_column", "text"),
                "hf_rv_splits": hf_meta.get("rv"),
                "hf_ms_splits": hf_meta.get("ms"),
            }
        )

    print(f"\n{'=' * 60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"RV ratio: {rv_ratio:.0%}, MS ratio: {ms_ratio:.0%}")
    print(f"Text scheme: input={input_scheme}, model={model_scheme}")
    print(f"Model base: {model_base} (revision={model_revision})")
    print(f"Data source: {data_source}")
    if hf_meta is not None:
        print(
            f"  HF dataset: {hf_meta.get('dataset_id')}"
            f" (revision={hf_meta.get('revision')})"
        )
    print(f"{'=' * 60}\n")

    # Initialize WandB
    wandb_cfg = cfg.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "vedic-ocr-accent-experiment"),
        name=experiment_name,
        config={
            **OmegaConf.to_container(cfg, resolve=True),  # ty: ignore[invalid-argument-type]
            "experiment_name": experiment_name,
            "rv_ratio": rv_ratio,
            "ms_ratio": ms_ratio,
            "run_suffix": run_suffix,
            "input_scheme": input_scheme,
            "model_scheme": model_scheme,
            "model_source": model_source_meta,
            "data_source": data_source_meta,
            "freeze_mode": freeze_mode,
            "parallel_subset_seed": subset_seed_effective,
            "parallel_subset_seed_explicit": parallel_subset_seed,
        },
        tags=build_wandb_tags(
            rv_ratio,
            ms_ratio,
            model_scheme,
            freeze_mode,
            seed=cfg.get("seed"),
            extra_tags=wandb_cfg.get("tags"),
        ),
        reinit=True,
        mode=cfg.get("wandb_mode", "online"),
    )

    # Load model and processor
    model, processor, tokenizer = load_model_and_processor(cfg)

    # Configure for generation
    params = cfg.params
    # Build generation config with defaults
    generation_defaults = {
        "max_length": params.max_length,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "length_penalty": 2.0,
        "num_beams": 4,
    }
    # Merge with user-provided generation config (if any)
    if "generation" in cfg and cfg.generation is not None:
        generation_cfg = OmegaConf.merge(
            OmegaConf.create(generation_defaults), cfg.generation
        )
    else:
        generation_cfg = OmegaConf.create(generation_defaults)

    configure_model_for_generation(model, tokenizer, generation_cfg)  # ty: ignore[invalid-argument-type]

    # Encoder-decoder loss alignment (transformers 4.57): VisionEncoderDecoder
    # already teacher-forcing-shifts the decoder input via shift_tokens_right,
    # so train on a no-shift cross-entropy instead of the default
    # ForCausalLMLoss, which would shift a second time and supervise the model
    # two positions ahead (free-running generate then emits every other token).
    # See scripts/training/trocr/seq2seq_loss.py.
    install_no_shift_loss(model)
    print(
        "[loss] installed no-shift cross-entropy for encoder-decoder training",
        flush=True,
    )

    # Apply encoder/decoder freeze controls . Done after generation +
    # loss setup and before Trainer construction so the optimizer sees the
    # intended trainable set; ``none`` leaves the model fully trainable
    # (historical behaviour). The parameter counts go to WandB + results.json so
    # a finished run is auditable without reloading the model.
    freeze_summary = apply_freeze_mode(model, freeze_mode)
    print(f"[freeze] {freeze_summary.describe()}", flush=True)
    wandb.log(
        {
            "freeze/total_params": freeze_summary.total_params,
            "freeze/trainable_params": freeze_summary.trainable_params,
            "freeze/encoder_trainable_params": freeze_summary.encoder_trainable_params,
            "freeze/decoder_trainable_params": freeze_summary.decoder_trainable_params,
        }
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load datasets. ``data.source`` selects the local ``file_name,text`` CSV +
    # image tree (compatibility / debug) or a published Hugging Face Dataset
    # (preferred long-term input); ``load_ocr_dataset_from_config`` hides the
    # source from this caller and returns ``.df``-backed datasets either way so the
    # parallel mix below is source-agnostic . ``input_scheme`` /
    # ``model_scheme`` were read up front (near the WandB init) so they could be
    # logged there.
    data_cfg = cfg.data

    # Train-only image augmentation. ``None`` when ``augmentation.enabled=false``
    # (or the section is absent), preserving the pre-augmentation pixel path.
    train_transform = build_train_transform(cfg)

    def _load_split(
        domain: str,
        split: str,
        image_transform: Optional[Callable] = None,
    ):
        return load_ocr_dataset_from_config(
            data_cfg,
            split=split,
            domain=domain,
            root_dir=ROOT_DIR,
            processor=processor,
            tokenizer=tokenizer,
            input_scheme=input_scheme,
            model_scheme=model_scheme,
            max_target_length=params.max_length,
            image_transform=image_transform,
            accent_system=domain,
        )

    # Augmentation is applied *only* to train datasets so val/test stay
    # comparable across runs (and to Job 46).
    rv_train_dataset = _load_split("rv", "train", image_transform=train_transform)
    ms_train_dataset = _load_split("ms", "train", image_transform=train_transform)
    # val splits drive in-training eval; test splits stay reserved for the
    # post-training final eval block.
    rv_val_dataset = _load_split("rv", "val")
    ms_val_dataset = _load_split("ms", "val")
    rv_test_dataset = _load_split("rv", "test")
    ms_test_dataset = _load_split("ms", "test")

    print(f"RV train samples: {len(rv_train_dataset)}")
    print(f"MS train samples: {len(ms_train_dataset)}")
    print(f"RV val samples: {len(rv_val_dataset)}")
    print(f"MS val samples: {len(ms_val_dataset)}")
    print(f"RV test samples: {len(rv_test_dataset)}")
    print(f"MS test samples: {len(ms_test_dataset)}")

    # Create mixed training dataset. RV/MS are parallel corpora (identical text
    # labels, only the image differs), so partition the *shared* texts by ratio
    # instead of sampling two independent pools: fixed N across the sweep and no
    # text entering as both an RV and an MS image .
    total_samples = cfg.get("total_train_samples")
    mixed_train_dataset = create_parallel_mixed_dataset(
        rv_train_dataset,
        ms_train_dataset,
        ratio_a=rv_ratio,
        total_samples=total_samples,
        seed=trainer_seed,
        # ``None`` ⇒ mixer selects with ``seed`` (previous behaviour); an explicit
        # value pins the subset ladder independent of the Trainer seed .
        subset_seed=parallel_subset_seed,
    )

    print(f"Mixed train samples: {len(mixed_train_dataset)}")
    print(f"  - RV samples: {mixed_train_dataset.n_a}")
    print(f"  - MS samples: {mixed_train_dataset.n_b}")
    print(f"  - mix mode: {mixed_train_dataset.mix_mode}")
    print(
        f"  - parallel texts used: {mixed_train_dataset.parallel_keys_used}"
        f" / {mixed_train_dataset.parallel_keys_total} shared"
    )
    print(
        f"  - subset seed: {mixed_train_dataset.subset_seed}"
        f" (trainer seed: {trainer_seed}) | key_list_sha256:"
        f" {mixed_train_dataset.key_list_sha256}"
    )

    # Log dataset info to WandB
    wandb.log(
        {
            "dataset/source": data_source,
            "dataset/rv_train_total": len(rv_train_dataset),
            "dataset/ms_train_total": len(ms_train_dataset),
            "dataset/rv_train_used": mixed_train_dataset.n_a,
            "dataset/ms_train_used": mixed_train_dataset.n_b,
            "dataset/mixed_train_total": len(mixed_train_dataset),
            "dataset/mix_mode": mixed_train_dataset.mix_mode,
            "dataset/parallel_keys_total": mixed_train_dataset.parallel_keys_total,
            "dataset/parallel_keys_used": mixed_train_dataset.parallel_keys_used,
            "dataset/parallel_subset_seed": mixed_train_dataset.subset_seed,
            "dataset/key_list_sha256": mixed_train_dataset.key_list_sha256,
            "dataset/rv_val_total": len(rv_val_dataset),
            "dataset/ms_val_total": len(ms_val_dataset),
            "dataset/rv_test_total": len(rv_test_dataset),
            "dataset/ms_test_total": len(ms_test_dataset),
        }
    )

    # Augmentation summary (train-only). Logged regardless of enabled state so
    # WandB run summaries can be filtered/grouped by aug knobs.
    aug_cfg = cfg.get("augmentation")
    aug_enabled = bool(aug_cfg.get("enabled", False)) if aug_cfg is not None else False
    aug_log: dict = {"augmentation/enabled": aug_enabled}
    if aug_enabled:
        aug_log.update(
            {
                "augmentation/affine_degrees": float(aug_cfg.get("affine_degrees", 2)),
                "augmentation/affine_translate": float(
                    aug_cfg.get("affine_translate", 0.02)
                ),
                "augmentation/affine_scale_min": float(
                    aug_cfg.get("affine_scale_min", 0.95)
                ),
                "augmentation/affine_scale_max": float(
                    aug_cfg.get("affine_scale_max", 1.02)
                ),
                "augmentation/color_jitter_brightness": float(
                    aug_cfg.get("color_jitter_brightness", 0.15)
                ),
                "augmentation/color_jitter_contrast": float(
                    aug_cfg.get("color_jitter_contrast", 0.15)
                ),
                "augmentation/blur_kernel": int(aug_cfg.get("blur_kernel", 3)),
                "augmentation/blur_sigma_min": float(
                    aug_cfg.get("blur_sigma_min", 0.1)
                ),
                "augmentation/blur_sigma_max": float(
                    aug_cfg.get("blur_sigma_max", 1.0)
                ),
            }
        )
    wandb.log(aug_log)
    print(f"Augmentation enabled: {aug_enabled}")

    # Output directory
    output_dir = Path(cfg.get("output_dir", "./outputs/experiments")) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the deterministic parallel-subset provenance  up front,
    # so the record survives even if training later crashes. parallel_keys.txt is
    # the full ordered (ratio-invariant) selection; subset_manifest.json adds the
    # routing split + hashes and keeps ``seed`` / ``parallel_subset_seed`` distinct.
    subset_metadata = build_subset_metadata(
        mixed_train_dataset,
        experiment_name=experiment_name,
        rv_ratio=rv_ratio,
        ms_ratio=ms_ratio,
        seed=trainer_seed,
        parallel_subset_seed_explicit=parallel_subset_seed,
        total_train_samples_requested=total_samples,
    )
    write_subset_manifest(output_dir, mixed_train_dataset, subset_metadata)
    print(
        f"[subset] wrote subset_manifest.json + parallel_keys.txt "
        f"({subset_metadata['parallel_keys_used']} keys) to {output_dir}",
        flush=True,
    )

    # Calculate training steps
    total_train_steps = (
        len(mixed_train_dataset) // params.train_batch_size * params.num_train_epochs
    )
    warmup_ratio = params.get("warmup_ratio", 0.1)
    warmup_steps = int(total_train_steps * warmup_ratio)

    # Build the in-training eval dataset(s) from ``eval_targets``. Multiple
    # targets are passed as ``dict[str, Dataset]`` so the trainer prefixes the
    # metric keys with the target name (e.g. ``eval_rv_cer``, ``eval_ms_cer``).
    # Reject unknown targets up front: silently dropping them would change the
    # eval mode (e.g. ``["rv", "foo"]`` → RV-only) and the best-model metric
    # without warning.
    eval_targets = list(cfg.get("eval_targets", ["rv", "ms"]))
    val_target_to_dataset = {"rv": rv_val_dataset, "ms": ms_val_dataset}
    if not eval_targets:
        raise ValueError(
            "eval_targets must be a non-empty subset of {'rv', 'ms'}; got []"
        )
    unknown = [t for t in eval_targets if t not in val_target_to_dataset]
    if unknown:
        raise ValueError(
            f"eval_targets contains unknown entries {unknown!r}; "
            f"expected a non-empty subset of {sorted(val_target_to_dataset)}"
        )
    selected_val = {target: val_target_to_dataset[target] for target in eval_targets}
    if len(selected_val) == 1:
        eval_dataset = next(iter(selected_val.values()))
    else:
        eval_dataset = selected_val

    metric_for_best_model = _resolve_metric_for_best_model(
        eval_targets=list(selected_val.keys()),
        rv_ratio=rv_ratio,
        policy=cfg.get("best_model_policy", "dominant_cer"),
        tie_breaker=cfg.get("best_model_tie_breaker", "avg_cer"),
    )
    print(
        f"Eval targets: {list(selected_val.keys())} | "
        f"metric_for_best_model: eval_{metric_for_best_model}"
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        run_name=experiment_name,
        # Thread the experiment seed into the Trainer so the *training* RNG (data
        # order, dropout) varies across the multi-seed study , not
        # just the RV/MS parallel-text partition (create_parallel_mixed_dataset
        # already seeds that). transformers' default is 42, so seed=42 is
        # bit-identical to the previous runs; only seed!=42 changes behaviour.
        seed=trainer_seed,
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.train_batch_size,
        per_device_eval_batch_size=params.eval_batch_size,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=params.get("lr_scheduler_type", "linear"),
        logging_dir=str(output_dir / "logs"),
        logging_steps=params.get("logging_steps", 50),
        save_strategy=params.get("save_strategy", "epoch"),
        eval_strategy=params.get("eval_strategy", "epoch"),
        save_total_limit=params.get("save_total_limit", 2),
        predict_with_generate=True,
        fp16=torch.cuda.is_available() and params.get("fp16", True),
        dataloader_num_workers=params.get("num_workers", 4),
        dataloader_pin_memory=torch.cuda.is_available(),  # Only pin memory if CUDA available
        load_best_model_at_end=params.get("load_best_model_at_end", True),
        metric_for_best_model=metric_for_best_model,
        greater_is_better=params.get("greater_is_better", False),
        report_to=cfg.get("report_to", "wandb"),
    )

    # Create compute_metrics function
    compute_metrics = create_compute_metrics_fn(
        processor=processor,
        tokenizer=tokenizer,
        use_chrf=params.get("use_chrf", True),
    )

    # Use the multi-eval subclass so ``eval_avg_<metric>`` keys (e.g.
    # ``eval_avg_cer``) are injected when both rv and ms are evaluated.
    trainer = MultiEvalSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=mixed_train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        processing_class=processor
        if hasattr(processor, "save_pretrained")
        else tokenizer,
        compute_metrics=compute_metrics,
    )

    es_callback = early_stopping_callback_from_params(params)
    if es_callback is not None:
        trainer.add_callback(es_callback)
        print(
            f"[early-stopping] enabled: patience={es_callback.early_stopping_patience}",
            flush=True,
        )

    # Train
    print("\nStarting training...")
    trainer.train()
    print("Training completed!")

    # Capture training-run provenance (best epoch, optimizer steps, best-checkpoint
    # selection) now, while trainer.state is populated — the launcher purges the
    # checkpoint dir (and its trainer_state.json) on success, so results.json is
    # the only durable ledger of how the run trained .
    training_summary = summarize_training_state(trainer.state, metric_for_best_model)
    print(f"[training-state] {training_summary}", flush=True)

    # Save model
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    results = None
    if cfg.get("skip_final_eval", False):
        print("\nSkipping final evaluation (skip_final_eval=true)")
    else:
        # Final evaluation on both test sets
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)

        eval_num_workers = params.get("num_workers", 4)
        use_chrf = params.get("use_chrf", True)

        rv_results = evaluate_on_dataset(
            model,
            processor,
            tokenizer,
            rv_test_dataset,
            "RV Test",
            batch_size=params.eval_batch_size,
            num_workers=eval_num_workers,
            use_chrf=use_chrf,
            model_scheme=model_scheme,
        )
        ms_results = evaluate_on_dataset(
            model,
            processor,
            tokenizer,
            ms_test_dataset,
            "MS Test",
            batch_size=params.eval_batch_size,
            num_workers=eval_num_workers,
            use_chrf=use_chrf,
            model_scheme=model_scheme,
        )

        # Calculate and log final metrics
        cer_gap = abs(rv_results["cer"] - ms_results["cer"])
        avg_cer = (rv_results["cer"] + ms_results["cer"]) / 2
        accent_f1_gap = abs(rv_results["accent_f1"] - ms_results["accent_f1"])
        accent_stripped_cer_gap = abs(
            rv_results["accent_stripped_cer"] - ms_results["accent_stripped_cer"]
        )

        final_metrics = {
            "final/rv_cer": rv_results["cer"],
            "final/rv_wer": rv_results["wer"],
            "final/rv_exact_match": rv_results["exact_match"],
            "final/rv_accent_precision": rv_results["accent_precision"],
            "final/rv_accent_recall": rv_results["accent_recall"],
            "final/rv_accent_f1": rv_results["accent_f1"],
            "final/rv_accent_f1_labeled": rv_results["accent_f1_labeled"],
            "final/rv_accent_stripped_cer": rv_results["accent_stripped_cer"],
            "final/ms_cer": ms_results["cer"],
            "final/ms_wer": ms_results["wer"],
            "final/ms_exact_match": ms_results["exact_match"],
            "final/ms_accent_precision": ms_results["accent_precision"],
            "final/ms_accent_recall": ms_results["accent_recall"],
            "final/ms_accent_f1": ms_results["accent_f1"],
            "final/ms_accent_f1_labeled": ms_results["accent_f1_labeled"],
            "final/ms_accent_stripped_cer": ms_results["accent_stripped_cer"],
            "final/avg_cer": avg_cer,
            "final/cer_gap": cer_gap,
            "final/accent_f1_gap": accent_f1_gap,
            "final/accent_stripped_cer_gap": accent_stripped_cer_gap,
        }
        wandb.log(final_metrics)

        # Log summary table
        summary_table = wandb.Table(
            columns=["Test Set", "CER", "WER", "Exact Match", "CHRF"],
            data=[
                [
                    "RV",
                    rv_results["cer"],
                    rv_results["wer"],
                    rv_results["exact_match"],
                    rv_results.get("chrf", 0),
                ],
                [
                    "MS",
                    ms_results["cer"],
                    ms_results["wer"],
                    ms_results["exact_match"],
                    ms_results.get("chrf", 0),
                ],
            ],
        )
        wandb.log({"final/summary_table": summary_table})

        # Save results to JSON
        import json

        results = {
            "experiment_name": experiment_name,
            "seed": trainer_seed,
            "num_train_epochs": params.num_train_epochs,
            "training_state": training_summary,
            "run_suffix": run_suffix,
            "input_scheme": input_scheme,
            "model_scheme": model_scheme,
            "data_source": data_source_meta,
            "model_base": model_base,
            "model_revision": model_revision,
            "model_source": model_source_meta,
            "freeze_mode": freeze_mode,
            "freeze_summary": freeze_summary.as_dict(),
            "rv_ratio": rv_ratio,
            "ms_ratio": ms_ratio,
            "rv_train_samples": mixed_train_dataset.n_a,
            "ms_train_samples": mixed_train_dataset.n_b,
            "mix_mode": mixed_train_dataset.mix_mode,
            "parallel_keys_total": mixed_train_dataset.parallel_keys_total,
            "parallel_keys_used": mixed_train_dataset.parallel_keys_used,
            # Deterministic-subset provenance . ``seed`` (above) is the
            # Trainer RNG; ``parallel_subset_seed`` is the resolved data-selection
            # seed — kept distinct so the data-efficiency ladder is auditable. The
            # full ordered selection lives in the sibling ``subset_manifest.json`` /
            # ``parallel_keys.txt``.
            "parallel_subset_seed": subset_seed_effective,
            "parallel_subset_seed_explicit": parallel_subset_seed,
            "key_column": mixed_train_dataset.key_column,
            "selection_algorithm": mixed_train_dataset.selection_algorithm,
            "key_list_sha256": mixed_train_dataset.key_list_sha256,
            "total_train_samples_requested": total_samples,
            "subset_manifest": "subset_manifest.json",
            "rv_test": {
                k: v
                for k, v in rv_results.items()
                if k not in ["predictions", "references"]
            },
            "ms_test": {
                k: v
                for k, v in ms_results.items()
                if k not in ["predictions", "references"]
            },
            "avg_cer": avg_cer,
            "cer_gap": cer_gap,
            "accent_f1_gap": accent_f1_gap,
            "accent_stripped_cer_gap": accent_stripped_cer_gap,
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save detailed predictions
        pd.DataFrame(
            {
                "reference": rv_results["references"],
                "prediction": rv_results["predictions"],
            }
        ).to_csv(output_dir / "rv_predictions.csv", index=False)

        pd.DataFrame(
            {
                "reference": ms_results["references"],
                "prediction": ms_results["predictions"],
            }
        ).to_csv(output_dir / "ms_predictions.csv", index=False)

        print(f"\nResults saved to: {output_dir}")

    wandb.finish()

    return results


@hydra.main(version_base="1.3", config_path="conf", config_name="experiment")
def main(cfg: DictConfig):
    """Entry point for Hydra."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
