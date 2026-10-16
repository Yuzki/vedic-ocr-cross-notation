"""Test evaluate initial checkpoint: PACLIC 2026 v2 release."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.evaluation.evaluate_initial_checkpoint import (
    CONDITION,
    DECODE_FIELDS,
    SPLIT,
    _headline_metrics,
    build_manifest,
    build_run_config,
    resolved_decode_config,
    sample_ids_for,
)
from scripts.evaluation.prediction_metrics import (
    evaluate_prediction_file,
    load_prediction_rows,
)
from scripts.evaluation.prediction_io import write_predictions_csv


class _FakeHFDataset:
    """Minimal stand-in for a ``datasets.Dataset``: column-addressable + counted."""

    def __init__(self, columns: dict[str, list[Any]]):
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, key: str) -> list[Any]:
        return self._columns[key]


def _fake_dataset(file_names: list[str] | None) -> Any:
    columns: dict[str, list[Any]] = {"text": [f"t{i}" for i in range(3)]}
    if file_names is not None:
        columns["file_name"] = file_names
    return SimpleNamespace(hf_dataset=_FakeHFDataset(columns))


# --- build_run_config -------------------------------------------------------


def test_build_run_config_shapes_the_hf_loader_contract():
    cfg = build_run_config(
        model_id="yzk/model",
        model_revision="abc123",
        dataset_id="yzk/data",
        dataset_revision="def456",
    )
    assert cfg.model.base == "yzk/model"
    assert cfg.model.revision == "abc123"
    assert cfg.data.source == "hf"
    assert cfg.data.hf.dataset_id == "yzk/data"
    assert cfg.data.hf.revision == "def456"
    # domain→split maps the reused loader indexes as data.hf.<domain>.<split>.
    assert cfg.data.hf.rv.test == "rv_test"
    assert cfg.data.hf.ms.test == "ms_test"
    # Token-free load (offline from the shared cache).
    assert cfg.data.hf.token is None


def test_build_run_config_allows_null_revisions():
    cfg = build_run_config(
        model_id="local/dir",
        model_revision=None,
        dataset_id="yzk/data",
        dataset_revision=None,
    )
    assert cfg.model.revision is None
    assert cfg.data.hf.revision is None


# --- sample_ids_for ---------------------------------------------------------


def test_sample_ids_from_file_name_column():
    ds = _fake_dataset(["a.png", "b.png", "c.png"])
    ids, source = sample_ids_for(ds, expected=3)
    assert ids == ["a.png", "b.png", "c.png"]
    assert source == "file_name"


def test_sample_ids_fall_back_to_index_when_column_absent():
    ds = _fake_dataset(None)
    ids, source = sample_ids_for(ds, expected=3)
    assert ids == ["0", "1", "2"]
    assert source == "index"


def test_sample_ids_fall_back_on_length_mismatch():
    # A file_name column that does not line up with the evaluated row count must
    # NOT be trusted (misaligned sample ids would be worse than positional ones).
    ds = _fake_dataset(["a.png", "b.png", "c.png"])
    ids, source = sample_ids_for(ds, expected=2)
    assert ids == ["0", "1"]
    assert source == "index"


def test_sample_ids_handle_missing_hf_dataset():
    ids, source = sample_ids_for(SimpleNamespace(), expected=2)
    assert ids == ["0", "1"]
    assert source == "index"


# --- resolved_decode_config -------------------------------------------------


def test_resolved_decode_config_snapshots_saved_fields():
    gen_cfg = SimpleNamespace(
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
        max_length=256,
        decoder_start_token_id=2,
        eos_token_id=[2],
        bos_token_id=0,
        pad_token_id=1,
    )
    decode = resolved_decode_config(SimpleNamespace(generation_config=gen_cfg))
    assert set(decode) == set(DECODE_FIELDS)
    assert decode["num_beams"] == 4
    assert decode["length_penalty"] == 2.0
    assert decode["decoder_start_token_id"] == 2
    assert decode["eos_token_id"] == [2]


def test_resolved_decode_config_missing_fields_are_none():
    decode = resolved_decode_config(
        SimpleNamespace(generation_config=SimpleNamespace())
    )
    assert all(decode[field] is None for field in DECODE_FIELDS)


# --- _headline_metrics ------------------------------------------------------


def test_headline_metrics_selects_scalar_keys_only():
    payload = {
        "cer": 0.1,
        "wer": 0.2,
        "chrf": 88.0,
        "exact_match": 0.5,
        "exact_match_count": 289,
        "accent_precision": 0.9,
        "accent_recall": 0.8,
        "accent_f1": 0.85,
        "accent_stripped_cer": 0.05,
        "num_samples": 578,
        "num_scored": 578,
        # Structured blocks must be dropped from the manifest summary.
        "per_font": {"x": {}},
        "reliability": {"y": 1},
        "edit_events": {"z": 2},
    }
    headline = _headline_metrics(payload)
    assert headline["cer"] == 0.1
    assert headline["accent_f1"] == 0.85
    assert headline["num_samples"] == 578
    assert "per_font" not in headline
    assert "reliability" not in headline
    assert "edit_events" not in headline


# --- build_manifest ---------------------------------------------------------


def _view_stub(view: str) -> dict[str, Any]:
    return {
        "view": view,
        "domain": view.split("_")[0],
        "split": SPLIT,
        "n": 578,
        "sample_id_source": "file_name",
        "predictions_csv": f"predictions/{view}.csv",
        "predictions_sha256": "sha256:deadbeef",
        "metrics_json": f"metrics/{view}.json",
        "metrics": {"cer": 0.4},
    }


def test_build_manifest_records_pins_scope_and_provenance():
    manifest = build_manifest(
        model_id="yzk/model",
        model_revision="abc123",
        dataset_id="yzk/data",
        dataset_revision="def456",
        decode={"num_beams": 4, "decoder_start_token_id": 2},
        start_token_resolution="preserved saved EOS=2",
        eval_settings={"batch_size": 16, "model_scheme": "iso"},
        environment={"token_env_present": False, "hf_hub_offline": "1"},
        views=[_view_stub("rv_test"), _view_stub("ms_test")],
        generated_date="2026-07-22",
    )
    assert manifest["issue"] == 146
    assert manifest["parent_issue"] == 122
    assert manifest["fine_tuning"] is False
    assert manifest["condition"] == CONDITION
    assert manifest["model"]["hub_revision"] == "abc123"
    assert manifest["dataset"]["hub_revision"] == "def456"
    assert manifest["dataset"]["splits"] == ["rv_test", "ms_test"]
    assert manifest["decode"]["on_policy"] is True
    assert manifest["decode"]["config"]["decoder_start_token_id"] == 2
    assert manifest["environment"]["token_env_present"] is False
    # The initialization-bias framing  must be carried in the artifact.
    assert "initialization-bias" in manifest["note"]
    assert "not attributed" in manifest["note"].lower() or "NOT" in manifest["note"]
    # study evaluator is cited as the metric source.
    assert "study" in manifest["reused_code"]["metrics_evaluator"]


# --- real round-trip through the reused study writer + evaluator --------------


def test_written_csv_carries_sample_id_and_recomputes_metrics(tmp_path: Path):
    """The versioned CSV keeps sample_id + axes and regenerates metrics (cond. 2)."""
    references = ["kṛṣṇa", "agni", "soma"]
    predictions = ["kṛṣṇa", "agmi", "soma"]
    sample_ids = ["rv/0001.png", "rv/0002.png", "rv/0003.png"]
    csv_path = tmp_path / "predictions" / "rv_test.csv"
    write_predictions_csv(
        csv_path,
        references=references,
        predictions=predictions,
        sample_ids=sample_ids,
        condition=CONDITION,
        view="rv_test",
        split=SPLIT,
    )

    rows = load_prediction_rows(csv_path)
    assert [r.sample_id for r in rows] == sample_ids
    assert [r.reference for r in rows] == references
    assert [r.prediction for r in rows] == predictions
    assert all(r.condition == CONDITION for r in rows)
    assert all(r.view == "rv_test" and r.split == SPLIT for r in rows)

    payload = evaluate_prediction_file(csv_path)
    assert payload["num_samples"] == 3
    assert payload["condition"] == CONDITION
    assert payload["view"] == "rv_test"
    # Two of three exact after NFC+strip → EM count 2.
    assert payload["exact_match_count"] == 2
    for key in (
        "cer",
        "wer",
        "chrf",
        "exact_match",
        "accent_f1",
        "accent_stripped_cer",
    ):
        assert key in payload
