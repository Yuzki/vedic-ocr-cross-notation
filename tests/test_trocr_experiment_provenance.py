"""Test trocr experiment provenance: PACLIC 2026 v2 release."""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from scripts.training.trocr.experiment_accent_ratio import (
    _parse_checkpoint_step,
    build_model_source,
    build_wandb_tags,
    summarize_training_state,
)


# --- build_wandb_tags --------------------------------------------------------


def test_builtin_tags_without_seed_or_extras():
    assert build_wandb_tags(0.5, 0.5, "iso", "none") == [
        "rv_50",
        "ms_50",
        "text_iso",
        "freeze_none",
    ]


def test_seed_tag_appended_after_builtins():
    tags = build_wandb_tags(1.0, 0.0, "iso", "none", seed=43)
    assert tags == ["rv_100", "ms_0", "text_iso", "freeze_none", "seed_43"]


def test_seed_zero_is_tagged_not_dropped():
    # seed 0 is a valid seed; only ``None`` suppresses the seed tag.
    assert "seed_0" in build_wandb_tags(0.5, 0.5, "iso", "none", seed=0)


def test_extra_tags_appended_in_order():
    tags = build_wandb_tags(
        0.25,
        0.75,
        "iso",
        "none",
        seed=44,
        extra_tags=["paclic", "multi-seed", "accent-ratio-multiseed"],
    )
    assert tags == [
        "rv_25",
        "ms_75",
        "text_iso",
        "freeze_none",
        "seed_44",
        "paclic",
        "multi-seed",
        "accent-ratio-multiseed",
    ]


def test_extra_tags_dedup_against_builtins_and_each_other():
    tags = build_wandb_tags(
        0.5,
        0.5,
        "iso",
        "none",
        seed=42,
        extra_tags=["rv_50", "seed_42", "paclic", "paclic"],
    )
    # collisions with built-ins / the seed tag / earlier extras are dropped
    assert tags == ["rv_50", "ms_50", "text_iso", "freeze_none", "seed_42", "paclic"]


def test_extra_tags_blank_and_whitespace_dropped_and_stripped():
    tags = build_wandb_tags(
        0.5, 0.5, "iso", "none", extra_tags=["", "   ", "  paclic  "]
    )
    assert tags == ["rv_50", "ms_50", "text_iso", "freeze_none", "paclic"]


def test_extra_tags_none_is_noop():
    assert build_wandb_tags(0.0, 1.0, "iso", "none", extra_tags=None) == [
        "rv_0",
        "ms_100",
        "text_iso",
        "freeze_none",
    ]


def test_slp1_scheme_and_freeze_mode_flow_through():
    tags = build_wandb_tags(0.75, 0.25, "slp1_accented", "encoder", seed=1)
    assert tags == [
        "rv_75",
        "ms_25",
        "text_slp1_accented",
        "freeze_encoder",
        "seed_1",
    ]


# --- _parse_checkpoint_step --------------------------------------------------


def test_parse_checkpoint_step_variants():
    assert _parse_checkpoint_step("/x/checkpoints/checkpoint-1302") == 1302
    assert _parse_checkpoint_step("checkpoint-0") == 0
    assert _parse_checkpoint_step(None) is None
    assert _parse_checkpoint_step("") is None
    assert _parse_checkpoint_step("/x/final_model") is None
    assert _parse_checkpoint_step("/x/checkpoint-abc") is None


# --- summarize_training_state ------------------------------------------------


def _state(**kw) -> SimpleNamespace:
    return SimpleNamespace(**kw)


def _log_history() -> list[dict]:
    return [
        {"loss": 0.5, "epoch": 1.0, "step": 100},
        {"eval_avg_cer": 0.09, "epoch": 1.0, "step": 100},
        {"loss": 0.3, "epoch": 2.0, "step": 200},
        {"eval_avg_cer": 0.05, "epoch": 2.0, "step": 200},
        {"loss": 0.2, "epoch": 3.0, "step": 300},
        {"eval_avg_cer": 0.06, "epoch": 3.0, "step": 300},
    ]


def test_summary_resolves_best_epoch_from_checkpoint_step():
    state = _state(
        global_step=300,
        epoch=3.0,
        best_metric=0.05,
        best_model_checkpoint="/o/checkpoints/checkpoint-200",
        log_history=_log_history(),
    )
    summary = summarize_training_state(state, "avg_cer")
    assert summary["global_step"] == 300  # optimizer steps
    assert summary["final_epoch"] == 3.0
    assert summary["best_metric_key"] == "eval_avg_cer"
    assert summary["best_metric"] == 0.05
    assert summary["best_step"] == 200
    assert summary["best_epoch"] == 2.0
    assert summary["best_model_checkpoint"] == "/o/checkpoints/checkpoint-200"


def test_summary_falls_back_to_metric_match_without_checkpoint_path():
    # No best_model_checkpoint (e.g. save path pruned): resolve via the eval entry
    # whose metric equals best_metric.
    state = _state(
        global_step=300,
        epoch=3.0,
        best_metric=0.05,
        best_model_checkpoint=None,
        log_history=_log_history(),
    )
    summary = summarize_training_state(state, "avg_cer")
    assert summary["best_epoch"] == 2.0
    assert summary["best_step"] == 200


def test_summary_degrades_to_none_on_empty_state():
    summary = summarize_training_state(SimpleNamespace(), "cer")
    assert summary == {
        "global_step": None,
        "final_epoch": None,
        "best_metric_key": "eval_cer",
        "best_metric": None,
        "best_epoch": None,
        "best_step": None,
        "best_model_checkpoint": None,
    }


def test_summary_best_epoch_none_when_step_absent_from_log():
    # best checkpoint points at a step with no matching log entry and no metric
    # fallback hit -> best_epoch stays None (never raises).
    state = _state(
        global_step=300,
        epoch=3.0,
        best_metric=0.123,  # matches nothing in the log
        best_model_checkpoint="/o/checkpoints/checkpoint-999",
        log_history=_log_history(),
    )
    summary = summarize_training_state(state, "avg_cer")
    assert summary["best_step"] == 999
    assert summary["best_epoch"] is None


# --- build_model_source + revision forwarding  -------------------


def test_build_model_source_from_dict():
    assert build_model_source({"base": "yzk/repo", "revision": "abc123"}) == {
        "base": "yzk/repo",
        "revision": "abc123",
    }


def test_build_model_source_missing_and_none():
    # No revision key -> None; empty/None cfg -> both None (encoder/decoder path).
    assert build_model_source({"base": "local/dir"}) == {
        "base": "local/dir",
        "revision": None,
    }
    assert build_model_source(None) == {"base": None, "revision": None}


def test_build_model_source_reads_omegaconf():
    cfg = OmegaConf.create({"base": "yzk/repo", "revision": "deadbeef"})
    assert build_model_source(cfg) == {"base": "yzk/repo", "revision": "deadbeef"}


def _patch_from_pretrained(monkeypatch, captured: dict):
    """Replace the two ``from_pretrained`` on the base path with capturing fakes."""
    import scripts.training.trocr.experiment_accent_ratio as mod

    def fake_proc(name, **kwargs):
        captured["proc"] = {"name": name, **kwargs}
        return SimpleNamespace(tokenizer=object())

    def fake_model(name, **kwargs):
        captured["model"] = {"name": name, **kwargs}
        return object()

    monkeypatch.setattr(mod.TrOCRProcessor, "from_pretrained", staticmethod(fake_proc))
    monkeypatch.setattr(
        mod.VisionEncoderDecoderModel, "from_pretrained", staticmethod(fake_model)
    )
    return mod


def test_load_model_and_processor_forwards_revision(monkeypatch):
    captured: dict = {}
    mod = _patch_from_pretrained(monkeypatch, captured)
    cfg = OmegaConf.create({"model": {"base": "yzk/repo", "revision": "cafef00d"}})
    _model, processor, tokenizer = mod.load_model_and_processor(cfg)
    assert captured["proc"]["name"] == "yzk/repo"
    assert captured["proc"]["revision"] == "cafef00d"
    assert captured["proc"]["use_fast"] is False
    assert captured["model"]["name"] == "yzk/repo"
    assert captured["model"]["revision"] == "cafef00d"
    # tokenizer is taken from the processor on the base path.
    assert tokenizer is processor.tokenizer


def test_load_model_and_processor_revision_defaults_to_none(monkeypatch):
    # Unpinned base (no model.revision) forwards revision=None — the previous
    # behaviour and a no-op for a local dir.
    captured: dict = {}
    mod = _patch_from_pretrained(monkeypatch, captured)
    cfg = OmegaConf.create({"model": {"base": "local/dir"}})
    mod.load_model_and_processor(cfg)
    assert captured["proc"]["revision"] is None
    assert captured["model"]["revision"] is None
