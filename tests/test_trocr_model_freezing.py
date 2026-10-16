"""Test trocr model freezing: PACLIC 2026 v2 release."""

import pytest
from torch import nn

from scripts.training.trocr.model_freezing import (
    FreezeSummary,
    apply_freeze_mode,
    normalize_freeze_mode,
)

# nn.Linear(3, 4) -> weight 4*3=12 + bias 4 = 16 params.
# nn.Linear(4, 2) -> weight 2*4=8  + bias 2 = 10 params.
ENCODER_PARAMS = 16
DECODER_PARAMS = 10
TOTAL_PARAMS = ENCODER_PARAMS + DECODER_PARAMS


class _TinyEncoderDecoder(nn.Module):
    """Minimal stand-in for ``VisionEncoderDecoderModel``: an ``.encoder`` and a
    ``.decoder`` submodule, each with real (small) parameters."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.decoder = nn.Linear(4, 2)


def _all_trainable(module) -> bool:
    return all(p.requires_grad for p in module.parameters())


def _none_trainable(module) -> bool:
    params = list(module.parameters())
    return bool(params) and all(not p.requires_grad for p in params)


def test_none_leaves_all_parameters_trainable():
    model = _TinyEncoderDecoder()
    summary = apply_freeze_mode(model, "none")

    assert _all_trainable(model.encoder)
    assert _all_trainable(model.decoder)
    assert summary.mode == "none"
    assert summary.total_params == TOTAL_PARAMS
    assert summary.trainable_params == TOTAL_PARAMS
    assert summary.encoder_trainable_params == ENCODER_PARAMS
    assert summary.decoder_trainable_params == DECODER_PARAMS


def test_encoder_freezes_only_encoder():
    model = _TinyEncoderDecoder()
    summary = apply_freeze_mode(model, "encoder")

    assert _none_trainable(model.encoder)
    assert _all_trainable(model.decoder)
    assert summary.mode == "encoder"
    assert summary.encoder_trainable_params == 0
    assert summary.decoder_trainable_params == DECODER_PARAMS
    assert summary.trainable_params == DECODER_PARAMS
    # Total is unaffected by freezing -- it counts existence, not grad state.
    assert summary.total_params == TOTAL_PARAMS


def test_decoder_freezes_only_decoder():
    model = _TinyEncoderDecoder()
    summary = apply_freeze_mode(model, "decoder")

    assert _all_trainable(model.encoder)
    assert _none_trainable(model.decoder)
    assert summary.mode == "decoder"
    assert summary.encoder_trainable_params == ENCODER_PARAMS
    assert summary.decoder_trainable_params == 0
    assert summary.trainable_params == ENCODER_PARAMS


def test_summary_counts_reflect_requires_grad_state():
    """The summary must be recomputed from the post-freeze ``requires_grad``
    state, i.e. equal to an independent recount of trainable params."""
    model = _TinyEncoderDecoder()
    summary = apply_freeze_mode(model, "encoder")

    recount = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert summary.trainable_params == recount
    enc_recount = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    assert summary.encoder_trainable_params == enc_recount


@pytest.mark.parametrize("bad_mode", ["bogus", "enc", "both", "freeze", "encoder!"])
def test_invalid_mode_raises_value_error(bad_mode):
    model = _TinyEncoderDecoder()
    with pytest.raises(ValueError, match="freeze mode"):
        apply_freeze_mode(model, bad_mode)


def test_invalid_mode_does_not_mutate_grad_state():
    """A bad mode must fail before any parameter is frozen."""
    model = _TinyEncoderDecoder()
    with pytest.raises(ValueError):
        apply_freeze_mode(model, "bogus")
    assert _all_trainable(model.encoder)
    assert _all_trainable(model.decoder)


def test_normalize_freeze_mode_canonicalises():
    assert normalize_freeze_mode(None) == "none"
    assert normalize_freeze_mode("") == "none"
    assert normalize_freeze_mode("   ") == "none"
    assert normalize_freeze_mode("ENCODER") == "encoder"
    assert normalize_freeze_mode("  Decoder  ") == "decoder"
    assert normalize_freeze_mode("none") == "none"


def test_normalize_freeze_mode_rejects_unknown():
    with pytest.raises(ValueError, match="freeze mode"):
        normalize_freeze_mode("partial")


def test_summary_describe_and_as_dict():
    model = _TinyEncoderDecoder()
    summary = apply_freeze_mode(model, "decoder")

    text = summary.describe()
    assert "freeze=decoder" in text
    assert isinstance(text, str)

    as_dict = summary.as_dict()
    assert as_dict == {
        "mode": "decoder",
        "total_params": TOTAL_PARAMS,
        "trainable_params": ENCODER_PARAMS,
        "encoder_trainable_params": ENCODER_PARAMS,
        "decoder_trainable_params": 0,
    }


def test_freeze_summary_is_frozen_dataclass():
    summary = FreezeSummary(
        mode="none",
        total_params=10,
        trainable_params=10,
        encoder_trainable_params=6,
        decoder_trainable_params=4,
    )
    with pytest.raises(Exception):
        summary.mode = "encoder"  # ty: ignore[invalid-assignment]
