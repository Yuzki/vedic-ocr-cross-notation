"""Test trocr generation policy: PACLIC 2026 v2 release."""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf
from transformers import GenerationConfig

from scripts.training.trocr.generation_policy import (
    SOURCE_GENERATION_CONFIG,
    SOURCE_GENERIC_FALLBACK,
    SOURCE_MODEL_CONFIG,
    SOURCE_OVERRIDE,
    SOURCE_TROCR_FALLBACK,
    apply_generation_policy,
    collect_special_tokens,
    is_trocr_model,
    resolve_decoder_start_token_id,
)


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class FakeTokenizer:
    """Minimal tokenizer exposing only the special-token ids the policy reads.

    Defaults mirror the microsoft/trocr (RoBERTa) tokenizer: ``<s>``=0,
    ``</s>``=2, ``<pad>``=1, and no separate cls/sep.
    """

    def __init__(
        self,
        *,
        bos: int | None = 0,
        eos: int | None = 2,
        pad: int | None = 1,
        cls: int | None = None,
        sep: int | None = None,
        vocab_size: int | None = 64,
    ) -> None:
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad
        self.cls_token_id = cls
        self.sep_token_id = sep
        self.vocab_size = vocab_size


def make_model(
    *,
    model_type: str = "trocr",
    config_start: int | None = None,
    gen_start: int | None = None,
    has_gen_config: bool = True,
    **gen_fields,
):
    """A stand-in ``VisionEncoderDecoderModel`` exposing only what the policy touches.

    ``model.config.decoder.model_type`` drives :func:`is_trocr_model`;
    ``config_start`` / ``gen_start`` seed the two saved start-token locations.
    ``SimpleNamespace`` lets ``apply_generation_policy`` set attributes freely.
    """
    config = SimpleNamespace(
        decoder=SimpleNamespace(model_type=model_type),
        decoder_start_token_id=config_start,
    )
    if has_gen_config:
        generation_config = GenerationConfig(
            decoder_start_token_id=gen_start, **gen_fields
        )
    else:
        generation_config = None
    return SimpleNamespace(config=config, generation_config=generation_config)


# --------------------------------------------------------------------------- #
# is_trocr_model                                                              #
# --------------------------------------------------------------------------- #


def test_is_trocr_model_true_for_trocr_decoder():
    assert is_trocr_model(make_model(model_type="trocr")) is True


def test_is_trocr_model_false_for_roberta_decoder():
    assert is_trocr_model(make_model(model_type="roberta")) is False


def test_is_trocr_model_handles_missing_config_or_decoder():
    assert is_trocr_model(SimpleNamespace()) is False
    assert is_trocr_model(SimpleNamespace(config=SimpleNamespace())) is False


def test_is_trocr_model_against_real_hf_configs():
    """Guard the discriminator against an upstream change to decoder.model_type."""
    from transformers import (
        RobertaConfig,
        TrOCRConfig,
        VisionEncoderDecoderConfig,
        ViTConfig,
    )

    trocr = SimpleNamespace(
        config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            ViTConfig(), TrOCRConfig()
        ),
    )
    roberta = SimpleNamespace(
        config=VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            ViTConfig(), RobertaConfig()
        ),
    )
    assert is_trocr_model(trocr) is True
    assert is_trocr_model(roberta) is False


# --------------------------------------------------------------------------- #
# collect_special_tokens                                                      #
# --------------------------------------------------------------------------- #


def test_collect_special_tokens_start_end_like_roberta():
    st = collect_special_tokens(FakeTokenizer(bos=0, eos=2, pad=1))
    assert st.start_like == 0
    assert st.end_like == 2


def test_collect_special_tokens_bert_style_fallback():
    st = collect_special_tokens(FakeTokenizer(bos=None, eos=None, cls=101, sep=102))
    assert st.start_like == 101  # cls stands in for bos
    assert st.end_like == 102  # sep stands in for eos


# --------------------------------------------------------------------------- #
# resolve_decoder_start_token_id -- priority order                           #
# --------------------------------------------------------------------------- #


def test_preserves_saved_generation_config_start_token():
    # TrOCR checkpoint saved decoder_start_token_id=2; eval must keep 2, not bos.
    model = make_model(model_type="trocr", gen_start=2, config_start=2)
    res = resolve_decoder_start_token_id(model, FakeTokenizer())
    assert res.value == 2
    assert res.source == SOURCE_GENERATION_CONFIG
    assert res.is_trocr is True


def test_falls_back_to_model_config_when_gen_config_has_no_value():
    model = make_model(model_type="trocr", gen_start=None, config_start=2)
    res = resolve_decoder_start_token_id(model, FakeTokenizer())
    assert res.value == 2
    assert res.source == SOURCE_MODEL_CONFIG


def test_explicit_override_wins_over_saved_value():
    model = make_model(model_type="trocr", gen_start=2, config_start=2)
    res = resolve_decoder_start_token_id(model, FakeTokenizer(), override=0)
    assert res.value == 0
    assert res.source == SOURCE_OVERRIDE


def test_override_zero_is_honoured_not_skipped():
    # bos=0 is a legitimate id; treating 0 as "unset" was the latent `or` bug.
    model = make_model(model_type="trocr", gen_start=2)
    res = resolve_decoder_start_token_id(model, FakeTokenizer(), override=0)
    assert res.value == 0
    assert res.source == SOURCE_OVERRIDE


def test_trocr_fallback_is_eos_when_nothing_saved():
    model = make_model(model_type="trocr", gen_start=None, config_start=None)
    res = resolve_decoder_start_token_id(model, FakeTokenizer(bos=0, eos=2))
    assert res.value == 2  # eos, the TrOCR convention
    assert res.source == SOURCE_TROCR_FALLBACK


def test_trocr_fallback_uses_sep_when_tokenizer_has_no_eos():
    model = make_model(model_type="trocr", gen_start=None, config_start=None)
    res = resolve_decoder_start_token_id(
        model, FakeTokenizer(bos=None, eos=None, cls=101, sep=102)
    )
    assert res.value == 102  # sep
    assert res.source == SOURCE_TROCR_FALLBACK


def test_generic_fallback_is_bos_never_trocr_eos():
    # A custom ViT+RoBERTa decoder must NOT inherit the TrOCR eos-start rule.
    model = make_model(model_type="roberta", gen_start=None, config_start=None)
    res = resolve_decoder_start_token_id(model, FakeTokenizer(bos=0, eos=2))
    assert res.value == 0  # bos
    assert res.source == SOURCE_GENERIC_FALLBACK
    assert res.is_trocr is False


def test_generic_fallback_uses_cls_when_tokenizer_has_no_bos():
    model = make_model(model_type="roberta", gen_start=None, config_start=None)
    res = resolve_decoder_start_token_id(
        model, FakeTokenizer(bos=None, eos=None, cls=101, sep=102)
    )
    assert res.value == 101  # cls
    assert res.source == SOURCE_GENERIC_FALLBACK


# --------------------------------------------------------------------------- #
# apply_generation_policy -- writes a preserve-first GenerationConfig         #
# --------------------------------------------------------------------------- #


def test_apply_sets_resolved_start_on_generation_config_and_model_config():
    model = make_model(model_type="trocr", gen_start=2, config_start=2)
    res = apply_generation_policy(
        model, FakeTokenizer(), decode_overrides={"num_beams": 4}
    )
    assert res.value == 2
    assert model.generation_config.decoder_start_token_id == 2
    assert model.config.decoder_start_token_id == 2  # mirrored for right-shift
    assert model.generation_config.num_beams == 4


def test_apply_preserves_other_saved_generation_fields():
    # The whole point of preserve-first: a saved num_beams/length_penalty survives
    # while only the explicitly-passed knobs (and the start token) are updated.
    model = make_model(model_type="trocr", gen_start=2, num_beams=7, length_penalty=3.0)
    apply_generation_policy(
        model, FakeTokenizer(), decode_overrides={"max_length": 128}
    )
    assert model.generation_config.num_beams == 7  # preserved
    assert model.generation_config.length_penalty == 3.0  # preserved
    assert model.generation_config.max_length == 128  # overlaid
    assert model.generation_config.decoder_start_token_id == 2  # preserved


def test_apply_decode_override_wins_over_saved_value():
    model = make_model(model_type="trocr", gen_start=2, num_beams=7)
    apply_generation_policy(model, FakeTokenizer(), decode_overrides={"num_beams": 4})
    assert model.generation_config.num_beams == 4


def test_apply_skips_none_decode_overrides():
    model = make_model(model_type="trocr", gen_start=2, num_beams=7)
    apply_generation_policy(
        model, FakeTokenizer(), decode_overrides={"num_beams": None}
    )
    assert model.generation_config.num_beams == 7  # unset knob keeps saved value


def test_apply_explicit_override_changes_start_token():
    model = make_model(model_type="trocr", gen_start=2, config_start=2)
    res = apply_generation_policy(model, FakeTokenizer(), decoder_start_token_id=0)
    assert res.value == 0
    assert res.source == SOURCE_OVERRIDE
    assert model.generation_config.decoder_start_token_id == 0
    assert model.config.decoder_start_token_id == 0


def test_apply_builds_fresh_config_and_fills_special_tokens():
    model = make_model(
        model_type="trocr", gen_start=None, config_start=2, has_gen_config=False
    )
    res = apply_generation_policy(
        model, FakeTokenizer(), decode_overrides={"num_beams": 4}
    )
    assert res.value == 2
    assert res.source == SOURCE_MODEL_CONFIG
    assert model.generation_config.decoder_start_token_id == 2
    # Identity tokens filled from the tokenizer when the fresh config lacks them.
    assert model.generation_config.bos_token_id == 0
    assert model.generation_config.eos_token_id == 2
    assert model.generation_config.pad_token_id == 1


# --------------------------------------------------------------------------- #
# Wiring: train / test / experiment share one source of truth                 #
# --------------------------------------------------------------------------- #


def _generation_cfg(**overrides):
    base = {
        "max_length": 256,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "length_penalty": 2.0,
        "num_beams": 4,
        "decoder_start_token_id": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_test_and_experiment_paths_resolve_identically():
    from scripts.training.trocr import experiment_accent_ratio
    from scripts.training.trocr import experiment_accent_ratio as trocr_test

    cfg = _generation_cfg()
    model_t = make_model(model_type="trocr", gen_start=2, config_start=2)
    model_e = make_model(model_type="trocr", gen_start=2, config_start=2)

    res_t = trocr_test.configure_model_for_generation(model_t, FakeTokenizer(), cfg)
    res_e = experiment_accent_ratio.configure_model_for_generation(
        model_e, FakeTokenizer(), cfg
    )

    assert res_t.value == res_e.value == 2
    assert (
        model_t.generation_config.decoder_start_token_id
        == model_e.generation_config.decoder_start_token_id
        == 2
    )


def test_train_bakes_start_token_that_eval_then_preserves():
    from scripts.training.trocr import experiment_accent_ratio as trocr_test
    from scripts.training.trocr import experiment_accent_ratio as trocr_train

    cfg = _generation_cfg()
    # Fresh TrOCR base: generation_config ships decoder_start=2.
    base = make_model(model_type="trocr", gen_start=2, config_start=2)
    trocr_train.configure_model_for_generation(base, FakeTokenizer(), cfg)
    saved_start = base.config.decoder_start_token_id
    assert saved_start == 2

    # A checkpoint saved with that value is later evaluated by test.py -> kept.
    ckpt = make_model(
        model_type="trocr", gen_start=saved_start, config_start=saved_start
    )
    res = trocr_test.configure_model_for_generation(ckpt, FakeTokenizer(), cfg)
    assert res.value == saved_start == 2


def test_eval_path_no_longer_forces_bos_for_trocr_checkpoint():
    # Direct regression for the the study bug: the old test.py forced bos=0.
    from scripts.training.trocr import experiment_accent_ratio as trocr_test

    ckpt = make_model(model_type="trocr", gen_start=2, config_start=2)
    res = trocr_test.configure_model_for_generation(
        ckpt, FakeTokenizer(), _generation_cfg()
    )
    assert res.value == 2  # NOT 0 (bos)
    assert ckpt.generation_config.decoder_start_token_id == 2


def test_config_decoder_start_override_is_applied_on_eval_path():
    from scripts.training.trocr import experiment_accent_ratio as trocr_test

    cfg = _generation_cfg(decoder_start_token_id=0)
    ckpt = make_model(model_type="trocr", gen_start=2, config_start=2)
    res = trocr_test.configure_model_for_generation(ckpt, FakeTokenizer(), cfg)
    assert res.value == 0  # explicit override honoured
    assert res.source == SOURCE_OVERRIDE
    assert ckpt.generation_config.decoder_start_token_id == 0
