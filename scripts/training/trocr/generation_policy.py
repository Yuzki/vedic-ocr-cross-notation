"""Generation policy: PACLIC 2026 v2 release."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from transformers import GenerationConfig

# Resolution source labels, highest priority first. Exposed as constants so the
# call sites and tests refer to the same strings instead of bare literals.
SOURCE_OVERRIDE = "override"
SOURCE_GENERATION_CONFIG = "generation_config"
SOURCE_MODEL_CONFIG = "model_config"
SOURCE_TROCR_FALLBACK = "trocr_fallback"
SOURCE_GENERIC_FALLBACK = "generic_fallback"


@dataclass(frozen=True)
class SpecialTokens:
    """The special token ids a tokenizer exposes (any may be ``None``)."""

    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    cls_token_id: int | None
    sep_token_id: int | None

    @property
    def start_like(self) -> int | None:
        """BOS, or CLS when the tokenizer has no BOS (BERT-style decoders)."""
        return self.bos_token_id if self.bos_token_id is not None else self.cls_token_id

    @property
    def end_like(self) -> int | None:
        """EOS, or SEP when the tokenizer has no EOS (BERT-style decoders)."""
        return self.eos_token_id if self.eos_token_id is not None else self.sep_token_id


def collect_special_tokens(tokenizer) -> SpecialTokens:
    """Read the special-token ids off ``tokenizer`` defensively (missing -> ``None``)."""
    return SpecialTokens(
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        cls_token_id=getattr(tokenizer, "cls_token_id", None),
        sep_token_id=getattr(tokenizer, "sep_token_id", None),
    )


def is_trocr_model(model) -> bool:
    """Return ``True`` for a microsoft/trocr-style ``VisionEncoderDecoderModel``.

    Such a model exposes its decoder sub-config at ``model.config.decoder``; for
    the TrOCR family that sub-config is a ``TrOCRConfig`` whose ``model_type`` is
    ``"trocr"``. A custom ViT+RoBERTa decoder reports ``"roberta"`` (or
    ``"bert"``). Keying off this keeps the TrOCR EOS-start *fallback* from ever
    being applied to a custom decoder (the study risk: TrOCR / custom mix-up).
    """
    decoder_cfg = getattr(getattr(model, "config", None), "decoder", None)
    return getattr(decoder_cfg, "model_type", None) == "trocr"


@dataclass(frozen=True)
class StartTokenResolution:
    """Outcome of resolving ``decoder_start_token_id`` for a single model."""

    value: int | None
    source: str
    is_trocr: bool

    def describe(self) -> str:
        """One-line, log-friendly summary of the decision."""
        family = "trocr" if self.is_trocr else "generic"
        return (
            f"decoder_start_token_id={self.value} "
            f"(source={self.source}, family={family})"
        )


def resolve_decoder_start_token_id(
    model,
    tokenizer,
    *,
    override: int | None = None,
) -> StartTokenResolution:
    """Resolve ``decoder_start_token_id`` with a preserve-first policy.

    Priority (highest first); the chosen step is reported as
    :attr:`StartTokenResolution.source`:

    1. ``override`` (``SOURCE_OVERRIDE``) -- an explicit value from config/CLI.
    2. ``model.generation_config.decoder_start_token_id``
       (``SOURCE_GENERATION_CONFIG``) -- the value the checkpoint was saved with.
    3. ``model.config.decoder_start_token_id`` (``SOURCE_MODEL_CONFIG``) -- the
       model-config mirror of the same.
    4. TrOCR fallback (``SOURCE_TROCR_FALLBACK``): EOS, else SEP. microsoft/trocr
       decoders start generation from ``</s>`` (id 2).
    5. Generic fallback (``SOURCE_GENERIC_FALLBACK``): BOS, else CLS. A custom
       ViT+RoBERTa decoder starts from ``<s>``.

    Steps 2-3 are what make the eval scripts agree with how the checkpoint was
    trained instead of silently forcing BOS (the the study bug). Steps 4-5 only
    matter for a freshly loaded base model with no saved start token.

    ``None``-checks use ``is not None`` throughout so a legitimate id of ``0``
    (BOS for the TrOCR tokenizer) is never skipped.
    """
    is_trocr = is_trocr_model(model)

    if override is not None:
        return StartTokenResolution(int(override), SOURCE_OVERRIDE, is_trocr)

    gen_cfg = getattr(model, "generation_config", None)
    gen_val = getattr(gen_cfg, "decoder_start_token_id", None)
    if gen_val is not None:
        return StartTokenResolution(int(gen_val), SOURCE_GENERATION_CONFIG, is_trocr)

    model_cfg = getattr(model, "config", None)
    cfg_val = getattr(model_cfg, "decoder_start_token_id", None)
    if cfg_val is not None:
        return StartTokenResolution(int(cfg_val), SOURCE_MODEL_CONFIG, is_trocr)

    tokens = collect_special_tokens(tokenizer)
    if is_trocr:
        return StartTokenResolution(tokens.end_like, SOURCE_TROCR_FALLBACK, is_trocr)
    return StartTokenResolution(tokens.start_like, SOURCE_GENERIC_FALLBACK, is_trocr)


def apply_generation_policy(
    model,
    tokenizer,
    *,
    decode_overrides: dict | None = None,
    decoder_start_token_id: int | None = None,
    preserve_existing: bool = True,
) -> StartTokenResolution:
    """Resolve the start token and write a preserve-first ``GenerationConfig``.

    - ``decoder_start_token_id`` is set to the value chosen by
      :func:`resolve_decoder_start_token_id` (``decoder_start_token_id`` here is
      the *explicit override*; ``None`` means "decide by policy").
    - When ``preserve_existing`` is true and the model already carries a
      ``GenerationConfig`` (e.g. one loaded from a checkpoint's
      ``generation_config.json``), it is **deep-copied** and only the fields we
      explicitly manage are overwritten, so saved decode settings (``num_beams``,
      ``length_penalty``, ...) survive instead of being silently reset (the study
      risk: over-writing loaded config).
    - ``decode_overrides`` (e.g. the Hydra ``generation`` block) win over saved
      values; ``None`` entries are skipped so an unset knob keeps the saved value.
    - Identity special tokens (bos/eos/pad) are filled only when the config does
      not already carry them, again to avoid clobbering fine-tuned values.

    The chosen ``decoder_start_token_id`` (and the special tokens) are mirrored
    onto ``model.config`` as well, because the training-time right-shift
    (``shift_tokens_right``) reads ``model.config.decoder_start_token_id`` while
    ``generate()`` reads ``model.generation_config`` -- both must agree.

    Returns the :class:`StartTokenResolution` so callers can log which value was
    used and why.
    """
    resolution = resolve_decoder_start_token_id(
        model, tokenizer, override=decoder_start_token_id
    )

    existing = getattr(model, "generation_config", None)
    if preserve_existing and isinstance(existing, GenerationConfig):
        gen_cfg = copy.deepcopy(existing)
    else:
        gen_cfg = GenerationConfig()

    # The one field the study is about: always set to the resolved value.
    gen_cfg.decoder_start_token_id = resolution.value

    # Fill identity tokens only when absent so we never clobber fine-tuned values.
    # (``vocab_size`` is deliberately not propagated: it is not a generation
    # parameter -- ``generate()`` takes the vocabulary from ``model.config`` --
    # so the old code's ``GenerationConfig(vocab_size=...)`` was vestigial.)
    tokens = collect_special_tokens(tokenizer)
    if getattr(gen_cfg, "bos_token_id", None) is None:
        gen_cfg.bos_token_id = tokens.start_like
    if getattr(gen_cfg, "eos_token_id", None) is None:
        gen_cfg.eos_token_id = tokens.end_like
    if getattr(gen_cfg, "pad_token_id", None) is None:
        gen_cfg.pad_token_id = tokens.pad_token_id

    # Caller-provided decode knobs win over saved values; skip unset (None) ones.
    for key, value in (decode_overrides or {}).items():
        if value is not None:
            setattr(gen_cfg, key, value)

    model.generation_config = gen_cfg

    # Keep model.config in sync with the generation config (right-shift + any
    # code path that still reads model.config.*).
    model.config.decoder_start_token_id = gen_cfg.decoder_start_token_id
    if gen_cfg.bos_token_id is not None:
        model.config.bos_token_id = gen_cfg.bos_token_id
    if gen_cfg.eos_token_id is not None:
        model.config.eos_token_id = gen_cfg.eos_token_id
    if gen_cfg.pad_token_id is not None:
        model.config.pad_token_id = gen_cfg.pad_token_id

    return resolution
