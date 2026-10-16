"""Model freezing: PACLIC 2026 v2 release."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Public freeze modes. ``none`` is the default and must preserve full
# fine-tuning; ``encoder`` / ``decoder`` freeze that side and leave the other
# (plus any projection/output params outside both submodules) trainable.
VALID_FREEZE_MODES = ("none", "encoder", "decoder")


@dataclass(frozen=True)
class FreezeSummary:
    """Audit record for a single :func:`apply_freeze_mode` call.

    ``trainable_params`` is the count actually optimised; the per-side counts let
    a reviewer confirm the intended side was frozen and spot params that live
    outside ``.encoder`` / ``.decoder`` (``total - encoder - decoder``).
    """

    mode: str
    total_params: int
    trainable_params: int
    encoder_trainable_params: int
    decoder_trainable_params: int

    def describe(self) -> str:
        """One-line human summary for stdout / SLURM logs."""
        pct = (
            100.0 * self.trainable_params / self.total_params
            if self.total_params
            else 0.0
        )
        return (
            f"freeze={self.mode}: trainable {self.trainable_params:,}/"
            f"{self.total_params:,} ({pct:.1f}%) | "
            f"encoder_trainable={self.encoder_trainable_params:,} "
            f"decoder_trainable={self.decoder_trainable_params:,}"
        )

    def as_dict(self) -> dict:
        """Plain dict for ``results.json`` / WandB config serialisation."""
        return asdict(self)


def normalize_freeze_mode(mode) -> str:
    """Validate and canonicalise a freeze-mode value (fail fast on unknowns).

    ``None`` / empty / whitespace map to ``"none"`` so an absent or blank Hydra
    field behaves like full fine-tuning. Case and surrounding whitespace are
    ignored. Any other value raises :class:`ValueError` *before* training starts
    rather than silently training the wrong condition.
    """
    if mode is None:
        return "none"
    normalized = str(mode).strip().lower()
    if normalized == "":
        return "none"
    if normalized not in VALID_FREEZE_MODES:
        raise ValueError(
            f"Unknown freeze mode {mode!r}; expected one of "
            f"{VALID_FREEZE_MODES} (case-insensitive)"
        )
    return normalized


def _count_params(module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts for ``module``.

    Uses the module's own ``parameters()`` (which de-duplicates tied weights),
    so the model-level total counts shared embeddings once.
    """
    total = 0
    trainable = 0
    for param in module.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def apply_freeze_mode(model, mode) -> FreezeSummary:
    """Freeze the encoder or decoder of ``model`` in place; return a summary.

    ``mode`` is normalised via :func:`normalize_freeze_mode` (so an unknown mode
    raises before any parameter is touched):

    - ``none``: leave every parameter trainable (default; no-op besides counting);
    - ``encoder``: set ``model.encoder.parameters()`` to ``requires_grad=False``;
    - ``decoder``: set ``model.decoder.parameters()`` to ``requires_grad=False``.

    The returned :class:`FreezeSummary` is computed *after* the flip so the counts
    reflect the resulting ``requires_grad`` state. ``model`` is expected to expose
    ``.encoder`` and ``.decoder`` (the ``VisionEncoderDecoderModel`` contract).
    """
    normalized = normalize_freeze_mode(mode)

    if normalized == "encoder":
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif normalized == "decoder":
        for param in model.decoder.parameters():
            param.requires_grad = False
    # "none": every parameter stays trainable.

    total, trainable = _count_params(model)
    _, encoder_trainable = _count_params(model.encoder)
    _, decoder_trainable = _count_params(model.decoder)

    return FreezeSummary(
        mode=normalized,
        total_params=total,
        trainable_params=trainable,
        encoder_trainable_params=encoder_trainable,
        decoder_trainable_params=decoder_trainable,
    )
