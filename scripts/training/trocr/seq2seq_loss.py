"""Seq2seq loss: PACLIC 2026 v2 release."""

from __future__ import annotations

import torch
from transformers.loss.loss_utils import fixed_cross_entropy


def no_shift_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """Cross-entropy supervising ``logits[i] -> labels[i]`` (no extra shift).

    Signature matches ``transformers`` ``ForCausalLMLoss`` so it can be dropped
    in as ``model.loss_function``; the only difference is that it does *not*
    shift logits/labels, because ``VisionEncoderDecoderModel.forward`` has
    already right-shifted the decoder input via ``shift_tokens_right``.
    """
    logits = logits.float()
    labels = labels.to(logits.device)
    return fixed_cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        num_items_in_batch,
        ignore_index,
    )


def install_no_shift_loss(model):
    """Override ``model.loss_function`` with no-shift CE; return the model.

    Idempotent and cheap. Call once after loading an encoder-decoder model in
    the training path. ``loss_function`` is a runtime attribute (not persisted),
    so it only affects the live training/eval forward passes -- exactly where the
    double-shift would otherwise corrupt supervision.
    """
    model.loss_function = no_shift_cross_entropy_loss
    return model
