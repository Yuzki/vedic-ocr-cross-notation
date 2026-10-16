"""Metrics: PACLIC 2026 v2 release."""

from functools import lru_cache

import evaluate


@lru_cache(maxsize=None)
def load_metric(name: str) -> "evaluate.EvaluationModule":
    """``evaluate.load`` memoized per process.

    ``evaluate.load`` performs a Hub round-trip on **every** call even when the
    module is already cached locally — measured at ~1 s each on this cluster. The
    offline analysis builders construct an :class:`OCRMetrics` per cell per view,
    so a 60-cell bundle paid 120 x 3 x ~1 s ≈ 6 min of pure network wait for
    metric objects that are immutable and identical every time (`/usr/bin/time -v`
    on the study builder: 6 min 40 s wall for 47 s of CPU, 0 local filesystem
    reads, blocked in ``ssl.read`` under ``requests``).

    The loaded metric objects are stateless with respect to ``compute`` — each
    call passes its own predictions/references — so sharing one instance across
    call sites is safe, and it makes the builders network-free after the first
    load rather than proportional to the cell count.
    """
    return evaluate.load(name)


class OCRMetrics:
    """Compute metrics for OCR evaluation.

    Supports CER, WER, CHRF, and Exact Match metrics.
    """

    def __init__(self, use_chrf: bool = True):
        """Initialize metrics.

        Args:
            use_chrf: Whether to compute CHRF score (slower but useful for morphologically rich languages)
        """
        self.cer_metric = load_metric("cer")
        self.wer_metric = load_metric("wer")
        self.chrf_metric = load_metric("chrf") if use_chrf else None

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Compute all metrics.

        Args:
            predictions: List of predicted strings
            references: List of reference strings

        Returns:
            Dictionary with metric names and values
        """
        # Drop only empty *references* (degenerate ground truth). Keep empty
        # predictions: a blank prediction is a total model failure that must be
        # charged as a full deletion in CER/WER, not removed from the denominator
        # (which artificially lowered CER and skewed model selection) — the study.
        valid_pairs = [
            (p.strip(), r.strip()) for p, r in zip(predictions, references) if r.strip()
        ]

        if not valid_pairs:
            return {
                "cer": 1.0,
                "wer": 1.0,
                "chrf": 0.0,
                "exact_match": 0.0,
            }

        preds, refs = zip(*valid_pairs)
        preds, refs = list(preds), list(refs)

        # Compute metrics
        results = {
            "cer": self.cer_metric.compute(predictions=preds, references=refs),  # ty: ignore[missing-argument]
            "wer": self.wer_metric.compute(predictions=preds, references=refs),  # ty: ignore[missing-argument]
        }

        # CHRF (optional). Test ``is not None`` explicitly: an ``evaluate`` module
        # with no buffered examples is falsy (``__len__`` returns 0), so a bare
        # ``if self.chrf_metric:`` silently skipped chrF on every real call and the
        # metric was never emitted (the study requires it).
        if self.chrf_metric is not None:
            chrf_result = self.chrf_metric.compute(predictions=preds, references=refs)  # ty: ignore[missing-argument]
            results["chrf"] = chrf_result["score"]  # ty: ignore[not-subscriptable]

        # Exact match accuracy
        exact_matches = sum(1 for p, r in zip(preds, refs) if p == r)
        results["exact_match"] = exact_matches / len(preds)  # ty: ignore[invalid-assignment]

        return results  # ty: ignore[invalid-return-type]


def create_compute_metrics_fn(processor, tokenizer=None, use_chrf: bool = True):
    """Create a compute_metrics function for Seq2SeqTrainer.

    Args:
        processor: TrOCRProcessor or image processor for decoding
        tokenizer: Optional separate tokenizer (if not using processor.tokenizer)
        use_chrf: Whether to compute CHRF score

    Returns:
        A compute_metrics function compatible with HuggingFace Trainer
    """
    metrics_calculator = OCRMetrics(use_chrf=use_chrf)

    # Determine which tokenizer to use for decoding
    if tokenizer is not None:
        _tokenizer = tokenizer
    elif hasattr(processor, "tokenizer"):
        _tokenizer = processor.tokenizer
    else:
        _tokenizer = processor

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Decode predictions
        if hasattr(processor, "batch_decode"):
            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        else:
            pred_str = _tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # Replace -100 with pad token id for decoding
        labels_ids[labels_ids == -100] = _tokenizer.pad_token_id

        if hasattr(processor, "batch_decode"):
            label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        else:
            label_str = _tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        return metrics_calculator.compute(pred_str, label_str)

    return compute_metrics
