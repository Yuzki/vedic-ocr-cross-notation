"""Test trocr metrics: PACLIC 2026 v2 release."""

from __future__ import annotations

import pytest

from scripts.training.trocr.metrics import OCRMetrics


class _RecordingMetric:
    """Fake ``evaluate`` metric recording the (predictions, references) it sees."""

    def __init__(self, value: float = 0.0):
        self.value = value
        self.calls: list[tuple[list[str], list[str]]] = []

    def compute(self, predictions, references):
        self.calls.append((list(predictions), list(references)))
        return self.value


def _metrics_with_fakes() -> tuple[OCRMetrics, _RecordingMetric, _RecordingMetric]:
    """An ``OCRMetrics`` with recording fakes for cer/wer and no chrf.

    The fakes are returned separately so assertions read their typed ``.calls``
    rather than ``m.cer_metric`` (declared as the evaluate ``EvaluationModule``).
    """
    m = object.__new__(OCRMetrics)
    cer = _RecordingMetric()
    wer = _RecordingMetric()
    m.cer_metric = cer
    m.wer_metric = wer
    m.chrf_metric = None
    return m, cer, wer


def test_empty_prediction_is_kept_for_metric():
    m, cer, wer = _metrics_with_fakes()
    m.compute(["", "abxd"], ["abcd", "abcd"])
    # The empty-prediction row reaches the CER/WER backend (not filtered) (study).
    assert cer.calls[0] == (["", "abxd"], ["abcd", "abcd"])
    assert wer.calls[0] == (["", "abxd"], ["abcd", "abcd"])


def test_whitespace_only_prediction_treated_as_empty_but_kept():
    m, cer, _ = _metrics_with_fakes()
    m.compute([" ", "abcd"], ["abcd", "abcd"])
    assert cer.calls[0] == (["", "abcd"], ["abcd", "abcd"])


def test_empty_reference_is_dropped():
    m, cer, _ = _metrics_with_fakes()
    m.compute(["x", "abcd"], ["", "abcd"])
    # Degenerate ground truth (empty reference) is the only thing dropped.
    assert cer.calls[0] == (["abcd"], ["abcd"])


def test_exact_match_denominator_includes_empty_prediction():
    m, _, _ = _metrics_with_fakes()
    res = m.compute(["", "abcd"], ["abcd", "abcd"])
    # 1 exact match out of 2 kept rows; the empty-pred row counts in the total.
    assert res["exact_match"] == 0.5


def test_all_empty_references_returns_fallback_without_calling_backend():
    m, cer, _ = _metrics_with_fakes()
    res = m.compute(["x", "y"], ["", "  "])
    assert res == {"cer": 1.0, "wer": 1.0, "chrf": 0.0, "exact_match": 0.0}
    assert cer.calls == []


def test_real_backend_charges_empty_prediction():
    # Real evaluate/jiwer backend: keeping the empty-pred row raises corpus CER
    # vs dropping it (the the study probe). 5 errors (4 del + 1 sub) / 8 ref chars.
    m = OCRMetrics(use_chrf=False)
    with_empty = m.compute(["", "abxd"], ["abcd", "abcd"])["cer"]
    without_empty = m.compute(["abxd"], ["abcd"])["cer"]
    assert with_empty == pytest.approx(0.625)
    assert with_empty > without_empty


def test_real_backend_emits_chrf_when_enabled():
    # Regression: an evaluate module with no buffered examples is falsy, so a bare
    # ``if self.chrf_metric:`` skipped chrF on every real call. use_chrf=True must
    # actually produce a numeric chrF .
    m = OCRMetrics(use_chrf=True)
    res = m.compute(["abcd", "abxd"], ["abcd", "abcd"])
    assert "chrf" in res
    assert isinstance(res["chrf"], float)
    assert 0.0 <= res["chrf"] <= 100.0
