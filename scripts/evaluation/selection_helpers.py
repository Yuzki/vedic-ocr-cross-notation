"""Checkpoint selection sensitivity: PACLIC 2026 v2 release."""

from __future__ import annotations


import argparse


import csv


import io


import json


import sys


from collections.abc import Sequence


from dataclasses import dataclass, field, replace


from pathlib import Path


from typing import Any, Optional


from . import aggregate_accent_ratio_multiseed as agg


from .aggregate_accent_ratio_multiseed import STEPS_PER_EPOCH


CRITERIA: tuple[str, ...] = ("current", "mean_val_cer", "worst_val_cer")


SENSITIVITY_CRITERIA: tuple[str, ...] = ("mean_val_cer", "worst_val_cer")


REPORT_METRICS: tuple[str, ...] = (
    "rv_cer",
    "ms_cer",
    "avg_cer",
    "worst_view_cer",
    "rv_accent_f1",
    "ms_accent_f1",
    "worst_accent_f1",
)


def _mean2(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else (a + b) / 2.0


def _worst_cer(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Worst (largest) of two view CERs; ``None`` unless both are present."""
    return None if a is None or b is None else max(a, b)


def _worst_accent_f1(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Worst (lowest) of two view accent F1s; ``None`` unless both are present."""
    return None if a is None or b is None else min(a, b)


def derive_report_metrics(
    *,
    rv_cer: Optional[float],
    ms_cer: Optional[float],
    rv_accent_f1: Optional[float],
    ms_accent_f1: Optional[float],
    avg_cer: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """The seven :data:`REPORT_METRICS` from the four per-view test numbers.

    The single place ``avg_cer`` / ``worst_view_cer`` / ``worst_accent_f1`` are
    derived, so a cell scored from its committed ``results.json`` and a cell
    scored from a re-inference record go through *identical* arithmetic. A missing
    input stays ``None`` — never 0, never carried over from another epoch.
    """
    return {
        "rv_cer": rv_cer,
        "ms_cer": ms_cer,
        "avg_cer": avg_cer if avg_cer is not None else _mean2(rv_cer, ms_cer),
        "worst_view_cer": _worst_cer(rv_cer, ms_cer),
        "rv_accent_f1": rv_accent_f1,
        "ms_accent_f1": ms_accent_f1,
        "worst_accent_f1": _worst_accent_f1(rv_accent_f1, ms_accent_f1),
    }


def report_metrics(cell: agg.SeedCell) -> dict[str, Optional[float]]:
    """The seven study test metrics for one cell, from its committed ``results.json``."""
    return derive_report_metrics(
        rv_cer=cell.rv.get("cer"),
        ms_cer=cell.ms.get("cer"),
        rv_accent_f1=cell.rv.get("accent_f1"),
        ms_accent_f1=cell.ms.get("accent_f1"),
        avg_cer=cell.avg_cer,
    )


def rank_ratios(per_ratio: dict[float, dict[str, Any]], metric: str) -> dict[str, Any]:
    """Internal-ratio ranking (ascending mean of ``metric``; lower = better).

    Ratios with no computable cell (no mean) are omitted from the order and
    counted, so ``complete`` is true only when all five ratios rank with the full
    n=3 — otherwise the ranking is explicitly partial.
    """
    ranked = [
        (rv, per_ratio[rv]["stats"][metric].mean, per_ratio[rv]["n"])
        for rv in agg.RV_RATIOS
        if per_ratio[rv]["stats"][metric].mean is not None
    ]
    ranked.sort(key=lambda t: t[1])
    complete = len(ranked) == len(agg.RV_RATIOS) and all(
        n == len(agg.SEEDS) for _, _, n in ranked
    )
    return {
        "metric": metric,
        "order": [{"rv_ratio": rv, "mean": mean, "n": n} for rv, mean, n in ranked],
        "n_ratios_ranked": len(ranked),
        "n_ratios_total": len(agg.RV_RATIOS),
        "complete": complete,
    }

