"""Build paclic accent transfer analysis: PACLIC 2026 v2 release."""

from __future__ import annotations


import argparse


import csv


import hashlib


import io


import json


import math


import statistics


import sys


import unicodedata


from collections import Counter


from collections.abc import Iterable, Sequence


from pathlib import Path


from typing import Any, Optional


from . import aggregate_accent_ratio_multiseed as agg


from .prediction_metrics import (
    DEFAULT_NORMALIZATION,
    Normalization,
    iter_edit_events,
    load_prediction_rows,
)


ENUMERATION_MAX_N: int = 20


RESAMPLE_RNG_SEED: int = 20260721


N_RESAMPLES: int = 100_000


_PERM_EPS: float = 1e-12


def sign_flip_permutation_test(diffs: Sequence[float]) -> dict[str, Any]:
    """Exact paired sign-flip permutation test on oriented paired differences.

    ``diffs`` are per-pair differences already oriented so a positive value means
    "RV→MS is better" (see :func:`cross_notation_significance`). Under H0 each
    difference is symmetric about 0, so every sign vector ``s ∈ {+1,−1}**n`` is
    equally likely. The statistic is the signed sum; the observed value is
    ``sum(diffs)``.

    With ``n <= ENUMERATION_MAX_N`` all ``2**n`` assignments are enumerated, so the
    p-values are exact (this is the PACLIC path: ``n = 3`` → 8 assignments). Above
    that, a deterministic RNG (seed :data:`RESAMPLE_RNG_SEED`) draws
    :data:`N_RESAMPLES` sign vectors and the p-values use the ``(1 + b)/(1 + m)``
    Monte-Carlo estimator.

    Returns the two-sided p (``|T| >= |T_obs|``), the one-sided p in the
    RV→MS-better direction (``T >= T_obs``), and — for the exact path — the
    ``min_p_two_sided`` floor ``2 / 2**n`` (0.25 at ``n = 3``): the smallest
    two-sided p the test can *ever* return at this sample size.
    """
    vals = [float(d) for d in diffs]
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "exact": True,
            "enumerated": 0,
            "observed_sum": None,
            "observed_mean": None,
            "p_two_sided": None,
            "p_one_sided": None,
            "min_p_two_sided": None,
        }
    observed = math.fsum(vals)
    exact = n <= ENUMERATION_MAX_N
    if exact:
        total = 2**n
        ge = abs_ge = 0
        for mask in range(total):
            t = math.fsum((-vals[i] if (mask >> i) & 1 else vals[i]) for i in range(n))
            if t >= observed - _PERM_EPS:
                ge += 1
            if abs(t) >= abs(observed) - _PERM_EPS:
                abs_ge += 1
        return {
            "n": n,
            "exact": True,
            "enumerated": total,
            "observed_sum": observed,
            "observed_mean": observed / n,
            "p_two_sided": abs_ge / total,
            "p_one_sided": ge / total,
            "min_p_two_sided": 2 / total,
        }
    # Generalized large-n path (never hit at n=3; kept deterministic + honest).
    import random

    rng = random.Random(RESAMPLE_RNG_SEED)
    ge = abs_ge = 0
    for _ in range(N_RESAMPLES):
        t = math.fsum((v if rng.random() < 0.5 else -v) for v in vals)
        if t >= observed - _PERM_EPS:
            ge += 1
        if abs(t) >= abs(observed) - _PERM_EPS:
            abs_ge += 1
    return {
        "n": n,
        "exact": False,
        "enumerated": N_RESAMPLES,
        "observed_sum": observed,
        "observed_mean": observed / n,
        "p_two_sided": (1 + abs_ge) / (1 + N_RESAMPLES),
        "p_one_sided": (1 + ge) / (1 + N_RESAMPLES),
        "min_p_two_sided": None,
    }


PLATEAU_RATIOS: tuple[float, ...] = (0.75, 0.5, 0.25)


ERROR_CLASSES: tuple[str, ...] = (
    "accent_insertion",
    "accent_deletion",
    "accent_substitution",
    "base_insertion",
    "base_deletion",
    "base_substitution",
)


def direction_of(rv_ratio: float, view: str) -> str:
    """Map a (ratio, view) cell to its transfer direction label."""
    if rv_ratio == 1.0:
        return "in_domain_rv" if view == "rv" else "rv_to_ms"
    if rv_ratio == 0.0:
        return "ms_to_rv" if view == "rv" else "in_domain_ms"
    return "mixed"

