"""Build paclic v2 effect sizes and plateau: PACLIC 2026 v2 release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import aggregate_accent_ratio_multiseed as agg
from .build_paclic_accent_transfer_analysis import sign_flip_permutation_test

# --------------------------------------------------------------------------- #
# Study identity + inputs.
# --------------------------------------------------------------------------- #
#: Study slug, used in log prefixes and stamped into every artifact.
STUDY: str = "paclic-v2-effect-sizes-and-plateau"
#: The issue whose statistics half this bundle closes.
ISSUE: int = 125
#: The bundles this one reads. Both are committed analysis artifacts, not run dirs:
#: this module never touches ``outputs/``, ``data/`` or external run directories.
PARENT_STUDY: str = "paclic-v2-maingrid"
SELECTION_STUDY: str = "paclic-v2-checkpoint-selection-sensitivity"

_EXPERIMENTS = Path("results")
DEFAULT_MAINGRID_DIR: Path = _EXPERIMENTS / PARENT_STUDY
DEFAULT_SELECTION_DIR: Path = _EXPERIMENTS / SELECTION_STUDY
DEFAULT_OUT_DIR: Path = _EXPERIMENTS / STUDY

#: The corpus both inputs must attest. A v1 bundle silently substituted here would
#: otherwise produce v2-labelled effect sizes from abandoned pixels.
EXPECTED_CORPUS: str = "v2"
#: The selection-sensitivity criterion whose rows are the *reported* rule — the one
#: the committed predictions were decoded under. The uniform ``mean_val_cer`` /
#: ``worst_val_cer`` criteria are a different (and incomplete) grid; see study.
SELECTION_CRITERION: str = "current"

#: Seeds and the interior ratios the plateau question is about.
SEEDS: tuple[int, ...] = agg.SEEDS
RV_RATIOS: tuple[float, ...] = agg.RV_RATIOS
INTERIOR_RV_RATIOS: tuple[float, ...] = (0.75, 0.5, 0.25)
#: The target cell count behind every number here (all five ratios are needed: the
#: endpoints for the contrasts, the interior three for the plateau).
TARGET_N_CELLS: int = len(RV_RATIOS) * len(SEEDS)

# --------------------------------------------------------------------------- #
# Reconciliation tolerances — set by how the *published* table prints, not by taste.
# --------------------------------------------------------------------------- #
#: ``per_seed_diff`` and ``per_ratio.csv``'s ``values`` are printed with ``%.6f``, so
#: a re-derived value may differ from the printed one by up to 5e-7 per operand.
PRINTED_6DP_TOLERANCE: float = 1e-6
#: ``diff_mean`` / ``diff_ci95_*`` are printed with ``%.8f``.
PRINTED_8DP_TOLERANCE: float = 1e-8
#: ``scorable_per_ratio.csv`` prints ``repr``-precision floats, so the worst-view
#: reconciliation is exact up to float round-off in the mean.
FULL_PRECISION_TOLERANCE: float = 1e-12

# --------------------------------------------------------------------------- #
# Metric properties. Only two things are ever needed: which direction is better,
# and the metric's range (used solely to flag a floor-level relative-effect
# denominator, never to rescale a reported value).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricSpec:
    """Which direction is better for one reported metric."""

    name: str
    higher_is_better: bool

    def better(self, a: float, b: float) -> str:
        """``"a"`` or ``"b"`` — which of two arm means is the better reading."""
        if a == b:
            return "tie"
        if self.higher_is_better:
            return "a" if a > b else "b"
        return "a" if a < b else "b"


METRICS: dict[str, MetricSpec] = {
    "cer": MetricSpec("cer", higher_is_better=False),
    "wer": MetricSpec("wer", higher_is_better=False),
    "chrf": MetricSpec("chrf", higher_is_better=True),
    "accent_f1": MetricSpec("accent_f1", higher_is_better=True),
    "accent_stripped_cer": MetricSpec("accent_stripped_cer", higher_is_better=False),
}

# --------------------------------------------------------------------------- #
# The eight published endpoint contrasts.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Arm:
    """One side of a paired contrast: a (mix ratio, test view, metric) reading."""

    rv_ratio: float
    view: str
    metric: str
    label: str

    @property
    def column(self) -> str:
        """The ``per_seed.csv`` column holding this reading."""
        return f"{self.view}_{self.metric}"


@dataclass(frozen=True)
class ContrastSpec:
    """One published paired contrast, and how its effect sizes are defined.

    ``comparison`` is the published table's ``comparison`` cell **verbatim**: it is
    the join key against ``asymmetry.csv`` / ``retention_adaptation.csv``, so a typo
    fails loudly instead of producing an unreconciled row. ``a`` / ``b`` are the arms
    in that label's own order, so the reported difference is always ``b − a`` exactly
    as published. ``reference`` names the arm that is the relative effect's
    denominator, with the reason it is that arm rather than the other.
    """

    comparison: str
    table: str
    a: Arm
    b: Arm
    reference: str
    reference_rationale: str

    def __post_init__(self) -> None:
        if self.reference not in ("a", "b"):
            raise ValueError(f"reference must be 'a' or 'b', got {self.reference!r}")
        if self.a.metric != self.b.metric:
            raise ValueError(f"{self.comparison}: arms differ in metric")

    @property
    def metric(self) -> str:
        return self.a.metric

    @property
    def spec(self) -> MetricSpec:
        return METRICS[self.metric]

    def reference_arm(self) -> Arm:
        return self.a if self.reference == "a" else self.b


_ASYMMETRY_TABLE = "asymmetry.csv"
_RETENTION_TABLE = "retention_adaptation.csv"

#: Rationale strings, factored out so the same sentence cannot drift between rows.
_ZERO_SHOT_REFERENCE_RATIONALE = (
    "the MS→RV arm, preserving the pre-registered v1 denominator for this contrast "
    "(paclic-accent-transfer-analysis/manifest.json statistical_test.effect_sizes)"
)
_BASELINE_REFERENCE_RATIONALE = (
    "the single-notation baseline arm, i.e. the reading the minority share is added to"
)
_IN_DOMAIN_REFERENCE_RATIONALE = (
    "the RV-own arm, the first term of the published label's subtraction"
)

CONTRASTS: tuple[ContrastSpec, ...] = (
    ContrastSpec(
        comparison="zero_shot_cer(MS→RV − RV→MS)",
        table=_ASYMMETRY_TABLE,
        a=Arm(1.0, "ms", "cer", "RV→MS (RV-only model on MS test)"),
        b=Arm(0.0, "rv", "cer", "MS→RV (MS-only model on RV test)"),
        reference="b",
        reference_rationale=_ZERO_SHOT_REFERENCE_RATIONALE,
    ),
    ContrastSpec(
        comparison="zero_shot_chrf(MS→RV − RV→MS)",
        table=_ASYMMETRY_TABLE,
        a=Arm(1.0, "ms", "chrf", "RV→MS (RV-only model on MS test)"),
        b=Arm(0.0, "rv", "chrf", "MS→RV (MS-only model on RV test)"),
        reference="b",
        reference_rationale=_ZERO_SHOT_REFERENCE_RATIONALE,
    ),
    ContrastSpec(
        comparison="zero_shot_accent_f1(MS→RV − RV→MS)",
        table=_ASYMMETRY_TABLE,
        a=Arm(1.0, "ms", "accent_f1", "RV→MS (RV-only model on MS test)"),
        b=Arm(0.0, "rv", "accent_f1", "MS→RV (MS-only model on RV test)"),
        reference="b",
        reference_rationale=_ZERO_SHOT_REFERENCE_RATIONALE,
    ),
    ContrastSpec(
        comparison="in_domain_cer(MS_own − RV_own)",
        table=_ASYMMETRY_TABLE,
        a=Arm(1.0, "rv", "cer", "RV-own (RV-only model on RV test)"),
        b=Arm(0.0, "ms", "cer", "MS-own (MS-only model on MS test)"),
        reference="a",
        reference_rationale=_IN_DOMAIN_REFERENCE_RATIONALE,
    ),
    ContrastSpec(
        comparison="rv_primary_add_ms:rv_retention_cer",
        table=_RETENTION_TABLE,
        a=Arm(1.0, "rv", "cer", "RV-only baseline on RV test"),
        b=Arm(0.75, "rv", "cer", "RV-primary +25% MS on RV test"),
        reference="a",
        reference_rationale=_BASELINE_REFERENCE_RATIONALE,
    ),
    ContrastSpec(
        comparison="rv_primary_add_ms:ms_adaptation_accent_f1",
        table=_RETENTION_TABLE,
        a=Arm(1.0, "ms", "accent_f1", "RV-only baseline on MS test"),
        b=Arm(0.75, "ms", "accent_f1", "RV-primary +25% MS on MS test"),
        reference="a",
        reference_rationale=_BASELINE_REFERENCE_RATIONALE,
    ),
    ContrastSpec(
        comparison="ms_primary_add_rv:ms_retention_cer",
        table=_RETENTION_TABLE,
        a=Arm(0.0, "ms", "cer", "MS-only baseline on MS test"),
        b=Arm(0.25, "ms", "cer", "MS-primary +25% RV on MS test"),
        reference="a",
        reference_rationale=_BASELINE_REFERENCE_RATIONALE,
    ),
    ContrastSpec(
        comparison="ms_primary_add_rv:rv_adaptation_accent_f1",
        table=_RETENTION_TABLE,
        a=Arm(0.0, "rv", "accent_f1", "MS-only baseline on RV test"),
        b=Arm(0.25, "rv", "accent_f1", "MS-primary +25% RV on RV test"),
        reference="a",
        reference_rationale=_BASELINE_REFERENCE_RATIONALE,
    ),
)

# --------------------------------------------------------------------------- #
# The plateau quantities.
# --------------------------------------------------------------------------- #
#: Combine the two test views of one cell into the mean-view or the worst-view
#: reading. "Worst" is direction-aware: the larger error, the smaller score.
MODE_MEAN: str = "mean_of_views"
MODE_WORST: str = "worst_view"


@dataclass(frozen=True)
class PlateauQuantity:
    """One per-cell derived quantity of the plateau table."""

    name: str
    metric: str
    mode: str
    #: Column name in ``scorable_per_ratio.csv`` that publishes the same quantity at
    #: n=3, or ``None`` when this bundle is the first to derive it. Reported per row;
    #: a quantity with no published counterpart is marked, never presented as checked.
    published_as: Optional[str]

    @property
    def spec(self) -> MetricSpec:
        return METRICS[self.metric]


#: Names follow v1's ``plateau_comparison.csv`` (``avg_*`` / ``worst_*``) so the table
#: shape is recognisable; the *published* counterpart names differ and are carried in
#: ``published_as`` rather than guessed at read time.
PLATEAU_QUANTITIES: tuple[PlateauQuantity, ...] = (
    PlateauQuantity("avg_cer", "cer", MODE_MEAN, "avg_cer"),
    PlateauQuantity("worst_cer", "cer", MODE_WORST, "worst_view_cer"),
    PlateauQuantity("avg_wer", "wer", MODE_MEAN, None),
    PlateauQuantity("worst_wer", "wer", MODE_WORST, None),
    PlateauQuantity("avg_accent_f1", "accent_f1", MODE_MEAN, None),
    PlateauQuantity("worst_accent_f1", "accent_f1", MODE_WORST, "worst_accent_f1"),
    PlateauQuantity("avg_accent_stripped_cer", "accent_stripped_cer", MODE_MEAN, None),
    PlateauQuantity(
        "worst_accent_stripped_cer", "accent_stripped_cer", MODE_WORST, None
    ),
)

#: The quantity the ranking is on, and its tie-break. Worst-view CER first: the
#: plateau question is "which mix is safe to deploy on either notation".
RANK_PRIMARY: str = "worst_cer"
RANK_TIEBREAK: str = "avg_cer"

#: The three pairs of interior ratios, as ``(a, b)`` so the contrast is ``b − a``.
PLATEAU_PAIRS: tuple[tuple[float, float], ...] = (
    (0.75, 0.5),
    (0.75, 0.25),
    (0.5, 0.25),
)

#: Verdicts for one pair. Neither says "separable": at n=3 none is available.
ORDER_CONSISTENT: str = "consistently_ordered_but_not_separable"
ORDER_UNSTABLE: str = "order_unstable_across_seeds"

#: Why no pair can be separable here, quoted into every artifact.
SEPARABILITY_NOTE: str = (
    "No pair is statistically separable at n=3. The exact paired sign-flip "
    "permutation test's smallest attainable two-sided p is 2/2**3 = 0.25, so the "
    "ordering of two ratios cannot be established in either direction; a pair is "
    "classified only as consistently ordered across the three seeds (a description "
    "of those seeds) or order-unstable."
)

#: Why *dz* gets no interval.
NO_DZ_INTERVAL_REASON: str = (
    "No interval is reported for dz. A defensible interval for a standardized paired "
    "difference requires the noncentral-t distribution, which is not available here "
    "and is deliberately not approximated: at n=3 an invented interval would be worse "
    "than none. The unstandardized difference's Student-t 95% CI (published, and "
    "reproduced in this table) is the interval on the quantity the paper quotes; the "
    "leave-one-seed-out refits below show how unstable dz itself is, and are NOT a "
    "confidence interval."
)


class InputError(RuntimeError):
    """An input bundle is absent, unreadable, or missing a required column/row."""


# --------------------------------------------------------------------------- #
# Reading the published tables.
# --------------------------------------------------------------------------- #
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Every row of a committed CSV as ``{column: text}``."""
    if not path.is_file():
        raise InputError(f"required input table is absent: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    """A committed JSON payload (used for the two inputs' own status blocks)."""
    if not path.is_file():
        raise InputError(f"required input payload is absent: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt committed file
        raise InputError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputError(f"{path} is not a JSON object")
    return payload


def sha256_of(path: Path) -> Optional[str]:
    """Content digest of an input file, so provenance is checkable, not asserted."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_per_seed_diff(text: str) -> dict[int, float]:
    """``"42:0.034;43:0.031"`` → ``{42: 0.034, 43: 0.031}``.

    The published format is ``agg._contrast_rows``' ``f"{seed}:{diff:.6f}"`` join, so
    the parsed values carry 6 decimals — which is why they are reconciled against,
    rather than used as, this bundle's inputs.
    """
    out: dict[int, float] = {}
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        seed_text, _, value_text = token.partition(":")
        if not value_text:
            raise InputError(f"malformed per_seed_diff token {token!r}")
        out[int(seed_text)] = float(value_text)
    return out


def parse_values(text: str) -> list[float]:
    """``per_ratio.csv``'s ``values`` column → its floats, in published order."""
    return [float(tok) for tok in text.split(";") if tok.strip()]


@dataclass(frozen=True)
class PublishedCell:
    """One published ``(rv_ratio, seed)`` cell of the v2 main grid.

    Read from ``per_seed.csv`` — the only seed-labelled per-cell table — and holding
    just the view metrics the two deliverables need.
    """

    rv_ratio: float
    seed: int
    run_name: str
    values: Mapping[str, float]

    @property
    def ms_ratio(self) -> float:
        return round(1.0 - self.rv_ratio, 2)

    def get(self, column: str) -> Optional[float]:
        return self.values.get(column)


#: The ``per_seed.csv`` columns this module reads. ``avg_cer`` is read to check the
#: mean-of-views definition against the runner's own value.
_REQUIRED_CELL_COLUMNS: tuple[str, ...] = ("avg_cer",) + tuple(
    f"{view}_{metric}" for metric in METRICS for view in ("rv", "ms")
)


def load_published_cells(maingrid_dir: Path) -> list[PublishedCell]:
    """The present cells of ``per_seed.csv``, in ratio-then-seed order.

    A row whose ``present`` is not ``True`` is dropped (it carries no metrics); a
    missing required column is a hard :class:`InputError` rather than a silent
    ``None``, because every downstream number would otherwise vanish quietly.
    """
    path = maingrid_dir / "tables" / "per_seed.csv"
    rows = read_csv_rows(path)
    if rows:
        missing = [c for c in _REQUIRED_CELL_COLUMNS if c not in rows[0]]
        if missing:
            raise InputError(f"{path} is missing column(s): {', '.join(missing)}")
    cells: list[PublishedCell] = []
    for row in rows:
        if (row.get("present") or "").strip().lower() != "true":
            continue
        values: dict[str, float] = {}
        for column in _REQUIRED_CELL_COLUMNS:
            text = (row.get(column) or "").strip()
            if text:
                values[column] = float(text)
        cells.append(
            PublishedCell(
                rv_ratio=float(row["rv_ratio"]),
                seed=int(row["seed"]),
                run_name=row.get("run_name") or "",
                values=values,
            )
        )
    cells.sort(key=lambda c: (-c.rv_ratio, c.seed))
    return cells


def load_published_contrasts(maingrid_dir: Path) -> dict[str, dict[str, str]]:
    """``comparison`` → published row, over both contrast tables."""
    out: dict[str, dict[str, str]] = {}
    for table in (_ASYMMETRY_TABLE, _RETENTION_TABLE):
        for row in read_csv_rows(maingrid_dir / "tables" / table):
            out[row["comparison"]] = {**row, "_table": table}
    return out


def load_published_per_ratio(
    maingrid_dir: Path,
) -> dict[tuple[float, str], dict[str, str]]:
    """``(rv_ratio, metric)`` → published per-ratio row."""
    return {
        (_ratio_of(row["rv_ratio"]), row["metric"]): row
        for row in read_csv_rows(maingrid_dir / "tables" / "per_ratio.csv")
    }


def load_published_worst(
    selection_dir: Path, *, criterion: str = SELECTION_CRITERION
) -> dict[tuple[float, str], dict[str, str]]:
    """``(rv_ratio, metric)`` → published row of the reported-rule per-ratio table.

    Only rows of ``criterion`` are kept: the uniform study criteria describe
    checkpoints the committed predictions were never decoded from.
    """
    return {
        (_ratio_of(row["rv_ratio"]), row["metric"]): row
        for row in read_csv_rows(selection_dir / "tables" / "scorable_per_ratio.csv")
        if row.get("criterion") == criterion
    }


def _ratio_of(text: str) -> float:
    """Parse a published ``rv_ratio`` cell into the float this module keys on.

    The committed tables do not agree on the *rendering* of a ratio: the CSVs print
    the raw float (``1.0``, ``0.0``) while the JSON payloads key on ``%g`` (``1``,
    ``0``). Keying a join on either text form silently loses the two endpoint ratios,
    so every lookup here is on the rounded float instead.
    """
    return round(float(text), 4)


def _ratio_key(rv_ratio: float) -> str:
    """The ``%g`` ratio key used in this bundle's own JSON payload (house convention)."""
    return f"{rv_ratio:g}"


# --------------------------------------------------------------------------- #
# The two arithmetic definitions this module adds.
# --------------------------------------------------------------------------- #
def cohens_dz(diff: agg.Stat) -> Optional[float]:
    """``mean(diff) / sample_sd(diff)`` — the paired standardized mean difference.

    The one and only *dz* convention in this repo: the mean of the **paired**
    differences over their **sample** SD (``ddof = 1``, exactly the SD
    :func:`~scripts.evaluation.aggregate_accent_ratio_multiseed.summarize` computes,
    which is also the denominator the v1 analysis manifest pre-registered). Not
    Cohen's *d*: there is no pooling of two independent groups here, and dividing by
    a pooled SD would answer a different question.

    ``None`` when the spread cannot be estimated (``n < 2``) or is exactly zero —
    a zero-SD *dz* is ``±inf``, which is not a reportable effect size.
    """
    if diff.mean is None or diff.sd is None or diff.sd == 0.0:
        return None
    return diff.mean / diff.sd


def leave_one_seed_out_dz(diffs: Mapping[int, float]) -> dict[str, Any]:
    """Refit *dz* three times, dropping one seed each time. **Not an interval.**

    Each fold estimates a mean *and* an SD from the remaining ``n − 1 = 2``
    differences, so the folds are unstable by construction — which is the point. The
    spread of the folds is the honest, assumption-free demonstration of how little a
    three-point *dz* pins down; it is neither a confidence interval nor a jackknife
    standard error, and is labelled as such everywhere it appears.
    """
    seeds = sorted(diffs)
    folds: list[dict[str, Any]] = []
    for dropped in seeds:
        kept = [diffs[s] for s in seeds if s != dropped]
        folds.append(
            {
                "dropped_seed": dropped,
                "n": len(kept),
                "dz": cohens_dz(agg.summarize(kept)),
            }
        )
    finite = [f["dz"] for f in folds if f["dz"] is not None]
    return {
        "n_folds": len(folds),
        "n_per_fold": max(len(seeds) - 1, 0),
        "folds": folds,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "is_confidence_interval": False,
        "note": (
            "Leave-one-seed-out refits of dz, each on n=2 paired differences. NOT a "
            "confidence interval and not a standard error — a spread, reported to "
            "show how far dz moves when one of three observations is removed."
        ),
    }


def relative_effect(
    diff_mean: Optional[float], reference_mean: Optional[float]
) -> Optional[float]:
    """``mean(diff) / reference_arm_mean`` — the difference as a share of one arm.

    Reported in the published contrast's own ``b − a`` orientation, with the
    denominator arm named in the same row so the ratio is checkable by hand. ``None``
    when either term is missing or the denominator is exactly zero.
    """
    if diff_mean is None or reference_mean is None or reference_mean == 0.0:
        return None
    return diff_mean / reference_mean


def relative_effect_exceeds_reference(rel: Optional[float]) -> bool:
    """Is the difference larger than the arm it is expressed as a share of?

    ``|relative_effect| > 1`` means the denominator is smaller than the effect itself,
    so the ratio is not "an x% improvement of a baseline" — it is a ratio against a
    reading near the metric's floor and it inflates without bound as that reading
    approaches zero. On the v2 grid this is exactly the accent-F1-denominated rows:
    the untrained notation scores 0.028–0.060, which is what turns the adaptation
    contrasts into relative effects of +1472% and +3218%. Flagged on the row rather
    than suppressed, because the *unstandardized* difference in the same row is the
    number that means something.
    """
    return rel is not None and abs(rel) > 1.0


# --------------------------------------------------------------------------- #
# Endpoint effect sizes.
# --------------------------------------------------------------------------- #
def paired_arm(cells: Sequence[PublishedCell], arm: Arm) -> dict[int, float]:
    """``seed → value`` for one arm, over the present cells of its ratio."""
    out: dict[int, float] = {}
    for cell in cells:
        if cell.rv_ratio != arm.rv_ratio:
            continue
        value = cell.get(arm.column)
        if value is not None:
            out[cell.seed] = value
    return out


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else abs(a - b)


def _float_or_none(text: Optional[str]) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    return float(text)


def reconcile_contrast(
    spec: ContrastSpec,
    contrast: dict[str, Any],
    published: Optional[dict[str, str]],
) -> dict[str, Any]:
    """Does the re-derived contrast equal the published row it claims to be?

    Three independent comparisons: every per-seed difference (published at 6dp), the
    difference's mean and both CI bounds (published at 8dp), and the ``n`` /
    ``sign_consistent`` / ``direction`` flags. An unmatched ``comparison`` is
    ``ok=False`` with ``reason``, never a silently unchecked row — the whole point of
    keying on the published label is that a contrast this bundle invented cannot pass.
    """
    diff = contrast["diff"]
    if published is None:
        return {
            "comparison": spec.comparison,
            "published_table": spec.table,
            "ok": False,
            "reason": "no published row with this comparison label",
            "max_abs_delta_per_seed": None,
            "delta_mean": None,
            "delta_ci95_lo": None,
            "delta_ci95_hi": None,
            "flags_match": False,
        }
    published_per_seed = parse_per_seed_diff(published.get("per_seed_diff", ""))
    derived_per_seed = {int(s): v for s, v in contrast["per_seed_diff"].items()}
    shared = sorted(set(published_per_seed) & set(derived_per_seed))
    per_seed_deltas = [abs(derived_per_seed[s] - published_per_seed[s]) for s in shared]
    max_per_seed = max(per_seed_deltas) if per_seed_deltas else None
    delta_mean = _delta(diff["mean"], _float_or_none(published.get("diff_mean")))
    delta_lo = _delta(diff["ci95_lo"], _float_or_none(published.get("diff_ci95_lo")))
    delta_hi = _delta(diff["ci95_hi"], _float_or_none(published.get("diff_ci95_hi")))
    seeds_match = set(published_per_seed) == set(derived_per_seed)
    flags_match = (
        str(contrast["sign_consistent"]) == (published.get("sign_consistent") or "")
        and contrast["direction"] == (published.get("direction") or "")
        and str(diff["n"]) == (published.get("n") or "")
    )
    within = (
        seeds_match
        and max_per_seed is not None
        and max_per_seed <= PRINTED_6DP_TOLERANCE
        and delta_mean is not None
        and delta_mean <= PRINTED_8DP_TOLERANCE
        and delta_lo is not None
        and delta_lo <= PRINTED_8DP_TOLERANCE
        and delta_hi is not None
        and delta_hi <= PRINTED_8DP_TOLERANCE
        and flags_match
    )
    reason = None
    if not within:
        if not seeds_match:
            reason = "per-seed diff seed sets differ from the published row"
        elif not flags_match:
            reason = "n / direction / sign_consistent differ from the published row"
        else:
            reason = "re-derived values exceed the printed-precision tolerance"
    return {
        "comparison": spec.comparison,
        "published_table": spec.table,
        "ok": bool(within),
        "reason": reason,
        "max_abs_delta_per_seed": max_per_seed,
        "delta_mean": delta_mean,
        "delta_ci95_lo": delta_lo,
        "delta_ci95_hi": delta_hi,
        "flags_match": flags_match,
    }


def endpoint_effect_size(
    spec: ContrastSpec,
    cells: Sequence[PublishedCell],
    published: Optional[dict[str, str]],
) -> dict[str, Any]:
    """Effect sizes for one published contrast, with everything that bounds them.

    The contrast itself is
    :func:`~scripts.evaluation.aggregate_accent_ratio_multiseed._contrast` (paired on
    the shared seeds, ``b − a``); *dz* and the relative effect are the two definitions
    above; the p-value is the exact paired sign-flip permutation test, reused whole.
    Only its two-sided p and its floor are reported: a one-sided p needs a
    pre-registered direction, and the published table's orientation is not one.
    """
    arm_a = paired_arm(cells, spec.a)
    arm_b = paired_arm(cells, spec.b)
    contrast = agg._contrast(arm_a, arm_b)
    seeds: list[int] = list(contrast["seeds"])
    diffs = {s: contrast["per_seed_diff"][str(s)] for s in seeds}
    diff_stat = agg.summarize(diffs[s] for s in seeds)
    stat_a = agg.summarize(arm_a[s] for s in seeds)
    stat_b = agg.summarize(arm_b[s] for s in seeds)
    reference_stat = stat_a if spec.reference == "a" else stat_b
    permutation = sign_flip_permutation_test([diffs[s] for s in seeds])
    better = (
        spec.spec.better(stat_a.mean, stat_b.mean)
        if stat_a.mean is not None and stat_b.mean is not None
        else None
    )
    dz = cohens_dz(diff_stat)
    rel = relative_effect(diff_stat.mean, reference_stat.mean)
    return {
        "comparison": spec.comparison,
        "published_table": spec.table,
        "metric": spec.metric,
        "higher_is_better": spec.spec.higher_is_better,
        "orientation": f"{spec.b.label} − {spec.a.label} (the published b − a)",
        "n": diff_stat.n,
        "seeds": seeds,
        "arm_a": {"label": spec.a.label, **stat_a.as_dict()},
        "arm_b": {"label": spec.b.label, **stat_b.as_dict()},
        "better_arm": better,
        "better_arm_label": (
            None
            if better is None
            else "tie"
            if better == "tie"
            else (spec.a.label if better == "a" else spec.b.label)
        ),
        "per_seed_diff": {str(s): diffs[s] for s in seeds},
        "diff": diff_stat.as_dict(),
        "direction": contrast["direction"],
        "sign_consistent": contrast["sign_consistent"],
        "cohens_dz": dz,
        "cohens_dz_definition": "mean(paired diff) / sample_sd(paired diff), ddof=1",
        "cohens_dz_ci": None,
        "cohens_dz_ci_available": False,
        "cohens_dz_no_interval_reason": NO_DZ_INTERVAL_REASON,
        "cohens_dz_leave_one_seed_out": leave_one_seed_out_dz(diffs),
        "relative_effect": rel,
        "relative_effect_definition": "mean(paired diff) / reference_arm_mean",
        "reference_arm": spec.reference,
        "reference_arm_label": spec.reference_arm().label,
        "reference_arm_mean": reference_stat.mean,
        "reference_arm_rationale": spec.reference_rationale,
        "relative_effect_exceeds_reference_arm": relative_effect_exceeds_reference(rel),
        "exact_sign_test": {
            "p_two_sided": permutation["p_two_sided"],
            "min_p_two_sided": permutation["min_p_two_sided"],
            "enumerated": permutation["enumerated"],
            "exact": permutation["exact"],
            "one_sided_p_withheld_because": (
                "a one-sided p requires a pre-registered direction; the published "
                "table's b − a orientation is a reporting convention, not one"
            ),
        },
        "significance_claim_available": False,
        "significance_note": (
            "n=3: the exact paired sign-flip permutation test cannot return a "
            "two-sided p below 2/2**3 = 0.25, so no significance claim is available "
            "in either direction regardless of how large dz is."
        ),
        "reconciliation": reconcile_contrast(spec, contrast, published),
    }


def endpoint_effect_sizes(
    cells: Sequence[PublishedCell],
    published: Mapping[str, dict[str, str]],
    contrasts: Sequence[ContrastSpec] = CONTRASTS,
) -> list[dict[str, Any]]:
    """One :func:`endpoint_effect_size` row per declared contrast, in report order."""
    return [
        endpoint_effect_size(spec, cells, published.get(spec.comparison))
        for spec in contrasts
    ]


def unexpected_published_contrasts(
    published: Mapping[str, dict[str, str]],
    contrasts: Sequence[ContrastSpec] = CONTRASTS,
) -> list[str]:
    """Published contrast rows this module has no spec for.

    Reported so a contrast added to the main grid later cannot be silently omitted
    from the effect-size table.
    """
    return sorted(set(published) - {spec.comparison for spec in contrasts})


# --------------------------------------------------------------------------- #
# The plateau comparison.
# --------------------------------------------------------------------------- #
def plateau_value(cell: PublishedCell, quantity: PlateauQuantity) -> Optional[float]:
    """One cell's mean-of-views or worst-view reading of ``quantity``'s metric.

    Worst is direction-aware: ``max`` for an error metric, ``min`` for a score. Both
    views must be present — a one-view "worst" is not a worst case.
    """
    rv = cell.get(f"rv_{quantity.metric}")
    ms = cell.get(f"ms_{quantity.metric}")
    if rv is None or ms is None:
        return None
    if quantity.mode == MODE_MEAN:
        return (rv + ms) / 2
    if quantity.mode == MODE_WORST:
        return max(rv, ms) if not quantity.spec.higher_is_better else min(rv, ms)
    raise ValueError(f"unknown plateau mode {quantity.mode!r}")


def plateau_paired(
    cells: Sequence[PublishedCell], rv_ratio: float, quantity: PlateauQuantity
) -> dict[int, float]:
    """``seed → derived value`` at one ratio (the paired input to a pair contrast)."""
    out: dict[int, float] = {}
    for cell in cells:
        if cell.rv_ratio != rv_ratio:
            continue
        value = plateau_value(cell, quantity)
        if value is not None:
            out[cell.seed] = value
    return out


def plateau_summaries(
    cells: Sequence[PublishedCell],
    ratios: Sequence[float] = INTERIOR_RV_RATIOS,
    quantities: Sequence[PlateauQuantity] = PLATEAU_QUANTITIES,
) -> dict[float, dict[str, agg.Stat]]:
    """Per interior ratio, each derived quantity summarized across seeds."""
    out: dict[float, dict[str, agg.Stat]] = {}
    for rv_ratio in ratios:
        paired = {q.name: plateau_paired(cells, rv_ratio, q) for q in quantities}
        out[rv_ratio] = {
            q.name: agg.summarize(paired[q.name][s] for s in sorted(paired[q.name]))
            for q in quantities
        }
    return out


def plateau_ranking(
    summaries: Mapping[float, Mapping[str, agg.Stat]],
    ratios: Sequence[float] = INTERIOR_RV_RATIOS,
) -> list[dict[str, Any]]:
    """Ranked interior ratios by :data:`RANK_PRIMARY`, tie-broken on :data:`RANK_TIEBREAK`.

    Rank 1 is the best (lowest worst-view CER). A ratio whose ranking quantity is
    unavailable sorts last rather than crashing, and carries ``rankable=False`` so no
    table presents it as ranked.
    """

    def key(rv_ratio: float) -> tuple[float, float]:
        primary = summaries[rv_ratio][RANK_PRIMARY].mean
        tiebreak = summaries[rv_ratio][RANK_TIEBREAK].mean
        return (
            math.inf if primary is None else primary,
            math.inf if tiebreak is None else tiebreak,
        )

    ordered = sorted(ratios, key=key)
    rows: list[dict[str, Any]] = []
    for rank, rv_ratio in enumerate(ordered, start=1):
        primary = summaries[rv_ratio][RANK_PRIMARY]
        tiebreak = summaries[rv_ratio][RANK_TIEBREAK]
        rows.append(
            {
                "rank": rank,
                "rv_ratio": rv_ratio,
                "ms_ratio": round(1.0 - rv_ratio, 2),
                "n": primary.n,
                "rank_quantity": RANK_PRIMARY,
                "rank_quantity_mean": primary.mean,
                "rank_quantity_ci95_lo": primary.ci95_lo,
                "rank_quantity_ci95_hi": primary.ci95_hi,
                "tiebreak_quantity": RANK_TIEBREAK,
                "tiebreak_quantity_mean": tiebreak.mean,
                "rankable": primary.mean is not None,
            }
        )
    return rows


def _cis_overlap(a: agg.Stat, b: agg.Stat) -> Optional[bool]:
    """Do two marginal 95% CIs overlap? ``None`` when either is uncomputable.

    Closed intervals: touching bounds count as overlapping, which is the reading that
    refuses to call a pair separated on a boundary case.
    """
    a_lo, a_hi = a.ci95_lo, a.ci95_hi
    b_lo, b_hi = b.ci95_lo, b.ci95_hi
    if a_lo is None or a_hi is None or b_lo is None or b_hi is None:
        return None
    return not (a_hi < b_lo or b_hi < a_lo)


def plateau_separability(
    cells: Sequence[PublishedCell],
    summaries: Mapping[float, Mapping[str, agg.Stat]],
    pairs: Sequence[tuple[float, float]] = PLATEAU_PAIRS,
    quantities: Sequence[PlateauQuantity] = PLATEAU_QUANTITIES,
) -> list[dict[str, Any]]:
    """Pairwise separability of the interior ratios, per derived quantity.

    Three readings per pair, because a ranking without them is misleading: the
    **paired** seed-wise contrast (the powerful one — the ratios share seeds) with its
    own CI and sign consistency, whether the two **marginal** CIs overlap, and the
    exact sign test's two-sided p. ``separable`` is ``False`` on every row by
    construction (:data:`SEPARABILITY_NOTE`); what varies is whether the three seeds
    agree on the order at all.
    """
    rows: list[dict[str, Any]] = []
    for a_ratio, b_ratio in pairs:
        for quantity in quantities:
            paired_a = plateau_paired(cells, a_ratio, quantity)
            paired_b = plateau_paired(cells, b_ratio, quantity)
            contrast = agg._contrast(paired_a, paired_b)
            diff = contrast["diff"]
            seeds: list[int] = list(contrast["seeds"])
            diffs = [contrast["per_seed_diff"][str(s)] for s in seeds]
            permutation = sign_flip_permutation_test(diffs)
            lo, hi = diff["ci95_lo"], diff["ci95_hi"]
            paired_ci_excludes_zero = (
                None if lo is None or hi is None else not (lo <= 0.0 <= hi)
            )
            rows.append(
                {
                    "pair": f"{_pct(b_ratio)} − {_pct(a_ratio)}",
                    "a_rv_ratio": a_ratio,
                    "b_rv_ratio": b_ratio,
                    "quantity": quantity.name,
                    "metric": quantity.metric,
                    "mode": quantity.mode,
                    "higher_is_better": quantity.spec.higher_is_better,
                    "n": diff["n"],
                    "seeds": seeds,
                    "paired_diff": diff,
                    "per_seed_diff": contrast["per_seed_diff"],
                    "direction": contrast["direction"],
                    "sign_consistent": contrast["sign_consistent"],
                    "paired_ci95_excludes_zero": paired_ci_excludes_zero,
                    "marginal_ci95_overlap": _cis_overlap(
                        summaries[a_ratio][quantity.name],
                        summaries[b_ratio][quantity.name],
                    ),
                    "exact_sign_test_p_two_sided": permutation["p_two_sided"],
                    "exact_sign_test_min_p_two_sided": permutation["min_p_two_sided"],
                    "separable": False,
                    "verdict": (
                        ORDER_CONSISTENT
                        if contrast["sign_consistent"]
                        else ORDER_UNSTABLE
                    ),
                }
            )
    return rows


def _pct(rv_ratio: float) -> str:
    """``0.75`` → ``"75:25"`` — the RV:MS label the paper uses."""
    rv = int(round(rv_ratio * 100))
    return f"{rv}:{100 - rv}"


def plateau_verdict(
    ranking: Sequence[Mapping[str, Any]],
    separability: Sequence[Mapping[str, Any]],
    *,
    quantity: str = RANK_PRIMARY,
) -> dict[str, Any]:
    """The one sentence the paper may quote about the ranking, and its bounds."""
    rows = [r for r in separability if r["quantity"] == quantity]
    consistent = [r["pair"] for r in rows if r["verdict"] == ORDER_CONSISTENT]
    unstable = [r["pair"] for r in rows if r["verdict"] == ORDER_UNSTABLE]
    overlapping = [r["pair"] for r in rows if r["marginal_ci95_overlap"] is True]
    ranked = [_pct(r["rv_ratio"]) for r in ranking if r["rankable"]]
    return {
        "rank_quantity": quantity,
        "ranking_best_first": ranked,
        "n_pairs": len(rows),
        "n_pairs_separable": 0,
        "pairs_consistently_ordered": consistent,
        "pairs_order_unstable": unstable,
        "pairs_with_overlapping_marginal_cis": overlapping,
        "all_marginal_cis_overlap": len(overlapping) == len(rows) and bool(rows),
        "separability_note": SEPARABILITY_NOTE,
        "ranking_statement": (
            f"Ranked best-first on {quantity}: {' < '.join(ranked)}."
        ),
        "separability_statement": (
            f"{len(overlapping)}/{len(rows)} pairs have overlapping marginal 95% CIs "
            f"and 0/{len(rows)} pairs are separable at n=3; "
            f"{len(consistent)}/{len(rows)} are consistently ordered across the three "
            f"seeds and {len(unstable)}/{len(rows)} flip order by seed."
        ),
        "statement": (
            f"Ranked best-first on {quantity}: {' < '.join(ranked)}. "
            f"{len(overlapping)}/{len(rows)} pairs have overlapping marginal 95% CIs "
            f"and 0/{len(rows)} pairs are separable at n=3; "
            f"{len(consistent)}/{len(rows)} are consistently ordered across the three "
            f"seeds and {len(unstable)}/{len(rows)} flip order by seed."
        ),
    }


# --------------------------------------------------------------------------- #
# Reconciliation of the derived plateau quantities + the per-ratio value order.
# --------------------------------------------------------------------------- #
def reconcile_plateau(
    summaries: Mapping[float, Mapping[str, agg.Stat]],
    published_worst: Mapping[tuple[float, str], Mapping[str, str]],
    ratios: Sequence[float] = INTERIOR_RV_RATIOS,
    quantities: Sequence[PlateauQuantity] = PLATEAU_QUANTITIES,
) -> list[dict[str, Any]]:
    """Derived worst-view / mean-view means vs their published counterparts.

    A quantity with a ``published_as`` counterpart must match it to
    :data:`FULL_PRECISION_TOLERANCE` — that is what makes ``max(rv, ms)`` a
    re-derivation rather than a new definition. A quantity with no counterpart is
    emitted with ``checked=False``, so the table states plainly which of its rows
    stand on a published number and which are derived here for the first time.
    """
    rows: list[dict[str, Any]] = []
    for rv_ratio in ratios:
        for quantity in quantities:
            stat = summaries[rv_ratio][quantity.name]
            if quantity.published_as is None:
                rows.append(
                    {
                        "rv_ratio": rv_ratio,
                        "quantity": quantity.name,
                        "published_as": None,
                        "checked": False,
                        "derived_mean": stat.mean,
                        "published_mean": None,
                        "delta": None,
                        "ok": True,
                        "reason": "no published counterpart at n=3; derived here",
                    }
                )
                continue
            published = published_worst.get((round(rv_ratio, 4), quantity.published_as))
            published_mean = (
                None if published is None else _float_or_none(published.get("mean"))
            )
            delta = _delta(stat.mean, published_mean)
            ok = delta is not None and delta <= FULL_PRECISION_TOLERANCE
            rows.append(
                {
                    "rv_ratio": rv_ratio,
                    "quantity": quantity.name,
                    "published_as": quantity.published_as,
                    "checked": True,
                    "derived_mean": stat.mean,
                    "published_mean": published_mean,
                    "delta": delta,
                    "ok": bool(ok),
                    "reason": None
                    if ok
                    else (
                        "no published row for this (ratio, quantity)"
                        if published is None
                        else "derived mean differs from the published mean"
                    ),
                }
            )
    return rows


#: The ``per_ratio.csv`` metrics whose unlabelled ``values`` column is checked against
#: the seed-labelled cells. Every metric this module reads, both views.
_VALUE_ORDER_METRICS: tuple[str, ...] = ("avg_cer",) + tuple(
    f"{view}_{metric}" for metric in METRICS for view in ("rv", "ms")
)


def reconcile_per_ratio_values(
    cells: Sequence[PublishedCell],
    published_per_ratio: Mapping[tuple[float, str], Mapping[str, str]],
    ratios: Sequence[float] = RV_RATIOS,
) -> list[dict[str, Any]]:
    """Is ``per_ratio.csv``'s unlabelled ``values`` column in ascending-seed order?

    This is the check that licenses everything above. ``per_ratio.csv`` publishes the
    per-seed numbers but not which seed each belongs to, so it cannot be paired; this
    bundle pairs from ``per_seed.csv`` instead and verifies here that the two agree
    element-by-element with the seeds sorted ascending. A mismatch means one of the
    two tables is not what it appears to be, and is publication-blocking.
    """
    rows: list[dict[str, Any]] = []
    for rv_ratio in ratios:
        group = sorted(
            (c for c in cells if c.rv_ratio == rv_ratio), key=lambda c: c.seed
        )
        for metric in _VALUE_ORDER_METRICS:
            published = published_per_ratio.get((round(rv_ratio, 4), metric))
            derived = [c.get(metric) for c in group]
            if published is None:
                rows.append(
                    {
                        "rv_ratio": rv_ratio,
                        "metric": metric,
                        "n_derived": len(derived),
                        "n_published": 0,
                        "max_abs_delta": None,
                        "ok": False,
                        "reason": "no published per_ratio row for this metric",
                    }
                )
                continue
            values = parse_values(published.get("values", ""))
            deltas = [
                abs(d - p)
                for d, p in zip(derived, values, strict=False)
                if d is not None
            ]
            max_delta = max(deltas) if deltas else None
            ok = (
                len(values) == len(derived)
                and max_delta is not None
                and max_delta <= PRINTED_6DP_TOLERANCE
            )
            rows.append(
                {
                    "rv_ratio": rv_ratio,
                    "metric": metric,
                    "n_derived": len(derived),
                    "n_published": len(values),
                    "max_abs_delta": max_delta,
                    "ok": bool(ok),
                    "reason": None
                    if ok
                    else (
                        "value count differs from the cells present in per_seed.csv"
                        if len(values) != len(derived)
                        else "seed-ordered values differ beyond printed precision"
                    ),
                }
            )
    return rows


def reconcile_mean_of_views(cells: Sequence[PublishedCell]) -> dict[str, Any]:
    """Is ``mean_of_views(cer)`` the runner's own ``avg_cer``?

    The mean-view column of the plateau table is defined here as ``(rv + ms) / 2``.
    The runner already publishes exactly that as ``avg_cer`` per cell, so the
    definition is checkable rather than merely plausible.
    """
    deltas: list[float] = []
    for cell in cells:
        rv = cell.get("rv_cer")
        ms = cell.get("ms_cer")
        published = cell.get("avg_cer")
        if rv is None or ms is None or published is None:
            continue
        deltas.append(abs((rv + ms) / 2 - published))
    max_delta = max(deltas) if deltas else None
    return {
        "n_cells": len(deltas),
        "max_abs_delta": max_delta,
        "ok": bool(
            deltas and max_delta is not None and max_delta <= FULL_PRECISION_TOLERANCE
        ),
    }


# --------------------------------------------------------------------------- #
# Coverage, input attestation, warnings.
# --------------------------------------------------------------------------- #
def coverage_summary(cells: Sequence[PublishedCell]) -> dict[str, Any]:
    """Cell coverage of the target grid behind every number in this bundle.

    Measured against the **target** (:data:`TARGET_N_CELLS`), never against the rows
    that happened to be readable: a truncated ``per_seed.csv`` must not report a
    complete grid. ``n`` is carried per ratio because it is the n behind that ratio's
    plateau row, and per contrast because it is the n behind that *dz*.
    """
    per_ratio: dict[str, Any] = {}
    for rv_ratio in RV_RATIOS:
        present = sorted(c.seed for c in cells if c.rv_ratio == rv_ratio)
        per_ratio[_ratio_key(rv_ratio)] = {
            "rv_ratio": rv_ratio,
            "ms_ratio": round(1.0 - rv_ratio, 2),
            "n_present": len(present),
            "seeds_present": present,
            "seeds_missing": sorted(set(SEEDS) - set(present)),
            "complete": set(present) == set(SEEDS),
        }
    n_present = len(cells)
    return {
        "n_cells": TARGET_N_CELLS,
        "n_present": n_present,
        "complete": n_present == TARGET_N_CELLS
        and all(block["complete"] for block in per_ratio.values()),
        "target_seeds": list(SEEDS),
        "target_rv_ratios": list(RV_RATIOS),
        "interior_rv_ratios": list(INTERIOR_RV_RATIOS),
        "per_ratio": per_ratio,
    }


def status_label(coverage: Mapping[str, Any]) -> str:
    """``"COMPLETE"`` or ``"PARTIAL"`` — the word every artifact leads with."""
    return "COMPLETE" if coverage["complete"] else "PARTIAL"


def partial_banner(coverage: Mapping[str, Any]) -> Optional[str]:
    """The refusal banner a partial input must carry, or ``None`` when complete."""
    if coverage["complete"]:
        return None
    thin = [
        f"{_pct(block['rv_ratio'])} (n={block['n_present']})"
        for block in coverage["per_ratio"].values()
        if not block["complete"]
    ]
    return (
        f"PARTIAL: {coverage['n_present']}/{coverage['n_cells']} published cells were "
        f"read; incomplete ratios: {', '.join(thin) or 'none'}. No effect size or "
        "ranking below may be presented as the v2 result: an effect size over a "
        "subset of the seeds is not the grid's effect size, and a ranking is not a "
        "ranking of ratios whose n differ."
    )


def input_attestation(maingrid_dir: Path, selection_dir: Path) -> dict[str, Any]:
    """What the two input bundles say about themselves, and whether that is enough.

    Nothing here is assumed from a directory name. The main grid must declare itself
    ``COMPLETE`` **and** ``publishable`` with a healthy corpus attestation whose only
    corpus is :data:`EXPECTED_CORPUS`; the selection bundle must declare the same
    corpus and the same 15-cell grid. Either failing is publication-blocking — the
    alternative is v2-labelled effect sizes computed over v1 pixels.
    """
    maingrid = read_json(maingrid_dir / "aggregate.json")
    selection = read_json(selection_dir / "aggregate.json")
    attestation = maingrid.get("corpus_attestation") or {}
    corpora = list(attestation.get("corpora") or [])
    maingrid_ok = (
        maingrid.get("status") == "COMPLETE"
        and maingrid.get("publishable") is True
        and bool(attestation.get("healthy"))
        and corpora == [EXPECTED_CORPUS]
        and maingrid.get("corpus") == EXPECTED_CORPUS
    )
    selection_grid = selection.get("grid") or {}
    selection_ok = (
        selection.get("corpus") == EXPECTED_CORPUS
        and selection.get("n_cells") == TARGET_N_CELLS
        and list(selection_grid.get("seeds") or []) == list(SEEDS)
    )
    return {
        "expected_corpus": EXPECTED_CORPUS,
        "maingrid": {
            "study": maingrid.get("study"),
            "status": maingrid.get("status"),
            "publishable": maingrid.get("publishable"),
            "corpus": maingrid.get("corpus"),
            "attested_corpora": corpora,
            "attestation_healthy": bool(attestation.get("healthy")),
            "n_attested": attestation.get("n_attested"),
            "ok": bool(maingrid_ok),
        },
        "selection": {
            "study": selection.get("study"),
            "corpus": selection.get("corpus"),
            "n_cells": selection.get("n_cells"),
            "seeds": list(selection_grid.get("seeds") or []),
            "criterion_used": SELECTION_CRITERION,
            "ok": bool(selection_ok),
        },
        "ok": bool(maingrid_ok and selection_ok),
    }


def input_files(maingrid_dir: Path, selection_dir: Path) -> list[dict[str, Any]]:
    """Every file read, with its sha256 — provenance that a reader can verify."""
    paths = [
        maingrid_dir / "aggregate.json",
        maingrid_dir / "tables" / "per_seed.csv",
        maingrid_dir / "tables" / "per_ratio.csv",
        maingrid_dir / "tables" / _ASYMMETRY_TABLE,
        maingrid_dir / "tables" / _RETENTION_TABLE,
        selection_dir / "aggregate.json",
        selection_dir / "tables" / "scorable_per_ratio.csv",
    ]
    return [
        {"path": str(path), "sha256": sha256_of(path), "present": path.is_file()}
        for path in paths
    ]


def audit_warnings(
    coverage: Mapping[str, Any],
    attestation: Mapping[str, Any],
    effects: Sequence[Mapping[str, Any]],
    plateau_reconciliation: Sequence[Mapping[str, Any]],
    value_order: Sequence[Mapping[str, Any]],
    mean_of_views: Mapping[str, Any],
    extra_published: Sequence[str],
) -> list[str]:
    """Every integrity failure, in the order it should be read."""
    warnings: list[str] = []
    if not coverage["complete"]:
        warnings.append(
            f"incomplete input: {coverage['n_present']}/{coverage['n_cells']} cells "
            "present in the published per_seed table"
        )
    if not attestation["ok"]:
        warnings.append(
            "input corpus attestation does not establish "
            f"{EXPECTED_CORPUS}: maingrid ok={attestation['maingrid']['ok']}, "
            f"selection ok={attestation['selection']['ok']}"
        )
    bad_contrasts = [e["comparison"] for e in effects if not e["reconciliation"]["ok"]]
    if bad_contrasts:
        warnings.append(
            "contrast(s) not reconciled against the published table: "
            + ", ".join(bad_contrasts)
        )
    thin_contrasts = [
        f"{e['comparison']} (n={e['n']})" for e in effects if e["n"] != len(SEEDS)
    ]
    if thin_contrasts:
        warnings.append(
            f"contrast(s) not paired on all {len(SEEDS)} seeds: "
            + ", ".join(thin_contrasts)
        )
    bad_plateau = [
        f"{_pct(r['rv_ratio'])}:{r['quantity']}"
        for r in plateau_reconciliation
        if not r["ok"]
    ]
    if bad_plateau:
        warnings.append(
            "derived plateau quantity/-ies disagree with the published worst-view "
            "table: " + ", ".join(bad_plateau)
        )
    bad_order = [
        f"{_pct(r['rv_ratio'])}:{r['metric']}" for r in value_order if not r["ok"]
    ]
    if bad_order:
        warnings.append(
            "per_ratio.csv values do not match the seed-ordered cells for: "
            + ", ".join(bad_order)
        )
    if not mean_of_views["ok"]:
        warnings.append(
            "mean_of_views(cer) does not reproduce the runner's avg_cer "
            f"(max |Δ| = {mean_of_views['max_abs_delta']})"
        )
    if extra_published:
        warnings.append(
            "published contrast row(s) with no effect-size spec (so absent from this "
            "bundle): " + ", ".join(extra_published)
        )
    return warnings


def hard_warnings(warnings: Sequence[str]) -> list[str]:
    """Every warning here is publication-blocking.

    There is no soft class: the inputs are committed artifacts, so an unreconciled
    contrast, a thin grid or an unattested corpus is a defect in *this* bundle's
    reading of them, not a mid-flight state to be tolerated.
    """
    return list(warnings)


# --------------------------------------------------------------------------- #
# Aggregate payload.
# --------------------------------------------------------------------------- #
def build_aggregate(
    cells: Sequence[PublishedCell],
    effects: Sequence[Mapping[str, Any]],
    summaries: Mapping[float, Mapping[str, agg.Stat]],
    ranking: Sequence[Mapping[str, Any]],
    separability: Sequence[Mapping[str, Any]],
    *,
    coverage: Mapping[str, Any],
    attestation: Mapping[str, Any],
    plateau_reconciliation: Sequence[Mapping[str, Any]],
    value_order: Sequence[Mapping[str, Any]],
    mean_of_views: Mapping[str, Any],
    extra_published: Sequence[str],
    inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The machine-readable payload (→ ``aggregate.json``)."""
    warnings = audit_warnings(
        coverage,
        attestation,
        effects,
        plateau_reconciliation,
        value_order,
        mean_of_views,
        extra_published,
    )
    hard = hard_warnings(warnings)
    publishable = not hard and coverage["complete"] and attestation["ok"]
    verdict = plateau_verdict(ranking, separability)
    return {
        "study": STUDY,
        "issue": ISSUE,
        "parent_study": PARENT_STUDY,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "corpus": EXPECTED_CORPUS,
        "analysis_only": True,
        "no_training_inference_or_gpu": True,
        "derives_from": "published v2 tables only (no run dir, no prediction CSV)",
        "status": status_label(coverage),
        "partial_banner": partial_banner(coverage),
        "publishable": publishable,
        "coverage": coverage,
        "input_attestation": attestation,
        "inputs": list(inputs),
        "n_ceiling": n_ceiling_block(),
        "endpoint_effect_sizes": list(effects),
        "plateau": {
            "interior_rv_ratios": list(INTERIOR_RV_RATIOS),
            "quantities": [
                {
                    "name": q.name,
                    "metric": q.metric,
                    "mode": q.mode,
                    "higher_is_better": q.spec.higher_is_better,
                    "published_as": q.published_as,
                }
                for q in PLATEAU_QUANTITIES
            ],
            "per_ratio": {
                _ratio_key(rv_ratio): {
                    name: stat.as_dict() for name, stat in metrics.items()
                }
                for rv_ratio, metrics in summaries.items()
            },
            "ranking": list(ranking),
            "separability": list(separability),
            "verdict": verdict if publishable else None,
            "verdict_withheld_because": None
            if publishable
            else "the inputs did not pass the coverage / attestation / "
            "reconciliation gates, so no ranking may be quoted",
        },
        "reconciliation": {
            "contrasts": [dict(e["reconciliation"]) for e in effects],
            "plateau": list(plateau_reconciliation),
            "per_ratio_value_order": list(value_order),
            "mean_of_views_is_runner_avg_cer": dict(mean_of_views),
            "tolerances": {
                "printed_6dp": PRINTED_6DP_TOLERANCE,
                "printed_8dp": PRINTED_8DP_TOLERANCE,
                "full_precision": FULL_PRECISION_TOLERANCE,
            },
        },
        "warnings": warnings,
        "hard_warnings": hard,
        "generated_by": f"scripts.evaluation.{Path(__file__).stem}",
    }


def n_ceiling_block() -> dict[str, Any]:
    """What n=3 does and does not permit — attached to every artifact."""
    floor = sign_flip_permutation_test([1.0] * len(SEEDS))
    return {
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "exact_sign_test_min_p_two_sided": floor["min_p_two_sided"],
        "sign_assignments_enumerated": floor["enumerated"],
        "significance_claim_available": False,
        "dz_interval_available": False,
        "dz_no_interval_reason": NO_DZ_INTERVAL_REASON,
        "separability_note": SEPARABILITY_NOTE,
        "what_these_numbers_support": [
            "the direction and the standardized size of the difference *among these "
            "three seeds*, and whether all three agree in sign",
            "the unstandardized difference with a Student-t 95% CI (already published "
            "by the main grid and reproduced here)",
            "that the three interior ratios' worst-view CERs sit inside each other's "
            "intervals, i.e. a broad plateau rather than an identified optimum",
        ],
        "what_these_numbers_do_not_support": [
            "any significance claim, in either direction: the exact paired test's "
            "floor at n=3 is p = 0.25",
            "a population effect size: dz here is a ratio of two estimates from three "
            "points, and its leave-one-seed-out refits move by factors, not percents",
            "a claim that one interior mix ratio is better than another: no pair is "
            "separable at n=3",
            "any comparison with v1, whose corpus is abandoned",
        ],
    }


# --------------------------------------------------------------------------- #
# Rendering: CSV.
# --------------------------------------------------------------------------- #
def _fixed(value: Optional[float], digits: int = 8) -> str:
    """Fixed-point, never scientific; empty for an uncomputable value.

    Same rule and the same reason as
    :func:`~scripts.evaluation.aggregate_accent_ratio_multiseed._fixed`: a value
    printed as ``4.97e-05`` reads as ``4.97`` when a cell is skimmed or truncated,
    which is a five-order-of-magnitude error in a claim. ``digits`` is looser for
    *dz*, which runs to three integer digits here and needs no eight decimals.
    """
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _flag(value: Optional[bool]) -> str:
    return "" if value is None else str(value)


def _write_rows(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buf.getvalue()


ENDPOINT_COLUMNS: tuple[str, ...] = (
    "comparison",
    "published_table",
    "metric",
    "n",
    "seeds",
    "orientation",
    "arm_a_label",
    "arm_a_mean",
    "arm_b_label",
    "arm_b_mean",
    "better_arm_label",
    "diff_mean",
    "diff_sd",
    "diff_ci95_lo",
    "diff_ci95_hi",
    "cohens_dz",
    "cohens_dz_ci",
    "dz_leave_one_seed_out_min",
    "dz_leave_one_seed_out_max",
    "relative_effect",
    "reference_arm_label",
    "reference_arm_mean",
    "relative_effect_exceeds_reference_arm",
    "direction",
    "sign_consistent",
    "exact_sign_test_p_two_sided",
    "exact_sign_test_min_p_two_sided",
    "significance_claim_available",
    "reconciled",
    "per_seed_diff",
)


def render_endpoint_csv(effects: Sequence[Mapping[str, Any]]) -> str:
    """The endpoint effect-size table.

    Every row carries its own ``n``, an empty ``cohens_dz_ci`` (there is none — see
    :data:`NO_DZ_INTERVAL_REASON`), the leave-one-seed-out spread that replaces it,
    the named denominator of its relative effect, and
    ``significance_claim_available=False`` beside the p-floor. A reader who lifts one
    row out of context still cannot mistake it for a significance result.
    """
    rows = []
    for e in effects:
        diff = e["diff"]
        loo = e["cohens_dz_leave_one_seed_out"]
        rows.append(
            {
                "comparison": e["comparison"],
                "published_table": e["published_table"],
                "metric": e["metric"],
                "n": e["n"],
                "seeds": ";".join(str(s) for s in e["seeds"]),
                "orientation": e["orientation"],
                "arm_a_label": e["arm_a"]["label"],
                "arm_a_mean": _fixed(e["arm_a"]["mean"], 6),
                "arm_b_label": e["arm_b"]["label"],
                "arm_b_mean": _fixed(e["arm_b"]["mean"], 6),
                "better_arm_label": e["better_arm_label"] or "",
                "diff_mean": _fixed(diff["mean"]),
                "diff_sd": _fixed(diff["sd"]),
                "diff_ci95_lo": _fixed(diff["ci95_lo"]),
                "diff_ci95_hi": _fixed(diff["ci95_hi"]),
                "cohens_dz": _fixed(e["cohens_dz"], 4),
                # Deliberately always empty; the column exists so that no reader
                # wonders whether an interval was simply forgotten.
                "cohens_dz_ci": "",
                "dz_leave_one_seed_out_min": _fixed(loo["min"], 4),
                "dz_leave_one_seed_out_max": _fixed(loo["max"], 4),
                "relative_effect": _fixed(e["relative_effect"], 6),
                "reference_arm_label": e["reference_arm_label"],
                "reference_arm_mean": _fixed(e["reference_arm_mean"], 6),
                "relative_effect_exceeds_reference_arm": _flag(
                    e["relative_effect_exceeds_reference_arm"]
                ),
                "direction": e["direction"],
                "sign_consistent": _flag(e["sign_consistent"]),
                "exact_sign_test_p_two_sided": _fixed(
                    e["exact_sign_test"]["p_two_sided"], 6
                ),
                "exact_sign_test_min_p_two_sided": _fixed(
                    e["exact_sign_test"]["min_p_two_sided"], 6
                ),
                "significance_claim_available": _flag(
                    e["significance_claim_available"]
                ),
                "reconciled": _flag(e["reconciliation"]["ok"]),
                "per_seed_diff": ";".join(
                    f"{s}:{e['per_seed_diff'][str(s)]:.6f}" for s in e["seeds"]
                ),
            }
        )
    return _write_rows(ENDPOINT_COLUMNS, rows)


DZ_FOLD_COLUMNS: tuple[str, ...] = (
    "comparison",
    "dropped_seed",
    "n",
    "dz",
    "is_confidence_interval",
)


def render_dz_folds_csv(effects: Sequence[Mapping[str, Any]]) -> str:
    """The leave-one-seed-out *dz* refits, one row per fold.

    Published as its own table because it is the evidence that the *dz* column is
    not a decisive number: three fits at n=2, each labelled ``is_confidence_interval
    = False``.
    """
    rows = []
    for e in effects:
        for fold in e["cohens_dz_leave_one_seed_out"]["folds"]:
            rows.append(
                {
                    "comparison": e["comparison"],
                    "dropped_seed": fold["dropped_seed"],
                    "n": fold["n"],
                    "dz": _fixed(fold["dz"], 4),
                    "is_confidence_interval": False,
                }
            )
    return _write_rows(DZ_FOLD_COLUMNS, rows)


PLATEAU_COLUMNS: tuple[str, ...] = (
    "rv_ratio",
    "ms_ratio",
    "quantity",
    "metric",
    "mode",
    "n",
    "seeds",
    "mean",
    "sd",
    "sem",
    "ci95_lo",
    "ci95_hi",
    "ci95_halfwidth",
    "values",
    "published_as",
    "reconciled_against_published",
    "worst_cer_rank",
)


def render_plateau_csv(
    cells: Sequence[PublishedCell],
    summaries: Mapping[float, Mapping[str, agg.Stat]],
    ranking: Sequence[Mapping[str, Any]],
    plateau_reconciliation: Sequence[Mapping[str, Any]],
) -> str:
    """The plateau comparison, in v1's ``plateau_comparison.csv`` shape plus provenance.

    Added over v1: ``seeds`` (so the unlabelled ``values`` column of this table is
    never the only ordering evidence), ``published_as`` /
    ``reconciled_against_published`` (which rows stand on a published number), and
    ``mode`` (what "worst" means for a score metric).
    """
    rank_by_ratio = {r["rv_ratio"]: r["rank"] for r in ranking}
    reconciled = {(r["rv_ratio"], r["quantity"]): r for r in plateau_reconciliation}
    rows = []
    for rv_ratio in INTERIOR_RV_RATIOS:
        seeds = sorted(c.seed for c in cells if c.rv_ratio == rv_ratio)
        for quantity in PLATEAU_QUANTITIES:
            stat = summaries[rv_ratio][quantity.name]
            check = reconciled.get((rv_ratio, quantity.name)) or {}
            rows.append(
                {
                    "rv_ratio": rv_ratio,
                    "ms_ratio": round(1.0 - rv_ratio, 2),
                    "quantity": quantity.name,
                    "metric": quantity.metric,
                    "mode": quantity.mode,
                    "n": stat.n,
                    "seeds": ";".join(str(s) for s in seeds),
                    "mean": _fixed(stat.mean),
                    "sd": _fixed(stat.sd),
                    "sem": _fixed(stat.sem),
                    "ci95_lo": _fixed(stat.ci95_lo),
                    "ci95_hi": _fixed(stat.ci95_hi),
                    "ci95_halfwidth": _fixed(stat.ci95_halfwidth),
                    "values": ";".join(f"{v:.6f}" for v in stat.values),
                    "published_as": quantity.published_as or "",
                    "reconciled_against_published": _flag(check.get("checked")),
                    "worst_cer_rank": rank_by_ratio.get(rv_ratio, ""),
                }
            )
    return _write_rows(PLATEAU_COLUMNS, rows)


SEPARABILITY_COLUMNS: tuple[str, ...] = (
    "pair",
    "quantity",
    "metric",
    "mode",
    "n",
    "seeds",
    "paired_diff_mean",
    "paired_diff_ci95_lo",
    "paired_diff_ci95_hi",
    "paired_ci95_excludes_zero",
    "marginal_ci95_overlap",
    "sign_consistent",
    "direction",
    "exact_sign_test_p_two_sided",
    "exact_sign_test_min_p_two_sided",
    "separable",
    "verdict",
    "per_seed_diff",
)


def render_separability_csv(separability: Sequence[Mapping[str, Any]]) -> str:
    """Pairwise separability of the interior ratios — the table the ranking needs.

    ``separable`` is ``False`` in every row and ``verdict`` says which kind of
    non-separability it is, so the ranking table can never travel without the reason
    it must not be read as an ordering claim.
    """
    rows = []
    for r in separability:
        diff = r["paired_diff"]
        rows.append(
            {
                "pair": r["pair"],
                "quantity": r["quantity"],
                "metric": r["metric"],
                "mode": r["mode"],
                "n": diff["n"],
                "seeds": ";".join(str(s) for s in r["seeds"]),
                "paired_diff_mean": _fixed(diff["mean"]),
                "paired_diff_ci95_lo": _fixed(diff["ci95_lo"]),
                "paired_diff_ci95_hi": _fixed(diff["ci95_hi"]),
                "paired_ci95_excludes_zero": _flag(r["paired_ci95_excludes_zero"]),
                "marginal_ci95_overlap": _flag(r["marginal_ci95_overlap"]),
                "sign_consistent": _flag(r["sign_consistent"]),
                "direction": r["direction"],
                "exact_sign_test_p_two_sided": _fixed(
                    r["exact_sign_test_p_two_sided"], 6
                ),
                "exact_sign_test_min_p_two_sided": _fixed(
                    r["exact_sign_test_min_p_two_sided"], 6
                ),
                "separable": _flag(r["separable"]),
                "verdict": r["verdict"],
                "per_seed_diff": ";".join(
                    f"{s}:{r['per_seed_diff'][str(s)]:.6f}" for s in r["seeds"]
                ),
            }
        )
    return _write_rows(SEPARABILITY_COLUMNS, rows)


RECONCILIATION_COLUMNS: tuple[str, ...] = (
    "check",
    "subject",
    "published_source",
    "checked",
    "derived",
    "published",
    "max_abs_delta",
    "tolerance",
    "ok",
    "reason",
)


def render_reconciliation_csv(aggregate: Mapping[str, Any]) -> str:
    """Every reconciliation this bundle passed (or failed), in one place."""
    rec = aggregate["reconciliation"]
    rows: list[dict[str, Any]] = []
    for row in rec["contrasts"]:
        rows.append(
            {
                "check": "published_contrast",
                "subject": row["comparison"],
                "published_source": f"{PARENT_STUDY}/tables/{row['published_table']}",
                "checked": True,
                "derived": "",
                "published": "",
                "max_abs_delta": _fixed(row["max_abs_delta_per_seed"], 10),
                "tolerance": _fixed(PRINTED_6DP_TOLERANCE, 10),
                "ok": row["ok"],
                "reason": row["reason"] or "",
            }
        )
    for row in rec["plateau"]:
        rows.append(
            {
                "check": "published_worst_view",
                "subject": f"{_pct(row['rv_ratio'])}:{row['quantity']}",
                "published_source": (
                    f"{SELECTION_STUDY}/tables/scorable_per_ratio.csv"
                    f"[{SELECTION_CRITERION}:{row['published_as']}]"
                    if row["published_as"]
                    else ""
                ),
                "checked": row["checked"],
                "derived": _fixed(row["derived_mean"], 10),
                "published": _fixed(row["published_mean"], 10),
                "max_abs_delta": _fixed(row["delta"], 12),
                "tolerance": _fixed(FULL_PRECISION_TOLERANCE, 12),
                "ok": row["ok"],
                "reason": row["reason"] or "",
            }
        )
    for row in rec["per_ratio_value_order"]:
        rows.append(
            {
                "check": "per_ratio_value_order",
                "subject": f"{_pct(row['rv_ratio'])}:{row['metric']}",
                "published_source": f"{PARENT_STUDY}/tables/per_ratio.csv",
                "checked": True,
                "derived": "",
                "published": "",
                "max_abs_delta": _fixed(row["max_abs_delta"], 10),
                "tolerance": _fixed(PRINTED_6DP_TOLERANCE, 10),
                "ok": row["ok"],
                "reason": row["reason"] or "",
            }
        )
    mov = rec["mean_of_views_is_runner_avg_cer"]
    rows.append(
        {
            "check": "mean_of_views_is_runner_avg_cer",
            "subject": f"{mov['n_cells']} cells",
            "published_source": f"{PARENT_STUDY}/tables/per_seed.csv[avg_cer]",
            "checked": True,
            "derived": "",
            "published": "",
            "max_abs_delta": _fixed(mov["max_abs_delta"], 12),
            "tolerance": _fixed(FULL_PRECISION_TOLERANCE, 12),
            "ok": mov["ok"],
            "reason": "",
        }
    )
    return _write_rows(RECONCILIATION_COLUMNS, rows)


# --------------------------------------------------------------------------- #
# Rendering: results.md.
# --------------------------------------------------------------------------- #
def _pm(stat: Mapping[str, Any], digits: int = 4) -> str:
    """``mean ± halfwidth (n=…)`` for a :class:`~...agg.Stat` dict."""
    if stat.get("mean") is None:
        return "—"
    body = f"{stat['mean']:.{digits}f}"
    if stat.get("ci95_halfwidth") is not None:
        body += f" ± {stat['ci95_halfwidth']:.{digits}f}"
    return f"{body} (n={stat.get('n')})"


def render_results_md(aggregate: Mapping[str, Any]) -> str:
    """The human-readable results page, leading with what n=3 does not permit."""
    out: list[str] = []
    ceiling = aggregate["n_ceiling"]
    out.append("# PACLIC v2 — endpoint effect sizes and the interior-ratio plateau")
    out.append("")
    out.append(
        f"**Status: {aggregate['status']}** — "
        f"{aggregate['coverage']['n_present']}/{aggregate['coverage']['n_cells']} "
        f"published cells, seeds {ceiling['seeds']}, corpus "
        f"`{aggregate['corpus']}`, `publishable: {aggregate['publishable']}`. "
        "Analysis-only: every number below is derived from the committed v2 tables "
        "listed under *Inputs*, with no run dir, no prediction CSV, no inference and "
        "no GPU."
    )
    out.append("")
    banner = aggregate.get("partial_banner")
    if banner:
        out.append(f"> **{banner}**")
        out.append("")

    out.append("## What these two numbers can and cannot support")
    out.append("")
    out.append(
        f"n = {ceiling['n_seeds']} seeds is the entire sample. The exact paired "
        "sign-flip permutation test enumerates "
        f"{ceiling['sign_assignments_enumerated']} sign assignments, so its smallest "
        f"attainable two-sided p is **{ceiling['exact_sign_test_min_p_two_sided']}**."
    )
    out.append("")
    out.append("**They support:**")
    out.append("")
    for item in ceiling["what_these_numbers_support"]:
        out.append(f"- {item}")
    out.append("")
    out.append("**They do not support:**")
    out.append("")
    for item in ceiling["what_these_numbers_do_not_support"]:
        out.append(f"- {item}")
    out.append("")
    out.append(f"*On the missing interval for dz.* {ceiling['dz_no_interval_reason']}")
    out.append("")

    out.append("## 1. Endpoint contrasts — Cohen's *dz* and the relative effect")
    out.append("")
    out.append(
        "Orientation is the published table's own `b − a`; direction is given by the "
        "**better arm** column rather than by a sign convention. `dz = mean(diff) / "
        "sample_sd(diff)`; `relative effect = mean(diff) / reference-arm mean`, with "
        "the denominator named per row."
    )
    out.append("")
    out.append(
        "| comparison | n | diff (b − a) | *dz* | *dz* interval | *dz* leave-one-seed-out | "
        "relative effect (denominator) | exact p (floor) |"
    )
    out.append("| :--- | ---: | :--- | ---: | :--- | :--- | :--- | :--- |")
    for e in aggregate["endpoint_effect_sizes"]:
        loo = e["cohens_dz_leave_one_seed_out"]
        dz = "—" if e["cohens_dz"] is None else f"{e['cohens_dz']:+.2f}"
        loo_text = (
            "—"
            if loo["min"] is None or loo["max"] is None
            else f"[{loo['min']:+.1f}, {loo['max']:+.1f}] (n=2 each, **not a CI**)"
        )
        rel = (
            "—"
            if e["relative_effect"] is None
            else f"{e['relative_effect']:+.4f} ({e['reference_arm_label']}"
            f" = {e['reference_arm_mean']:.4f})"
            + (
                " — **denominator smaller than the effect**"
                if e["relative_effect_exceeds_reference_arm"]
                else ""
            )
        )
        p = e["exact_sign_test"]
        out.append(
            f"| `{e['comparison']}` | {e['n']} | {_pm(e['diff'], 6)} | {dz} | "
            f"none reported | {loo_text} | {rel} | "
            f"{p['p_two_sided']:.2f} (floor {p['min_p_two_sided']:.2f}) |"
        )
    out.append("")
    out.append(
        "Every row is `significance_claim_available: false`. Each is also reconciled "
        "against the published contrast row it names — per-seed differences, mean, "
        "both CI bounds, `n`, `direction` and `sign_consistent` — so an effect size "
        "here can only belong to a contrast the main grid already published."
    )
    out.append("")

    out.append("## 2. The interior-ratio plateau (75:25 / 50:50 / 25:75)")
    out.append("")
    plateau = aggregate["plateau"]
    out.append(
        "Per-cell **worst view** is the harder of the two test views (larger error / "
        "smaller score); **mean of views** is their unweighted mean. Both are "
        "summarized across seeds with the same Student-t helper as every other v2 "
        "table."
    )
    out.append("")
    out.append("| quantity | 75:25 | 50:50 | 25:75 | reconciled against |")
    out.append("| :--- | ---: | ---: | ---: | :--- |")
    for quantity in PLATEAU_QUANTITIES:
        cells_text = " | ".join(
            _pm(plateau["per_ratio"][_ratio_key(r)][quantity.name])
            for r in INTERIOR_RV_RATIOS
        )
        source = (
            f"`{quantity.published_as}`" if quantity.published_as else "derived here"
        )
        out.append(f"| `{quantity.name}` | {cells_text} | {source} |")
    out.append("")
    verdict = plateau["verdict"]
    if verdict is None:
        out.append(f"**No ranking is quoted.** {plateau['verdict_withheld_because']}.")
    else:
        out.append(
            f"**{verdict['ranking_statement']}** {verdict['separability_statement']}"
        )
        out.append("")
        out.append(f"> {verdict['separability_note']}")
    out.append("")
    out.append(f"### Pairwise separability on `{RANK_PRIMARY}`")
    out.append("")
    out.append(
        "| pair | paired Δ | paired CI excludes 0 | marginal CIs overlap | "
        "all seeds agree | exact p | verdict |"
    )
    out.append("| :--- | :--- | :---: | :---: | :---: | :--- | :--- |")
    for r in plateau["separability"]:
        if r["quantity"] != RANK_PRIMARY:
            continue
        out.append(
            f"| {r['pair']} | {_pm(r['paired_diff'], 6)} | "
            f"{r['paired_ci95_excludes_zero']} | {r['marginal_ci95_overlap']} | "
            f"{r['sign_consistent']} | "
            f"{r['exact_sign_test_p_two_sided']:.2f} | `{r['verdict']}` |"
        )
    out.append("")
    out.append(
        "The full table (all eight quantities × three pairs) is "
        "`tables/plateau_separability.csv`."
    )
    out.append("")

    out.append("## 3. Reconciliation")
    out.append("")
    rec = aggregate["reconciliation"]
    n_contrast_ok = sum(1 for r in rec["contrasts"] if r["ok"])
    n_plateau_checked = [r for r in rec["plateau"] if r["checked"]]
    n_plateau_ok = sum(1 for r in n_plateau_checked if r["ok"])
    n_order_ok = sum(1 for r in rec["per_ratio_value_order"] if r["ok"])
    mov = rec["mean_of_views_is_runner_avg_cer"]
    out.append(
        f"- Published contrasts re-derived: **{n_contrast_ok}/"
        f"{len(rec['contrasts'])}** (per-seed differences, mean, both CI bounds, "
        "flags)."
    )
    out.append(
        f"- Derived worst-view / mean-view quantities matched against their published "
        f"counterparts: **{n_plateau_ok}/{len(n_plateau_checked)}**; "
        f"{len(rec['plateau']) - len(n_plateau_checked)} quantities have no published "
        "counterpart and are marked as derived here."
    )
    out.append(
        f"- `per_ratio.csv` unlabelled `values` in ascending-seed order: "
        f"**{n_order_ok}/{len(rec['per_ratio_value_order'])}** — the check that "
        "licenses pairing from `per_seed.csv`."
    )
    out.append(
        f"- `mean_of_views(cer)` reproduces the runner's own `avg_cer` on "
        f"**{mov['n_cells']}** cells (max |Δ| "
        f"{_fixed(mov['max_abs_delta'], 12) or '—'})."
    )
    out.append("")

    out.append("## Inputs")
    out.append("")
    for item in aggregate["inputs"]:
        digest = (item["sha256"] or "")[:12]
        out.append(f"- `{item['path']}` — sha256 `{digest}…`")
    out.append("")
    att = aggregate["input_attestation"]
    out.append(
        f"Input attestation: main grid `{att['maingrid']['status']}` / "
        f"`publishable: {att['maingrid']['publishable']}` / corpora "
        f"{att['maingrid']['attested_corpora']} attested on "
        f"{att['maingrid']['n_attested']} cells; selection bundle corpus "
        f"`{att['selection']['corpus']}` over {att['selection']['n_cells']} cells at "
        f"criterion `{att['selection']['criterion_used']}`."
    )
    out.append("")
    if aggregate["warnings"]:
        out.append("## Warnings (all publication-blocking)")
        out.append("")
        for warning in aggregate["warnings"]:
            out.append(f"- {warning}")
        out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Manifest.
# --------------------------------------------------------------------------- #
def build_manifest(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance + the two arithmetic definitions this bundle adds."""
    return {
        "study": STUDY,
        "issue": ISSUE,
        "parent_issue": 122,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "corpus": EXPECTED_CORPUS,
        "analysis_only": True,
        "no_training_inference_or_gpu": True,
        "inputs": aggregate["inputs"],
        "input_attestation": aggregate["input_attestation"],
        "reused_without_reimplementation": {
            "mean_sd_student_t_ci": (
                "scripts.evaluation.aggregate_accent_ratio_multiseed.summarize"
            ),
            "paired_contrast": (
                "scripts.evaluation.aggregate_accent_ratio_multiseed._contrast"
            ),
            "exact_paired_sign_flip_test": (
                "scripts.evaluation.build_paclic_accent_transfer_analysis."
                "sign_flip_permutation_test"
            ),
        },
        "definitions_added_here": {
            "cohens_dz": "mean(paired diff) / sample_sd(paired diff), ddof=1 "
            "(paired standardized mean difference; NOT Cohen's d — no pooled SD)",
            "relative_effect": "mean(paired diff) / reference_arm_mean, with the "
            "reference arm declared per contrast",
            "worst_view": "max(rv, ms) for an error metric, min(rv, ms) for a score",
            "mean_of_views": "(rv + ms) / 2",
        },
        "orientation": (
            "every contrast is reported in the published v2 table's own b − a "
            "orientation and never re-oriented; direction is carried by better_arm"
        ),
        "reference_arms": [
            {
                "comparison": spec.comparison,
                "reference_arm": spec.reference,
                "reference_arm_label": spec.reference_arm().label,
                "rationale": spec.reference_rationale,
            }
            for spec in CONTRASTS
        ],
        "n_ceiling": aggregate["n_ceiling"],
        "reconciliation_tolerances": aggregate["reconciliation"]["tolerances"],
        "v1_comparison": (
            "none. v1 is abandoned; its table shapes are reused for recognisability "
            "and its zero-shot relative-effect denominator convention is preserved, "
            "but no v1 number is read, quoted or differenced."
        ),
        "reproduce": (
            f"uv run python -m scripts.evaluation.{Path(__file__).stem} "
            f"--out-dir {DEFAULT_OUT_DIR}"
        ),
        "generated_by": f"scripts.evaluation.{Path(__file__).stem}",
    }


# --------------------------------------------------------------------------- #
# Bundle + CLI.
# --------------------------------------------------------------------------- #
def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def analyze(
    maingrid_dir: Path, selection_dir: Path
) -> tuple[list[PublishedCell], dict[str, Any]]:
    """Read both bundles and produce the full aggregate payload."""
    cells = load_published_cells(maingrid_dir)
    published_contrasts = load_published_contrasts(maingrid_dir)
    published_per_ratio = load_published_per_ratio(maingrid_dir)
    published_worst = load_published_worst(selection_dir)

    effects = endpoint_effect_sizes(cells, published_contrasts)
    summaries = plateau_summaries(cells)
    ranking = plateau_ranking(summaries)
    separability = plateau_separability(cells, summaries)

    aggregate = build_aggregate(
        cells,
        effects,
        summaries,
        ranking,
        separability,
        coverage=coverage_summary(cells),
        attestation=input_attestation(maingrid_dir, selection_dir),
        plateau_reconciliation=reconcile_plateau(summaries, published_worst),
        value_order=reconcile_per_ratio_values(cells, published_per_ratio),
        mean_of_views=reconcile_mean_of_views(cells),
        extra_published=unexpected_published_contrasts(published_contrasts),
        inputs=input_files(maingrid_dir, selection_dir),
    )
    return cells, aggregate


def build_bundle(
    cells: Sequence[PublishedCell], aggregate: Mapping[str, Any], out_dir: Path
) -> None:
    """Write ``tables/`` + ``results.md`` + ``aggregate.json`` + ``manifest.json``."""
    tables = out_dir / "tables"
    plateau = aggregate["plateau"]
    summaries = plateau_summaries(cells)
    _write(
        tables / "endpoint_effect_sizes.csv",
        render_endpoint_csv(aggregate["endpoint_effect_sizes"]),
    )
    _write(
        tables / "endpoint_dz_leave_one_seed_out.csv",
        render_dz_folds_csv(aggregate["endpoint_effect_sizes"]),
    )
    _write(
        tables / "plateau_comparison.csv",
        render_plateau_csv(
            cells,
            summaries,
            plateau["ranking"],
            aggregate["reconciliation"]["plateau"],
        ),
    )
    _write(
        tables / "plateau_separability.csv",
        render_separability_csv(plateau["separability"]),
    )
    _write(tables / "reconciliation.csv", render_reconciliation_csv(aggregate))
    _write(out_dir / "results.md", render_results_md(aggregate))
    _write_json(out_dir / "aggregate.json", aggregate)
    _write_json(out_dir / "manifest.json", build_manifest(aggregate))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maingrid-dir",
        type=Path,
        default=DEFAULT_MAINGRID_DIR,
        help=(
            "Committed v2 main-grid bundle (per_seed.csv, per_ratio.csv, "
            "asymmetry.csv, retention_adaptation.csv, aggregate.json)."
        ),
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=DEFAULT_SELECTION_DIR,
        help=(
            "Committed v2 checkpoint-selection-sensitivity bundle, read only for its "
            f"published worst-view per-ratio rows at criterion={SELECTION_CRITERION}."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Bundle dir: writes tables/, results.md, aggregate.json, manifest.json. "
            f"Omit to print the results page to stdout. Conventional: {DEFAULT_OUT_DIR}"
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        cells, aggregate = analyze(args.maingrid_dir, args.selection_dir)
    except InputError as exc:
        print(f"[{STUDY}][error] {exc}", file=sys.stderr)
        return 1

    for warning in aggregate["warnings"]:
        print(f"[{STUDY}][warn] {warning}", file=sys.stderr)

    if args.out_dir is not None:
        build_bundle(cells, aggregate, args.out_dir)
        print(f"[{STUDY}] bundle -> {args.out_dir}", file=sys.stderr)
    else:
        print(render_results_md(aggregate))

    return 1 if aggregate["hard_warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
