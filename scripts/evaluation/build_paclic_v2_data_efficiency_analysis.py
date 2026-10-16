"""Build paclic v2 data efficiency analysis: PACLIC 2026 v2 release."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from . import aggregate_accent_ratio_multiseed as agg
from . import build_paclic_v2_effect_sizes_and_plateau as esp
from . import build_paclic_v2_maingrid_analysis as v2mg
from .build_paclic_accent_transfer_analysis import sign_flip_permutation_test

# --------------------------------------------------------------------------- #
# Study identity + the four-point axis.
# --------------------------------------------------------------------------- #
#: Study slug, used in log prefixes and the emitted artifacts.
STUDY: str = "paclic-v2-data-efficiency"
#: The launcher that trained the NEW (low-data) cells.
LAUNCHER: str = "scripts/reproduce.py data-efficiency"
#: Its immutable pins + planned grid.
MANIFEST_YAML: str = (
    "release_manifest.json"
)
#: The launcher whose cells are inherited by reference as the 100% point.
INHERITED_LAUNCHER: str = "scripts/reproduce.py maingrid"

#: The lowest point of the curve (10% of the shared parallel keys).
LOW_TOTAL: int = 1042
#: The 100% point, inherited from the v2 main grid — never re-run here.
FULL_TOTAL: int = 10420
#: The totals THIS study's launcher trains, ascending: 10 / 25 / 50 %. All three live
#: in one study root and are named by one ``run_suffix`` template; they differ only in
#: which submission of the launcher trained them, which is not a provenance
#: distinction (same pins, same recipe, same corpus, same launcher file).
TRAINED_TOTALS: tuple[int, ...] = (LOW_TOTAL, 2605, 5210)
#: The curve's x axis, ascending: the trained totals then the inherited 100% point.
TOTALS: tuple[int, ...] = (*TRAINED_TOTALS, FULL_TOTAL)
#: ``total`` → integer percent of :data:`FULL_TOTAL`, for labels and tables. Exact by
#: construction: 10420 is divisible by 10, 4 and 2.
PCT_BY_TOTAL: dict[int, int] = {LOW_TOTAL: 10, 2605: 25, 5210: 50, FULL_TOTAL: 100}

#: The three mix ratios shared by every point of the curve — a strict subset of
#: :data:`agg.RV_RATIOS`, so every reused ``agg`` helper still applies per total.
#: 0.75 / 0.25 exist at 10420 only (main grid) and are part of no point here.
RV_RATIOS: tuple[float, ...] = (1.0, 0.5, 0.0)
#: Seeds, all three trained at EVERY point. This is what gives each point a CI.
SEEDS: tuple[int, ...] = agg.SEEDS

#: Target size of the join. ``complete`` is measured against THIS, never against
#: ``len(cells)``: a caller that loads a subset of the axes must not be told its
#: 12-cell join is a finished curve.
TARGET_N_CELLS: int = len(TOTALS) * len(RV_RATIOS) * len(SEEDS)
#: Of those, the ones this study had to train (the rest are inherited).
TARGET_N_NEW_CELLS: int = len(TRAINED_TOTALS) * len(RV_RATIOS) * len(SEEDS)

#: ``source`` stamped on each cell, so no reader has to infer provenance from a
#: directory name. Written into every table that lists cells.
CELL_SOURCE_BY_TOTAL: dict[int, str] = {
    **{total: "v2-data-efficiency" for total in TRAINED_TOTALS},
    FULL_TOTAL: "v2-maingrid-inherited",
}
#: ``inherited`` vs ``newly_trained``, for the provenance column.
PROVENANCE_BY_TOTAL: dict[int, str] = {
    **{total: "newly_trained" for total in TRAINED_TOTALS},
    FULL_TOTAL: "inherited",
}

DEFAULT_LOW_STUDY_ROOT = Path("outputs/experiments_paclic_v2_data_efficiency")
DEFAULT_FULL_STUDY_ROOT = v2mg.DEFAULT_STUDY_ROOT
DEFAULT_SHARED_LOW_STUDY_ROOT = (
    "./outputs/experiments_paclic_v2_data_efficiency"
)
DEFAULT_SHARED_FULL_STUDY_ROOT = v2mg.DEFAULT_SHARED_STUDY_ROOT
DEFAULT_SLURM_LOG_DIR = v2mg.DEFAULT_SLURM_LOG_DIR

#: Slurm ``.out`` basenames per arm. The two arms are attested from DIFFERENT log
#: sets: run names differ (``…_n<total>_seed_n`` vs ``…_seed_n``), but keeping the globs
#: apart also means a main-grid log can never be matched to a trained cell even if a
#: future launcher collided the names. All trained totals share one glob because one
#: launcher wrote them; they are still told apart by the run name inside the log.
DATAEFF_SLURM_LOG_GLOB: str = "paclic_v2_data_efficiency_*.out"
SLURM_LOG_GLOB_BY_TOTAL: dict[int, str] = {
    **{total: DATAEFF_SLURM_LOG_GLOB for total in TRAINED_TOTALS},
    FULL_TOTAL: v2mg.SLURM_LOG_GLOB,
}

DEFAULT_OUT_DIR = Path(
    "results/paclic-v2-data-efficiency"
)
#: Its OWN figure dir. The committed main-grid figures live in
#: ``docs/papers/paclic2026/figures/`` next to a ``manifest.json`` / ``README.md``
#: that describe *that* study; writing here would overwrite them (and
#: :func:`build_figures` refuses to write into another study's figure dir anyway).
DEFAULT_FIGURES_OUT_DIR = Path("outputs/figures/data-efficiency")

#: The committed main-grid bundle whose 10420 cells this curve inherits. Read (never
#: written) to verify the inherited arm is the published one.
DEFAULT_MAINGRID_AGGREGATE = v2mg.DEFAULT_OUT_DIR / "aggregate.json"

#: Curve metrics summarized per ``(total, ratio)`` — a subset of
#: :data:`agg.AGG_METRICS`, so ``agg.aggregate_by_ratio`` yields them directly.
#: Ordered for the tables: the aggregate CER picture, then the two channels the
#: accent question actually lives in.
CURVE_METRICS: tuple[str, ...] = (
    "avg_cer",
    "rv_cer",
    "ms_cer",
    "rv_accent_stripped_cer",  # base letters, RV test
    "ms_accent_stripped_cer",  # base letters, MS test
    "rv_accent_f1",  # accent marks, RV test
    "ms_accent_f1",  # accent marks, MS test
)

#: Absolute tolerance when checking this join's inherited means against the committed
#: main-grid ``aggregate.json``. Both sides are the same ``agg.summarize`` over the
#: same ``results.json`` floats, so the only expected difference is JSON
#: serialisation round-tripping; anything above this is a different set of cells.
MAINGRID_MEAN_TOLERANCE: float = 1e-12

# --------------------------------------------------------------------------- #
# Which contrasts are published — four of the six pairs, and why.
# --------------------------------------------------------------------------- #
#: ``consecutive_step`` — one rung of the ladder. These are what a shape question is
#: about: "where does the gain sit".
KIND_STEP: str = "consecutive_step"
#: ``endpoint`` — 10%→100%, the number the paper quotes. Also the sum of the three
#: consecutive steps, per seed, which is why it is not double counting: it is a
#: *presentation* of the same paired differences, not a fifth measurement.
KIND_ENDPOINT: str = "endpoint"

#: The three consecutive steps, ``(from_total, to_total)`` ascending.
CONSECUTIVE_STEPS: tuple[tuple[int, int], ...] = tuple(zip(TOTALS, TOTALS[1:]))
#: The endpoint contrast.
ENDPOINT_PAIR: tuple[int, int] = (TOTALS[0], TOTALS[-1])
#: Every published contrast, in report order: the steps, then the endpoint.
CONTRAST_PAIRS: tuple[tuple[int, int], ...] = (*CONSECUTIVE_STEPS, ENDPOINT_PAIR)
#: The pairs deliberately NOT emitted. Recomputed here rather than hard-coded so a
#: change to the axis cannot leave a stale list behind.
OMITTED_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (a, b)
    for i, a in enumerate(TOTALS)
    for b in TOTALS[i + 1 :]
    if (a, b) not in CONTRAST_PAIRS
)

#: Why four contrasts and not all six. Carried into the aggregate, the manifest and the
#: results page, because "we did not publish that comparison" has to be a stated
#: decision rather than a gap a reader discovers.
CONTRAST_SELECTION_NOTE: str = (
    f"{len(TOTALS)} points admit {len(TOTALS) * (len(TOTALS) - 1) // 2} ordered pairs; "
    f"this bundle publishes {len(CONTRAST_PAIRS)} — the "
    f"{len(CONSECUTIVE_STEPS)} consecutive steps plus the endpoint "
    f"({ENDPOINT_PAIR[0]}→{ENDPOINT_PAIR[1]}) — and omits "
    + ", ".join(f"{a}→{b}" for a, b in OMITTED_PAIRS)
    + ". Publishing all of them at n=3 would multiply the family for no new "
    "information: every contrast here is a WITHIN-SEED difference of the same seed's "
    "values, so each omitted pair is the exact per-seed sum of published consecutive "
    "steps (e.g. 1042→5210 = [1042→2605] + [2605→5210], seed by seed) and is "
    "reconstructible by adding two per_seed_diff columns of tables/curve_deltas.csv. "
    "The published set is therefore minimal and complete: the steps carry the shape "
    "question, the endpoint carries the headline, and nothing that was measured is "
    "hidden."
)

# --------------------------------------------------------------------------- #
# Separability vocabulary — reused verbatim from the plateau analysis.
# --------------------------------------------------------------------------- #
#: A step whose three per-seed differences all point the same way. A DESCRIPTION of
#: three seeds, never a separability claim.
STEP_SIGN_CONSISTENT: str = "consistently_signed_but_not_separable"
#: A step whose per-seed differences disagree in sign.
STEP_SIGN_UNSTABLE: str = "sign_unstable_across_seeds"
#: Two steps whose order is the same in all three seeds / not.
ORDER_CONSISTENT: str = esp.ORDER_CONSISTENT
ORDER_UNSTABLE: str = esp.ORDER_UNSTABLE
#: No paired seed at all — mid-flight, or a cell that never landed. Distinct from
#: "unstable" on purpose: ``agg._contrast`` reports ``sign_consistent=False`` for an
#: empty difference set, and printing that as instability would turn a missing cell
#: into a measured disagreement.
NOT_COMPUTABLE: str = "not_computable_no_paired_seeds"

#: Why nothing here can be separable, quoted into every artifact that ranks steps.
#: The plateau bundle's note, restated for step magnitudes rather than mix ratios; the
#: p-floor argument and the ``2/2**3 = 0.25`` figure are the same and come from there.
SEPARABILITY_NOTE: str = (
    "No step and no pair of steps is statistically separable at n=3. The exact paired "
    "sign-flip permutation test's smallest attainable two-sided p is 2/2**3 = 0.25, so "
    "neither 'this step is a real gain' nor 'this step is larger than that one' can be "
    "established in either direction. A step is classified only as consistently signed "
    "across the three seeds, and a pair of steps only as consistently ordered across "
    f"them (a description of those seeds) or unstable. Source of the floor: {esp.STUDY} "
    "(SEPARABILITY_NOTE), which reaches the same ceiling on the ratio plateau."
)

#: The verdict a four-point curve at n=3 is allowed to reach about its own shape.
SHAPE_NOT_DETERMINED: str = "shape_not_determined_at_n3"

#: What may and may not be written about the shape. This is the whole reason the
#: interior points were trained, so the limit is stated at the same volume as the gain.
SHAPE_CLAIM_NOTE: str = (
    "The four points do NOT determine a curve shape. 'Front-loaded', 'log-linear', "
    "'saturates by X%' and 'the gain is concentrated in the first step' are all claims "
    "about how the step magnitudes compare, and comparing them needs the steps to be "
    "separable from each other, which n=3 cannot deliver (separability_note). What the "
    "curve does establish is each point's own mean with an interval, and each step's "
    "paired difference with an interval and a sign-consistency flag. Step rankings in "
    "this bundle are printed only next to the separability table and are ordered "
    "DESCRIPTIONS of three seeds, never findings. The interior points still earn their "
    "cost: they replace an unmeasured interval between 10% and 100% with measured "
    "intermediate levels, which is what rules out 'the curve was never looked at "
    "between the endpoints' — a weaker and honest statement than a shape."
)

#: The metric the step ranking / shape verdict is reported on, per mix ratio. Chosen
#: as the aggregate CER picture, the same quantity the curve tables lead with; the
#: separability table itself covers every :data:`CURVE_METRICS` entry, and the verdict
#: is emitted per ratio because a shape need not be shared across mixes.
SHAPE_PRIMARY_METRIC: str = "avg_cer"


# Re-exported so callers gate on one exception hierarchy across both v2 builders.
FigureGateError = v2mg.FigureGateError
IncompleteGridError = v2mg.IncompleteGridError
UnverifiedCorpusError = v2mg.UnverifiedCorpusError
GridInvariantError = v2mg.GridInvariantError


class FigureDirConflictError(FigureGateError):
    """Raised when the requested figure dir already belongs to another study.

    ``docs/papers/paclic2026/figures/`` holds the committed main-grid figure set with
    its own ``manifest.json`` / ``README.md``. Rendering this study there would
    silently replace that provenance with this study's, leaving two figures described
    by a manifest that names only one of them.
    """


def pct_of(total: int) -> int:
    """Percent-of-full label for a total (falls back to a rounded ratio)."""
    return PCT_BY_TOTAL.get(total, round(100 * total / FULL_TOTAL))


# --------------------------------------------------------------------------- #
# Cell naming + loading (two roots: the trained totals' root and the main grid's).
# --------------------------------------------------------------------------- #
def run_name(rv_ratio: float, seed: int, *, total: int) -> str:
    """Run-dir name for a cell of this curve, which depends on **which arm** it is.

    * ``total == FULL_TOTAL`` → delegated to :func:`v2mg.run_name`, i.e.
      ``rv_<pct>_ms_<pct>_seed_<n>``. Delegating rather than re-deriving is the point:
      the inherited arm can only ever address the dirs the main-grid bundle addressed,
      so the two studies cannot drift apart on naming.
    * otherwise → ``rv_<pct>_ms_<pct>_n<total>_seed_<n>``, this launcher's
      ``run_suffix_template: "n{total}_seed_{seed}"``. Encoding **both** total and seed
      is what makes a trained cell self-describing, keeps the three trained totals
      apart inside one study root, and makes it impossible to confuse any of them with
      an inherited 100% dir in an aggregation.

    That the two arms need two different names is exactly why a single-root loader
    cannot read this study.
    """
    if total == FULL_TOTAL:
        return v2mg.run_name(rv_ratio, seed)
    rv_pct = int(round(rv_ratio * 100))
    ms_pct = int(round((1.0 - rv_ratio) * 100))
    return f"rv_{rv_pct}_ms_{ms_pct}_n{total}_seed_{seed}"


def study_root_for(total: int, *, low_study_root: Path, full_study_root: Path) -> Path:
    """Which root a total's cells live under.

    ``low_study_root`` holds every TRAINED total (they are one launcher's output root,
    told apart by the ``n<total>`` token in the run name); ``full_study_root`` holds the
    inherited 100% cells and is never written to.
    """
    return full_study_root if total == FULL_TOTAL else low_study_root


def load_curve(
    low_study_root: Path,
    full_study_root: Path,
    *,
    metrics_fn: Optional[agg.MetricsFn] = None,
    cross_check: bool = True,
    shared_low_study_root: Optional[str] = None,
    shared_full_study_root: Optional[str] = None,
    totals: Sequence[int] = TOTALS,
    rv_ratios: Sequence[float] = RV_RATIOS,
    seeds: Sequence[int] = SEEDS,
) -> tuple[dict[int, list[agg.SeedCell]], list[dict[str, Any]]]:
    """Load ``total → [cells]`` from the two roots, ratio-major within each total.

    Every total's cell list is a plain ``agg`` grid over the three shared ratios, so
    ``agg.aggregate_by_ratio`` / ``agg.asymmetry`` apply to it unchanged; the *curve*
    is the series of those per-total summaries.

    Absent cells come back ``present=False`` rather than raising — the trained arm is
    expected to be read mid-flight — and a ``results.json`` that exists but does not
    parse is treated as absent (:func:`v2mg._results_json_parses`) instead of killing
    the aggregation, then re-detected as a hard warning by :func:`grid_integrity`.

    The metrics cache is shared across all totals, so a prediction CSV is re-evaluated
    once no matter how many series read it.
    """
    metrics = agg._MetricsCache(metrics_fn or agg.default_metrics_fn)
    curve: dict[int, list[agg.SeedCell]] = {}
    checks: list[dict[str, Any]] = []
    for total in totals:
        root = study_root_for(
            total, low_study_root=low_study_root, full_study_root=full_study_root
        )
        shared_root = (
            shared_full_study_root if total == FULL_TOTAL else shared_low_study_root
        )
        cells: list[agg.SeedCell] = []
        for rv_ratio in rv_ratios:
            for seed in seeds:
                name = run_name(rv_ratio, seed, total=total)
                shared_path = (
                    f"{shared_root.rstrip('/')}/{name}" if shared_root else None
                )
                source = CELL_SOURCE_BY_TOTAL.get(total, f"v2-n{total}")
                if not v2mg._results_json_parses(root / name):
                    cells.append(
                        agg.SeedCell(
                            rv_ratio=rv_ratio,
                            seed=seed,
                            run_name=name,
                            source=source,
                            present=False,
                            shared_path=shared_path,
                        )
                    )
                    continue
                cell, cell_checks = agg.read_cell_dir(
                    root / name,
                    rv_ratio=rv_ratio,
                    seed=seed,
                    run_name=name,
                    source=source,
                    shared_path=shared_path,
                    metrics=metrics,
                    cross_check=cross_check,
                )
                cells.append(cell)
                checks.extend(cell_checks)
        curve[total] = cells
    return curve, checks


def all_cells(curve: dict[int, list[agg.SeedCell]]) -> list[agg.SeedCell]:
    """Every cell of the join, in curve order (ascending total, then ratio-major)."""
    return [cell for total in sorted(curve) for cell in curve[total]]


def total_of(curve: dict[int, list[agg.SeedCell]], cell: agg.SeedCell) -> Optional[int]:
    """Which total a cell belongs to (``agg.SeedCell`` carries no total field)."""
    for total, cells in curve.items():
        if any(c is cell for c in cells):
            return total
    return None


# --------------------------------------------------------------------------- #
# Coverage — per point, per seed, and the PAIRED n behind every delta.
# --------------------------------------------------------------------------- #
def contrast_key(from_total: int, to_total: int) -> str:
    """``"1042->2605"`` — the stable key every contrast is filed under."""
    return f"{from_total}->{to_total}"


def contrast_kind(from_total: int, to_total: int) -> str:
    """:data:`KIND_STEP` for a consecutive rung, :data:`KIND_ENDPOINT` for 10%→100%."""
    return KIND_STEP if (from_total, to_total) in CONSECUTIVE_STEPS else KIND_ENDPOINT


def contrast_label(from_total: int, to_total: int) -> str:
    """``"10→25%"`` — the human label used in tables and statements."""
    return f"{pct_of(from_total)}→{pct_of(to_total)}%"


def coverage_summary(curve: dict[int, list[agg.SeedCell]]) -> dict[str, Any]:
    """Coverage of the target join: per total, per point, per seed, and paired.

    ``complete`` is the only flag an artifact may use to claim a finished curve.

    ``per_point[…].n_present`` is the n behind that *point's* mean/CI, and
    ``paired[…].n_paired`` is the n behind ONE published contrast at one ratio — the
    number of seeds present at BOTH of that contrast's totals. They are different
    numbers and all of them are reported, because a paired design with 3/3 at one end
    and 2/3 at the other has an n=2 delta, not an n=3 one, and on a four-point curve a
    single missing cell can leave some steps at n=3 and others at n=2.
    """
    per_total: dict[str, Any] = {}
    per_point: dict[str, Any] = {}
    for total in TOTALS:
        cells = curve.get(total, [])
        present = [c for c in cells if c.present]
        per_total[str(total)] = {
            "total_train_samples": total,
            "pct": pct_of(total),
            "source": CELL_SOURCE_BY_TOTAL.get(total),
            "provenance": PROVENANCE_BY_TOTAL.get(total),
            "n_cells": len(RV_RATIOS) * len(SEEDS),
            "n_cells_loaded": len(cells),
            "n_present": len(present),
            "complete": len(present) == len(RV_RATIOS) * len(SEEDS),
        }
        for rv_ratio in RV_RATIOS:
            group = [c for c in cells if c.rv_ratio == rv_ratio]
            group_present = [c for c in group if c.present]
            per_point[f"{total}:{rv_ratio:g}"] = {
                "total_train_samples": total,
                "pct": pct_of(total),
                "rv_ratio": rv_ratio,
                "ms_ratio": round(1.0 - rv_ratio, 2),
                "n_cells": len(SEEDS),
                "n_cells_loaded": len(group),
                "n_present": len(group_present),
                "seeds_present": sorted(c.seed for c in group_present),
                "seeds_missing": sorted(set(SEEDS) - {c.seed for c in group_present}),
                "complete": len(group_present) == len(SEEDS),
            }

    # One paired block per (published contrast × ratio): on a four-point curve the
    # paired n is a property of a PAIR, not of a ratio, so a ratio-keyed block would
    # have to pick one of four numbers.
    paired: dict[str, Any] = {}
    for from_total, to_total in CONTRAST_PAIRS:
        for rv_ratio in RV_RATIOS:
            seeds_by_total = {
                total: {
                    c.seed
                    for c in curve.get(total, [])
                    if c.present and c.rv_ratio == rv_ratio
                }
                for total in (from_total, to_total)
            }
            shared = sorted(set.intersection(*seeds_by_total.values()))
            paired[f"{contrast_key(from_total, to_total)}:{rv_ratio:g}"] = {
                "contrast": contrast_key(from_total, to_total),
                "contrast_kind": contrast_kind(from_total, to_total),
                "label": contrast_label(from_total, to_total),
                "from_total": from_total,
                "to_total": to_total,
                "from_pct": pct_of(from_total),
                "to_pct": pct_of(to_total),
                "rv_ratio": rv_ratio,
                "ms_ratio": round(1.0 - rv_ratio, 2),
                "seeds_paired": shared,
                "n_paired": len(shared),
                "n_expected": len(SEEDS),
                "complete": len(shared) == len(SEEDS),
                "seeds_by_total": {
                    str(total): sorted(seeds)
                    for total, seeds in sorted(seeds_by_total.items())
                },
            }

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        cells = [c for total in TOTALS for c in curve.get(total, []) if c.seed == seed]
        present = [c for c in cells if c.present]
        per_seed[str(seed)] = {
            "seed": seed,
            "n_cells": len(TOTALS) * len(RV_RATIOS),
            "n_cells_loaded": len(cells),
            "n_present": len(present),
            "complete": len(present) == len(TOTALS) * len(RV_RATIOS),
        }

    present_names = {c.run_name for c in all_cells(curve) if c.present}
    n_present = len(present_names)
    n_new_present = sum(
        1 for total in TRAINED_TOTALS for c in curve.get(total, []) if c.present
    )
    return {
        "n_cells": TARGET_N_CELLS,
        "n_cells_loaded": len(all_cells(curve)),
        "n_present": n_present,
        "complete": n_present == TARGET_N_CELLS,
        "n_new_cells": TARGET_N_NEW_CELLS,
        "n_new_present": n_new_present,
        "new_cells_complete": n_new_present == TARGET_N_NEW_CELLS,
        "target_totals": list(TOTALS),
        "target_trained_totals": list(TRAINED_TOTALS),
        "target_inherited_total": FULL_TOTAL,
        "target_rv_ratios": list(RV_RATIOS),
        "target_seeds": list(SEEDS),
        "published_contrasts": [contrast_key(a, b) for a, b in CONTRAST_PAIRS],
        "per_total": per_total,
        "per_point": per_point,
        "paired": paired,
        "per_seed": per_seed,
        # Enumerated from the TARGET join, so a cell that was never even loaded
        # (subset axes) counts as missing rather than vanishing.
        "missing_cells": [
            name
            for name in (
                run_name(rv_ratio, seed, total=total)
                for total in TOTALS
                for rv_ratio in RV_RATIOS
                for seed in SEEDS
            )
            if name not in present_names
        ],
    }


def status_label(coverage: dict[str, Any]) -> str:
    """``"COMPLETE"`` or ``"PARTIAL"`` — the word every artifact leads with."""
    return "COMPLETE" if coverage["complete"] else "PARTIAL"


def per_point_n_phrase(coverage: dict[str, Any]) -> str:
    """``n1042 100:0 n=3/3; … n10420 0:100 n=3/3`` — never a single join-wide n."""
    return "; ".join(
        f"n{blk['total_train_samples']} "
        f"{int(round(blk['rv_ratio'] * 100))}:{int(round(blk['ms_ratio'] * 100))} "
        f"n={blk['n_present']}/{blk['n_cells']}"
        for blk in coverage["per_point"].values()
    )


def paired_n_phrase(coverage: dict[str, Any]) -> str:
    """``10→25% 100:0 paired n=3/3; …`` — the n behind each published delta.

    One entry per (contrast × ratio), never one per ratio: which seeds a delta has
    depends on which two totals it spans.
    """
    return "; ".join(
        f"{blk['label']} "
        f"{int(round(blk['rv_ratio'] * 100))}:{int(round(blk['ms_ratio'] * 100))} "
        f"paired n={blk['n_paired']}/{blk['n_expected']}"
        for blk in coverage["paired"].values()
    )


def partial_banner(coverage: dict[str, Any]) -> Optional[str]:
    """The PARTIAL banner, or ``None`` when the curve really is whole.

    Same convention as ``build_paclic_v2_maingrid_analysis`` and
    ``issue161_retrain_deltas``: the words "PARTIAL" and "NOT A COMPLETE RESULT", the
    present/expected counts, and the explicit list of what is missing. Every published
    contrast's paired n is in the banner too, because on a multi-point curve one missing
    cell silently drops the n of *some* deltas and not others, and a reader will
    otherwise assume 3 everywhere.
    """
    if coverage["complete"]:
        return None
    missing = coverage["missing_cells"]
    shown = ", ".join(missing[:18]) + (" …" if len(missing) > 18 else "")
    return (
        f"PARTIAL — NOT A COMPLETE RESULT: {coverage['n_present']}/"
        f"{coverage['n_cells']} cells present "
        f"({coverage['n_new_present']}/{coverage['n_new_cells']} newly trained at "
        f"totals {list(TRAINED_TOTALS)}; the n={FULL_TOTAL} point is inherited from the "
        f"v2 main grid). Per point: {per_point_n_phrase(coverage)}. Paired: "
        f"{paired_n_phrase(coverage)}. No value below may be presented as the v2 "
        "data-efficiency result, no point's mean/CI is a 3-seed number unless its own "
        "n says 3, and no step or endpoint delta is a 3-seed delta unless its PAIRED n "
        f"says 3. Still missing: {shown}"
    )


# --------------------------------------------------------------------------- #
# Curve statistics — every number here comes out of ``agg``.
# --------------------------------------------------------------------------- #
def aggregate_curve(
    curve: dict[int, list[agg.SeedCell]],
) -> dict[int, dict[float, dict[str, agg.Stat]]]:
    """``total → ratio → metric → Stat``, via ``agg.aggregate_by_ratio`` per total.

    ``agg.aggregate_by_ratio`` summarizes over the *five* ``agg.RV_RATIOS`` and returns
    ``n=0`` stats for the two this study does not train. Those are dropped here rather
    than carried: at 10420 the 0.75 / 0.25 cells genuinely exist (in the main grid), so
    an ``n=0`` row for them in *this* bundle would assert something false.
    """
    return {
        total: {
            rv_ratio: metrics
            for rv_ratio, metrics in agg.aggregate_by_ratio(cells).items()
            if rv_ratio in RV_RATIOS
        }
        for total, cells in curve.items()
    }


def curve_series(
    per_total: dict[int, dict[float, dict[str, agg.Stat]]],
    rv_ratio: float,
    metric: str,
) -> list[tuple[int, agg.Stat]]:
    """``[(total, Stat)]`` ascending — one ratio's series for one metric."""
    return [
        (total, per_total[total][rv_ratio][metric])
        for total in TOTALS
        if total in per_total and rv_ratio in per_total[total]
    ]


def _seed_values(
    cells: Sequence[agg.SeedCell], rv_ratio: float, metric: str
) -> dict[int, float]:
    """``seed → value`` for one ``(ratio, metric)`` over present cells.

    Reuses ``agg._flat_metrics`` for the flattening, so the metric names here are
    exactly the names ``agg.aggregate_by_ratio`` summarizes and ``avg_cer`` (which is
    not a per-view metric, so ``agg._paired`` cannot reach it) is included on the same
    footing.
    """
    out: dict[int, float] = {}
    for cell in cells:
        if not cell.present or cell.rv_ratio != rv_ratio:
            continue
        value = agg._flat_metrics(cell).get(metric)
        if value is not None:
            out[cell.seed] = float(value)
    return out


def one_contrast(
    curve: dict[int, list[agg.SeedCell]],
    from_total: int,
    to_total: int,
    rv_ratio: float,
    metric: str,
) -> dict[str, Any]:
    """One published contrast: ``value(to_total) − value(from_total)``, paired by seed.

    ``agg._contrast`` is used verbatim — the same paired machinery behind the main
    grid's asymmetry and retention tables — so the delta's mean, sample SD, Student-t
    CI, per-seed differences and sign-consistency flag are computed exactly as they are
    there. Only the two arms differ: instead of two ratios at one total, this pairs one
    ratio across two totals.

    A positive ``diff`` means the metric *rose* with more data (worse for CER, better
    for accent F1) — the direction is carried by ``direction`` / ``sign_consistent``
    rather than being editorialised into the key names.
    """
    return {
        "contrast": contrast_key(from_total, to_total),
        "contrast_kind": contrast_kind(from_total, to_total),
        "label": contrast_label(from_total, to_total),
        "from_total": from_total,
        "to_total": to_total,
        "from_pct": pct_of(from_total),
        "to_pct": pct_of(to_total),
        **agg._contrast(
            _seed_values(curve.get(from_total, []), rv_ratio, metric),
            _seed_values(curve.get(to_total, []), rv_ratio, metric),
        ),
    }


def curve_deltas(curve: dict[int, list[agg.SeedCell]]) -> dict[str, Any]:
    """The published paired contrasts: three consecutive steps + the endpoint.

    Shaped ``by_pair[<contrast>][<ratio>][<metric>]`` with a ``pairs`` index and the
    selection rationale alongside, so a reader of ``aggregate.json`` sees which
    comparisons exist, which were deliberately not emitted, and why, without leaving
    the payload (:data:`CONTRAST_SELECTION_NOTE`).
    """
    by_pair: dict[str, Any] = {}
    for from_total, to_total in CONTRAST_PAIRS:
        by_pair[contrast_key(from_total, to_total)] = {
            f"{rv_ratio:g}": {
                metric: one_contrast(curve, from_total, to_total, rv_ratio, metric)
                for metric in CURVE_METRICS
            }
            for rv_ratio in RV_RATIOS
        }
    return {
        "pairs": [
            {
                "contrast": contrast_key(a, b),
                "contrast_kind": contrast_kind(a, b),
                "label": contrast_label(a, b),
                "from_total": a,
                "to_total": b,
                "from_pct": pct_of(a),
                "to_pct": pct_of(b),
            }
            for a, b in CONTRAST_PAIRS
        ],
        "consecutive_steps": [contrast_key(a, b) for a, b in CONSECUTIVE_STEPS],
        "endpoint": contrast_key(*ENDPOINT_PAIR),
        "omitted_pairs": [contrast_key(a, b) for a, b in OMITTED_PAIRS],
        "selection_note": CONTRAST_SELECTION_NOTE,
        "by_pair": by_pair,
    }


def delta_rows(deltas: dict[str, Any]) -> list[tuple[dict[str, Any], float, str]]:
    """``[(pair_index_entry, rv_ratio, metric)]`` in report order — one per table row.

    One iteration order, shared by the CSV renderer, the markdown renderer and the
    tests, so a row can never appear in one artifact and not another.
    """
    return [
        (pair, rv_ratio, metric)
        for pair in deltas["pairs"]
        for rv_ratio in RV_RATIOS
        for metric in CURVE_METRICS
    ]


def contrast_of(
    deltas: dict[str, Any], contrast: str, rv_ratio: float, metric: str
) -> Optional[dict[str, Any]]:
    """The contrast payload for one ``(pair, ratio, metric)``, or ``None``."""
    return ((deltas["by_pair"].get(contrast) or {}).get(f"{rv_ratio:g}") or {}).get(
        metric
    )


# --------------------------------------------------------------------------- #
# Step separability + the shape verdict — the reading a four-point curve needs.
# --------------------------------------------------------------------------- #
def _step_seed_values(
    curve: dict[int, list[agg.SeedCell]],
    from_total: int,
    to_total: int,
    rv_ratio: float,
    metric: str,
) -> dict[int, float]:
    """``seed → (value at to_total − value at from_total)`` over paired seeds.

    The per-seed magnitude of one step. Comparing two steps means pairing THESE by
    seed, i.e. a second difference, which is why the map is built rather than the
    summarized step being differenced.
    """
    a = _seed_values(curve.get(from_total, []), rv_ratio, metric)
    b = _seed_values(curve.get(to_total, []), rv_ratio, metric)
    return {seed: b[seed] - a[seed] for seed in sorted(set(a) & set(b))}


def _stat_of(payload: dict[str, Any]) -> agg.Stat:
    """Rehydrate an ``agg.Stat`` from a ``Stat.as_dict()`` payload."""
    return agg.Stat(**agg._stat_from_dict(payload))


def _sign_test(diffs: Sequence[float]) -> dict[str, Any]:
    """The exact paired sign-flip permutation test, reused verbatim.

    Same function the plateau bundle calls, so the p and the ``min_p_two_sided`` floor
    mean exactly what they mean there.
    """
    return sign_flip_permutation_test(list(diffs))


def step_separability(curve: dict[int, list[agg.SeedCell]]) -> list[dict[str, Any]]:
    """Is each step a gain, and is any step bigger than another? Neither, separably.

    Two row kinds, both paired by seed and both carrying ``separable=False`` by
    construction (:data:`SEPARABILITY_NOTE`):

    * ``step_vs_zero`` — one consecutive step against no change. Reports the step's own
      paired CI, whether that CI excludes zero, sign consistency across the three seeds,
      and the exact sign test's p next to the p it can never go below.
    * ``step_vs_step`` — one step against another, differenced **within a seed** (a
      second difference), plus whether the two steps' own CIs overlap. This is the row a
      shape claim would need, and it is the row that shows the claim is unavailable.

    A row with no paired seed gets :data:`NOT_COMPUTABLE`, never "unstable": mid-flight
    that is a missing cell, and ``agg._contrast``'s ``sign_consistent=False`` on an empty
    difference set would otherwise print as a measured disagreement.

    Every :data:`CURVE_METRICS` entry gets both, for every ratio: a shape need not be
    shared across mixes or across the CER and accent channels, and collapsing them would
    hide exactly that.
    """
    rows: list[dict[str, Any]] = []
    for rv_ratio in RV_RATIOS:
        rv_pct = int(round(rv_ratio * 100))
        mix = f"{rv_pct}:{100 - rv_pct}"
        for metric in CURVE_METRICS:
            steps = {
                contrast_key(a, b): _step_seed_values(curve, a, b, rv_ratio, metric)
                for a, b in CONSECUTIVE_STEPS
            }
            for a, b in CONSECUTIVE_STEPS:
                key = contrast_key(a, b)
                contrast = one_contrast(curve, a, b, rv_ratio, metric)
                diff = contrast["diff"]
                seeds = list(contrast["seeds"])
                diffs = [contrast["per_seed_diff"][str(s)] for s in seeds]
                permutation = _sign_test(diffs)
                lo, hi = diff["ci95_lo"], diff["ci95_hi"]
                rows.append(
                    {
                        "comparison_kind": "step_vs_zero",
                        "rv_ratio": rv_ratio,
                        "mix": mix,
                        "metric": metric,
                        "step": key,
                        "step_label": contrast_label(a, b),
                        "other_step": None,
                        "other_step_label": None,
                        "n": diff["n"],
                        "seeds": seeds,
                        "diff_mean": diff["mean"],
                        "diff_ci95_lo": lo,
                        "diff_ci95_hi": hi,
                        "per_seed_diff": contrast["per_seed_diff"],
                        # With no paired seed there is no direction and no sign to be
                        # consistent about; agg._contrast's "mixed"/False on an empty
                        # difference set would read as a measured disagreement.
                        "direction": contrast["direction"] if seeds else None,
                        "sign_consistent": contrast["sign_consistent"]
                        if seeds
                        else None,
                        "ci95_excludes_zero": (
                            None if lo is None or hi is None else not (lo <= 0.0 <= hi)
                        ),
                        "ci95_overlap": None,
                        "exact_sign_test_p_two_sided": permutation["p_two_sided"],
                        "exact_sign_test_min_p_two_sided": permutation[
                            "min_p_two_sided"
                        ],
                        "separable": False,
                        "verdict": (
                            NOT_COMPUTABLE
                            if not seeds
                            else STEP_SIGN_CONSISTENT
                            if contrast["sign_consistent"]
                            else STEP_SIGN_UNSTABLE
                        ),
                    }
                )
            for i, (a, b) in enumerate(CONSECUTIVE_STEPS):
                for c, d in CONSECUTIVE_STEPS[i + 1 :]:
                    first, second = contrast_key(a, b), contrast_key(c, d)
                    # b − a on the two steps' per-seed magnitudes: a positive mean
                    # means the LATER step moved the metric more in the positive
                    # direction, which is a statement about the two magnitudes and not
                    # about either one's sign.
                    contrast = agg._contrast(steps[first], steps[second])
                    diff = contrast["diff"]
                    seeds = list(contrast["seeds"])
                    diffs = [contrast["per_seed_diff"][str(s)] for s in seeds]
                    permutation = _sign_test(diffs)
                    lo, hi = diff["ci95_lo"], diff["ci95_hi"]
                    rows.append(
                        {
                            "comparison_kind": "step_vs_step",
                            "rv_ratio": rv_ratio,
                            "mix": mix,
                            "metric": metric,
                            "step": first,
                            "step_label": contrast_label(a, b),
                            "other_step": second,
                            "other_step_label": contrast_label(c, d),
                            "n": diff["n"],
                            "seeds": seeds,
                            "diff_mean": diff["mean"],
                            "diff_ci95_lo": lo,
                            "diff_ci95_hi": hi,
                            "per_seed_diff": contrast["per_seed_diff"],
                            "direction": contrast["direction"] if seeds else None,
                            "sign_consistent": (
                                contrast["sign_consistent"] if seeds else None
                            ),
                            "ci95_excludes_zero": (
                                None
                                if lo is None or hi is None
                                else not (lo <= 0.0 <= hi)
                            ),
                            # The plateau bundle's own overlap predicate, on the two
                            # steps' marginal CIs (closed intervals: touching counts as
                            # overlapping, i.e. never called separated on a boundary).
                            "ci95_overlap": esp._cis_overlap(
                                agg.summarize(steps[first].values()),
                                agg.summarize(steps[second].values()),
                            ),
                            "exact_sign_test_p_two_sided": permutation["p_two_sided"],
                            "exact_sign_test_min_p_two_sided": permutation[
                                "min_p_two_sided"
                            ],
                            "separable": False,
                            "verdict": (
                                NOT_COMPUTABLE
                                if not seeds
                                else ORDER_CONSISTENT
                                if contrast["sign_consistent"]
                                else ORDER_UNSTABLE
                            ),
                        }
                    )
    return rows


def step_ranking(
    curve: dict[int, list[agg.SeedCell]], rv_ratio: float, metric: str
) -> list[dict[str, Any]]:
    """The three steps ranked by |mean paired difference|, largest first.

    An ordered DESCRIPTION of three seeds, emitted only next to
    :func:`step_separability`. Steps whose delta cannot be computed sort last and are
    flagged ``rankable=False`` rather than dropped, so a partial curve cannot silently
    produce a three-item ranking out of two steps.
    """
    rows: list[dict[str, Any]] = []
    for a, b in CONSECUTIVE_STEPS:
        contrast = one_contrast(curve, a, b, rv_ratio, metric)
        mean = contrast["diff"]["mean"]
        rows.append(
            {
                "step": contrast_key(a, b),
                "step_label": contrast_label(a, b),
                "n": contrast["diff"]["n"],
                "diff_mean": mean,
                "abs_diff_mean": None if mean is None else abs(mean),
                "diff_ci95_lo": contrast["diff"]["ci95_lo"],
                "diff_ci95_hi": contrast["diff"]["ci95_hi"],
                "sign_consistent": contrast["sign_consistent"],
                "rankable": mean is not None,
            }
        )
    ordered = sorted(
        rows,
        key=lambda r: (
            r["abs_diff_mean"] is None,
            -(r["abs_diff_mean"] or 0.0),
            r["step"],
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank if row["rankable"] else None
    return ordered


def _pair_labels(rows: Any) -> list[str]:
    """``["10→25% vs 25→50%", …]`` for a selection of ``step_vs_step`` rows."""
    return [f"{r['step_label']} vs {r['other_step_label']}" for r in rows]


def shape_verdict(
    curve: dict[int, list[agg.SeedCell]],
    separability: Sequence[dict[str, Any]],
    *,
    metric: str = SHAPE_PRIMARY_METRIC,
) -> dict[str, Any]:
    """What the curve's shape may be claimed to be, per mix ratio: nothing, at n=3.

    ``determined`` is ``False`` by construction — no pair of steps can be separable at
    n=3 — and the payload says which kind of non-determination each mix is in: all three
    seeds agreeing on the step order (an ordered description), the order flipping by
    seed, or the pair not being computable at all because a cell has not landed. Those
    three are counted separately: mid-flight, calling an absent cell "order-unstable"
    would report missing data as a measured disagreement. The ranking travels inside
    this block so it cannot be quoted without it.
    """
    by_ratio: dict[str, Any] = {}
    for rv_ratio in RV_RATIOS:
        rows = [
            r
            for r in separability
            if r["comparison_kind"] == "step_vs_step"
            and r["rv_ratio"] == rv_ratio
            and r["metric"] == metric
        ]

        consistent = _pair_labels(r for r in rows if r["verdict"] == ORDER_CONSISTENT)
        unstable = _pair_labels(r for r in rows if r["verdict"] == ORDER_UNSTABLE)
        not_computable = _pair_labels(r for r in rows if r["verdict"] == NOT_COMPUTABLE)
        overlapping = _pair_labels(r for r in rows if r["ci95_overlap"] is True)
        ranking = step_ranking(curve, rv_ratio, metric)
        ranked = [r["step_label"] for r in ranking if r["rankable"]]
        n_comparable = len(rows) - len(not_computable)
        rv_pct = int(round(rv_ratio * 100))
        by_ratio[f"{rv_ratio:g}"] = {
            "rv_ratio": rv_ratio,
            "mix": f"{rv_pct}:{100 - rv_pct}",
            "metric": metric,
            "ranking_largest_step_first": ranked,
            "ranking": ranking,
            "n_steps": len(CONSECUTIVE_STEPS),
            "n_steps_ranked": len(ranked),
            "n_step_pairs": len(rows),
            "n_step_pairs_comparable": n_comparable,
            "n_step_pairs_separable": 0,
            "step_pairs_consistently_ordered": consistent,
            "step_pairs_order_unstable": unstable,
            "step_pairs_not_computable": not_computable,
            "step_pairs_with_overlapping_cis": overlapping,
            "determined": False,
            "verdict": SHAPE_NOT_DETERMINED,
            "statement": (
                f"{rv_pct}:{100 - rv_pct} on {metric}: "
                + (
                    f"{len(ranked)}/{len(CONSECUTIVE_STEPS)} steps computable, by |Δ| "
                    "largest first " + " > ".join(ranked)
                    if ranked
                    else "no step computable"
                )
                + ". Shape NOT determined — "
                f"0/{n_comparable} comparable step pair(s) separable at n=3; "
                f"{len(consistent)}/{n_comparable} consistently ordered across the "
                f"three seeds, {len(unstable)}/{n_comparable} order-unstable, "
                f"{len(overlapping)}/{n_comparable} with overlapping step CIs"
                + (
                    ""
                    if not not_computable
                    else f"; {len(not_computable)}/{len(rows)} pair(s) not computable "
                    "yet (a cell has not landed), which is missing data and NOT "
                    "instability"
                )
                + "."
            ),
        }
    return {
        "primary_metric": metric,
        "determined": False,
        "verdict": SHAPE_NOT_DETERMINED,
        "by_ratio": by_ratio,
        "separability_note": SEPARABILITY_NOTE,
        "claim_note": SHAPE_CLAIM_NOTE,
        "statement": (
            "The shape of the curve is NOT determined at n=3: no pair of consecutive "
            "steps is separable, in any mix or any metric. "
            + " ".join(blk["statement"] for blk in by_ratio.values())
        ),
    }


def asymmetry_by_total(curve: dict[int, list[agg.SeedCell]]) -> dict[str, Any]:
    """``agg.asymmetry`` per total — the mix-endpoint contrast at every curve point.

    Applicable at every point because all of them train ``rv_ratio`` 1.0 and 0.0. Reused
    verbatim, so the zero-shot asymmetry read at any trained total is the same quantity,
    computed the same way, as the one the main-grid bundle reports at 10420.
    """
    return {str(total): agg.asymmetry(cells) for total, cells in sorted(curve.items())}


#: Why no retention/adaptation block exists here. Carried into the aggregate and the
#: manifest so its absence is a stated design fact, not an omission a reader has to
#: notice.
RETENTION_ADAPTATION_NOTE: str = (
    "Not emitted. agg.retention_adaptation contrasts a single-notation baseline "
    "against a 25% minority mix, i.e. rv_ratio 0.75 and 0.25. This study trains only "
    "{1.0, 0.5, 0.0}, so the contrast has NO counterpart at ANY trained total "
    f"({list(TRAINED_TOTALS)}) and cannot be a curve. Emitting it as an empty or "
    f"{FULL_TOTAL}-only table would read as 'measured, no effect'. The {FULL_TOTAL} "
    "retention/adaptation contrasts are in the v2 main-grid bundle, where all five "
    "ratios exist."
)

#: Why no minimum-target-notation-ratio table exists here.
MIN_TARGET_RATIO_NOTE: str = (
    "Not emitted. The v1 study adequacy rule scans the five mix ratios for the smallest "
    "target-notation training share whose mean target-view CER clears a pre-registered "
    "threshold. Over three ratios the scan has a different resolution, so running it "
    "here would silently redefine a pre-registered rule on a coarser grid rather than "
    "reproduce it. PENDING(v2-data-efficiency-study) keeps that table open."
)


# --------------------------------------------------------------------------- #
# Recipe / subset provenance, scoped to an N-total design.
# --------------------------------------------------------------------------- #
#: Recipe pins that must agree across the WHOLE join — the fields whose constancy is
#: what makes any two points of the curve comparable at all. ``total_train_samples_
#: requested`` and ``parallel_keys_used`` are deliberately NOT here: they are the axis.
JOIN_WIDE_FIELDS: tuple[str, ...] = (
    "mix_mode",
    "key_column",
    "selection_algorithm",
    "parallel_keys_total",
    "num_train_epochs",
    "model_base",
    "freeze_mode",
    "input_scheme",
)

#: Pins that must agree WITHIN a total but differ across totals by design.
PER_TOTAL_FIELDS: tuple[str, ...] = (
    "total_train_samples_requested",
    "parallel_keys_used",
)

#: Why the subset hash is scoped to ``(seed, total)`` here rather than to ``seed``.
KEY_LIST_SHA_SCOPE_NOTE: str = (
    "key_list_sha256 hashes the ORDERED selected key list. dataset.py sorts the shared "
    "keys, applies a seeded shuffle (parallel_subset_seed is left null, so it follows "
    "`seed`) and takes a PREFIX of length total_train_samples. So within one (seed, "
    f"total) all {len(RV_RATIOS)} ratios share one list — rv_ratio only reroutes an "
    "already-selected text to an RV or an MS image — while the same seed at a "
    "different total is a different ordered list with a different hash. The invariant "
    "is therefore WITHIN (seed, total), one step weaker than the main grid's "
    f"within-seed one, and a healthy join carries exactly {len(SEEDS) * len(TOTALS)} "
    f"distinct hashes ({len(SEEDS)} seeds x {len(TOTALS)} totals). Checking it "
    "join-wide, or even per seed, makes a healthy curve report 'this is not a single "
    "study'."
)

#: Why the four selections nest without pinning a subset seed — and why nesting is
#: nevertheless VERIFIED rather than argued (:func:`subset_nesting`).
NESTING_NOTE: str = (
    "The launcher leaves parallel_subset_seed null (unlike the v1 study study, which "
    "pinned 42) so every point's interval means 'all variation downstream of `seed`'. "
    "Nesting is not sacrificed by that: dataset.py sorts the shared keys, shuffles them "
    "with a seed that does NOT depend on total_train_samples, and takes a prefix, so "
    f"within one seed {' ⊂ '.join(str(t) for t in TOTALS)} holds by construction, and "
    f"at {FULL_TOTAL} the selected key SET is the whole shared-parallel corpus anyway. "
    "Because that is what makes a consecutive step mean 'the same lines plus these', it "
    "is CHECKED against each cell's committed parallel_keys.txt (subset_nesting) rather "
    "than trusted. What the unpinned seed buys is that a low-data interval also covers "
    "subset-sampling variation, which is the honest interval for 'only n lines were "
    "available'; the cost is that two points' error bars are not subset-matched ACROSS "
    "seeds, so a paired delta is a delta between two conditions, not between two runs "
    "on one subset."
)

#: The committed per-cell key list a nesting check reads. Written by the runner next to
#: ``results.json``; absent for a cell means "unchecked", never "not nested".
PARALLEL_KEYS_FILE: str = "parallel_keys.txt"


def subset_provenance_by_total(
    curve: dict[int, list[agg.SeedCell]],
    *,
    low_study_root: Path,
    full_study_root: Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    """``total → run_name → {v2mg.SUBSET_FIELDS}``, via ``v2mg.subset_provenance``."""
    return {
        total: v2mg.subset_provenance(
            study_root_for(
                total, low_study_root=low_study_root, full_study_root=full_study_root
            ),
            cells,
        )
        for total, cells in curve.items()
    }


def _read_key_list(path: Path) -> Optional[list[str]]:
    """The committed ordered key list of one cell, or ``None`` if unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return [line for line in text.splitlines() if line]


def subset_nesting(
    curve: dict[int, list[agg.SeedCell]],
    *,
    low_study_root: Path,
    full_study_root: Path,
) -> dict[str, Any]:
    """Within each seed, is every total's key list a PREFIX of the next total's?

    This is the invariant that makes a consecutive-step delta readable as "the same
    lines plus these" instead of "a different draw of more lines" — the single most
    load-bearing property of a multi-point curve whose subset seed is not pinned, and
    the one thing ``key_list_sha256`` cannot show (two prefixes of one seed hash
    differently whether or not one contains the other).

    One representative cell per ``(total, seed)`` is enough: the selection is
    ratio-invariant and ``key_list_sha256`` is already checked constant within each
    ``(total, seed)``, so any present ratio's list is that pair's list.

    A pair with a missing or unreadable ``parallel_keys.txt`` is ``unchecked`` — never
    "nested" and never "violated". Only an actual prefix violation is a defect.
    """
    lists: dict[tuple[int, int], tuple[str, list[str]]] = {}
    for total, cells in curve.items():
        root = study_root_for(
            total, low_study_root=low_study_root, full_study_root=full_study_root
        )
        for cell in cells:
            if not cell.present or (total, cell.seed) in lists:
                continue
            keys = _read_key_list(root / cell.run_name / PARALLEL_KEYS_FILE)
            if keys:
                lists[(total, cell.seed)] = (cell.run_name, keys)

    pairs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for inner, outer in zip(TOTALS, TOTALS[1:]):
            row: dict[str, Any] = {
                "seed": seed,
                "inner_total": inner,
                "outer_total": outer,
                "inner_run": None,
                "outer_run": None,
                "inner_n_keys": None,
                "outer_n_keys": None,
                "checked": False,
                "nested": None,
                "reason": None,
            }
            inner_blk, outer_blk = lists.get((inner, seed)), lists.get((outer, seed))
            if inner_blk is None or outer_blk is None:
                row["reason"] = f"no readable {PARALLEL_KEYS_FILE} at " + ", ".join(
                    str(t)
                    for t, blk in ((inner, inner_blk), (outer, outer_blk))
                    if blk is None
                )
                pairs.append(row)
                continue
            inner_run, inner_keys = inner_blk
            outer_run, outer_keys = outer_blk
            row.update(
                {
                    "inner_run": inner_run,
                    "outer_run": outer_run,
                    "inner_n_keys": len(inner_keys),
                    "outer_n_keys": len(outer_keys),
                    "checked": True,
                    "nested": outer_keys[: len(inner_keys)] == inner_keys,
                }
            )
            if not row["nested"]:
                row["reason"] = (
                    "the smaller total's ordered key list is not a prefix of the "
                    "larger's, so this step adds a DIFFERENT draw rather than more "
                    "lines"
                )
                row["inner_is_subset_of_outer"] = set(inner_keys) <= set(outer_keys)
            pairs.append(row)

    checked = [r for r in pairs if r["checked"]]
    violations = [r for r in checked if r["nested"] is False]
    return {
        "scope": (
            "within each seed, every consecutive pair of totals on the curve's x axis"
        ),
        "n_pairs": len(pairs),
        "n_checked": len(checked),
        "n_unchecked": len(pairs) - len(checked),
        "n_violations": len(violations),
        "violations": [
            f"seed {r['seed']}: {r['inner_total']} ⊄ {r['outer_total']} "
            f"({r['inner_run']} vs {r['outer_run']})"
            for r in violations
        ],
        "pairs": pairs,
        "nested": not violations,
        "note": NESTING_NOTE,
    }


def subset_consistency(
    prov: dict[int, dict[str, dict[str, Any]]],
    curve: dict[int, list[agg.SeedCell]],
    *,
    nesting: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Is the join one study apart from the axis, and did each cell train its mix?

    Five checks, at four different scopes — the scoping *is* the content here, because
    the main grid's grid-wide check would flag this healthy design as broken:

    * **join-wide** (:data:`JOIN_WIDE_FIELDS`) — one distinct value each. These are
      what make a 1042 cell and a 10420 cell comparable; more than one value means the
      points are not the same experiment.
    * **per total** (:data:`PER_TOTAL_FIELDS`) — one distinct value per total, and
      ``total_train_samples_requested`` must equal the total itself. This is where the
      axis is verified rather than assumed: a 2605 cell whose ``results.json`` requests
      10420 is a mislabelled cell, and no join-wide check can see it.
    * **per (seed, total) subset hash** — see :data:`KEY_LIST_SHA_SCOPE_NOTE`.
    * **within-seed nesting** — :func:`subset_nesting`, passed in so the caller decides
      whether the key lists are read at all; ``None`` records it as not checked, which
      is *not* the same as passing.
    * **mix realization** — ``rv/ms_train_samples`` equals the floor split
      (:func:`v2mg.expected_mix_counts`) of that cell's OWN total at its ratio.
    """
    flat: dict[str, dict[str, Any]] = {}
    total_by_run: dict[str, int] = {}
    for total, records in prov.items():
        for name, rec in records.items():
            flat[name] = rec
            total_by_run[name] = total

    def _distinct(names: Sequence[str], field: str) -> list[Any]:
        values = sorted(
            {
                json.dumps(flat[n].get(field), sort_keys=True)
                for n in names
                if flat[n].get(field) is not None
            }
        )
        return [json.loads(v) for v in values]

    join_wide = {field: _distinct(sorted(flat), field) for field in JOIN_WIDE_FIELDS}
    join_inconsistent = sorted(f for f, vals in join_wide.items() if len(vals) > 1)

    per_total: dict[str, Any] = {}
    total_mislabelled: list[dict[str, Any]] = []
    for total in sorted(prov):
        names = sorted(prov[total])
        blk = {field: _distinct(names, field) for field in PER_TOTAL_FIELDS}
        split = sorted(f for f, vals in blk.items() if len(vals) > 1)
        requested = blk.get("total_train_samples_requested") or []
        matches_axis = requested == [total]
        per_total[str(total)] = {
            "total_train_samples": total,
            "n_cells_checked": len(names),
            "held_fixed": blk,
            "inconsistent_fields": split,
            "consistent": not split,
            "total_train_samples_requested_matches_axis": matches_axis,
        }
        if names and not matches_axis:
            total_mislabelled.append(
                {
                    "total_train_samples": total,
                    "requested_values": requested,
                    "runs": names,
                }
            )

    by_seed_total: dict[str, dict[str, Any]] = {}
    for name, rec in sorted(flat.items()):
        sha = rec.get("key_list_sha256")
        total = total_by_run.get(name)
        seed = next(
            (c.seed for c in all_cells(curve) if c.run_name == name and c.present),
            None,
        )
        if sha is None or total is None or seed is None:
            continue
        blk = by_seed_total.setdefault(
            f"{total}:{seed}",
            {
                "total_train_samples": total,
                "seed": seed,
                "runs": [],
                "key_list_sha256": [],
            },
        )
        blk["runs"].append(name)
        if sha not in blk["key_list_sha256"]:
            blk["key_list_sha256"].append(sha)
    split_groups = sorted(
        key for key, blk in by_seed_total.items() if len(blk["key_list_sha256"]) > 1
    )

    ratio_by_run = {c.run_name: c.rv_ratio for c in all_cells(curve)}
    mix_rows: list[dict[str, Any]] = []
    for name, rec in sorted(flat.items()):
        total = rec.get("total_train_samples_requested")
        rv_ratio = ratio_by_run.get(name)
        if total is None or rv_ratio is None:
            continue
        exp_rv, exp_ms = v2mg.expected_mix_counts(int(total), rv_ratio)
        ok = (
            rec.get("rv_train_samples") == exp_rv
            and rec.get("ms_train_samples") == exp_ms
        )
        mix_rows.append(
            {
                "run_name": name,
                "total_train_samples_requested": total,
                "rv_ratio": rv_ratio,
                "expected_rv_train_samples": exp_rv,
                "expected_ms_train_samples": exp_ms,
                "rv_train_samples": rec.get("rv_train_samples"),
                "ms_train_samples": rec.get("ms_train_samples"),
                "matches_ratio": bool(ok),
            }
        )

    return {
        "n_cells_checked": len(flat),
        "join_wide": {
            "fields": list(JOIN_WIDE_FIELDS),
            "held_fixed": join_wide,
            "inconsistent_fields": join_inconsistent,
            "consistent": not join_inconsistent,
            "scope": (
                "every present cell of EVERY total; total_train_samples_requested and "
                "parallel_keys_used are excluded on purpose — they are the curve's axis"
            ),
        },
        "per_total": per_total,
        "total_axis_mismatches": total_mislabelled,
        "key_list_sha256_per_seed_total": {
            key: by_seed_total[key] for key in sorted(by_seed_total)
        },
        "key_list_sha256_split_groups": split_groups,
        "key_list_sha256_distinct_join_wide": sorted(
            {sha for blk in by_seed_total.values() for sha in blk["key_list_sha256"]}
        ),
        "key_list_sha256_expected_distinct": len(SEEDS) * len(TOTALS),
        "key_list_sha256_consistent": not split_groups,
        "key_list_sha256_scope_note": KEY_LIST_SHA_SCOPE_NOTE,
        "key_list_sha256_note": v2mg.KEY_LIST_SHA_NOTE,
        "nesting": nesting,
        "nesting_note": NESTING_NOTE,
        "mix_realization": mix_rows,
        "mix_mismatches": [r["run_name"] for r in mix_rows if not r["matches_ratio"]],
        "consistent": (
            not join_inconsistent
            and not total_mislabelled
            and not split_groups
            and all(blk["consistent"] for blk in per_total.values())
            # A nesting VIOLATION is inconsistency; not having looked is not.
            and not (nesting or {}).get("violations")
        ),
    }


# --------------------------------------------------------------------------- #
# Corpus attestation — per arm, from that arm's OWN logs and study root.
# --------------------------------------------------------------------------- #
def corpus_attestation(
    curve: dict[int, list[agg.SeedCell]],
    *,
    log_dir: Optional[Path],
    low_study_root: Path,
    full_study_root: Path,
) -> dict[str, Any]:
    """``v2mg.corpus_attestation_summary`` per TOTAL, plus a join-wide verdict.

    Run once per point on purpose. The two arms are written by two launchers into two
    study roots and logged under two glob patterns, so each is attested from its own log
    set and validated against its own root: a main-grid log cannot attest a trained
    cell, and ``study_root_mismatches`` still catches a ``…_smoke`` log borrowing either
    arm's run names. The three trained totals share one launcher, hence one glob, and are
    still summarized separately so each point reports its own attested count and its own
    ``healthy`` — a total whose logs are missing cannot hide behind its siblings. Each
    block's ``per_run`` is scoped to that total's own cells for the same reason.

    ``healthy`` is the conjunction over totals — the one flag a publication artifact may
    read. A total with no present cell is *not* healthy (``v2mg`` requires
    ``bool(present)``), so an empty point cannot green the gate by vacuity.
    """
    per_total: dict[str, Any] = {}
    for total, cells in sorted(curve.items()):
        root = study_root_for(
            total, low_study_root=low_study_root, full_study_root=full_study_root
        )
        attested = v2mg.attest_corpus_from_slurm_logs(
            log_dir, glob=SLURM_LOG_GLOB_BY_TOTAL.get(total, v2mg.SLURM_LOG_GLOB)
        )
        summary = v2mg.corpus_attestation_summary(attested, cells, study_root=root)
        target_names = {c.run_name for c in cells}
        summary["per_run"] = {
            name: rec
            for name, rec in (summary.get("per_run") or {}).items()
            if name in target_names
        }
        summary["total_train_samples"] = total
        summary["slurm_log_glob"] = SLURM_LOG_GLOB_BY_TOTAL.get(total)
        summary["launcher"] = INHERITED_LAUNCHER if total == FULL_TOTAL else LAUNCHER
        per_total[str(total)] = summary

    corpora = sorted(
        {c for blk in per_total.values() for c in blk.get("corpora") or []}
    )
    data_dirs = sorted(
        {
            tuple(pair)
            for blk in per_total.values()
            for pair in blk.get("data_dirs") or []
        }
    )
    return {
        "source": "immutable Slurm .out headers echoed by the two launchers",
        "per_total": per_total,
        "corpora": corpora,
        "data_dirs": [list(pair) for pair in data_dirs],
        "single_corpus": len(corpora) == 1 and len(data_dirs) <= 1,
        "expected_corpus": {
            "corpus": v2mg.EXPECTED_CORPUS,
            "rv_data_dir": v2mg.EXPECTED_RV_DATA_DIR,
            "ms_data_dir": v2mg.EXPECTED_MS_DATA_DIR,
            "data_source": v2mg.EXPECTED_DATA_SOURCE,
        },
        "n_present_cells": sum(
            int(blk.get("n_present_cells") or 0) for blk in per_total.values()
        ),
        "n_attested": sum(
            int(blk.get("n_attested") or 0) for blk in per_total.values()
        ),
        "n_unattested": sum(
            int(blk.get("n_unattested") or 0) for blk in per_total.values()
        ),
        "healthy": bool(per_total)
        and all(bool(blk.get("healthy")) for blk in per_total.values())
        and len(corpora) == 1
        and len(data_dirs) <= 1,
        "note": (
            "Each arm is attested from its OWN Slurm log glob and validated against its "
            "OWN study root, so an attestation can never cross the join. Absence of a "
            "log means UNATTESTED, not v1: results.json records no corpus path, and "
            "key_list_sha256 cannot discriminate the corpora (identical ISO text). The "
            "corpus= token is a LABEL derived by the launcher from the resolved data "
            "dirs; the dirs and data_source are the evidence, and single_corpus catches "
            "only mixtures, never a uniformly wrong corpus."
        ),
    }


# --------------------------------------------------------------------------- #
# Per-cell invariants + the inherited-arm cross-check.
# --------------------------------------------------------------------------- #
def grid_integrity(
    curve: dict[int, list[agg.SeedCell]],
    *,
    low_study_root: Optional[Path] = None,
    full_study_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Does each cell's ``results.json`` agree with the dir and point it was read as?

    * **seed ↔ dir** — ``read_cell_dir`` takes the seed from ``results.json`` while the
      dir name encodes the requested seed (``v2mg._SEED_SUFFIX_RE`` matches both
      namings), so these are two independent records that must agree.
    * **distinct seeds per point** — a point's mean/CI is an n-seed number only if its
      present cells carry n *different* seeds.
    * **unreadable ``results.json``** — on disk but not parseable (a torn mid-flight
      write). Counted as absent rather than crashing, but publication-blocking.
    """
    cells = all_cells(curve)
    seed_mismatches = [
        {
            "run_name": c.run_name,
            "dir_seed": int(match.group(1)),
            "results_json_seed": c.seed,
        }
        for c in cells
        if c.present
        for match in [v2mg._SEED_SUFFIX_RE.search(c.run_name)]
        if match is not None and int(match.group(1)) != c.seed
    ]
    point_duplicates = []
    for total in TOTALS:
        for rv_ratio in RV_RATIOS:
            seeds = [
                c.seed
                for c in curve.get(total, [])
                if c.present and c.rv_ratio == rv_ratio
            ]
            if len(seeds) != len(set(seeds)):
                point_duplicates.append(
                    {
                        "total_train_samples": total,
                        "rv_ratio": rv_ratio,
                        "n_present": len(seeds),
                        "seeds": sorted(seeds),
                        "n_distinct_seeds": len(set(seeds)),
                    }
                )
    unreadable: list[str] = []
    checked = low_study_root is not None and full_study_root is not None
    if checked:
        assert low_study_root is not None and full_study_root is not None
        for total, group in curve.items():
            root = study_root_for(
                total, low_study_root=low_study_root, full_study_root=full_study_root
            )
            unreadable.extend(
                c.run_name
                for c in group
                if not c.present
                and (root / c.run_name / "results.json").is_file()
                and not v2mg._results_json_parses(root / c.run_name)
            )
    return {
        "seed_mismatches": seed_mismatches,
        "point_seed_duplicates": point_duplicates,
        "unreadable_results_json": sorted(unreadable),
        "unreadable_check_performed": checked,
        "ok": not seed_mismatches and not point_duplicates and not unreadable,
        "note": (
            "The dir name and results.json are two independent records of a cell's "
            "seed and must agree. A point whose present cells do not carry distinct "
            "seeds is not an n-seed point no matter what n the table prints."
        ),
    }


def maingrid_consistency(
    per_total: dict[int, dict[float, dict[str, agg.Stat]]],
    maingrid_aggregate: Optional[dict[str, Any]],
    *,
    tolerance: float = MAINGRID_MEAN_TOLERANCE,
) -> dict[str, Any]:
    """Are this join's inherited means the ones the committed main-grid bundle published?

    The 100% end of the curve is a claim about *another study's* result, so it is
    verified rather than asserted: for the three shared ratios and every curve metric,
    the ``Stat.mean`` and ``Stat.n`` this join computed from the main-grid run dirs must
    match the committed ``paclic-v2-maingrid/aggregate.json`` ``per_ratio`` block.

    Without this, pointing ``--full-study-root`` at a stale, moved or partially re-run
    main-grid root would produce a curve whose 100% point silently disagrees with the
    published figure of the same number — and nothing else in the bundle could tell.
    """
    if maingrid_aggregate is None:
        return {
            "checked": False,
            "reason": (
                "no main-grid aggregate.json was read, so the inherited arm is "
                "unverified: it is whatever --full-study-root contained"
            ),
            "mismatches": [],
            "ok": False,
        }
    published = maingrid_aggregate.get("per_ratio") or {}
    inherited = per_total.get(FULL_TOTAL) or {}
    mismatches: list[dict[str, Any]] = []
    n_compared = 0
    for rv_ratio in RV_RATIOS:
        block = published.get(f"{rv_ratio:g}")
        if block is None:
            mismatches.append(
                {
                    "rv_ratio": rv_ratio,
                    "metric": None,
                    "problem": "ratio absent from the committed main-grid aggregate",
                }
            )
            continue
        for metric in CURVE_METRICS:
            stat = (inherited.get(rv_ratio) or {}).get(metric)
            ref = block.get(metric)
            if stat is None or ref is None:
                mismatches.append(
                    {
                        "rv_ratio": rv_ratio,
                        "metric": metric,
                        "problem": "metric missing on one side",
                        "joined": None if stat is None else stat.mean,
                        "published": None if ref is None else ref.get("mean"),
                    }
                )
                continue
            n_compared += 1
            ref_mean = ref.get("mean")
            if (
                stat.n != ref.get("n")
                or ((stat.mean is None) != (ref_mean is None))
                or (
                    stat.mean is not None
                    and ref_mean is not None
                    and abs(float(stat.mean) - float(ref_mean)) > tolerance
                )
            ):
                mismatches.append(
                    {
                        "rv_ratio": rv_ratio,
                        "metric": metric,
                        "problem": "mean or n differs from the published main grid",
                        "joined_mean": stat.mean,
                        "published_mean": ref_mean,
                        "joined_n": stat.n,
                        "published_n": ref.get("n"),
                    }
                )
    return {
        "checked": True,
        "maingrid_study": maingrid_aggregate.get("study"),
        "maingrid_status": maingrid_aggregate.get("status"),
        "maingrid_complete": maingrid_aggregate.get("complete"),
        "shared_rv_ratios": list(RV_RATIOS),
        "metrics_compared": list(CURVE_METRICS),
        "n_comparisons": n_compared,
        "tolerance": tolerance,
        "mismatches": mismatches,
        "ok": not mismatches and n_compared > 0,
        "note": (
            "The inherited 100% point is another study's published result, so it is "
            "checked against that study's committed aggregate.json rather than trusted "
            "because a path was passed. Both sides are agg.summarize over the same "
            "results.json floats, so the tolerance covers JSON round-tripping only."
        ),
    }


def load_maingrid_aggregate(path: Optional[Path]) -> Optional[dict[str, Any]]:
    """Read the committed main-grid ``aggregate.json`` (``None`` if absent/unreadable)."""
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Warnings.
# --------------------------------------------------------------------------- #
#: Substrings marking a warning as **publication-blocking**: these fail the CLI
#: (exit 1) *and* refuse the figures. Everything else is reported and carried into the
#: artifacts but does not block, so a merely ambiguous condition cannot make the gate
#: un-greenable.
HARD_WARNING_MARKERS: tuple[str, ...] = (
    "not held fixed",
    "freeze_mode",
    "not a single study",
    "not constant within",
    "do not match their mix ratio",
    "does not equal the total",
    "different study root",
    "does not establish the expected v2 corpus",
    "no Slurm-log corpus attestation",
    "more than one corpus",
    "disagrees with its run dir",
    "do not carry distinct seeds",
    "results.json that did not parse",
    "disagrees with the committed v2 main-grid bundle",
    "inherited 100% point is unverified",
    "are not nested",
)


def hard_warnings(warnings: Sequence[str]) -> list[str]:
    """The publication-blocking subset of :func:`audit_warnings`.

    One predicate shared by the CLI exit status and the figure gate, so "the figures
    rendered" and "the run exited 0" can never disagree.
    """
    return [
        w
        for w in warnings
        if w.startswith("PARTIAL") or any(m in w for m in HARD_WARNING_MARKERS)
    ]


def audit_warnings(
    curve: dict[int, list[agg.SeedCell]],
    coverage: dict[str, Any],
    *,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
    maingrid: Optional[dict[str, Any]] = None,
) -> list[str]:
    """The shared ``agg`` checks plus this study's join / recipe / corpus ones."""
    cells = all_cells(curve)
    warnings: list[str] = []
    banner = partial_banner(coverage)
    if banner is not None:
        warnings.append(banner)
    warnings.extend(agg.audit_warnings(cells))

    # A point can be complete while a DELTA that reads it is not: that is the failure
    # mode a paired multi-point design has and a single-total grid does not. Reported per
    # (contrast × ratio), because one absent cell drops the n of the two steps and the
    # endpoint that touch it while leaving every other delta at n=3.
    unpaired = [
        blk
        for blk in coverage["paired"].values()
        if blk["n_paired"] < blk["n_expected"]
    ]
    if unpaired:
        warnings.append(
            f"published delta(s) resting on fewer than {len(SEEDS)} paired seeds, so "
            "the delta's n is NOT the per-point n: "
            + "; ".join(
                f"{blk['label']} {int(round(blk['rv_ratio'] * 100))}:"
                f"{int(round(blk['ms_ratio'] * 100))} paired n={blk['n_paired']} "
                f"(seeds {blk['seeds_paired']}; by total {blk['seeds_by_total']})"
                for blk in unpaired
            )
        )

    if integrity is not None:
        if integrity["seed_mismatches"]:
            warnings.append(
                "cell(s) whose results.json seed disagrees with its run dir: "
                + ", ".join(
                    f"{m['run_name']} (dir seed={m['dir_seed']}, "
                    f"results.json seed={m['results_json_seed']})"
                    for m in integrity["seed_mismatches"]
                )
            )
        if integrity["point_seed_duplicates"]:
            warnings.append(
                "curve point(s) whose present cells do not carry distinct seeds, so "
                "their n overstates the number of seeds behind the mean/CI: "
                + "; ".join(
                    f"n{d['total_train_samples']} "
                    f"{int(round(d['rv_ratio'] * 100))}:"
                    f"{int(round((1 - d['rv_ratio']) * 100))} n={d['n_present']} over "
                    f"{d['n_distinct_seeds']} distinct seed(s) {d['seeds']}"
                    for d in integrity["point_seed_duplicates"]
                )
            )
        if integrity["unreadable_results_json"]:
            warnings.append(
                "cell(s) with a results.json that did not parse (torn/partial write) "
                "and are therefore counted as ABSENT: "
                + ", ".join(integrity["unreadable_results_json"])
            )

    if subset is not None:
        if subset["join_wide"]["inconsistent_fields"]:
            warnings.append(
                "present cells disagree on join-wide recipe field(s) "
                f"{subset['join_wide']['inconsistent_fields']} — the points of the "
                "curve are not held fixed, so this is not a single study"
            )
        for blk in subset["per_total"].values():
            if blk["inconsistent_fields"]:
                warnings.append(
                    f"cells at total={blk['total_train_samples']} disagree on "
                    f"{blk['inconsistent_fields']} — that field is not held fixed "
                    "within a curve point"
                )
        for bad in subset["total_axis_mismatches"]:
            warnings.append(
                "cell(s) whose total_train_samples_requested "
                f"{bad['requested_values']} does not equal the total they were read "
                f"as ({bad['total_train_samples']}): {', '.join(bad['runs'])}"
            )
        if subset["key_list_sha256_split_groups"]:
            warnings.append(
                "key_list_sha256 is not constant within (total, seed) group(s) "
                f"{subset['key_list_sha256_split_groups']} — the ratio arms of one "
                "(total, seed) did not train on the same text subset (the hash is "
                "EXPECTED to differ across seeds and across totals, never within a "
                "(total, seed) pair)"
            )
        if subset["mix_mismatches"]:
            warnings.append(
                "cell(s) whose rv/ms train counts do not match their mix ratio at "
                "their own total: " + ", ".join(subset["mix_mismatches"])
            )
        nesting = subset.get("nesting")
        if nesting is None:
            # Not blocking: the check needs the run dirs, and a caller may legitimately
            # aggregate without them. But it is the property the STEPS rest on, so a
            # bundle that skipped it must not read as one that passed it.
            warnings.append(
                "within-seed subset nesting was NOT checked (no parallel_keys.txt read), "
                "so 'each step adds lines to the previous subset' is unverified in this "
                "run — a step delta is only 'the same lines plus these' if it holds"
            )
        else:
            if nesting["violations"]:
                warnings.append(
                    "consecutive totals of one seed are not nested — the smaller "
                    "total's ordered key list is not a prefix of the larger's, so that "
                    "step adds a DIFFERENT draw rather than more lines: "
                    + "; ".join(nesting["violations"])
                )
            if nesting["n_unchecked"]:
                warnings.append(
                    f"{nesting['n_unchecked']}/{nesting['n_pairs']} (seed, step) "
                    "nesting pair(s) could not be checked because a cell's "
                    f"{PARALLEL_KEYS_FILE} is absent or unreadable — recorded as "
                    "unchecked, never as nested"
                )

    if attestation is not None:
        for key, blk in sorted(attestation["per_total"].items()):
            if blk["n_unattested"]:
                warnings.append(
                    f"n{key}: {blk['n_unattested']}/{blk['n_present_cells']} present "
                    "cell(s) have no Slurm-log corpus attestation (corpus unattested, "
                    "NOT assumed): " + ", ".join(blk["unattested_runs"])
                )
            if blk["n_attested"] and not blk["single_corpus"]:
                warnings.append(
                    f"n{key}: attested cells name more than one corpus / data dir "
                    f"pair: corpora={blk['corpora']} dirs={blk['data_dirs']}"
                )
            if blk.get("corpus_mismatches"):
                warnings.append(
                    f"n{key}: attested log(s) whose echoed corpus evidence does not "
                    f"establish the expected v2 corpus (rv={v2mg.EXPECTED_RV_DATA_DIR} "
                    f"ms={v2mg.EXPECTED_MS_DATA_DIR} "
                    f"data_source={v2mg.EXPECTED_DATA_SOURCE}) — the corpus= label "
                    "alone is not evidence: "
                    + "; ".join(
                        f"{m['run_name']}←{m['log']} ({'; '.join(m['defects'])})"
                        for m in blk["corpus_mismatches"]
                    )
                )
            if blk.get("study_root_mismatches"):
                warnings.append(
                    f"n{key}: attested log(s) belong to a different study root than "
                    "the arm being aggregated (run names are shared with the _smoke "
                    "roots): "
                    + ", ".join(
                        f"{m['run_name']}←{m['log']} (output_dir={m['log_output_dir']})"
                        for m in blk["study_root_mismatches"]
                    )
                )
            if blk.get("superseded"):
                # Not publication-blocking: the winner is the newest by (jobid, task)
                # and is validated like any other. But it IS ambiguous, so not silent.
                warnings.append(
                    f"n{key}: cell(s) matched by several Slurm logs; the highest "
                    "(jobid, task) wins and the rest are recorded as superseded_logs "
                    "— check the winner is the intended run: "
                    + "; ".join(
                        f"{s['run_name']}←{s['log']} (superseded: "
                        f"{', '.join(s['superseded_logs'])})"
                        for s in blk["superseded"]
                    )
                )
        if not attestation["single_corpus"]:
            warnings.append(
                "the two arms of the curve do not attest ONE corpus: "
                f"corpora={attestation['corpora']} dirs={attestation['data_dirs']} — "
                "a 10%-vs-100% delta across two corpora is not a data-efficiency "
                "measurement"
            )

    if maingrid is not None:
        if not maingrid["checked"]:
            warnings.append(
                f"the inherited 100% point is unverified: {maingrid['reason']}"
            )
        elif maingrid["mismatches"]:
            warnings.append(
                "the inherited 100% point disagrees with the committed v2 main-grid "
                "bundle it is supposed to BE: "
                + "; ".join(
                    f"{int(round(m['rv_ratio'] * 100))}:"
                    f"{int(round((1 - m['rv_ratio']) * 100))} "
                    f"{m.get('metric')} {m['problem']}"
                    for m in maingrid["mismatches"][:8]
                )
                + (
                    f" (+{len(maingrid['mismatches']) - 8} more)"
                    if len(maingrid["mismatches"]) > 8
                    else ""
                )
            )
    return warnings


# --------------------------------------------------------------------------- #
# Aggregate payload.
# --------------------------------------------------------------------------- #
def build_aggregate(
    curve: dict[int, list[agg.SeedCell]],
    per_total: dict[int, dict[float, dict[str, agg.Stat]]],
    checks: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    *,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
    maingrid: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Machine-readable payload (→ ``aggregate.json``).

    ``curves`` is shaped exactly as ``build_paclic_paper_figures.fig_data_efficiency``
    reads it (``curves[<ratio>][<metric>] = [{total_train_samples, mean,
    ci95_halfwidth, …}]``), so that figure renders from this payload with no change to
    the figure's data access — it already plots four totals for the v1 study.

    ``step_separability`` and ``shape`` always travel together: the ranking inside the
    shape block is meaningless without the separability rows, so neither is optional.
    """
    warnings = audit_warnings(
        curve,
        coverage,
        subset=subset,
        attestation=attestation,
        integrity=integrity,
        maingrid=maingrid,
    )
    separability = step_separability(curve)
    curves = {
        f"{rv_ratio:g}": {
            metric: [
                {
                    "total_train_samples": total,
                    "pct": pct_of(total),
                    "source": CELL_SOURCE_BY_TOTAL.get(total),
                    **stat.as_dict(),
                }
                for total, stat in curve_series(per_total, rv_ratio, metric)
            ]
            for metric in CURVE_METRICS
        }
        for rv_ratio in RV_RATIOS
    }
    return {
        "study": STUDY,
        "issue": 126,
        "parent_issue": 122,
        "corpus": "v2",
        "analysis_only": True,
        "no_training_no_inference": True,
        "status": status_label(coverage),
        "complete": coverage["complete"],
        "partial_banner": partial_banner(coverage),
        "axis": {
            "totals": list(TOTALS),
            "pct_by_total": {str(t): pct_of(t) for t in TOTALS},
            "rv_ratios": list(RV_RATIOS),
            "seeds": list(SEEDS),
            "seeds_present_by_total": {
                str(total): sorted({c.seed for c in curve.get(total, []) if c.present})
                for total in TOTALS
            },
            "trained_totals": list(TRAINED_TOTALS),
            "inherited_total": FULL_TOTAL,
            "n_points": len(TOTALS),
            "consecutive_steps": [contrast_key(a, b) for a, b in CONSECUTIVE_STEPS],
            "endpoint": contrast_key(*ENDPOINT_PAIR),
        },
        "n_cells": coverage["n_cells"],
        "n_present": coverage["n_present"],
        "coverage": coverage,
        "curves": curves,
        "per_total": {
            str(total): {
                f"{rv:g}": {metric: stat.as_dict() for metric, stat in metrics.items()}
                for rv, metrics in by_ratio.items()
            }
            for total, by_ratio in per_total.items()
        },
        "curve_deltas": curve_deltas(curve),
        "step_separability": separability,
        "shape": shape_verdict(curve, separability),
        "asymmetry_by_total": asymmetry_by_total(curve),
        "retention_adaptation": None,
        "retention_adaptation_note": RETENTION_ADAPTATION_NOTE,
        "minimum_target_ratio": None,
        "minimum_target_ratio_note": MIN_TARGET_RATIO_NOTE,
        "cross_check": agg.cross_check_summary(checks),
        "subset_consistency": subset,
        "corpus_attestation": attestation,
        "grid_integrity": integrity,
        "maingrid_consistency": maingrid,
        "warnings": warnings,
        "hard_warnings": hard_warnings(warnings),
        "publishable": not hard_warnings(warnings)
        and bool((attestation or {}).get("healthy"))
        and bool((maingrid or {}).get("ok")),
    }


# --------------------------------------------------------------------------- #
# Rendering — CSV.
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float], digits: int = 12) -> str:
    """Full-precision CSV cell (empty for ``None``)."""
    return "" if value is None else f"{value:.{digits}f}"


def _write_rows(fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


#: The joined curve table. ``agg.PER_RATIO_COLUMNS`` plus the three columns the join
#: needs — which point, what fraction of the corpus, and which study trained it. There
#: is no ``agg`` renderer for this shape because ``agg`` has no total axis.
CURVE_COLUMNS: tuple[str, ...] = (
    "total_train_samples",
    "pct",
    "source_study",
    "provenance",
    *agg.PER_RATIO_COLUMNS,
)


def render_curve_csv(
    per_total: dict[int, dict[float, dict[str, agg.Stat]]],
) -> str:
    """One row per ``(ratio, metric, total)`` — the headline table.

    Every numeric column is read straight off the ``agg.Stat`` the reused
    ``agg.aggregate_by_ratio`` produced, in the same column names and order
    ``agg.render_per_ratio_csv`` uses, so a reader can diff this against the main
    grid's ``per_ratio.csv`` field by field.
    """
    rows: list[dict[str, Any]] = []
    for rv_ratio in RV_RATIOS:
        for metric in CURVE_METRICS:
            for total, stat in curve_series(per_total, rv_ratio, metric):
                rows.append(
                    {
                        "total_train_samples": total,
                        "pct": pct_of(total),
                        "source_study": CELL_SOURCE_BY_TOTAL.get(total),
                        "provenance": PROVENANCE_BY_TOTAL.get(total),
                        "rv_ratio": rv_ratio,
                        "ms_ratio": round(1.0 - rv_ratio, 2),
                        "metric": metric,
                        "n": stat.n,
                        "mean": _fmt(stat.mean),
                        "sd": _fmt(stat.sd),
                        "sem": _fmt(stat.sem),
                        "ci95_lo": _fmt(stat.ci95_lo),
                        "ci95_hi": _fmt(stat.ci95_hi),
                        "ci95_halfwidth": _fmt(stat.ci95_halfwidth),
                        "values": ";".join(f"{v:.6f}" for v in stat.values),
                    }
                )
    return _write_rows(CURVE_COLUMNS, rows)


#: The paired-delta table: ``agg.ASYMMETRY_COLUMNS`` (so the rows are built by
#: ``agg._contrast_rows`` verbatim) plus which two totals were paired and which kind of
#: contrast it is, so the three steps and the endpoint are never confused for four
#: independent measurements.
DELTA_COLUMNS: tuple[str, ...] = (
    "contrast",
    "contrast_kind",
    "from_total",
    "to_total",
    "from_pct",
    "to_pct",
    *agg.ASYMMETRY_COLUMNS,
)


def render_delta_csv(deltas: dict[str, Any]) -> str:
    """The published paired deltas, one row per ``(contrast, ratio, metric)``.

    Rows come from ``agg._contrast_rows`` — the same row builder behind the main
    grid's ``asymmetry.csv`` and ``retention_adaptation.csv`` — so ``diff_mean``,
    ``diff_ci95_lo/hi``, ``direction``, ``sign_consistent`` and ``per_seed_diff`` mean
    exactly what they mean there. ``contrast_kind`` separates the three consecutive
    steps from the endpoint, and the omitted pairs are reconstructible by adding
    consecutive steps' ``per_seed_diff`` columns (:data:`CONTRAST_SELECTION_NOTE`).
    """
    rows: list[dict[str, Any]] = []
    for pair, rv_ratio, metric in delta_rows(deltas):
        contrast = contrast_of(deltas, pair["contrast"], rv_ratio, metric)
        if contrast is None:
            continue
        rv_pct = int(round(rv_ratio * 100))
        ms_pct = int(round((1.0 - rv_ratio) * 100))
        name = (
            f"{rv_pct}:{ms_pct}:{metric}(n{pair['to_total']} − n{pair['from_total']})"
        )
        for row in agg._contrast_rows(name, contrast):
            rows.append(
                {
                    "contrast": pair["contrast"],
                    "contrast_kind": pair["contrast_kind"],
                    "from_total": pair["from_total"],
                    "to_total": pair["to_total"],
                    "from_pct": pair["from_pct"],
                    "to_pct": pair["to_pct"],
                    **row,
                }
            )
    return _write_rows(DELTA_COLUMNS, rows)


#: The step-separability table — the one that has to travel with any statement about
#: the curve's shape.
STEP_SEPARABILITY_COLUMNS: tuple[str, ...] = (
    "comparison_kind",
    "rv_ratio",
    "mix",
    "metric",
    "step",
    "step_label",
    "other_step",
    "other_step_label",
    "n",
    "diff_mean",
    "diff_ci95_lo",
    "diff_ci95_hi",
    "direction",
    "sign_consistent",
    "ci95_excludes_zero",
    "ci95_overlap",
    "exact_sign_test_p_two_sided",
    "exact_sign_test_min_p_two_sided",
    "separable",
    "verdict",
    "per_seed_diff",
)


def _flag(value: Optional[bool]) -> str:
    """Tri-state cell: ``true`` / ``false`` / empty for "not computable"."""
    return "" if value is None else ("true" if value else "false")


def render_step_separability_csv(rows: Sequence[dict[str, Any]]) -> str:
    """``step_vs_zero`` + ``step_vs_step`` rows, with ``separable`` false throughout.

    ``verdict`` says which kind of non-separability each row is, so the ranking in
    ``results.md`` §3 can never be read without the reason it is only a ranking.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "comparison_kind": row["comparison_kind"],
                "rv_ratio": row["rv_ratio"],
                "mix": row["mix"],
                "metric": row["metric"],
                "step": row["step"],
                "step_label": row["step_label"],
                "other_step": row["other_step"] or "",
                "other_step_label": row["other_step_label"] or "",
                "n": row["n"],
                "diff_mean": agg._fixed(row["diff_mean"]),
                "diff_ci95_lo": agg._fixed(row["diff_ci95_lo"]),
                "diff_ci95_hi": agg._fixed(row["diff_ci95_hi"]),
                "direction": row["direction"] or "",
                "sign_consistent": _flag(row["sign_consistent"]),
                "ci95_excludes_zero": _flag(row["ci95_excludes_zero"]),
                "ci95_overlap": _flag(row["ci95_overlap"]),
                "exact_sign_test_p_two_sided": (
                    ""
                    if row["exact_sign_test_p_two_sided"] is None
                    else f"{row['exact_sign_test_p_two_sided']:.6f}"
                ),
                "exact_sign_test_min_p_two_sided": (
                    ""
                    if row["exact_sign_test_min_p_two_sided"] is None
                    else f"{row['exact_sign_test_min_p_two_sided']:.6f}"
                ),
                "separable": _flag(row["separable"]),
                "verdict": row["verdict"],
                "per_seed_diff": ";".join(
                    f"{seed}:{value:.6f}"
                    for seed, value in sorted(row["per_seed_diff"].items())
                ),
            }
        )
    return _write_rows(STEP_SEPARABILITY_COLUMNS, out)


PROVENANCE_COLUMNS: tuple[str, ...] = (
    "total_train_samples",
    "pct",
    "rv_ratio",
    "ms_ratio",
    "seed",
    "run_name",
    "source_study",
    "provenance",
    "present",
    "results_path",
    "shared_path",
    "corpus_attested",
    "slurm_log",
)


def render_provenance_csv(
    curve: dict[int, list[agg.SeedCell]], attestation: Optional[dict[str, Any]]
) -> str:
    """One row per cell: which study trained it, whether it is here, and its corpus."""
    rows: list[dict[str, Any]] = []
    for total in TOTALS:
        per_run = (
            ((attestation or {}).get("per_total") or {}).get(str(total)) or {}
        ).get("per_run") or {}
        for cell in curve.get(total, []):
            att = per_run.get(cell.run_name) or {}
            rows.append(
                {
                    "total_train_samples": total,
                    "pct": pct_of(total),
                    "rv_ratio": cell.rv_ratio,
                    "ms_ratio": cell.ms_ratio,
                    "seed": cell.seed,
                    "run_name": cell.run_name,
                    "source_study": cell.source,
                    "provenance": PROVENANCE_BY_TOTAL.get(total),
                    "present": cell.present,
                    "results_path": cell.results_path,
                    "shared_path": cell.shared_path,
                    "corpus_attested": att.get("corpus") or "unattested",
                    "slurm_log": att.get("log"),
                }
            )
    return _write_rows(PROVENANCE_COLUMNS, rows)


# --------------------------------------------------------------------------- #
# Rendering — Markdown.
# --------------------------------------------------------------------------- #
def _pm_n(stat: agg.Stat, digits: int = 4) -> str:
    """``mean ± halfwidth (n=k)`` — a value never appears without its own n."""
    if stat.mean is None:
        return "—"
    return f"{agg._pm(stat, digits)} (n={stat.n})"


#: Metric → digits, for the markdown tables.
_METRIC_DIGITS: dict[str, int] = {
    "avg_cer": 4,
    "rv_cer": 4,
    "ms_cer": 4,
    "rv_accent_stripped_cer": 4,
    "ms_accent_stripped_cer": 4,
    "rv_accent_f1": 3,
    "ms_accent_f1": 3,
}


def render_results_md(
    curve: dict[int, list[agg.SeedCell]],
    per_total: dict[int, dict[float, dict[str, agg.Stat]]],
    deltas: dict[str, Any],
    checks: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    *,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
    maingrid: Optional[dict[str, Any]] = None,
    separability: Optional[Sequence[dict[str, Any]]] = None,
    shape: Optional[dict[str, Any]] = None,
) -> str:
    """Human-readable results page: status first, then the curve, n on every row.

    ``separability`` / ``shape`` default to freshly computed ones rather than to
    "omitted", so the page can never print a step ranking without the table that says
    the ranking is not a finding.
    """
    if separability is None:
        separability = step_separability(curve)
    if shape is None:
        shape = shape_verdict(curve, separability)
    status = status_label(coverage)
    banner = partial_banner(coverage)
    pct_axis = " / ".join(f"{pct_of(t)}%" for t in TOTALS)
    out: list[str] = []
    out.append(
        f"# PACLIC v2 data efficiency — {len(TOTALS)}-point curve "
        f"({pct_axis}) × 3 mix ratios (corrected corpus)"
    )
    out.append("")
    out.append(
        f"**Status:** {status} · **Corpus:** v2 (accent marks corrected) · "
        f"**Cells:** {coverage['n_present']}/{coverage['n_cells']} "
        f"({coverage['n_new_present']}/{coverage['n_new_cells']} newly trained at "
        f"totals {list(TRAINED_TOTALS)}) · **Analysis-only** (no training, no inference)."
    )
    out.append("")
    if banner is not None:
        out.append(f"> **{banner}**")
        out.append("")
    other_warnings = [
        w
        for w in audit_warnings(
            curve,
            coverage,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
            maingrid=maingrid,
        )
        if w != banner
    ]
    if other_warnings:
        out.append("> **Integrity warnings:**")
        for warning in other_warnings:
            out.append(f"> - {warning}")
        out.append("")
    out.append(
        f"The curve has **{len(TOTALS)} points**: "
        f"`total_train_samples ∈ {list(TRAINED_TOTALS)}` "
        f"({', '.join(f'{pct_of(t)}%' for t in TRAINED_TOTALS)} of the {FULL_TOTAL} "
        f"shared parallel texts), all trained by `{LAUNCHER}` into "
        f"`{DEFAULT_LOW_STUDY_ROOT}`; and `total_train_samples={FULL_TOTAL}` "
        f"({pct_of(FULL_TOTAL)}%), **inherited by reference** from the v2 main grid "
        f"(`{INHERITED_LAUNCHER}`, `{DEFAULT_FULL_STUDY_ROOT}`) and never re-run. Which "
        "point came from which study is recorded per cell (`source_study` / "
        "`provenance` in every table), and the inherited point is verified against the "
        "committed main-grid `aggregate.json` (§6). Statistics — mean, sample SD, "
        "Student-t 95% CI with df=n−1 — the paired contrasts and the exact-match "
        "recompute are reused verbatim from "
        "`scripts.evaluation.aggregate_accent_ratio_multiseed`, the same code the "
        "main-grid bundle uses; the separability vocabulary in §3 is the plateau "
        "bundle's. **v2 numbers are never pooled with, or subtracted from, v1 numbers**: "
        "the two are different image corpora."
    )
    out.append("")

    out.append("## 0. Coverage")
    out.append("")
    out.append(
        "`cells present` is the point's own present-cell count. **`paired n` is a "
        "property of a CONTRAST, not of a point** — the number of seeds present at both "
        "of its totals — so it is listed per published contrast below the point table. "
        "One absent cell lowers the n of the two steps and the endpoint that touch it "
        "and leaves every other delta at n=3."
    )
    out.append("")
    out.append(
        "| total (pct) | RV:MS | provenance | cells present | seeds present | "
        "seeds missing |"
    )
    out.append("| :--- | :--- | :--- | ---: | :--- | :--- |")
    for total in TOTALS:
        for rv_ratio in RV_RATIOS:
            blk = coverage["per_point"][f"{total}:{rv_ratio:g}"]
            rv_pct = int(round(blk["rv_ratio"] * 100))
            ms_pct = int(round(blk["ms_ratio"] * 100))
            out.append(
                f"| {total} ({blk['pct']}%) | {rv_pct}:{ms_pct} | "
                f"{PROVENANCE_BY_TOTAL.get(total)} | "
                f"{blk['n_present']}/{blk['n_cells']} | "
                f"{blk['seeds_present'] or '—'} | {blk['seeds_missing'] or '—'} |"
            )
    out.append("")
    out.append("| contrast | kind | RV:MS | paired n | paired seeds |")
    out.append("| :--- | :--- | :--- | ---: | :--- |")
    for blk in coverage["paired"].values():
        rv_pct = int(round(blk["rv_ratio"] * 100))
        ms_pct = int(round(blk["ms_ratio"] * 100))
        out.append(
            f"| {blk['label']} | {blk['contrast_kind']} | {rv_pct}:{ms_pct} | "
            f"{blk['n_paired']}/{blk['n_expected']} | {blk['seeds_paired'] or '—'} |"
        )
    out.append("")

    out.append(
        f"## 1. The {len(TOTALS)}-point curve — mean ± 95% CI across the present seeds"
    )
    out.append("")
    out.append(
        "One block per mix ratio, one row per training total. Every value carries its "
        "own `(n=k)`: the runner writes `results.json` before the prediction CSVs and "
        "exact-match is re-sourced from those CSVs (study), so mid-flight a cell can "
        "contribute CER but not every metric. A point with n<3 is not a 3-seed point; "
        "with n<2 it has no CI at all."
    )
    out.append("")
    for rv_ratio in RV_RATIOS:
        rv_pct = int(round(rv_ratio * 100))
        ms_pct = int(round((1.0 - rv_ratio) * 100))
        out.append(f"### 1.{RV_RATIOS.index(rv_ratio) + 1} RV:MS {rv_pct}:{ms_pct}")
        out.append("")
        out.append(
            "| total (pct) | provenance | avg CER | RV CER | MS CER | "
            "RV base CER | MS base CER | RV accF1 | MS accF1 |"
        )
        out.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for total in TOTALS:
            metrics = (per_total.get(total) or {}).get(rv_ratio)
            if metrics is None:
                continue
            cells = " | ".join(
                _pm_n(metrics[metric], _METRIC_DIGITS[metric])
                for metric in CURVE_METRICS
            )
            out.append(
                f"| {total} ({pct_of(total)}%) | "
                f"{PROVENANCE_BY_TOTAL.get(total)} | {cells} |"
            )
        out.append("")

    out.append("## 2. The published steps and the endpoint (paired by seed)")
    out.append("")
    out.append(
        "Δ = (value at the larger total) − (value at the smaller), differenced **within "
        "a seed** and then summarized, so the interval is the interval of the step and "
        "not the difference of two intervals. `n` is the paired n. A negative Δ on a "
        "CER row means more data lowered the error; a positive Δ on an accent-F1 row "
        "means more data raised it. `sign-consistent` says every paired seed moved the "
        "same way — with n=3 that is a direction statement, not a significance one "
        "(the exact paired sign-flip test cannot go below two-sided p=0.25 at n=3)."
    )
    out.append("")
    out.append(deltas["selection_note"])
    out.append("")
    out.append("| contrast | kind | RV:MS | quantity | n | Δ |")
    out.append("| :--- | :--- | :--- | :--- | ---: | :--- |")
    for pair, rv_ratio, metric in delta_rows(deltas):
        contrast = contrast_of(deltas, pair["contrast"], rv_ratio, metric)
        if contrast is None:
            continue
        rv_pct = int(round(rv_ratio * 100))
        ms_pct = int(round((1.0 - rv_ratio) * 100))
        out.append(
            f"| {pair['label']} | {pair['contrast_kind']} | {rv_pct}:{ms_pct} | "
            f"{metric} | {contrast['diff']['n']} | "
            f"{agg._diff_phrase(contrast, _METRIC_DIGITS[metric])} |"
        )
    out.append("")
    out.append(
        f"**Retention–adaptation is not in this bundle.** {RETENTION_ADAPTATION_NOTE}"
    )
    out.append("")
    out.append(f"**No minimum-target-ratio table.** {MIN_TARGET_RATIO_NOTE}")
    out.append("")

    out.append("## 3. Is the curve's SHAPE determined? No — step separability at n=3")
    out.append("")
    out.append(shape["claim_note"])
    out.append("")
    out.append(f"> {shape['separability_note']}")
    out.append("")
    out.append(
        f"Ranking quantity: `{shape['primary_metric']}`. Full table over every metric "
        "and both comparison kinds: `tables/step_separability.csv`."
    )
    out.append("")
    for blk in shape["by_ratio"].values():
        out.append(f"**RV:MS {blk['mix']}** — {blk['statement']}")
        out.append("")
        out.append(
            "| step | n | Δ | CI excludes 0 | sign-consistent | exact p (floor) | "
            "verdict |"
        )
        out.append("| :--- | ---: | :--- | :--- | :--- | :--- | :--- |")
        for row in separability:
            if (
                row["comparison_kind"] != "step_vs_zero"
                or row["rv_ratio"] != blk["rv_ratio"]
                or row["metric"] != shape["primary_metric"]
            ):
                continue
            digits = _METRIC_DIGITS[row["metric"]]
            mean = row["diff_mean"]
            body = "—" if mean is None else f"{mean:+.{digits}f}"
            if row["diff_ci95_lo"] is not None and row["diff_ci95_hi"] is not None:
                body += (
                    f" [{row['diff_ci95_lo']:+.{digits}f}, "
                    f"{row['diff_ci95_hi']:+.{digits}f}]"
                )
            p = row["exact_sign_test_p_two_sided"]
            floor = row["exact_sign_test_min_p_two_sided"]
            out.append(
                f"| {row['step_label']} | {row['n']} | {body} | "
                f"{_flag(row['ci95_excludes_zero']) or '—'} | "
                f"{_flag(row['sign_consistent']) or '—'} | "
                + ("—" if p is None else f"{p:.3f}")
                + (" (" + (f"floor {floor:.3f}" if floor is not None else "—") + ")")
                + f" | {row['verdict']} |"
            )
        out.append("")
        out.append(
            "| step vs step | n | Δ of Δ | step CIs overlap | exact p (floor) | verdict |"
        )
        out.append("| :--- | ---: | :--- | :--- | :--- | :--- |")
        for row in separability:
            if (
                row["comparison_kind"] != "step_vs_step"
                or row["rv_ratio"] != blk["rv_ratio"]
                or row["metric"] != shape["primary_metric"]
            ):
                continue
            digits = _METRIC_DIGITS[row["metric"]]
            mean = row["diff_mean"]
            body = "—" if mean is None else f"{mean:+.{digits}f}"
            if row["diff_ci95_lo"] is not None and row["diff_ci95_hi"] is not None:
                body += (
                    f" [{row['diff_ci95_lo']:+.{digits}f}, "
                    f"{row['diff_ci95_hi']:+.{digits}f}]"
                )
            p = row["exact_sign_test_p_two_sided"]
            floor = row["exact_sign_test_min_p_two_sided"]
            out.append(
                f"| {row['step_label']} vs {row['other_step_label']} | {row['n']} | "
                f"{body} | {_flag(row['ci95_overlap']) or '—'} | "
                + ("—" if p is None else f"{p:.3f}")
                + (" (" + (f"floor {floor:.3f}" if floor is not None else "—") + ")")
                + f" | {row['verdict']} |"
            )
        out.append("")

    out.append("## 4. Zero-shot asymmetry at every point of the curve (paired by seed)")
    out.append("")
    out.append(
        "RV→MS = the RV-only model (`rv_ratio=1.0`) read on **MS** test; MS→RV = the "
        "MS-only model (`rv_ratio=0.0`) read on **RV** test. Computed by "
        "`agg.asymmetry` at each total independently, so every low-data row is the same "
        "quantity the main-grid bundle reports at 100%."
    )
    out.append("")
    out.append(
        "| total (pct) | metric | n paired | RV→MS | MS→RV | diff (MS→RV − RV→MS) |"
    )
    out.append("| :--- | :--- | ---: | ---: | ---: | :--- |")
    asym = asymmetry_by_total(curve)
    for total in TOTALS:
        block_all = asym.get(str(total))
        if block_all is None:
            continue
        for metric, digits in (("cer", 4), ("chrf", 2), ("accent_f1", 3)):
            block = block_all["zero_shot"][metric]
            rv2ms = _stat_of(block["rv_to_ms"])
            ms2rv = _stat_of(block["ms_to_rv"])
            out.append(
                f"| {total} ({pct_of(total)}%) | {metric} | {block['diff']['n']} | "
                f"{_pm_n(rv2ms, digits)} | {_pm_n(ms2rv, digits)} | "
                f"{agg._diff_phrase(block, digits)} |"
            )
    out.append("")

    out.append("## 5. Per-cell ledger + corpus attestation")
    out.append("")
    out.append(
        "`best epoch`/`best step`/`global step` come from each cell's own "
        "`results.json` `training_state`. `corpus` is attested from the immutable "
        "Slurm `.out` header of **that arm's own** log set — `results.json` records no "
        "corpus path, and `key_list_sha256` cannot discriminate v1 from v2."
    )
    out.append("")
    out.append(
        "| total | RV:MS | seed | present | study | best epoch | best step | "
        "global step | corpus (attested) |"
    )
    out.append("| :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | :--- |")
    for total in TOTALS:
        per_run = (
            ((attestation or {}).get("per_total") or {}).get(str(total)) or {}
        ).get("per_run") or {}
        for cell in curve.get(total, []):
            ts = cell.training_state or {}
            be = ts.get("best_epoch")
            bs = ts.get("best_step")
            gs = ts.get("global_step")
            att = per_run.get(cell.run_name) or {}
            corpus = att.get("corpus") or "unattested"
            if not cell.present and att.get("corpus"):
                # A log without a results.json: the cell is running or failed. The
                # attestation is about the log, so say so rather than imply a result.
                corpus = f"{corpus} (log only)"
            rv_pct = int(round(cell.rv_ratio * 100))
            ms_pct = int(round(cell.ms_ratio * 100))
            out.append(
                f"| {total} | {rv_pct}:{ms_pct} | {cell.seed} | "
                f"{'yes' if cell.present else 'NO'} | {cell.source} | "
                f"{'—' if be is None else f'{be:g}'} | {'—' if bs is None else bs} | "
                f"{'—' if gs is None else gs} | {corpus} |"
            )
    out.append("")

    if maingrid is not None:
        out.append("## 6. Is the inherited 100% point the published main-grid result?")
        out.append("")
        if not maingrid["checked"]:
            out.append(f"**NOT CHECKED** — {maingrid['reason']}.")
        else:
            verdict = "verified" if maingrid["ok"] else "DISAGREES"
            out.append(
                f"**{verdict}** — {maingrid['n_comparisons']} comparison(s) over "
                f"{len(RV_RATIOS)} shared ratio(s) × {len(CURVE_METRICS)} metric(s) "
                f"against `{DEFAULT_MAINGRID_AGGREGATE}` "
                f"(status `{maingrid['maingrid_status']}`, complete "
                f"`{maingrid['maingrid_complete']}`), tolerance "
                f"{maingrid['tolerance']:g}"
                + (
                    "."
                    if maingrid["ok"]
                    else f" — mismatches: {maingrid['mismatches']}."
                )
            )
            out.append("")
            out.append(maingrid["note"])
        out.append("")

    if subset is not None:
        out.append("## 7. Recipe / subset consistency (four scopes)")
        out.append("")
        jw = subset["join_wide"]
        out.append(
            f"**Join-wide held-fixed fields** — across all {subset['n_cells_checked']} "
            f"present cell(s) of every total ({', '.join(jw['fields'])}): "
            f"**{'consistent' if jw['consistent'] else 'INCONSISTENT'}**"
            + ("" if jw["consistent"] else f" — differing: {jw['inconsistent_fields']}")
            + f". {jw['scope']}."
        )
        out.append("")
        per_total_phrase = "; ".join(
            f"total {blk['total_train_samples']}: {blk['n_cells_checked']} cell(s), "
            f"requested={blk['held_fixed'].get('total_train_samples_requested')}, "
            + (
                "axis OK"
                if blk["total_train_samples_requested_matches_axis"]
                else "AXIS MISMATCH"
            )
            for blk in subset["per_total"].values()
        )
        out.append(
            "**Per-total pins** — `total_train_samples_requested` / "
            "`parallel_keys_used` must be constant *within* a total and must equal "
            f"that total. {per_total_phrase or 'no cell present'}."
        )
        out.append("")
        sha_verdict = (
            "consistent"
            if subset["key_list_sha256_consistent"]
            else "INCONSISTENT WITHIN A (total, seed)"
        )
        out.append(
            "**`key_list_sha256`** — checked WITHIN each `(total, seed)` across the "
            f"three ratios: **{sha_verdict}**"
            + (
                ""
                if subset["key_list_sha256_consistent"]
                else f" — split group(s): {subset['key_list_sha256_split_groups']}"
            )
            + f". {len(subset['key_list_sha256_distinct_join_wide'])} distinct value(s) "
            f"join-wide, {subset['key_list_sha256_expected_distinct']} expected "
            f"({len(SEEDS)} seeds × {len(TOTALS)} totals). "
            f"{KEY_LIST_SHA_SCOPE_NOTE}"
        )
        out.append("")
        nesting = subset.get("nesting")
        if nesting is None:
            out.append(
                "**Subset nesting — NOT CHECKED in this run** (no `parallel_keys.txt` "
                f"was read). {NESTING_NOTE}"
            )
        else:
            out.append(
                "**Subset nesting** — within each seed, is every total's ordered key "
                "list a PREFIX of the next total's? **"
                + ("nested" if nesting["nested"] else "VIOLATED")
                + f"** — {nesting['n_checked']}/{nesting['n_pairs']} (seed, step) "
                f"pair(s) checked, {nesting['n_unchecked']} unchecked, "
                f"{nesting['n_violations']} violation(s)"
                + ("" if nesting["nested"] else f": {nesting['violations']}")
                + f". {NESTING_NOTE}"
            )
        out.append("")
        out.append(
            "**Mix realization** (`rv/ms_train_samples` == floor split of that cell's "
            "OWN requested total at its ratio): "
            + (
                "all cells match"
                if not subset["mix_mismatches"]
                else f"MISMATCH in {subset['mix_mismatches']}"
            )
            + f". {v2mg.KEY_LIST_SHA_NOTE}"
        )
        out.append("")

    if integrity is not None:
        out.append("## 8. Per-cell invariants (seed ↔ dir, distinct seeds, parse)")
        out.append("")
        out.append(
            "Verdict: **"
            + ("all hold" if integrity["ok"] else "VIOLATED")
            + "** — seed↔dir mismatches: "
            + (
                "none"
                if not integrity["seed_mismatches"]
                else str([m["run_name"] for m in integrity["seed_mismatches"]])
            )
            + "; curve points without distinct seeds: "
            + (
                "none"
                if not integrity["point_seed_duplicates"]
                else str(
                    [
                        (d["total_train_samples"], d["rv_ratio"])
                        for d in integrity["point_seed_duplicates"]
                    ]
                )
            )
            + "; unreadable `results.json`: "
            + (
                (
                    "none"
                    if integrity["unreadable_check_performed"]
                    else "not checked (no study roots given)"
                )
                if not integrity["unreadable_results_json"]
                else str(integrity["unreadable_results_json"])
            )
            + "."
        )
        out.append("")

    cc = agg.cross_check_summary(checks)
    if cc["n_views_checked"]:
        parts = []
        for key, label in (
            ("delta_cer", "CER"),
            ("delta_wer", "WER"),
            ("delta_accent_f1", "accent F1"),
            ("delta_exact_match", "exact-match"),
        ):
            blk = cc[key]
            if blk["max"] is not None:
                parts.append(f"max |Δ{label}|={blk['max']:.5f}")
        out.append(
            "**Provenance cross-check** — over "
            f"{cc['n_views_checked']} re-evaluated views across every point, the runner "
            "`results.json` metrics reproduce under the study inference-free evaluator "
            "to "
            + (", ".join(parts) if parts else "no comparable metric")
            + " (exact-match is *sourced* from that evaluator for every cell, study)."
        )
        out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Provenance manifest.
# --------------------------------------------------------------------------- #
def build_manifest(
    curve: dict[int, list[agg.SeedCell]],
    checks: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    *,
    low_study_root: Path,
    full_study_root: Path,
    shared_low_study_root: Optional[str],
    shared_full_study_root: Optional[str],
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
    maingrid: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Data-provenance manifest for the v2 data-efficiency bundle."""
    return {
        "study": STUDY,
        "issue": 126,
        "parent_issue": 122,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "corpus": "v2 (accent marks corrected; same ISO text as v1, different pixels)",
        "analysis_only": True,
        "no_training_no_inference": True,
        "status": status_label(coverage),
        "complete": coverage["complete"],
        "partial_banner": partial_banner(coverage),
        "coverage": coverage,
        "design": {
            "shape": (
                f"{len(TOTALS)}-point curve: total_train_samples ∈ {list(TOTALS)} "
                f"(= {[pct_of(t) for t in TOTALS]}% of {FULL_TOTAL}) "
                f"× rv_ratio ∈ {list(RV_RATIOS)} × seed ∈ {list(SEEDS)}"
            ),
            "n_cells": TARGET_N_CELLS,
            "n_newly_trained": TARGET_N_NEW_CELLS,
            "n_inherited": TARGET_N_CELLS - TARGET_N_NEW_CELLS,
            "n_per_point": len(SEEDS),
            "trained_totals": list(TRAINED_TOTALS),
            "inherited_total": FULL_TOTAL,
            "manifest_yaml": MANIFEST_YAML,
            "why_four_points": (
                "The study shipped as {1042, 10420}. Two points fix the endpoints of a "
                "comparison and say nothing about the shape between them, which is what "
                "'where does the gain saturate' asks, so 2605 and 5210 were trained as a "
                "second submission of the same launcher under the same pins. The 100% "
                "end is still not re-run: the v2 main grid already trains these three "
                "ratios at 10420 under these three seeds with the same foundation, "
                "recipe and corpus. n=3 at EVERY point is what makes each step a paired "
                "difference with an interval rather than a two-number gap; seeds were "
                "deliberately not traded for more totals (manifest "
                "design_notes.why_three_seeds)."
            ),
            "why_three_ratios": (
                "Endpoints (100:0, 0:100) plus the balanced middle (50:50) carry the "
                "retention/adaptation question; 0.75 / 0.25 refine the mix axis and "
                "exist at 100% only, in the main grid. Dropping them bought the third "
                "seed at every retained point."
            ),
            "published_contrasts": [contrast_key(a, b) for a, b in CONTRAST_PAIRS],
            "omitted_contrasts": [contrast_key(a, b) for a, b in OMITTED_PAIRS],
            "contrast_selection": CONTRAST_SELECTION_NOTE,
            "shape_claim": SHAPE_CLAIM_NOTE,
            "separability": SEPARABILITY_NOTE,
            "retention_adaptation": RETENTION_ADAPTATION_NOTE,
            "minimum_target_ratio": MIN_TARGET_RATIO_NOTE,
        },
        "roots": {
            "trained": {
                "totals": list(TRAINED_TOTALS),
                "pct": [pct_of(t) for t in TRAINED_TOTALS],
                "launcher": LAUNCHER,
                "study_root": str(low_study_root),
                "shared_study_root": shared_low_study_root,
                "run_name_pattern": "rv_<pct>_ms_<pct>_n<total>_seed_<n>",
                "slurm_log_glob": DATAEFF_SLURM_LOG_GLOB,
                "provenance": PROVENANCE_BY_TOTAL[LOW_TOTAL],
                "note": (
                    "One root, one launcher, one run-name template for all three "
                    "trained totals; they were submitted in two waves (1042 first, then "
                    "2605 + 5210) under identical pins, which is a scheduling fact and "
                    "not a provenance distinction."
                ),
            },
            "inherited": {
                "total_train_samples": FULL_TOTAL,
                "pct": pct_of(FULL_TOTAL),
                "launcher": INHERITED_LAUNCHER,
                "study_root": str(full_study_root),
                "shared_study_root": shared_full_study_root,
                "run_name_pattern": "rv_<pct>_ms_<pct>_seed_<n>",
                "slurm_log_glob": SLURM_LOG_GLOB_BY_TOTAL[FULL_TOTAL],
                "provenance": PROVENANCE_BY_TOTAL[FULL_TOTAL],
                "reused_by_reference": True,
                "bundle": str(v2mg.DEFAULT_OUT_DIR),
            },
        },
        "why_not_the_neighbouring_builders": (
            "build_paclic_v2_maingrid_analysis is single-root and single-total: "
            "TARGET_N_CELLS is len(RV_RATIOS)*len(SEEDS) over five ratios, run_name "
            "emits no n<total> token, and subset_consistency holds "
            "total_train_samples_requested fixed GRID-WIDE — the field this study "
            "varies by design. build_paclic_accent_ratio_data_efficiency_analysis has "
            "the right shape but is wired to v1 provenance: three roots with the "
            "seed-42→freeze-factorial routing (unsuffixed rv_<pct>_ms_<pct> dirs), "
            "is_inherited testing for source in ('prior-study-reference', "
            "'multiseed'), five ratios in every table, no corpus attestation at all, "
            "and best-effort exit-0 semantics on a partial curve — the opposite of this "
            "study's contract. Both produced committed bundles that must keep "
            "regenerating unchanged, so this is a sibling that reuses their pieces."
        ),
        "reuse": {
            "cell_reader": "scripts.evaluation.aggregate_accent_ratio_multiseed.read_cell_dir",
            "statistics": "aggregate_accent_ratio_multiseed.summarize (mean / sample SD / Student-t 95% CI, df=n-1)",
            "per_point_summary": "aggregate_accent_ratio_multiseed.aggregate_by_ratio (applied per total)",
            "paired_contrasts": "aggregate_accent_ratio_multiseed._contrast (every step, the endpoint, and every step-vs-step second difference) / .asymmetry (per total)",
            "cross_check": "aggregate_accent_ratio_multiseed.cross_check_summary",
            "tables": "aggregate_accent_ratio_multiseed.render_per_seed_csv / render_asymmetry_csv / _contrast_rows / _fixed",
            "corpus_attestation": "build_paclic_v2_maingrid_analysis.attest_corpus_from_slurm_logs / .corpus_attestation_summary / .attestation_defects",
            "subset_provenance": "build_paclic_v2_maingrid_analysis.subset_provenance / .expected_mix_counts / .SUBSET_FIELDS",
            "inherited_run_names": "build_paclic_v2_maingrid_analysis.run_name",
            "significance_test": "build_paclic_accent_transfer_analysis.sign_flip_permutation_test (exact paired sign-flip, the same function the plateau bundle uses)",
            "separability_vocabulary": (
                "build_paclic_v2_effect_sizes_and_plateau.ORDER_CONSISTENT / "
                ".ORDER_UNSTABLE / ._cis_overlap — one vocabulary for 'not separable at "
                "n=3' across the layer"
            ),
            "figure_layer": "scripts.evaluation.build_paclic_paper_figures (fig6_data_efficiency)",
            "not_reimplemented": (
                "No statistic is computed in this module. Its own code is the join, the "
                "per-point / paired coverage, the scoping of the recipe checks, the "
                "choice of which contrasts to publish, and three table shapes agg has no "
                "renderer for (the joined curve, the paired step/endpoint deltas and the "
                "step separability), whose values are read straight off Stat / _contrast "
                "/ sign_flip_permutation_test payloads."
            ),
        },
        "cells": [
            {
                "total_train_samples": total,
                "pct": pct_of(total),
                "rv_ratio": c.rv_ratio,
                "ms_ratio": c.ms_ratio,
                "seed": c.seed,
                "run_name": c.run_name,
                "source": c.source,
                "provenance": PROVENANCE_BY_TOTAL.get(total),
                "present": c.present,
                "results_path": c.results_path,
                "shared_path": c.shared_path,
                "rv_chrf_source": c.rv_chrf_source,
                "ms_chrf_source": c.ms_chrf_source,
                "rv_exact_match_source": c.rv_exact_match_source,
                "ms_exact_match_source": c.ms_exact_match_source,
                "training_state_source": c.training_state_source,
                "training_state_recovery_method": c.training_state_recovery_method,
                "model_base": c.model_base,
                "corpus_attestation": (
                    (
                        ((attestation or {}).get("per_total") or {}).get(str(total))
                        or {}
                    ).get("per_run")
                    or {}
                ).get(c.run_name),
            }
            for total in TOTALS
            for c in curve.get(total, [])
        ],
        "subset_consistency": subset,
        "corpus_attestation": attestation,
        "grid_integrity": integrity,
        "maingrid_consistency": maingrid,
        "metric_sourcing_note": (
            "CER / WER / chrF / accent metrics are the runner's canonical training-eval "
            "numbers from each cell's own results.json; the v2 runner records chrF for "
            "every cell, so nothing is recomputed to fill a gap. exact_match (rate + "
            "count) is recomputed inference-free from the committed "
            "{rv,ms}_predictions.csv via scripts.evaluation.prediction_metrics under "
            "the pinned NFC+strip normalization (study/study); the runner value feeds only "
            "the cross_check delta."
        ),
        "evaluator": {
            "module": "scripts.evaluation.prediction_metrics",
            "generic_ocr_backend": "scripts.training.trocr.metrics.OCRMetrics",
            "normalization": {
                "form": "NFC",
                "strip": True,
                "collapse_whitespace": False,
            },
            "dependency_pin": agg.evaluator_dependency_pin(),
        },
        "cross_check": agg.cross_check_summary(checks),
        "limitations": [
            "The points' error bars are not subset-matched ACROSS seeds. "
            "parallel_subset_seed is left null (unlike the v1 study study), so a "
            "low-data cell's interval covers subset sampling as well as routing and "
            "training stochasticity, while the 10420 interval cannot cover subset "
            "sampling — at full N the key set is the whole corpus. All of them are 'all "
            "variation downstream of `seed`', which is the honest interval for 'only n "
            "lines were available', but a paired delta is a delta between two "
            "CONDITIONS, not between two runs on one subset. WITHIN a seed the four "
            "selections do nest exactly, and that is verified "
            "(subset_consistency.nesting), so a step is 'these lines plus those'.",
            "Four points do NOT establish a curve shape at n=3. No pair of consecutive "
            "steps is separable, so 'front-loaded', 'log-linear' and 'saturates by X%' "
            "are all unavailable; the bundle reports each step's paired difference and "
            "the pairwise order stability of the steps instead, and its shape verdict is "
            "shape_not_determined_at_n3 (design.shape_claim, aggregate.shape).",
            "n=3 per point: the exact paired sign-flip permutation test cannot go "
            "below two-sided p=0.25, so 'sign-consistent' is a direction statement, "
            "never a significance claim.",
            "Of the six ordered pairs of points, four are published (three consecutive "
            "steps + the endpoint). The omitted two are exact per-seed sums of published "
            "steps rather than unmeasured comparisons (design.contrast_selection).",
            "No retention/adaptation and no minimum-target-notation-ratio table — "
            "both need the 0.75 / 0.25 ratios, which this study does not train "
            "(design.retention_adaptation, design.minimum_target_ratio).",
            "results.json records no corpus path; corpus identity is attested only "
            "from the Slurm .out headers, per arm, and an unattested cell is reported "
            "as such. The header's corpus= token is a LABEL derived by the launcher, so "
            "the echoed data dirs + data_source are what is validated.",
            "key_list_sha256 hashes ISO text keys, identical between v1 and v2 by "
            "construction, so it is not corpus evidence. It is also seeded AND "
            "prefixed, hence a within-(seed, total) invariant here.",
            "v1 and v2 are different image corpora. No v1 cell is pooled in and no "
            "v1→v2 delta is reported; the v1 study bundle is a record of the abandoned "
            "corpus, and its 2605 / 5210 cells are NOT the interior points of this "
            "curve even though they carry the same totals.",
            "Figures are gated on complete==true AND a healthy corpus attestation on "
            "EVERY point AND a verified inherited point AND no publication-blocking "
            "warning.",
        ],
        "reproduce": (
            "uv run python -m scripts.evaluation.build_paclic_v2_data_efficiency_analysis "
            f"--low-study-root {DEFAULT_LOW_STUDY_ROOT} "
            f"--full-study-root {DEFAULT_FULL_STUDY_ROOT} "
            f"--out-dir {DEFAULT_OUT_DIR}"
        ),
        "generated_by": "scripts.evaluation.build_paclic_v2_data_efficiency_analysis",
    }


# --------------------------------------------------------------------------- #
# Bundle orchestration.
# --------------------------------------------------------------------------- #
def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[{STUDY}] wrote {path}", file=sys.stderr)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write(
        path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def build_bundle(
    curve: dict[int, list[agg.SeedCell]],
    checks: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    low_study_root: Path,
    full_study_root: Path,
    shared_low_study_root: Optional[str] = None,
    shared_full_study_root: Optional[str] = None,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
    maingrid: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write ``tables/`` + ``aggregate.json`` + ``manifest.json`` + ``results.md``.

    Returns the aggregate payload (also the input to :func:`build_figures`).

    ``integrity`` defaults to a freshly computed :func:`grid_integrity` over both
    roots rather than to "not checked", so a caller cannot get a bundle whose
    per-cell invariants were silently skipped.
    """
    if integrity is None:
        integrity = grid_integrity(
            curve, low_study_root=low_study_root, full_study_root=full_study_root
        )
    coverage = coverage_summary(curve)
    per_total = aggregate_curve(curve)
    deltas = curve_deltas(curve)
    separability = step_separability(curve)
    shape = shape_verdict(curve, separability)

    tables = out_dir / "tables"
    _write(tables / "curve.csv", render_curve_csv(per_total))
    _write(tables / "curve_deltas.csv", render_delta_csv(deltas))
    _write(
        tables / "step_separability.csv",
        render_step_separability_csv(separability),
    )
    _write(tables / "provenance.csv", render_provenance_csv(curve, attestation))
    for total in TOTALS:
        cells = curve.get(total, [])
        # agg's renderers verbatim, one file per point: the schema then matches the main
        # grid's tables field for field, and the file name carries the total.
        _write(tables / f"per_seed_n{total}.csv", agg.render_per_seed_csv(cells))
        _write(
            tables / f"asymmetry_n{total}.csv",
            agg.render_asymmetry_csv(agg.asymmetry(cells)),
        )
    _write(
        out_dir / "results.md",
        render_results_md(
            curve,
            per_total,
            deltas,
            checks,
            coverage,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
            maingrid=maingrid,
            separability=separability,
            shape=shape,
        ),
    )
    aggregate = build_aggregate(
        curve,
        per_total,
        checks,
        coverage,
        subset=subset,
        attestation=attestation,
        integrity=integrity,
        maingrid=maingrid,
    )
    _write_json(out_dir / "aggregate.json", aggregate)
    _write_json(
        out_dir / "manifest.json",
        build_manifest(
            curve,
            checks,
            coverage,
            low_study_root=low_study_root,
            full_study_root=full_study_root,
            shared_low_study_root=shared_low_study_root,
            shared_full_study_root=shared_full_study_root,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
            maingrid=maingrid,
        ),
    )
    return aggregate


# --------------------------------------------------------------------------- #
# Figures — reuse the PACLIC figure layer, gated on a whole, attested curve.
# --------------------------------------------------------------------------- #
#: The one paper figure whose input is exactly a data-efficiency ``curves`` payload.
V2_FIGURE_STEMS: tuple[str, ...] = ("fig6_data_efficiency",)

#: Title burned into the figure. The v1 default asserts a v1 *finding* ("base letters
#: transfer across notation, accent marks do not"); a v2 render must not inherit a
#: conclusion in its pixels, so the v2 title states the design and nothing more.
#: Short enough to fit the figure at print type size. The x-axis labels already
#: carry the totals and their percentages, so the suptitle does not repeat them.
V2_FIGURE_SUPTITLE: str = "Data efficiency by training mix — mean ± 95% CI (n=3)"

#: v2-safe caption. Descriptive only: what is plotted, from where, and the honesty
#: condition a four-point curve at n=3 carries — that the drawn line is not a shape.
V2_FIGURE_SHOWS: dict[str, str] = {
    "fig6_data_efficiency": (
        "Four panels against the training total for each of the three RV:MS mixes "
        "trained at every point (100:0, 50:50, 0:100), mean ± 95% CI over n=3 seeds. "
        "(a)/(b) all-symbol CER on the RV and MS test sets; (c)/(d) the same MS test "
        "set split into its base-letter (accent-stripped CER) and accent-mark (accent "
        "F1) channels. Four points — 1042 / 2605 / 5210 (10/25/50%) trained by the v2 "
        "data-efficiency launcher and 10420 (100%) inherited from the v2 main grid. The "
        "line joining them is a guide to the eye, NOT a fitted shape: at n=3 no pair of "
        "consecutive steps is separable, so the figure supports 'the curve was measured "
        "at these four levels' and not 'the gain is front-loaded' or 'it saturates by "
        "X%' (tables/step_separability.csv, aggregate.json shape). Descriptive: the "
        "direction and size of each step are read off the plotted values and the paired "
        "deltas in tables/curve_deltas.csv, not asserted here."
    ),
}


def _figure_dir_conflict(out_dir: Path) -> Optional[str]:
    """Does ``out_dir`` already hold another study's figure manifest?"""
    manifest = out_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"{manifest} exists but does not parse"
    study = payload.get("study") if isinstance(payload, dict) else None
    if study is not None and study != STUDY:
        return f"{manifest} belongs to study {study!r}, not {STUDY!r}"
    return None


def build_figures(
    aggregate: dict[str, Any],
    out_dir: Path,
    *,
    bundle_dir: str = str(DEFAULT_OUT_DIR),
    make_pdf: bool = True,
    dpi: int = 300,
) -> list[dict[str, Any]]:
    """Render the v2 data-efficiency figure from a **complete**, verified aggregate.

    Reuses ``build_paclic_paper_figures.fig_data_efficiency`` — same colorblind-safe
    palette, same deterministic Agg rendering, same ≥300-dpi PNG + vector PDF —
    parameterised on the ratio list and the suptitle, because the v1 defaults are five
    ratios and a v1 conclusion.

    Five gates, all of which must pass before anything is written:

    * **coverage** — :class:`IncompleteGridError` unless all
      :data:`TARGET_N_CELLS` cells are present: the figure captions itself "n=3
      seeds", which a partial curve cannot honour.
    * **corpus** — :class:`UnverifiedCorpusError` unless the attestation is healthy on
      **both** arms. A skipped attestation counts as unverified: not checking is not
      the same as passing.
    * **inherited point** — :class:`GridInvariantError` unless
      :func:`maingrid_consistency` verified the 100% end against the committed
      main-grid bundle. A figure whose right-hand point silently disagrees with the
      published figure of the same number is the worst artifact this study could ship.
    * **invariants** — :class:`GridInvariantError` on any publication-blocking
      warning, the same predicate the CLI exits nonzero on, so figures on disk and a
      zero exit can never disagree.
    * **dir ownership** — :class:`FigureDirConflictError` if ``out_dir`` already holds
      another study's figure manifest, so this study cannot overwrite the committed
      main-grid figure set's provenance.
    """
    if not aggregate.get("complete"):
        coverage = aggregate.get("coverage") or {}
        raise IncompleteGridError(
            "refusing to render the v2 data-efficiency figure from a partial curve: "
            f"{coverage.get('n_present')}/{coverage.get('n_cells')} cells present. "
            f"{aggregate.get('partial_banner')}"
        )
    attestation = aggregate.get("corpus_attestation")
    if attestation is None:
        raise UnverifiedCorpusError(
            "refusing to render the v2 data-efficiency figure without a corpus "
            "attestation: the curve's corpus was not checked "
            "(--no-slurm-attestation, or no attestation passed to build_aggregate). "
            "results.json records no corpus path, so nothing else in the bundle can "
            "tell v1 pixels from v2 pixels, and this figure is labelled v2."
        )
    if not attestation.get("healthy"):
        raise UnverifiedCorpusError(
            "refusing to render the v2 data-efficiency figure from an unhealthy corpus "
            "attestation: "
            f"{attestation.get('n_attested')}/{attestation.get('n_present_cells')} "
            f"cells attested, single_corpus={attestation.get('single_corpus')}, "
            "per-arm health="
            + str(
                {
                    key: blk.get("healthy")
                    for key, blk in sorted((attestation.get("per_total") or {}).items())
                }
            )
            + f". Expected corpus: {attestation.get('expected_corpus')}"
        )
    maingrid = aggregate.get("maingrid_consistency")
    if not (maingrid or {}).get("ok"):
        raise GridInvariantError(
            "refusing to render the v2 data-efficiency figure while the inherited "
            "100% point is unverified or disagrees with the committed v2 main-grid "
            f"bundle: {maingrid}"
        )
    blocking = hard_warnings(aggregate.get("warnings") or [])
    if blocking:
        raise GridInvariantError(
            "refusing to render the v2 data-efficiency figure while "
            f"publication-blocking warning(s) stand: {blocking}"
        )
    conflict = _figure_dir_conflict(out_dir)
    if conflict is not None:
        raise FigureDirConflictError(
            f"refusing to write this study's figures into {out_dir}: {conflict}. "
            "Rendering here would replace another study's figure manifest/README with "
            "this study's, leaving figures described by a manifest that does not name "
            "them. Use a dir of this study's own "
            f"(conventional: {DEFAULT_FIGURES_OUT_DIR})."
        )

    from . import build_paclic_paper_figures as bpf

    figure = bpf.fig_data_efficiency(
        aggregate,
        out_dir,
        make_pdf=make_pdf,
        dpi=dpi,
        rv_ratios=RV_RATIOS,
        suptitle=V2_FIGURE_SUPTITLE,
    )
    sources = [
        {**s, "file": str(s["file"]).replace(bpf.DATA_EFFICIENCY_DIR, bundle_dir)}
        for s in figure.get("sources", [])
    ]
    figures = [
        {
            **figure,
            "sources": sources,
            "shows": V2_FIGURE_SHOWS.get(figure["stem"], figure["shows"]),
        }
    ]
    stems = tuple(f["stem"] for f in figures)
    if stems != V2_FIGURE_STEMS:
        raise RuntimeError(f"unexpected v2 figure set {stems!r}")
    leaked = [
        s["file"]
        for f in figures
        for s in f["sources"]
        if bpf.DATA_EFFICIENCY_DIR in str(s["file"])
    ]
    if leaked:
        raise RuntimeError(
            f"v1 bundle paths leaked into v2 figure provenance: {leaked}"
        )

    _write_json(
        out_dir / "manifest.json",
        build_figures_manifest(
            figures, aggregate, bundle_dir=bundle_dir, make_pdf=make_pdf, dpi=dpi
        ),
    )
    _write(
        out_dir / "README.md",
        render_figures_readme(figures, aggregate, out_dir=out_dir, dpi=dpi),
    )
    return figures


def build_figures_manifest(
    figures: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    bundle_dir: str,
    make_pdf: bool,
    dpi: int,
) -> dict[str, Any]:
    """Provenance for the v2 data-efficiency figure set.

    The corpus claim, the inherited-point verification and the bundle's warnings are
    **carried, not restated**: the label sits next to the evidence that earned it.
    """
    return {
        "study": STUDY,
        "issue": 126,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "corpus": "v2 (accent marks corrected)",
        "corpus_claim_basis": (
            "Not asserted by this builder: the figure is gated on a HEALTHY "
            "corpus_attestation on EVERY point (below) — every present cell attested from "
            "its own launcher's immutable Slurm .out header, one corpus across the "
            "join, and that corpus validated against the expected v2 dataset dirs and "
            "data_source. Without that the figure is refused."
        ),
        "corpus_attestation": aggregate.get("corpus_attestation"),
        "inherited_point_basis": (
            "The 100% point is the v2 main grid's published result, not a re-run. The "
            "figure is gated on maingrid_consistency.ok, i.e. this join's inherited "
            "per-ratio means matched the committed paclic-v2-maingrid/aggregate.json "
            "for every shared ratio and curve metric."
        ),
        "maingrid_consistency": aggregate.get("maingrid_consistency"),
        "analysis_only": True,
        "no_training_no_inference": True,
        "plot_only_committed_numbers": True,
        "status": aggregate["status"],
        "complete": aggregate["complete"],
        "publishable": aggregate.get("publishable"),
        "warnings": aggregate.get("warnings") or [],
        "hard_warnings": aggregate.get("hard_warnings") or [],
        "axis": aggregate["axis"],
        "coverage": aggregate["coverage"],
        "grid_integrity": aggregate.get("grid_integrity"),
        "subset_consistency": aggregate.get("subset_consistency"),
        "input_bundles": {
            "v2_data_efficiency": f"{bundle_dir}/aggregate.json",
            "v2_maingrid": f"{v2mg.DEFAULT_OUT_DIR}/aggregate.json",
        },
        "renders_with": (
            "scripts.evaluation.build_paclic_paper_figures.fig_data_efficiency "
            "(rv_ratios and suptitle overridden for v2)"
        ),
        "caption_note": (
            "The v1 figure caption and suptitle assert v1 findings; the v2 records "
            "here describe only what is plotted, and say outright that four points at "
            "n=3 do not establish a curve shape — the joining line is a guide to the "
            "eye (see V2_FIGURE_SHOWS, V2_FIGURE_SUPTITLE, aggregate.json shape)."
        ),
        "not_rendered": {
            "reason": (
                "The v1 study figure set also carried a minimum-target-notation-ratio "
                "reading over five ratios; that rule is not reproduced on three ratios "
                "(aggregate.json minimum_target_ratio_note), so no figure of it exists "
                "here. The per-ratio grid and retention/adaptation figures belong to "
                "the v2 main-grid bundle, not to this one."
            ),
            "stems": [],
        },
        "rendering": {
            "backend": "matplotlib Agg",
            "png_dpi": dpi,
            "pdf": "vector" if make_pdf else "skipped",
            "deterministic": (
                "no RNG, no timestamp; PDF CreationDate cleared and PNG Software "
                "pinned so figures regenerate byte-identically"
            ),
            "palette": "inherited from scripts.evaluation.build_paclic_paper_figures",
        },
        "figures": [
            {
                "stem": f["stem"],
                "title": f["title"],
                "shows": f["shows"],
                "files": f["files"],
                "sources": f["sources"],
            }
            for f in figures
        ],
        "generated_by": "scripts.evaluation.build_paclic_v2_data_efficiency_analysis",
    }


def render_figures_readme(
    figures: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    out_dir: Path,
    dpi: int,
) -> str:
    """README for the figure dir — status, provenance, and the exact command used.

    ``out_dir`` is the dir actually written, so the regenerate command in the README
    reproduces *this* README rather than naming the module default.
    """
    cov = aggregate["coverage"]
    att = aggregate.get("corpus_attestation") or {}
    mg = aggregate.get("maingrid_consistency") or {}
    lines = [
        f"# PACLIC 2026 figures — v2 data efficiency ({len(TOTALS)}-point curve, "
        "corrected corpus)",
        "",
        f"**Status:** {aggregate['status']} · **Cells:** "
        f"{cov['n_present']}/{cov['n_cells']} "
        f"({cov['n_new_present']}/{cov['n_new_cells']} newly trained at "
        f"totals {list(TRAINED_TOTALS)}) · **Seeds per point:** `{list(SEEDS)}`.",
        "",
        f"**Corpus attested:** {att.get('n_attested')}/"
        f"{att.get('n_present_cells')} present cells, from each arm's own immutable "
        f"Slurm `.out` headers; corpora `{att.get('corpora')}`, data dirs "
        f"`{att.get('data_dirs')}`, validated against `{att.get('expected_corpus')}`. "
        "This figure is refused unless that attestation is healthy at every point — the "
        "v2 label is never asserted by the figure builder itself, because "
        "`results.json` records no corpus path.",
        "",
        f"**Inherited 100% point verified:** `{mg.get('ok')}` over "
        f"`{mg.get('n_comparisons')}` comparison(s) against "
        f"`{v2mg.DEFAULT_OUT_DIR}/aggregate.json`. The right-hand point of every "
        "series is the v2 main grid's published result, reused by reference and never "
        "re-run.",
        "",
    ]
    warnings = aggregate.get("warnings") or []
    if warnings:
        lines.append(
            "> **Warnings carried from the data bundle** (none of them "
            "publication-blocking, or this figure would not exist):"
        )
        lines.extend(f"> - {w}" for w in warnings)
        lines.append("")
    lines += [
        "Data-viz only — no training or inference. Every value is read from the v2 "
        "data-efficiency bundle's `aggregate.json`; the figure code, palette and "
        "determinism are reused from `scripts.evaluation.build_paclic_paper_figures`, "
        "with the ratio list and the suptitle overridden so no v1 conclusion is burned "
        "into a v2 render. Regenerate:",
        "",
        "```bash",
        "uv run python -m scripts.evaluation.build_paclic_v2_data_efficiency_analysis \\",
        f"  --out-dir {DEFAULT_OUT_DIR} \\",
        f"  --figures-out-dir {out_dir}",
        "```",
        "",
        f"Each figure is a ≥{dpi}-dpi PNG plus a vector PDF (matplotlib Agg, "
        "deterministic). Machine-readable provenance: `manifest.json`.",
        "",
        "| # | figure | shows | source data |",
        "| :-- | :--- | :--- | :--- |",
    ]
    for i, fig in enumerate(figures, start=1):
        files = ", ".join(f"`{name}`" for name in fig["files"])
        srcs = "; ".join(f"`{s['file']}`" for s in fig["sources"])
        lines.append(f"| {i} | {files} | {fig['shows']} | {srcs} |")
    lines.append("")
    shape = aggregate.get("shape") or {}
    lines.append(
        "> **The joining line is a guide to the eye, not a fitted shape.** Four points "
        "at n=3 cannot separate one step from another, so this figure supports 'the "
        "curve was measured at 10/25/50/100%' and not 'the gain is front-loaded' or 'it "
        f"saturates by X%'. Verdict: `{shape.get('verdict')}` "
        f"(determined=`{shape.get('determined')}`), with the per-mix step rankings and "
        "their order stability in the bundle's `tables/step_separability.csv`. The v1 "
        "study's 2605 / 5210 cells were trained on the abandoned corpus and are not "
        "plotted, joined or subtracted here."
    )
    lines.append("")
    lines.append(
        '> These figures use the corrected v2 corpus and the released numeric tables.'
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--low-study-root",
        type=Path,
        default=DEFAULT_LOW_STUDY_ROOT,
        help=(
            f"Study root of the {TARGET_N_NEW_CELLS} newly trained cells — ALL of the "
            f"trained totals {list(TRAINED_TOTALS)} live in this one root, told apart by "
            "the n<total> token in each run name."
        ),
    )
    parser.add_argument(
        "--full-study-root",
        type=Path,
        default=DEFAULT_FULL_STUDY_ROOT,
        help=(
            f"Study root of the v2 MAIN GRID, whose total_train_samples={FULL_TOTAL} "
            "cells are inherited by reference as the 100% point (never re-run, never "
            "written to)."
        ),
    )
    parser.add_argument(
        "--shared-low-study-root",
        default=DEFAULT_SHARED_LOW_STUDY_ROOT,
        help="Optional path recorded in provenance for the newly trained cells.",
    )
    parser.add_argument(
        "--shared-full-study-root",
        default=DEFAULT_SHARED_FULL_STUDY_ROOT,
        help="Optional path recorded in provenance for the inherited cells.",
    )
    parser.add_argument(
        "--maingrid-aggregate",
        type=Path,
        default=DEFAULT_MAINGRID_AGGREGATE,
        help=(
            "Committed v2 main-grid aggregate.json. The inherited 100% point is "
            "verified against it; absent/unreadable → the inherited arm is reported "
            "UNVERIFIED (a hard failure), never assumed."
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
    parser.add_argument(
        "--slurm-log-dir",
        type=Path,
        default=DEFAULT_SLURM_LOG_DIR,
        help=(
            "Dir of immutable Slurm .out logs. Each arm is attested from its own glob "
            f"({DATAEFF_SLURM_LOG_GLOB} for the trained totals / "
            f"{SLURM_LOG_GLOB_BY_TOTAL[FULL_TOTAL]} for the inherited one). Absent dir → "
            "every cell 'unattested' (never assumed)."
        ),
    )
    parser.add_argument(
        "--no-slurm-attestation",
        action="store_true",
        help="Skip the Slurm-log corpus attestation entirely.",
    )
    parser.add_argument(
        "--no-cross-check",
        action="store_true",
        help="Skip re-evaluating every view against results.json.",
    )
    parser.add_argument(
        "--figures-out-dir",
        type=Path,
        default=None,
        help=(
            "Render the v2 data-efficiency figure here. Requires --out-dir (the figure "
            "cites the bundle as its source). Refused, with a note, while the curve is "
            "incomplete, either arm's attestation unhealthy, the inherited point "
            "unverified, any publication-blocking warning standing, or the dir already "
            f"belonging to another study. Conventional: {DEFAULT_FIGURES_OUT_DIR}"
        ),
    )
    parser.add_argument(
        "--no-pdf", action="store_true", help="Skip the vector PDF copies."
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (>=300 for publication)."
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # The figure's own manifest/README cite the bundle as their source, so rendering
    # without writing that bundle publishes a pointer at a path this run never made.
    if args.figures_out_dir is not None and args.out_dir is None:
        print(
            f"[{STUDY}][fatal] --figures-out-dir requires --out-dir: the figure "
            "manifest and README cite the bundle's aggregate.json as their data "
            "source, and without --out-dir no bundle is written, so they would cite a "
            "path this run never produced.",
            file=sys.stderr,
        )
        return 2

    curve, checks = load_curve(
        args.low_study_root,
        args.full_study_root,
        cross_check=not args.no_cross_check,
        shared_low_study_root=args.shared_low_study_root,
        shared_full_study_root=args.shared_full_study_root,
    )
    coverage = coverage_summary(curve)
    per_total = aggregate_curve(curve)
    subset = subset_consistency(
        subset_provenance_by_total(
            curve,
            low_study_root=args.low_study_root,
            full_study_root=args.full_study_root,
        ),
        curve,
        nesting=subset_nesting(
            curve,
            low_study_root=args.low_study_root,
            full_study_root=args.full_study_root,
        ),
    )
    integrity = grid_integrity(
        curve,
        low_study_root=args.low_study_root,
        full_study_root=args.full_study_root,
    )
    attestation = (
        None
        if args.no_slurm_attestation
        else corpus_attestation(
            curve,
            log_dir=args.slurm_log_dir,
            low_study_root=args.low_study_root,
            full_study_root=args.full_study_root,
        )
    )
    maingrid = maingrid_consistency(
        per_total, load_maingrid_aggregate(args.maingrid_aggregate)
    )

    warnings = audit_warnings(
        curve,
        coverage,
        subset=subset,
        attestation=attestation,
        integrity=integrity,
        maingrid=maingrid,
    )
    for warning in warnings:
        print(f"[{STUDY}][warn] {warning}", file=sys.stderr)
    # Computed BEFORE the figures: a hard failure must not leave a publishable figure
    # bundle on disk just because the exit code came last.
    hard = hard_warnings(warnings)

    if args.out_dir is not None:
        aggregate = build_bundle(
            curve,
            checks,
            args.out_dir,
            low_study_root=args.low_study_root,
            full_study_root=args.full_study_root,
            shared_low_study_root=args.shared_low_study_root,
            shared_full_study_root=args.shared_full_study_root,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
            maingrid=maingrid,
        )
    else:
        aggregate = build_aggregate(
            curve,
            per_total,
            checks,
            coverage,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
            maingrid=maingrid,
        )
        print(
            render_results_md(
                curve,
                per_total,
                curve_deltas(curve),
                checks,
                coverage,
                subset=subset,
                attestation=attestation,
                integrity=integrity,
                maingrid=maingrid,
            )
        )

    if args.figures_out_dir is not None:
        if hard:
            print(
                f"[{STUDY}][warn] refusing to render the v2 data-efficiency figure: "
                f"this run is a hard failure ({len(hard)} publication-blocking "
                f"warning(s)), so no figure bundle is written. First: {hard[0]}",
                file=sys.stderr,
            )
        else:
            try:
                figures = build_figures(
                    aggregate,
                    args.figures_out_dir,
                    bundle_dir=str(args.out_dir),
                    make_pdf=not args.no_pdf,
                    dpi=args.dpi,
                )
            except FigureGateError as exc:
                print(f"[{STUDY}][warn] {exc}", file=sys.stderr)
            else:
                print(
                    f"[{STUDY}] {len(figures)} figures -> {args.figures_out_dir}",
                    file=sys.stderr,
                )

    # Nonzero while the curve is incomplete or an invariant is broken, so a caller can
    # gate publication on the exit status alone. Same predicate the figure gate uses.
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
