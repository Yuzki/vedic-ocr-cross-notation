"""Aggregate accent ratio multiseed: PACLIC 2026 v2 release."""

from __future__ import annotations

STEPS_PER_EPOCH = 326


import argparse


import ast


import csv


import io


import json


import math


import re


import statistics


import sys


from collections.abc import Callable, Iterable, Iterator, Sequence


from dataclasses import dataclass, field


from pathlib import Path


from typing import Any, Optional


RV_RATIOS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)


SEEDS: tuple[int, ...] = (42, 43, 44)


_RUN_LOG_BASENAME: dict[str, str] = {}


VIEW_METRICS: tuple[str, ...] = (
    "cer",
    "wer",
    "chrf",
    "exact_match",
    "full_cer",
    "accent_stripped_cer",
    "accent_precision",
    "accent_recall",
    "accent_f1",
)


_T95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


Z95: float = 1.959963984540054


def t_critical_95(df: int) -> Optional[float]:
    """Two-sided 95% critical value for ``df`` degrees of freedom (None if df<1)."""
    if df < 1:
        return None
    return _T95.get(df, Z95)


@dataclass(frozen=True)
class SeedCell:
    """One (mix ratio, seed) cell, loaded from its ``results.json``.

    ``present=False`` records a cell whose ``results.json`` is absent so a
    partially-finished grid still aggregates with the gaps reported. Present
    cells carry the two per-view metric dicts (keys :data:`VIEW_METRICS`) plus
    the top-level gap metrics and recovered training state / provenance.
    """

    rv_ratio: float
    seed: int
    run_name: str
    source: str  # "multiseed" | "prior-study-reference"
    present: bool
    results_path: Optional[str] = None
    shared_path: Optional[str] = None
    rv: dict[str, Optional[float]] = field(default_factory=dict)
    ms: dict[str, Optional[float]] = field(default_factory=dict)
    avg_cer: Optional[float] = None
    cer_gap: Optional[float] = None
    accent_f1_gap: Optional[float] = None
    accent_stripped_cer_gap: Optional[float] = None
    rv_chrf_source: Optional[str] = None
    ms_chrf_source: Optional[str] = None
    rv_exact_match_source: Optional[str] = None
    ms_exact_match_source: Optional[str] = None
    training_state: Optional[dict[str, Any]] = None
    training_state_source: str = "unavailable"
    training_state_confidence: Optional[str] = None
    training_state_recovery_method: Optional[str] = None
    model_base: Optional[str] = None
    freeze_mode: Optional[str] = None

    @property
    def ms_ratio(self) -> float:
        return round(1.0 - self.rv_ratio, 2)


@dataclass(frozen=True)
class Stat:
    """A metric summarized across seeds: mean, sample SD, and 95% t-CI."""

    n: int
    values: tuple[float, ...]
    mean: Optional[float]
    sd: Optional[float]
    sem: Optional[float]
    ci95_lo: Optional[float]
    ci95_hi: Optional[float]
    ci95_halfwidth: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "values": list(self.values),
            "mean": self.mean,
            "sd": self.sd,
            "sem": self.sem,
            "ci95_lo": self.ci95_lo,
            "ci95_hi": self.ci95_hi,
            "ci95_halfwidth": self.ci95_halfwidth,
        }


def summarize(values: Iterable[Optional[float]]) -> Stat:
    """Mean / sample-SD / 95% Student-t CI over the non-``None`` values.

    ``n < 2`` cannot estimate a spread, so SD/SEM/CI are ``None`` and the mean
    (if any) still reports; ``n == 0`` is an all-``None`` metric.
    """
    vals = tuple(float(v) for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return Stat(0, (), None, None, None, None, None, None)
    mean = statistics.fmean(vals)
    if n < 2:
        return Stat(n, vals, mean, None, None, None, None, None)
    sd = statistics.stdev(vals)
    sem = sd / math.sqrt(n)
    t = t_critical_95(n - 1)
    hw = t * sem if t is not None else None
    lo = mean - hw if hw is not None else None
    hi = mean + hw if hw is not None else None
    return Stat(n, vals, mean, sd, sem, lo, hi, hw)


def _training_state_from_results(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    ts = data.get("training_state")
    if not isinstance(ts, dict):
        return None
    return {
        "best_epoch": ts.get("best_epoch"),
        "best_step": ts.get("best_step"),
        "global_step": ts.get("global_step"),
        "source": "results.json",
        "confidence": "high",
        "recovery_method": "results_json",
    }


def _training_state_from_checkpoints(cell_dir: Path) -> Optional[dict[str, Any]]:
    """Recover best epoch / optimizer steps from a surviving ``trainer_state.json``.

    HF Trainer writes ``trainer_state.json`` in every checkpoint dir; the
    highest-step one records the final ``best_model_checkpoint`` decision and the
    full ``log_history`` from which the best epoch is read. Used for the seed-42
    cells, whose previous ``results.json`` predate the inline ``training_state``.
    """
    ck_root = cell_dir / "checkpoints"
    if not ck_root.is_dir():
        return None
    states = sorted(
        ck_root.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-", 1)[1]),
    )
    if not states:
        return None
    state_path = states[-1]
    try:
        ts = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    best_ckpt = ts.get("best_model_checkpoint")
    best_step = int(best_ckpt.rsplit("-", 1)[1]) if best_ckpt else None
    best_epoch = None
    for entry in ts.get("log_history", []):
        if entry.get("step") == best_step and "epoch" in entry:
            best_epoch = entry["epoch"]
            break
    return {
        "best_epoch": best_epoch,
        "best_step": best_step,
        "global_step": ts.get("global_step"),
        "source": f"checkpoints ({state_path.parent.name}/trainer_state.json)",
        "confidence": "high",
        "recovery_method": "checkpoint_trainer_state",
    }


_METRIC_FOR_BEST_RE = re.compile(r"metric_for_best_model:\s*(\S+)")


def parse_metric_for_best_model(text: str) -> Optional[str]:
    """The configured ``metric_for_best_model`` from a Slurm ``.out`` header.

    The runner prints ``Eval targets: [...] | metric_for_best_model: eval_rv_cer``
    once at startup. The two seed-42 target cells use ``eval_rv_cer``
    (``rv_75_ms_25``) and ``eval_ms_cer`` (``rv_25_ms_75``) — **not**
    ``eval_avg_cer`` — so the selection metric is read from the log, never assumed.
    """
    m = _METRIC_FOR_BEST_RE.search(text)
    return m.group(1) if m else None


def _iter_trainer_dicts(text: str) -> Iterator[dict[str, Any]]:
    """Yield each log line that is a ``{...}`` Python dict literal.

    HF Trainer prints each per-epoch eval line and the closing ``train_runtime``
    summary as a dict literal; ``ast.literal_eval`` over the ``{`` … ``}`` span
    parses it structurally (no brittle per-float regex). Lines whose brace span is
    not a literal dict are skipped.
    """
    for line in text.splitlines():
        lo = line.find("{")
        hi = line.rfind("}")
        if lo == -1 or hi <= lo:
            continue
        try:
            parsed = ast.literal_eval(line[lo : hi + 1])
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            yield parsed


def parse_slurm_eval_trajectory(
    text: str, metric: str
) -> tuple[list[tuple[float, float]], Optional[float]]:
    """``([(epoch, metric_value), …], final_epoch)`` for ``metric`` from a log.

    ``metric`` is the configured ``metric_for_best_model`` (e.g. ``eval_rv_cer``),
    which appears once per epoch in that view's eval dict; the trajectory keeps the
    per-epoch validation values in log order. ``final_epoch`` is the epoch of the
    closing ``{'train_runtime': …, 'epoch': N.0}`` summary — the realized training
    length — or ``None`` if the run never reached it.
    """
    trajectory: list[tuple[float, float]] = []
    final_epoch: Optional[float] = None
    for d in _iter_trainer_dicts(text):
        if "train_runtime" in d and "epoch" in d:
            final_epoch = float(d["epoch"])
        elif metric in d and "epoch" in d:
            trajectory.append((float(d["epoch"]), float(d[metric])))
    return trajectory, final_epoch


def _training_state_from_slurm_log(
    run_name: str, slurm_log_dir: Optional[Path]
) -> Optional[dict[str, Any]]:
    """Recover best epoch / optimizer steps from the immutable Slurm ``.out`` (study).

    Last-resort recovery for a seed-42 cell whose previous ``results.json`` has no
    ``training_state`` **and** whose ``checkpoints/`` were purged (no surviving
    ``trainer_state.json``) — the ``rv_75_ms_25`` / ``rv_25_ms_75`` gap. Reads the
    configured ``metric_for_best_model`` from the log header, then selects the
    earliest epoch achieving the minimum logged validation metric: exactly HF
    Trainer's ``load_best_model_at_end`` decision under ``greater_is_better=False``
    (CER), the same read the study OCR recovery makes. ``best_step`` /
    ``global_step`` use the fixed PACLIC step contract (``STEPS_PER_EPOCH`` optimizer
    steps per epoch, imported from the study ledger). ``best_metric`` is the
    *validation* CER at the selected epoch — distinct from the saved-prediction
    *test* CER. Returns ``None`` (→ unavailable) if the log is absent or carries no
    usable trajectory; the mapping is only defined for the registered runs, so an
    unrelated ``run_name`` also yields ``None``.
    """
    if slurm_log_dir is None:
        return None
    log_basename = _RUN_LOG_BASENAME.get(run_name)
    if log_basename is None:
        return None
    log_path = slurm_log_dir / f"{log_basename}.out"
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="replace")
    metric = parse_metric_for_best_model(text)
    if metric is None:
        return None
    trajectory, final_epoch = parse_slurm_eval_trajectory(text, metric)
    if not trajectory:
        return None
    # HF load_best_model_at_end (greater_is_better=False) keeps the earliest epoch
    # achieving the minimum validation metric (strict-< update → earliest on ties).
    best_epoch_f, best_value = min(trajectory, key=lambda ev: (ev[1], ev[0]))
    best_epoch = int(round(best_epoch_f))
    best_step = best_epoch * STEPS_PER_EPOCH
    global_step = (
        int(round(final_epoch * STEPS_PER_EPOCH)) if final_epoch is not None else None
    )
    return {
        "best_epoch": float(best_epoch),
        "best_step": best_step,
        "global_step": global_step,
        "final_epoch": final_epoch,
        "best_metric_key": metric,
        "best_metric": best_value,
        "best_model_checkpoint": f"checkpoint-{best_step}",
        "source": (
            f"slurm-log eval trajectory (outputs/slurm/{log_basename}.out; "
            f"metric={metric})"
        ),
        "confidence": "high",
        "recovery_method": "slurm_eval_trajectory",
    }


@dataclass(frozen=True)
class SelectionCandidate:
    """One epoch's paired RV/MS validation CER — a checkpoint-selection candidate.

    Built from an HF-Trainer per-epoch eval record (a Slurm ``.out`` dict line or a
    checkpoint ``trainer_state.json`` ``log_history`` entry). ``mean_val_cer`` and
    ``worst_val_cer`` are the two uniform selection objectives the study studies —
    the unweighted mean and the max of the two view CERs. ``step`` is the epoch's
    optimizer step under the fixed PACLIC contract (``round(epoch *
    STEPS_PER_EPOCH)``), so the candidate names ``checkpoint-<step>``.
    """

    epoch: int
    step: int
    rv_val_cer: float
    ms_val_cer: float

    @property
    def mean_val_cer(self) -> float:
        """Unweighted mean of the two view CERs (== HF Trainer's ``eval_avg_cer``)."""
        return (self.rv_val_cer + self.ms_val_cer) / 2.0

    @property
    def worst_val_cer(self) -> float:
        """The worse (larger) of the two view CERs."""
        return max(self.rv_val_cer, self.ms_val_cer)

    def value(self, field: str) -> float:
        """The candidate's value on one of :data:`CANDIDATE_VALUE_FIELDS`."""
        if field not in CANDIDATE_VALUE_FIELDS:
            raise ValueError(f"unknown selection field {field!r}")
        return float(getattr(self, field))

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "rv_val_cer": self.rv_val_cer,
            "ms_val_cer": self.ms_val_cer,
            "mean_val_cer": self.mean_val_cer,
            "worst_val_cer": self.worst_val_cer,
        }


CANDIDATE_VALUE_FIELDS: tuple[str, ...] = (
    "rv_val_cer",
    "ms_val_cer",
    "mean_val_cer",
    "worst_val_cer",
)


METRIC_TO_CANDIDATE_FIELD: dict[str, str] = {
    "eval_rv_cer": "rv_val_cer",
    "eval_ms_cer": "ms_val_cer",
    "eval_avg_cer": "mean_val_cer",
}


def paired_candidates_from_trainer_dicts(
    dicts: Iterable[dict[str, Any]], *, steps_per_epoch: int = STEPS_PER_EPOCH
) -> list[SelectionCandidate]:
    """Per-epoch paired RV/MS validation CER → selection candidates.

    HF Trainer logs the RV-view and MS-view eval dicts separately (each carries its
    own ``epoch``); this keys them by epoch and keeps only epochs carrying *both*
    ``eval_rv_cer`` and ``eval_ms_cer`` (a candidate needs both views to score the
    mean/worst criteria). Source-agnostic: feed it the Slurm ``.out`` dict lines
    (via :func:`_iter_trainer_dicts`) or a checkpoint ``trainer_state.json``
    ``log_history``. Candidates come back in ascending epoch order with
    ``step = round(epoch * steps_per_epoch)``.
    """
    rv: dict[float, float] = {}
    ms: dict[float, float] = {}
    for d in dicts:
        if "epoch" not in d:
            continue
        epoch = float(d["epoch"])
        if "eval_rv_cer" in d:
            rv[epoch] = float(d["eval_rv_cer"])
        if "eval_ms_cer" in d:
            ms[epoch] = float(d["eval_ms_cer"])
    return [
        SelectionCandidate(
            epoch=int(round(epoch)),
            step=round(epoch * steps_per_epoch),
            rv_val_cer=rv[epoch],
            ms_val_cer=ms[epoch],
        )
        for epoch in sorted(rv.keys() & ms.keys())
    ]


def parse_paired_validation_trajectory(
    text: str, *, steps_per_epoch: int = STEPS_PER_EPOCH
) -> list[SelectionCandidate]:
    """Paired RV/MS validation-CER trajectory from a Slurm ``.out`` .

    The study counterpart to :func:`parse_slurm_eval_trajectory` (which keeps a
    single configured metric): parses every trainer dict line via
    :func:`_iter_trainer_dicts` and pairs ``eval_rv_cer`` with ``eval_ms_cer`` per
    epoch, so both uniform criteria (mean, worst) score from one structural pass.
    """
    return paired_candidates_from_trainer_dicts(
        _iter_trainer_dicts(text), steps_per_epoch=steps_per_epoch
    )


def select_earliest_min(
    candidates: Sequence[SelectionCandidate], field: str
) -> Optional[SelectionCandidate]:
    """The earliest-epoch candidate minimizing ``field`` (HF best-model rule).

    ``field`` is one of :data:`CANDIDATE_VALUE_FIELDS`. Ties on the value break to
    the earliest epoch — exactly HF Trainer's ``load_best_model_at_end`` under
    ``greater_is_better=False`` (a strict-``<`` update keeps the first minimizer).
    ``None`` for an empty trajectory.
    """
    if field not in CANDIDATE_VALUE_FIELDS:
        raise ValueError(f"unknown selection field {field!r}")
    if not candidates:
        return None
    return min(candidates, key=lambda c: (c.value(field), c.epoch))


STATUS_COMPUTABLE = "computable_from_saved_predictions"


STATUS_DEFERRED = "requires_reinference_deferred_post_126"


@dataclass(frozen=True)
class SelectionDecision:
    """The checkpoint a criterion selects for one run, and whether it is scorable.

    ``criterion`` is ``current`` (the run's configured mixed rule) or a uniform
    study objective (``mean_val_cer`` / ``worst_val_cer``). The selected epoch/step
    names ``checkpoint-<step>``; the ``*_at_selected`` fields snapshot all four
    view/summary validation CERs there. ``matches_current_saved_predictions`` is
    whether this epoch equals the run's current-rule best epoch — the epoch the
    committed ``{rv,ms}_predictions.csv`` were decoded from. If so the cell's test
    metrics recompute inference-free (:data:`STATUS_COMPUTABLE`); otherwise scoring
    it needs a fresh decode (:data:`STATUS_DEFERRED`), never a silent fallback to
    the current checkpoint.
    """

    criterion: str
    selected_epoch: int
    selected_step: int
    selected_value: float
    rv_val_cer_at_selected: float
    ms_val_cer_at_selected: float
    mean_val_cer_at_selected: float
    worst_val_cer_at_selected: float
    matches_current_saved_predictions: bool
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "selected_epoch": self.selected_epoch,
            "selected_step": self.selected_step,
            "selected_value": self.selected_value,
            "rv_val_cer_at_selected": self.rv_val_cer_at_selected,
            "ms_val_cer_at_selected": self.ms_val_cer_at_selected,
            "mean_val_cer_at_selected": self.mean_val_cer_at_selected,
            "worst_val_cer_at_selected": self.worst_val_cer_at_selected,
            "matches_current_saved_predictions": (
                self.matches_current_saved_predictions
            ),
            "status": self.status,
        }


def build_selection_decision(
    criterion: str,
    field: str,
    candidate: SelectionCandidate,
    *,
    current_selected_epoch: int,
) -> SelectionDecision:
    """Assemble a :class:`SelectionDecision` for ``criterion`` from its candidate.

    ``field`` is the :data:`CANDIDATE_VALUE_FIELDS` objective ``criterion`` scores
    on (``current`` passes the run's mapped current field). ``matches`` /
    ``status`` follow from whether the candidate's epoch equals the run's
    current-rule best epoch.
    """
    matches = candidate.epoch == current_selected_epoch
    return SelectionDecision(
        criterion=criterion,
        selected_epoch=candidate.epoch,
        selected_step=candidate.step,
        selected_value=candidate.value(field),
        rv_val_cer_at_selected=candidate.rv_val_cer,
        ms_val_cer_at_selected=candidate.ms_val_cer,
        mean_val_cer_at_selected=candidate.mean_val_cer,
        worst_val_cer_at_selected=candidate.worst_val_cer,
        matches_current_saved_predictions=matches,
        status=STATUS_COMPUTABLE if matches else STATUS_DEFERRED,
    )


MetricsFn = Callable[[Path], dict[str, Any]]


def default_metrics_fn(csv_path: Path) -> dict[str, Any]:
    """The real study evaluator over a saved prediction CSV (inference-free)."""
    from .prediction_metrics import evaluate_prediction_file

    return evaluate_prediction_file(csv_path)


class _MetricsCache:
    """Compute each prediction CSV's study payload at most once."""

    def __init__(self, fn: MetricsFn) -> None:
        self._fn = fn
        self._cache: dict[Path, dict[str, Any]] = {}

    def get(self, csv_path: Path) -> dict[str, Any]:
        if csv_path not in self._cache:
            self._cache[csv_path] = self._fn(csv_path)
        return self._cache[csv_path]


def _extract_view(block: dict[str, Any]) -> dict[str, Optional[float]]:
    return {key: block.get(key) for key in VIEW_METRICS}


def read_cell_dir(
    cell_dir: Path,
    *,
    rv_ratio: float,
    seed: int,
    run_name: str,
    source: str,
    shared_path: Optional[str],
    metrics: _MetricsCache,
    cross_check: bool,
    slurm_log_dir: Optional[Path] = None,
) -> tuple[SeedCell, list[dict[str, Any]]]:
    """Read one cell directory (``results.json`` + optional study chrf/cross-check).

    The single ``results.json`` → :class:`SeedCell` reader behind both
    :func:`load_cell` (the study factorial/multiseed routing) and the study
    data-efficiency loader, so there is exactly one cell reader and one chrf-fill /
    cross-check path. An absent ``results.json`` yields a ``present=False`` cell so
    a partial grid still aggregates with the gap reported.
    """
    results_path = cell_dir / "results.json"

    if not results_path.is_file():
        return (
            SeedCell(
                rv_ratio=rv_ratio,
                seed=seed,
                run_name=run_name,
                source=source,
                present=False,
                shared_path=shared_path,
            ),
            [],
        )

    data = json.loads(results_path.read_text(encoding="utf-8"))
    rv = _extract_view(data.get("rv_test") or {})
    ms = _extract_view(data.get("ms_test") or {})

    checks: list[dict[str, Any]] = []
    rv_chrf_source = "results.json" if rv.get("chrf") is not None else None
    ms_chrf_source = "results.json" if ms.get("chrf") is not None else None
    # exact-match is *always* re-sourced from the committed predictions (study), so
    # the reported value regenerates from saved predictions alone. This means the
    # prediction CSV is re-evaluated whenever it exists (not only for the chrF fill
    # or the cross-check); the runner's results.json exact-match is kept aside per
    # view to feed the delta_exact_match cross-check.
    exact_source: dict[str, Optional[str]] = {"rv": None, "ms": None}
    for view_name, view, csv_name in (
        ("rv", rv, "rv_predictions.csv"),
        ("ms", ms, "ms_predictions.csv"),
    ):
        need_chrf = view.get("chrf") is None
        runner_exact = view.get("exact_match")
        # Clear the results.json-derived exact-match; only a prediction recompute
        # sets it, so a cell whose prediction CSV is absent reports it as a gap.
        view["exact_match"] = None
        view["exact_match_count"] = None
        csv_path = cell_dir / csv_name
        if not csv_path.is_file():
            continue
        payload = metrics.get(csv_path)
        view["exact_match"] = payload.get("exact_match")
        view["exact_match_count"] = payload.get("exact_match_count")
        exact_source[view_name] = "recomputed-predictions"
        if need_chrf:
            view["chrf"] = payload.get("chrf")
            if view_name == "rv":
                rv_chrf_source = "recomputed-predictions"
            else:
                ms_chrf_source = "recomputed-predictions"
        if cross_check:
            checks.append(
                _cross_check_record(
                    run_name,
                    seed,
                    view_name,
                    view,
                    payload,
                    recomputed_chrf=need_chrf,
                    runner_exact_match=runner_exact,
                )
            )

    # Fallback order (study): inline results.json training_state → a surviving
    # checkpoint trainer_state.json → the immutable Slurm .out eval trajectory →
    # unavailable. Seeds 43/44 stop at the first; the 3 seed-42 cells with surviving
    # checkpoints stop at the second; the 2 purged seed-42 cells (rv_75_ms_25,
    # rv_25_ms_75) reach the Slurm-log recovery.
    ts = (
        _training_state_from_results(data)
        or _training_state_from_checkpoints(cell_dir)
        or _training_state_from_slurm_log(run_name, slurm_log_dir)
    )

    cell = SeedCell(
        rv_ratio=rv_ratio,
        seed=data.get("seed", seed) if data.get("seed") is not None else seed,
        run_name=run_name,
        source=source,
        present=True,
        results_path=str(results_path),
        shared_path=shared_path,
        rv=rv,
        ms=ms,
        avg_cer=data.get("avg_cer"),
        cer_gap=data.get("cer_gap"),
        accent_f1_gap=data.get("accent_f1_gap"),
        accent_stripped_cer_gap=data.get("accent_stripped_cer_gap"),
        rv_chrf_source=rv_chrf_source,
        ms_chrf_source=ms_chrf_source,
        rv_exact_match_source=exact_source["rv"],
        ms_exact_match_source=exact_source["ms"],
        training_state=ts,
        training_state_source=ts["source"]
        if ts
        else "unavailable (no results.json/checkpoint/Slurm-log training_state)",
        training_state_confidence=ts.get("confidence") if ts else None,
        training_state_recovery_method=ts.get("recovery_method") if ts else None,
        model_base=data.get("model_base"),
        freeze_mode=data.get("freeze_mode"),
    )
    return cell, checks


def _cross_check_record(
    run_name: str,
    seed: int,
    view: str,
    results_view: dict[str, Optional[float]],
    payload: dict[str, Any],
    *,
    recomputed_chrf: bool,
    runner_exact_match: Optional[float] = None,
) -> dict[str, Any]:
    """|Δ| between the runner ``results.json`` metrics and the study re-evaluation.

    chrF that had to be recomputed (seed 42) has no runner value to diff, so its
    delta is ``None`` — it is *sourced* from this evaluator, not cross-checked.
    exact-match is *always* sourced from the recompute (study), so ``results_view``
    no longer holds the runner value; it is passed in via ``runner_exact_match``
    (``None`` when the runner recorded none) to diff against the recompute.
    """
    deltas: dict[str, Optional[float]] = {}
    for key in ("cer", "wer", "chrf", "accent_f1"):
        runner = results_view.get(key)
        recomputed = payload.get(key)
        if key == "chrf" and recomputed_chrf:
            deltas[f"delta_{key}"] = None
            continue
        deltas[f"delta_{key}"] = (
            abs(runner - recomputed)
            if runner is not None and recomputed is not None
            else None
        )
    recomputed_exact = payload.get("exact_match")
    deltas["delta_exact_match"] = (
        abs(runner_exact_match - recomputed_exact)
        if runner_exact_match is not None and recomputed_exact is not None
        else None
    )
    return {"run_name": run_name, "seed": seed, "view": view, **deltas}


def _flat_metrics(cell: SeedCell) -> dict[str, Optional[float]]:
    """Flatten a cell to ``{rv_<m>, ms_<m>, avg_cer}`` for per-ratio summarizing."""
    flat: dict[str, Optional[float]] = {"avg_cer": cell.avg_cer}
    for key in VIEW_METRICS:
        flat[f"rv_{key}"] = cell.rv.get(key)
        flat[f"ms_{key}"] = cell.ms.get(key)
    return flat


AGG_METRICS: tuple[str, ...] = (
    "avg_cer",
    "rv_cer",
    "ms_cer",
    "rv_wer",
    "ms_wer",
    "rv_chrf",
    "ms_chrf",
    "rv_exact_match",
    "ms_exact_match",
    "rv_accent_precision",
    "rv_accent_recall",
    "rv_accent_f1",
    "ms_accent_precision",
    "ms_accent_recall",
    "ms_accent_f1",
    "rv_accent_stripped_cer",
    "ms_accent_stripped_cer",
)


def aggregate_by_ratio(cells: Sequence[SeedCell]) -> dict[float, dict[str, Stat]]:
    """Per mix ratio, summarize each metric across the present seeds."""
    by_ratio: dict[float, dict[str, Stat]] = {}
    for rv_ratio in RV_RATIOS:
        present = [c for c in cells if c.rv_ratio == rv_ratio and c.present]
        flats = [_flat_metrics(c) for c in present]
        by_ratio[rv_ratio] = {
            metric: summarize(f.get(metric) for f in flats) for metric in AGG_METRICS
        }
    return by_ratio


def _paired(
    cells: Sequence[SeedCell], rv_ratio: float, view: str, metric: str
) -> dict[int, float]:
    """``seed → metric`` for one arm, over present cells (for paired contrasts)."""
    out: dict[int, float] = {}
    for c in cells:
        if c.rv_ratio == rv_ratio and c.present:
            value = (c.rv if view == "rv" else c.ms).get(metric)
            if value is not None:
                out[c.seed] = value
    return out


def _contrast(a: dict[int, float], b: dict[int, float]) -> dict[str, Any]:
    """Paired contrast ``b − a`` over shared seeds: stats + per-seed sign check.

    ``a`` and ``b`` are ``seed → value`` maps for the two arms; the difference is
    taken only on seeds present in both, so the comparison is properly paired.
    """
    seeds = sorted(set(a) & set(b))
    diffs = [b[s] - a[s] for s in seeds]
    stat = summarize(diffs)
    positive = all(d > 0 for d in diffs) if diffs else False
    negative = all(d < 0 for d in diffs) if diffs else False
    return {
        "seeds": seeds,
        "a": summarize(a[s] for s in seeds).as_dict(),
        "b": summarize(b[s] for s in seeds).as_dict(),
        "per_seed_diff": {str(s): b[s] - a[s] for s in seeds},
        "diff": stat.as_dict(),
        "sign_consistent": bool(positive or negative),
        "direction": ("b>a" if positive else "b<a" if negative else "mixed"),
    }


def asymmetry(cells: Sequence[SeedCell]) -> dict[str, Any]:
    """RV↔MS asymmetries at the endpoints, paired by seed.

    * ``zero_shot`` — the cross-notation transfer question: the RV-only model
      (rv_ratio 1.0) read on **MS** test vs the MS-only model (rv_ratio 0.0) read
      on **RV** test. ``diff = MS→RV − RV→MS`` per seed, for CER (headline), chrF,
      and accent F1; ``direction b>a`` (MS→RV worse) means RV→OCRs better.
    * ``in_domain`` — is one notation intrinsically harder? Own-notation RV
      (rv_cer at rv_ratio 1.0) vs own-notation MS (ms_cer at rv_ratio 0.0).
    """
    zero_shot: dict[str, Any] = {}
    for metric in ("cer", "chrf", "accent_f1"):
        rv2ms = _paired(cells, 1.0, "ms", metric)  # RV-only model, MS test
        ms2rv = _paired(cells, 0.0, "rv", metric)  # MS-only model, RV test
        zero_shot[metric] = {
            "rv_to_ms": summarize(rv2ms[s] for s in sorted(rv2ms)).as_dict(),
            "ms_to_rv": summarize(ms2rv[s] for s in sorted(ms2rv)).as_dict(),
            **_contrast(rv2ms, ms2rv),
        }
    in_domain_rv = _paired(cells, 1.0, "rv", "cer")  # own-notation RV
    in_domain_ms = _paired(cells, 0.0, "ms", "cer")  # own-notation MS
    return {
        "zero_shot": zero_shot,
        "in_domain_cer": {
            "rv_own": summarize(
                in_domain_rv[s] for s in sorted(in_domain_rv)
            ).as_dict(),
            "ms_own": summarize(
                in_domain_ms[s] for s in sorted(in_domain_ms)
            ).as_dict(),
            **_contrast(in_domain_rv, in_domain_ms),
        },
    }


def retention_adaptation(cells: Sequence[SeedCell]) -> dict[str, Any]:
    """Retention + adaptation when a 25% minority notation is added, paired by seed.

    * RV-primary +25% MS (rv_ratio 0.75) vs RV-only (1.0): RV **retention** =
      Δ RV CER; MS **adaptation** = Δ MS accent F1.
    * MS-primary +25% RV (rv_ratio 0.25) vs MS-only (0.0): MS **retention** =
      Δ MS CER; RV **adaptation** = Δ RV accent F1.
    """
    return {
        "rv_primary_add_ms": {
            "rv_retention_cer": _contrast(
                _paired(cells, 1.0, "rv", "cer"), _paired(cells, 0.75, "rv", "cer")
            ),
            "ms_adaptation_accent_f1": _contrast(
                _paired(cells, 1.0, "ms", "accent_f1"),
                _paired(cells, 0.75, "ms", "accent_f1"),
            ),
        },
        "ms_primary_add_rv": {
            "ms_retention_cer": _contrast(
                _paired(cells, 0.0, "ms", "cer"), _paired(cells, 0.25, "ms", "cer")
            ),
            "rv_adaptation_accent_f1": _contrast(
                _paired(cells, 0.0, "rv", "accent_f1"),
                _paired(cells, 0.25, "rv", "accent_f1"),
            ),
        },
    }


def cross_check_summary(checks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Max / mean |Δ| between runner ``results.json`` and the study re-evaluation."""
    summary: dict[str, Any] = {"n_views_checked": len(checks)}
    for key in (
        "delta_cer",
        "delta_wer",
        "delta_chrf",
        "delta_accent_f1",
        "delta_exact_match",
    ):
        vals = [c[key] for c in checks if c.get(key) is not None]
        summary[key] = (
            {"max": max(vals), "mean": statistics.fmean(vals), "n": len(vals)}
            if vals
            else {"max": None, "mean": None, "n": 0}
        )
    return summary


def audit_warnings(cells: Sequence[SeedCell]) -> list[str]:
    """Integrity checks: missing cells, split foundation, unavailable state."""
    warnings: list[str] = []
    missing = [c.run_name for c in cells if not c.present]
    if missing:
        warnings.append(
            f"{len(missing)}/{len(cells)} cells missing results.json: "
            + ", ".join(missing)
        )
    bases = {c.model_base for c in cells if c.present and c.model_base}
    if len(bases) > 1:
        warnings.append(
            "multiple model.base across present cells (foundation not held fixed): "
            + ", ".join(sorted(bases))
        )
    freezes = {c.freeze_mode for c in cells if c.present and c.freeze_mode}
    if freezes - {"none"}:
        warnings.append(
            f"non-'none' freeze_mode among present cells: {sorted(freezes)}"
        )
    no_state = [
        c.run_name
        for c in cells
        if c.present and c.training_state_source.startswith("unavailable")
    ]
    if no_state:
        warnings.append(
            f"{len(no_state)} cell(s) without recoverable training_state "
            "(no results.json training_state, no surviving checkpoint "
            "trainer_state.json, no recoverable Slurm eval trajectory): "
            + ", ".join(no_state)
        )
    return warnings


PER_SEED_COLUMNS: tuple[str, ...] = (
    "rv_ratio",
    "ms_ratio",
    "seed",
    "run_name",
    "source",
    "present",
    "avg_cer",
    "rv_cer",
    "ms_cer",
    "rv_wer",
    "ms_wer",
    "rv_chrf",
    "ms_chrf",
    "rv_chrf_source",
    "ms_chrf_source",
    "rv_exact_match",
    "ms_exact_match",
    "rv_exact_match_count",
    "ms_exact_match_count",
    "rv_exact_match_source",
    "ms_exact_match_source",
    "rv_accent_precision",
    "rv_accent_recall",
    "rv_accent_f1",
    "ms_accent_precision",
    "ms_accent_recall",
    "ms_accent_f1",
    "rv_accent_stripped_cer",
    "ms_accent_stripped_cer",
    "best_epoch",
    "best_step",
    "global_step",
    "training_state_source",
    "training_state_confidence",
    "training_state_recovery_method",
    "model_base",
    "results_path",
)


def render_per_seed_csv(cells: Sequence[SeedCell]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(PER_SEED_COLUMNS))
    writer.writeheader()
    for c in cells:
        ts = c.training_state or {}
        writer.writerow(
            {
                "rv_ratio": c.rv_ratio,
                "ms_ratio": c.ms_ratio,
                "seed": c.seed,
                "run_name": c.run_name,
                "source": c.source,
                "present": c.present,
                "avg_cer": c.avg_cer,
                "rv_cer": c.rv.get("cer"),
                "ms_cer": c.ms.get("cer"),
                "rv_wer": c.rv.get("wer"),
                "ms_wer": c.ms.get("wer"),
                "rv_chrf": c.rv.get("chrf"),
                "ms_chrf": c.ms.get("chrf"),
                "rv_chrf_source": c.rv_chrf_source,
                "ms_chrf_source": c.ms_chrf_source,
                "rv_exact_match": c.rv.get("exact_match"),
                "ms_exact_match": c.ms.get("exact_match"),
                "rv_exact_match_count": c.rv.get("exact_match_count"),
                "ms_exact_match_count": c.ms.get("exact_match_count"),
                "rv_exact_match_source": c.rv_exact_match_source,
                "ms_exact_match_source": c.ms_exact_match_source,
                "rv_accent_precision": c.rv.get("accent_precision"),
                "rv_accent_recall": c.rv.get("accent_recall"),
                "rv_accent_f1": c.rv.get("accent_f1"),
                "ms_accent_precision": c.ms.get("accent_precision"),
                "ms_accent_recall": c.ms.get("accent_recall"),
                "ms_accent_f1": c.ms.get("accent_f1"),
                "rv_accent_stripped_cer": c.rv.get("accent_stripped_cer"),
                "ms_accent_stripped_cer": c.ms.get("accent_stripped_cer"),
                "best_epoch": ts.get("best_epoch"),
                "best_step": ts.get("best_step"),
                "global_step": ts.get("global_step"),
                "training_state_source": c.training_state_source,
                "training_state_confidence": c.training_state_confidence,
                "training_state_recovery_method": c.training_state_recovery_method,
                "model_base": c.model_base,
                "results_path": c.results_path,
            }
        )
    return buf.getvalue()


PER_RATIO_COLUMNS: tuple[str, ...] = (
    "rv_ratio",
    "ms_ratio",
    "metric",
    "n",
    "mean",
    "sd",
    "sem",
    "ci95_lo",
    "ci95_hi",
    "ci95_halfwidth",
    "values",
)


def render_per_ratio_csv(by_ratio: dict[float, dict[str, Stat]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(PER_RATIO_COLUMNS))
    writer.writeheader()
    for rv_ratio in RV_RATIOS:
        for metric in AGG_METRICS:
            stat = by_ratio[rv_ratio][metric]
            writer.writerow(
                {
                    "rv_ratio": rv_ratio,
                    "ms_ratio": round(1.0 - rv_ratio, 2),
                    "metric": metric,
                    "n": stat.n,
                    "mean": stat.mean,
                    "sd": stat.sd,
                    "sem": stat.sem,
                    "ci95_lo": stat.ci95_lo,
                    "ci95_hi": stat.ci95_hi,
                    "ci95_halfwidth": stat.ci95_halfwidth,
                    "values": ";".join(f"{v:.6f}" for v in stat.values),
                }
            )
    return buf.getvalue()


def _fixed(value: Optional[float]) -> str:
    """Fixed-point, never scientific; empty for an uncomputable contrast.

    A small diff written as ``4.9779724718123273e-05`` reads as "4.98" when a table
    cell is skimmed or truncated — a five-order-of-magnitude error in a claim. The
    value stays ``None`` when a contrast has too few cells to compute, and that must
    render as an empty cell rather than crash or print "None".
    """
    return "" if value is None else f"{value:.8f}"


def _contrast_rows(name: str, contrast: dict[str, Any]) -> list[dict[str, Any]]:
    diff = contrast["diff"]
    return [
        {
            "comparison": name,
            "n": diff["n"],
            # Fixed-point, never scientific: a small diff written as
            # 4.9779724718123273e-05 reads as "4.98" when a table cell is skimmed
            # or truncated, which is a five-order-of-magnitude error in a claim.
            "diff_mean": _fixed(diff["mean"]),
            "diff_ci95_lo": _fixed(diff["ci95_lo"]),
            "diff_ci95_hi": _fixed(diff["ci95_hi"]),
            "direction": contrast["direction"],
            "sign_consistent": contrast["sign_consistent"],
            "per_seed_diff": ";".join(
                f"{s}:{d:.6f}" for s, d in sorted(contrast["per_seed_diff"].items())
            ),
        }
    ]


ASYMMETRY_COLUMNS: tuple[str, ...] = (
    "comparison",
    "n",
    "diff_mean",
    "diff_ci95_lo",
    "diff_ci95_hi",
    "direction",
    "sign_consistent",
    "per_seed_diff",
)


def render_asymmetry_csv(asym: dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(ASYMMETRY_COLUMNS))
    writer.writeheader()
    for metric, block in asym["zero_shot"].items():
        writer.writerows(_contrast_rows(f"zero_shot_{metric}(MS→RV − RV→MS)", block))
    writer.writerows(
        _contrast_rows("in_domain_cer(MS_own − RV_own)", asym["in_domain_cer"])
    )
    return buf.getvalue()


def render_retention_csv(ra: dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(ASYMMETRY_COLUMNS))
    writer.writeheader()
    for group, block in ra.items():
        for name, contrast in block.items():
            writer.writerows(_contrast_rows(f"{group}:{name}", contrast))
    return buf.getvalue()


def _pm(stat: Stat, digits: int = 4) -> str:
    """``mean ± halfwidth`` (or just mean when n<2, or ``—`` when empty)."""
    if stat.mean is None:
        return "—"
    if stat.ci95_halfwidth is None:
        return f"{stat.mean:.{digits}f}"
    return f"{stat.mean:.{digits}f} ± {stat.ci95_halfwidth:.{digits}f}"


def _diff_phrase(contrast: dict[str, Any], digits: int = 4) -> str:
    diff = contrast["diff"]
    if diff["mean"] is None:
        return "—"
    hw = diff["ci95_halfwidth"]
    body = f"{diff['mean']:+.{digits}f}"
    if hw is not None:
        body += f" ± {hw:.{digits}f}"
    consistent = "sign-consistent" if contrast["sign_consistent"] else "sign varies"
    return f"{body} (n={diff['n']}, {consistent})"


def _stat_from_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Rebuild :class:`Stat` kwargs from its ``as_dict`` (values → tuple)."""
    return {**d, "values": tuple(d.get("values") or ())}


_EVALUATOR_BACKENDS: tuple[str, ...] = ("jiwer", "evaluate", "sacrebleu", "rapidfuzz")


def evaluator_dependency_pin() -> dict[str, Any]:
    """Python + metric-backend versions used for the inference-free recompute.

    Records the versions actually resolved at aggregation time (uv.lock-locked)
    so the exact-match recompute reported in this bundle is reproducible (study).
    A backend that is somehow absent is stamped ``"not-installed"`` rather than
    failing the aggregation.
    """
    import platform
    from importlib.metadata import PackageNotFoundError, version

    backends: dict[str, str] = {}
    for pkg in _EVALUATOR_BACKENDS:
        try:
            backends[pkg] = version(pkg)
        except PackageNotFoundError:
            backends[pkg] = "not-installed"
    return {"python": platform.python_version(), "backends": backends}

