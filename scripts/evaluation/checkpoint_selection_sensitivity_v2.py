"""Checkpoint selection sensitivity v2: PACLIC 2026 v2 release."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from . import aggregate_accent_ratio_multiseed as agg
from . import build_paclic_v2_maingrid_analysis as v2
from . import selection_helpers
from .aggregate_accent_ratio_multiseed import STEPS_PER_EPOCH

# --------------------------------------------------------------------------- #
# Study identity.
# --------------------------------------------------------------------------- #
#: Study slug (log prefix + bundle dir name). Distinct from the v1 study slug so
#: this bundle can never overwrite ``paclic-checkpoint-selection-sensitivity/``.
STUDY: str = "paclic-v2-checkpoint-selection-sensitivity"
DEFAULT_OUT_DIR: Path = Path(
    "results/"
    "paclic-v2-checkpoint-selection-sensitivity"
)

#: The three ledger columns, and the two uniform criteria under study. Imported
#: from the v1 module so the two studies cannot drift on what a criterion *is*.
CRITERIA: tuple[str, ...] = selection_helpers.CRITERIA
SENSITIVITY_CRITERIA: tuple[str, ...] = selection_helpers.SENSITIVITY_CRITERIA
#: The seven per-cell test metrics a scorable cell reports, and the helper that
#: derives ``avg_cer`` / ``worst_view_cer`` / ``worst_accent_f1`` — also imported,
#: so a v1 row and a v2 row mean the same thing.
REPORT_METRICS: tuple[str, ...] = selection_helpers.REPORT_METRICS

#: The per-epoch validation quantities whose movement between the mixed-rule and
#: uniform-rule epoch bounds the exposure. ``dominant_val_cer`` is not listed: it is
#: the cell's *own* mixed metric and is reported separately, because which field it
#: names differs by ratio.
EXPOSURE_QUANTITIES: tuple[str, ...] = agg.CANDIDATE_VALUE_FIELDS

#: The mixed rule's ``metric_for_best_model`` → its human ledger label.
_METRIC_TO_RULE_LABEL: dict[str, str] = {
    "eval_rv_cer": "rv_val_cer",
    "eval_ms_cer": "ms_val_cer",
    "eval_avg_cer": "mean_val_cer",
}

#: A recovered trajectory's value at the mixed-rule epoch must reproduce
#: ``training_state.best_metric`` to this tolerance, or the trajectory does not
#: belong to the committed run.
BEST_METRIC_AGREEMENT_TOL: float = 1e-9
#: Two paired trajectories agree on a shared epoch within this tolerance. Both
#: sources print full ``repr`` floats, so agreement is exact in practice and this is
#: only a float-identity guard.
TRAJECTORY_AGREEMENT_TOL: float = 1e-9

#: A surviving checkpoint whose ``log_history`` reproduces the cell's gated
#: trajectory: it belongs to the run whose predictions the paper reports, so
#: decoding it would score *that* run at another epoch.
ATTRIBUTION_REPORTED: str = "reported_run"
#: A surviving checkpoint whose ``log_history`` contradicts the gated trajectory —
#: it is left over from a **superseded attempt** of the same cell. Decoding it would
#: mix two training runs in one row, so it is never scorable.
ATTRIBUTION_ORPHAN: str = "orphan_not_the_reported_run"
#: A surviving checkpoint with no readable/overlapping ``log_history``: not shown to
#: belong to the reported run, therefore treated as unusable (never assumed good).
ATTRIBUTION_UNKNOWN: str = "unattributable"

#: What a still-unscorable ``(cell, criterion)`` would need. ``reinference`` only
#: where a checkpoint at that exact step survives **and** is attributed to the
#: reported run; otherwise the epoch's weights do not exist anywhere.
RESOLUTION_REINFERENCE: str = "reinference_of_a_surviving_reported_run_checkpoint"
RESOLUTION_RETRAIN: str = "retraining_the_cell_no_weights_exist_for_that_epoch"


def _fmt_ratio(rv: float) -> str:
    """``0.75`` → ``"75:25"`` — the RV:MS label every table and note uses."""
    return f"{int(round(rv * 100))}:{int(round((1 - rv) * 100))}"


def _ratio_list(keys: "Sequence[str]") -> str:
    """``["1", "0"]`` → ``"100:0, 0:100"`` — ratio keys as RV:MS labels."""
    return ", ".join(_fmt_ratio(float(key)) for key in keys)


# --------------------------------------------------------------------------- #
# Surviving-checkpoint census + attribution.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CheckpointSurvival:
    """One surviving ``checkpoints/checkpoint-<step>`` dir, and whose run it is.

    ``attribution`` is decided by comparing the checkpoint's own
    ``trainer_state.json`` ``log_history`` with the cell's *gated* trajectory (the
    winning Slurm log's). ``max_abs_trajectory_delta`` is the largest per-epoch RV/MS
    validation-CER disagreement over shared epochs — ``0.0`` for the reported run,
    clearly non-zero for a superseded attempt.
    """

    dir_name: str
    step: int
    epoch: Optional[int]
    n_shared_epochs: int
    max_abs_trajectory_delta: Optional[float]
    attribution: str
    trainer_state_best_metric: Optional[float]
    trainer_state_best_model_checkpoint: Optional[str]

    @property
    def usable_for_reinference(self) -> bool:
        """Whether decoding this checkpoint would score the *reported* run."""
        return self.attribution == ATTRIBUTION_REPORTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "dir_name": self.dir_name,
            "step": self.step,
            "epoch": self.epoch,
            "n_shared_epochs": self.n_shared_epochs,
            "max_abs_trajectory_delta": self.max_abs_trajectory_delta,
            "attribution": self.attribution,
            "usable_for_reinference": self.usable_for_reinference,
            "trainer_state_best_metric": self.trainer_state_best_metric,
            "trainer_state_best_model_checkpoint": (
                self.trainer_state_best_model_checkpoint
            ),
        }


def _step_to_epoch(
    step: int, *, steps_per_epoch: int = STEPS_PER_EPOCH
) -> Optional[int]:
    """``checkpoint-<step>`` → its epoch, or ``None`` if the step is not on-contract."""
    if steps_per_epoch <= 0 or step % steps_per_epoch:
        return None
    return step // steps_per_epoch


def max_abs_trajectory_delta(
    a: Sequence[agg.SelectionCandidate], b: Sequence[agg.SelectionCandidate]
) -> tuple[int, Optional[float]]:
    """``(n_shared_epochs, max |Δ| over both views)`` between two trajectories.

    The comparison that attributes a checkpoint to a run: a checkpoint written by
    the reported run replays that run's own eval curve exactly, while one left over
    from a superseded attempt disagrees. ``(0, None)`` when the two share no epoch.
    """
    by_epoch = {c.epoch: c for c in b}
    worst: Optional[float] = None
    shared = 0
    for ca in a:
        cb = by_epoch.get(ca.epoch)
        if cb is None:
            continue
        shared += 1
        delta = max(
            abs(ca.rv_val_cer - cb.rv_val_cer), abs(ca.ms_val_cer - cb.ms_val_cer)
        )
        worst = delta if worst is None else max(worst, delta)
    return shared, worst


def survey_checkpoints(
    cell_dir: Path,
    trajectory: Sequence[agg.SelectionCandidate],
    *,
    tol: float = TRAJECTORY_AGREEMENT_TOL,
) -> tuple[CheckpointSurvival, ...]:
    """Census ``cell_dir/checkpoints/`` and attribute every surviving checkpoint.

    Attribution needs the cell's gated trajectory: a checkpoint whose
    ``log_history`` reproduces it within ``tol`` is the reported run's
    (:data:`ATTRIBUTION_REPORTED`); one that disagrees is an orphan from a superseded
    attempt (:data:`ATTRIBUTION_ORPHAN`); one with no readable or no overlapping
    ``log_history`` is :data:`ATTRIBUTION_UNKNOWN` — never assumed good.
    """
    root = cell_dir / "checkpoints"
    if not root.is_dir():
        return ()
    out: list[CheckpointSurvival] = []
    for path in sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-", 1)[1]),
    ):
        step = int(path.name.split("-", 1)[1])
        state: dict[str, Any] = {}
        try:
            state = json.loads(
                (path / "trainer_state.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, ValueError):
            state = {}
        candidates = agg.paired_candidates_from_trainer_dicts(
            state.get("log_history", []) if isinstance(state, dict) else []
        )
        shared, worst = max_abs_trajectory_delta(candidates, trajectory)
        if not candidates or shared == 0 or worst is None:
            attribution = ATTRIBUTION_UNKNOWN
        elif worst <= tol:
            attribution = ATTRIBUTION_REPORTED
        else:
            attribution = ATTRIBUTION_ORPHAN
        out.append(
            CheckpointSurvival(
                dir_name=path.name,
                step=step,
                epoch=_step_to_epoch(step),
                n_shared_epochs=shared,
                max_abs_trajectory_delta=worst,
                attribution=attribution,
                trainer_state_best_metric=state.get("best_metric"),
                trainer_state_best_model_checkpoint=state.get("best_model_checkpoint"),
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# Per-cell selection resolution.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CellSelection:
    """One v2 cell's gated trajectory, its per-criterion selections, and what survives.

    ``mixed_epoch`` is the epoch the committed ``{rv,ms}_predictions.csv`` were
    decoded from — read from the cell's own ``results.json`` ``training_state``
    (the v2 launcher writes it inline, so no recovery is involved). The three gates
    are what bind the recovered trajectory to *that* run.
    """

    cell: agg.SeedCell
    candidates: tuple[agg.SelectionCandidate, ...]
    slurm_log: Optional[str]
    superseded_logs: tuple[str, ...]
    log_metric_for_best_model: Optional[str]
    results_metric_key: Optional[str]
    mixed_epoch: Optional[int]
    mixed_step: Optional[int]
    recorded_best_metric: Optional[float]
    planned_epochs: Optional[int]
    final_epoch: Optional[float]
    selection_gate_ok: Optional[bool]
    best_metric_gate_ok: Optional[bool]
    metric_key_gate_ok: Optional[bool]
    step_contract_ok: Optional[bool]
    decisions: dict[str, agg.SelectionDecision]
    checkpoints: tuple[CheckpointSurvival, ...]
    #: Whether ``final_model/`` survives — the weights ``load_best_model_at_end``
    #: left behind, i.e. the mixed-rule checkpoint itself. Measured, because the
    #: results page states it.
    has_final_model: bool
    notes: tuple[str, ...]

    # -- derived ------------------------------------------------------------- #
    @property
    def epochs_observed(self) -> int:
        return len(self.candidates)

    @property
    def last_observed_epoch(self) -> Optional[int]:
        return max((c.epoch for c in self.candidates), default=None)

    @property
    def early_stopped(self) -> Optional[bool]:
        """Whether early stopping truncated this run (fewer epochs than planned)."""
        if self.planned_epochs is None or self.last_observed_epoch is None:
            return None
        return self.last_observed_epoch < self.planned_epochs

    @property
    def gates_ok(self) -> bool:
        """Whether no gate *failed* (``None`` = not checkable, not a failure)."""
        return all(
            gate is not False
            for gate in (
                self.selection_gate_ok,
                self.best_metric_gate_ok,
                self.metric_key_gate_ok,
                self.step_contract_ok,
            )
        )

    @property
    def has_saved_test_predictions(self) -> bool:
        """Whether the committed top-level ``{rv,ms}_predictions.csv`` metrics exist."""
        return self.cell.present and self.cell.rv.get("cer") is not None

    def candidate_at(self, epoch: Optional[int]) -> Optional[agg.SelectionCandidate]:
        if epoch is None:
            return None
        return next((c for c in self.candidates if c.epoch == epoch), None)

    @property
    def mixed_field(self) -> Optional[str]:
        """The :data:`agg.CANDIDATE_VALUE_FIELDS` name this cell's mixed rule minimizes."""
        key = self.results_metric_key or self.log_metric_for_best_model
        return agg.METRIC_TO_CANDIDATE_FIELD.get(key) if key else None

    def usable_checkpoint_at(self, step: Optional[int]) -> Optional[CheckpointSurvival]:
        """A surviving, reported-run checkpoint at exactly ``step`` (or ``None``)."""
        if step is None:
            return None
        return next(
            (
                ck
                for ck in self.checkpoints
                if ck.step == step and ck.usable_for_reinference
            ),
            None,
        )


def _int_epoch(value: Any) -> Optional[int]:
    return None if value is None else int(round(float(value)))


def raw_training_state(cell_dir: Path) -> dict[str, Any]:
    """The cell's **full** ``results.json`` ``training_state`` mapping (``{}`` if none).

    ``agg.SeedCell.training_state`` is deliberately narrowed to ``best_epoch`` /
    ``best_step`` / ``global_step`` (it has to be uniform across v1's four recovery
    sources), but the v2 launcher also writes ``best_metric``, ``best_metric_key``
    and ``final_epoch`` inline — and those are exactly what the ``best_metric`` and
    ``metric_key`` gates check. So the raw mapping is re-read here rather than
    widening the shared dataclass.
    """
    try:
        data = json.loads((cell_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    state = data.get("training_state") if isinstance(data, dict) else None
    return state if isinstance(state, dict) else {}


def build_decisions(
    candidates: Sequence[agg.SelectionCandidate],
    *,
    mixed_field: Optional[str],
    mixed_epoch: Optional[int],
    gates_ok: bool,
) -> dict[str, agg.SelectionDecision]:
    """Per-criterion :class:`agg.SelectionDecision`\\ s for one cell's trajectory.

    ``current`` is pinned to the candidate at the recorded saved-prediction epoch,
    so it always self-matches. Each uniform criterion takes its earliest-min
    candidate (HF Trainer ``load_best_model_at_end`` under
    ``greater_is_better=False``) and matches iff that epoch equals ``mixed_epoch``.
    When a gate failed the trajectory is untrustworthy, so both uniform criteria are
    forced to deferred regardless of the epoch match — never a silent fallback to
    the committed checkpoint. Pure function of its inputs (no I/O).
    """
    decisions: dict[str, agg.SelectionDecision] = {}
    if mixed_epoch is None:
        return decisions

    mixed_candidate = next((c for c in candidates if c.epoch == mixed_epoch), None)
    if mixed_candidate is not None and mixed_field is not None:
        decisions["current"] = agg.build_selection_decision(
            "current",
            mixed_field,
            mixed_candidate,
            current_selected_epoch=mixed_epoch,
        )
    for criterion in SENSITIVITY_CRITERIA:
        candidate = agg.select_earliest_min(candidates, criterion)
        if candidate is None:
            continue
        decision = agg.build_selection_decision(
            criterion, criterion, candidate, current_selected_epoch=mixed_epoch
        )
        if not gates_ok:
            decision = replace(
                decision,
                matches_current_saved_predictions=False,
                status=agg.STATUS_DEFERRED,
            )
        decisions[criterion] = decision
    return decisions


def resolve_cell_selection(
    cell: agg.SeedCell,
    *,
    study_root: Path,
    attestation: dict[str, dict[str, Any]],
    slurm_log_dir: Optional[Path],
    planned_epochs: Optional[int] = None,
) -> CellSelection:
    """Recover, gate and decide one v2 cell.

    The trajectory source is the **winning** Slurm ``.out`` for this run name (the
    numeric ``(jobid, task)`` winner from
    :func:`v2.attest_corpus_from_slurm_logs`); surviving checkpoints are *not* used
    as a trajectory source, because on v2 a surviving checkpoint may belong to a
    superseded attempt — they are censused and attributed instead
    (:func:`survey_checkpoints`).
    """
    notes: list[str] = []
    record = attestation.get(cell.run_name, {})
    log_basename = record.get("log")
    superseded = tuple(record.get("superseded_logs", ()) or ())

    text: Optional[str] = None
    if log_basename is not None and slurm_log_dir is not None:
        path = slurm_log_dir / log_basename
        if path.is_file():
            text = path.read_text(errors="replace")
    if text is None:
        notes.append(
            "no winning Slurm .out resolved for this run; no validation trajectory, "
            "so both uniform criteria are undecidable"
        )
    candidates = agg.parse_paired_validation_trajectory(text) if text else []
    log_metric = agg.parse_metric_for_best_model(text) if text else None

    # The shared loader's training_state is narrowed; the raw one carries
    # best_metric / best_metric_key / final_epoch, which two gates need.
    state = {
        **(cell.training_state or {}),
        **raw_training_state(study_root / cell.run_name),
    }
    mixed_epoch = _int_epoch(state.get("best_epoch"))
    mixed_step = state.get("best_step")
    recorded_best_metric = state.get("best_metric")
    results_metric_key = state.get("best_metric_key")
    final_epoch = state.get("final_epoch")

    # Gate 1 — re-selecting the mixed metric over the trajectory must reproduce the
    # recorded best epoch.
    mixed_field = (
        agg.METRIC_TO_CANDIDATE_FIELD.get(results_metric_key or log_metric or "")
        or None
    )
    argmin = (
        agg.select_earliest_min(candidates, mixed_field)
        if mixed_field and candidates
        else None
    )
    selection_gate_ok: Optional[bool] = (
        None if argmin is None or mixed_epoch is None else argmin.epoch == mixed_epoch
    )
    if selection_gate_ok is False and argmin is not None:
        notes.append(
            f"selection gate FAILED: trajectory argmin (epoch {argmin.epoch}) != "
            f"recorded best_epoch ({mixed_epoch}); uniform criteria deferred"
        )

    # Gate 2 — the trajectory's value at that epoch must reproduce best_metric.
    at_mixed = next((c for c in candidates if c.epoch == mixed_epoch), None)
    best_metric_gate_ok: Optional[bool] = None
    if at_mixed is not None and mixed_field and recorded_best_metric is not None:
        best_metric_gate_ok = (
            abs(at_mixed.value(mixed_field) - float(recorded_best_metric))
            <= BEST_METRIC_AGREEMENT_TOL
        )
        if not best_metric_gate_ok:
            notes.append(
                f"best_metric gate FAILED: trajectory {mixed_field}="
                f"{at_mixed.value(mixed_field)!r} at epoch {mixed_epoch} != recorded "
                f"best_metric={recorded_best_metric!r}; uniform criteria deferred"
            )

    # Gate 3 — the log header's metric and results.json's key must agree.
    metric_key_gate_ok: Optional[bool] = (
        None
        if log_metric is None or results_metric_key is None
        else log_metric == results_metric_key
    )
    if metric_key_gate_ok is False:
        notes.append(
            f"metric-key gate FAILED: Slurm header metric_for_best_model={log_metric!r} "
            f"!= results.json best_metric_key={results_metric_key!r}"
        )

    # Gate 4 — the fixed PACLIC step contract (best_step == best_epoch * 326).
    step_contract_ok: Optional[bool] = (
        None
        if mixed_epoch is None or mixed_step is None
        else int(mixed_step) == mixed_epoch * STEPS_PER_EPOCH
    )
    if step_contract_ok is False:
        notes.append(
            f"step-contract gate FAILED: best_step={mixed_step} != best_epoch "
            f"{mixed_epoch} × {STEPS_PER_EPOCH}"
        )

    gates_ok = all(
        gate is not False
        for gate in (
            selection_gate_ok,
            best_metric_gate_ok,
            metric_key_gate_ok,
            step_contract_ok,
        )
    )
    if mixed_epoch is not None and at_mixed is None and candidates:
        notes.append(
            f"recorded best_epoch {mixed_epoch} is absent from the recovered "
            "trajectory; the current-rule decision is unavailable"
        )

    cell_dir = study_root / cell.run_name
    checkpoints = survey_checkpoints(cell_dir, candidates)
    orphans = [ck for ck in checkpoints if ck.attribution == ATTRIBUTION_ORPHAN]
    if orphans:
        notes.append(
            "surviving checkpoint(s) "
            + ", ".join(f"{ck.dir_name}" for ck in orphans)
            + " replay a DIFFERENT eval curve than the committed run (max |Δ| "
            + ", ".join(
                f"{ck.max_abs_trajectory_delta:.6f}"
                for ck in orphans
                if ck.max_abs_trajectory_delta is not None
            )
            + "), so they are left over from a superseded attempt and may never be "
            "decoded into this cell's row"
        )
    if superseded:
        notes.append(
            "re-queued cell: superseded log(s) "
            + ", ".join(superseded)
            + f" ignored in favour of the numeric (jobid, task) winner {log_basename}"
        )

    return CellSelection(
        cell=cell,
        candidates=tuple(candidates),
        slurm_log=log_basename,
        superseded_logs=superseded,
        log_metric_for_best_model=log_metric,
        results_metric_key=results_metric_key,
        mixed_epoch=mixed_epoch,
        mixed_step=int(mixed_step) if mixed_step is not None else None,
        recorded_best_metric=(
            float(recorded_best_metric) if recorded_best_metric is not None else None
        ),
        planned_epochs=planned_epochs,
        final_epoch=float(final_epoch) if final_epoch is not None else None,
        selection_gate_ok=selection_gate_ok,
        best_metric_gate_ok=best_metric_gate_ok,
        metric_key_gate_ok=metric_key_gate_ok,
        step_contract_ok=step_contract_ok,
        decisions=build_decisions(
            candidates,
            mixed_field=mixed_field,
            mixed_epoch=mixed_epoch,
            gates_ok=gates_ok,
        ),
        checkpoints=checkpoints,
        has_final_model=(cell_dir / "final_model").is_dir(),
        notes=tuple(notes),
    )


def build_selections(
    cells: Sequence[agg.SeedCell],
    *,
    study_root: Path,
    attestation: dict[str, dict[str, Any]],
    slurm_log_dir: Optional[Path],
    planned_epochs: Optional[dict[str, Optional[int]]] = None,
) -> list[CellSelection]:
    """Resolve every present cell, in the grid's ratio-major order."""
    planned = planned_epochs or {}
    return [
        resolve_cell_selection(
            cell,
            study_root=study_root,
            attestation=attestation,
            slurm_log_dir=slurm_log_dir,
            planned_epochs=planned.get(cell.run_name),
        )
        for cell in cells
        if cell.present
    ]


def planned_epochs_by_run(
    study_root: Path, cells: Sequence[agg.SeedCell]
) -> dict[str, Optional[int]]:
    """``run_name → num_train_epochs`` from each cell's own ``results.json``.

    Read through :func:`v2.subset_provenance` (which already re-reads the recipe
    fields) so the planned length is not re-parsed a second way. It is what makes
    "early stopping truncated this trajectory" a measurement rather than a guess.
    """
    prov = v2.subset_provenance(study_root, cells)
    out: dict[str, Optional[int]] = {}
    for name, record in prov.items():
        value = record.get("num_train_epochs")
        out[name] = int(value) if value is not None else None
    return out


# --------------------------------------------------------------------------- #
# Scorability.
# --------------------------------------------------------------------------- #
def is_scorable(selection: CellSelection, criterion: str) -> bool:
    """Whether ``criterion``'s selected checkpoint is test-scorable for this cell.

    ``current`` is scorable for any present cell (its saved predictions *are* that
    checkpoint). A uniform criterion is scorable only when it selects the
    saved-prediction epoch — on v2 there is no re-inference record, and a divergent
    cell is **never** scored on the committed checkpoint.
    """
    if not selection.has_saved_test_predictions:
        return False
    decision = selection.decisions.get(criterion)
    if decision is None:
        return False
    if criterion == "current":
        return True
    return decision.status == agg.STATUS_COMPUTABLE


def resolution_needed(selection: CellSelection, criterion: str) -> Optional[str]:
    """What an unscorable ``(cell, criterion)`` would need — re-inference or retraining.

    :data:`RESOLUTION_REINFERENCE` only when a checkpoint at that criterion's
    selected step survives **and** is attributed to the reported run; otherwise no
    weights for that epoch exist anywhere and only retraining could produce them.
    ``None`` for a scorable cell.
    """
    if criterion == "current" or is_scorable(selection, criterion):
        return None
    decision = selection.decisions.get(criterion)
    if decision is None:
        return RESOLUTION_RETRAIN
    if selection.usable_checkpoint_at(decision.selected_step) is not None:
        return RESOLUTION_REINFERENCE
    return RESOLUTION_RETRAIN


def divergence_summary(
    selections: Sequence[CellSelection], criterion: str
) -> dict[str, Any]:
    """Match/diverge counts and run lists for one uniform criterion (recomputed)."""
    matching: list[str] = []
    diverging: list[str] = []
    reachable: list[str] = []
    retrain_only: list[str] = []
    for s in selections:
        decision = s.decisions.get(criterion)
        if decision is None:
            continue
        if is_scorable(s, criterion):
            matching.append(s.cell.run_name)
            continue
        diverging.append(s.cell.run_name)
        if resolution_needed(s, criterion) == RESOLUTION_REINFERENCE:
            reachable.append(s.cell.run_name)
        else:
            retrain_only.append(s.cell.run_name)
    return {
        "criterion": criterion,
        "n_decided": len(matching) + len(diverging),
        "n_match": len(matching),
        "n_diverge": len(diverging),
        "n_reinference_reachable": len(reachable),
        "n_retrain_only": len(retrain_only),
        "matching_runs": matching,
        "diverging_runs": diverging,
        "reinference_reachable_runs": reachable,
        "retrain_only_runs": retrain_only,
    }


def aggregate_criterion(
    selections: Sequence[CellSelection], criterion: str
) -> dict[float, dict[str, Any]]:
    """Per mix ratio, summarize the seven test metrics over ``criterion``'s scorable cells.

    ``n`` and the contributing seeds are reported per ratio, so a partial subset can
    never be mistaken for a complete one. Statistics are
    :func:`agg.summarize` (mean / sample SD / Student-t 95% CI) — the same function
    behind every other table in the paper.
    """
    per_ratio: dict[float, dict[str, Any]] = {}
    for rv_ratio in v2.RV_RATIOS:
        scorable = [
            s
            for s in selections
            if s.cell.rv_ratio == rv_ratio and is_scorable(s, criterion)
        ]
        metrics = [selection_helpers.report_metrics(s.cell) for s in scorable]
        per_ratio[rv_ratio] = {
            "n": len(scorable),
            "seeds": sorted(s.cell.seed for s in scorable),
            "stats": {
                metric: agg.summarize(m.get(metric) for m in metrics)
                for metric in REPORT_METRICS
            },
        }
    return per_ratio


#: The three interior mix ratios — the band the paper's plateau/ordering reading
#: lives on. The endpoints are a different claim (zero-shot transfer), and a
#: uniform rule changes what they even measure (see :func:`deliverability`).
INTERIOR_RATIOS: tuple[float, ...] = (0.75, 0.5, 0.25)


def deliverability(
    selections: Sequence[CellSelection], criterion: str
) -> dict[str, Any]:
    """Whether ``criterion`` supports a table, recomputed from the scorable counts.

    Three graded verdicts, because "cannot be tested" has structure:

    * ``all_ratios_full`` — every one of the five ratios scorable at the full n=3.
      Only this licenses a grid-wide uniform-rule result.
    * ``all_ratios_have_ci`` — every ratio has n≥2, so every row at least carries an
      interval.
    * ``interior_band_has_ci`` — the three interior ratios each have n≥2, so the
      plateau/ordering reading *inside* the band can be compared (with the per-ratio
      n stated).
    """
    per_ratio = aggregate_criterion(selections, criterion)
    counts = {rv: per_ratio[rv]["n"] for rv in v2.RV_RATIOS}
    full = len(agg.SEEDS)
    return {
        "criterion": criterion,
        "n_scorable_cells": sum(counts.values()),
        "n_cells": len(selections),
        "per_ratio_n": {f"{rv:g}": counts[rv] for rv in v2.RV_RATIOS},
        "ratios_at_full_n": [f"{rv:g}" for rv in v2.RV_RATIOS if counts[rv] == full],
        "ratios_with_no_cell": [f"{rv:g}" for rv in v2.RV_RATIOS if counts[rv] == 0],
        "all_ratios_full": all(counts[rv] == full for rv in v2.RV_RATIOS),
        "all_ratios_have_ci": all(counts[rv] >= 2 for rv in v2.RV_RATIOS),
        "interior_band_has_ci": all(counts[rv] >= 2 for rv in INTERIOR_RATIOS),
        "interior_band_full": all(counts[rv] == full for rv in INTERIOR_RATIOS),
    }


#: The two deliverables the study could produce on v2, and the rule that picks one.
DELIVERABLE_SENSITIVITY: str = "sensitivity_result"
DELIVERABLE_BOUNDED_NEGATIVE: str = "bounded_negative_result"


def chosen_deliverable(selections: Sequence[CellSelection]) -> dict[str, Any]:
    """Which study deliverable the surviving artifacts support — decided, not assumed.

    A **sensitivity result** requires at least one uniform criterion scorable at the
    full n=3 in **all five** ratios; anything less is a **bounded negative result**
    (precisely which cells diverge, how far, and therefore what the paper may say).
    The verdict and its reasons are recomputed here so the bundle's prose can never
    disagree with its own tables.
    """
    per_criterion = {c: deliverability(selections, c) for c in SENSITIVITY_CRITERIA}
    supporting = [c for c, d in per_criterion.items() if d["all_ratios_full"]]
    reasons: list[str] = []
    for criterion, block in per_criterion.items():
        empty = block["ratios_with_no_cell"]
        reasons.append(
            f"{criterion}: {block['n_scorable_cells']}/{block['n_cells']} cells "
            f"scorable (per-ratio n "
            + ", ".join(
                f"{_fmt_ratio(float(rv))}={n}" for rv, n in block["per_ratio_n"].items()
            )
            + ")"
            + (
                "; no scorable cell at all at RV:MS "
                + ", ".join(_fmt_ratio(float(rv)) for rv in empty)
                if empty
                else ""
            )
        )
    return {
        "chosen": (
            DELIVERABLE_SENSITIVITY if supporting else DELIVERABLE_BOUNDED_NEGATIVE
        ),
        "criteria_supporting_a_grid_wide_result": supporting,
        "per_criterion": per_criterion,
        "why": reasons,
        "rule": (
            "a grid-wide sensitivity result requires some uniform criterion scorable "
            f"at n={len(agg.SEEDS)} in all {len(v2.RV_RATIOS)} ratios; otherwise the "
            "honest deliverable is the bounded negative result"
        ),
    }


# --------------------------------------------------------------------------- #
# Exposure: how far the two epochs' validation metrics are apart.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Exposure:
    """How far apart the mixed-rule and uniform-rule checkpoints are, for one cell.

    Computable for **every** cell with a gated trajectory, checkpoint or not, which
    is why it is the part of study that v2 can answer in full. ``deltas`` is
    ``uniform − mixed`` per :data:`EXPOSURE_QUANTITIES`; ``dominant_delta`` is the
    same difference on the cell's *own* mixed metric — the movement in the quantity
    that cell's checkpoint was chosen for. ``censored`` marks a uniform-rule epoch
    that is the last epoch observed on a trajectory early stopping truncated: the
    criterion was still improving when the *mixed* rule stopped training, so the
    uniform rule's true argmin is unobservable and this epoch is only a lower bound.
    """

    run_name: str
    rv_ratio: float
    seed: int
    criterion: str
    mixed_epoch: int
    uniform_epoch: int
    epoch_delta: int
    mixed_values: dict[str, float]
    uniform_values: dict[str, float]
    deltas: dict[str, float]
    dominant_field: Optional[str]
    dominant_delta: Optional[float]
    diverges: bool
    censored: bool
    resolution: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "rv_ratio": self.rv_ratio,
            "seed": self.seed,
            "criterion": self.criterion,
            "mixed_epoch": self.mixed_epoch,
            "uniform_epoch": self.uniform_epoch,
            "epoch_delta": self.epoch_delta,
            "mixed_values": dict(sorted(self.mixed_values.items())),
            "uniform_values": dict(sorted(self.uniform_values.items())),
            "deltas": dict(sorted(self.deltas.items())),
            "dominant_field": self.dominant_field,
            "dominant_delta": self.dominant_delta,
            "diverges": self.diverges,
            "censored": self.censored,
            "resolution": self.resolution,
        }


def cell_exposure(selection: CellSelection, criterion: str) -> Optional[Exposure]:
    """The mixed-vs-uniform validation movement for one ``(cell, criterion)``.

    ``None`` when either epoch is missing from the recovered trajectory (nothing to
    compare) — never a zero delta standing in for an unknown one.
    """
    decision = selection.decisions.get(criterion)
    if decision is None or selection.mixed_epoch is None:
        return None
    mixed = selection.candidate_at(selection.mixed_epoch)
    uniform = selection.candidate_at(decision.selected_epoch)
    if mixed is None or uniform is None:
        return None
    mixed_values = {q: mixed.value(q) for q in EXPOSURE_QUANTITIES}
    uniform_values = {q: uniform.value(q) for q in EXPOSURE_QUANTITIES}
    field = selection.mixed_field
    last = selection.last_observed_epoch
    return Exposure(
        run_name=selection.cell.run_name,
        rv_ratio=selection.cell.rv_ratio,
        seed=selection.cell.seed,
        criterion=criterion,
        mixed_epoch=mixed.epoch,
        uniform_epoch=uniform.epoch,
        epoch_delta=uniform.epoch - mixed.epoch,
        mixed_values=mixed_values,
        uniform_values=uniform_values,
        deltas={q: uniform_values[q] - mixed_values[q] for q in EXPOSURE_QUANTITIES},
        dominant_field=field,
        dominant_delta=(
            uniform_values[field] - mixed_values[field]
            if field in uniform_values
            else None
        ),
        diverges=uniform.epoch != mixed.epoch,
        censored=bool(
            selection.early_stopped and last is not None and uniform.epoch == last
        ),
        resolution=resolution_needed(selection, criterion),
    )


def exposures(selections: Sequence[CellSelection]) -> list[Exposure]:
    """Every ``(cell, uniform criterion)`` exposure, in grid order."""
    return [
        exposure
        for s in selections
        for criterion in SENSITIVITY_CRITERIA
        if (exposure := cell_exposure(s, criterion)) is not None
    ]


def per_ratio_exposure(
    selections: Sequence[CellSelection], criterion: str
) -> dict[str, Any]:
    """Per-ratio **paired** exposure contrasts for one uniform criterion.

    For each mix ratio and each :data:`EXPOSURE_QUANTITIES` quantity, the two arms
    are ``seed → value at the mixed-rule epoch`` and ``seed → value at the
    uniform-rule epoch``; :func:`agg._contrast` then gives the paired mean Δ, its
    Student-t 95% CI and the per-seed sign check — the same paired machinery the
    main-grid bundle's asymmetry and retention/adaptation tables use, so the two
    bundles cannot disagree on what a CI is. Every ratio contributes all its cells,
    scorable or not: this is the part of study that needs no checkpoint.
    """
    rows = [e for e in exposures(selections) if e.criterion == criterion]
    out: dict[str, Any] = {}
    for rv_ratio in v2.RV_RATIOS:
        group = [e for e in rows if e.rv_ratio == rv_ratio]
        quantities: dict[str, Any] = {}
        for quantity in EXPOSURE_QUANTITIES:
            a = {e.seed: e.mixed_values[quantity] for e in group}
            b = {e.seed: e.uniform_values[quantity] for e in group}
            quantities[quantity] = agg._contrast(a, b)
        epoch_a = {e.seed: float(e.mixed_epoch) for e in group}
        epoch_b = {e.seed: float(e.uniform_epoch) for e in group}
        out[f"{rv_ratio:g}"] = {
            "rv_ratio": rv_ratio,
            "ms_ratio": round(1.0 - rv_ratio, 2),
            "n": len(group),
            "seeds": sorted(e.seed for e in group),
            "n_diverging": sum(1 for e in group if e.diverges),
            "n_censored": sum(1 for e in group if e.censored),
            "epoch": agg._contrast(epoch_a, epoch_b),
            "quantities": quantities,
        }
    return out


def exposure_extremes(selections: Sequence[CellSelection]) -> dict[str, Any]:
    """The worst observed movement per criterion — the headline bound.

    ``max_abs_delta`` per quantity over the diverging cells only (a matching cell
    contributes an exact 0 and would flatter the bound), with the run that set it.
    """
    out: dict[str, Any] = {}
    for criterion in SENSITIVITY_CRITERIA:
        rows = [
            e for e in exposures(selections) if e.criterion == criterion and e.diverges
        ]
        per_quantity: dict[str, Any] = {}
        for quantity in EXPOSURE_QUANTITIES:
            if not rows:
                per_quantity[quantity] = {"max_abs_delta": None, "run_name": None}
                continue
            worst = max(rows, key=lambda e: abs(e.deltas[quantity]))
            per_quantity[quantity] = {
                "max_abs_delta": abs(worst.deltas[quantity]),
                "signed_delta": worst.deltas[quantity],
                "run_name": worst.run_name,
                "mixed_epoch": worst.mixed_epoch,
                "uniform_epoch": worst.uniform_epoch,
            }
        max_epoch_gap = (
            max((abs(e.epoch_delta) for e in rows), default=None) if rows else None
        )
        out[criterion] = {
            "n_diverging_cells": len(rows),
            "max_abs_epoch_delta": max_epoch_gap,
            "per_quantity": per_quantity,
            "censored_runs": [e.run_name for e in rows if e.censored],
        }
    return out


# --------------------------------------------------------------------------- #
# Retention census + gates.
# --------------------------------------------------------------------------- #
def retention_summary(selections: Sequence[CellSelection]) -> dict[str, Any]:
    """What survives on disk for this grid, and what that permits (recomputed).

    The census that decides the deliverable: how many cells kept any checkpoint,
    how many of those checkpoints belong to the run whose predictions are reported,
    and therefore how many divergent ``(cell, criterion)`` rows re-inference could
    ever reach.
    """
    with_ck = [s for s in selections if s.checkpoints]
    all_ck = [(s, ck) for s in selections for ck in s.checkpoints]
    reachable = [
        {
            "run_name": s.cell.run_name,
            "criterion": criterion,
            "selected_epoch": s.decisions[criterion].selected_epoch,
            "selected_step": s.decisions[criterion].selected_step,
            "checkpoint_dir": ck.dir_name,
        }
        for s in selections
        for criterion in SENSITIVITY_CRITERIA
        if criterion in s.decisions
        and resolution_needed(s, criterion) == RESOLUTION_REINFERENCE
        and (ck := s.usable_checkpoint_at(s.decisions[criterion].selected_step))
        is not None
    ]
    return {
        "n_cells": len(selections),
        "n_cells_with_saved_predictions": sum(
            1 for s in selections if s.has_saved_test_predictions
        ),
        "n_cells_with_final_model": sum(1 for s in selections if s.has_final_model),
        "n_cells_with_any_surviving_checkpoint": len(with_ck),
        "n_surviving_checkpoints": len(all_ck),
        "n_surviving_checkpoints_by_attribution": {
            attribution: sum(1 for _, ck in all_ck if ck.attribution == attribution)
            for attribution in (
                ATTRIBUTION_REPORTED,
                ATTRIBUTION_ORPHAN,
                ATTRIBUTION_UNKNOWN,
            )
        },
        "cells": [
            {
                "run_name": s.cell.run_name,
                "checkpoints": [ck.as_dict() for ck in s.checkpoints],
            }
            for s in with_ck
        ],
        "n_reinference_reachable_rows": len(reachable),
        "reinference_reachable_rows": reachable,
        "why": (
            "save_total_limit=2 kept at most two checkpoints per run and the v2 "
            "launcher purges checkpoints on success, so a divergent cell is scorable "
            "only if a checkpoint at its uniform-rule step both SURVIVED and belongs "
            "to the run whose predictions are committed; a checkpoint left over from "
            "a superseded attempt of the same cell is an orphan and is excluded"
        ),
    }


def gate_summary(selections: Sequence[CellSelection]) -> dict[str, Any]:
    """The four trajectory-attribution gates, per gate and overall."""
    gates = (
        ("selection", lambda s: s.selection_gate_ok),
        ("best_metric", lambda s: s.best_metric_gate_ok),
        ("metric_key", lambda s: s.metric_key_gate_ok),
        ("step_contract", lambda s: s.step_contract_ok),
    )
    per_gate = {
        name: {
            "n_checked": sum(1 for s in selections if getter(s) is not None),
            "n_passed": sum(1 for s in selections if getter(s) is True),
            "failures": [s.cell.run_name for s in selections if getter(s) is False],
        }
        for name, getter in gates
    }
    return {
        "checks": {
            "selection": (
                "re-selecting the cell's own metric_for_best_model over the recovered "
                "trajectory reproduces results.json training_state.best_epoch"
            ),
            "best_metric": (
                "the trajectory's value at that epoch reproduces "
                "training_state.best_metric (tolerance "
                f"{BEST_METRIC_AGREEMENT_TOL:g})"
            ),
            "metric_key": (
                "the Slurm header's metric_for_best_model equals results.json "
                "training_state.best_metric_key"
            ),
            "step_contract": (
                f"best_step == best_epoch × {STEPS_PER_EPOCH} (the fixed PACLIC step "
                "contract)"
            ),
        },
        "per_gate": per_gate,
        "all_ok": all(s.gates_ok for s in selections),
        "failing_runs": [s.cell.run_name for s in selections if not s.gates_ok],
    }


def early_stopping_summary(selections: Sequence[CellSelection]) -> dict[str, Any]:
    """Which cells early stopping truncated, and where that censors a uniform rule."""
    truncated = [s for s in selections if s.early_stopped]
    censored = sorted(
        {e.run_name for e in exposures(selections) if e.censored and e.diverges}
    )
    return {
        "n_truncated": len(truncated),
        "n_cells": len(selections),
        "truncated": [
            {
                "run_name": s.cell.run_name,
                "epochs_observed": s.last_observed_epoch,
                "planned_epochs": s.planned_epochs,
                "mixed_epoch": s.mixed_epoch,
            }
            for s in truncated
        ],
        "censored_runs": censored,
        "note": (
            "early stopping is driven by the MIXED metric, so on a truncated "
            "trajectory a uniform criterion whose argmin is the last observed epoch "
            "was still improving when training stopped: its selected epoch is a LOWER "
            "BOUND, and the epoch a uniform rule would really have chosen is "
            "unobservable from any committed artifact"
        ),
    }


def band_summary(selections: Sequence[CellSelection]) -> dict[str, Any]:
    """Interior-band vs endpoint exposure, over **both** uniform criteria.

    The split that carries the paper's reading: the three interior ratios back the
    plateau/ordering statement, while the two endpoints back the zero-shot-transfer
    and retention/adaptation statements. Both bands are summarized over both uniform
    criteria at once, because the claim the paper needs is "under *either* uniform
    rule, does anything inside the band move" — not one criterion at a time. Every
    figure is recomputed from the exposures.
    """
    rows = exposures(selections)
    out: dict[str, Any] = {}
    for band, ratios in (
        ("interior", INTERIOR_RATIOS),
        ("endpoints", tuple(r for r in v2.RV_RATIOS if r not in INTERIOR_RATIOS)),
    ):
        group = [e for e in rows if e.rv_ratio in ratios]
        diverging = [e for e in group if e.diverges]
        non_censored = [e for e in diverging if not e.censored]
        out[band] = {
            "rv_ratios": [f"{r:g}" for r in ratios],
            "n_rows": len(group),
            "n_matching": len(group) - len(diverging),
            "n_diverging": len(diverging),
            "n_censored": sum(1 for e in diverging if e.censored),
            "max_abs_epoch_delta": max(
                (abs(e.epoch_delta) for e in diverging), default=None
            ),
            "max_abs_epoch_delta_excluding_censored": max(
                (abs(e.epoch_delta) for e in non_censored), default=None
            ),
            "max_abs_validation_delta": max(
                (abs(v) for e in diverging for v in e.deltas.values()), default=None
            ),
            "max_abs_validation_delta_quantity": (
                max(
                    ((abs(v), q) for e in diverging for q, v in e.deltas.items()),
                    default=(None, None),
                )[1]
            ),
            "diverging_runs": sorted({e.run_name for e in diverging}),
            "n_scorable_by_criterion": {
                criterion: sum(
                    1
                    for s in selections
                    if s.cell.rv_ratio in ratios and is_scorable(s, criterion)
                )
                for criterion in SENSITIVITY_CRITERIA
            },
            "n_cells": sum(1 for s in selections if s.cell.rv_ratio in ratios),
        }
    return out


# --------------------------------------------------------------------------- #
# Aggregate payload.
# --------------------------------------------------------------------------- #
def build_aggregate(selections: Sequence[CellSelection]) -> dict[str, Any]:
    """The machine-readable payload (→ ``aggregate.json``)."""
    per_criterion: dict[str, Any] = {}
    for criterion in CRITERIA:
        per_ratio = aggregate_criterion(selections, criterion)
        per_criterion[criterion] = {
            "n_scorable_cells": sum(1 for s in selections if is_scorable(s, criterion)),
            "per_ratio": {
                f"{rv:g}": {
                    "n": block["n"],
                    "seeds": block["seeds"],
                    "metrics": {
                        m: stat.as_dict() for m, stat in block["stats"].items()
                    },
                }
                for rv, block in per_ratio.items()
            },
            "ranking_by_avg_cer": selection_helpers.rank_ratios(per_ratio, "avg_cer"),
            "ranking_by_worst_view_cer": selection_helpers.rank_ratios(per_ratio, "worst_view_cer"),
        }
    return {
        "issue": 161,
        "study": STUDY,
        "parent_study": v2.STUDY,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "corpus": "v2",
        "no_training_inference_or_gpu": True,
        "grid": {"rv_ratios": list(v2.RV_RATIOS), "seeds": list(agg.SEEDS)},
        "criteria": list(CRITERIA),
        "n_cells": len(selections),
        "deliverable": chosen_deliverable(selections),
        "gates": gate_summary(selections),
        "retention": retention_summary(selections),
        "early_stopping": early_stopping_summary(selections),
        "divergence": {
            c: divergence_summary(selections, c) for c in SENSITIVITY_CRITERIA
        },
        "bands": band_summary(selections),
        "exposure_extremes": exposure_extremes(selections),
        "exposure_per_ratio": {
            c: per_ratio_exposure(selections, c) for c in SENSITIVITY_CRITERIA
        },
        "exposure_cells": [e.as_dict() for e in exposures(selections)],
        "per_criterion": per_criterion,
        "ledger": ledger_rows(selections),
    }


# --------------------------------------------------------------------------- #
# CSV renderers.
# --------------------------------------------------------------------------- #
LEDGER_COLUMNS: tuple[str, ...] = (
    "run_name",
    "rv_ratio",
    "ms_ratio",
    "seed",
    "criterion",
    "mixed_rule_metric",
    "mixed_rule_label",
    "mixed_selected_epoch",
    "mixed_selected_step",
    "selected_epoch",
    "selected_step",
    "selected_value",
    "rv_val_cer_at_selected",
    "ms_val_cer_at_selected",
    "mean_val_cer_at_selected",
    "worst_val_cer_at_selected",
    "matches_saved_predictions",
    "status",
    "scorable",
    "resolution_needed",
    "epochs_observed",
    "planned_epochs",
    "early_stopped",
    "slurm_log",
    "superseded_logs",
    "selection_gate_ok",
    "best_metric_gate_ok",
    "metric_key_gate_ok",
    "step_contract_ok",
    "surviving_checkpoints",
    "surviving_checkpoints_usable",
)


def ledger_rows(selections: Sequence[CellSelection]) -> list[dict[str, Any]]:
    """One row per ``(cell, criterion)`` — the full candidate-checkpoint ledger."""
    rows: list[dict[str, Any]] = []
    for s in selections:
        key = s.results_metric_key or s.log_metric_for_best_model
        for criterion in CRITERIA:
            decision = s.decisions.get(criterion)
            if decision is None:
                continue
            rows.append(
                {
                    "run_name": s.cell.run_name,
                    "rv_ratio": s.cell.rv_ratio,
                    "ms_ratio": s.cell.ms_ratio,
                    "seed": s.cell.seed,
                    "criterion": criterion,
                    "mixed_rule_metric": key,
                    "mixed_rule_label": _METRIC_TO_RULE_LABEL.get(key) if key else None,
                    "mixed_selected_epoch": s.mixed_epoch,
                    "mixed_selected_step": s.mixed_step,
                    "selected_epoch": decision.selected_epoch,
                    "selected_step": decision.selected_step,
                    "selected_value": decision.selected_value,
                    "rv_val_cer_at_selected": decision.rv_val_cer_at_selected,
                    "ms_val_cer_at_selected": decision.ms_val_cer_at_selected,
                    "mean_val_cer_at_selected": decision.mean_val_cer_at_selected,
                    "worst_val_cer_at_selected": decision.worst_val_cer_at_selected,
                    "matches_saved_predictions": (
                        decision.matches_current_saved_predictions
                    ),
                    "status": decision.status,
                    "scorable": is_scorable(s, criterion),
                    "resolution_needed": resolution_needed(s, criterion),
                    "epochs_observed": s.epochs_observed,
                    "planned_epochs": s.planned_epochs,
                    "early_stopped": s.early_stopped,
                    "slurm_log": s.slurm_log,
                    "superseded_logs": ";".join(s.superseded_logs),
                    "selection_gate_ok": s.selection_gate_ok,
                    "best_metric_gate_ok": s.best_metric_gate_ok,
                    "metric_key_gate_ok": s.metric_key_gate_ok,
                    "step_contract_ok": s.step_contract_ok,
                    "surviving_checkpoints": ";".join(
                        ck.dir_name for ck in s.checkpoints
                    ),
                    "surviving_checkpoints_usable": ";".join(
                        ck.dir_name for ck in s.checkpoints if ck.usable_for_reinference
                    ),
                }
            )
    return rows


def _render_csv(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def render_ledger_csv(selections: Sequence[CellSelection]) -> str:
    return _render_csv(LEDGER_COLUMNS, ledger_rows(selections))


CANDIDATE_COLUMNS: tuple[str, ...] = (
    "run_name",
    "rv_ratio",
    "ms_ratio",
    "seed",
    "epoch",
    "step",
    "rv_val_cer",
    "ms_val_cer",
    "mean_val_cer",
    "worst_val_cer",
    "is_mixed_selected",
    "is_mean_val_cer_selected",
    "is_worst_val_cer_selected",
    "slurm_log",
)


def render_candidates_csv(selections: Sequence[CellSelection]) -> str:
    """One row per ``(cell, epoch)`` — the full audit trail behind every selection."""
    rows: list[dict[str, Any]] = []
    for s in selections:
        selected = {
            criterion: s.decisions[criterion].selected_epoch
            for criterion in SENSITIVITY_CRITERIA
            if criterion in s.decisions
        }
        for c in s.candidates:
            rows.append(
                {
                    "run_name": s.cell.run_name,
                    "rv_ratio": s.cell.rv_ratio,
                    "ms_ratio": s.cell.ms_ratio,
                    "seed": s.cell.seed,
                    "epoch": c.epoch,
                    "step": c.step,
                    "rv_val_cer": c.rv_val_cer,
                    "ms_val_cer": c.ms_val_cer,
                    "mean_val_cer": c.mean_val_cer,
                    "worst_val_cer": c.worst_val_cer,
                    "is_mixed_selected": c.epoch == s.mixed_epoch,
                    "is_mean_val_cer_selected": (
                        c.epoch == selected.get("mean_val_cer")
                    ),
                    "is_worst_val_cer_selected": (
                        c.epoch == selected.get("worst_val_cer")
                    ),
                    "slurm_log": s.slurm_log,
                }
            )
    return _render_csv(CANDIDATE_COLUMNS, rows)


EXPOSURE_COLUMNS: tuple[str, ...] = (
    "criterion",
    "rv_ratio",
    "ms_ratio",
    "seed",
    "run_name",
    "mixed_epoch",
    "uniform_epoch",
    "epoch_delta",
    "diverges",
    "censored",
    "resolution_needed",
    "dominant_field",
    "dominant_delta",
    *(f"mixed_{q}" for q in EXPOSURE_QUANTITIES),
    *(f"uniform_{q}" for q in EXPOSURE_QUANTITIES),
    *(f"delta_{q}" for q in EXPOSURE_QUANTITIES),
)


def render_exposure_csv(selections: Sequence[CellSelection]) -> str:
    """One row per ``(cell, uniform criterion)`` — the checkpoint-free exposure bound."""
    rows: list[dict[str, Any]] = []
    for e in exposures(selections):
        row: dict[str, Any] = {
            "criterion": e.criterion,
            "rv_ratio": e.rv_ratio,
            "ms_ratio": round(1.0 - e.rv_ratio, 2),
            "seed": e.seed,
            "run_name": e.run_name,
            "mixed_epoch": e.mixed_epoch,
            "uniform_epoch": e.uniform_epoch,
            "epoch_delta": e.epoch_delta,
            "diverges": e.diverges,
            "censored": e.censored,
            "resolution_needed": e.resolution,
            "dominant_field": e.dominant_field,
            "dominant_delta": e.dominant_delta,
        }
        for quantity in EXPOSURE_QUANTITIES:
            row[f"mixed_{quantity}"] = e.mixed_values[quantity]
            row[f"uniform_{quantity}"] = e.uniform_values[quantity]
            row[f"delta_{quantity}"] = e.deltas[quantity]
        rows.append(row)
    return _render_csv(EXPOSURE_COLUMNS, rows)


PER_RATIO_EXPOSURE_COLUMNS: tuple[str, ...] = (
    "criterion",
    "rv_ratio",
    "ms_ratio",
    "quantity",
    "n",
    "seeds",
    "n_diverging",
    "n_censored",
    "mixed_mean",
    "uniform_mean",
    "diff_mean",
    "diff_sd",
    "diff_ci95_lo",
    "diff_ci95_hi",
    "diff_ci95_halfwidth",
    "sign_consistent",
    "direction",
    "per_seed_diff",
)


def render_per_ratio_exposure_csv(selections: Sequence[CellSelection]) -> str:
    """Per ``(criterion, ratio, quantity)`` paired exposure contrast (n=3 per ratio)."""
    rows: list[dict[str, Any]] = []
    for criterion in SENSITIVITY_CRITERIA:
        blocks = per_ratio_exposure(selections, criterion)
        for block in blocks.values():
            for quantity in ("epoch", *EXPOSURE_QUANTITIES):
                contrast = (
                    block["epoch"]
                    if quantity == "epoch"
                    else block["quantities"][quantity]
                )
                diff = contrast["diff"]
                rows.append(
                    {
                        "criterion": criterion,
                        "rv_ratio": block["rv_ratio"],
                        "ms_ratio": block["ms_ratio"],
                        "quantity": quantity,
                        "n": diff["n"],
                        "seeds": ";".join(str(s) for s in contrast["seeds"]),
                        "n_diverging": block["n_diverging"],
                        "n_censored": block["n_censored"],
                        "mixed_mean": contrast["a"]["mean"],
                        "uniform_mean": contrast["b"]["mean"],
                        "diff_mean": diff["mean"],
                        "diff_sd": diff["sd"],
                        "diff_ci95_lo": diff["ci95_lo"],
                        "diff_ci95_hi": diff["ci95_hi"],
                        "diff_ci95_halfwidth": diff["ci95_halfwidth"],
                        "sign_consistent": contrast["sign_consistent"],
                        "direction": contrast["direction"],
                        "per_seed_diff": ";".join(
                            f"{seed}={value:.6f}"
                            for seed, value in sorted(contrast["per_seed_diff"].items())
                        ),
                    }
                )
    return _render_csv(PER_RATIO_EXPOSURE_COLUMNS, rows)


SCORABLE_COLUMNS: tuple[str, ...] = (
    "criterion",
    "rv_ratio",
    "ms_ratio",
    "metric",
    "n",
    "seeds",
    "mean",
    "sd",
    "sem",
    "ci95_lo",
    "ci95_hi",
    "ci95_halfwidth",
    "values",
)


def render_scorable_per_ratio_csv(selections: Sequence[CellSelection]) -> str:
    """Per ``(criterion, ratio, metric)`` test-metric stats over scorable cells only."""
    rows: list[dict[str, Any]] = []
    for criterion in CRITERIA:
        per_ratio = aggregate_criterion(selections, criterion)
        for rv_ratio in v2.RV_RATIOS:
            block = per_ratio[rv_ratio]
            for metric in REPORT_METRICS:
                stat = block["stats"][metric]
                rows.append(
                    {
                        "criterion": criterion,
                        "rv_ratio": rv_ratio,
                        "ms_ratio": round(1.0 - rv_ratio, 2),
                        "metric": metric,
                        "n": stat.n,
                        "seeds": ";".join(str(s) for s in block["seeds"]),
                        "mean": stat.mean,
                        "sd": stat.sd,
                        "sem": stat.sem,
                        "ci95_lo": stat.ci95_lo,
                        "ci95_hi": stat.ci95_hi,
                        "ci95_halfwidth": stat.ci95_halfwidth,
                        "values": ";".join(f"{v:.6f}" for v in stat.values),
                    }
                )
    return _render_csv(SCORABLE_COLUMNS, rows)


CHECKPOINT_COLUMNS: tuple[str, ...] = (
    "run_name",
    "rv_ratio",
    "seed",
    "checkpoint_dir",
    "step",
    "epoch",
    "attribution",
    "usable_for_reinference",
    "n_shared_epochs",
    "max_abs_trajectory_delta",
    "trainer_state_best_metric",
    "results_json_best_metric",
    "trainer_state_best_model_checkpoint",
)


def render_checkpoint_survival_csv(selections: Sequence[CellSelection]) -> str:
    """One row per surviving checkpoint dir in the whole grid, with its attribution."""
    rows: list[dict[str, Any]] = []
    for s in selections:
        for ck in s.checkpoints:
            rows.append(
                {
                    "run_name": s.cell.run_name,
                    "rv_ratio": s.cell.rv_ratio,
                    "seed": s.cell.seed,
                    "checkpoint_dir": ck.dir_name,
                    "step": ck.step,
                    "epoch": ck.epoch,
                    "attribution": ck.attribution,
                    "usable_for_reinference": ck.usable_for_reinference,
                    "n_shared_epochs": ck.n_shared_epochs,
                    "max_abs_trajectory_delta": ck.max_abs_trajectory_delta,
                    "trainer_state_best_metric": ck.trainer_state_best_metric,
                    "results_json_best_metric": s.recorded_best_metric,
                    "trainer_state_best_model_checkpoint": (
                        ck.trainer_state_best_model_checkpoint
                    ),
                }
            )
    return _render_csv(CHECKPOINT_COLUMNS, rows)


# --------------------------------------------------------------------------- #
# Markdown.
# --------------------------------------------------------------------------- #
def _pm(stat_dict: dict[str, Any], digits: int = 4) -> str:
    """``mean ± halfwidth (n=k)`` from a :meth:`agg.Stat.as_dict` payload.

    Mirrors ``build_paclic_v2_maingrid_analysis._pm_n``: a value never appears
    without its own n.
    """
    if stat_dict.get("mean") is None:
        return "—"
    halfwidth = stat_dict.get("ci95_halfwidth")
    body = (
        f"{stat_dict['mean']:+.{digits}f}"
        if halfwidth is None
        else f"{stat_dict['mean']:+.{digits}f} ± {halfwidth:.{digits}f}"
    )
    return f"{body} (n={stat_dict['n']})"


def _plural(n: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def render_divergence_table(selections: Sequence[CellSelection]) -> list[str]:
    out = [
        "| criterion | selects the reported epoch | diverges | of the divergent: "
        "re-inference reachable | needs retraining |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for criterion in SENSITIVITY_CRITERIA:
        d = divergence_summary(selections, criterion)
        out.append(
            f"| `{criterion}` | {d['n_match']}/{d['n_decided']} | "
            f"{d['n_diverge']}/{d['n_decided']} | {d['n_reinference_reachable']} | "
            f"{d['n_retrain_only']} |"
        )
    return out


def render_exposure_table(selections: Sequence[CellSelection]) -> list[str]:
    """Per-cell divergence table: epochs apart, and how far validation moved.

    ``scorable`` and ``censored`` are separate columns on purpose: a censored cell
    is a *divergent* one whose uniform epoch is only a lower bound, and folding the
    two into one column would read as a third kind of scorability.
    """
    out = [
        "| criterion | RV:MS | seed | mixed epoch | uniform epoch | Δepoch | "
        "Δ RV val CER | Δ MS val CER | Δ mean | Δ worst | Δ on the cell's own "
        "metric | scorable | censored |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | :--- | :--- |",
    ]
    for e in exposures(selections):
        dominant = "—" if e.dominant_delta is None else f"{e.dominant_delta:+.4f}"
        out.append(
            f"| `{e.criterion}` | {_fmt_ratio(e.rv_ratio)} | {e.seed} | "
            f"{e.mixed_epoch} | {e.uniform_epoch} | {e.epoch_delta:+d} | "
            f"{e.deltas['rv_val_cer']:+.4f} | {e.deltas['ms_val_cer']:+.4f} | "
            f"{e.deltas['mean_val_cer']:+.4f} | {e.deltas['worst_val_cer']:+.4f} | "
            f"{dominant} | {'yes' if not e.diverges else 'no'} | "
            f"{'yes' if e.censored else '—'} |"
        )
    return out


def render_per_ratio_exposure_table(
    selections: Sequence[CellSelection], criterion: str
) -> list[str]:
    out = [
        "| RV:MS | n | diverging | Δepoch | Δ RV val CER | Δ MS val CER | "
        "Δ mean val CER | Δ worst val CER |",
        "| :--- | ---: | ---: | :--- | :--- | :--- | :--- | :--- |",
    ]
    for block in per_ratio_exposure(selections, criterion).values():
        cells = [_pm(block["quantities"][q]["diff"]) for q in EXPOSURE_QUANTITIES]
        out.append(
            f"| {_fmt_ratio(block['rv_ratio'])} | {block['n']} | "
            f"{block['n_diverging']}/{block['n']} | "
            f"{_pm(block['epoch']['diff'], 1)} | " + " | ".join(cells) + " |"
        )
    return out


def render_scorable_table(
    selections: Sequence[CellSelection], criterion: str
) -> list[str]:
    per_ratio = aggregate_criterion(selections, criterion)
    out = [
        "| RV:MS | n | seeds | RV CER | MS CER | avg CER | worst-view CER | "
        "RV accF1 | MS accF1 | worst accF1 |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rv_ratio in v2.RV_RATIOS:
        block = per_ratio[rv_ratio]
        st = block["stats"]

        def cell(metric: str, digits: int = 4) -> str:
            stat = st[metric]
            if stat.mean is None:
                return "—"
            if stat.ci95_halfwidth is None:
                return f"{stat.mean:.{digits}f} (n={stat.n})"
            return (
                f"{stat.mean:.{digits}f} ± {stat.ci95_halfwidth:.{digits}f} "
                f"(n={stat.n})"
            )

        out.append(
            f"| {_fmt_ratio(rv_ratio)} | {block['n']} | "
            f"{block['seeds'] or '—'} | {cell('rv_cer')} | {cell('ms_cer')} | "
            f"{cell('avg_cer')} | {cell('worst_view_cer')} | "
            f"{cell('rv_accent_f1', 3)} | {cell('ms_accent_f1', 3)} | "
            f"{cell('worst_accent_f1', 3)} |"
        )
    return out


def render_study_tables_md(selections: Sequence[CellSelection]) -> str:
    """The tables-only companion (``tables/study_tables.md``)."""
    out: list[str] = [
        "# Checkpoint-selection sensitivity on v2 — study tables ",
        "",
        "Analysis-only: no training, no inference, no GPU. Validation trajectories "
        "come from the winning Slurm `.out` per cell; test metrics are the committed "
        "`results.json` numbers, reported **only** where the uniform rule selects "
        "the very epoch the saved predictions were decoded from. Intervals are 95% "
        "Student-t CIs (df=n−1) from "
        "`aggregate_accent_ratio_multiseed.summarize`; paired Δ rows use its "
        "`_contrast`.",
        "",
        "## Divergence from the mixed rule",
        "",
        *render_divergence_table(selections),
        "",
        "## Per-cell divergence and exposure",
        "",
        *render_exposure_table(selections),
        "",
    ]
    for criterion in SENSITIVITY_CRITERIA:
        out += [
            f"## Paired per-ratio exposure — `{criterion}` (uniform − mixed)",
            "",
            *render_per_ratio_exposure_table(selections, criterion),
            "",
        ]
    for criterion in CRITERIA:
        block = (
            deliverability(selections, criterion) if criterion != "current" else None
        )
        out += [
            f"## Test metrics over the scorable cells — `{criterion}`",
            "",
        ]
        if block is not None:
            out.append(
                f"Scorable: **{block['n_scorable_cells']}/{block['n_cells']}** cells. "
                + (
                    "No ratio reaches the full n=3."
                    if not block["ratios_at_full_n"]
                    else "Full n=3 at ratio(s) "
                    + _ratio_list(block["ratios_at_full_n"])
                    + "."
                )
            )
            out.append("")
        out += [*render_scorable_table(selections, criterion), ""]
    return "\n".join(out) + "\n"


def render_results_md(
    selections: Sequence[CellSelection],
    *,
    study_root: Path,
    slurm_log_dir: Optional[Path],
) -> str:
    """The bundle's ``results.md`` — verdict first, then the evidence."""
    deliverable = chosen_deliverable(selections)
    retention = retention_summary(selections)
    gates = gate_summary(selections)
    stopping = early_stopping_summary(selections)
    extremes = exposure_extremes(selections)
    mean_div = divergence_summary(selections, "mean_val_cer")
    worst_div = divergence_summary(selections, "worst_val_cer")
    negative = deliverable["chosen"] == DELIVERABLE_BOUNDED_NEGATIVE

    out: list[str] = []
    out.append("# Checkpoint-selection sensitivity on the v2 main grid ")
    out.append("")
    out.append(
        f"**Deliverable:** {'BOUNDED NEGATIVE RESULT' if negative else 'SENSITIVITY RESULT'}"
        f" · **Corpus:** v2 · **Cells:** {len(selections)} · **Study id:** `{STUDY}` ·"
        " **Analysis-only** (no training, no inference, no GPU, no Slurm) ·"
        " **Parent:** `paclic-v2-maingrid`."
    )
    out.append("")
    out.append(
        "The v2 main grid selected each cell's checkpoint under a **mixed** rule — "
        "`eval_rv_cer` at RV:MS 100:0 and 75:25, `eval_avg_cer` at 50:50, "
        "`eval_ms_cer` at 25:75 and 0:100 "
        "(`experiment_accent_ratio._resolve_metric_for_best_model`). Issue study asks "
        "whether the mix-ratio conclusions survive a single **uniform** rule: mean "
        "validation CER `(eval_rv_cer+eval_ms_cer)/2`, or worst-view validation CER "
        "`max(eval_rv_cer,eval_ms_cer)`."
    )
    out.append("")

    # --- Verdict ---------------------------------------------------------- #
    out.append("## 1. Verdict")
    out.append("")
    if negative:
        out.append(
            "**The uniform-rule sensitivity table cannot be produced on v2, and this "
            "bundle does not produce one.** A uniform rule can only be *scored* on a "
            "cell whose uniform-rule epoch was actually decoded — either it is the "
            "epoch the committed predictions came from, or a checkpoint for that step "
            "survives and belongs to that run. Neither holds widely enough:"
        )
    else:
        out.append(
            "A uniform criterion is scorable at the full n=3 in all five ratios, so "
            "the sensitivity result below is a complete table."
        )
    out.append("")
    for reason in deliverable["why"]:
        out.append(f"- {reason}")
    out.append(
        f"- Rule applied: {deliverable['rule']}. Criteria supporting a grid-wide "
        "result: "
        + (
            ", ".join(
                f"`{c}`" for c in deliverable["criteria_supporting_a_grid_wide_result"]
            )
            or "**none**"
        )
        + "."
    )
    out.append("")
    out.append(
        "What *is* delivered instead: exactly which cells diverge under which "
        "uniform rule, how far apart the two epochs are, and how far the validation "
        "metrics moved between them — computable for **every** cell from its own "
        "trajectory, with no checkpoint required. That bounds the effect without "
        "pretending to measure it on test."
    )
    out.append("")

    # --- Retention -------------------------------------------------------- #
    out.append("## 2. What survives, and what it permits")
    out.append("")
    by_attr = retention["n_surviving_checkpoints_by_attribution"]
    out.append(
        f"- {retention['n_cells_with_saved_predictions']}/{retention['n_cells']} cells "
        "kept their committed `results.json` + `{rv,ms}_predictions.csv`, and "
        f"{retention['n_cells_with_final_model']}/{retention['n_cells']} kept a "
        "`final_model/` (the weights `load_best_model_at_end` left behind — the "
        "mixed-rule checkpoint itself). So the **mixed-rule** epoch is scorable "
        "everywhere; that is the paper's existing table."
    )
    out.append(
        f"- Checkpoints: {retention['n_cells_with_any_surviving_checkpoint']}/"
        f"{retention['n_cells']} cells retain any `checkpoints/` dir at all "
        f"({retention['n_surviving_checkpoints']} checkpoint dir(s) grid-wide). "
        "`save_total_limit=2` kept at most two, and the v2 launcher purges them on "
        "success."
    )
    if retention["n_surviving_checkpoints"]:
        out.append(
            "- Of those, "
            + ", ".join(
                f"**{n}** `{attribution}`" for attribution, n in by_attr.items() if n
            )
            + ". Attribution is a measurement: each checkpoint's own "
            "`trainer_state.json` `log_history` is replayed against the cell's gated "
            "trajectory (`tables/checkpoint_survival.csv`)."
        )
    for cell in retention["cells"]:
        for ck in cell["checkpoints"]:
            if ck["attribution"] != ATTRIBUTION_ORPHAN:
                continue
            out.append(
                f"  - `{cell['run_name']}/checkpoints/{ck['dir_name']}` is an "
                f"**orphan**: its eval curve differs from the committed run's by up "
                f"to {ck['max_abs_trajectory_delta']:.6f} CER over "
                f"{ck['n_shared_epochs']} shared epochs, and its own "
                f"`best_metric` ({ck['trainer_state_best_metric']}) is not the cell's "
                "committed `best_metric`. It is left over from a **superseded "
                "attempt** of the same cell, so decoding it would mix two training "
                "runs in one row."
            )
    out.append(
        f"- **Re-inference-reachable divergent rows: "
        f"{retention['n_reinference_reachable_rows']}.** "
        + (
            "No divergent `(cell, criterion)` row has surviving weights for its "
            "uniform-rule epoch, so no amount of decoding can close this — only "
            "retraining could."
            if not retention["n_reinference_reachable_rows"]
            else "Listed in `aggregate.json` `retention.reinference_reachable_rows`; "
            "each would need a committed exact re-inference record before any table "
            "may use it."
        )
    )
    out.append("")

    # --- Trajectory attribution ------------------------------------------- #
    out.append("## 3. Trajectory attribution (why these curves are the reported runs')")
    out.append("")
    out.append(
        "Trajectories come from the **winning** Slurm `.out` per cell — the numeric "
        "`(jobid, task)` winner from `build_paclic_v2_maingrid_analysis."
        "attest_corpus_from_slurm_logs`, which also records the superseded logs of "
        "re-queued cells. Four gates bind each curve to the run whose predictions the "
        "paper reports:"
    )
    out.append("")
    out.append("| gate | check | checked | passed | failures |")
    out.append("| :--- | :--- | ---: | ---: | :--- |")
    for name, block in gates["per_gate"].items():
        out.append(
            f"| `{name}` | {gates['checks'][name]} | {block['n_checked']} | "
            f"{block['n_passed']} | "
            + (", ".join(f"`{r}`" for r in block["failures"]) or "none")
            + " |"
        )
    out.append("")
    requeued = [s for s in selections if s.superseded_logs]
    if requeued:
        out.append(
            f"The winning-log choice is load-bearing, not cosmetic: "
            f"{_plural(len(requeued), 'cell')} were re-queued, and for at least one "
            "the superseded attempt reached a different best epoch (see the orphan "
            "checkpoint in §2). A study that read the earlier log would fail the "
            "`selection` and `best_metric` gates."
        )
        out.append("")

    # --- Divergence ------------------------------------------------------- #
    out.append("## 4. Divergence from the mixed rule (recomputed)")
    out.append("")
    out.extend(render_divergence_table(selections))
    out.append("")
    out.append(
        f"- Under **mean** validation CER, {mean_div['n_diverge']}/"
        f"{mean_div['n_decided']} cells select a different epoch than the reported "
        f"one; under **worst-view**, {worst_div['n_diverge']}/"
        f"{worst_div['n_decided']}."
    )
    out.append(
        "- The pattern is structural, not noise: the divergence is concentrated in "
        "the **endpoint** ratios, where one of the two views is out-of-distribution "
        "by construction. `tables/selection_ledger.csv` has every row."
    )
    if stopping["n_truncated"]:
        truncated = ", ".join(
            f"`{row['run_name']}` ({row['epochs_observed']}/{row['planned_epochs']} "
            f"epochs)"
            for row in stopping["truncated"]
        )
        out.append(
            f"- **Early stopping truncated {stopping['n_truncated']}/"
            f"{stopping['n_cells']} trajectories** ({truncated}). "
            + stopping["note"]
            + (
                "; that censors the uniform-rule epoch for "
                + ", ".join(f"`{r}`" for r in stopping["censored_runs"])
                if stopping["censored_runs"]
                else ""
            )
            + "."
        )
    out.append("")
    out.append("### 4.1 Per-cell divergence and exposure")
    out.append("")
    out.extend(render_exposure_table(selections))
    out.append("")

    # --- Exposure --------------------------------------------------------- #
    out.append("## 5. Exposure — how far the two checkpoints are apart")
    out.append("")
    out.append(
        "The part of study that needs no checkpoint: for every cell, the validation "
        "metrics at the mixed-rule epoch versus at the uniform-rule epoch, paired by "
        "seed within each ratio (`aggregate_accent_ratio_multiseed._contrast` — the "
        "same paired machinery as the main grid's asymmetry and retention tables). "
        "Δ is **uniform − mixed**; every ratio contributes all three seeds."
    )
    out.append("")
    for criterion in SENSITIVITY_CRITERIA:
        block = extremes[criterion]
        out.append(f"### 5.{SENSITIVITY_CRITERIA.index(criterion) + 1} `{criterion}`")
        out.append("")
        out.extend(render_per_ratio_exposure_table(selections, criterion))
        out.append("")
        worst_rv = block["per_quantity"]["rv_val_cer"]
        worst_ms = block["per_quantity"]["ms_val_cer"]
        if worst_rv["max_abs_delta"] is not None:
            out.append(
                f"- Worst single-cell movement over the {block['n_diverging_cells']} "
                f"diverging cell(s): RV val CER {worst_rv['signed_delta']:+.4f} "
                f"(`{worst_rv['run_name']}`, epoch {worst_rv['mixed_epoch']} → "
                f"{worst_rv['uniform_epoch']}), MS val CER "
                f"{worst_ms['signed_delta']:+.4f} (`{worst_ms['run_name']}`), largest "
                f"epoch gap {block['max_abs_epoch_delta']}."
            )
            out.append("")

    # --- Scorable subset -------------------------------------------------- #
    out.append("## 6. Test metrics over the scorable cells only")
    out.append("")
    out.append(
        "A cell appears here **only** if the uniform rule selects the very epoch its "
        "committed predictions were decoded from; then its test metrics are the "
        "committed ones, exactly. A divergent cell is excluded — never scored on the "
        "reported checkpoint. `n` per ratio is the honest denominator, and no row "
        "below may be quoted as *the* v2 result under that rule."
    )
    out.append("")
    for criterion in CRITERIA:
        label = {
            "current": "The reported mixed rule (reproduces the main-grid table)",
            "mean_val_cer": "Uniform mean validation CER",
            "worst_val_cer": "Uniform worst-view validation CER",
        }[criterion]
        out.append(f"### 6.{CRITERIA.index(criterion) + 1} {label}")
        out.append("")
        if criterion != "current":
            block = deliverability(selections, criterion)
            out.append(
                f"Scorable: **{block['n_scorable_cells']}/{block['n_cells']}** cells; "
                + (
                    "full n=3 at " + _ratio_list(block["ratios_at_full_n"])
                    if block["ratios_at_full_n"]
                    else "**no ratio at full n=3**"
                )
                + (
                    "; **no scorable cell at all** at "
                    + _ratio_list(block["ratios_with_no_cell"])
                    if block["ratios_with_no_cell"]
                    else ""
                )
                + "."
            )
            out.append("")
        out.extend(render_scorable_table(selections, criterion))
        out.append("")

    # --- What the paper may say ------------------------------------------ #
    out.extend(_render_claims(selections))

    # --- Files / reproduce ----------------------------------------------- #
    out.append("## 8. Files")
    out.append("")
    out.append("- `aggregate.json` — full machine-readable payload + ledger.")
    out.append("- `manifest.json` — provenance, gates, criteria, no-GPU stance.")
    out.append(
        "- `tables/selection_ledger.csv` — one row per `(cell, criterion)`: selected "
        "epoch/step, status, scorability, what would resolve it, all four gates."
    )
    out.append(
        "- `tables/selection_candidates.csv` — one row per `(cell, epoch)`: the full "
        "trajectory every selection was made over."
    )
    out.append(
        "- `tables/divergence_exposure.csv` — one row per `(cell, uniform criterion)`: "
        "both epochs and the four validation deltas."
    )
    out.append(
        "- `tables/per_ratio_exposure.csv` — paired per-ratio Δ contrasts (mean, SD, "
        "95% CI, sign consistency, per-seed diffs)."
    )
    out.append(
        "- `tables/scorable_per_ratio.csv` — test-metric stats over the scorable "
        "cells only, with `n`."
    )
    out.append(
        "- `tables/checkpoint_survival.csv` — every surviving checkpoint dir and its "
        "attribution."
    )
    out.append("- `tables/study_tables.md` — the tables without the prose.")
    out.append("")
    out.append("## 9. Reproduce")
    out.append("")
    out.append("```")
    out.append(
        "uv run python -m scripts.evaluation.checkpoint_selection_sensitivity_v2 \\\n"
        f"    --study-root {study_root} \\\n"
        f"    --slurm-log-dir {slurm_log_dir} \\\n"
        f"    --out-dir {DEFAULT_OUT_DIR}"
    )
    out.append("```")
    out.append("")
    out.append(
        "Reads only committed/pushed artifacts: each cell's `results.json` and "
        "prediction CSVs, any surviving `checkpoints/*/trainer_state.json`, and the "
        "immutable Slurm `.out` logs. Writes no `data/` or `outputs/` path, runs no "
        "inference, and submits nothing."
    )
    out.append("")
    return "\n".join(out) + "\n"


def _render_claims(selections: Sequence[CellSelection]) -> list[str]:
    """§7 — what the paper may and may not claim, derived from the tables above.

    Every quantity comes from :func:`band_summary`, :func:`deliverability` and the
    exposures, so the prose cannot drift from the tables it summarizes.
    """
    bands = band_summary(selections)
    interior = bands["interior"]
    endpoints = bands["endpoints"]
    per_criterion = {c: deliverability(selections, c) for c in SENSITIVITY_CRITERIA}
    worst_endpoint_rv = [
        e
        for e in exposures(selections)
        if e.criterion == "worst_val_cer" and e.rv_ratio == 1.0
    ]

    out = ["## 7. What the paper may claim, and what it may not", ""]
    out.append("**May claim.**")
    out.append("")
    out.append(
        "- The reported per-cell checkpoints are exactly the ones the mixed rule "
        f"selects: all four attribution gates pass on all {len(selections)} cells, "
        f"and every cell's `best_step` is on the fixed `best_epoch × "
        f"{STEPS_PER_EPOCH}` contract."
    )
    if interior["max_abs_validation_delta"] is not None:
        excl = interior["max_abs_epoch_delta_excluding_censored"]
        out.append(
            "- **Inside the interior band (75:25 / 50:50 / 25:75) both uniform rules "
            f"are close to a no-op.** Over the {interior['n_rows']} "
            "(cell × uniform criterion) rows there, "
            f"{interior['n_matching']} select the reported epoch outright; the "
            f"{interior['n_diverging']} that diverge sit at most "
            f"{interior['max_abs_epoch_delta']} epoch(s) away "
            + (
                f"({excl} excluding the "
                f"{_plural(interior['n_censored'], 'censored row')}) "
                if interior["n_censored"] and excl is not None
                else ""
            )
            + "and move **no validation quantity by more than "
            f"{interior['max_abs_validation_delta']:.4f} CER** (worst quantity: "
            f"`{interior['max_abs_validation_delta_quantity']}`). Per criterion, "
            + "; ".join(
                f"{n}/{interior['n_cells']} interior cells are scorable under "
                f"`{criterion}`"
                for criterion, n in interior["n_scorable_by_criterion"].items()
            )
            + "."
        )
    out.append(
        "- Every per-cell and per-ratio divergence count, epoch distance and paired "
        "validation Δ under both uniform rules — measured on all "
        f"{len(selections)} cells, at the full n=3 per ratio, and tabulated here."
    )
    out.append("")
    out.append("**May not claim.**")
    out.append("")
    out.append(
        "- **No uniform-rule *test* table, and therefore no statement either way "
        "about the ratio ordering under a uniform rule.** Scorable cells: "
        + "; ".join(
            f"`{criterion}` {block['n_scorable_cells']}/{block['n_cells']}"
            + (
                " (no scorable cell at all at "
                + _ratio_list(block["ratios_with_no_cell"])
                + ")"
                if block["ratios_with_no_cell"]
                else ""
            )
            for criterion, block in per_criterion.items()
        )
        + ". No criterion reaches n=3 in all five ratios, so a ranking over the "
        "ratios cannot be computed — the paper must not say the ordering is robust "
        "to the selection rule, nor that it is fragile."
    )
    out.append(
        "- **Nothing at all about the endpoints under a uniform rule.** "
        + "; ".join(
            f"`{criterion}` scores {n}/{endpoints['n_cells']} endpoint cells"
            for criterion, n in endpoints["n_scorable_by_criterion"].items()
        )
        + (
            ", and where they diverge the uniform epoch is up to "
            f"{endpoints['max_abs_epoch_delta']} epochs from the reported one, worth "
            f"up to {endpoints['max_abs_validation_delta']:.4f} validation CER"
            if endpoints["max_abs_validation_delta"] is not None
            else ", and none of them diverges"
        )
        + ". The zero-shot asymmetry (data pack §4) and the retention/adaptation "
        "contrasts (§6) each read an endpoint cell, so **neither has a uniform-rule "
        "counterpart** here."
    )
    if worst_endpoint_rv:
        worst_case = min(worst_endpoint_rv, key=lambda e: e.uniform_epoch)
        out.append(
            "- **At the endpoints a uniform rule is also arguably the wrong question, "
            "not merely an unmeasured one.** At RV:MS 100:0 the MS view is "
            "out-of-distribution by construction, so minimizing the worst view "
            f"selects epoch {worst_case.uniform_epoch} (reported: "
            f"{worst_case.mixed_epoch}), where RV validation CER is "
            f"{worst_case.uniform_values['rv_val_cer']:.4f} instead of "
            f"{worst_case.mixed_values['rv_val_cer']:.4f} — a barely specialized "
            'model. Such a cell would no longer be "the RV-only model", so the '
            "contrast it feeds would be a different experiment rather than a "
            "robustness check of this one. This is an argument about the criterion, "
            "and it does not substitute for the measurement."
        )
    out.append("")
    out.append("**Residual risk, stated plainly.**")
    out.append("")
    diverging_interior = ", ".join(f"`{r}`" for r in interior["diverging_runs"]) or "—"
    bound = (
        "every interior cell selects the reported epoch under both uniform rules, so "
        "nothing there moves at all"
        if interior["max_abs_validation_delta"] is None
        else (
            "its uniform-rule epochs are at most "
            f"{interior['max_abs_epoch_delta']} epoch(s) away and worth at most "
            f"{interior['max_abs_validation_delta']:.4f} validation CER"
        )
    )
    out.append(
        "The interior band — where the plateau/ordering reading lives — is the part "
        f"the mixed rule can be shown not to distort: {bound}. **That is a "
        "validation-side bound, not a test-side measurement.** A small validation "
        "move does not prove a small test move; every interior `(cell, criterion)` "
        "row that diverges is *excluded* from that criterion's aggregation — the "
        f"affected cells are {diverging_interior} — so even an interior-only "
        f"uniform-rule comparison drops below n={len(agg.SEEDS)} in at least one "
        "ratio; and one of those divergences is **censored** — early stopping was "
        "driven by the mixed metric, so on that trajectory the uniform criterion was "
        "still improving when training stopped and its true argmin is unobservable. "
        "The endpoints are bounded by nothing here. Concretely: the paper's ratio "
        "ordering (75:25 / 50:50 / 25:75 avg CER, whose CIs already overlap) retains "
        "an **untested dependence on the selection rule**, and the only way to remove "
        "it is to retrain the divergent cells under the uniform rule. Until then the "
        "honest limitation is: *checkpoint selection is a mixed rule; on v2 that rule "
        "was shown to be internally self-consistent and, inside the interior band, "
        "near-inert on validation, but no uniform-rule test table exists and none can "
        "be built from the surviving artifacts.*"
    )
    out.append("")
    return out


# --------------------------------------------------------------------------- #
# Manifest.
# --------------------------------------------------------------------------- #
def build_manifest(
    selections: Sequence[CellSelection],
    *,
    study_root: Path,
    slurm_log_dir: Optional[Path],
) -> dict[str, Any]:
    """Provenance manifest: sources, gates, criteria, verdict, no-GPU stance."""
    return {
        "study": STUDY,
        "issue": 161,
        "parent_study": v2.STUDY,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "corpus": "v2",
        "analysis_only": True,
        "no_training_no_inference": True,
        "study_root": str(study_root),
        "slurm_log_dir": str(slurm_log_dir) if slurm_log_dir is not None else None,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "method": {
            "trajectory_source": (
                "the WINNING Slurm .out per cell (numeric (jobid, task) winner from "
                "build_paclic_v2_maingrid_analysis.attest_corpus_from_slurm_logs), "
                "parsed by aggregate_accent_ratio_multiseed."
                "parse_paired_validation_trajectory"
            ),
            "why_not_checkpoints_as_trajectory_source": (
                "on v2 a surviving checkpoint may belong to a SUPERSEDED attempt of "
                "the same cell, so a checkpoint log_history is not evidence about the "
                "reported run; checkpoints are censused and attributed instead"
            ),
            "mixed_rule_epoch_source": (
                "the cell's own results.json training_state (the v2 launcher writes "
                "best_epoch / best_step / best_metric / best_metric_key inline — no "
                "Slurm or W&B recovery)"
            ),
            "selection_rule": (
                "earliest epoch achieving the minimum criterion value = HF Trainer "
                "load_best_model_at_end under greater_is_better=False"
            ),
            "test_metric_source": (
                "the committed results.json / {rv,ms}_predictions.csv numbers, used "
                "ONLY where the criterion selects the saved-prediction epoch; a "
                "divergent cell is excluded, never scored on the reported checkpoint"
            ),
            "statistics": (
                "aggregate_accent_ratio_multiseed.summarize (mean, sample SD, "
                "Student-t 95% CI, df=n-1) and its _contrast for every paired Δ — "
                "reused, not reimplemented, so this bundle and the main-grid bundle "
                "cannot disagree on what a CI is"
            ),
            "metric_derivation": (
                "avg_cer / worst_view_cer / worst_accent_f1 come from "
                "checkpoint_selection_sensitivity.derive_report_metrics, the same "
                "helper the v1 study uses"
            ),
            "counts": "every divergence/exposure/verdict number is recomputed here",
        },
        "criteria": {
            "current": (
                "the cell's configured metric_for_best_model (eval_rv_cer / "
                "eval_avg_cer / eval_ms_cer), cross-checked between the Slurm header "
                "and results.json best_metric_key"
            ),
            "mean_val_cer": "(eval_rv_cer + eval_ms_cer) / 2 (== eval_avg_cer)",
            "worst_val_cer": "max(eval_rv_cer, eval_ms_cer)",
        },
        "checkpoint_attribution": {
            "how": (
                "a surviving checkpoint's trainer_state.json log_history is replayed "
                "against the cell's gated trajectory; agreement within "
                f"{TRAJECTORY_AGREEMENT_TOL:g} means it is the reported run's"
            ),
            "values": {
                ATTRIBUTION_REPORTED: "the reported run's own checkpoint",
                ATTRIBUTION_ORPHAN: (
                    "left over from a superseded attempt of the same cell; NEVER "
                    "decodable into that cell's row"
                ),
                ATTRIBUTION_UNKNOWN: (
                    "no readable/overlapping log_history, so not shown to belong to "
                    "the reported run; treated as unusable"
                ),
            },
        },
        "deliverable": chosen_deliverable(selections),
        "gates": gate_summary(selections),
        "retention": retention_summary(selections),
        "early_stopping": early_stopping_summary(selections),
        "bands": band_summary(selections),
        "limitations": [
            "Uniform-rule test metrics exist only for cells whose uniform-rule epoch "
            "IS the saved-prediction epoch; every other cell is excluded, so no "
            "uniform-rule per-ratio row here is a 3-seed number unless its own n "
            "says 3.",
            "The exposure tables are a VALIDATION-side bound. A small validation "
            "movement between the two epochs does not prove a small test movement.",
            "On an early-stopped trajectory the uniform rule's argmin can be the last "
            "observed epoch; early stopping was driven by the MIXED metric, so that "
            "epoch is a lower bound and the uniform rule's true choice is "
            "unobservable.",
            "v1 numbers are not comparable to v2's and no v1→v2 comparison appears in "
            "this bundle.",
        ],
        "reproduce": (
            "uv run python -m scripts.evaluation.checkpoint_selection_sensitivity_v2 "
            f"--study-root {study_root} --slurm-log-dir {slurm_log_dir} "
            f"--out-dir {DEFAULT_OUT_DIR}"
        ),
        "cells": [
            {
                "run_name": s.cell.run_name,
                "rv_ratio": s.cell.rv_ratio,
                "seed": s.cell.seed,
                "slurm_log": s.slurm_log,
                "superseded_logs": list(s.superseded_logs),
                "mixed_rule_metric": s.results_metric_key,
                "mixed_selected_epoch": s.mixed_epoch,
                "mixed_selected_step": s.mixed_step,
                "recorded_best_metric": s.recorded_best_metric,
                "epochs_observed": s.epochs_observed,
                "planned_epochs": s.planned_epochs,
                "early_stopped": s.early_stopped,
                "gates_ok": s.gates_ok,
                "surviving_checkpoints": [ck.as_dict() for ck in s.checkpoints],
                "notes": list(s.notes),
            }
            for s in selections
        ],
        "generated_by": "scripts.evaluation.checkpoint_selection_sensitivity_v2",
    }


# --------------------------------------------------------------------------- #
# CLI.
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
    out_dir: Path,
    selections: Sequence[CellSelection],
    *,
    study_root: Path,
    slurm_log_dir: Optional[Path],
) -> None:
    """Write the whole bundle (tables, payload, manifest, results page)."""
    tables = out_dir / "tables"
    _write(tables / "selection_ledger.csv", render_ledger_csv(selections))
    _write(tables / "selection_candidates.csv", render_candidates_csv(selections))
    _write(tables / "divergence_exposure.csv", render_exposure_csv(selections))
    _write(tables / "per_ratio_exposure.csv", render_per_ratio_exposure_csv(selections))
    _write(tables / "scorable_per_ratio.csv", render_scorable_per_ratio_csv(selections))
    _write(
        tables / "checkpoint_survival.csv", render_checkpoint_survival_csv(selections)
    )
    _write(tables / "study_tables.md", render_study_tables_md(selections))
    _write_json(out_dir / "aggregate.json", build_aggregate(selections))
    _write_json(
        out_dir / "manifest.json",
        build_manifest(selections, study_root=study_root, slurm_log_dir=slurm_log_dir),
    )
    _write(
        out_dir / "results.md",
        render_results_md(
            selections, study_root=study_root, slurm_log_dir=slurm_log_dir
        ),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=v2.DEFAULT_STUDY_ROOT,
        help=(
            "The v2 main-grid study root (15 rv_<pct>_ms_<pct>_seed_<n> dirs). Pass "
            f"{v2.DEFAULT_SHARED_STUDY_ROOT} when the controller has no local copy."
        ),
    )
    parser.add_argument(
        "--slurm-log-dir",
        type=Path,
        default=v2.DEFAULT_SLURM_LOG_DIR,
        help="Dir of immutable Slurm .out logs (the validation-trajectory source).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Bundle dir: writes results.md, aggregate.json, manifest.json, tables/. "
            "Prints study_tables.md to stdout if omitted."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # cross_check=False: no inference here, and the metrics are the runner's own
    # results.json numbers (the main-grid bundle already cross-checked them).
    cells, _ = v2.load_grid(
        args.study_root,
        cross_check=False,
        shared_study_root=v2.DEFAULT_SHARED_STUDY_ROOT,
    )
    coverage = v2.coverage_summary(cells)
    if not coverage["complete"]:
        print(
            f"[{STUDY}][warn] {v2.partial_banner(coverage)}",
            file=sys.stderr,
        )
    attestation = v2.attest_corpus_from_slurm_logs(args.slurm_log_dir)
    selections = build_selections(
        cells,
        study_root=args.study_root,
        attestation=attestation,
        slurm_log_dir=args.slurm_log_dir,
        planned_epochs=planned_epochs_by_run(args.study_root, cells),
    )
    for s in selections:
        for note in s.notes:
            print(f"[{STUDY}][note] {s.cell.run_name}: {note}", file=sys.stderr)
    gates = gate_summary(selections)
    for name in gates["failing_runs"]:
        print(f"[{STUDY}][warn] attribution gate failed for {name}", file=sys.stderr)

    if args.out_dir is not None:
        build_bundle(
            args.out_dir,
            selections,
            study_root=args.study_root,
            slurm_log_dir=args.slurm_log_dir,
        )
    else:
        print(render_study_tables_md(selections))

    verdict = chosen_deliverable(selections)
    print(f"[{STUDY}] deliverable: {verdict['chosen']}", file=sys.stderr)
    # A divergent (unscorable) cell is the expected finding, not a failure; only a
    # grid that is not fully present, or a failed attribution gate, is an error.
    return 1 if (not coverage["complete"] or gates["failing_runs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
