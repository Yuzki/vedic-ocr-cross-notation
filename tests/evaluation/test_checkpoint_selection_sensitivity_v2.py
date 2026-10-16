"""Test checkpoint selection sensitivity v2: PACLIC 2026 v2 release."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import pytest

from scripts.evaluation import aggregate_accent_ratio_multiseed as agg
from scripts.evaluation import build_paclic_v2_maingrid_analysis as v2
from scripts.evaluation import selection_helpers as v1
from scripts.evaluation import checkpoint_selection_sensitivity_v2 as css2

TOTAL = 10420
PLANNED_EPOCHS = 20

#: Per-ratio (rv_cer, ms_cer) test metrics, with the endpoint asymmetry baked in.
_TEST_SPEC: dict[float, tuple[float, float]] = {
    1.0: (0.037, 0.264),
    0.75: (0.040, 0.041),
    0.5: (0.041, 0.037),
    0.25: (0.044, 0.036),
    0.0: (0.297, 0.036),
}


# --------------------------------------------------------------------------- #
# Synthetic trajectories.
# --------------------------------------------------------------------------- #
def _cand(epoch: int, rv: float, ms: float) -> agg.SelectionCandidate:
    return agg.SelectionCandidate(
        epoch=epoch,
        step=epoch * agg.STEPS_PER_EPOCH,
        rv_val_cer=rv,
        ms_val_cer=ms,
    )


def _endpoint_curve(
    n_epochs: int = PLANNED_EPOCHS, *, rv_only: bool = True
) -> list[agg.SelectionCandidate]:
    """A single-notation cell: the in-domain view improves, the other degrades faster.

    So the mixed rule (the dominant view's CER) picks the last epoch while mean and
    worst-view both pick the first — the real endpoint pattern, in miniature, and the
    reason a uniform rule at an endpoint returns a barely specialized model.
    ``rv_only=False`` mirrors it for the MS-only endpoint.
    """
    return [
        _cand(
            epoch,
            0.20 - 0.008 * epoch if rv_only else 0.20 + 0.012 * epoch,
            0.20 + 0.012 * epoch if rv_only else 0.20 - 0.008 * epoch,
        )
        for epoch in range(1, n_epochs + 1)
    ]


def _interior_curve(
    n_epochs: int = PLANNED_EPOCHS, *, best: int = 18
) -> list[agg.SelectionCandidate]:
    """An interior cell: both views bottom out at ``best`` and every rule agrees."""
    return [
        _cand(
            epoch, 0.05 + 0.0002 * abs(epoch - best), 0.05 + 0.0003 * abs(epoch - best)
        )
        for epoch in range(1, n_epochs + 1)
    ]


def _log_text(
    run_name: str,
    candidates: list[agg.SelectionCandidate],
    *,
    metric: str,
    job: int,
    task: int,
) -> str:
    """A Slurm ``.out`` the v2 attestation parser and the trajectory parser both read."""
    header = (
        f"[paclic-v2-maingrid] corpus CSVs + first images present: "
        f"rv=data/datasets/rv_v2 ms=data/datasets/ms_v2\n"
        f"[paclic-v2-maingrid] task={task}/14 run_name={run_name}  smoke=0\n"
        f"[paclic-v2-maingrid] node=gpu-worker  job={job} array={job}_{task}\n"
        f"[paclic-v2-maingrid] corpus=v2 rv=data/datasets/rv_v2 "
        f"ms=data/datasets/ms_v2 data_source=csv\n"
        f"[paclic-v2-maingrid] output_dir=./outputs/experiments_paclic_v2_maingrid\n"
        f"Eval targets: ['rv', 'ms'] | metric_for_best_model: {metric}\n"
        "Starting training\n"
    )
    body = "".join(
        f"{{'eval_rv_loss': 0.1, 'eval_rv_cer': {c.rv_val_cer!r}, "
        f"'epoch': {float(c.epoch)!r}}}\n"
        f"{{'eval_ms_loss': 0.1, 'eval_ms_cer': {c.ms_val_cer!r}, "
        f"'epoch': {float(c.epoch)!r}}}\n"
        for c in candidates
    )
    last = candidates[-1].epoch if candidates else 0
    return header + body + f"{{'train_runtime': 100.0, 'epoch': {float(last)!r}}}\n"


def _mixed_metric(rv_ratio: float) -> str:
    """The rule ``_resolve_metric_for_best_model`` resolves for this ratio."""
    if rv_ratio > 0.5:
        return "eval_rv_cer"
    if rv_ratio < 0.5:
        return "eval_ms_cer"
    return "eval_avg_cer"


def _mixed_epoch(candidates: list[agg.SelectionCandidate], metric: str) -> int:
    field = agg.METRIC_TO_CANDIDATE_FIELD[metric]
    chosen = agg.select_earliest_min(candidates, field)
    assert chosen is not None
    return chosen.epoch


# --------------------------------------------------------------------------- #
# Synthetic study root.
# --------------------------------------------------------------------------- #
def _view(cer: float) -> dict[str, float]:
    return {
        "cer": cer,
        "wer": cer * 3,
        "chrf": 90.0,
        "exact_match": 0.17,
        "full_cer": cer,
        "accent_stripped_cer": cer / 2,
        "accent_precision": 0.97,
        "accent_recall": 0.92,
        "accent_f1": 0.94,
    }


def _write_cell(
    root: Path,
    rv_ratio: float,
    seed: int,
    candidates: list[agg.SelectionCandidate],
    *,
    metric: str,
    best_epoch: int | None = None,
    best_step: int | None = None,
    best_metric: float | None = None,
    best_metric_key: str | None = None,
    planned_epochs: int = PLANNED_EPOCHS,
) -> Path:
    """One v2 run dir: ``results.json`` with inline ``training_state`` + predictions."""
    name = v2.run_name(rv_ratio, seed)
    cell_dir = root / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    epoch = best_epoch if best_epoch is not None else _mixed_epoch(candidates, metric)
    by_epoch = {c.epoch: c for c in candidates}
    field = agg.METRIC_TO_CANDIDATE_FIELD[metric]
    value = (
        best_metric
        if best_metric is not None
        else (by_epoch[epoch].value(field) if epoch in by_epoch else None)
    )
    rv_cer, ms_cer = _TEST_SPEC[rv_ratio]
    jitter = (seed - 43) * 0.001
    n_rv = int(TOTAL * rv_ratio)
    payload = {
        "experiment_name": name,
        "seed": seed,
        "model_base": "fixed-foundation",
        "freeze_mode": "none",
        "input_scheme": "iso",
        "num_train_epochs": planned_epochs,
        "training_state": {
            "best_epoch": float(epoch),
            "best_step": (
                best_step if best_step is not None else epoch * agg.STEPS_PER_EPOCH
            ),
            "global_step": candidates[-1].step if candidates else None,
            "final_epoch": float(candidates[-1].epoch) if candidates else None,
            "best_metric": value,
            "best_metric_key": (
                best_metric_key if best_metric_key is not None else metric
            ),
        },
        "mix_mode": "parallel_by_text",
        "key_column": "text",
        "selection_algorithm": "sorted_common_keys+seeded_shuffle+prefix",
        "parallel_keys_total": TOTAL,
        "parallel_keys_used": TOTAL,
        "parallel_subset_seed": seed,
        "key_list_sha256": f"sha-{seed}",
        "total_train_samples_requested": TOTAL,
        "rv_train_samples": n_rv,
        "ms_train_samples": TOTAL - n_rv,
        "rv_test": _view(rv_cer + jitter),
        "ms_test": _view(ms_cer + jitter),
        "avg_cer": (rv_cer + ms_cer) / 2 + jitter,
    }
    (cell_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    for view in ("rv", "ms"):
        (cell_dir / f"{view}_predictions.csv").write_text(
            "file_name,reference,prediction\na.png,x,x\n", encoding="utf-8"
        )
    return cell_dir


def _write_checkpoint(
    cell_dir: Path,
    step: int,
    candidates: list[agg.SelectionCandidate],
    *,
    best_metric: float = 0.04,
    best_model_checkpoint: str = "checkpoint-1",
) -> Path:
    """A surviving ``checkpoints/checkpoint-<step>/trainer_state.json``."""
    path = cell_dir / "checkpoints" / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    log_history: list[dict[str, float]] = []
    for c in candidates:
        log_history.append({"eval_rv_cer": c.rv_val_cer, "epoch": float(c.epoch)})
        log_history.append({"eval_ms_cer": c.ms_val_cer, "epoch": float(c.epoch)})
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "best_metric": best_metric,
                "best_model_checkpoint": best_model_checkpoint,
                "log_history": log_history,
            }
        ),
        encoding="utf-8",
    )
    return path


def _curve_for(rv_ratio: float, seed: int) -> list[agg.SelectionCandidate]:
    """Endpoint ratios get the diverging curve, interior ratios the agreeing one."""
    if rv_ratio in (1.0, 0.0):
        return _endpoint_curve(rv_only=rv_ratio == 1.0)
    return _interior_curve(best=18 - (seed - 42))


@pytest.fixture
def grid(tmp_path: Path) -> tuple[Path, Path]:
    """A complete synthetic 15-cell grid with one healthy log per cell."""
    root = tmp_path / "study"
    logs = tmp_path / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    task = 0
    for rv_ratio in v2.RV_RATIOS:
        for seed in v2.SEEDS:
            candidates = _curve_for(rv_ratio, seed)
            metric = _mixed_metric(rv_ratio)
            _write_cell(root, rv_ratio, seed, candidates, metric=metric)
            (logs / f"paclic_v2_maingrid_600_{task}.out").write_text(
                _log_text(
                    v2.run_name(rv_ratio, seed),
                    candidates,
                    metric=metric,
                    job=600,
                    task=task,
                ),
                encoding="utf-8",
            )
            task += 1
    return root, logs


def _resolve(root: Path, logs: Path) -> list[css2.CellSelection]:
    cells, _ = v2.load_grid(root, cross_check=False)
    return css2.build_selections(
        cells,
        study_root=root,
        attestation=v2.attest_corpus_from_slurm_logs(logs),
        slurm_log_dir=logs,
        planned_epochs=css2.planned_epochs_by_run(root, cells),
    )


def _by_run(selections: list[css2.CellSelection]) -> dict[str, css2.CellSelection]:
    return {s.cell.run_name: s for s in selections}


# --------------------------------------------------------------------------- #
# 1. Gates.
# --------------------------------------------------------------------------- #
def test_all_four_gates_pass_on_a_healthy_grid(grid):
    selections = _resolve(*grid)
    assert len(selections) == 15
    gates = css2.gate_summary(selections)
    assert gates["all_ok"] is True
    assert gates["failing_runs"] == []
    for name, block in gates["per_gate"].items():
        assert block["n_checked"] == 15, name
        assert block["n_passed"] == 15, name
        assert block["failures"] == []


def test_selection_gate_fails_when_the_log_is_another_runs_curve(grid, tmp_path):
    """A trajectory that does not reproduce the recorded best epoch is untrustworthy."""
    root, logs = grid
    name = "rv_50_ms_50_seed_43"
    # Rewrite that cell's log with a curve whose argmin is a different epoch.
    other = _interior_curve(best=4)
    (logs / "paclic_v2_maingrid_600_7.out").write_text(
        _log_text(name, other, metric="eval_avg_cer", job=600, task=7),
        encoding="utf-8",
    )
    selection = _by_run(_resolve(root, logs))[name]
    assert selection.selection_gate_ok is False
    assert selection.gates_ok is False
    assert any("selection gate FAILED" in n for n in selection.notes)
    # Both uniform criteria are deferred regardless of any epoch coincidence.
    for criterion in css2.SENSITIVITY_CRITERIA:
        assert selection.decisions[criterion].status == agg.STATUS_DEFERRED
        assert css2.is_scorable(selection, criterion) is False


def test_best_metric_gate_fails_on_a_mismatched_recorded_value(tmp_path):
    root, logs = tmp_path / "study", tmp_path / "slurm"
    logs.mkdir(parents=True)
    candidates = _interior_curve()
    _write_cell(root, 0.5, 42, candidates, metric="eval_avg_cer", best_metric=0.99999)
    (logs / "paclic_v2_maingrid_600_0.out").write_text(
        _log_text(
            v2.run_name(0.5, 42), candidates, metric="eval_avg_cer", job=600, task=0
        ),
        encoding="utf-8",
    )
    cells, _ = v2.load_grid(root, cross_check=False, seeds=(42,), rv_ratios=(0.5,))
    selection = css2.build_selections(
        cells,
        study_root=root,
        attestation=v2.attest_corpus_from_slurm_logs(logs),
        slurm_log_dir=logs,
    )[0]
    assert selection.best_metric_gate_ok is False
    assert selection.gates_ok is False


def test_metric_key_and_step_contract_gates_fail_independently(tmp_path):
    root, logs = tmp_path / "study", tmp_path / "slurm"
    logs.mkdir(parents=True)
    candidates = _interior_curve()
    # results.json claims a different selection metric than the log header, and a
    # best_step off the fixed 326-steps-per-epoch contract.
    _write_cell(
        root,
        0.5,
        42,
        candidates,
        metric="eval_avg_cer",
        best_metric_key="eval_rv_cer",
        best_step=99,
    )
    (logs / "paclic_v2_maingrid_600_0.out").write_text(
        _log_text(
            v2.run_name(0.5, 42), candidates, metric="eval_avg_cer", job=600, task=0
        ),
        encoding="utf-8",
    )
    cells, _ = v2.load_grid(root, cross_check=False, seeds=(42,), rv_ratios=(0.5,))
    selection = css2.build_selections(
        cells,
        study_root=root,
        attestation=v2.attest_corpus_from_slurm_logs(logs),
        slurm_log_dir=logs,
    )[0]
    assert selection.metric_key_gate_ok is False
    assert selection.step_contract_ok is False


def test_raw_training_state_recovers_the_fields_the_shared_loader_drops(grid):
    """``best_metric`` / ``best_metric_key`` are what two gates check, so they must
    come from the raw ``results.json`` mapping, not the narrowed ``SeedCell`` one."""
    root, _ = grid
    state = css2.raw_training_state(root / "rv_50_ms_50_seed_42")
    assert set(state) >= {"best_metric", "best_metric_key", "final_epoch"}
    cells, _ = v2.load_grid(root, cross_check=False)
    narrowed = next(c for c in cells if c.run_name == "rv_50_ms_50_seed_42")
    assert "best_metric" not in (narrowed.training_state or {})
    assert css2.raw_training_state(root / "nope") == {}


# --------------------------------------------------------------------------- #
# 2. Winning-log resolution.
# --------------------------------------------------------------------------- #
def test_a_superseded_log_with_a_different_curve_is_recorded_not_used(grid):
    root, logs = grid
    name = "rv_75_ms_25_seed_44"
    reported = _curve_for(0.75, 44)
    # An EARLIER job id for the same cell, carrying a different eval curve.
    (logs / "paclic_v2_maingrid_500_5.out").write_text(
        _log_text(name, _interior_curve(best=3), metric="eval_rv_cer", job=500, task=5),
        encoding="utf-8",
    )
    selection = _by_run(_resolve(root, logs))[name]
    assert selection.slurm_log == "paclic_v2_maingrid_600_5.out"
    assert selection.superseded_logs == ("paclic_v2_maingrid_500_5.out",)
    assert selection.gates_ok is True
    assert selection.mixed_epoch == _mixed_epoch(reported, "eval_rv_cer")
    assert any("re-queued cell" in n for n in selection.notes)


# --------------------------------------------------------------------------- #
# 3. Checkpoint attribution.
# --------------------------------------------------------------------------- #
def test_an_orphan_checkpoint_is_never_usable_and_leaves_the_cell_on_retraining(grid):
    """The v2 finding: the surviving checkpoints belong to a superseded attempt."""
    root, logs = grid
    name = "rv_100_ms_0_seed_42"
    superseded_curve = _interior_curve(best=3)
    _write_checkpoint(root / name, 5216, superseded_curve, best_metric=0.0400)
    selection = _by_run(_resolve(root, logs))[name]
    (survival,) = selection.checkpoints
    assert survival.attribution == css2.ATTRIBUTION_ORPHAN
    assert survival.usable_for_reinference is False
    assert survival.max_abs_trajectory_delta is not None
    assert survival.max_abs_trajectory_delta > css2.TRAJECTORY_AGREEMENT_TOL
    assert any("superseded attempt" in n for n in selection.notes)
    # An orphan cannot resolve any criterion, even at the exact selected step.
    assert selection.usable_checkpoint_at(5216) is None
    for criterion in css2.SENSITIVITY_CRITERIA:
        assert css2.resolution_needed(selection, criterion) == css2.RESOLUTION_RETRAIN
    retention = css2.retention_summary(_resolve(root, logs))
    assert retention["n_surviving_checkpoints"] == 1
    assert (
        retention["n_surviving_checkpoints_by_attribution"][css2.ATTRIBUTION_ORPHAN]
        == 1
    )
    assert retention["n_reinference_reachable_rows"] == 0


def test_a_reported_run_checkpoint_at_the_uniform_step_is_reachable(grid):
    root, logs = grid
    name = "rv_100_ms_0_seed_42"
    reported = _curve_for(1.0, 42)
    selections = _resolve(root, logs)
    step = _by_run(selections)[name].decisions["mean_val_cer"].selected_step
    # A checkpoint written by the reported run itself, at that very step.
    _write_checkpoint(
        root / name, step, [c for c in reported if c.step <= step], best_metric=0.04
    )
    selection = _by_run(_resolve(root, logs))[name]
    (survival,) = selection.checkpoints
    assert survival.attribution == css2.ATTRIBUTION_REPORTED
    assert survival.max_abs_trajectory_delta == 0.0
    assert survival.usable_for_reinference is True
    assert (
        css2.resolution_needed(selection, "mean_val_cer") == css2.RESOLUTION_REINFERENCE
    )
    retention = css2.retention_summary(_resolve(root, logs))
    assert retention["n_reinference_reachable_rows"] >= 1
    assert retention["reinference_reachable_rows"][0]["run_name"] == name
    # Reachable is NOT scorable: no re-inference record exists, so the cell is
    # still excluded from the criterion's aggregation.
    assert css2.is_scorable(selection, "mean_val_cer") is False


def test_final_model_presence_is_measured_not_asserted(grid):
    """§2 of the results page states it, so it must come from disk."""
    root, logs = grid
    assert all(not s.has_final_model for s in _resolve(root, logs))
    assert css2.retention_summary(_resolve(root, logs))["n_cells_with_final_model"] == 0
    for rv_ratio in v2.RV_RATIOS:
        for seed in v2.SEEDS:
            (root / v2.run_name(rv_ratio, seed) / "final_model").mkdir()
    selections = _resolve(root, logs)
    assert all(s.has_final_model for s in selections)
    assert css2.retention_summary(selections)["n_cells_with_final_model"] == 15


def test_an_unreadable_checkpoint_is_unattributable_not_assumed_good(grid):
    root, logs = grid
    path = root / "rv_50_ms_50_seed_42" / "checkpoints" / "checkpoint-4000"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text("{not json", encoding="utf-8")
    selection = _by_run(_resolve(root, logs))["rv_50_ms_50_seed_42"]
    (survival,) = selection.checkpoints
    assert survival.attribution == css2.ATTRIBUTION_UNKNOWN
    assert survival.usable_for_reinference is False
    assert survival.epoch is None  # 4000 is not a multiple of 326


# --------------------------------------------------------------------------- #
# 4. Scorability + verdict.
# --------------------------------------------------------------------------- #
def test_divergent_cells_are_excluded_and_per_ratio_n_is_reported(grid):
    selections = _resolve(*grid)
    per_ratio = css2.aggregate_criterion(selections, "mean_val_cer")
    # The endpoint curves diverge (mixed picks the last epoch, mean an early one).
    assert per_ratio[1.0]["n"] == 0
    assert per_ratio[0.0]["n"] == 0
    # The interior curves agree, so all three seeds score from saved predictions.
    for rv_ratio in css2.INTERIOR_RATIOS:
        assert per_ratio[rv_ratio]["n"] == 3
        assert per_ratio[rv_ratio]["seeds"] == list(agg.SEEDS)
    # The mixed column always reproduces the full grid.
    current = css2.aggregate_criterion(selections, "current")
    assert [current[rv]["n"] for rv in v2.RV_RATIOS] == [3, 3, 3, 3, 3]


def test_an_excluded_cell_contributes_no_value_at_all(grid):
    """A divergent cell must not leak its committed (mixed-rule) numbers."""
    selections = _resolve(*grid)
    endpoint = css2.aggregate_criterion(selections, "mean_val_cer")[1.0]
    for metric in css2.REPORT_METRICS:
        stat = endpoint["stats"][metric]
        assert stat.n == 0
        assert stat.mean is None
        assert stat.values == ()


def test_verdict_is_the_bounded_negative_result_when_a_ratio_has_no_cell(grid):
    selections = _resolve(*grid)
    verdict = css2.chosen_deliverable(selections)
    assert verdict["chosen"] == css2.DELIVERABLE_BOUNDED_NEGATIVE
    assert verdict["criteria_supporting_a_grid_wide_result"] == []
    mean = verdict["per_criterion"]["mean_val_cer"]
    assert mean["all_ratios_full"] is False
    assert mean["interior_band_full"] is True
    assert mean["ratios_with_no_cell"] == ["1", "0"]
    assert any("no scorable cell at all" in reason for reason in verdict["why"])


def test_verdict_is_the_sensitivity_result_when_every_cell_agrees(tmp_path):
    """The verdict is decided from the data, not hardcoded: make every cell agree."""
    root, logs = tmp_path / "study", tmp_path / "slurm"
    logs.mkdir(parents=True)
    task = 0
    for rv_ratio in v2.RV_RATIOS:
        for seed in v2.SEEDS:
            # A curve whose single minimum is shared by all three criteria.
            candidates = _interior_curve(best=12)
            metric = _mixed_metric(rv_ratio)
            _write_cell(root, rv_ratio, seed, candidates, metric=metric)
            (logs / f"paclic_v2_maingrid_600_{task}.out").write_text(
                _log_text(
                    v2.run_name(rv_ratio, seed),
                    candidates,
                    metric=metric,
                    job=600,
                    task=task,
                ),
                encoding="utf-8",
            )
            task += 1
    selections = _resolve(root, logs)
    verdict = css2.chosen_deliverable(selections)
    assert verdict["chosen"] == css2.DELIVERABLE_SENSITIVITY
    assert set(verdict["criteria_supporting_a_grid_wide_result"]) == set(
        css2.SENSITIVITY_CRITERIA
    )


def test_divergence_summary_counts_are_recomputed(grid):
    selections = _resolve(*grid)
    summary = css2.divergence_summary(selections, "mean_val_cer")
    assert summary["n_decided"] == 15
    assert summary["n_match"] + summary["n_diverge"] == 15
    assert summary["n_diverge"] == 6  # the six endpoint cells
    assert summary["n_reinference_reachable"] == 0
    assert summary["n_retrain_only"] == 6
    assert sorted(summary["diverging_runs"]) == sorted(
        v2.run_name(rv, seed) for rv in (1.0, 0.0) for seed in agg.SEEDS
    )


# --------------------------------------------------------------------------- #
# 5. Exposure.
# --------------------------------------------------------------------------- #
def test_exposure_deltas_are_uniform_minus_mixed(grid):
    selections = _resolve(*grid)
    endpoint = next(
        e
        for e in css2.exposures(selections)
        if e.run_name == "rv_100_ms_0_seed_42" and e.criterion == "mean_val_cer"
    )
    assert endpoint.diverges is True
    assert endpoint.epoch_delta == endpoint.uniform_epoch - endpoint.mixed_epoch
    for quantity in css2.EXPOSURE_QUANTITIES:
        assert endpoint.deltas[quantity] == pytest.approx(
            endpoint.uniform_values[quantity] - endpoint.mixed_values[quantity]
        )
    # An RV-only cell: the uniform epoch is earlier, so RV val CER gets WORSE and
    # MS val CER gets better — the trade the mixed rule declines to make.
    assert endpoint.epoch_delta < 0
    assert endpoint.deltas["rv_val_cer"] > 0
    assert endpoint.deltas["ms_val_cer"] < 0
    # The 'own metric' delta is the mixed rule's own field.
    assert endpoint.dominant_field == "rv_val_cer"
    assert endpoint.dominant_delta == endpoint.deltas["rv_val_cer"]


def test_a_matching_cell_has_exactly_zero_exposure(grid):
    selections = _resolve(*grid)
    interior = next(
        e
        for e in css2.exposures(selections)
        if e.run_name == "rv_50_ms_50_seed_42" and e.criterion == "mean_val_cer"
    )
    assert interior.diverges is False
    assert interior.epoch_delta == 0
    assert set(interior.deltas.values()) == {0.0}
    assert interior.resolution is None


def test_a_uniform_argmin_at_the_last_epoch_of_a_truncated_run_is_censored(tmp_path):
    """Early stopping is driven by the MIXED metric, so such an argmin is a bound."""
    root, logs = tmp_path / "study", tmp_path / "slurm"
    logs.mkdir(parents=True)
    # 12 of 20 planned epochs; MS keeps improving to the end while RV bottoms out
    # at 9, so eval_rv_cer selects 9 and mean/worst select the last epoch.
    candidates = [
        _cand(epoch, 0.05 + 0.0004 * abs(epoch - 9), 0.09 - 0.003 * epoch)
        for epoch in range(1, 13)
    ]
    _write_cell(root, 0.75, 42, candidates, metric="eval_rv_cer", planned_epochs=20)
    (logs / "paclic_v2_maingrid_600_3.out").write_text(
        _log_text(
            v2.run_name(0.75, 42), candidates, metric="eval_rv_cer", job=600, task=3
        ),
        encoding="utf-8",
    )
    cells, _ = v2.load_grid(root, cross_check=False, seeds=(42,), rv_ratios=(0.75,))
    selections = css2.build_selections(
        cells,
        study_root=root,
        attestation=v2.attest_corpus_from_slurm_logs(logs),
        slurm_log_dir=logs,
        planned_epochs=css2.planned_epochs_by_run(root, cells),
    )
    selection = selections[0]
    assert selection.planned_epochs == 20
    assert selection.last_observed_epoch == 12
    assert selection.early_stopped is True
    assert selection.mixed_epoch == 9
    exposure = next(
        e for e in css2.exposures(selections) if e.criterion == "mean_val_cer"
    )
    assert exposure.uniform_epoch == 12
    assert exposure.censored is True
    stopping = css2.early_stopping_summary(selections)
    assert stopping["n_truncated"] == 1
    assert stopping["censored_runs"] == [v2.run_name(0.75, 42)]


def test_a_full_length_run_is_not_early_stopped(grid):
    selections = _resolve(*grid)
    assert css2.early_stopping_summary(selections)["n_truncated"] == 0
    assert all(s.early_stopped is False for s in selections)


def test_per_ratio_exposure_uses_the_shared_paired_contrast(grid):
    selections = _resolve(*grid)
    block = css2.per_ratio_exposure(selections, "mean_val_cer")["1"]
    assert block["n"] == 3
    assert block["seeds"] == list(agg.SEEDS)
    assert block["n_diverging"] == 3
    rv = block["quantities"]["rv_val_cer"]
    # _contrast's own shape, so this bundle cannot invent a different CI.
    assert set(rv) >= {"seeds", "a", "b", "per_seed_diff", "diff", "sign_consistent"}
    assert rv["diff"]["n"] == 3
    assert rv["sign_consistent"] is True
    assert rv["direction"] == "b>a"
    expected = agg._contrast(
        {
            s: e.mixed_values["rv_val_cer"]
            for s in agg.SEEDS
            for e in [_e(selections, s)]
        },
        {
            s: e.uniform_values["rv_val_cer"]
            for s in agg.SEEDS
            for e in [_e(selections, s)]
        },
    )
    assert rv["diff"]["mean"] == pytest.approx(expected["diff"]["mean"])


def _e(selections, seed):
    return next(
        e
        for e in css2.exposures(selections)
        if e.criterion == "mean_val_cer" and e.rv_ratio == 1.0 and e.seed == seed
    )


def test_exposure_extremes_ignore_matching_cells(grid):
    selections = _resolve(*grid)
    extremes = css2.exposure_extremes(selections)["mean_val_cer"]
    assert extremes["n_diverging_cells"] == 6
    rv = extremes["per_quantity"]["rv_val_cer"]
    assert rv["max_abs_delta"] > 0
    # The reported worst run must really be one of the divergent ones.
    assert (
        rv["run_name"]
        in css2.divergence_summary(selections, "mean_val_cer")["diverging_runs"]
    )


def test_band_summary_separates_the_interior_band_from_the_endpoints(grid):
    selections = _resolve(*grid)
    bands = css2.band_summary(selections)
    interior, endpoints = bands["interior"], bands["endpoints"]
    assert interior["n_cells"] == 9
    assert endpoints["n_cells"] == 6
    # Interior curves agree with every rule here, endpoints diverge under both.
    assert interior["n_diverging"] == 0
    assert interior["max_abs_validation_delta"] is None
    assert endpoints["n_diverging"] == 12
    assert endpoints["max_abs_epoch_delta"] > 0
    assert endpoints["n_scorable_by_criterion"] == {
        "mean_val_cer": 0,
        "worst_val_cer": 0,
    }


# --------------------------------------------------------------------------- #
# 6. Reuse of the shared contract.
# --------------------------------------------------------------------------- #
def test_criteria_report_metrics_and_ratios_are_the_shared_ones():
    assert css2.CRITERIA == v1.CRITERIA
    assert css2.SENSITIVITY_CRITERIA == v1.SENSITIVITY_CRITERIA
    assert css2.REPORT_METRICS == v1.REPORT_METRICS
    assert css2.EXPOSURE_QUANTITIES == agg.CANDIDATE_VALUE_FIELDS
    assert css2.INTERIOR_RATIOS == tuple(r for r in v2.RV_RATIOS if 0.0 < r < 1.0)


def test_statistics_come_from_the_shared_summarize(grid):
    selections = _resolve(*grid)
    block = css2.aggregate_criterion(selections, "current")[0.5]
    values = [
        v1.report_metrics(s.cell)["avg_cer"]
        for s in selections
        if s.cell.rv_ratio == 0.5
    ]
    expected = agg.summarize(values)
    assert block["stats"]["avg_cer"] == expected


# --------------------------------------------------------------------------- #
# 7. Renderers + bundle.
# --------------------------------------------------------------------------- #
def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def test_ledger_has_one_row_per_cell_and_criterion_with_scorability(grid):
    selections = _resolve(*grid)
    rows = _rows(css2.render_ledger_csv(selections))
    assert len(rows) == 15 * len(css2.CRITERIA)
    assert list(rows[0]) == list(css2.LEDGER_COLUMNS)
    endpoint = [
        r
        for r in rows
        if r["run_name"] == "rv_100_ms_0_seed_42" and r["criterion"] == "mean_val_cer"
    ]
    assert endpoint[0]["status"] == agg.STATUS_DEFERRED
    assert endpoint[0]["scorable"] == "False"
    assert endpoint[0]["resolution_needed"] == css2.RESOLUTION_RETRAIN
    assert endpoint[0]["mixed_rule_metric"] == "eval_rv_cer"
    assert endpoint[0]["mixed_rule_label"] == "rv_val_cer"


def test_candidates_csv_flags_each_rules_selected_epoch(grid):
    selections = _resolve(*grid)
    rows = [
        r
        for r in _rows(css2.render_candidates_csv(selections))
        if r["run_name"] == "rv_100_ms_0_seed_42"
    ]
    assert len(rows) == PLANNED_EPOCHS
    assert sum(1 for r in rows if r["is_mixed_selected"] == "True") == 1
    assert sum(1 for r in rows if r["is_mean_val_cer_selected"] == "True") == 1
    assert sum(1 for r in rows if r["is_worst_val_cer_selected"] == "True") == 1


def test_exposure_and_per_ratio_csvs_are_complete(grid):
    selections = _resolve(*grid)
    exposure = _rows(css2.render_exposure_csv(selections))
    assert len(exposure) == 15 * len(css2.SENSITIVITY_CRITERIA)
    assert list(exposure[0]) == list(css2.EXPOSURE_COLUMNS)
    per_ratio = _rows(css2.render_per_ratio_exposure_csv(selections))
    # (epoch + four quantities) × five ratios × two criteria.
    assert len(per_ratio) == 5 * len(v2.RV_RATIOS) * len(css2.SENSITIVITY_CRITERIA)
    assert {r["quantity"] for r in per_ratio} == {"epoch", *css2.EXPOSURE_QUANTITIES}


def test_scorable_csv_reports_n_even_for_an_empty_ratio(grid):
    selections = _resolve(*grid)
    rows = _rows(css2.render_scorable_per_ratio_csv(selections))
    empty = [
        r for r in rows if r["criterion"] == "mean_val_cer" and r["rv_ratio"] == "1.0"
    ]
    assert empty and all(r["n"] == "0" and r["mean"] == "" for r in empty)






def test_a_cell_with_no_log_is_reported_not_guessed(grid):
    root, logs = grid
    (logs / "paclic_v2_maingrid_600_0.out").unlink()
    selection = _by_run(_resolve(root, logs))["rv_100_ms_0_seed_42"]
    assert selection.slurm_log is None
    assert selection.candidates == ()
    assert selection.decisions == {}
    assert any("no winning Slurm .out" in n for n in selection.notes)
    for criterion in css2.SENSITIVITY_CRITERIA:
        assert css2.is_scorable(selection, criterion) is False
        assert css2.resolution_needed(selection, criterion) == css2.RESOLUTION_RETRAIN


def test_main_exits_nonzero_on_a_partial_grid(tmp_path, capsys):
    root, logs = tmp_path / "study", tmp_path / "slurm"
    logs.mkdir(parents=True)
    candidates = _interior_curve()
    _write_cell(root, 0.5, 42, candidates, metric="eval_avg_cer")
    (logs / "paclic_v2_maingrid_600_0.out").write_text(
        _log_text(
            v2.run_name(0.5, 42), candidates, metric="eval_avg_cer", job=600, task=0
        ),
        encoding="utf-8",
    )
    code = css2.main(
        [
            "--study-root",
            str(root),
            "--slurm-log-dir",
            str(logs),
            "--out-dir",
            str(tmp_path / "bundle"),
        ]
    )
    assert code == 1
    assert "PARTIAL" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 8. The committed bundle and the paper prose must not drift apart.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = (
    REPO_ROOT / "results"
    / "paclic-v2-checkpoint-selection-sensitivity"
)




@pytest.fixture(scope="module")
def committed() -> dict:
    """The committed v2 bundle's payload (skips if the bundle is not checked in)."""
    path = BUNDLE / "aggregate.json"
    if not path.is_file():
        pytest.skip(f"{path} is not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_committed_bundle_is_the_bounded_negative_result(committed):
    assert committed["deliverable"]["chosen"] == css2.DELIVERABLE_BOUNDED_NEGATIVE
    assert committed["deliverable"]["criteria_supporting_a_grid_wide_result"] == []
    assert committed["n_cells"] == 15
    assert committed["gates"]["all_ok"] is True
    # The finding the negative result rests on: the only surviving checkpoints are
    # orphans of a superseded attempt, so nothing is reachable by re-inference.
    retention = committed["retention"]
    assert retention["n_cells_with_any_surviving_checkpoint"] == 1
    assert retention["n_reinference_reachable_rows"] == 0
    assert (
        retention["n_surviving_checkpoints_by_attribution"][css2.ATTRIBUTION_REPORTED]
        == 0
    )




