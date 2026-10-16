"""Test paclic v2 maingrid analysis: PACLIC 2026 v2 release."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.evaluation import aggregate_accent_ratio_multiseed as agg
from scripts.evaluation import build_paclic_v2_maingrid_analysis as v2

TOTAL = 10420


# --------------------------------------------------------------------------- #
# Synthetic study root.
# --------------------------------------------------------------------------- #
def _view(cer: float, f1: float, chrf: float = 88.0) -> dict[str, float]:
    return {
        "cer": cer,
        "wer": cer * 3,
        "chrf": chrf,
        "exact_match": 0.1,
        "full_cer": cer,
        "accent_stripped_cer": cer / 2,
        "accent_precision": f1,
        "accent_recall": f1,
        "accent_f1": f1,
    }


#: Per-ratio (rv_cer, ms_cer, rv_accent_f1, ms_accent_f1) with an endpoint
#: asymmetry baked in (RV->MS easier than MS->RV) so the contrasts have signal.
_SPEC: dict[float, tuple[float, float, float, float]] = {
    1.0: (0.04, 0.10, 0.95, 0.30),
    0.75: (0.05, 0.06, 0.94, 0.90),
    0.5: (0.05, 0.05, 0.93, 0.93),
    0.25: (0.06, 0.05, 0.90, 0.94),
    0.0: (0.20, 0.04, 0.30, 0.95),
}


def key_list_sha(seed: int) -> str:
    """The subset hash a healthy cell records — **per seed**, never grid-wide.

    ``dataset.py`` sorts the shared keys then applies a *seeded* shuffle
    (``parallel_subset_seed=null`` → follows ``seed``), so at full N the selected key
    set is identical across seeds while its order, hence the hash, is not. The
    fixture used to hard-code one hash for all 15 cells, which is exactly why the
    grid-wide check looked correct: on the live grid seed 42 hashed to
    ``ca03ed92dd7b46b4`` and seed 43 to ``f58204bf5572bd12``.
    """
    return {42: "ca03ed92dd7b46b4", 43: "f58204bf5572bd12", 44: "0f1e2d3c4b5a6978"}[
        seed
    ]


def _payload(rv_ratio: float, seed: int, *, total: int = TOTAL) -> dict:
    """A v2 ``results.json`` — chrF and inline ``training_state`` for every cell."""
    rv_cer, ms_cer, rv_f1, ms_f1 = _SPEC[rv_ratio]
    # A tiny per-seed jitter so the across-seed SD/CI is non-degenerate.
    jitter = (seed - 43) * 0.001
    n_rv, n_ms = v2.expected_mix_counts(total, rv_ratio)
    return {
        "experiment_name": v2.run_name(rv_ratio, seed),
        "seed": seed,
        "model_base": "fixed-foundation",
        "freeze_mode": "none",
        "input_scheme": "iso",
        "num_train_epochs": 20,
        "training_state": {
            "best_epoch": 12.0,
            "best_step": 3912,
            "global_step": 6520,
        },
        "mix_mode": "parallel_by_text",
        "key_column": "text",
        "selection_algorithm": "sorted_common_keys+seeded_shuffle+prefix",
        "parallel_keys_total": TOTAL,
        "parallel_keys_used": total,
        "parallel_subset_seed": seed,
        "key_list_sha256": key_list_sha(seed),
        "total_train_samples_requested": total,
        "rv_train_samples": n_rv,
        "ms_train_samples": n_ms,
        "rv_test": _view(rv_cer + jitter, rv_f1),
        "ms_test": _view(ms_cer + jitter, ms_f1),
        "avg_cer": (rv_cer + ms_cer) / 2 + jitter,
        "cer_gap": abs(rv_cer - ms_cer),
        "accent_f1_gap": abs(rv_f1 - ms_f1),
        "accent_stripped_cer_gap": abs(rv_cer - ms_cer) / 2,
    }


def _write_grid(
    root: Path,
    *,
    skip: frozenset[str] = frozenset(),
    with_preds: bool = False,
    mutate: dict[str, dict] | None = None,
    preds_skip: frozenset[str] = frozenset(),
    corrupt: frozenset[str] = frozenset(),
) -> Path:
    """Write the 15 v2 run dirs; ``skip`` names cells to leave absent.

    ``preds_skip`` writes a cell's ``results.json`` but not its prediction CSVs — the
    normal mid-flight state, since the runner writes ``results.json`` first and
    exact-match is re-sourced from the CSVs. ``corrupt`` writes a truncated
    ``results.json`` (a torn write).
    """
    for rv_ratio in v2.RV_RATIOS:
        for seed in v2.SEEDS:
            name = v2.run_name(rv_ratio, seed)
            if name in skip:
                continue
            cell = root / name
            cell.mkdir(parents=True, exist_ok=True)
            if name in corrupt:
                (cell / "results.json").write_text(
                    '{"seed": 42, "rv_test": {"cer": 0.0', encoding="utf-8"
                )
                continue
            payload = _payload(rv_ratio, seed)
            if mutate and name in mutate:
                payload.update(mutate[name])
            (cell / "results.json").write_text(json.dumps(payload), encoding="utf-8")
            if with_preds and name not in preds_skip:
                for view in ("rv", "ms"):
                    (cell / f"{view}_predictions.csv").write_text(
                        "reference,prediction\na,a\nb,c\n", encoding="utf-8"
                    )
    return root


def _log_text(
    run_name: str,
    *,
    output_dir: str,
    corpus: str = "v2",
    rv: str = "data/datasets/rv_v2",
    ms: str = "data/datasets/ms_v2",
    data_source: str | None = "csv",
    job: int = 506,
    task: int = 0,
    legacy: bool = False,
) -> str:
    """One Slurm ``.out`` header as the launcher echoes it.

    ``legacy=True`` reproduces the form the CURRENTLY RUNNING grid writes: the
    hard-coded ``corpus=v2 (accent marks corrected)`` literal, no ``data_source=``
    token, and the CSV presence guard line from which the data source is recovered.
    Those logs are the only attestation the in-flight cells will ever have, so the
    parser must keep reading them.
    """
    guard = (
        f"[paclic-v2-maingrid] corpus CSVs + first images present: rv={rv} ms={ms}\n"
        if data_source == "csv"
        else ""
    )
    if legacy:
        corpus_line = (
            f"[paclic-v2-maingrid] corpus={corpus} (accent marks corrected) "
            f"rv={rv} ms={ms}"
        )
    else:
        source = "" if data_source is None else f" data_source={data_source}"
        corpus_line = f"[paclic-v2-maingrid] corpus={corpus} rv={rv} ms={ms}{source}"
    return (
        guard
        + f"[paclic-v2-maingrid] task={task}/14 run_name={run_name}  seed=42 smoke=0\n"
        f"[paclic-v2-maingrid] node=gpu-worker  job={job} array={job}_{task}\n"
        + corpus_line
        + "\n"
        f"[paclic-v2-maingrid] output_dir={output_dir} tags=[paclic,corpus-v2]\n"
        "Starting training\n"
    )


def _write_logs(
    log_dir: Path,
    *,
    output_dir: str = "./outputs/experiments_paclic_v2_maingrid",
    job: int = 506,
    **kwargs,
) -> Path:
    """A healthy log per target cell, so the figure gate's corpus check can pass."""
    log_dir.mkdir(parents=True, exist_ok=True)
    task = 0
    for rv_ratio in v2.RV_RATIOS:
        for seed in v2.SEEDS:
            (log_dir / f"paclic_v2_maingrid_{job}_{task}.out").write_text(
                _log_text(
                    v2.run_name(rv_ratio, seed),
                    output_dir=output_dir,
                    job=job,
                    task=task,
                    **kwargs,
                ),
                encoding="utf-8",
            )
            task += 1
    return log_dir


def _healthy_attestation(cells, tmp_path: Path, root: Path, **kwargs) -> dict:
    """Attestation summary over a full set of healthy logs for ``cells``."""
    logs = _write_logs(
        tmp_path / "slurm", output_dir=f"./outputs/{root.name}", **kwargs
    )
    return v2.corpus_attestation_summary(
        v2.attest_corpus_from_slurm_logs(logs), cells, study_root=root
    )


def _fake_metrics_fn(csv_path: Path) -> dict:
    """Deterministic stand-in for the study evaluator (no jiwer/torch import)."""
    return {
        "cer": 0.041,
        "wer": 0.205,
        "chrf": 88.0,
        "accent_f1": 0.952,
        "exact_match": 0.5,
        "exact_match_count": 1,
    }


# --------------------------------------------------------------------------- #
# 1. Naming: why the v1 aggregator cannot be pointed at the v2 root.
# --------------------------------------------------------------------------- #
def test_run_name_carries_the_seed_suffix_for_every_seed():
    assert v2.run_name(1.0, 42) == "rv_100_ms_0_seed_42"
    assert v2.run_name(0.75, 43) == "rv_75_ms_25_seed_43"
    assert v2.run_name(0.0, 44) == "rv_0_ms_100_seed_44"


def test_run_name_matches_the_runner_name_builder():
    """The launcher passes run_suffix=seed_<n>; the loader must agree exactly."""
    from scripts.training.trocr.experiment_accent_ratio import build_experiment_name

    for rv_ratio in v2.RV_RATIOS:
        for seed in v2.SEEDS:
            assert v2.run_name(rv_ratio, seed) == build_experiment_name(
                rv_ratio, round(1.0 - rv_ratio, 2), f"seed_{seed}"
            )


def test_v2_loader_finds_all_fifteen_cells(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    assert len(cells) == 15
    assert all(c.present for c in cells)
    assert {c.source for c in cells} == {v2.CELL_SOURCE}
    # Ratio-major order == the launcher's array decomposition.
    assert [(c.rv_ratio, c.seed) for c in cells[:4]] == [
        (1.0, 42),
        (1.0, 43),
        (1.0, 44),
        (0.75, 42),
    ]


def test_v2_cells_read_training_state_inline_no_recovery(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    assert {c.training_state_recovery_method for c in cells} == {"results_json"}


# --------------------------------------------------------------------------- #
# 2. Coverage / partial honesty.
# --------------------------------------------------------------------------- #
def _partial_cells(tmp_path):
    """Grid with the 0:100 ratio absent and one 75:25 seed absent (mid-flight)."""
    skip = frozenset([v2.run_name(0.0, s) for s in v2.SEEDS] + [v2.run_name(0.75, 44)])
    root = _write_grid(tmp_path / "v2", skip=skip)
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    return root, cells, checks


def test_coverage_reports_n_per_ratio_and_per_seed(tmp_path):
    _root, cells, _ = _partial_cells(tmp_path)
    cov = v2.coverage_summary(cells)
    assert cov["n_cells"] == 15
    assert cov["n_present"] == 11
    assert cov["complete"] is False
    assert cov["per_ratio"]["1"]["n_present"] == 3
    assert cov["per_ratio"]["0.75"]["n_present"] == 2
    assert cov["per_ratio"]["0.75"]["seeds_missing"] == [44]
    assert cov["per_ratio"]["0"]["n_present"] == 0
    assert cov["per_ratio"]["0"]["complete"] is False
    assert cov["per_seed"]["42"]["n_present"] == 4
    assert cov["per_seed"]["44"]["n_present"] == 3
    assert len(cov["missing_cells"]) == 4
    assert v2.status_label(cov) == "PARTIAL"


def test_complete_grid_is_labelled_complete(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    cov = v2.coverage_summary(cells)
    assert cov["complete"] is True
    assert v2.status_label(cov) == "COMPLETE"
    assert v2.partial_banner(cov) is None
    assert cov["seeds_present"] == [42, 43, 44]


def test_partial_banner_says_not_a_complete_result_and_lists_gaps(tmp_path):
    _root, cells, _ = _partial_cells(tmp_path)
    banner = v2.partial_banner(v2.coverage_summary(cells))
    assert banner is not None
    assert banner.startswith("PARTIAL")
    assert "NOT A COMPLETE RESULT" in banner
    assert "11/15" in banner
    assert "n=2/3" in banner  # the per-ratio n, not one grid-wide n
    assert v2.run_name(0.0, 42) in banner


def test_grid_seeds_never_overstate_n(tmp_path):
    """``grid.seeds`` feeds the figure caption, so it holds seeds actually present."""
    skip = frozenset(v2.run_name(r, 44) for r in v2.RV_RATIOS)
    root = _write_grid(tmp_path / "v2", skip=skip)
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    aggregate = v2.build_aggregate(cells, checks, v2.coverage_summary(cells))
    assert aggregate["grid"]["seeds"] == [42, 43]
    assert aggregate["grid"]["target_seeds"] == [42, 43, 44]
    assert aggregate["status"] == "PARTIAL"


# --------------------------------------------------------------------------- #
# 3. Statistics are reused, not reimplemented.
# --------------------------------------------------------------------------- #
def test_per_ratio_stats_equal_the_v1_summarize_contract(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    aggregate = v2.build_aggregate(cells, checks, v2.coverage_summary(cells))
    block = aggregate["per_ratio"]["1"]["rv_cer"]
    expected = agg.summarize(
        [_payload(1.0, s)["rv_test"]["cer"] for s in v2.SEEDS]
    ).as_dict()
    assert block["n"] == 3
    assert block["mean"] == pytest.approx(expected["mean"])
    assert block["ci95_halfwidth"] == pytest.approx(expected["ci95_halfwidth"])
    # Student-t (df=2), not z.
    assert block["ci95_halfwidth"] == pytest.approx(4.303 * expected["sem"])


def test_partial_ratio_reports_its_own_smaller_n(tmp_path):
    _root, cells, checks = _partial_cells(tmp_path)
    aggregate = v2.build_aggregate(cells, checks, v2.coverage_summary(cells))
    assert aggregate["per_ratio"]["0.75"]["rv_cer"]["n"] == 2
    assert aggregate["per_ratio"]["0"]["rv_cer"]["n"] == 0
    assert aggregate["per_ratio"]["0"]["rv_cer"]["mean"] is None
    # The zero-shot contrast pairs the two endpoints; one endpoint is absent.
    assert aggregate["asymmetry"]["zero_shot"]["cer"]["diff"]["n"] == 0


def test_exact_match_is_resourced_from_committed_predictions(tmp_path):
    root = _write_grid(tmp_path / "v2", with_preds=True)
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=True)
    assert {c.rv_exact_match_source for c in cells} == {"recomputed-predictions"}
    assert cells[0].rv["exact_match"] == pytest.approx(0.5)
    assert cells[0].rv["exact_match_count"] == 1
    # chrF is present in every v2 results.json, so it is never a recompute-fill.
    assert {c.rv_chrf_source for c in cells} == {"results.json"}
    assert len(checks) == 30  # 15 cells x 2 views


# --------------------------------------------------------------------------- #
# 4. Rendered results page.
# --------------------------------------------------------------------------- #
def _render(cells, checks, **kwargs) -> str:
    cov = v2.coverage_summary(cells)
    return v2.render_results_md(
        cells,
        agg.aggregate_by_ratio(cells),
        agg.asymmetry(cells),
        agg.retention_adaptation(cells),
        checks,
        cov,
        **kwargs,
    )


def test_results_md_leads_with_partial_and_shows_per_ratio_n(tmp_path):
    _root, cells, checks = _partial_cells(tmp_path)
    md = _render(cells, checks)
    assert "**Status:** PARTIAL" in md
    assert "NOT A COMPLETE RESULT" in md
    assert "11/15" in md
    assert "## 0. Coverage" in md
    assert "| 0:100 | 0/3 |" in md
    assert "COMPLETE" not in md.replace("NOT A COMPLETE RESULT", "")


def test_results_md_says_complete_only_when_complete(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    md = _render(cells, checks)
    assert "**Status:** COMPLETE" in md
    assert "PARTIAL" not in md
    assert "never pooled" in md  # v1/v2 must not be pooled


def test_results_md_marks_unattested_corpus(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    attestation = v2.corpus_attestation_summary({}, cells)
    md = _render(cells, checks, attestation=attestation)
    assert "unattested" in md


def test_results_md_surfaces_integrity_warnings_in_the_page(tmp_path):
    """A reader of the committed page must see the warnings, not only stderr."""
    bad = v2.run_name(0.5, 44)
    root = _write_grid(tmp_path / "v2", mutate={bad: {"rv_train_samples": 1}})
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    subset = v2.subset_consistency(v2.subset_provenance(root, cells), cells)
    md = _render(cells, checks, subset=subset)
    assert "Integrity warnings" in md
    assert "do not match their mix ratio" in md


def test_results_md_labels_a_log_only_cell_as_such(tmp_path):
    """An attested log with no results.json must not read as a finished cell."""
    root = _write_grid(tmp_path / "v2", skip=frozenset({v2.run_name(1.0, 42)}))
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    attestation = v2.corpus_attestation_summary(
        {"rv_100_ms_0_seed_42": {"corpus": "v2", "log": "x.out"}}, cells
    )
    md = _render(cells, checks, attestation=attestation)
    assert "| 100:0 | 42 | NO |" in md
    assert "v2 (log only)" in md


def test_results_md_attaches_n_to_every_asymmetry_arm(tmp_path):
    """An arm's value never appears without its own n (the paired n differs)."""
    skip = frozenset(v2.run_name(0.0, s) for s in v2.SEEDS)
    root = _write_grid(tmp_path / "v2", skip=skip)
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    md = _render(cells, checks)
    assert "n paired" in md
    assert "(n=3)" in md  # the surviving RV->MS arm
    assert v2._pm_n(agg.summarize([])) == "—"


# --------------------------------------------------------------------------- #
# 5. Recipe / subset consistency.
# --------------------------------------------------------------------------- #
def test_subset_consistency_passes_on_a_clean_grid(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    subset = v2.subset_consistency(v2.subset_provenance(root, cells), cells)
    assert subset["n_cells_checked"] == 15
    assert subset["consistent"] is True
    assert subset["mix_mismatches"] == []
    assert "CANNOT attest" in subset["key_list_sha256_note"]


def test_a_healthy_grid_with_per_seed_subset_hashes_is_not_flagged(tmp_path):
    """The false alarm: three per-seed hashes are the HEALTHY state, not a split study.

    ``key_list_sha256`` used to sit in the grid-wide ``fixed_fields``, so the second
    seed landing would publish "Held-fixed fields … INCONSISTENT" plus "this is not a
    single study" for a grid that is one — and because ``main`` hard-fails on that
    string, the publication gate could never turn green again.
    """
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    subset = v2.subset_consistency(v2.subset_provenance(root, cells), cells)

    # Three distinct hashes grid-wide — one per seed — and that is CONSISTENT.
    assert len(subset["key_list_sha256_distinct_grid_wide"]) == len(v2.SEEDS)
    assert "key_list_sha256" not in subset["held_fixed"]
    assert "key_list_sha256" not in subset["inconsistent_fields"]
    assert subset["key_list_sha256_split_seeds"] == []
    assert subset["key_list_sha256_consistent"] is True
    assert subset["grid_wide_consistent"] is True
    assert subset["consistent"] is True
    # Each seed contributes exactly one hash over its five ratios.
    for seed in v2.SEEDS:
        blk = subset["key_list_sha256_per_seed"][str(seed)]
        assert blk["key_list_sha256"] == [key_list_sha(seed)]
        assert len(blk["runs"]) == len(v2.RV_RATIOS)

    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), subset=subset)
    assert not [w for w in warnings if "not a single study" in w]
    assert v2.hard_warnings(warnings) == []


def test_results_md_says_what_scope_each_recipe_check_used(tmp_path):
    """§5 must not describe a per-seed check as a grid-wide held-fixed verdict."""
    root = _write_grid(tmp_path / "v2")
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    subset = v2.subset_consistency(v2.subset_provenance(root, cells), cells)
    md = _render(cells, checks, subset=subset)
    assert "INCONSISTENT" not in md
    assert "WITHIN each seed across the five ratios, NOT grid-wide" in md
    assert "3 distinct value(s) grid-wide, 3 expected (one per seed)" in md


def test_key_list_sha256_split_WITHIN_a_seed_is_still_caught(tmp_path):
    """The real defect the hash can detect: one seed's ratios on different subsets."""
    bad = v2.run_name(0.5, 44)
    root = _write_grid(tmp_path / "v2", mutate={bad: {"key_list_sha256": "other"}})
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    subset = v2.subset_consistency(v2.subset_provenance(root, cells), cells)
    assert subset["key_list_sha256_split_seeds"] == [44]
    assert subset["key_list_sha256_consistent"] is False
    assert subset["consistent"] is False
    # ...and it stays grid-wide-consistent: the split is within seed 44 only.
    assert subset["grid_wide_consistent"] is True
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), subset=subset)
    assert any("not constant within seed(s) [44]" in w for w in warnings)
    assert v2.hard_warnings(warnings)  # publication-blocking


def test_a_genuine_cross_seed_recipe_split_is_still_reported(tmp_path):
    """Dropping the hash must not weaken the fields that DO pin the recipe.

    A different foundation under one seed is the failure the grid-wide check exists
    for, and it must survive the Blocker-1 fix.
    """
    root = _write_grid(
        tmp_path / "v2",
        mutate={
            v2.run_name(r, 44): {"model_base": "some-other-foundation"}
            for r in v2.RV_RATIOS
        },
    )
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    subset = v2.subset_consistency(v2.subset_provenance(root, cells), cells)
    assert subset["inconsistent_fields"] == ["model_base"]
    assert subset["grid_wide_consistent"] is False
    assert subset["consistent"] is False
    # The per-seed hash check is untouched by this — the two are independent.
    assert subset["key_list_sha256_split_seeds"] == []
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), subset=subset)
    assert any("not a single study" in w for w in warnings)
    assert v2.hard_warnings(warnings)
    # agg's own foundation check fires too, on the same grid.
    assert any("foundation not held fixed" in w for w in warnings)


def test_total_train_samples_split_across_seeds_is_still_reported(tmp_path):
    """The other pin the brief named explicitly: N must stay grid-wide."""
    root = _write_grid(
        tmp_path / "v2",
        mutate={
            v2.run_name(r, 43): {"total_train_samples_requested": 5210}
            for r in v2.RV_RATIOS
        },
    )
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    subset = v2.subset_consistency(v2.subset_provenance(root, cells), cells)
    assert "total_train_samples_requested" in subset["inconsistent_fields"]
    assert v2.hard_warnings(
        v2.audit_warnings(cells, v2.coverage_summary(cells), subset=subset)
    )


def test_subset_consistency_flags_a_mislabelled_mix(tmp_path):
    bad = v2.run_name(0.5, 44)
    root = _write_grid(tmp_path / "v2", mutate={bad: {"rv_train_samples": 1}})
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    subset = v2.subset_consistency(v2.subset_provenance(root, cells), cells)
    assert subset["consistent"] is True  # the recipe pins are all fine
    assert subset["mix_mismatches"] == [bad]
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), subset=subset)
    assert any("do not match their mix ratio" in w for w in warnings)


def test_expected_mix_counts_follow_the_floor_rule():
    assert v2.expected_mix_counts(10420, 1.0) == (10420, 0)
    assert v2.expected_mix_counts(10420, 0.75) == (7815, 2605)
    assert v2.expected_mix_counts(10420, 0.5) == (5210, 5210)
    assert v2.expected_mix_counts(10420, 0.25) == (2605, 7815)
    assert v2.expected_mix_counts(10420, 0.0) == (0, 10420)


# --------------------------------------------------------------------------- #
# 6. Corpus attestation from the Slurm header.
# --------------------------------------------------------------------------- #
_LOG = """\
[paclic-v2-maingrid] corpus CSVs + first images present: rv=data/datasets/rv_v2 ms=data/datasets/ms_v2
[paclic-v2-maingrid] task=0/14 run_name=rv_100_ms_0_seed_42  rv_ratio=1.0 ms_ratio=0.0 seed=42 smoke=0
[paclic-v2-maingrid] node=gpu-worker  job=506 array=506_0
[paclic-v2-maingrid] corpus=v2 (accent marks corrected) rv=data/datasets/rv_v2 ms=data/datasets/ms_v2
[paclic-v2-maingrid] output_dir=./outputs/experiments_paclic_v2_maingrid tags=[paclic,corpus-v2]
Starting training
"""


def test_parse_corpus_attestation_reads_the_header_echo():
    rec = v2.parse_corpus_attestation(_LOG)
    assert rec is not None
    assert rec["run_name"] == "rv_100_ms_0_seed_42"
    assert rec["corpus"] == "v2"
    assert rec["rv_data_dir"] == "data/datasets/rv_v2"
    assert rec["ms_data_dir"] == "data/datasets/ms_v2"
    assert rec["node"] == "gpu-worker"
    assert rec["output_dir"] == "./outputs/experiments_paclic_v2_maingrid"


def test_parse_corpus_attestation_returns_none_without_a_run_name():
    assert v2.parse_corpus_attestation("nothing useful here\n") is None


def test_the_running_grids_legacy_log_form_still_parses_and_validates():
    """The in-flight cells' only attestation is the OLD `corpus=v2 …` literal form.

    No ``data_source=`` token there, so the data source is recovered from the CSV
    presence guard the launcher prints only on the csv path. Validating the echoed
    dirs is what makes those logs trustworthy, so this must keep working.
    """
    legacy = _log_text(
        "rv_100_ms_0_seed_42",
        output_dir="./outputs/experiments_paclic_v2_maingrid",
        legacy=True,
    )
    assert "corpus=v2 (accent marks corrected)" in legacy
    assert "data_source=" not in legacy
    rec = v2.parse_corpus_attestation(legacy)
    assert rec is not None
    assert rec["corpus"] == "v2"
    assert rec["rv_data_dir"] == "data/datasets/rv_v2"
    assert rec["ms_data_dir"] == "data/datasets/ms_v2"
    assert rec["data_source"] == "csv"
    assert "CSV presence guard" in rec["data_source_evidence"]
    assert v2.attestation_defects(rec) == []


def test_a_uniformly_v1_grid_is_rejected_even_though_single_corpus_holds(tmp_path):
    """The Blocker-2 module half: ``single_corpus`` sees mixtures, not a wrong corpus.

    Every cell agrees, so ``single_corpus`` is True and there are no mixture
    warnings — yet every cell trained on v1 pixels. Before the fix this produced
    ``corpus: v2``, ``single_corpus: True``, zero warnings, exit 0, figures written.
    """
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    summary = _healthy_attestation(
        cells,
        tmp_path,
        root,
        corpus="v1",
        rv="data/datasets/rv",
        ms="data/datasets/ms",
    )
    assert summary["n_unattested"] == 0
    assert summary["single_corpus"] is True  # ...and that is not enough
    assert summary["healthy"] is False
    assert len(summary["corpus_mismatches"]) == 15
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), attestation=summary)
    assert any("does not establish the expected v2 corpus" in w for w in warnings)
    assert v2.hard_warnings(warnings)  # publication-blocking, not a note


def test_an_hf_grid_echoing_the_unused_v2_dirs_is_rejected(tmp_path):
    """``data.source=hf`` has no v2 artifact but still echoes the v2 CSV dirs."""
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    summary = _healthy_attestation(cells, tmp_path, root, data_source="hf")
    assert summary["single_corpus"] is True
    assert summary["healthy"] is False
    assert all(
        any("data_source=hf is not csv" in d for d in m["defects"])
        for m in summary["corpus_mismatches"]
    )
    assert v2.hard_warnings(
        v2.audit_warnings(cells, v2.coverage_summary(cells), attestation=summary)
    )


def test_an_unrecoverable_data_source_is_not_assumed_to_be_csv(tmp_path):
    """No ``data_source=`` and no CSV guard → unverified, not "probably csv"."""
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    summary = _healthy_attestation(cells, tmp_path, root, data_source=None)
    assert summary["healthy"] is False
    assert all(
        any("data_source is not recoverable" in d for d in m["defects"])
        for m in summary["corpus_mismatches"]
    )


def test_a_healthy_v2_attestation_passes_cleanly(tmp_path):
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    summary = _healthy_attestation(cells, tmp_path, root)
    assert summary["healthy"] is True
    assert summary["corpus_mismatches"] == []
    assert summary["n_unattested"] == 0
    assert summary["expected_corpus"]["rv_data_dir"] == "data/datasets/rv_v2"
    assert (
        v2.hard_warnings(
            v2.audit_warnings(cells, v2.coverage_summary(cells), attestation=summary)
        )
        == []
    )


def test_a_healthy_legacy_form_attestation_also_passes(tmp_path):
    """The currently running grid, once complete, must be able to go green."""
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    summary = _healthy_attestation(cells, tmp_path, root, legacy=True)
    assert summary["healthy"] is True
    assert summary["corpus_mismatches"] == []


def test_logs_are_ordered_numerically_not_lexicographically(tmp_path):
    """Job 100 supersedes job 99; basename sort puts ``…_99_0`` last and wins wrongly."""
    logs = tmp_path / "slurm"
    logs.mkdir()
    (logs / "paclic_v2_maingrid_99_0.out").write_text(
        _log_text("rv_100_ms_0_seed_42", output_dir="./outputs/old", job=99),
        encoding="utf-8",
    )
    (logs / "paclic_v2_maingrid_100_0.out").write_text(
        _log_text("rv_100_ms_0_seed_42", output_dir="./outputs/new", job=100),
        encoding="utf-8",
    )
    rec = v2.attest_corpus_from_slurm_logs(logs)["rv_100_ms_0_seed_42"]
    assert rec["log"] == "paclic_v2_maingrid_100_0.out"
    assert rec["superseded_logs"] == ["paclic_v2_maingrid_99_0.out"]
    assert rec["output_dir"] == "./outputs/new"


def test_superseded_logs_are_warned_about_but_do_not_block_publication(tmp_path):
    """Live case: ``rv_100_ms_0_seed_42`` matches both the 505 smoke and 506 real log.

    Ambiguous, so it must not be silent; but the winner is the newest by (jobid, task)
    and is validated like any other log, so it must not make the gate un-greenable.
    """
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    logs = _write_logs(tmp_path / "slurm", output_dir=f"./outputs/{root.name}", job=506)
    (logs / "paclic_v2_maingrid_505_0.out").write_text(
        _log_text(
            "rv_100_ms_0_seed_42",
            output_dir="./outputs/experiments_paclic_v2_maingrid_smoke",
            job=505,
        ),
        encoding="utf-8",
    )
    summary = v2.corpus_attestation_summary(
        v2.attest_corpus_from_slurm_logs(logs), cells, study_root=root
    )
    assert [s["run_name"] for s in summary["superseded"]] == ["rv_100_ms_0_seed_42"]
    assert summary["superseded"][0]["superseded_logs"] == [
        "paclic_v2_maingrid_505_0.out"
    ]
    assert summary["healthy"] is True  # the winning log is this study's own
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), attestation=summary)
    assert any("superseded_logs" in w for w in warnings)
    assert v2.hard_warnings(warnings) == []  # ambiguous, not blocking


def test_attest_corpus_prefers_the_later_log_and_records_the_superseded_one(tmp_path):
    logs = tmp_path / "slurm"
    logs.mkdir()
    (logs / "paclic_v2_maingrid_506_0.out").write_text(_LOG, encoding="utf-8")
    (logs / "paclic_v2_maingrid_507_0.out").write_text(
        _LOG.replace("job=506", "job=507"), encoding="utf-8"
    )
    attest = v2.attest_corpus_from_slurm_logs(logs)
    rec = attest["rv_100_ms_0_seed_42"]
    assert rec["log"] == "paclic_v2_maingrid_507_0.out"
    assert rec["superseded_logs"] == ["paclic_v2_maingrid_506_0.out"]


def test_attest_corpus_missing_log_dir_is_empty_not_an_error(tmp_path):
    assert v2.attest_corpus_from_slurm_logs(tmp_path / "nope") == {}
    assert v2.attest_corpus_from_slurm_logs(None) == {}


def test_attestation_summary_reports_unattested_cells(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    logs = tmp_path / "slurm"
    logs.mkdir()
    (logs / "paclic_v2_maingrid_506_0.out").write_text(_LOG, encoding="utf-8")
    summary = v2.corpus_attestation_summary(
        v2.attest_corpus_from_slurm_logs(logs), cells
    )
    assert summary["n_attested"] == 1
    assert summary["n_unattested"] == 14
    assert summary["corpora"] == ["v2"]
    assert summary["single_corpus"] is True
    assert "not v1" in summary["note"]
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), attestation=summary)
    assert any("unattested" in w for w in warnings)


def test_attestation_summary_flags_a_log_from_a_different_study_root(tmp_path):
    """The smoke root reuses the same run names, so a name match is not enough.

    Reproduces a real hazard: aggregating ``…_maingrid_smoke`` matched the *real*
    grid's ``paclic_v2_maingrid_506_0.out`` purely by run name.
    """
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid_smoke")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    logs = tmp_path / "slurm"
    logs.mkdir()
    (logs / "paclic_v2_maingrid_506_0.out").write_text(_LOG, encoding="utf-8")
    summary = v2.corpus_attestation_summary(
        v2.attest_corpus_from_slurm_logs(logs), cells, study_root=root
    )
    assert summary["expected_study_root"] == "experiments_paclic_v2_maingrid_smoke"
    assert [m["run_name"] for m in summary["study_root_mismatches"]] == [
        "rv_100_ms_0_seed_42"
    ]
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), attestation=summary)
    assert any("different study root" in w for w in warnings)


def test_attestation_summary_accepts_a_log_from_the_matching_study_root(tmp_path):
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    logs = tmp_path / "slurm"
    logs.mkdir()
    (logs / "paclic_v2_maingrid_506_0.out").write_text(_LOG, encoding="utf-8")
    summary = v2.corpus_attestation_summary(
        v2.attest_corpus_from_slurm_logs(logs), cells, study_root=root
    )
    assert summary["study_root_mismatches"] == []


def test_attestation_summary_flags_two_different_corpora(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    logs = tmp_path / "slurm"
    logs.mkdir()
    (logs / "paclic_v2_maingrid_506_0.out").write_text(_LOG, encoding="utf-8")
    (logs / "paclic_v2_maingrid_506_1.out").write_text(
        _LOG.replace("rv_100_ms_0_seed_42", "rv_100_ms_0_seed_43")
        .replace("corpus=v2", "corpus=v1")
        .replace("rv_v2", "rv")
        .replace("ms_v2", "ms"),
        encoding="utf-8",
    )
    summary = v2.corpus_attestation_summary(
        v2.attest_corpus_from_slurm_logs(logs), cells
    )
    assert summary["corpora"] == ["v1", "v2"]
    assert summary["single_corpus"] is False
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), attestation=summary)
    assert any("more than one corpus" in w for w in warnings)


# --------------------------------------------------------------------------- #
# 7. Manifest provenance — no v1 leakage.
# --------------------------------------------------------------------------- #
def _manifest(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    return v2.build_manifest(
        cells,
        checks,
        v2.coverage_summary(cells),
        study_root=root,
        shared_study_root=v2.DEFAULT_SHARED_STUDY_ROOT,
        subset=v2.subset_consistency(v2.subset_provenance(root, cells), cells),
        attestation=v2.corpus_attestation_summary({}, cells),
    )


def test_manifest_records_status_coverage_and_every_cell(tmp_path):
    manifest = _manifest(tmp_path)
    assert manifest["study"] == v2.STUDY
    assert manifest["status"] == "COMPLETE"
    assert manifest["complete"] is True
    assert manifest["partial_banner"] is None
    assert len(manifest["cells"]) == 15
    assert {c["source"] for c in manifest["cells"]} == {v2.CELL_SOURCE}
    assert manifest["grid"]["single_root"] is True


def test_manifest_carries_no_v1_provenance(tmp_path):
    """A v2 bundle that names a v1 root or reuses a v1 note is a lie.

    ``why_not_the_v1_aggregator`` is excluded: it *documents* the v1 contract on
    purpose (it is the record of the decision), rather than claiming v1 provenance.
    """
    manifest = _manifest(tmp_path)
    manifest.pop("why_not_the_v1_aggregator")
    blob = json.dumps(manifest)
    for forbidden in (
        "factorial",
        "experiments_paclic_accent_ratio_multiseed",
        "paclic_accent_ratio_multiseed.manifest.yaml",
        "chrf_recompute_note",
        "seed42_training_state_note",
    ):
        assert forbidden not in blob, forbidden


def test_manifest_explains_why_the_v1_aggregator_is_not_used(tmp_path):
    why = _manifest(tmp_path)["why_not_the_v1_aggregator"]
    assert "expected_run_name" in why
    assert "REUSED_SEED" in why
    assert "ABSENT" in why


def test_manifest_states_the_corpus_limitations(tmp_path):
    limits = " ".join(_manifest(tmp_path)["limitations"])
    assert "different image corpora" in limits
    assert "key_list_sha256" in limits
    assert "complete==true" in limits


# --------------------------------------------------------------------------- #
# 8. Figures — gated on a complete grid, and re-attributed to the v2 bundle.
# --------------------------------------------------------------------------- #
def _aggregate(tmp_path, *, skip: frozenset[str] = frozenset(), attested: bool = True):
    """A v2 aggregate. ``attested=True`` supplies the healthy corpus attestation the
    figure gate now requires, so the figure tests exercise a publishable bundle."""
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid", skip=skip)
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    attestation = _healthy_attestation(cells, tmp_path, root) if attested else None
    return v2.build_aggregate(
        cells,
        checks,
        v2.coverage_summary(cells),
        attestation=attestation,
        integrity=v2.grid_integrity(cells, study_root=root),
    )


def test_figures_refuse_a_partial_grid(tmp_path):
    aggregate = _aggregate(tmp_path, skip=frozenset({v2.run_name(0.5, 44)}))
    with pytest.raises(v2.IncompleteGridError) as excinfo:
        v2.build_figures(aggregate, tmp_path / "figs")
    assert "14/15" in str(excinfo.value)
    assert not (tmp_path / "figs").exists()


def test_figures_refuse_an_unattested_grid(tmp_path):
    """Coverage alone was the only gate, so a complete-but-unverified grid shipped."""
    aggregate = _aggregate(tmp_path, attested=False)
    assert aggregate["complete"] is True  # coverage is fine...
    with pytest.raises(v2.UnverifiedCorpusError) as excinfo:
        v2.build_figures(aggregate, tmp_path / "figs")
    assert "without a corpus attestation" in str(excinfo.value)
    assert not (tmp_path / "figs").exists()


def test_figures_refuse_a_grid_attested_to_the_wrong_corpus(tmp_path):
    """A complete grid trained on v1 pixels must render no v2-labelled figure."""
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    aggregate = v2.build_aggregate(
        cells,
        checks,
        v2.coverage_summary(cells),
        attestation=_healthy_attestation(
            cells,
            tmp_path,
            root,
            corpus="v1",
            rv="data/datasets/rv",
            ms="data/datasets/ms",
        ),
    )
    assert aggregate["complete"] is True
    assert aggregate["publishable"] is False
    with pytest.raises(v2.FigureGateError):
        v2.build_figures(aggregate, tmp_path / "figs")
    assert not (tmp_path / "figs").exists()


def test_figures_refuse_a_grid_with_a_blocking_integrity_warning(tmp_path):
    """Attestation healthy, coverage complete — but a cell's mix is mislabelled."""
    root = _write_grid(
        tmp_path / "experiments_paclic_v2_maingrid",
        mutate={v2.run_name(0.5, 44): {"rv_train_samples": 1}},
    )
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    aggregate = v2.build_aggregate(
        cells,
        checks,
        v2.coverage_summary(cells),
        subset=v2.subset_consistency(v2.subset_provenance(root, cells), cells),
        attestation=_healthy_attestation(cells, tmp_path, root),
    )
    assert aggregate["complete"] is True
    with pytest.raises(v2.GridInvariantError) as excinfo:
        v2.build_figures(aggregate, tmp_path / "figs")
    assert "publication-blocking" in str(excinfo.value)
    assert not (tmp_path / "figs").exists()


def test_figure_manifest_and_readme_carry_the_attestation_and_warnings(tmp_path):
    """The figure bundle republished ``corpus: v2`` with the evidence stripped out."""
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    # A non-blocking warning that must survive into the figure bundle.
    logs = _write_logs(tmp_path / "slurm", output_dir=f"./outputs/{root.name}", job=506)
    (logs / "paclic_v2_maingrid_505_0.out").write_text(
        _log_text("rv_100_ms_0_seed_42", output_dir=f"./outputs/{root.name}", job=505),
        encoding="utf-8",
    )
    attestation = v2.corpus_attestation_summary(
        v2.attest_corpus_from_slurm_logs(logs), cells, study_root=root
    )
    aggregate = v2.build_aggregate(
        cells, checks, v2.coverage_summary(cells), attestation=attestation
    )
    assert aggregate["warnings"]  # the superseded-log note
    out = tmp_path / "figs"
    v2.build_figures(aggregate, out, make_pdf=False, dpi=300)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["corpus_attestation"]["healthy"] is True
    assert manifest["corpus_attestation"]["expected_corpus"]["corpus"] == "v2"
    assert manifest["warnings"] == aggregate["warnings"]
    assert manifest["hard_warnings"] == []
    assert manifest["publishable"] is True
    readme = (out / "README.md").read_text()
    assert "Corpus attested:" in readme
    assert "superseded_logs" in readme  # the warning is visible to a human reader


def test_figures_render_from_a_complete_grid(tmp_path):
    aggregate = _aggregate(tmp_path)
    out = tmp_path / "figs"
    figures = v2.build_figures(aggregate, out, make_pdf=True, dpi=300)
    assert [f["stem"] for f in figures] == list(v2.V2_FIGURE_STEMS)
    for fig in figures:
        png = out / f"{fig['stem']}.png"
        pdf = out / f"{fig['stem']}.pdf"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert pdf.read_bytes()[:5] == b"%PDF-"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE"
    assert len(manifest["figures"]) == 2
    assert (out / "README.md").is_file()


def test_figure_provenance_points_at_the_v2_bundle_only(tmp_path):
    from scripts.evaluation import build_paclic_paper_figures as bpf

    out = tmp_path / "figs"
    v2.build_figures(_aggregate(tmp_path), out, make_pdf=False, dpi=300)
    blob = (out / "manifest.json").read_text() + (out / "README.md").read_text()
    assert bpf.MULTISEED_DIR not in blob
    assert str(v2.DEFAULT_OUT_DIR) in blob


def test_v2_captions_do_not_repeat_v1_findings(tmp_path):
    out = tmp_path / "figs"
    figures = v2.build_figures(_aggregate(tmp_path), out, make_pdf=False, dpi=300)
    shows = " ".join(f["shows"] for f in figures)
    assert "jumps sharply" not in shows
    assert "no catastrophic forgetting" not in shows
    assert "v2" in shows


def test_figures_are_byte_deterministic(tmp_path):
    aggregate = _aggregate(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    v2.build_figures(aggregate, a, make_pdf=True, dpi=300)
    v2.build_figures(aggregate, b, make_pdf=True, dpi=300)
    for stem in v2.V2_FIGURE_STEMS:
        for ext in ("png", "pdf"):
            assert (a / f"{stem}.{ext}").read_bytes() == (
                b / f"{stem}.{ext}"
            ).read_bytes()


# --------------------------------------------------------------------------- #
# 9. Bundle + CLI.
# --------------------------------------------------------------------------- #
def test_build_bundle_writes_every_artifact(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    out = tmp_path / "bundle"
    aggregate = v2.build_bundle(cells, checks, out, study_root=root)
    assert aggregate["complete"] is True
    for rel in (
        "aggregate.json",
        "manifest.json",
        "results.md",
        "tables/per_seed.csv",
        "tables/per_ratio.csv",
        "tables/asymmetry.csv",
        "tables/retention_adaptation.csv",
    ):
        assert (out / rel).is_file(), rel
    per_seed = (out / "tables/per_seed.csv").read_text()
    assert per_seed.count("\n") == 16  # header + 15 cells
    assert "rv_100_ms_0_seed_42" in per_seed


def test_cli_partial_grid_writes_bundle_and_exits_nonzero(tmp_path, capsys):
    skip = frozenset(v2.run_name(0.0, s) for s in v2.SEEDS)
    root = _write_grid(tmp_path / "v2", skip=skip)
    out = tmp_path / "bundle"
    rc = v2.main(
        [
            "--study-root",
            str(root),
            "--out-dir",
            str(out),
            "--no-slurm-attestation",
            "--no-cross-check",
        ]
    )
    assert rc == 1
    aggregate = json.loads((out / "aggregate.json").read_text())
    assert aggregate["status"] == "PARTIAL"
    assert aggregate["complete"] is False
    assert "NOT A COMPLETE RESULT" in aggregate["partial_banner"]
    assert "PARTIAL" in capsys.readouterr().err


def test_cli_complete_grid_exits_zero_and_can_render_figures(tmp_path):
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    logs = _write_logs(tmp_path / "slurm", output_dir=f"./outputs/{root.name}")
    out = tmp_path / "bundle"
    figs = tmp_path / "figs"
    rc = v2.main(
        [
            "--study-root",
            str(root),
            "--out-dir",
            str(out),
            "--figures-out-dir",
            str(figs),
            "--slurm-log-dir",
            str(logs),
            "--no-cross-check",
            "--no-pdf",
        ]
    )
    assert rc == 0
    aggregate = json.loads((out / "aggregate.json").read_text())
    assert aggregate["status"] == "COMPLETE"
    assert aggregate["publishable"] is True
    assert aggregate["hard_warnings"] == []
    for stem in v2.V2_FIGURE_STEMS:
        assert (figs / f"{stem}.png").is_file()


def test_cli_writes_no_figures_when_it_will_return_nonzero(tmp_path, capsys):
    """The figures were written to disk BEFORE ``main`` returned 1.

    A hard failure therefore left a complete, publishable-looking figure bundle
    behind. Here the grid is complete and attested, so coverage would not stop it —
    only the hard warning does.
    """
    root = _write_grid(
        tmp_path / "experiments_paclic_v2_maingrid",
        mutate={v2.run_name(0.5, 44): {"rv_train_samples": 1}},
    )
    logs = _write_logs(tmp_path / "slurm", output_dir=f"./outputs/{root.name}")
    figs = tmp_path / "figs"
    rc = v2.main(
        [
            "--study-root",
            str(root),
            "--out-dir",
            str(tmp_path / "bundle"),
            "--figures-out-dir",
            str(figs),
            "--slurm-log-dir",
            str(logs),
            "--no-cross-check",
        ]
    )
    assert rc == 1
    assert not figs.exists()
    err = capsys.readouterr().err
    assert "refusing to render v2 paper figures" in err
    assert "hard failure" in err


def test_cli_refuses_figures_without_an_out_dir(tmp_path, capsys):
    """The figure manifest cites the bundle it never wrote."""
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    figs = tmp_path / "figs"
    rc = v2.main(
        [
            "--study-root",
            str(root),
            "--figures-out-dir",
            str(figs),
            "--no-slurm-attestation",
            "--no-cross-check",
        ]
    )
    assert rc == 2
    assert not figs.exists()
    assert "--figures-out-dir requires --out-dir" in capsys.readouterr().err


def test_cli_healthy_complete_grid_with_per_seed_hashes_can_go_green(tmp_path):
    """End-to-end Blocker 1: a healthy 15-cell grid must be able to exit 0.

    Three per-seed ``key_list_sha256`` values used to trip the grid-wide check, and
    ``main`` hard-fails on "not a single study" — so this exit status could never be
    0 once the second seed landed, no matter how healthy the grid was.
    """
    root = _write_grid(tmp_path / "experiments_paclic_v2_maingrid")
    logs = _write_logs(tmp_path / "slurm", output_dir=f"./outputs/{root.name}")
    out = tmp_path / "bundle"
    rc = v2.main(
        [
            "--study-root",
            str(root),
            "--out-dir",
            str(out),
            "--slurm-log-dir",
            str(logs),
            "--no-cross-check",
        ]
    )
    assert rc == 0
    results = (out / "results.md").read_text()
    assert "INCONSISTENT" not in results
    assert "not a single study" not in results
    aggregate = json.loads((out / "aggregate.json").read_text())
    assert aggregate["subset_consistency"]["consistent"] is True
    assert (
        len(aggregate["subset_consistency"]["key_list_sha256_distinct_grid_wide"]) == 3
    )


def test_cli_skips_figures_on_a_partial_grid_without_failing_the_run(tmp_path, capsys):
    root = _write_grid(tmp_path / "v2", skip=frozenset({v2.run_name(0.5, 42)}))
    figs = tmp_path / "figs"
    rc = v2.main(
        [
            "--study-root",
            str(root),
            "--out-dir",
            str(tmp_path / "bundle"),
            "--figures-out-dir",
            str(figs),
            "--no-slurm-attestation",
            "--no-cross-check",
        ]
    )
    assert rc == 1  # partial is reported through the exit status...
    assert not figs.exists()  # ...and no figure is written
    assert "refusing to render" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 10. The launcher and this module must agree on names/paths.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 11. Per-metric n, per-cell invariants, and the coverage target.
# --------------------------------------------------------------------------- #
def test_section1_gives_each_metric_its_own_n_when_denominators_differ(tmp_path):
    """§1's single row ``n`` spanned metrics with different denominators.

    ``read_cell_dir`` clears exact-match when the prediction CSV is absent, and the
    runner writes ``results.json`` *before* those CSVs — the normal mid-flight state.
    Observed: the row said n=3 while ``ms_exact_match`` n was 2.
    """
    no_preds = v2.run_name(1.0, 44)
    root = _write_grid(
        tmp_path / "v2", with_preds=True, preds_skip=frozenset({no_preds})
    )
    cells, checks = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    by_ratio = agg.aggregate_by_ratio(cells)
    # The premise: CER has n=3 for this ratio, exact-match only n=2.
    assert by_ratio[1.0]["rv_cer"].n == 3
    assert by_ratio[1.0]["ms_exact_match"].n == 2

    md = _render(cells, checks)
    section1 = md.split("## 1. Per mix ratio")[1].split("## 2.")[0]
    row = next(line for line in section1.splitlines() if line.startswith("| 100:0 |"))
    # Coverage is stated as coverage, not as a metric's n...
    assert row.startswith("| 100:0 | 3/3 |")
    # ...and every value carries its own n, so the n=2 metric cannot borrow n=3.
    assert row.count("(n=3)") == 7
    assert row.count("(n=2)") == 2
    assert "cells` is coverage, never a metric's n" in md


def test_results_json_seed_must_match_its_run_dir(tmp_path):
    """A ratio backed by 2 distinct seeds published as n=3 before this check.

    ``read_cell_dir`` takes the seed from ``results.json``; the dir name encodes the
    requested seed. A copied run dir therefore duplicates a seed silently.
    """
    root = _write_grid(tmp_path / "v2", mutate={v2.run_name(0.5, 44): {"seed": 43}})
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    integrity = v2.grid_integrity(cells, study_root=root)
    assert integrity["ok"] is False
    assert [m["run_name"] for m in integrity["seed_mismatches"]] == [
        v2.run_name(0.5, 44)
    ]
    assert integrity["seed_mismatches"][0]["dir_seed"] == 44
    assert integrity["seed_mismatches"][0]["results_json_seed"] == 43
    # ...and the same grid is caught on the axis that matters: the 0.5 ratio now has
    # three present cells over two distinct seeds.
    dup = integrity["ratio_seed_duplicates"]
    assert [d["rv_ratio"] for d in dup] == [0.5]
    assert dup[0]["n_present"] == 3
    assert dup[0]["n_distinct_seeds"] == 2

    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), integrity=integrity)
    assert any("disagrees with its run dir" in w for w in warnings)
    assert any("do not carry distinct seeds" in w for w in warnings)
    assert v2.hard_warnings(warnings)


def test_a_clean_grid_has_no_integrity_findings(tmp_path):
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    integrity = v2.grid_integrity(cells, study_root=root)
    assert integrity["ok"] is True
    assert integrity["unreadable_check_performed"] is True
    assert (
        v2.audit_warnings(cells, v2.coverage_summary(cells), integrity=integrity) == []
    )


def test_a_torn_results_json_is_reported_not_fatal(tmp_path):
    """An unguarded ``json.loads`` in the shared reader killed the whole aggregation.

    The grid is read *while* the launcher writes these files, so a partial write is a
    normal state; it must degrade to one absent cell, loudly.
    """
    torn = v2.run_name(0.25, 43)
    root = _write_grid(tmp_path / "v2", corrupt=frozenset({torn}))
    # The shared v1 reader still raises on it — this module guards its own loader.
    with pytest.raises(json.JSONDecodeError):
        agg.read_cell_dir(
            root / torn,
            rv_ratio=0.25,
            seed=43,
            run_name=torn,
            source="x",
            shared_path=None,
            metrics=agg._MetricsCache(_fake_metrics_fn),
            cross_check=False,
        )

    cells, _ = v2.load_grid(root, metrics_fn=_fake_metrics_fn, cross_check=False)
    assert len(cells) == 15
    assert [c.run_name for c in cells if not c.present] == [torn]
    integrity = v2.grid_integrity(cells, study_root=root)
    assert integrity["unreadable_results_json"] == [torn]
    warnings = v2.audit_warnings(cells, v2.coverage_summary(cells), integrity=integrity)
    assert any("did not parse" in w for w in warnings)
    assert v2.hard_warnings(warnings)


def test_complete_is_measured_against_the_fifteen_cell_target(tmp_path):
    """``n_present == len(cells)`` called a two-seed load a complete grid."""
    root = _write_grid(tmp_path / "v2")
    cells, _ = v2.load_grid(
        root, metrics_fn=_fake_metrics_fn, cross_check=False, seeds=(42, 43)
    )
    assert len(cells) == 10
    assert all(c.present for c in cells)  # every LOADED cell is present...
    cov = v2.coverage_summary(cells)
    assert cov["complete"] is False  # ...but the grid is not complete
    assert cov["n_cells"] == 15
    assert cov["n_cells_loaded"] == 10
    assert cov["n_present"] == 10
    assert v2.status_label(cov) == "PARTIAL"
    assert sorted(cov["missing_cells"]) == sorted(
        v2.run_name(r, 44) for r in v2.RV_RATIOS
    )
    assert cov["per_seed"]["44"]["n_present"] == 0
    banner = v2.partial_banner(cov)
    assert banner is not None and "10/15" in banner


def test_missing_study_root_reports_fifteen_gaps_not_a_crash(tmp_path):
    cells, _ = v2.load_grid(
        tmp_path / "nope", metrics_fn=_fake_metrics_fn, cross_check=False
    )
    cov = v2.coverage_summary(cells)
    assert cov["n_present"] == 0
    assert len(cov["missing_cells"]) == 15
    banner = v2.partial_banner(cov)
    assert banner is not None and "0/15" in banner
