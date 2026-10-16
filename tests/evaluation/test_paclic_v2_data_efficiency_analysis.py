"""Test paclic v2 data efficiency analysis: PACLIC 2026 v2 release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation import aggregate_accent_ratio_multiseed as agg
from scripts.evaluation import build_paclic_v2_data_efficiency_analysis as de
from scripts.evaluation import build_paclic_v2_maingrid_analysis as v2mg


# --------------------------------------------------------------------------- #
# Synthetic study roots.
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


#: ``(total, rv_ratio) → (rv_cer, ms_cer, rv_accent_f1, ms_accent_f1)``, all four
#: points of the curve. Shaped after the live data: the 10% point is worst on the
#: own-notation channels, 100:0's ms_* metrics *degrade* with more data (so a
#: sign-flipped delta is exercised), and 50:50's rv_cer falls in front-loaded steps
#: (−0.04, −0.02, −0.01) so the step ranking is deterministic and its "consistently
#: ordered, not separable" verdict is a real assertion rather than a coin flip.
_SPEC: dict[tuple[int, float], tuple[float, float, float, float]] = {
    (1042, 1.0): (0.07, 0.22, 0.88, 0.18),
    (2605, 1.0): (0.055, 0.24, 0.91, 0.13),
    (5210, 1.0): (0.045, 0.25, 0.93, 0.09),
    (10420, 1.0): (0.04, 0.26, 0.95, 0.06),
    (1042, 0.5): (0.11, 0.09, 0.70, 0.82),
    (2605, 0.5): (0.07, 0.06, 0.85, 0.90),
    (5210, 0.5): (0.05, 0.05, 0.90, 0.92),
    (10420, 0.5): (0.04, 0.04, 0.93, 0.94),
    (1042, 0.0): (0.30, 0.06, 0.05, 0.95),
    (2605, 0.0): (0.30, 0.05, 0.04, 0.955),
    (5210, 0.0): (0.30, 0.045, 0.035, 0.955),
    (10420, 0.0): (0.30, 0.04, 0.03, 0.96),
}


def _key_lines(total: int, seed: int) -> str:
    """A cell's ``parallel_keys.txt``: a nested prefix ladder within a seed.

    The real selection shuffles the sorted shared keys with a seed that does not
    depend on the total and then takes a prefix, so within one seed every total's key
    list is a prefix of the next. Reproduced here (scaled down) so the nesting check
    has something to verify rather than something to assume.
    """
    return "".join(f"key-{seed}-{i}\n" for i in range(total // 1000 + 1))


def key_list_sha(total: int, seed: int) -> str:
    """The subset hash a healthy cell records — per ``(total, seed)``, never wider.

    ``dataset.py`` sorts the shared keys, seed-shuffles them and takes a **prefix** of
    length ``total_train_samples``, so the same seed at two totals is two different
    ordered lists. A healthy 36-cell join therefore carries twelve distinct hashes (3
    seeds x 4 totals); the live data confirms it (seed 42 hashed ``73793f5216…`` at
    1042, ``a75bf2fece…`` at 2605, ``3f9fc7301d…`` at 5210 and ``ca03ed92dd…`` at
    10420 — four different hashes for four nested prefixes of one shuffle).
    """
    return f"sha-{total}-{seed}"


def _payload(total: int, rv_ratio: float, seed: int) -> dict:
    """A v2 ``results.json`` — chrF and inline ``training_state`` for every cell."""
    rv_cer, ms_cer, rv_f1, ms_f1 = _SPEC[(total, rv_ratio)]
    # Seed noise that grows with the total, so the per-seed STEP magnitudes differ
    # across seeds and a step's CI is not degenerate.
    jitter = (seed - 43) * 0.001 * (1.0 + total / de.FULL_TOTAL)
    n_rv, n_ms = v2mg.expected_mix_counts(total, rv_ratio)
    return {
        "experiment_name": de.run_name(rv_ratio, seed, total=total),
        "seed": seed,
        "model_base": "fixed-foundation",
        "freeze_mode": "none",
        "input_scheme": "iso",
        "num_train_epochs": 20,
        "training_state": {"best_epoch": 16.0, "best_step": 528, "global_step": 627},
        "mix_mode": "parallel_by_text",
        "key_column": "text",
        "selection_algorithm": "sorted_common_keys+seeded_shuffle+prefix",
        "parallel_keys_total": de.FULL_TOTAL,
        "parallel_keys_used": total,
        "parallel_subset_seed": seed,
        "key_list_sha256": key_list_sha(total, seed),
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


def _write_roots(
    tmp_path: Path,
    *,
    skip: frozenset[str] = frozenset(),
    with_preds: bool = False,
    mutate: dict[str, dict] | None = None,
    corrupt: frozenset[str] = frozenset(),
    no_key_lists: bool = False,
) -> tuple[Path, Path]:
    """Write both study roots (all four totals); ``skip`` names cells to leave absent.

    ``low`` holds every TRAINED total, exactly as the live launcher does — one root,
    told apart by the ``n<total>`` token — and ``full`` the inherited 100% cells.
    """
    low = tmp_path / "outputs" / "experiments_paclic_v2_data_efficiency"
    full = tmp_path / "outputs" / "experiments_paclic_v2_maingrid"
    for total in de.TOTALS:
        root = full if total == de.FULL_TOTAL else low
        for rv_ratio in de.RV_RATIOS:
            for seed in de.SEEDS:
                name = de.run_name(rv_ratio, seed, total=total)
                if name in skip:
                    continue
                cell = root / name
                cell.mkdir(parents=True, exist_ok=True)
                if name in corrupt:
                    (cell / "results.json").write_text(
                        '{"seed": 42, "rv_test": {"cer": 0.0', encoding="utf-8"
                    )
                    continue
                payload = _payload(total, rv_ratio, seed)
                if mutate and name in mutate:
                    payload.update(mutate[name])
                (cell / "results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                if not no_key_lists:
                    (cell / de.PARALLEL_KEYS_FILE).write_text(
                        _key_lines(total, seed), encoding="utf-8"
                    )
                if with_preds:
                    for view in ("rv", "ms"):
                        (cell / f"{view}_predictions.csv").write_text(
                            "reference,prediction\na,a\nb,c\n", encoding="utf-8"
                        )
    return low, full


def _log_text(
    run_name: str,
    *,
    prefix: str,
    output_dir: str,
    corpus: str = "v2",
    rv: str = "data/datasets/rv_v2",
    ms: str = "data/datasets/ms_v2",
    data_source: str | None = "csv",
    job: int = 520,
    task: int = 0,
) -> str:
    """One Slurm ``.out`` header, in the four-token form both launchers echo."""
    guard = (
        f"[{prefix}] corpus CSVs + first images present: rv={rv} ms={ms}\n"
        if data_source == "csv"
        else ""
    )
    source = "" if data_source is None else f" data_source={data_source}"
    return (
        guard + f"[{prefix}] task={task}/8 run_name={run_name}  seed=42 smoke=0\n"
        f"[{prefix}] node=gpu-worker  job={job} array={job}_{task}\n"
        f"[{prefix}] corpus={corpus} rv={rv} ms={ms}{source}\n"
        f"[{prefix}] output_dir={output_dir} tags=[paclic,corpus-v2]\n"
        "Starting training\n"
    )


def _write_logs(
    log_dir: Path, *, low: Path, full: Path, totals=de.TOTALS, **kwargs
) -> Path:
    """A healthy log per target cell, under each arm's OWN glob and output_dir.

    The three trained totals share the dataeff glob (one launcher wrote them), so they
    are given DIFFERENT job ids — as separate submissions really do — and are told
    apart by the run name inside the log, which is what the attestation matches on.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    for total in totals:
        stem = de.SLURM_LOG_GLOB_BY_TOTAL[total].replace("_*.out", "")
        inherited = total == de.FULL_TOTAL
        prefix = "paclic-v2-maingrid" if inherited else "paclic-v2-dataeff"
        root = full if inherited else low
        job = 506 if inherited else 520 + de.TRAINED_TOTALS.index(total)
        task = 0
        for rv_ratio in de.RV_RATIOS:
            for seed in de.SEEDS:
                (log_dir / f"{stem}_{job}_{task}.out").write_text(
                    _log_text(
                        de.run_name(rv_ratio, seed, total=total),
                        prefix=prefix,
                        output_dir=f"./outputs/{root.name}",
                        job=job,
                        task=task,
                        **kwargs,
                    ),
                    encoding="utf-8",
                )
                task += 1
    return log_dir


def _fake_metrics_fn(csv_path: Path) -> dict:
    """Deterministic stand-in for the study evaluator (no jiwer/torch import)."""
    return {
        "cer": 0.041,
        "wer": 0.205,
        "chrf": 88.0,
        "accent_f1": 0.952,
        "exact_match": 0.1,
        "exact_match_count": 58,
        "n": 578,
    }


def _maingrid_aggregate(
    per_total: dict[int, dict[float, dict[str, agg.Stat]]],
) -> dict:
    """A stand-in committed main-grid ``aggregate.json`` that MATCHES the join.

    Round-tripped through JSON, like the real file, so the tolerance is exercised
    rather than bypassed.
    """
    payload = {
        "study": v2mg.STUDY,
        "status": "COMPLETE",
        "complete": True,
        "per_ratio": {
            f"{rv:g}": {metric: stat.as_dict() for metric, stat in metrics.items()}
            for rv, metrics in per_total[de.FULL_TOTAL].items()
        },
    }
    return json.loads(json.dumps(payload))


@pytest.fixture
def whole_curve(tmp_path: Path):
    """A complete, healthy, attested, verified 36-cell join."""
    low, full = _write_roots(tmp_path, with_preds=True)
    curve, checks = de.load_curve(low, full, metrics_fn=_fake_metrics_fn)
    per_total = de.aggregate_curve(curve)
    logs = _write_logs(tmp_path / "slurm", low=low, full=full)
    attestation = de.corpus_attestation(
        curve, log_dir=logs, low_study_root=low, full_study_root=full
    )
    subset = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
        nesting=de.subset_nesting(curve, low_study_root=low, full_study_root=full),
    )
    integrity = de.grid_integrity(curve, low_study_root=low, full_study_root=full)
    maingrid = de.maingrid_consistency(per_total, _maingrid_aggregate(per_total))
    aggregate = de.build_aggregate(
        curve,
        per_total,
        checks,
        de.coverage_summary(curve),
        subset=subset,
        attestation=attestation,
        integrity=integrity,
        maingrid=maingrid,
    )
    return {
        "low": low,
        "full": full,
        "logs": logs,
        "curve": curve,
        "checks": checks,
        "per_total": per_total,
        "aggregate": aggregate,
        "tmp_path": tmp_path,
    }


# --------------------------------------------------------------------------- #
# 1. Why neither neighbouring builder can read this study.
# --------------------------------------------------------------------------- #
def test_maingrid_builder_cannot_address_the_low_data_cells() -> None:
    """``v2mg.run_name`` emits no ``n<total>`` token, so it names the wrong dir."""
    assert v2mg.run_name(1.0, 42) == "rv_100_ms_0_seed_42"
    assert de.run_name(1.0, 42, total=de.LOW_TOTAL) == "rv_100_ms_0_n1042_seed_42"
    assert de.run_name(1.0, 42, total=de.LOW_TOTAL) != v2mg.run_name(1.0, 42)


def test_maingrid_builder_target_is_a_five_ratio_single_total_grid() -> None:
    """Its target size has no total axis, and it counts two ratios this study lacks."""
    assert v2mg.TARGET_N_CELLS == len(v2mg.RV_RATIOS) * len(v2mg.SEEDS) == 15
    assert de.TARGET_N_CELLS == len(de.TOTALS) * len(de.RV_RATIOS) * len(de.SEEDS)
    assert set(de.RV_RATIOS) < set(v2mg.RV_RATIOS)
    assert {0.75, 0.25}.isdisjoint(de.RV_RATIOS)


def test_maingrid_builder_holds_the_curve_axis_fixed_grid_wide(tmp_path: Path) -> None:
    """``v2mg.subset_consistency`` would call this healthy design "not a single study".

    ``total_train_samples_requested`` is one of its grid-wide held-fixed fields, and it
    is precisely the field a data-efficiency curve varies.
    """
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    cells = de.all_cells(curve)
    prov: dict[str, dict] = {}
    for total in de.TOTALS:
        root = full if total == de.FULL_TOTAL else low
        prov.update(v2mg.subset_provenance(root, curve[total]))
    verdict = v2mg.subset_consistency(prov, cells)
    assert "total_train_samples_requested" in verdict["inconsistent_fields"]
    assert verdict["grid_wide_consistent"] is False
    # The study's own check, at the right scopes, passes on the same cells.
    ours = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
    )
    assert ours["consistent"] is True


# --------------------------------------------------------------------------- #
# 2. The join: two roots, two namings, explicit provenance.
# --------------------------------------------------------------------------- #
def test_inherited_run_names_delegate_to_the_maingrid_builder() -> None:
    """The 100% arm can only address the dirs the main-grid bundle addressed."""
    for rv_ratio in de.RV_RATIOS:
        for seed in de.SEEDS:
            assert de.run_name(rv_ratio, seed, total=de.FULL_TOTAL) == v2mg.run_name(
                rv_ratio, seed
            )


def test_load_curve_reads_both_roots_and_stamps_provenance(tmp_path: Path) -> None:
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    assert sorted(curve) == sorted(de.TOTALS)
    assert all(
        len(cells) == len(de.RV_RATIOS) * len(de.SEEDS) for cells in curve.values()
    )
    assert {c.source for c in curve[de.LOW_TOTAL]} == {
        de.CELL_SOURCE_BY_TOTAL[de.LOW_TOTAL]
    }
    assert {c.source for c in curve[de.FULL_TOTAL]} == {
        de.CELL_SOURCE_BY_TOTAL[de.FULL_TOTAL]
    }
    # Every cell resolved under its OWN root, never the other one.
    for total, root in ((de.LOW_TOTAL, low), (de.FULL_TOTAL, full)):
        for cell in curve[total]:
            assert cell.present
            assert str(root) in str(cell.results_path)


def test_provenance_csv_names_the_study_behind_every_cell(whole_curve) -> None:
    text = de.render_provenance_csv(
        whole_curve["curve"], whole_curve["aggregate"]["corpus_attestation"]
    )
    lines = text.strip().splitlines()
    assert lines[0].startswith("total_train_samples,pct,")
    assert "source_study" in lines[0] and "provenance" in lines[0]
    assert len(lines) == 1 + de.TARGET_N_CELLS
    # Counted with delimiters: the inherited cells' source_study
    # ("v2-maingrid-inherited") also contains the word.
    assert text.count(",newly_trained,") == de.TARGET_N_NEW_CELLS
    assert text.count(",inherited,") == de.TARGET_N_CELLS - de.TARGET_N_NEW_CELLS


def test_curve_csv_carries_total_pct_and_source(whole_curve) -> None:
    text = de.render_curve_csv(whole_curve["per_total"])
    header = text.splitlines()[0].split(",")
    # agg's per-ratio columns, verbatim, after the three join columns.
    assert header[:4] == ["total_train_samples", "pct", "source_study", "provenance"]
    assert header[4:] == list(agg.PER_RATIO_COLUMNS)
    assert len(text.strip().splitlines()) == 1 + len(de.TOTALS) * len(
        de.RV_RATIOS
    ) * len(de.CURVE_METRICS)
    # No phantom rows for the two ratios this study does not train.
    assert ",0.75," not in text and ",0.25," not in text


# --------------------------------------------------------------------------- #
# 3. Statistics are agg's, not reimplemented.
# --------------------------------------------------------------------------- #
def test_per_point_stats_are_exactly_agg_aggregate_by_ratio(whole_curve) -> None:
    curve = whole_curve["curve"]
    for total, cells in curve.items():
        reference = agg.aggregate_by_ratio(cells)
        for rv_ratio in de.RV_RATIOS:
            for metric in de.CURVE_METRICS:
                assert (
                    whole_curve["per_total"][total][rv_ratio][metric]
                    == reference[rv_ratio][metric]
                )


def test_curve_deltas_are_exactly_agg_contrast(whole_curve) -> None:
    curve = whole_curve["curve"]
    deltas = de.curve_deltas(curve)
    for from_total, to_total in de.CONTRAST_PAIRS:
        for rv_ratio in de.RV_RATIOS:
            for metric in de.CURVE_METRICS:
                reference = agg._contrast(
                    de._seed_values(curve[from_total], rv_ratio, metric),
                    de._seed_values(curve[to_total], rv_ratio, metric),
                )
                got = de.contrast_of(
                    deltas, de.contrast_key(from_total, to_total), rv_ratio, metric
                )
                assert got is not None
                assert got["diff"] == reference["diff"]
                assert got["per_seed_diff"] == reference["per_seed_diff"]
                assert got["sign_consistent"] == reference["sign_consistent"]
                assert (got["from_total"], got["to_total"]) == (from_total, to_total)


def test_published_contrasts_are_the_steps_plus_the_endpoint(whole_curve) -> None:
    """Four of the six pairs, and the other two stated as omitted, not forgotten."""
    deltas = whole_curve["aggregate"]["curve_deltas"]
    assert [p["contrast"] for p in deltas["pairs"]] == [
        "1042->2605",
        "2605->5210",
        "5210->10420",
        "1042->10420",
    ]
    assert [p["contrast_kind"] for p in deltas["pairs"]] == [
        de.KIND_STEP,
        de.KIND_STEP,
        de.KIND_STEP,
        de.KIND_ENDPOINT,
    ]
    assert deltas["omitted_pairs"] == ["1042->5210", "2605->10420"]
    assert len(deltas["pairs"]) + len(deltas["omitted_pairs"]) == 6
    assert "minimal and complete" in deltas["selection_note"]


def test_an_omitted_pair_is_the_exact_sum_of_published_steps(whole_curve) -> None:
    """The claim that justifies omitting it, checked rather than asserted.

    Every contrast is a within-seed difference, so 1042→5210 must equal
    [1042→2605] + [2605→5210] seed by seed. If it did not, omitting it would be
    hiding a measurement instead of not repeating one.
    """
    curve = whole_curve["curve"]
    deltas = whole_curve["aggregate"]["curve_deltas"]
    for rv_ratio in de.RV_RATIOS:
        for metric in de.CURVE_METRICS:
            omitted = agg._contrast(
                de._seed_values(curve[1042], rv_ratio, metric),
                de._seed_values(curve[5210], rv_ratio, metric),
            )["per_seed_diff"]
            first = de.contrast_of(deltas, "1042->2605", rv_ratio, metric)
            second = de.contrast_of(deltas, "2605->5210", rv_ratio, metric)
            assert first is not None and second is not None
            for seed, value in omitted.items():
                assert value == pytest.approx(
                    first["per_seed_diff"][seed] + second["per_seed_diff"][seed]
                )


def test_delta_direction_follows_the_data_not_the_key_name(whole_curve) -> None:
    """A metric that got WORSE with more data reports a positive Δ, not a hidden one."""
    deltas = de.curve_deltas(whole_curve["curve"])
    endpoint = de.contrast_key(*de.ENDPOINT_PAIR)

    def contrast(key: str, metric: str) -> dict:
        got = de.contrast_of(deltas, key, 1.0, metric)
        assert got is not None, (key, metric)
        return got

    # ms_cer at 100:0 rises from 0.22 to 0.26 with 10x the data (the fixture mirrors
    # the live behaviour), while rv_cer falls — on the endpoint and on every step.
    assert contrast(endpoint, "ms_cer")["diff"]["mean"] > 0
    assert contrast(endpoint, "ms_cer")["direction"] == "b>a"
    assert contrast(endpoint, "rv_cer")["diff"]["mean"] < 0
    assert contrast(endpoint, "ms_accent_f1")["diff"]["mean"] < 0
    for step in de.CONSECUTIVE_STEPS:
        assert contrast(de.contrast_key(*step), "ms_accent_f1")["diff"]["mean"] < 0


def test_delta_csv_rows_come_from_agg_contrast_rows(whole_curve) -> None:
    text = de.render_delta_csv(de.curve_deltas(whole_curve["curve"]))
    header = text.splitlines()[0].split(",")
    assert header[:6] == [
        "contrast",
        "contrast_kind",
        "from_total",
        "to_total",
        "from_pct",
        "to_pct",
    ]
    assert header[6:] == list(agg.ASYMMETRY_COLUMNS)
    assert len(text.strip().splitlines()) == 1 + len(de.CONTRAST_PAIRS) * len(
        de.RV_RATIOS
    ) * len(de.CURVE_METRICS)
    assert text.count(f",{de.KIND_ENDPOINT},") == len(de.RV_RATIOS) * len(
        de.CURVE_METRICS
    )


def test_asymmetry_is_agg_asymmetry_per_total(whole_curve) -> None:
    per_total = de.asymmetry_by_total(whole_curve["curve"])
    for total, cells in whole_curve["curve"].items():
        assert per_total[str(total)] == agg.asymmetry(cells)


def test_retention_and_min_ratio_are_omitted_with_a_stated_reason(whole_curve) -> None:
    """Absent, not empty: an empty table would read as "measured, no effect"."""
    aggregate = whole_curve["aggregate"]
    assert aggregate["retention_adaptation"] is None
    assert "0.75" in aggregate["retention_adaptation_note"]
    assert aggregate["minimum_target_ratio"] is None
    assert "pre-registered" in aggregate["minimum_target_ratio_note"]


# --------------------------------------------------------------------------- #
# 4. Coverage, the PAIRED n, and the PARTIAL banner.
# --------------------------------------------------------------------------- #
def test_complete_curve_reports_complete(whole_curve) -> None:
    coverage = de.coverage_summary(whole_curve["curve"])
    assert coverage["complete"] is True
    assert coverage["n_present"] == de.TARGET_N_CELLS
    assert coverage["new_cells_complete"] is True
    assert de.status_label(coverage) == "COMPLETE"
    assert de.partial_banner(coverage) is None
    assert coverage["missing_cells"] == []
    assert all(blk["n_paired"] == len(de.SEEDS) for blk in coverage["paired"].values())


def test_partial_curve_names_the_missing_cell_and_drops_the_paired_n(
    tmp_path: Path,
) -> None:
    """The real mid-flight state: one interior cell has not landed yet.

    On a four-point curve one absent cell does NOT lower every delta's n: it lowers
    exactly the two steps and the endpoint that touch its total, and leaves the rest at
    3. That asymmetry is the reason the paired n is reported per contrast.
    """
    missing = de.run_name(0.0, 44, total=2605)
    low, full = _write_roots(tmp_path, skip=frozenset({missing}))
    curve, _ = de.load_curve(low, full)
    coverage = de.coverage_summary(curve)

    assert coverage["complete"] is False
    assert coverage["n_present"] == de.TARGET_N_CELLS - 1
    assert coverage["n_new_present"] == de.TARGET_N_NEW_CELLS - 1
    assert coverage["missing_cells"] == [missing]
    assert de.status_label(coverage) == "PARTIAL"

    # The point loses one seed...
    assert coverage["per_point"]["2605:0"]["n_present"] == 2
    assert coverage["per_point"]["2605:0"]["seeds_missing"] == [44]
    # ...and so do exactly the deltas that read it, even though its neighbours are 3/3.
    assert coverage["per_point"][f"{de.LOW_TOTAL}:0"]["n_present"] == 3
    assert coverage["per_point"]["5210:0"]["n_present"] == 3
    assert coverage["paired"]["1042->2605:0"]["n_paired"] == 2
    assert coverage["paired"]["1042->2605:0"]["seeds_paired"] == [42, 43]
    assert coverage["paired"]["2605->5210:0"]["n_paired"] == 2
    # Untouched: the endpoint spans 1042→10420 and never reads 2605.
    assert coverage["paired"]["1042->10420:0"]["n_paired"] == 3
    assert coverage["paired"]["5210->10420:0"]["n_paired"] == 3
    assert coverage["paired"]["1042->2605:1"]["n_paired"] == 3

    banner = de.partial_banner(coverage)
    assert banner is not None
    assert banner.startswith("PARTIAL")
    assert "NOT A COMPLETE RESULT" in banner
    assert f"{de.TARGET_N_CELLS - 1}/{de.TARGET_N_CELLS}" in banner
    assert missing in banner
    assert "PAIRED n says" in banner


def test_partial_curve_warns_that_a_delta_lost_its_pairing(tmp_path: Path) -> None:
    low, full = _write_roots(
        tmp_path, skip=frozenset({de.run_name(0.0, 44, total=de.LOW_TOTAL)})
    )
    curve, _ = de.load_curve(low, full)
    warnings = de.audit_warnings(curve, de.coverage_summary(curve))
    assert any("paired seeds" in w for w in warnings)
    assert any("paired n=2" in w for w in warnings)


def test_partial_banner_is_a_hard_warning(tmp_path: Path) -> None:
    low, full = _write_roots(
        tmp_path, skip=frozenset({de.run_name(0.0, 44, total=de.LOW_TOTAL)})
    )
    curve, _ = de.load_curve(low, full)
    warnings = de.audit_warnings(curve, de.coverage_summary(curve))
    hard = de.hard_warnings(warnings)
    assert any(w.startswith("PARTIAL") for w in hard)


def test_unpaired_delta_is_not_hard_on_its_own() -> None:
    """The pairing warning alone must not block: it is implied by PARTIAL.

    Blocking on it twice would be harmless, but a warning that can never be cleared
    independently of coverage is not its own gate.
    """
    warning = (
        "published delta(s) resting on fewer than 3 paired seeds, so the delta's n is "
        "NOT the per-point n: 10→25% 0:100 paired n=2"
    )
    assert de.hard_warnings([warning]) == []


def test_results_md_leads_with_status_and_carries_every_n(tmp_path: Path) -> None:
    low, full = _write_roots(
        tmp_path, skip=frozenset({de.run_name(0.0, 44, total=de.LOW_TOTAL)})
    )
    curve, checks = de.load_curve(low, full)
    coverage = de.coverage_summary(curve)
    text = de.render_results_md(
        curve, de.aggregate_curve(curve), de.curve_deltas(curve), checks, coverage
    )
    assert "**Status:** PARTIAL" in text
    assert "PARTIAL — NOT A COMPLETE RESULT" in text
    assert "(n=2)" in text  # the 0:100 low-data point
    assert "(n=3)" in text
    assert "paired n" in text
    assert "newly_trained" in text and "inherited" in text
    # No v1 comparison anywhere in the page.
    assert "v1" in text  # the "never pooled with v1" statement
    assert "v1→v2" not in text and "vs v1" not in text


# --------------------------------------------------------------------------- #
# 5. Recipe / subset consistency at three scopes.
# --------------------------------------------------------------------------- #
def test_healthy_join_has_one_subset_hash_per_total_and_seed(whole_curve) -> None:
    subset = whole_curve["aggregate"]["subset_consistency"]
    assert subset["key_list_sha256_consistent"] is True
    assert subset["key_list_sha256_split_groups"] == []
    assert (
        len(subset["key_list_sha256_distinct_join_wide"])
        == subset["key_list_sha256_expected_distinct"]
        == len(de.SEEDS) * len(de.TOTALS)
    )
    assert subset["consistent"] is True


def test_subset_hash_split_within_a_total_and_seed_is_hard(tmp_path: Path) -> None:
    """Two ratios of one (total, seed) on different subsets is a real defect."""
    name = de.run_name(0.5, 42, total=de.LOW_TOTAL)
    low, full = _write_roots(
        tmp_path, mutate={name: {"key_list_sha256": "a-different-subset"}}
    )
    curve, _ = de.load_curve(low, full)
    subset = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
    )
    assert subset["key_list_sha256_split_groups"] == [f"{de.LOW_TOTAL}:42"]
    assert subset["consistent"] is False
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), subset=subset)
    assert any("not constant within (total, seed)" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_a_cell_requesting_the_wrong_total_is_caught(tmp_path: Path) -> None:
    """A 1042 dir whose results.json requested 10420 is a mislabelled cell.

    No join-wide check can see this — ``total_train_samples_requested`` is excluded
    there on purpose — so the per-total scope is what catches it.
    """
    name = de.run_name(1.0, 42, total=de.LOW_TOTAL)
    low, full = _write_roots(
        tmp_path,
        mutate={name: {"total_train_samples_requested": de.FULL_TOTAL}},
    )
    curve, _ = de.load_curve(low, full)
    subset = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
    )
    assert subset["total_axis_mismatches"]
    assert (
        subset["per_total"][str(de.LOW_TOTAL)][
            "total_train_samples_requested_matches_axis"
        ]
        is False
    )
    assert subset["consistent"] is False
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), subset=subset)
    assert any("does not equal the total" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_join_wide_recipe_split_is_hard(tmp_path: Path) -> None:
    name = de.run_name(1.0, 42, total=de.LOW_TOTAL)
    low, full = _write_roots(tmp_path, mutate={name: {"model_base": "other-model"}})
    curve, _ = de.load_curve(low, full)
    subset = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
    )
    assert "model_base" in subset["join_wide"]["inconsistent_fields"]
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), subset=subset)
    assert any("join-wide recipe field" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_mix_realization_uses_each_cell_own_total(tmp_path: Path) -> None:
    """The floor split is checked against the cell's own total, not a global one."""
    name = de.run_name(0.5, 42, total=de.LOW_TOTAL)
    low, full = _write_roots(
        tmp_path, mutate={name: {"rv_train_samples": 5210, "ms_train_samples": 5210}}
    )
    curve, _ = de.load_curve(low, full)
    subset = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
    )
    assert subset["mix_mismatches"] == [name]
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), subset=subset)
    assert de.hard_warnings(warnings)


def test_healthy_mix_realization_matches_both_totals(whole_curve) -> None:
    subset = whole_curve["aggregate"]["subset_consistency"]
    assert subset["mix_mismatches"] == []
    by_total = {
        row["total_train_samples_requested"] for row in subset["mix_realization"]
    }
    assert by_total == set(de.TOTALS)


# --------------------------------------------------------------------------- #
# 6. Corpus attestation — per arm, never assumed.
# --------------------------------------------------------------------------- #
def test_attestation_is_healthy_at_every_point(whole_curve) -> None:
    att = whole_curve["aggregate"]["corpus_attestation"]
    assert att["healthy"] is True
    assert att["single_corpus"] is True
    assert att["corpora"] == ["v2"]
    assert att["n_unattested"] == 0
    assert sorted(att["per_total"], key=int) == [str(t) for t in sorted(de.TOTALS)]
    assert all(blk["healthy"] for blk in att["per_total"].values())
    # Each point recorded its own launcher; the trained totals share one glob (one
    # launcher wrote them) and the inherited point has its own.
    for total in de.TRAINED_TOTALS:
        blk = att["per_total"][str(total)]
        assert blk["launcher"] == de.LAUNCHER
        assert blk["slurm_log_glob"] == de.DATAEFF_SLURM_LOG_GLOB
        # …and each block's per_run is scoped to its OWN cells, so the three trained
        # totals do not each carry all 27 runs.
        assert set(blk["per_run"]) == {
            de.run_name(rv, seed, total=total)
            for rv in de.RV_RATIOS
            for seed in de.SEEDS
        }
    assert att["per_total"][str(de.FULL_TOTAL)]["launcher"] == de.INHERITED_LAUNCHER
    assert (
        att["per_total"][str(de.LOW_TOTAL)]["slurm_log_glob"]
        != att["per_total"][str(de.FULL_TOTAL)]["slurm_log_glob"]
    )


def test_missing_logs_mean_unattested_not_v1(tmp_path: Path) -> None:
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    att = de.corpus_attestation(
        curve,
        log_dir=tmp_path / "no-such-dir",
        low_study_root=low,
        full_study_root=full,
    )
    assert att["healthy"] is False
    assert att["n_attested"] == 0
    assert att["n_unattested"] == de.TARGET_N_CELLS
    assert att["corpora"] == []
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), attestation=att)
    assert any("no Slurm-log corpus attestation" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_only_one_arm_attested_is_not_healthy(tmp_path: Path) -> None:
    """An unattested arm cannot be greened by the other arm's clean logs."""
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    logs = _write_logs(tmp_path / "slurm", low=low, full=full, totals=(de.FULL_TOTAL,))
    att = de.corpus_attestation(
        curve, log_dir=logs, low_study_root=low, full_study_root=full
    )
    assert att["per_total"][str(de.FULL_TOTAL)]["healthy"] is True
    assert att["per_total"][str(de.LOW_TOTAL)]["healthy"] is False
    assert att["healthy"] is False


def test_v1_data_dirs_are_a_hard_failure_even_when_labelled_v2(tmp_path: Path) -> None:
    """The ``corpus=`` token is a label; the dirs are the evidence."""
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    logs = _write_logs(
        tmp_path / "slurm",
        low=low,
        full=full,
        rv="data/datasets/rv",
        ms="data/datasets/ms",
    )
    att = de.corpus_attestation(
        curve, log_dir=logs, low_study_root=low, full_study_root=full
    )
    assert att["healthy"] is False
    assert all(blk["corpus_mismatches"] for blk in att["per_total"].values())
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), attestation=att)
    assert any("does not establish the expected v2 corpus" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_hf_data_source_is_a_hard_failure(tmp_path: Path) -> None:
    """``data.source=hf`` ignores the CSV dirs it still echoes, so it is unverified."""
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    logs = _write_logs(tmp_path / "slurm", low=low, full=full, data_source="hf")
    att = de.corpus_attestation(
        curve, log_dir=logs, low_study_root=low, full_study_root=full
    )
    assert att["healthy"] is False
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), attestation=att)
    assert de.hard_warnings(warnings)


def test_a_smoke_root_log_cannot_attest_this_study(tmp_path: Path) -> None:
    """Run names are reused verbatim by the ``_smoke`` roots."""
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    logs = tmp_path / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    task = 0
    for rv_ratio in de.RV_RATIOS:
        for seed in de.SEEDS:
            (logs / f"paclic_v2_data_efficiency_520_{task}.out").write_text(
                _log_text(
                    de.run_name(rv_ratio, seed, total=de.LOW_TOTAL),
                    prefix="paclic-v2-dataeff",
                    output_dir=f"./outputs/{low.name}_smoke",
                    task=task,
                ),
                encoding="utf-8",
            )
            task += 1
    att = de.corpus_attestation(
        curve, log_dir=logs, low_study_root=low, full_study_root=full
    )
    assert att["per_total"][str(de.LOW_TOTAL)]["study_root_mismatches"]
    assert att["healthy"] is False
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), attestation=att)
    assert any("different study root" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_two_arms_on_two_corpora_is_a_hard_failure(tmp_path: Path) -> None:
    """A 10%-vs-100% delta across two corpora is not a data-efficiency measurement."""
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    logs = tmp_path / "slurm"
    _write_logs(logs, low=low, full=full, totals=(de.FULL_TOTAL,))
    _write_logs(
        logs,
        low=low,
        full=full,
        totals=(de.LOW_TOTAL,),
        corpus="v1",
        rv="data/datasets/rv",
        ms="data/datasets/ms",
    )
    att = de.corpus_attestation(
        curve, log_dir=logs, low_study_root=low, full_study_root=full
    )
    assert att["single_corpus"] is False
    assert sorted(att["corpora"]) == ["v1", "v2"]
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), attestation=att)
    assert any("do not attest ONE corpus" in w for w in warnings)


# --------------------------------------------------------------------------- #
# 7. The inherited point is verified, not asserted.
# --------------------------------------------------------------------------- #
def test_inherited_point_matches_the_committed_maingrid_bundle(whole_curve) -> None:
    mg = whole_curve["aggregate"]["maingrid_consistency"]
    assert mg["checked"] is True
    assert mg["ok"] is True
    assert mg["mismatches"] == []
    assert mg["n_comparisons"] == len(de.RV_RATIOS) * len(de.CURVE_METRICS)
    assert mg["maingrid_study"] == v2mg.STUDY


def test_a_stale_maingrid_root_is_caught(whole_curve) -> None:
    """Perturb the published means: the join must refuse to call it the same result."""
    per_total = whole_curve["per_total"]
    published = _maingrid_aggregate(per_total)
    published["per_ratio"]["1"]["rv_cer"]["mean"] += 0.01
    mg = de.maingrid_consistency(per_total, published)
    assert mg["ok"] is False
    assert [(m["rv_ratio"], m["metric"]) for m in mg["mismatches"]] == [(1.0, "rv_cer")]
    warnings = de.audit_warnings(
        whole_curve["curve"],
        de.coverage_summary(whole_curve["curve"]),
        maingrid=mg,
    )
    assert any(
        "disagrees with the committed v2 main-grid bundle" in w for w in warnings
    )
    assert de.hard_warnings(warnings)


def test_a_different_n_on_the_published_side_is_caught(whole_curve) -> None:
    per_total = whole_curve["per_total"]
    published = _maingrid_aggregate(per_total)
    published["per_ratio"]["0.5"]["ms_cer"]["n"] = 2
    mg = de.maingrid_consistency(per_total, published)
    assert mg["ok"] is False
    assert any(m["metric"] == "ms_cer" for m in mg["mismatches"])


def test_no_maingrid_aggregate_means_unverified_not_ok(whole_curve) -> None:
    mg = de.maingrid_consistency(whole_curve["per_total"], None)
    assert mg["checked"] is False
    assert mg["ok"] is False
    warnings = de.audit_warnings(
        whole_curve["curve"],
        de.coverage_summary(whole_curve["curve"]),
        maingrid=mg,
    )
    assert any("inherited 100% point is unverified" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_load_maingrid_aggregate_tolerates_absent_and_corrupt(tmp_path: Path) -> None:
    assert de.load_maingrid_aggregate(None) is None
    assert de.load_maingrid_aggregate(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert de.load_maingrid_aggregate(bad) is None


# --------------------------------------------------------------------------- #
# 8. Per-cell invariants.
# --------------------------------------------------------------------------- #
def test_seed_dir_disagreement_is_hard(tmp_path: Path) -> None:
    name = de.run_name(1.0, 44, total=de.LOW_TOTAL)
    low, full = _write_roots(tmp_path, mutate={name: {"seed": 43}})
    curve, _ = de.load_curve(low, full)
    integrity = de.grid_integrity(curve, low_study_root=low, full_study_root=full)
    assert [m["run_name"] for m in integrity["seed_mismatches"]] == [name]
    # ...which also means the point no longer carries distinct seeds.
    assert integrity["point_seed_duplicates"]
    assert integrity["ok"] is False
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), integrity=integrity)
    assert any("disagrees with its run dir" in w for w in warnings)
    assert any("do not carry distinct seeds" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_torn_results_json_counts_as_absent_and_is_reported(tmp_path: Path) -> None:
    name = de.run_name(0.0, 43, total=de.LOW_TOTAL)
    low, full = _write_roots(tmp_path, corrupt=frozenset({name}))
    curve, _ = de.load_curve(low, full)
    assert not next(c for c in curve[de.LOW_TOTAL] if c.run_name == name).present
    integrity = de.grid_integrity(curve, low_study_root=low, full_study_root=full)
    assert integrity["unreadable_results_json"] == [name]
    assert integrity["unreadable_check_performed"] is True
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), integrity=integrity)
    assert any("did not parse" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_integrity_without_roots_reports_not_performed(tmp_path: Path) -> None:
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    integrity = de.grid_integrity(curve)
    assert integrity["unreadable_check_performed"] is False
    assert integrity["ok"] is True


# --------------------------------------------------------------------------- #
# 9. Bundle.
# --------------------------------------------------------------------------- #
def test_build_bundle_writes_every_artifact(whole_curve, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    aggregate = de.build_bundle(
        whole_curve["curve"],
        whole_curve["checks"],
        out,
        low_study_root=whole_curve["low"],
        full_study_root=whole_curve["full"],
        subset=whole_curve["aggregate"]["subset_consistency"],
        attestation=whole_curve["aggregate"]["corpus_attestation"],
        maingrid=whole_curve["aggregate"]["maingrid_consistency"],
    )
    expected = [
        "results.md",
        "aggregate.json",
        "manifest.json",
        "tables/curve.csv",
        "tables/curve_deltas.csv",
        "tables/step_separability.csv",
        "tables/provenance.csv",
    ]
    # One per_seed + asymmetry file per POINT, four of each now.
    expected += [f"tables/per_seed_n{total}.csv" for total in de.TOTALS]
    expected += [f"tables/asymmetry_n{total}.csv" for total in de.TOTALS]
    for name in expected:
        assert (out / name).is_file(), name
    assert aggregate["complete"] is True
    assert aggregate["publishable"] is True
    assert aggregate["hard_warnings"] == []
    # per_seed tables are agg's renderer, verbatim, one point each. Read with
    # newline="" so csv's \r\n line endings are compared, not normalised away.
    for total in de.TOTALS:
        assert (out / f"tables/per_seed_n{total}.csv").read_text(
            encoding="utf-8", newline=""
        ) == agg.render_per_seed_csv(whole_curve["curve"][total])


def test_build_bundle_computes_integrity_when_not_given(
    whole_curve, tmp_path: Path
) -> None:
    """A caller must not be able to get a bundle whose invariants were skipped."""
    out = tmp_path / "bundle2"
    aggregate = de.build_bundle(
        whole_curve["curve"],
        whole_curve["checks"],
        out,
        low_study_root=whole_curve["low"],
        full_study_root=whole_curve["full"],
    )
    assert aggregate["grid_integrity"]["unreadable_check_performed"] is True


def test_aggregate_curves_payload_matches_the_figure_contract(whole_curve) -> None:
    """``curves[<ratio>][<metric>]`` is exactly what ``fig_data_efficiency`` reads."""
    curves = whole_curve["aggregate"]["curves"]
    assert sorted(curves) == sorted(f"{r:g}" for r in de.RV_RATIOS)
    for rv_ratio in de.RV_RATIOS:
        for metric in de.CURVE_METRICS:
            series = curves[f"{rv_ratio:g}"][metric]
            assert [p["total_train_samples"] for p in series] == list(de.TOTALS)
            assert [p["pct"] for p in series] == [de.pct_of(t) for t in de.TOTALS]
            assert [p["source"] for p in series] == [
                de.CELL_SOURCE_BY_TOTAL[t] for t in de.TOTALS
            ]
            for point in series:
                assert point["mean"] is not None
                assert point["ci95_halfwidth"] is not None


def test_manifest_carries_the_route_justification_and_no_v1_roots(
    whole_curve, tmp_path: Path
) -> None:
    manifest = de.build_manifest(
        whole_curve["curve"],
        whole_curve["checks"],
        de.coverage_summary(whole_curve["curve"]),
        low_study_root=whole_curve["low"],
        full_study_root=whole_curve["full"],
        shared_low_study_root=de.DEFAULT_SHARED_LOW_STUDY_ROOT,
        shared_full_study_root=de.DEFAULT_SHARED_FULL_STUDY_ROOT,
    )
    blob = json.dumps(manifest)
    assert manifest["issue"] == 126
    assert "why_not_the_neighbouring_builders" in manifest
    assert manifest["roots"]["inherited"]["reused_by_reference"] is True
    assert manifest["roots"]["trained"]["totals"] == list(de.TRAINED_TOTALS)
    assert manifest["design"]["n_newly_trained"] == de.TARGET_N_NEW_CELLS
    assert "read_cell_dir" in blob and "summarize" in blob
    # No v1 study root leaks into a path or a root anywhere.
    assert "experiments_preadapt_freeze_factorial" not in blob
    assert "experiments_paclic_accent_ratio_data_efficiency" not in blob
    # v1's provenance stamps are NAMED only inside the route justification (which has
    # to quote them) and never stamped on a cell or a root.
    assert "prior-study-reference" in manifest["why_not_the_neighbouring_builders"]
    cells_blob = json.dumps(manifest["cells"]) + json.dumps(manifest["roots"])
    assert "prior-study-reference" not in cells_blob
    assert "multiseed" not in cells_blob
    assert len(manifest["cells"]) == de.TARGET_N_CELLS
    assert {c["source"] for c in manifest["cells"]} == set(
        de.CELL_SOURCE_BY_TOTAL.values()
    )


def test_manifest_states_the_unmatched_subset_limitation(whole_curve) -> None:
    """The honest reading of an unpinned subset seed is written down, not smoothed."""
    manifest = de.build_manifest(
        whole_curve["curve"],
        whole_curve["checks"],
        de.coverage_summary(whole_curve["curve"]),
        low_study_root=whole_curve["low"],
        full_study_root=whole_curve["full"],
        shared_low_study_root=None,
        shared_full_study_root=None,
    )
    limitations = " ".join(manifest["limitations"])
    assert "not subset-matched" in limitations
    assert "Four points do NOT establish a curve shape at n=3" in limitations
    assert de.SHAPE_NOT_DETERMINED in limitations
    assert "p=0.25" in limitations
    # The nesting the steps rest on is stated as verified, not as assumed.
    assert "verified" in limitations and "nesting" in limitations


# --------------------------------------------------------------------------- #
# 10. Figure gates.
# --------------------------------------------------------------------------- #
def test_figures_refused_on_a_partial_curve(tmp_path: Path) -> None:
    low, full = _write_roots(
        tmp_path, skip=frozenset({de.run_name(0.0, 44, total=de.LOW_TOTAL)})
    )
    curve, checks = de.load_curve(low, full)
    aggregate = de.build_aggregate(
        curve, de.aggregate_curve(curve), checks, de.coverage_summary(curve)
    )
    with pytest.raises(de.IncompleteGridError, match="partial curve"):
        de.build_figures(aggregate, tmp_path / "figs")
    assert not (tmp_path / "figs").exists()


def test_figures_refused_without_an_attestation(whole_curve, tmp_path: Path) -> None:
    aggregate = dict(whole_curve["aggregate"], corpus_attestation=None)
    with pytest.raises(de.UnverifiedCorpusError, match="without a corpus attestation"):
        de.build_figures(aggregate, tmp_path / "figs")


def test_figures_refused_on_an_unhealthy_attestation(
    whole_curve, tmp_path: Path
) -> None:
    att = json.loads(json.dumps(whole_curve["aggregate"]["corpus_attestation"]))
    att["healthy"] = False
    att["per_total"][str(de.LOW_TOTAL)]["healthy"] = False
    aggregate = dict(whole_curve["aggregate"], corpus_attestation=att)
    with pytest.raises(de.UnverifiedCorpusError, match="unhealthy corpus"):
        de.build_figures(aggregate, tmp_path / "figs")


def test_figures_refused_when_the_inherited_point_is_unverified(
    whole_curve, tmp_path: Path
) -> None:
    aggregate = dict(
        whole_curve["aggregate"],
        maingrid_consistency={"checked": False, "ok": False, "reason": "none read"},
    )
    with pytest.raises(de.GridInvariantError, match="inherited"):
        de.build_figures(aggregate, tmp_path / "figs")


def test_figures_refused_on_a_hard_warning(whole_curve, tmp_path: Path) -> None:
    aggregate = dict(
        whole_curve["aggregate"],
        warnings=["cell(s) whose results.json seed disagrees with its run dir: x"],
    )
    with pytest.raises(de.GridInvariantError, match="publication-blocking"):
        de.build_figures(aggregate, tmp_path / "figs")


def test_figures_refused_in_another_study_figure_dir(
    whole_curve, tmp_path: Path
) -> None:
    """The committed main-grid figure dir must not be overwritten by this study."""
    other = tmp_path / "figures"
    other.mkdir()
    (other / "manifest.json").write_text(
        json.dumps({"study": v2mg.STUDY}), encoding="utf-8"
    )
    with pytest.raises(de.FigureDirConflictError, match=v2mg.STUDY):
        de.build_figures(whole_curve["aggregate"], other)
    # Its manifest is untouched.
    assert json.loads((other / "manifest.json").read_text(encoding="utf-8")) == {
        "study": v2mg.STUDY
    }


def test_figure_dir_conflict_allows_this_study_own_dir(
    whole_curve, tmp_path: Path
) -> None:
    own = tmp_path / "figs"
    own.mkdir()
    (own / "manifest.json").write_text(
        json.dumps({"study": de.STUDY}), encoding="utf-8"
    )
    assert de._figure_dir_conflict(own) is None


def test_complete_curve_renders_the_figure(whole_curve, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "figs"
    figures = de.build_figures(
        whole_curve["aggregate"], out, bundle_dir="docs/x", make_pdf=False, dpi=100
    )
    assert [f["stem"] for f in figures] == list(de.V2_FIGURE_STEMS)
    assert (out / "fig6_data_efficiency.png").is_file()
    assert (out / "manifest.json").is_file()
    assert (out / "README.md").is_file()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["study"] == de.STUDY
    assert manifest["corpus_attestation"]["healthy"] is True
    assert manifest["maingrid_consistency"]["ok"] is True
    # The caption is the v2 one, and no v1 conclusion is carried.
    shows = manifest["figures"][0]["shows"]
    assert "Four points" in shows
    assert "guide to the eye" in shows  # the line is not a fitted shape
    assert "base letters transfer across notation" not in shows
    # Provenance points at the bundle passed in, not at the v1 dir.
    from scripts.evaluation import build_paclic_paper_figures as bpf

    assert all(
        bpf.DATA_EFFICIENCY_DIR not in s["file"]
        for s in manifest["figures"][0]["sources"]
    )
    readme = (out / "README.md").read_text(encoding="utf-8")
    # The regenerate command names the dir actually written, not a module default.
    assert str(out) in readme
    assert "guide to the eye, not a fitted shape" in readme
    assert de.SHAPE_NOT_DETERMINED in readme


def test_v2_figure_suptitle_is_not_the_v1_conclusion() -> None:
    import json
    from pathlib import Path

    from scripts.evaluation import build_paclic_paper_figures as bpf
    from scripts.evaluation import build_paclic_v2_maingrid_analysis as v2mg

    assert de.V2_FIGURE_SUPTITLE != bpf.DATA_EFFICIENCY_SUPTITLE
    assert "accent marks do not" in bpf.DATA_EFFICIENCY_SUPTITLE
    assert "accent marks do not" not in de.V2_FIGURE_SUPTITLE
    # The suptitle used to have to contain "v2". That was a proxy for "the figure
    # bundle says which corpus it is", and it cost the width the print type size
    # needs — with it, the suptitle overflowed both edges of the figure. Assert what
    # the proxy stood for instead, and assert it where it is actually checkable: the
    # committed figure manifest records the corpus, and records it per cell from the
    # Slurm headers rather than as a label the builder asserts about itself.
    manifest_path = Path(
        "outputs/figures/data-efficiency/manifest.json"
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["corpus"].startswith("v2")
        attestation = manifest["corpus_attestation"]
        assert attestation["corpora"] == ["v2"]
        assert attestation["expected_corpus"]["corpus"] == v2mg.EXPECTED_CORPUS
        per_run = [
            run
            for block in attestation["per_total"].values()
            for run in block["per_run"].values()
        ]
        assert per_run, "no per-cell attestation recorded"
        assert {run["corpus"] for run in per_run} == {"v2"}


def test_figure_function_defaults_reproduce_the_v1_ratio_set() -> None:
    """The parameterisation must not change the v1 figure."""
    import inspect

    from scripts.evaluation import build_paclic_paper_figures as bpf

    params = inspect.signature(bpf.fig_data_efficiency).parameters
    assert params["rv_ratios"].default == agg.RV_RATIOS
    assert params["suptitle"].default == bpf.DATA_EFFICIENCY_SUPTITLE


# --------------------------------------------------------------------------- #
# 11. CLI.
# --------------------------------------------------------------------------- #
def test_cli_exits_nonzero_on_a_partial_curve(tmp_path: Path, capsys) -> None:
    low, full = _write_roots(
        tmp_path, skip=frozenset({de.run_name(0.0, 44, total=de.LOW_TOTAL)})
    )
    out = tmp_path / "bundle"
    code = de.main(
        [
            "--low-study-root",
            str(low),
            "--full-study-root",
            str(full),
            "--out-dir",
            str(out),
            "--no-cross-check",
            "--no-slurm-attestation",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "PARTIAL" in err
    assert (out / "aggregate.json").is_file()
    payload = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PARTIAL"
    assert payload["publishable"] is False


def test_cli_exits_zero_on_a_whole_verified_curve(tmp_path: Path) -> None:
    low, full = _write_roots(tmp_path)
    curve, _ = de.load_curve(low, full)
    per_total = de.aggregate_curve(curve)
    mg_path = tmp_path / "maingrid_aggregate.json"
    mg_path.write_text(json.dumps(_maingrid_aggregate(per_total)), encoding="utf-8")
    logs = _write_logs(tmp_path / "slurm", low=low, full=full)
    out = tmp_path / "bundle"
    code = de.main(
        [
            "--low-study-root",
            str(low),
            "--full-study-root",
            str(full),
            "--slurm-log-dir",
            str(logs),
            "--maingrid-aggregate",
            str(mg_path),
            "--out-dir",
            str(out),
            "--no-cross-check",
        ]
    )
    assert code == 0
    payload = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    assert payload["publishable"] is True


def test_cli_refuses_figures_without_a_bundle(tmp_path: Path, capsys) -> None:
    code = de.main(["--figures-out-dir", str(tmp_path / "figs")])
    assert code == 2
    assert "requires --out-dir" in capsys.readouterr().err
    assert not (tmp_path / "figs").exists()


def test_cli_writes_no_figures_on_a_hard_failure(tmp_path: Path) -> None:
    low, full = _write_roots(
        tmp_path, skip=frozenset({de.run_name(0.0, 44, total=de.LOW_TOTAL)})
    )
    code = de.main(
        [
            "--low-study-root",
            str(low),
            "--full-study-root",
            str(full),
            "--out-dir",
            str(tmp_path / "bundle"),
            "--figures-out-dir",
            str(tmp_path / "figs"),
            "--no-cross-check",
            "--no-slurm-attestation",
        ]
    )
    assert code == 1
    assert not (tmp_path / "figs").exists()


def test_cli_prints_the_results_page_without_out_dir(tmp_path: Path, capsys) -> None:
    low, full = _write_roots(tmp_path)
    code = de.main(
        [
            "--low-study-root",
            str(low),
            "--full-study-root",
            str(full),
            "--no-cross-check",
            "--no-slurm-attestation",
        ]
    )
    out = capsys.readouterr().out
    assert "# PACLIC v2 data efficiency" in out
    assert "4-point curve" in out
    assert "Is the curve's SHAPE determined? No" in out
    # No attestation and no maingrid check ⇒ hard failure even though 36/36 present.
    assert code == 1


# --------------------------------------------------------------------------- #
# 12. Step separability and the shape verdict — what four points may be claimed
#     to show, which is not a shape.
# --------------------------------------------------------------------------- #
def test_step_separability_covers_every_step_and_every_pair_of_steps(
    whole_curve,
) -> None:
    rows = whole_curve["aggregate"]["step_separability"]
    n_metrics, n_ratios = len(de.CURVE_METRICS), len(de.RV_RATIOS)
    zero = [r for r in rows if r["comparison_kind"] == "step_vs_zero"]
    pairs = [r for r in rows if r["comparison_kind"] == "step_vs_step"]
    assert len(zero) == n_ratios * n_metrics * len(de.CONSECUTIVE_STEPS)
    # 3 steps ⇒ 3 unordered pairs of steps.
    assert len(pairs) == n_ratios * n_metrics * 3
    assert len(rows) == len(zero) + len(pairs)
    # Nothing is ever separable at n=3 — by construction, not by luck.
    assert all(r["separable"] is False for r in rows)
    assert {r["exact_sign_test_min_p_two_sided"] for r in rows} == {0.25}
    assert all(
        r["exact_sign_test_p_two_sided"] >= r["exact_sign_test_min_p_two_sided"]
        for r in rows
    )


def test_a_step_vs_step_row_is_the_second_difference_paired_by_seed(
    whole_curve,
) -> None:
    """The step comparison pairs the per-seed STEP magnitudes, not the two means."""
    curve = whole_curve["curve"]
    rows = whole_curve["aggregate"]["step_separability"]
    row = next(
        r
        for r in rows
        if r["comparison_kind"] == "step_vs_step"
        and r["rv_ratio"] == 0.5
        and r["metric"] == "avg_cer"
        and r["step"] == "1042->2605"
        and r["other_step"] == "2605->5210"
    )
    first = de._step_seed_values(curve, 1042, 2605, 0.5, "avg_cer")
    second = de._step_seed_values(curve, 2605, 5210, 0.5, "avg_cer")
    reference = agg._contrast(first, second)
    assert row["diff_mean"] == reference["diff"]["mean"]
    assert row["per_seed_diff"] == reference["per_seed_diff"]
    assert row["n"] == len(de.SEEDS)


def test_the_shape_is_not_determined_and_the_ranking_travels_with_that(
    whole_curve,
) -> None:
    """A ranking exists; ``determined`` is False everywhere; both are in one block."""
    shape = whole_curve["aggregate"]["shape"]
    assert shape["determined"] is False
    assert shape["verdict"] == de.SHAPE_NOT_DETERMINED
    assert shape["primary_metric"] == de.SHAPE_PRIMARY_METRIC
    assert "2/2**3 = 0.25" in shape["separability_note"]
    assert "front-loaded" in shape["claim_note"].lower()
    for blk in shape["by_ratio"].values():
        assert blk["determined"] is False
        assert blk["n_step_pairs_separable"] == 0
        assert blk["n_step_pairs"] == 3
        # The ranking is present AND the ordering evidence sits beside it.
        assert len(blk["ranking_largest_step_first"]) == len(de.CONSECUTIVE_STEPS)
        assert len(blk["ranking"]) == len(de.CONSECUTIVE_STEPS)
        assert (
            len(blk["step_pairs_consistently_ordered"])
            + len(blk["step_pairs_order_unstable"])
            + len(blk["step_pairs_not_computable"])
            == blk["n_step_pairs"]
        )
        assert "NOT determined" in blk["statement"]


def test_the_front_loaded_fixture_ranks_but_is_still_not_separable(
    whole_curve,
) -> None:
    """The 50:50 mix falls in −0.04 / −0.02 / −0.01 steps: a clean, ordered ranking…

    …which is exactly the case where a shape claim is tempting. All three seeds agree
    on the order, so every pair is ``consistently_ordered``; none is separable, and the
    verdict is still ``shape_not_determined_at_n3``.
    """
    blk = whole_curve["aggregate"]["shape"]["by_ratio"]["0.5"]
    assert blk["ranking_largest_step_first"] == ["10→25%", "25→50%", "50→100%"]
    assert len(blk["step_pairs_consistently_ordered"]) == 3
    assert blk["step_pairs_order_unstable"] == []
    assert blk["determined"] is False


def test_a_missing_cell_makes_a_step_pair_not_computable_not_unstable(
    tmp_path: Path,
) -> None:
    """Mid-flight, an absent cell must not be printed as a measured disagreement."""
    low, full = _write_roots(
        tmp_path,
        skip=frozenset(de.run_name(0.5, seed, total=5210) for seed in de.SEEDS),
    )
    curve, _ = de.load_curve(low, full)
    rows = de.step_separability(curve)
    affected = [
        r
        for r in rows
        if r["rv_ratio"] == 0.5
        and r["metric"] == "avg_cer"
        and r["comparison_kind"] == "step_vs_step"
        and "2605->5210" in (r["step"], r["other_step"])
    ]
    assert affected
    assert all(r["verdict"] == de.NOT_COMPUTABLE for r in affected)
    assert all(r["n"] == 0 for r in affected)
    assert de.NOT_COMPUTABLE != de.ORDER_UNSTABLE
    # …and the sign/direction columns are BLANK rather than agg._contrast's
    # "mixed"/False on an empty difference set, which would read as a disagreement.
    assert all(r["direction"] is None for r in affected)
    assert all(r["sign_consistent"] is None for r in affected)
    csv_text = de.render_step_separability_csv(affected)
    header = csv_text.splitlines()[0].split(",")
    for line in csv_text.strip().splitlines()[1:]:
        cells = line.split(",")
        assert cells[header.index("sign_consistent")] == ""
        assert cells[header.index("direction")] == ""
    shape = de.shape_verdict(curve, rows)
    blk = shape["by_ratio"]["0.5"]
    # Both steps that read 5210 are gone, so every PAIR of steps is uncomputable —
    # and none of them is reported as an order flip.
    assert len(blk["step_pairs_not_computable"]) == 3
    assert blk["step_pairs_order_unstable"] == []
    assert blk["step_pairs_consistently_ordered"] == []
    assert blk["n_step_pairs_comparable"] == 0
    # The ranking shrinks to the one step that exists, and says so.
    assert len(blk["ranking_largest_step_first"]) == 1
    assert "1/3 steps computable" in blk["statement"]
    assert "not computable yet" in blk["statement"]


def test_step_separability_csv_carries_the_verdict_next_to_the_numbers(
    whole_curve,
) -> None:
    rows = whole_curve["aggregate"]["step_separability"]
    text = de.render_step_separability_csv(rows)
    header = text.splitlines()[0].split(",")
    assert header == list(de.STEP_SEPARABILITY_COLUMNS)
    assert len(text.strip().splitlines()) == 1 + len(rows)
    # `separable` is false on every row, and never blank.
    body = text.strip().splitlines()[1:]
    assert all(row.split(",")[header.index("separable")] == "false" for row in body)
    assert de.ORDER_CONSISTENT in text or de.ORDER_UNSTABLE in text


# --------------------------------------------------------------------------- #
# 13. Within-seed subset nesting — what makes a STEP readable.
# --------------------------------------------------------------------------- #
def test_nested_prefixes_are_verified_not_assumed(whole_curve) -> None:
    nesting = whole_curve["aggregate"]["subset_consistency"]["nesting"]
    assert nesting["nested"] is True
    assert nesting["n_violations"] == 0
    assert nesting["n_unchecked"] == 0
    # One row per (seed, consecutive pair of totals).
    assert nesting["n_pairs"] == len(de.SEEDS) * (len(de.TOTALS) - 1)
    assert nesting["n_checked"] == nesting["n_pairs"]
    assert all(r["nested"] for r in nesting["pairs"])


def test_a_non_nested_step_is_a_hard_warning(tmp_path: Path) -> None:
    """If 2605's keys are not a superset-prefix of 1042's, the step is not 'more data'."""
    low, full = _write_roots(tmp_path)
    victim = low / de.run_name(1.0, 42, total=2605) / de.PARALLEL_KEYS_FILE
    victim.write_text("some-other-key\n" * 3, encoding="utf-8")
    curve, _ = de.load_curve(low, full)
    nesting = de.subset_nesting(curve, low_study_root=low, full_study_root=full)
    assert nesting["nested"] is False
    # One corrupted list breaks BOTH rungs it participates in, for that seed only.
    assert nesting["n_violations"] == 2
    assert nesting["violations"] == [
        "seed 42: 1042 ⊄ 2605 "
        f"({de.run_name(1.0, 42, total=1042)} vs {de.run_name(1.0, 42, total=2605)})",
        "seed 42: 2605 ⊄ 5210 "
        f"({de.run_name(1.0, 42, total=2605)} vs {de.run_name(1.0, 42, total=5210)})",
    ]
    assert all(r["nested"] for r in nesting["pairs"] if r["seed"] != 42)
    subset = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
        nesting=nesting,
    )
    assert subset["consistent"] is False
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), subset=subset)
    assert any("are not nested" in w for w in warnings)
    assert de.hard_warnings(warnings)


def test_missing_key_lists_are_unchecked_never_nested(tmp_path: Path) -> None:
    """Not looking is reported as not looking — and skipping the check is too."""
    low, full = _write_roots(tmp_path, no_key_lists=True)
    curve, _ = de.load_curve(low, full)
    nesting = de.subset_nesting(curve, low_study_root=low, full_study_root=full)
    assert nesting["n_checked"] == 0
    assert nesting["n_unchecked"] == nesting["n_pairs"]
    assert all(r["nested"] is None for r in nesting["pairs"])
    # No violation ⇒ not publication-blocking, but it IS reported.
    assert nesting["nested"] is True
    subset = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
        nesting=nesting,
    )
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), subset=subset)
    assert any("could not be checked" in w for w in warnings)
    assert not [w for w in de.hard_warnings(warnings) if "nest" in w]

    # And a bundle that never ran the check at all says so rather than passing.
    skipped = de.subset_consistency(
        de.subset_provenance_by_total(curve, low_study_root=low, full_study_root=full),
        curve,
    )
    assert skipped["nesting"] is None
    warnings = de.audit_warnings(curve, de.coverage_summary(curve), subset=skipped)
    assert any("was NOT checked" in w for w in warnings)


# --------------------------------------------------------------------------- #
# 14. The main-grid builder is unchanged by all of this.
# --------------------------------------------------------------------------- #
def test_maingrid_builder_still_reads_its_own_grid(tmp_path: Path) -> None:
    """A 15-cell v2 main grid still aggregates COMPLETE through ``v2mg`` alone."""
    root = tmp_path / "outputs" / "experiments_paclic_v2_maingrid"
    for rv_ratio in v2mg.RV_RATIOS:
        for seed in v2mg.SEEDS:
            cell = root / v2mg.run_name(rv_ratio, seed)
            cell.mkdir(parents=True, exist_ok=True)
            spec = _SPEC.get((de.FULL_TOTAL, rv_ratio), (0.05, 0.05, 0.9, 0.9))
            (cell / "results.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "model_base": "fixed-foundation",
                        "freeze_mode": "none",
                        "input_scheme": "iso",
                        "num_train_epochs": 20,
                        "total_train_samples_requested": de.FULL_TOTAL,
                        "parallel_keys_total": de.FULL_TOTAL,
                        "parallel_keys_used": de.FULL_TOTAL,
                        "mix_mode": "parallel_by_text",
                        "key_column": "text",
                        "selection_algorithm": "sorted_common_keys+seeded_shuffle+prefix",
                        "key_list_sha256": key_list_sha(de.FULL_TOTAL, seed),
                        "rv_train_samples": v2mg.expected_mix_counts(
                            de.FULL_TOTAL, rv_ratio
                        )[0],
                        "ms_train_samples": v2mg.expected_mix_counts(
                            de.FULL_TOTAL, rv_ratio
                        )[1],
                        "rv_test": _view(spec[0], spec[2]),
                        "ms_test": _view(spec[1], spec[3]),
                        "avg_cer": (spec[0] + spec[1]) / 2,
                    }
                ),
                encoding="utf-8",
            )
    cells, _ = v2mg.load_grid(root, cross_check=False)
    coverage = v2mg.coverage_summary(cells)
    assert coverage["complete"] is True
    assert coverage["n_present"] == v2mg.TARGET_N_CELLS
    assert v2mg.partial_banner(coverage) is None
