"""Test paclic v2 effect sizes and plateau: PACLIC 2026 v2 release."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Optional

import pytest

from scripts.evaluation import aggregate_accent_ratio_multiseed as agg
from scripts.evaluation import build_paclic_v2_effect_sizes_and_plateau as es
from scripts.evaluation import checkpoint_selection_sensitivity_v2 as css2

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_MAINGRID = (
    REPO_ROOT
    / "results"
    / "paclic-v2-maingrid"
)
COMMITTED_BUNDLE = (
    REPO_ROOT / "results" / es.STUDY
)


# --------------------------------------------------------------------------- #
# Synthetic published bundles.
# --------------------------------------------------------------------------- #
#: ``rv_ratio → seed → (rv_cer, ms_cer, rv_accent_f1, ms_accent_f1)``, chosen so that
#: every convention under test has a hand-checkable answer:
#:
#: * ``zero_shot_cer`` per-seed diffs are 0.10 / 0.20 / 0.30 → mean 0.20, sample SD
#:   0.10, ``dz`` exactly 2.0, and the MS→RV arm mean is exactly 0.40 → relative
#:   effect exactly 0.5;
#: * ``rv_retention_cer`` diffs are all +0.010 → zero spread → ``dz`` undefined while
#:   the effect is perfectly sign-consistent (the degenerate case a reader is most
#:   likely to see as "infinitely large");
#: * worst-view CER ties exactly between 75:25 and 50:50 (both ``{.048,.049,.050}``)
#:   and is broken by mean-view CER (.045 vs .040), so the tiebreak is exercised;
#: * 25:75 is worse than 50:50 on worst-view CER in all three seeds, with a paired CI
#:   that excludes zero **and** non-overlapping marginal CIs — and still must not be
#:   reported as separable.
SPEC: dict[float, dict[int, tuple[float, float, float, float]]] = {
    1.0: {
        42: (0.030, 0.200, 0.95, 0.10),
        43: (0.031, 0.200, 0.95, 0.11),
        44: (0.032, 0.200, 0.95, 0.12),
    },
    0.75: {
        42: (0.040, 0.050, 0.94, 0.90),
        43: (0.041, 0.049, 0.94, 0.91),
        44: (0.042, 0.048, 0.94, 0.92),
    },
    0.5: {
        42: (0.049, 0.030, 0.93, 0.93),
        43: (0.050, 0.031, 0.93, 0.93),
        44: (0.048, 0.032, 0.93, 0.93),
    },
    0.25: {
        42: (0.060, 0.020, 0.90, 0.94),
        43: (0.061, 0.019, 0.90, 0.94),
        44: (0.062, 0.018, 0.90, 0.94),
    },
    0.0: {
        42: (0.300, 0.030, 0.05, 0.95),
        43: (0.400, 0.031, 0.06, 0.95),
        44: (0.500, 0.032, 0.07, 0.95),
    },
}


def _view(cer: float, f1: float) -> dict[str, Optional[float]]:
    """One test view's metric block, with every key the real cell reader fills."""
    return {
        "cer": cer,
        "wer": cer * 3,
        "chrf": 100.0 * f1,
        "exact_match": 0.1,
        "exact_match_count": 58,
        "full_cer": cer,
        "accent_stripped_cer": cer / 2,
        "accent_precision": f1,
        "accent_recall": f1,
        "accent_f1": f1,
    }


def _cells(
    spec: dict[float, dict[int, tuple[float, float, float, float]]] = SPEC,
) -> list[agg.SeedCell]:
    cells: list[agg.SeedCell] = []
    for rv_ratio in es.RV_RATIOS:
        for seed, (rv_cer, ms_cer, rv_f1, ms_f1) in sorted(spec[rv_ratio].items()):
            rv_pct = int(round(rv_ratio * 100))
            cells.append(
                agg.SeedCell(
                    rv_ratio=rv_ratio,
                    seed=seed,
                    run_name=f"rv_{rv_pct}_ms_{100 - rv_pct}_seed_{seed}",
                    source="v2-maingrid",
                    present=True,
                    results_path=f"outputs/x/rv_{rv_pct}_seed_{seed}/results.json",
                    rv=_view(rv_cer, rv_f1),
                    ms=_view(ms_cer, ms_f1),
                    avg_cer=(rv_cer + ms_cer) / 2,
                    model_base="models/preadapt",
                )
            )
    return cells


def _m(cell: agg.SeedCell, view: str, metric: str) -> float:
    """One view metric of a fixture cell, asserted present (the fixture fills all)."""
    value = (cell.rv if view == "rv" else cell.ms)[metric]
    assert value is not None
    return float(value)


def _scorable_csv(cells: list[agg.SeedCell]) -> str:
    """A ``scorable_per_ratio.csv`` whose ``current`` rows agree with ``cells``.

    Written through ``css2.SCORABLE_COLUMNS`` so the column contract comes from the
    producing module. A decoy ``worst_val_cer`` criterion carries deliberately wrong
    values: if the reader ever stopped filtering on ``criterion``, the worst-view
    reconciliation would fail loudly instead of silently mixing two rules.
    """
    rows: list[dict[str, Any]] = []
    for criterion, offset in ((es.SELECTION_CRITERION, 0.0), ("worst_val_cer", 0.5)):
        for rv_ratio in es.RV_RATIOS:
            group = sorted(
                (c for c in cells if c.rv_ratio == rv_ratio), key=lambda c: c.seed
            )
            for metric, values in (
                (
                    "avg_cer",
                    [(_m(c, "rv", "cer") + _m(c, "ms", "cer")) / 2 for c in group],
                ),
                (
                    "worst_view_cer",
                    [max(_m(c, "rv", "cer"), _m(c, "ms", "cer")) for c in group],
                ),
                (
                    "worst_accent_f1",
                    [
                        min(_m(c, "rv", "accent_f1"), _m(c, "ms", "accent_f1"))
                        for c in group
                    ],
                ),
            ):
                stat = agg.summarize(v + offset for v in values)
                rows.append(
                    {
                        "criterion": criterion,
                        "rv_ratio": rv_ratio,
                        "ms_ratio": round(1.0 - rv_ratio, 2),
                        "metric": metric,
                        "n": stat.n,
                        "seeds": ";".join(str(c.seed) for c in group),
                        "mean": stat.mean,
                        "sd": stat.sd,
                        "sem": stat.sem,
                        "ci95_lo": stat.ci95_lo,
                        "ci95_hi": stat.ci95_hi,
                        "ci95_halfwidth": stat.ci95_halfwidth,
                        "values": ";".join(f"{v:.6f}" for v in stat.values),
                    }
                )
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(css2.SCORABLE_COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_inputs(
    root: Path,
    *,
    spec: dict[float, dict[int, tuple[float, float, float, float]]] = SPEC,
    drop_cells: tuple[tuple[float, int], ...] = (),
    corpus: str = "v2",
    publishable: bool = True,
) -> tuple[Path, Path]:
    """Write a synthetic main-grid + selection bundle pair; return their dirs."""
    cells = [c for c in _cells(spec) if (c.rv_ratio, c.seed) not in set(drop_cells)]
    maingrid = root / "paclic-v2-maingrid"
    selection = root / "paclic-v2-checkpoint-selection-sensitivity"
    _write(maingrid / "tables" / "per_seed.csv", agg.render_per_seed_csv(cells))
    _write(
        maingrid / "tables" / "per_ratio.csv",
        agg.render_per_ratio_csv(agg.aggregate_by_ratio(cells)),
    )
    _write(
        maingrid / "tables" / "asymmetry.csv",
        agg.render_asymmetry_csv(agg.asymmetry(cells)),
    )
    _write(
        maingrid / "tables" / "retention_adaptation.csv",
        agg.render_retention_csv(agg.retention_adaptation(cells)),
    )
    _write(
        maingrid / "aggregate.json",
        json.dumps(
            {
                "study": "paclic-v2-maingrid",
                "status": "COMPLETE",
                "publishable": publishable,
                "corpus": corpus,
                "corpus_attestation": {
                    "healthy": True,
                    "corpora": [corpus],
                    "n_attested": len(cells),
                },
            }
        ),
    )
    _write(
        selection / "tables" / "scorable_per_ratio.csv",
        _scorable_csv(
            [c for c in _cells(spec) if (c.rv_ratio, c.seed) not in set(drop_cells)]
        ),
    )
    _write(
        selection / "aggregate.json",
        json.dumps(
            {
                "study": "paclic-v2-checkpoint-selection-sensitivity",
                "corpus": corpus,
                "n_cells": 15,
                "grid": {"rv_ratios": list(es.RV_RATIOS), "seeds": list(es.SEEDS)},
            }
        ),
    )
    return maingrid, selection


@pytest.fixture
def bundle(tmp_path: Path) -> dict[str, Any]:
    """The aggregate payload over a healthy synthetic input pair."""
    maingrid, selection = build_inputs(tmp_path)
    _cells_read, aggregate = es.analyze(maingrid, selection)
    return aggregate


def _effect(aggregate: dict[str, Any], comparison: str) -> dict[str, Any]:
    for row in aggregate["endpoint_effect_sizes"]:
        if row["comparison"] == comparison:
            return row
    raise AssertionError(f"no effect-size row for {comparison!r}")


def _separability(
    aggregate: dict[str, Any], pair: str, quantity: str = es.RANK_PRIMARY
) -> dict[str, Any]:
    for row in aggregate["plateau"]["separability"]:
        if row["pair"] == pair and row["quantity"] == quantity:
            return row
    raise AssertionError(f"no separability row for {pair!r}/{quantity!r}")


# --------------------------------------------------------------------------- #
# 1. The dz convention, pinned against hand computation.
# --------------------------------------------------------------------------- #
def test_cohens_dz_is_the_mean_over_the_sample_sd():
    """Hand-computed: diffs 0.1/0.2/0.3 → mean 0.2, sample SD 0.1, dz = 2.0.

    The population-SD (``ddof=0``) answer for the same three numbers is 2.449, so this
    single assertion catches the convention slip that matters. It is also the reason
    the module divides by ``agg.summarize``'s SD rather than computing its own.
    """
    stat = agg.summarize([0.10, 0.20, 0.30])
    assert stat.mean == pytest.approx(0.20)
    assert stat.sd == pytest.approx(0.10)
    assert es.cohens_dz(stat) == pytest.approx(2.0, abs=1e-12)
    # The ddof=0 alternative, spelled out so the intent cannot be misread.
    population_sd = statistics.pstdev([0.10, 0.20, 0.30])
    assert 0.20 / population_sd == pytest.approx(2.4494897, abs=1e-6)


def test_cohens_dz_second_hand_computed_case():
    """diffs 1/2/6 → mean 3, sample SD sqrt(7) → dz = 3/sqrt(7) = 1.13389342."""
    assert es.cohens_dz(agg.summarize([1.0, 2.0, 6.0])) == pytest.approx(
        1.1338934190276817, abs=1e-12
    )


def test_cohens_dz_is_none_without_an_estimable_spread():
    assert es.cohens_dz(agg.summarize([0.5])) is None  # n < 2
    assert es.cohens_dz(agg.summarize([0.5, 0.5, 0.5])) is None  # sd == 0
    assert es.cohens_dz(agg.summarize([])) is None


def test_leave_one_seed_out_dz_is_pinned_and_is_not_an_interval():
    """Three refits at n=2 on the 0.1/0.2/0.3 differences, hand-computed.

    drop 42 → (0.2,0.3): 0.25/0.0707 = 3.5355; drop 43 → (0.1,0.3): 0.2/0.1414 =
    1.4142; drop 44 → (0.1,0.2): 0.15/0.0707 = 2.1213. The full-sample dz is 2.0, so
    removing one of three observations moves dz by ±75% — the number that stops a
    three-point dz from reading as decisive.
    """
    loo = es.leave_one_seed_out_dz({42: 0.10, 43: 0.20, 44: 0.30})
    assert loo["n_folds"] == 3
    assert loo["n_per_fold"] == 2
    assert loo["is_confidence_interval"] is False
    assert [f["n"] for f in loo["folds"]] == [2, 2, 2]
    by_dropped = {f["dropped_seed"]: f["dz"] for f in loo["folds"]}
    assert by_dropped[42] == pytest.approx(3.5355339, abs=1e-6)
    assert by_dropped[43] == pytest.approx(1.4142136, abs=1e-6)
    assert by_dropped[44] == pytest.approx(2.1213203, abs=1e-6)
    assert loo["min"] == pytest.approx(1.4142136, abs=1e-6)
    assert loo["max"] == pytest.approx(3.5355339, abs=1e-6)


def test_relative_effect_is_the_difference_over_the_named_reference_arm():
    assert es.relative_effect(0.20, 0.40) == pytest.approx(0.5)
    assert es.relative_effect(0.20, 0.0) is None
    assert es.relative_effect(None, 0.40) is None


def test_relative_effect_exceeds_reference_flag():
    assert es.relative_effect_exceeds_reference(0.5) is False
    assert es.relative_effect_exceeds_reference(-1.5) is True
    assert es.relative_effect_exceeds_reference(None) is False


# --------------------------------------------------------------------------- #
# 2. The endpoint table on a healthy fixture.
# --------------------------------------------------------------------------- #
def test_endpoint_effect_sizes_are_hand_checkable(bundle):
    zero_shot = _effect(bundle, "zero_shot_cer(MS→RV − RV→MS)")
    assert zero_shot["n"] == 3
    assert zero_shot["seeds"] == [42, 43, 44]
    assert zero_shot["diff"]["mean"] == pytest.approx(0.20)
    assert zero_shot["cohens_dz"] == pytest.approx(2.0, abs=1e-9)
    # MS→RV arm mean is exactly 0.40 by construction, so 0.20/0.40 = 0.5.
    assert zero_shot["reference_arm"] == "b"
    assert zero_shot["reference_arm_mean"] == pytest.approx(0.40)
    assert zero_shot["relative_effect"] == pytest.approx(0.5)
    # CER is an error metric, so the smaller arm (RV→MS, 0.20) is the better read.
    assert zero_shot["better_arm"] == "a"
    assert "RV→MS" in (zero_shot["better_arm_label"] or "")


def test_dz_sign_follows_the_published_b_minus_a_orientation(bundle):
    """A score metric is NOT re-oriented: chrF's published diff is negative, so is dz.

    v1's analysis flipped score metrics so that "positive = RV→MS better"; carrying
    both conventions in one repo is how a sign claim gets inverted, so this pins that
    v2 keeps exactly one.
    """
    cer = _effect(bundle, "zero_shot_cer(MS→RV − RV→MS)")
    chrf = _effect(bundle, "zero_shot_chrf(MS→RV − RV→MS)")
    assert cer["diff"]["mean"] > 0 and cer["cohens_dz"] > 0
    assert chrf["diff"]["mean"] < 0 and chrf["cohens_dz"] < 0
    # Both nevertheless say RV→MS is the better arm.
    assert cer["better_arm"] == "a" and chrf["better_arm"] == "a"


def test_a_zero_spread_contrast_reports_no_dz_but_keeps_its_difference(bundle):
    """All three seeds move by exactly +0.010: dz is undefined, the effect is not.

    ``mean/0`` is ``±inf``, which is not a reportable effect size; the unstandardized
    difference and its sign consistency still are.
    """
    row = _effect(bundle, "rv_primary_add_ms:rv_retention_cer")
    assert row["diff"]["sd"] == pytest.approx(0.0)
    assert row["cohens_dz"] is None
    assert row["diff"]["mean"] == pytest.approx(0.010)
    assert row["sign_consistent"] is True


def test_no_row_carries_a_dz_interval_or_a_significance_claim(bundle):
    for row in bundle["endpoint_effect_sizes"]:
        assert row["cohens_dz_ci"] is None
        assert row["cohens_dz_ci_available"] is False
        assert "noncentral-t" in row["cohens_dz_no_interval_reason"]
        assert row["significance_claim_available"] is False
        assert row["exact_sign_test"]["min_p_two_sided"] == pytest.approx(0.25)
        assert row["exact_sign_test"]["enumerated"] == 8
        assert "p_one_sided" not in row["exact_sign_test"]
    assert bundle["n_ceiling"]["dz_interval_available"] is False
    assert bundle["n_ceiling"]["significance_claim_available"] is False
    assert bundle["n_ceiling"]["exact_sign_test_min_p_two_sided"] == pytest.approx(0.25)


def test_sign_consistent_endpoint_contrasts_sit_exactly_on_the_n3_floor(bundle):
    for row in bundle["endpoint_effect_sizes"]:
        if row["sign_consistent"]:
            assert row["exact_sign_test"]["p_two_sided"] == pytest.approx(0.25)


def test_the_dz_column_is_empty_in_the_rendered_csv(bundle):
    text = es.render_endpoint_csv(bundle["endpoint_effect_sizes"])
    rows = list(csv.DictReader(text.splitlines()))
    assert rows
    for row in rows:
        assert row["cohens_dz_ci"] == ""
        assert row["significance_claim_available"] == "False"
        assert row["exact_sign_test_min_p_two_sided"] == "0.250000"
        # Fixed-point, never scientific: 4.98e-05 truncates into a 100000x error.
        for column in ("diff_mean", "relative_effect", "cohens_dz"):
            assert "e" not in row[column].lower()


# --------------------------------------------------------------------------- #
# 3. The plateau: ranking, tiebreak, and separability that must accompany it.
# --------------------------------------------------------------------------- #
def test_worst_view_is_max_for_an_error_metric_and_min_for_a_score():
    cell = _cells()[0]  # rv_ratio 1.0, seed 42: rv_cer 0.030 / ms_cer 0.200
    published = es.PublishedCell(
        rv_ratio=cell.rv_ratio,
        seed=cell.seed,
        run_name=cell.run_name,
        values={
            "rv_cer": 0.030,
            "ms_cer": 0.200,
            "rv_accent_f1": 0.95,
            "ms_accent_f1": 0.10,
        },
    )
    worst_cer = next(q for q in es.PLATEAU_QUANTITIES if q.name == "worst_cer")
    worst_f1 = next(q for q in es.PLATEAU_QUANTITIES if q.name == "worst_accent_f1")
    mean_cer = next(q for q in es.PLATEAU_QUANTITIES if q.name == "avg_cer")
    assert es.plateau_value(published, worst_cer) == pytest.approx(0.200)
    assert es.plateau_value(published, worst_f1) == pytest.approx(0.10)
    assert es.plateau_value(published, mean_cer) == pytest.approx(0.115)


def test_the_ranking_is_worst_cer_then_avg_cer(bundle):
    """75:25 and 50:50 tie exactly on worst-view CER; mean-view CER breaks it."""
    ranking = bundle["plateau"]["ranking"]
    per_ratio = bundle["plateau"]["per_ratio"]
    assert per_ratio["0.75"]["worst_cer"]["mean"] == pytest.approx(
        per_ratio["0.5"]["worst_cer"]["mean"], abs=1e-15
    )
    assert per_ratio["0.5"]["avg_cer"]["mean"] < per_ratio["0.75"]["avg_cer"]["mean"]
    assert [r["rv_ratio"] for r in ranking] == [0.5, 0.75, 0.25]
    assert [r["rank"] for r in ranking] == [1, 2, 3]
    assert all(r["n"] == 3 for r in ranking)
    assert all(r["rankable"] for r in ranking)
    assert bundle["plateau"]["verdict"]["ranking_best_first"] == [
        "50:50",
        "75:25",
        "25:75",
    ]


def test_every_ranked_pair_has_its_separability_row(bundle):
    pairs = {
        r["pair"]
        for r in bundle["plateau"]["separability"]
        if r["quantity"] == "worst_cer"
    }
    assert pairs == {"50:50 − 75:25", "25:75 − 75:25", "25:75 − 50:50"}
    verdict = bundle["plateau"]["verdict"]
    assert verdict["n_pairs"] == 3
    assert verdict["n_pairs_separable"] == 0
    assert "0/3 pairs are separable at n=3" in verdict["separability_statement"]


def test_no_pair_is_separable_even_with_a_ci_excluding_zero_and_no_overlap(bundle):
    """The row a reader would otherwise read as an ordering result.

    25:75 is worse than 50:50 on worst-view CER in all three seeds, its paired 95% CI
    excludes zero, and the two marginal CIs do not even overlap — and the exact paired
    test still cannot go below p = 0.25, so ``separable`` stays False. This is the
    single most important assertion in the file.
    """
    row = _separability(bundle, "25:75 − 50:50")
    assert row["sign_consistent"] is True
    assert row["paired_ci95_excludes_zero"] is True
    assert row["marginal_ci95_overlap"] is False
    assert row["exact_sign_test_p_two_sided"] == pytest.approx(0.25)
    assert row["separable"] is False
    assert row["verdict"] == es.ORDER_CONSISTENT
    assert all(r["separable"] is False for r in bundle["plateau"]["separability"])


def test_an_order_flipping_pair_is_reported_as_unstable(bundle):
    row = _separability(bundle, "50:50 − 75:25")
    assert row["sign_consistent"] is False
    assert row["verdict"] == es.ORDER_UNSTABLE
    assert row["marginal_ci95_overlap"] is True
    assert bundle["plateau"]["verdict"]["pairs_order_unstable"] == ["50:50 − 75:25"]


def test_separability_csv_states_the_verdict_on_every_row(bundle):
    text = es.render_separability_csv(bundle["plateau"]["separability"])
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == len(es.PLATEAU_PAIRS) * len(es.PLATEAU_QUANTITIES)
    for row in rows:
        assert row["separable"] == "False"
        assert row["verdict"] in {es.ORDER_CONSISTENT, es.ORDER_UNSTABLE}
        assert row["n"] == "3"


def test_plateau_rows_declare_which_are_reconciled_against_a_published_number(bundle):
    checked = {
        (r["rv_ratio"], r["quantity"]): r for r in bundle["reconciliation"]["plateau"]
    }
    assert checked[(0.5, "worst_cer")]["checked"] is True
    assert checked[(0.5, "worst_cer")]["published_as"] == "worst_view_cer"
    assert checked[(0.5, "worst_cer")]["ok"] is True
    # No published counterpart exists for WER at n=3; it must say so rather than
    # implying a check happened.
    assert checked[(0.5, "worst_wer")]["checked"] is False
    assert checked[(0.5, "worst_wer")]["published_as"] is None
    assert checked[(0.5, "worst_wer")]["ok"] is True


# --------------------------------------------------------------------------- #
# 4. Reconciliation and the gates.
# --------------------------------------------------------------------------- #
def test_a_healthy_input_pair_is_publishable(bundle):
    assert bundle["status"] == "COMPLETE"
    assert bundle["partial_banner"] is None
    assert bundle["publishable"] is True
    assert bundle["warnings"] == []
    assert bundle["hard_warnings"] == []
    assert bundle["coverage"]["n_present"] == 15
    assert all(r["ok"] for r in bundle["reconciliation"]["contrasts"])
    assert all(r["ok"] for r in bundle["reconciliation"]["per_ratio_value_order"])
    assert bundle["reconciliation"]["mean_of_views_is_runner_avg_cer"]["ok"] is True
    assert len(bundle["endpoint_effect_sizes"]) == len(es.CONTRASTS)


def test_a_partial_input_withholds_the_ranking(tmp_path):
    maingrid, selection = build_inputs(tmp_path, drop_cells=((0.5, 44),))
    _cells_read, aggregate = es.analyze(maingrid, selection)
    assert aggregate["status"] == "PARTIAL"
    assert "14/15" in (aggregate["partial_banner"] or "")
    assert aggregate["publishable"] is False
    assert aggregate["plateau"]["verdict"] is None
    assert aggregate["plateau"]["verdict_withheld_because"]
    assert aggregate["hard_warnings"]
    # The 50:50 plateau row is n=2 and says so rather than passing as n=3.
    assert aggregate["plateau"]["per_ratio"]["0.5"]["worst_cer"]["n"] == 2
    assert (
        es.main(["--maingrid-dir", str(maingrid), "--selection-dir", str(selection)])
        == 1
    )
    text = es.render_results_md(aggregate)
    assert "No ranking is quoted" in text


def test_a_tampered_per_seed_value_fails_the_contrast_reconciliation(tmp_path):
    maingrid, selection = build_inputs(tmp_path)
    path = maingrid / "tables" / "per_seed.csv"
    # Move the MS-only model's RV-test CER for seed 42, which is one arm of the
    # zero-shot contrast: the published asymmetry row no longer follows from the cells.
    text = path.read_text(encoding="utf-8").replace("0.3,", "0.9,", 1)
    path.write_text(text, encoding="utf-8")
    _cells_read, aggregate = es.analyze(maingrid, selection)
    zero_shot = _effect(aggregate, "zero_shot_cer(MS→RV − RV→MS)")
    assert zero_shot["reconciliation"]["ok"] is False
    assert aggregate["publishable"] is False
    assert any("not reconciled" in w for w in aggregate["hard_warnings"])


def test_a_contrast_missing_from_the_published_table_is_a_hard_failure(tmp_path):
    maingrid, selection = build_inputs(tmp_path)
    path = maingrid / "tables" / "asymmetry.csv"
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.startswith("zero_shot_chrf")
    ]
    path.write_text("".join(kept), encoding="utf-8")
    _cells_read, aggregate = es.analyze(maingrid, selection)
    chrf = _effect(aggregate, "zero_shot_chrf(MS→RV − RV→MS)")
    assert chrf["reconciliation"]["ok"] is False
    assert (
        chrf["reconciliation"]["reason"]
        == "no published row with this comparison label"
    )
    assert aggregate["publishable"] is False


def test_a_published_contrast_with_no_spec_is_reported(tmp_path):
    maingrid, selection = build_inputs(tmp_path)
    path = maingrid / "tables" / "asymmetry.csv"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("brand_new_contrast,3,0.1,0.0,0.2,b>a,True,42:0.1;43:0.1;44:0.1\n")
    _cells_read, aggregate = es.analyze(maingrid, selection)
    assert any("no effect-size spec" in w for w in aggregate["hard_warnings"])
    assert aggregate["publishable"] is False


def test_a_shuffled_per_ratio_values_column_is_caught(tmp_path):
    """``per_ratio.csv``'s ``values`` is unlabelled; the seed-order check is what
    licenses pairing from ``per_seed.csv`` at all."""
    maingrid, selection = build_inputs(tmp_path)
    path = maingrid / "tables" / "per_ratio.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        if row["rv_ratio"] == "0.0" and row["metric"] == "rv_cer":
            row["values"] = ";".join(reversed(row["values"].split(";")))
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(agg.PER_RATIO_COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")
    _cells_read, aggregate = es.analyze(maingrid, selection)
    assert any("per_ratio.csv values" in w for w in aggregate["hard_warnings"])
    assert aggregate["publishable"] is False


def test_a_non_v2_input_is_refused(tmp_path):
    maingrid, selection = build_inputs(tmp_path, corpus="v1")
    _cells_read, aggregate = es.analyze(maingrid, selection)
    assert aggregate["input_attestation"]["ok"] is False
    assert aggregate["publishable"] is False
    assert any("corpus attestation" in w for w in aggregate["hard_warnings"])


def test_an_unpublishable_maingrid_is_refused(tmp_path):
    maingrid, selection = build_inputs(tmp_path, publishable=False)
    _cells_read, aggregate = es.analyze(maingrid, selection)
    assert aggregate["input_attestation"]["maingrid"]["ok"] is False
    assert aggregate["publishable"] is False


def test_a_missing_input_table_is_an_input_error(tmp_path):
    maingrid, selection = build_inputs(tmp_path)
    (maingrid / "tables" / "per_seed.csv").unlink()
    with pytest.raises(es.InputError):
        es.analyze(maingrid, selection)
    assert (
        es.main(["--maingrid-dir", str(maingrid), "--selection-dir", str(selection)])
        == 1
    )


# --------------------------------------------------------------------------- #
# 5. The written bundle.
# --------------------------------------------------------------------------- #
def test_cli_writes_the_bundle(tmp_path):
    maingrid, selection = build_inputs(tmp_path)
    out_dir = tmp_path / "bundle"
    rc = es.main(
        [
            "--maingrid-dir",
            str(maingrid),
            "--selection-dir",
            str(selection),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    for name in (
        "tables/endpoint_effect_sizes.csv",
        "tables/endpoint_dz_leave_one_seed_out.csv",
        "tables/plateau_comparison.csv",
        "tables/plateau_separability.csv",
        "tables/reconciliation.csv",
        "results.md",
        "aggregate.json",
        "manifest.json",
    ):
        assert (out_dir / name).is_file(), name
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_only"] is True
    assert manifest["no_training_inference_or_gpu"] is True
    assert manifest["corpus"] == "v2"
    assert "ddof=1" in manifest["definitions_added_here"]["cohens_dz"]
    assert manifest["v1_comparison"].startswith("none")
    # Every input is a committed docs/ artifact — never a run dir or an NFS mirror.
    blob = json.dumps(manifest)
    assert "/archive/" not in blob
    assert "outputs/" not in blob
    plateau_rows = list(
        csv.DictReader(
            (out_dir / "tables" / "plateau_comparison.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    assert {r["worst_cer_rank"] for r in plateau_rows} == {"1", "2", "3"}
    assert all(r["seeds"] == "42;43;44" for r in plateau_rows)


def test_results_md_leads_with_what_the_numbers_cannot_support(bundle):
    text = es.render_results_md(bundle)
    assert "What these two numbers can and cannot support" in text
    assert "0.25" in text
    assert "not a CI" in text
    assert "no pair is separable at n=3" in text.lower()
    # The ranking sentence and the separability sentence must both be present.
    assert "Ranked best-first on worst_cer: 50:50 < 75:25 < 25:75." in text


# --------------------------------------------------------------------------- #
# 6. The committed bundle, and the paper documents that quote it.
# --------------------------------------------------------------------------- #
@pytest.fixture
def committed() -> dict[str, Any]:
    path = COMMITTED_BUNDLE / "aggregate.json"
    if not path.is_file():
        pytest.skip(f"{path} is not committed")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_committed_bundle_is_publishable_and_fully_reconciled(committed):
    assert committed["status"] == "COMPLETE"
    assert committed["publishable"] is True
    assert committed["hard_warnings"] == []
    assert committed["coverage"]["n_present"] == 15
    rec = committed["reconciliation"]
    assert [r["ok"] for r in rec["contrasts"]] == [True] * len(es.CONTRASTS)
    assert all(r["ok"] for r in rec["plateau"])
    assert all(r["ok"] for r in rec["per_ratio_value_order"])
    assert rec["mean_of_views_is_runner_avg_cer"]["n_cells"] == 15


def test_the_committed_dz_recomputes_from_the_published_per_seed_diffs(committed):
    """Independent recomputation straight from the committed main-grid CSV.

    Not a re-run of the module: the per-seed differences are read out of
    ``asymmetry.csv`` / ``retention_adaptation.csv`` and pushed through
    ``statistics`` directly, so a change in either the module's pairing or its dz
    formula shows up here.
    """
    if not COMMITTED_MAINGRID.is_dir():
        pytest.skip("the committed main-grid bundle is absent")
    published: dict[str, list[float]] = {}
    for table in ("asymmetry.csv", "retention_adaptation.csv"):
        path = COMMITTED_MAINGRID / "tables" / table
        for row in csv.DictReader(path.read_text(encoding="utf-8").splitlines()):
            diffs = es.parse_per_seed_diff(row["per_seed_diff"])
            published[row["comparison"]] = [diffs[s] for s in sorted(diffs)]
    checked = 0
    for effect in committed["endpoint_effect_sizes"]:
        diffs = published[effect["comparison"]]
        assert len(diffs) == 3
        if effect["cohens_dz"] is None:
            continue
        expected = statistics.fmean(diffs) / statistics.stdev(diffs)
        # The published per-seed diffs are printed at 6dp while the bundle pairs from
        # the full-precision cells, so the two agree to well within any reportable
        # digit. The absolute leg matters for the MS-retention row, whose mean is
        # 4.98e-05: there, 6dp rounding moves dz in its third decimal.
        assert effect["cohens_dz"] == pytest.approx(expected, rel=5e-3, abs=1e-3)
        checked += 1
    assert checked == len(es.CONTRASTS)


def test_the_committed_ranking_is_reported_with_its_non_separability(committed):
    verdict = committed["plateau"]["verdict"]
    assert verdict["ranking_best_first"] == ["50:50", "75:25", "25:75"]
    assert verdict["n_pairs_separable"] == 0
    assert verdict["all_marginal_cis_overlap"] is True
    assert verdict["pairs_order_unstable"] == ["50:50 − 75:25"]
    assert sorted(verdict["pairs_consistently_ordered"]) == [
        "25:75 − 50:50",
        "25:75 − 75:25",
    ]
    assert all(r["separable"] is False for r in committed["plateau"]["separability"])


#: ``PENDING(v2-<slug>-#<issue>)`` — a real tag, as opposed to the ``PENDING(v2-…)``
#: ellipsis the documents use when talking *about* the convention.






