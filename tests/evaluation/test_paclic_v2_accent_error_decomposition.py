"""Test paclic v2 accent error decomposition: PACLIC 2026 v2 release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation import accent_metrics as am
from scripts.evaluation import build_paclic_v2_accent_error_decomposition as dec
from scripts.evaluation import build_paclic_v2_maingrid_analysis as mg

TOTAL = 10420


# --------------------------------------------------------------------------- #
# Helpers: build a prediction CSV and the committed counts that go with it.
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, pairs: list[tuple[str, str]]) -> Path:
    """Write a legacy two-column prediction CSV of ``(reference, prediction)`` pairs."""
    lines = ["reference,prediction"]
    for reference, prediction in pairs:
        lines.append(f'"{reference}","{prediction}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _committed(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """The accent block a runner would have committed for these pairs.

    Computed with the untouched shared metric function on the runner's own transform,
    which is exactly what ``experiment_accent_ratio.evaluate_on_dataset`` does. So the
    fixture is not a restatement of the decomposition — it is the published-number
    side of the reconciliation.
    """
    references = [dec.normalize_runner_iso(r) for r, _ in pairs]
    predictions = [dec.normalize_runner_iso(p) for _, p in pairs]
    return am.compute_all_accent_aware_metrics(predictions, references)


def _decompose(tmp_path: Path, pairs: list[tuple[str, str]], **kwargs) -> dict:
    return dec.decompose_prediction_file(
        _write_csv(tmp_path / "one_predictions.csv", pairs), **kwargs
    )


# --------------------------------------------------------------------------- #
# 1. The restated walk must charge what the locked module charges.
# --------------------------------------------------------------------------- #
CASES: list[tuple[str, str]] = [
    ("agním", "agním"),  # clean match
    ("agním", "agnim"),  # mark not written
    ("agním", "agnìm"),  # other kind
    ("agnim", "agním"),  # mark invented
    ("ágnim", "agním"),  # mark one syllable off
    ("agním", "agnkm"),  # the vowel itself misread
    ("agním purohitám", "agnim purohitam"),  # two marks not written
    ("rā́yás", "rāyas"),  # long vowel + accent
    ("kŕ̥ṣṇa", "kr̥ṣṇa"),  # vocalic r with an accent
    ("vaí devá", "váí devà"),  # diphthong accent, exercised through the transform
    ("agním", ""),  # empty prediction
    ("", "agním"),  # empty reference
    ("", ""),  # both empty
    ("śatám", "śatam"),  # the letter ś next to a real accent
]


def test_walk_charges_exactly_what_compute_accent_metrics_charges():
    """The guard that replaces the shared code path accent_metrics cannot give us."""
    references = [dec.normalize_runner_iso(r) for r, _ in CASES]
    predictions = [dec.normalize_runner_iso(p) for _, p in CASES]
    alignments = dec.iter_accent_alignments(predictions, references)
    check = dec.self_check_counts(alignments, predictions, references)
    assert check["matches_compute_accent_metrics"], check["deltas"]


@pytest.mark.parametrize("pair", CASES)
def test_walk_matches_the_metric_pair_by_pair(pair: tuple[str, str]):
    reference = [dec.normalize_runner_iso(pair[0])]
    prediction = [dec.normalize_runner_iso(pair[1])]
    alignments = dec.iter_accent_alignments(prediction, reference)
    assert dec.self_check_counts(alignments, prediction, reference)[
        "matches_compute_accent_metrics"
    ]


def test_self_check_reports_a_mismatch_rather_than_hiding_it():
    """If the walk ever drifts, the self-check must say so with the deltas."""
    references = ["agním"]
    predictions = ["agnim"]
    alignments = dec.iter_accent_alignments(predictions, references)
    # Drop the one event: a walk that misses positions must be caught.
    broken = [
        dec.AccentAlignment(
            row=a.row,
            ref_nfd=a.ref_nfd,
            hyp_nfd=a.hyp_nfd,
            ref_vowels=a.ref_vowels,
            hyp_vowels=a.hyp_vowels,
            events=(),
        )
        for a in alignments
    ]
    check = dec.self_check_counts(broken, predictions, references)
    assert not check["matches_compute_accent_metrics"]
    assert check["deltas"]["unlabeled"]["fn"] == -1


def test_alignment_walk_skips_both_empty_rows_like_the_metric_does():
    alignments = dec.iter_accent_alignments(["", "a"], ["", "á"])
    assert [a.row for a in alignments] == [1]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        dec.iter_accent_alignments(["a"], ["a", "b"])


# --------------------------------------------------------------------------- #
# 2. One hand-built case per class.
# --------------------------------------------------------------------------- #
def _counts(tmp_path: Path, reference: str, prediction: str) -> dict[str, int]:
    return _decompose(tmp_path, [(reference, prediction)])["counts"]


def test_matched_same_kind_is_the_only_non_error(tmp_path):
    counts = _counts(tmp_path, "agním", "agním")
    assert counts["matched_same_kind"] == 1
    assert all(counts[c] == 0 for c in dec.ERROR_ONLY_CLASSES)


def test_other_kind_is_an_acute_where_a_grave_belongs(tmp_path):
    counts = _counts(tmp_path, "agnìm", "agním")
    assert counts["matched_other_kind"] == 1
    assert counts["matched_same_kind"] == 0


def test_no_mark_at_read_syllable_needs_the_same_base_letter(tmp_path):
    counts = _counts(tmp_path, "agním", "agnim")
    assert counts["no_mark_at_read_syllable"] == 1
    assert counts["ref_syllable_misread"] == 0


def test_ref_syllable_misread_when_the_vowel_itself_is_wrong(tmp_path):
    counts = _counts(tmp_path, "agním", "agnkm")
    assert counts["ref_syllable_misread"] == 1
    assert counts["no_mark_at_read_syllable"] == 0


def test_ref_syllable_misread_when_the_vowel_is_deleted(tmp_path):
    counts = _counts(tmp_path, "agním", "agnm")
    assert counts["ref_syllable_misread"] == 1


def test_extra_at_read_syllable_is_a_mark_on_a_correctly_read_letter(tmp_path):
    counts = _counts(tmp_path, "agnim", "agním")
    assert counts["extra_at_read_syllable"] == 1
    assert counts["extra_at_misread_syllable"] == 0


def test_extra_at_misread_syllable_when_the_letter_is_also_wrong(tmp_path):
    counts = _counts(tmp_path, "agnkm", "agním")
    assert counts["extra_at_misread_syllable"] == 1
    assert counts["extra_at_read_syllable"] == 0


def test_matched_on_substituted_base_is_a_subset_not_a_class(tmp_path):
    """A mark that matched although the letter did not: credited, and flagged."""
    counts = _counts(tmp_path, "ágnim", "ígnim")
    assert counts["matched_same_kind"] == 1
    assert counts["matched_on_substituted_base"] == 1
    assert counts["matched_on_substituted_base"] not in (None,)
    assert "matched_on_substituted_base" not in dec.ERROR_CLASSES


# --------------------------------------------------------------------------- #
# 3. Misplacement: a refinement, with a density null, that never moves the partition.
# --------------------------------------------------------------------------- #
def test_misplacement_pairs_an_adjacent_same_kind_mark(tmp_path):
    payload = _decompose(tmp_path, [("ágnim", "agním")])
    assert payload["misplacement"]["misplaced"] == 1
    assert payload["misplacement"]["distance_histogram"] == {"1": 1}


def test_misplacement_requires_the_same_kind(tmp_path):
    """A nearby mark of the other kind is not "right kind, wrong syllable"."""
    payload = _decompose(tmp_path, [("ágnim", "agnìm")])
    assert payload["misplacement"]["misplaced"] == 0
    assert payload["misplacement"]["misplaced_other_kind"] == 1


def test_misplacement_respects_the_window(tmp_path):
    """Three syllables apart is not a misplacement at the pinned window of 1."""
    pairs = [("ágnimana", "agnimaná")]
    assert _decompose(tmp_path, pairs, window=1)["misplacement"]["misplaced"] == 0
    assert _decompose(tmp_path, pairs, window=3)["misplacement"]["misplaced"] == 1


def test_the_partition_does_not_depend_on_the_window(tmp_path):
    """The window may only move the refinement; the classes are window-free."""
    pairs = [("ágnimana", "agnimaná"), ("ágnim", "agním")]
    narrow = _decompose(tmp_path, pairs, window=0)
    wide = _decompose(tmp_path, pairs, window=3)
    assert narrow["counts"] == wide["counts"]
    assert narrow["misplacement"]["misplaced"] != wide["misplacement"]["misplaced"]


def test_misplacement_always_travels_with_its_density_null(tmp_path):
    payload = _decompose(tmp_path, [("ágnimana", "agnímana")])
    refinement = payload["misplacement"]
    assert set(refinement) >= {
        "misplaced",
        "misplaced_shift_null",
        "misplaced_excess_over_null",
    }
    assert (
        refinement["misplaced_excess_over_null"]
        == refinement["misplaced"] - refinement["misplaced_shift_null"]
    )


def test_shift_null_is_zero_when_a_row_has_one_syllable(tmp_path):
    """Nothing to rotate into: the control must not invent a pairing."""
    payload = _decompose(tmp_path, [("á", "a")])
    assert payload["misplacement"]["misplaced_shift_null"] == 0


def test_sensitivity_windows_are_all_reported(tmp_path):
    payload = _decompose(tmp_path, [("ágnimana", "agnímana")])
    assert set(payload["misplacement_sensitivity"]) == {
        str(w) for w in dec.MISPLACEMENT_SENSITIVITY_WINDOWS
    }


# --------------------------------------------------------------------------- #
# 4. The partitions and the identities.
# --------------------------------------------------------------------------- #
def test_both_partitions_sum_to_the_channel_totals(tmp_path):
    payload = _decompose(tmp_path, CASES)
    counts = payload["counts"]
    assert (
        sum(counts[c] for c in dec.REF_SIDE_CLASSES) == payload["n_ref_accent_vowels"]
    )
    assert (
        sum(counts[c] for c in dec.HYP_SIDE_CLASSES) == payload["n_hyp_accent_vowels"]
    )


def test_every_identity_reconciles_against_the_committed_counts(tmp_path):
    payload = _decompose(tmp_path, CASES)
    reconciliation = dec.reconcile_view(payload["counts"], _committed(CASES))
    assert reconciliation["reconciles"], reconciliation["identities"]
    assert reconciliation["n_checked"] == len(dec.RECONCILIATION_IDENTITIES)
    assert reconciliation["mismatches"] == []


def test_a_perturbed_committed_count_fails_the_reconciliation(tmp_path):
    payload = _decompose(tmp_path, CASES)
    committed = dict(_committed(CASES))
    committed["accent_fn"] = committed["accent_fn"] + 1
    reconciliation = dec.reconcile_view(payload["counts"], committed)
    assert not reconciliation["reconciles"]
    assert reconciliation["mismatches"] == ["accent_fn"]
    assert reconciliation["identities"][1]["delta"] == -1


def test_a_missing_committed_key_is_unchecked_not_passing(tmp_path):
    payload = _decompose(tmp_path, CASES)
    committed = {
        k: v for k, v in _committed(CASES).items() if k != "n_hyp_accent_vowels"
    }
    reconciliation = dec.reconcile_view(payload["counts"], committed)
    assert reconciliation["n_unchecked"] == 1
    assert not reconciliation["reconciles"]


def test_labeled_identities_are_where_the_two_f1s_differ(tmp_path):
    """``matched_other_kind`` is a TP unlabelled and an FN *and* FP labelled."""
    pairs = [("agnìm", "agním")]
    payload = _decompose(tmp_path, pairs)
    committed = _committed(pairs)
    assert committed["accent_tp"] == 1
    assert committed["accent_tp_labeled"] == 0
    assert committed["accent_fn_labeled"] == 1
    assert committed["accent_fp_labeled"] == 1
    assert dec.reconcile_view(payload["counts"], committed)["reconciles"]


# --------------------------------------------------------------------------- #
# 5. The ś census and the transform.
# --------------------------------------------------------------------------- #
def test_the_letter_s_is_counted_as_a_letter_not_an_accent(tmp_path):
    """U+0301 is also the second half of ś; the channel must not charge it."""
    payload = _decompose(tmp_path, [("śaśá", "śaśa")])
    assert payload["n_ref_accent_marks_nfd"] == 3  # two ś + one real acute
    assert payload["n_ref_accent_marks_on_letter_s"] == 2
    assert payload["n_ref_accent_vowels"] == 1
    assert payload["n_ref_accent_marks_unanchored_other"] == 0


def test_the_census_residual_is_measured_not_assumed(tmp_path):
    """An accent on a consonant the channel cannot anchor is reported, not dropped."""
    payload = _decompose(tmp_path, [("ḱa", "ka")])
    assert payload["n_ref_accent_marks_nfd"] == 1
    assert payload["n_ref_accent_vowels"] == 0
    assert payload["n_ref_accent_marks_unanchored_other"] == 1


def test_the_runner_transform_shifts_the_diphthong_accent():
    """The transform that makes results.json reproducible, asserted directly."""
    assert dec.normalize_runner_iso("vái") == "vaí"


def test_the_alt_normalization_block_is_reported_next_to_the_used_one(tmp_path):
    payload = _decompose(tmp_path, CASES)
    assert payload["text_transform"] == dec.RUNNER_TRANSFORM_LABEL
    assert payload["alt_normalization"]["normalization"] == dec.ALT_TRANSFORM_LABEL
    assert set(payload["alt_normalization"]) >= {"accent_tp", "accent_fp", "accent_fn"}


def test_the_alt_normalization_can_be_switched_off(tmp_path):
    payload = _decompose(tmp_path, CASES, alt_normalization=None)
    assert "alt_normalization" not in payload


# --------------------------------------------------------------------------- #
# 6. A synthetic grid: coverage, aggregation, contrasts, bundle, CLI.
# --------------------------------------------------------------------------- #
#: Per view, the (reference, prediction) pairs a "trained" and an "untrained" cell
#: produce. The untrained one keeps writing marks — in the wrong places — which is
#: what the mark-budget row exists to expose.
TRAINED_PAIRS: list[tuple[str, str]] = [
    ("agním purohitám", "agním purohitám"),
    ("rā́yás vaí devá", "rā́yás vaí devá"),
]
UNTRAINED_PAIRS: list[tuple[str, str]] = [
    ("agním purohitám", "agním puróhitam"),
    ("rā́yás vaí devá", "rāyā́s váí déva"),
]


def _payload(rv_ratio: float, seed: int) -> dict:
    """A v2 ``results.json``: the accent blocks are the committed side of §4."""
    rv_pairs = TRAINED_PAIRS if rv_ratio >= 0.5 else UNTRAINED_PAIRS
    ms_pairs = TRAINED_PAIRS if rv_ratio < 0.5 else UNTRAINED_PAIRS
    n_rv, n_ms = mg.expected_mix_counts(TOTAL, rv_ratio)
    return {
        "experiment_name": mg.run_name(rv_ratio, seed),
        "seed": seed,
        "model_base": "fixed-foundation",
        "freeze_mode": "none",
        "input_scheme": "iso",
        "num_train_epochs": 20,
        "training_state": {"best_epoch": 12.0, "best_step": 3912, "global_step": 6520},
        "mix_mode": "parallel_by_text",
        "key_column": "text",
        "selection_algorithm": "sorted_common_keys+seeded_shuffle+prefix",
        "parallel_keys_total": TOTAL,
        "parallel_keys_used": TOTAL,
        "parallel_subset_seed": seed,
        "key_list_sha256": f"hash-{seed}",
        "total_train_samples_requested": TOTAL,
        "rv_train_samples": n_rv,
        "ms_train_samples": n_ms,
        "rv_test": {"dataset": "RV Test", "cer": 0.04, **_committed(rv_pairs)},
        "ms_test": {"dataset": "MS Test", "cer": 0.04, **_committed(ms_pairs)},
    }


def _write_grid(
    root: Path,
    *,
    skip: frozenset[str] = frozenset(),
    preds_skip: frozenset[str] = frozenset(),
    mutate: dict[str, dict] | None = None,
) -> Path:
    for rv_ratio in mg.RV_RATIOS:
        for seed in mg.SEEDS:
            name = mg.run_name(rv_ratio, seed)
            if name in skip:
                continue
            cell = root / name
            cell.mkdir(parents=True, exist_ok=True)
            payload = _payload(rv_ratio, seed)
            if mutate and name in mutate:
                for view, changes in mutate[name].items():
                    payload[view].update(changes)
            (cell / "results.json").write_text(json.dumps(payload), encoding="utf-8")
            if name in preds_skip:
                continue
            rv_pairs = TRAINED_PAIRS if rv_ratio >= 0.5 else UNTRAINED_PAIRS
            ms_pairs = TRAINED_PAIRS if rv_ratio < 0.5 else UNTRAINED_PAIRS
            _write_csv(cell / "rv_predictions.csv", rv_pairs)
            _write_csv(cell / "ms_predictions.csv", ms_pairs)
    return root


def _grid(tmp_path: Path, **kwargs):
    root = _write_grid(tmp_path / "study", **kwargs)
    cells, _ = mg.load_grid(root, metrics_fn=dec._no_generic_metrics, cross_check=False)
    rows, skipped = dec.decompose_grid(cells, root)
    coverage = mg.coverage_summary(cells)
    view_cov = dec.view_coverage(rows, skipped, coverage)
    aggregate = dec.build_aggregate(cells, rows, coverage, view_cov)
    return root, cells, rows, aggregate


def test_a_complete_grid_decomposes_thirty_views(tmp_path):
    _root, _cells, rows, aggregate = _grid(tmp_path)
    assert len(rows) == dec.TARGET_N_VIEWS == 30
    assert aggregate["status"] == "COMPLETE"
    assert aggregate["partial_banner"] is None


def test_every_view_reconciles_and_self_checks_on_the_grid(tmp_path):
    _root, _cells, _rows, aggregate = _grid(tmp_path)
    assert aggregate["reconciliation"]["all_reconcile"]
    assert aggregate["alignment_self_check"]["all_match"]
    assert aggregate["alignment_self_check"]["n_views"] == 30


def test_a_present_cell_without_predictions_is_named_not_imputed(tmp_path):
    missing = mg.run_name(0.5, 43)
    _root, _cells, rows, aggregate = _grid(tmp_path, preds_skip=frozenset({missing}))
    assert len(rows) == 28
    assert aggregate["status"] == "PARTIAL"
    banner = aggregate["partial_banner"]
    assert "NOT A COMPLETE RESULT" in banner
    assert f"{missing}:rv" in banner and f"{missing}:ms" in banner
    assert dec.hard_warnings(aggregate["warnings"])


def test_an_absent_cell_is_partial_too(tmp_path):
    _root, _cells, rows, aggregate = _grid(
        tmp_path, skip=frozenset({mg.run_name(1.0, 44)})
    )
    assert len(rows) == 28
    assert not aggregate["complete"]
    assert aggregate["coverage"]["cells_complete"] is False


def test_a_perturbed_committed_count_blocks_publication(tmp_path):
    """The bundle must fail loudly when a class sum contradicts a published number."""
    broken = mg.run_name(1.0, 42)
    _root, _cells, _rows, aggregate = _grid(
        tmp_path, mutate={broken: {"rv_test": {"accent_fn": 999}}}
    )
    assert not aggregate["reconciliation"]["all_reconcile"]
    blocking = dec.hard_warnings(aggregate["warnings"])
    assert any("does not reconcile" in w for w in blocking)
    assert aggregate["publishable"] is False


def test_per_ratio_view_rows_carry_their_own_n_and_seeds(tmp_path):
    _root, _cells, _rows, aggregate = _grid(tmp_path)
    rows = aggregate["per_ratio_view"]
    assert len(rows) == len(mg.RV_RATIOS) * len(dec.VIEWS) * len(dec.ERROR_CLASSES)
    for row in rows:
        assert row["per_ref_accent_vowel"]["n"] == 3
        assert row["seeds"] == list(mg.SEEDS)
        assert row["per_ref_accent_vowel"]["ci95_halfwidth"] is not None or (
            row["per_ref_accent_vowel"]["sd"] == 0.0
        )


def test_the_reference_side_shares_sum_to_one_per_ratio_view(tmp_path):
    _root, _cells, rows, _aggregate = _grid(tmp_path)
    for row in rows:
        total = sum(row["per_ref_accent_vowel"][c] for c in dec.REF_SIDE_CLASSES)
        assert total == pytest.approx(1.0)


def test_mark_budget_is_hyp_over_ref(tmp_path):
    _root, _cells, rows, aggregate = _grid(tmp_path)
    for row in rows:
        assert row["hyp_marks_per_ref_accent"] == pytest.approx(
            row["n_hyp_accent_vowels"] / row["n_ref_accent_vowels"]
        )
    assert len(aggregate["mark_budget_per_ratio_view"]) == len(mg.RV_RATIOS) * 2


def test_trained_on_view_is_only_defined_at_the_endpoints(tmp_path):
    assert dec.trained_on_view(1.0, "rv") is True
    assert dec.trained_on_view(1.0, "ms") is False
    assert dec.trained_on_view(0.0, "ms") is True
    assert dec.trained_on_view(0.5, "rv") is None


def test_contrasts_are_oriented_untrained_minus_trained(tmp_path):
    _root, _cells, _rows, aggregate = _grid(tmp_path)
    contrasts = aggregate["trained_vs_untrained"]
    assert set(contrasts) == {"rv_only_model", "ms_only_model"}
    for label in contrasts:
        block = contrasts[label]
        assert "untrained_view - trained_view" in block["orientation"]
        matched = block["per_class"]["matched_same_kind"]["diff"]
        # The fixture's untrained view mangles every mark, so the non-error class must
        # be *less* frequent there: a positive Δ would mean the orientation flipped.
        assert matched["mean"] < 0
        assert block["per_class"]["no_mark_at_read_syllable"]["diff"]["mean"] > 0


def test_contrasts_only_use_the_two_endpoint_models(tmp_path):
    _root, _cells, rows, _aggregate = _grid(tmp_path)
    contrasts = dec.trained_vs_untrained_contrasts(rows)
    for block in contrasts.values():
        assert block["rv_ratio"] in (1.0, 0.0)


def test_reference_set_identity_is_single_and_says_it_cannot_attest_the_corpus(
    tmp_path,
):
    _root, _cells, rows, aggregate = _grid(tmp_path)
    identity = aggregate["reference_set_identity"]
    # The fixture's RV and MS references are the same text (parallel by text).
    assert identity["single_reference_set"]
    assert identity["n_distinct_reference_sequences"] == 1
    assert "CANNOT attest the corpus" in identity["note"]


def test_a_split_reference_set_is_a_blocking_warning(tmp_path):
    root, cells, rows, _aggregate = _grid(tmp_path)
    rows[0]["reference_sha256"] = "deadbeef"
    identity = dec.reference_set_identity(rows)
    assert not identity["single_reference_set"]
    warnings = dec.audit_warnings(
        cells,
        mg.coverage_summary(cells),
        dec.view_coverage(rows, [], mg.coverage_summary(cells)),
        rows,
        identity,
    )
    assert any("more than one reference set" in w for w in warnings)
    assert dec.hard_warnings(warnings)


# --------------------------------------------------------------------------- #
# 7. Artifacts + CLI.
# --------------------------------------------------------------------------- #
def test_build_bundle_writes_every_artifact(tmp_path):
    root, _cells, rows, aggregate = _grid(tmp_path)
    out = tmp_path / "bundle"
    dec.build_bundle(rows, aggregate, out, study_root=root)
    for name in (
        "results.md",
        "aggregate.json",
        "manifest.json",
        "tables/per_cell.csv",
        "tables/by_ratio_view.csv",
        "tables/mark_budget.csv",
        "tables/misplacement.csv",
        "tables/reconciliation.csv",
        "tables/trained_vs_untrained.csv",
        "tables/class_definitions.md",
    ):
        assert (out / name).is_file(), name


def test_results_page_leads_with_status_and_carries_the_reconciliation(tmp_path):
    _root, _cells, _rows, aggregate = _grid(tmp_path)
    page = dec.render_results_md(aggregate)
    assert page.startswith("# PACLIC v2 accent error decomposition")
    assert "**Status:** COMPLETE" in page
    assert "30/30 decomposed views satisfy all 8 identities exactly" in page
    assert "source lock" in page
    # No v1 comparison anywhere: the page says so, and cites no v1 bundle.
    assert "**No v1 number appears here**" in page
    assert "paclic-accent-transfer-analysis" not in page


def test_results_page_says_partial_and_lists_the_gap(tmp_path):
    missing = mg.run_name(0.25, 42)
    _root, _cells, _rows, aggregate = _grid(tmp_path, preds_skip=frozenset({missing}))
    page = dec.render_results_md(aggregate)
    assert "**Status:** PARTIAL" in page
    assert "NOT A COMPLETE RESULT" in page
    assert missing in page


def test_manifest_records_the_source_lock_workaround_and_no_v1_comparison(tmp_path):
    root, _cells, rows, aggregate = _grid(tmp_path)
    manifest = dec.build_manifest(
        aggregate, study_root=root, shared_study_root=None, rows=rows
    )
    assert "source lock" in manifest["method"]["aligner"]
    assert manifest["method"]["alignment_self_check"]["all_match"]
    assert "None." in manifest["not_in_this_bundle"]["v1_comparison"]
    assert manifest["corpus_claim_basis"].startswith("Not asserted")
    assert manifest["analysis_only"] and manifest["no_training_no_inference"]


def test_every_class_has_a_definition_and_no_definition_is_orphaned():
    assert set(dec.CLASS_DEFINITIONS) == set(dec.ERROR_CLASSES)
    for _name, classes, _key in dec.RECONCILIATION_IDENTITIES:
        assert set(classes) <= set(dec.ERROR_CLASSES)
    page = dec.render_class_definitions_md()
    for cls in dec.ERROR_CLASSES:
        assert f"`{cls}`" in page
    assert dec.WINDOW_ZERO_NOTE in page


def test_no_traditional_accent_terminology_in_this_module_or_its_pages():
    """The kinds are acute / grave / none; the banned glosses stay out of the prose."""
    source = Path(dec.__file__).read_text(encoding="utf-8")
    page = dec.render_class_definitions_md()
    for banned in ("udatta", "udātta", "anudatta", "anudātta", "svarita"):
        assert banned not in source.lower()
        assert banned not in page.lower()


def test_cli_writes_a_bundle_and_exits_zero_on_a_clean_grid(tmp_path):
    root = _write_grid(tmp_path / "study")
    out = tmp_path / "bundle"
    code = dec.main(
        [
            "--study-root",
            str(root),
            "--out-dir",
            str(out),
            "--no-slurm-attestation",
        ]
    )
    assert code == 0
    payload = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    # Attestation skipped is not attestation passed: the bundle may not call itself
    # publishable on the strength of a check nobody ran.
    assert payload["publishable"] is False


def test_cli_exits_nonzero_on_a_partial_grid(tmp_path):
    root = _write_grid(tmp_path / "study", preds_skip=frozenset({mg.run_name(0.0, 44)}))
    code = dec.main(
        [
            "--study-root",
            str(root),
            "--out-dir",
            str(tmp_path / "bundle"),
            "--no-slurm-attestation",
        ]
    )
    assert code == 1


def test_cli_prints_the_page_without_an_out_dir(tmp_path, capsys):
    root = _write_grid(tmp_path / "study")
    dec.main(["--study-root", str(root), "--no-slurm-attestation"])
    assert "# PACLIC v2 accent error decomposition" in capsys.readouterr().out


def test_defaults_point_at_the_v2_maingrid_study(tmp_path):
    """Cell discovery is the main grid's, not the v1 aggregator's."""
    assert dec.DEFAULT_STUDY_ROOT == mg.DEFAULT_STUDY_ROOT
    assert dec.SOURCE_STUDY == mg.STUDY
    assert dec.DEFAULT_OUT_DIR.name == "paclic-v2-accent-error-decomposition"
