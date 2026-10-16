"""Test prediction metrics: PACLIC 2026 v2 release."""

from __future__ import annotations

import csv
import json

import pytest

from scripts.evaluation.prediction_metrics import (
    DEFAULT_OVER_GENERATION_POLICY,
    EVALUATION_POLICY_VERSION,
    LEGACY_PREDICTION_COLUMNS,
    PREDICTION_COLUMNS,
    PREDICTION_SCHEMA_VERSION,
    RARE_CHARACTER_MAX_COUNT,
    RARE_CHARACTER_MIN_SUPPORT,
    REVIEWED_OVER_GENERATION_POLICY,
    REVIEWED_RARE_CHARACTERS,
    Normalization,
    OverGenerationPolicy,
    PredictionRow,
    PredictionSchemaError,
    evaluate_prediction_file,
    evaluate_prediction_rows,
    iter_edit_events,
    load_prediction_rows,
)


def _row(reference: str, prediction: str, **kwargs) -> PredictionRow:
    return PredictionRow(reference=reference, prediction=prediction, **kwargs)


def _write_csv(path, records, columns=PREDICTION_COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for record in records:
            writer.writerow({c: record.get(c, "") for c in columns})


# ---------- generic OCR + accent (accented Sanskrit, combining marks) ----------


def test_accented_sanskrit_missing_acute_scores_accent_but_not_baseletter():
    rows = [_row("agníṃ", "agniṃ")]  # dropped the udātta acute only
    out = evaluate_prediction_rows(rows)
    # Base letters intact ⇒ stripping accents makes them identical.
    assert out["accent_stripped_cer"] == pytest.approx(0.0)
    # The one accented vowel was not reproduced ⇒ zero accent recall.
    assert out["accent_recall"] == pytest.approx(0.0)
    # Full CER charges the missing acute, so it is non-zero.
    assert out["full_cer"] > 0.0
    assert out["cer"] > 0.0


def test_combining_mark_diacritic_counts_are_alignment_free_presence():
    rows = [_row("agníṃ", "agniṃ")]
    dia = evaluate_prediction_rows(rows)["diacritics"]
    # One combining acute in ref, none in hyp.
    assert dia["acute_udatta"]["ref_count"] == 1
    assert dia["acute_udatta"]["hyp_count"] == 0
    assert dia["acute_udatta"]["recall"] == pytest.approx(0.0)
    assert dia["acute_udatta"]["error_rate"] == pytest.approx(1.0)
    # The anusvāra dot-below survives on both sides ⇒ full recall.
    assert dia["dot_below"]["ref_count"] == 1
    assert dia["dot_below"]["recall"] == pytest.approx(1.0)


def test_perfect_prediction_scores_perfect():
    rows = [_row("agníṃ", "agníṃ"), _row("sóma", "sóma")]
    out = evaluate_prediction_rows(rows)
    assert out["cer"] == pytest.approx(0.0)
    assert out["exact_match"] == pytest.approx(1.0)
    assert out["exact_match_count"] == 2
    assert out["accent_f1"] == pytest.approx(1.0)
    assert out["edit_events"]["substitutions"] == 0
    assert out["edit_events"]["deletions"] == 0
    assert out["edit_events"]["insertions"] == 0


def test_exact_match_count_matches_rate_over_scored_samples():
    # the study: exact_match_count is the numerator of the exact_match rate, over
    # scored (non-empty-reference) samples — so count/num_scored reproduces the rate.
    rows = [
        _row("agníṃ", "agníṃ"),  # exact
        _row("sóma", "sóma"),  # exact
        _row("indra", "indr"),  # mismatch (dropped a char)
        _row("", "junk"),  # empty reference ⇒ degenerate, not scored
    ]
    out = evaluate_prediction_rows(rows)
    assert out["num_samples"] == 4
    assert out["num_scored"] == 3  # the empty-reference row is excluded
    assert out["exact_match_count"] == 2
    assert out["exact_match"] == pytest.approx(2 / 3)
    assert out["exact_match_count"] / out["num_scored"] == pytest.approx(
        out["exact_match"]
    )


# ---------- reliability: empty output, over-generation, decode failures ----------


def test_empty_prediction_counts_as_empty_output():
    rows = [_row("agni", ""), _row("soma", "soma"), _row("indra", "   ")]
    rel = evaluate_prediction_rows(rows)["reliability"]
    assert rel["n_empty_output"] == 2  # "" and whitespace-only
    assert rel["empty_output_rate"] == pytest.approx(2 / 3)


def test_over_generation_flagged_by_length_ratio():
    rows = [_row("a", "aaaa"), _row("soma", "soma")]
    rel = evaluate_prediction_rows(rows)["reliability"]
    # "aaaa" (len 4) > 2.0 * len("a") ⇒ over-generation; "soma" is not.
    assert rel["n_over_generation"] == 1
    assert rel["over_generation_denominator"] == 2
    assert rel["over_generation_rate"] == pytest.approx(0.5)


def test_over_generation_policy_is_configurable():
    rows = [_row("ab", "abcde")]  # len 5 vs 2
    strict = evaluate_prediction_rows(
        rows, over_generation_policy=OverGenerationPolicy(length_ratio=3.0)
    )["reliability"]
    assert strict["n_over_generation"] == 0  # 5 <= 3*2
    loose = evaluate_prediction_rows(
        rows, over_generation_policy=OverGenerationPolicy(length_ratio=1.0)
    )["reliability"]
    assert loose["n_over_generation"] == 1  # 5 > 1*2


def test_over_generation_boundary_is_strict_greater_than():
    # Reviewed criterion (study): flag iff pred_len > length_ratio*ref_len + margin.
    # Exactly at the bound (pred_len == 2*ref_len) is NOT over-generation.
    policy = REVIEWED_OVER_GENERATION_POLICY
    assert policy.is_over_generation(ref_len=3, pred_len=6) is False  # 6 == 2*3
    assert policy.is_over_generation(ref_len=3, pred_len=7) is True  # 7 > 2*3
    assert policy.is_over_generation(ref_len=3, pred_len=5) is False


def test_over_generation_empty_reference_never_flagged_or_counted():
    # Empty references are excluded from both numerator and denominator.
    assert REVIEWED_OVER_GENERATION_POLICY.is_over_generation(0, 999) is False
    rows = [_row("", "a runaway prediction"), _row("soma", "soma")]
    rel = evaluate_prediction_rows(rows)["reliability"]
    assert rel["n_over_generation"] == 0
    assert rel["over_generation_denominator"] == 1  # only the non-empty ref


def test_over_generation_unit_is_nfc_code_points_not_nfd():
    # "é" is one NFC code point (U+00E9) but two NFD (U+0065 U+0301). A 3-char
    # prediction over-generates under the NFC-code-point unit the reviewed policy
    # uses (3 > 2*1) but would NOT under an NFD unit (3 <= 2*2) — pinning the unit.
    rel = evaluate_prediction_rows([_row("é", "xxx")])["reliability"]
    assert rel["n_over_generation"] == 1
    assert rel["over_generation_policy"] == {"length_ratio": 2.0, "abs_margin": 0}


def test_reviewed_over_generation_policy_is_pinned():
    # The reviewed final policy equals the value every committed metrics JSON
    # reports, so finalizing it (study) changes no reported number.
    assert REVIEWED_OVER_GENERATION_POLICY == OverGenerationPolicy(
        length_ratio=2.0, abs_margin=0
    )
    assert REVIEWED_OVER_GENERATION_POLICY == DEFAULT_OVER_GENERATION_POLICY
    assert EVALUATION_POLICY_VERSION == "paclic-2026-v2"


def test_reviewed_rare_character_rule_is_pinned():
    import unicodedata

    # Rule constants versioned in code (not only prose).
    assert RARE_CHARACTER_MAX_COUNT == 10
    assert RARE_CHARACTER_MIN_SUPPORT == 2
    # The claimable injection set is exactly the five NFD grapheme clusters the
    # corpus derivation selects (rare, support >= 2, non-artifact).
    got = {unicodedata.normalize("NFD", c) for c in REVIEWED_RARE_CHARACTERS}
    assert got == {
        "ì",  # ì
        "ī̀",  # ī̀
        "r̥̀",  # r̥̀
        "r̥̄",  # r̥̄
        "ù",  # ù
    }
    # Every entry is a genuine ISO grapheme (all code points are named — no PUA
    # artifact such as U+E807 leaked into the claimable set).
    for grapheme in REVIEWED_RARE_CHARACTERS:
        for cp in unicodedata.normalize("NFD", grapheme):
            assert unicodedata.name(cp)  # raises for unnamed/PUA


def test_decode_failure_rate_from_explicit_metadata():
    rows = [
        _row("agni", "", decode_failed=True, error_type="RuntimeError"),
        _row("soma", "soma"),
    ]
    rel = evaluate_prediction_rows(rows)["reliability"]
    assert rel["n_decode_failed"] == 1
    assert rel["decode_failure_rate"] == pytest.approx(0.5)


# ---------- diacritic + rare-character rates ----------


def test_rare_character_error_rate_with_injected_set():
    rows = [_row("kḷptá", "kptá"), _row("bala", "bala")]  # dropped vocalic ḷ
    out = evaluate_prediction_rows(rows, rare_characters={"ḷ"})
    rare = out["rare_characters"]
    assert rare is not None
    assert rare["ref_count"] == 1
    assert rare["hyp_count"] == 0
    assert rare["recall"] == pytest.approx(0.0)
    assert rare["error_rate"] == pytest.approx(1.0)


def test_rare_characters_absent_when_not_requested():
    assert evaluate_prediction_rows([_row("a", "a")])["rare_characters"] is None


def test_rare_character_recall_when_preserved():
    out = evaluate_prediction_rows([_row("kḷp", "kḷp")], rare_characters={"ḷ"})
    assert out["rare_characters"]["recall"] == pytest.approx(1.0)


# ---------- edit events ----------


def test_edit_events_capture_substitutions_and_deletions():
    rows = [_row("abcd", "abxd"), _row("soma", "som")]  # 1 sub, 1 del
    events = evaluate_prediction_rows(rows)["edit_events"]
    assert events["substitutions"] == 1
    assert events["deletions"] == 1
    assert events["insertions"] == 0
    top = events["top_substitutions"]
    assert {"ref": "c", "hyp": "x", "count": 1} in top


def test_edit_events_handle_empty_prediction_as_deletions():
    events = evaluate_prediction_rows([_row("abc", "")])["edit_events"]
    assert events["deletions"] == 3
    assert events["substitutions"] == 0
    assert events["insertions"] == 0


# ---------- empty-prediction-as-deletion regression  ----------
# The PACLIC error-analysis path (scripts.evaluation.build_paclic_accent_transfer_analysis)
# decomposes errors from iter_edit_events, so an empty or whitespace-only
# prediction must survive normalization as a real prediction and be charged as
# full deletions there too — never silently dropped.


def test_iter_edit_events_empty_prediction_yields_only_deletions():
    events = list(iter_edit_events([PredictionRow(reference="abc", prediction="")]))
    assert [e["op"] for e in events] == ["delete", "delete", "delete"]
    assert [e["ref"] for e in events] == ["a", "b", "c"]
    assert all(e["hyp"] is None for e in events)
    # No spurious inserts/substitutions from the empty side.
    assert not any(e["op"] in ("insert", "substitute") for e in events)


def test_iter_edit_events_whitespace_only_prediction_is_all_deletions():
    # strip=True normalization collapses "   " to "" ⇒ still three deletions,
    # not an empty-vs-empty skip.
    events = list(iter_edit_events([PredictionRow(reference="abc", prediction="   ")]))
    assert [e["op"] for e in events] == ["delete", "delete", "delete"]
    assert all(e["op"] == "delete" and e["hyp"] is None for e in events)


def test_iter_edit_events_empty_prediction_charges_accent_deletion():
    # An accented reference wiped to empty deletes the combining acute too (NFD),
    # so the accent loss is visible as a deletion of U+0301 — the accent-error
    # decomposition depends on this.
    events = list(iter_edit_events([PredictionRow(reference=" á", prediction="")]))
    assert all(e["op"] == "delete" for e in events)
    assert "́" in {e["ref"] for e in events}


def test_evaluate_whitespace_only_prediction_is_full_deletion():
    events = evaluate_prediction_rows([_row("abc", "   ")])["edit_events"]
    assert events["deletions"] == 3
    assert events["substitutions"] == 0
    assert events["insertions"] == 0


# ---------- per-font grouping ----------


def test_per_font_groups_and_scores_independently():
    rows = [
        _row("agni", "agni", font="FontAlpha"),
        _row("soma", "sxma", font="FontAlpha"),
        _row("indra", "indra", font="FontBeta"),
        _row("vayu", "vayu"),  # no font ⇒ excluded from per_font
    ]
    out = evaluate_prediction_rows(rows)
    assert set(out["per_font"]) == {"FontAlpha", "FontBeta"}
    assert out["per_font"]["FontAlpha"]["num_samples"] == 2
    assert out["per_font"]["FontBeta"]["cer"] == pytest.approx(0.0)
    assert out["per_font"]["FontBeta"]["exact_match"] == pytest.approx(1.0)
    # Per-font block carries reliability + diacritics too.
    assert "reliability" in out["per_font"]["FontAlpha"]
    assert "diacritics" in out["per_font"]["FontAlpha"]


# ---------- axes / schema propagation ----------


def test_axes_are_the_constant_row_values():
    rows = [
        _row(
            "a",
            "a",
            condition="trocr_printed",
            fraction=10,
            seed=7,
            view="standard_test",
            split="test",
            schema_version=1,
        ),
        _row(
            "b",
            "b",
            condition="trocr_printed",
            fraction=10,
            seed=7,
            view="standard_test",
            split="test",
            schema_version=1,
        ),
    ]
    out = evaluate_prediction_rows(rows)
    assert out["condition"] == "trocr_printed"
    assert out["fraction"] == 10
    assert out["seed"] == 7
    assert out["view"] == "standard_test"
    assert out["split"] == "test"
    assert out["prediction_schema_version"] == 1
    assert out["is_legacy"] is False


def test_mixed_axis_values_collapse_to_none():
    rows = [_row("a", "a", fraction=10), _row("b", "b", fraction=25)]
    assert evaluate_prediction_rows(rows)["fraction"] is None


# ---------- loading: versioned + legacy ----------


def test_load_versioned_rows_round_trips_typed_fields(tmp_path):
    path = tmp_path / "standard_test.csv"
    _write_csv(
        path,
        [
            {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "sample_id": "ms_test_0",
                "reference": "agni",
                "prediction": "agnl",
                "font": "FontAlpha",
                "condition": "trocr_printed",
                "fraction": 10,
                "seed": 7,
                "view": "standard_test",
                "split": "test",
                "decode_failed": "false",
                "error_type": "",
                "error_message": "",
            }
        ],
    )
    (row,) = load_prediction_rows(path)
    assert row.sample_id == "ms_test_0"
    assert row.fraction == 10 and isinstance(row.fraction, int)
    assert row.seed == 7
    assert row.decode_failed is False
    assert row.is_legacy is False
    assert row.schema_version == PREDICTION_SCHEMA_VERSION


def test_load_legacy_three_column_csv_marks_metadata_missing(tmp_path):
    path = tmp_path / "legacy.csv"
    _write_csv(
        path,
        [{"sample_id": "s0", "reference": "agni", "prediction": "agni"}],
        columns=LEGACY_PREDICTION_COLUMNS,
    )
    (row,) = load_prediction_rows(path)
    assert row.is_legacy is True
    assert row.schema_version is None
    assert row.font is None and row.condition is None and row.fraction is None
    assert row.decode_failed is False


def test_load_rejects_missing_required_column(tmp_path):
    path = tmp_path / "bad.csv"
    _write_csv(
        path,
        [{"sample_id": "s0", "reference": "a"}],
        columns=("sample_id", "reference"),
    )
    with pytest.raises(PredictionSchemaError, match="prediction"):
        load_prediction_rows(path)


def test_load_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(PredictionSchemaError, match="empty"):
        load_prediction_rows(path)


def test_load_rejects_unparseable_int(tmp_path):
    path = tmp_path / "bad_int.csv"
    _write_csv(
        path,
        [{"schema_version": 1, "reference": "a", "prediction": "a", "fraction": "ten"}],
    )
    with pytest.raises(PredictionSchemaError, match="fraction"):
        load_prediction_rows(path)


def test_load_rejects_unsupported_schema_version(tmp_path):
    # A versioned header must carry a *supported* version; a future v2 file must
    # not silently evaluate under v1 reader semantics.
    path = tmp_path / "v2.csv"
    _write_csv(
        path,
        [
            {
                "schema_version": PREDICTION_SCHEMA_VERSION + 1,
                "reference": "a",
                "prediction": "a",
            }
        ],
    )
    with pytest.raises(
        PredictionSchemaError, match="unsupported prediction schema_version"
    ):
        load_prediction_rows(path)


def test_load_rejects_blank_schema_version_when_versioned(tmp_path):
    # Header has schema_version, but the cell is blank ⇒ missing version, reject.
    path = tmp_path / "blank_version.csv"
    _write_csv(path, [{"reference": "a", "prediction": "a"}])  # schema_version → ""
    with pytest.raises(PredictionSchemaError, match="schema_version"):
        load_prediction_rows(path)


def test_load_rejects_header_only_versioned_file(tmp_path):
    path = tmp_path / "header_only.csv"
    _write_csv(path, [])  # v1 header, zero data rows
    with pytest.raises(PredictionSchemaError, match="no prediction rows"):
        load_prediction_rows(path)


def test_load_rejects_header_only_legacy_file(tmp_path):
    path = tmp_path / "header_only_legacy.csv"
    _write_csv(path, [], columns=LEGACY_PREDICTION_COLUMNS)
    with pytest.raises(PredictionSchemaError, match="no prediction rows"):
        load_prediction_rows(path)


# ---------- determinism (no inference, identical JSON on re-evaluation) ----------


def test_reevaluation_is_deterministic(tmp_path):
    path = tmp_path / "standard_test.csv"
    _write_csv(
        path,
        [
            {
                "schema_version": 1,
                "sample_id": "s0",
                "reference": "agníṃ",
                "prediction": "agniṃ",
                "font": "FontAlpha",
                "condition": "c",
                "fraction": 10,
                "seed": 7,
                "view": "standard_test",
                "split": "test",
            },
            {
                "schema_version": 1,
                "sample_id": "s1",
                "reference": "sóma",
                "prediction": "",
                "font": "FontBeta",
                "condition": "c",
                "fraction": 10,
                "seed": 7,
                "view": "standard_test",
                "split": "test",
                "decode_failed": "true",
            },
        ],
    )
    first = evaluate_prediction_file(path, rare_characters={"ṃ"})
    second = evaluate_prediction_file(path, rare_characters={"ṃ"})
    assert json.dumps(first, sort_keys=True, ensure_ascii=False) == json.dumps(
        second, sort_keys=True, ensure_ascii=False
    )


def test_normalization_is_recorded_and_applied():
    # NFC composes the decomposed input so the two rows compare as identical.
    decomposed = "ágni"  # a + combining acute + gni
    composed = "ágni"  # precomposed á + gni
    out = evaluate_prediction_rows(
        [_row(decomposed, composed)], normalization=Normalization(form="NFC")
    )
    assert out["normalization"]["form"] == "NFC"
    assert out["cer"] == pytest.approx(0.0)
    assert out["exact_match"] == pytest.approx(1.0)
