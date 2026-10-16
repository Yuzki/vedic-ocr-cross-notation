"""Test accent metrics: PACLIC 2026 v2 release."""

import unicodedata

import pandas as pd
import pytest

from scripts.evaluation.accent_metrics import (
    compute_accent_metrics,
    compute_all_accent_aware_metrics,
    grapheme_accent_label,
    is_iso_vowel_grapheme,
    iter_grapheme_clusters,
    strip_iso_accents,
)


# ---------- strip_iso_accents ----------


def test_strip_iso_accents_removes_acute():
    assert strip_iso_accents("agníṃ purohitám") == "agniṃ purohitam"


def test_strip_iso_accents_removes_grave():
    assert strip_iso_accents("vīryàm") == "vīryam"


def test_strip_iso_accents_keeps_long_vowel():
    assert strip_iso_accents("rā́yás") == "rāyas"


def test_strip_iso_accents_keeps_vocalic_r():
    # kŕ̥ṣṇa: vocalic r should survive the strip; acute should not.
    out = strip_iso_accents("kŕ̥ṣṇa")
    assert "r̥" in out
    assert "́" not in out


def test_strip_iso_accents_empty_string():
    assert strip_iso_accents("") == ""


# ---------- the study: pinned NFC→NFD→strip(U+0300/U+0301)→NFC policy ----------
# The accent-stripping contract every PACLIC accent-stripped metric relies on:
# decompose the (possibly NFC) input to NFD, remove ONLY the combining acute
# (U+0301, udātta) and grave (U+0300, anudātta), then recompose to NFC. Every
# other combining mark — macron (U+0304), ring-below (U+0325), candrabindu
# (U+0310), dot-below (U+0323) — must survive untouched.


def _nfd_codepoints(text: str) -> list[str]:
    return [f"U+{ord(ch):04X}" for ch in unicodedata.normalize("NFD", text)]


def test_strip_policy_removes_only_acute_and_grave_from_nfc_input():
    # NFC-precomposed 'á' (U+00E1) / 'à' (U+00E0) decompose to base + accent; the
    # accent goes, leaving the bare base with nothing combining in NFD.
    assert strip_iso_accents("á") == "a"
    assert strip_iso_accents("à") == "a"
    assert _nfd_codepoints(strip_iso_accents("á")) == ["U+0061"]
    assert _nfd_codepoints(strip_iso_accents("à")) == ["U+0061"]


def test_strip_policy_keeps_macron_on_long_vowel_with_accent():
    # 'ā́' = a + macron(U+0304) + acute(U+0301); 'ī̀' = i + macron + grave(U+0300).
    # Only the accent is removed — the long-vowel macron survives.
    for text, base in (("ā́", "ā"), ("ī̀", "ī")):
        out = strip_iso_accents(text)
        assert out == base
        nfd = _nfd_codepoints(out)
        assert "U+0304" in nfd  # macron kept
        assert "U+0301" not in nfd and "U+0300" not in nfd  # both accents gone


def test_strip_policy_keeps_non_accent_marks_when_multiple_combine():
    # candrabindu (anunāsika) + acute on 'm': keep the candrabindu, drop the acute.
    out_m = strip_iso_accents("m̐́")
    assert "U+0310" in _nfd_codepoints(out_m)  # COMBINING CANDRABINDU kept
    assert "U+0301" not in _nfd_codepoints(out_m)
    # vocalic-r ring-below + grave: keep the ring-below, drop the grave.
    out_r = strip_iso_accents("r̥̀")
    assert "U+0325" in _nfd_codepoints(out_r)  # COMBINING RING BELOW kept
    assert "U+0300" not in _nfd_codepoints(out_r)


def test_strip_policy_output_is_recomposed_nfc():
    # The contract recomposes to NFC: 'ā́' → the single NFC code point U+0101
    # (LATIN SMALL LETTER A WITH MACRON), not the decomposed a + U+0304.
    out = strip_iso_accents("ā́")
    assert out == unicodedata.normalize("NFC", out)
    assert list(out) == ["ā"]


def test_empty_prediction_is_full_deletion_not_skipped():
    # A blank prediction against a non-empty reference must be charged as a full
    # deletion (CER = 1.0), never silently dropped, with the reference's accent
    # counted as a missed (FN) accent. The accent-stripped reference is still
    # non-empty ('agni'), so its CER is a full deletion too — not a skipped row.
    out = compute_all_accent_aware_metrics(predictions=[""], references=["agní"])
    assert out["full_cer"] == 1.0
    assert out["accent_stripped_cer"] == 1.0
    assert out["accent_recall"] == 0.0
    assert out["accent_fn"] == 1


def test_accent_only_error_zeroes_stripped_cer_but_not_full_cer():
    # End-to-end: when hypothesis and reference differ ONLY in accents, the
    # accent-stripped CER is 0 while the full CER is > 0.
    out = compute_all_accent_aware_metrics(
        predictions=["agni purohitam"], references=["agní puróhitam"]
    )
    assert out["accent_stripped_cer"] == 0.0
    assert out["full_cer"] > 0.0


# ---------- grapheme detection ----------


@pytest.mark.parametrize("grapheme", ["a", "á", "à", "ā", "ī", "í", "r̥", "l̥", "ā́"])
def test_is_iso_vowel_grapheme_true(grapheme):
    assert is_iso_vowel_grapheme(grapheme) is True


@pytest.mark.parametrize("grapheme", ["p", "k", "r", "l", "ṃ", ""])
def test_is_iso_vowel_grapheme_false(grapheme):
    assert is_iso_vowel_grapheme(grapheme) is False


def test_grapheme_accent_label_acute():
    assert grapheme_accent_label("á") == "acute"


def test_grapheme_accent_label_grave():
    assert grapheme_accent_label("à") == "grave"


def test_grapheme_accent_label_none():
    assert grapheme_accent_label("a") is None


def test_grapheme_accent_label_long_with_acute():
    assert grapheme_accent_label("ā́") == "acute"


def test_iter_grapheme_clusters_keeps_vocalic_r():
    # The function decomposes via NFD, so "ṣ"/"ṇ" come back as base + dot;
    # compare semantically by NFC-normalising both sides.
    import unicodedata as _u

    got = [_u.normalize("NFC", g) for g in iter_grapheme_clusters("kr̥ṣṇa")]
    assert got == ["k", "r̥", "ṣ", "ṇ", "a"]


# ---------- compute_accent_metrics: perfect match ----------


def test_perfect_match_f1_one():
    m = compute_accent_metrics(["agníṃ"], ["agníṃ"])
    assert m["unlabeled"]["f1"] == 1.0
    assert m["unlabeled"]["precision"] == 1.0
    assert m["unlabeled"]["recall"] == 1.0
    assert m["unlabeled"]["fp"] == 0
    assert m["unlabeled"]["fn"] == 0
    assert m["labeled"]["f1"] == 1.0


def test_perfect_match_two_accents():
    m = compute_accent_metrics(["agníṃ purohitám"], ["agníṃ purohitám"])
    assert m["unlabeled"]["tp"] == 2
    assert m["unlabeled"]["fp"] == 0
    assert m["unlabeled"]["fn"] == 0
    assert m["unlabeled"]["f1"] == 1.0


# ---------- accent-only mismatch ----------


def test_accent_dropped_in_prediction():
    # gt has 2 acutes, pred has none (baseletters match exactly).
    out = compute_all_accent_aware_metrics(["agniṃ purohitam"], ["agníṃ purohitám"])
    assert out["accent_fn"] == 2
    assert out["accent_fp"] == 0
    assert out["accent_tp"] == 0
    assert out["accent_stripped_cer"] == 0.0
    assert out["full_cer"] > 0.0
    assert out["accent_recall"] == 0.0


def test_accent_introduced_in_prediction():
    # gt has no accents, pred adds one (baseletters match exactly).
    out = compute_all_accent_aware_metrics(["agníṃ purohitam"], ["agniṃ purohitam"])
    assert out["accent_fp"] == 1
    assert out["accent_tp"] == 0
    assert out["accent_fn"] == 0
    assert out["accent_stripped_cer"] == 0.0
    assert out["full_cer"] > 0.0
    assert out["accent_precision"] == 0.0


# ---------- baseletter-only mismatch ----------


def test_baseletter_only_wrong_no_accents():
    out = compute_all_accent_aware_metrics(["agnem"], ["agnim"])
    assert out["accent_tp"] == 0
    assert out["accent_fp"] == 0
    assert out["accent_fn"] == 0
    assert out["full_cer"] > 0.0
    assert out["accent_stripped_cer"] > 0.0
    assert out["accent_f1"] == 0.0  # 0/0 → 0


# ---------- insertion / deletion of accented vowels ----------


def test_pred_insert_accented_vowel():
    # ref: "agni", pred: "agnái" (pred inserted "á")
    out = compute_all_accent_aware_metrics(["agnái"], ["agni"])
    assert out["accent_fp"] == 1
    assert out["accent_fn"] == 0


def test_pred_deletion_of_accented_vowel():
    # ref: "agní", pred: "agn" (pred dropped "í")
    out = compute_all_accent_aware_metrics(["agn"], ["agní"])
    assert out["accent_fn"] == 1
    assert out["accent_fp"] == 0


# ---------- udatta vs anudatta (labeled mode) ----------


def test_labeled_mode_distinguishes_acute_grave():
    # ref: acute on i; pred: grave on i; baseletters match.
    out_un = compute_accent_metrics(["agnìm"], ["agním"])
    out_lab = compute_accent_metrics(["agnìm"], ["agním"])
    # Unlabeled: both sides have *some* accent on matched vowel ⇒ TP.
    assert out_un["unlabeled"]["tp"] == 1
    # Labeled: type mismatch ⇒ FN + FP.
    assert out_lab["labeled"]["tp"] == 0
    assert out_lab["labeled"]["fp"] == 1
    assert out_lab["labeled"]["fn"] == 1


# ---------- edges ----------


def test_empty_pair():
    out = compute_all_accent_aware_metrics([""], [""])
    assert out["accent_tp"] == 0
    assert out["accent_fp"] == 0
    assert out["accent_fn"] == 0


def test_nan_inputs_treated_as_empty():
    # Mirrors evaluate_accent.py's pd.isna guard on raw CSV rows.
    out = compute_all_accent_aware_metrics([pd.NA, "á"], ["á", pd.NA])
    # First pair: pred is empty, ref had 1 acute ⇒ FN + 1 ref accent vowel.
    # Second pair: pred had 1 acute, ref is empty ⇒ FP + 1 hyp accent vowel.
    assert out["accent_fn"] == 1
    assert out["accent_fp"] == 1
    assert out["n_ref_accent_vowels"] == 1
    assert out["n_hyp_accent_vowels"] == 1


def test_empty_ref_counts_hyp_vowels():
    # Regression: hyp accent on empty ref must contribute to n_hyp_accent_vowels.
    out = compute_all_accent_aware_metrics(predictions=["á"], references=[""])
    assert out["accent_fp"] == 1
    assert out["accent_fp_labeled"] == 1
    assert out["n_hyp_accent_vowels"] == 1
    assert out["n_ref_accent_vowels"] == 0
    assert out["accent_precision"] == 0.0


def test_empty_hyp_counts_ref_vowels():
    # Regression: ref accent on empty hyp must contribute to n_ref_accent_vowels.
    out = compute_all_accent_aware_metrics(predictions=[""], references=["á"])
    assert out["accent_fn"] == 1
    assert out["accent_fn_labeled"] == 1
    assert out["n_ref_accent_vowels"] == 1
    assert out["n_hyp_accent_vowels"] == 0
    assert out["accent_recall"] == 0.0


def test_empty_side_counts_multiple_accents():
    # Multi-accent on empty-side guards against off-by-one in the new tally.
    out = compute_all_accent_aware_metrics(
        predictions=[""], references=["agníṃ purohitám"]
    )
    assert out["n_ref_accent_vowels"] == 2
    assert out["accent_fn"] == 2


def test_single_accented_vowel():
    out = compute_all_accent_aware_metrics(["á"], ["á"])
    assert out["accent_tp"] == 1
    assert out["accent_f1"] == 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_accent_metrics(["a"], ["a", "b"])


# ---------- aggregated keys present on results dict ----------


def test_compute_all_accent_aware_metrics_keys():
    out = compute_all_accent_aware_metrics(["agníṃ"], ["agniṃ"])
    expected = {
        "full_cer",
        "accent_stripped_cer",
        "accent_precision",
        "accent_recall",
        "accent_f1",
        "accent_tp",
        "accent_fp",
        "accent_fn",
        "accent_precision_labeled",
        "accent_recall_labeled",
        "accent_f1_labeled",
        "accent_tp_labeled",
        "accent_fp_labeled",
        "accent_fn_labeled",
        "n_ref_accent_vowels",
        "n_hyp_accent_vowels",
    }
    assert expected.issubset(out.keys())


# ---------- regression: realistic Vedic line ----------


def test_real_vedic_sample_all_accents_lost():
    # Pred lost every accent but kept all baseletters (including ā / ṣ / ṃ).
    ref = "rāyás póṣāya suprajās tvā́ya"
    pred = "rāyas poṣāya suprajās tvāya"
    out = compute_all_accent_aware_metrics([pred], [ref])
    assert out["accent_stripped_cer"] == 0.0
    assert out["accent_recall"] == 0.0
    assert out["accent_fn"] >= 3  # at least 3 accented vowels in ref
