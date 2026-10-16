"""Test transliteration: PACLIC 2026 v2 release."""

import unicodedata

import pytest

from scripts.utils.transliteration import (
    ACUTE,
    GRAVE,
    convert_text,
    iso_to_slp1,
    normalize_slp1,
    slp1_to_iso,
    strip_slp1_accents,
    transliterate,
)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ---------- iso_to_slp1 / slp1_to_iso ----------


def test_iso_to_slp1_acute_vowel():
    assert iso_to_slp1("agníṃ") == "agn^iM"


def test_iso_to_slp1_grave_vowel():
    assert iso_to_slp1("vīryàm") == "vIry_am"


def test_iso_to_slp1_long_vowel_acute():
    # ā́  -> ;A  -> ^A
    assert iso_to_slp1("rā́yás") == "r^Ay^as"


def test_iso_to_slp1_diphthong_acute():
    # áu  -> ;O  -> ^O
    assert iso_to_slp1("áu") == "^O"
    # ài  -> :E  -> _E
    assert iso_to_slp1("ài") == "_E"


def test_iso_to_slp1_unaccented_passes_through():
    assert iso_to_slp1("agnim") == "agnim"


def test_iso_to_slp1_accented_false_strips_markers():
    # ISO consonantal ``ḷ`` -> extended SLP1 ``L`` (not ``q``, which is ``ḍ``) (study).
    assert iso_to_slp1("agníṃ īḷé puróhitam", accented=False) == "agniM ILe purohitam"


# ---- retroflex lateral ḷ <-> L round-trip, distinct from ḍ <-> q (study) -------


def test_iso_retroflex_lateral_maps_to_extended_slp1_L():
    assert iso_to_slp1("īḷe") == "ILe"
    assert slp1_to_iso("ILe") == "īḷe"


def test_retroflex_d_still_uses_q():
    assert iso_to_slp1("ḍa") == "qa"
    assert slp1_to_iso("qa") == "ḍa"


def test_retroflex_lateral_and_d_stay_distinct():
    # Both used to collapse onto SLP1 ``q`` and reverse to ``ḍ`` (the study bug).
    assert iso_to_slp1("ḷa ḍa") == "La qa"
    assert slp1_to_iso(iso_to_slp1("ḷa ḍa")) == "ḷa ḍa"


def test_rv_1_1_1_second_word_round_trips():
    text = "agnim īḷe puróhitam"
    assert slp1_to_iso(iso_to_slp1(text)) == text


def test_legacy_transliterate_retroflex_lateral_round_trip():
    assert transliterate("īḷe", "iso", "slp1") == "ILe"
    assert transliterate("ILe", "slp1", "iso") == "īḷe"


# ---- precomposed vocalic ṛ / ṝ -> SLP1 f / F (study) --------------------------


def test_iso_precomposed_vocalic_r_maps_to_f():
    assert iso_to_slp1("ṛṣi", accented=False) == "fzi"
    assert iso_to_slp1("kṛṣṇa", accented=False) == "kfzRa"
    assert iso_to_slp1("gṛhya", accented=False) == "gfhya"


def test_iso_precomposed_long_vocalic_r_maps_to_F():
    assert iso_to_slp1("ṝṣi", accented=False) == "Fzi"


def test_iso_decomposed_vocalic_r_unchanged():
    # The decomposed spelling (r + COMBINING RING BELOW) already worked; keep it.
    assert iso_to_slp1("r̥ṣi", accented=False) == "fzi"


def test_iso_vocalic_r_keeps_accent_marker():
    assert iso_to_slp1("ṛ́ṣi") == "^fzi"
    assert iso_to_slp1("ṝ́ṣi") == "^Fzi"


def test_slp1_to_iso_canonical_markers():
    assert slp1_to_iso("pur^ohitam") == _nfc("puróhitam")


def test_slp1_to_iso_legacy_markers():
    assert slp1_to_iso("pur;ohitam") == _nfc("puróhitam")


def test_slp1_to_iso_canonical_and_legacy_agree():
    assert slp1_to_iso("pur^ohitam") == slp1_to_iso("pur;ohitam")


def test_round_trip_acute_on_short_vowel():
    iso = _nfc("agníṃ")
    assert slp1_to_iso(iso_to_slp1(iso)) == iso


def test_round_trip_acute_on_long_vowel():
    iso = _nfc("rā́yás")
    assert slp1_to_iso(iso_to_slp1(iso)) == iso


def test_round_trip_visarga_identity():
    # SanskritTransliteration maps ``ḥ`` (U+1E25, ISO 15919 visarga),
    # ``ḫ`` (U+1E2B, jihvāmūlīya) and ``ẖ`` (U+1E96) all to SLP1 ``H``,
    # and the reverse path picks the first table entry (``ḫ``).
    # ``slp1_to_iso`` normalizes back to ``ḥ`` so the round-trip is
    # identity for the corpus convention (ISO 15919 visarga).
    for src in ("ḥ", "ḫ"):
        slp1 = iso_to_slp1(src)
        assert slp1_to_iso(slp1) == "ḥ"

    sample = _nfc("agniḥ purohitaḥ yajñásya devám")
    assert slp1_to_iso(iso_to_slp1(sample)) == sample


def test_transliterate_legacy_wrapper_normalises_visarga():
    # ``transliterate`` is a backward-compat wrapper; it should share the
    # ``ḫ → ḥ`` post-processing with ``slp1_to_iso`` so legacy callers
    # don't see a divergent visarga mapping.
    assert transliterate("H", "slp1", "iso") == "ḥ"
    assert transliterate(transliterate("ḥ", "iso", "slp1"), "slp1", "iso") == "ḥ"


# ---------- literal punctuation preservation ----------


def test_literal_colon_preserved_through_iso_to_slp1():
    # The ":" between digits is NOT a grave marker — must survive intact.
    assert iso_to_slp1("3.1.10:13.16") == "3.1.10:13.16"


def test_literal_colon_preserved_through_strip():
    out = strip_slp1_accents("pur^ohitam 3.1.10:13.16 _Iqe")
    assert out == "purohitam 3.1.10:13.16 Iqe"


def test_normalize_slp1_keeps_literal_colon():
    # ";" / ":" before non-vowels are left alone.
    assert normalize_slp1("3.1.10:13.16 ;k") == "3.1.10:13.16 ;k"


def test_normalize_slp1_canonicalises_legacy_accent():
    assert normalize_slp1("pur;ohitam") == "pur^ohitam"


# ---------- anusvara variants ----------


@pytest.mark.parametrize("anusvara", ["ṃ", "ṁ", "m̐"])
def test_anusvara_variants_collapse_to_M(anusvara):
    assert iso_to_slp1(f"agni{anusvara}") == "agniM"


def test_anusvara_round_trip_canonicalises_to_first_match():
    # Documented one-way collapse: all three input variants come back as the
    # first ISO row in the upstream table — currently 'ṃ'.
    for anusvara in ("ṃ", "ṁ", "m̐"):
        slp1 = iso_to_slp1(f"agni{anusvara}")
        assert slp1_to_iso(slp1) == _nfc("agniṃ")


# ---------- dandas ----------


def test_single_danda_preserved():
    assert iso_to_slp1("agnim /") == "agnim /"


def test_double_danda_preserved():
    assert iso_to_slp1("agnim //") == "agnim //"


# ---------- strip_slp1_accents only touches vowel-adjacent markers ----------


def test_strip_slp1_accents_handles_canonical_and_legacy():
    # `^`/`_`/`;`/`:` directly in front of an SLP1 vowel is removed.
    # Colon between digits and underscores in front of consonants stay.
    text = "^aM ;indra:o pur;ohitam 3:13"
    expected = "aM indrao purohitam 3:13"
    assert strip_slp1_accents(text) == expected


# ---------- convert_text dispatcher ----------


def test_convert_text_iso_to_slp1_accented():
    assert convert_text("agníṃ", "iso", "slp1_accented") == "agn^iM"


def test_convert_text_iso_to_slp1_plain():
    assert convert_text("agníṃ", "iso", "slp1") == "agniM"


def test_convert_text_slp1_to_iso():
    assert convert_text("agn^iM", "slp1_accented", "iso") == _nfc("agníṃ")


def test_convert_text_same_scheme_iso_normalises_only():
    assert convert_text("agníṃ", "iso", "iso") == _nfc("agníṃ")


def test_convert_text_unknown_scheme():
    with pytest.raises(ValueError):
        convert_text("x", "foo", "bar")


# ---------- backward-compatible legacy alias ----------


def test_legacy_transliterate_returns_canonical_slp1():
    out = transliterate("agníṃ īḷé puróhitam", "iso", "slp1")
    # Whatever upstream returns, the wrapper must canonicalise legacy markers.
    assert ";" not in out
    assert ":" not in out  # no colon-before-vowel in this input
    assert ACUTE in out


def test_legacy_transliterate_slp1_input_accepts_canonical():
    # Round trip the canonical form back through the legacy entry point.
    assert transliterate("pur^ohitam", "slp1", "iso") == _nfc("puróhitam")


def test_legacy_transliterate_normalises_iso_diphthongs():
    # Regression: the legacy entry point must apply the same ISO normalisation
    # as ``iso_to_slp1`` so accents on the first vowel of a diphthong reach
    # upstream's per-diphthong rows instead of being split into ``^a`` + ``u``.
    assert transliterate("áu", "iso", "slp1") == "^O"
    assert transliterate("ài", "iso", "slp1") == "_E"


def test_legacy_transliterate_matches_iso_to_slp1():
    sample = "prā́cyai diśé svā́hā | tā́v ā́ yātam"
    assert transliterate(sample, "iso", "slp1") == iso_to_slp1(sample)


def test_canonical_marker_constants():
    assert ACUTE == "^"
    assert GRAVE == "_"
