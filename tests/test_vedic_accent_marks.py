"""Test vedic accent marks: PACLIC 2026 v2 release."""

from __future__ import annotations

import unicodedata

import pytest

from scripts.data.vedic_accent_marks import (
    PUA_VOCALIC_REPAIRS,
    STROKE_ABOVE,
    UNDERLINE,
    Akshara,
    danda_blocked_decisions,
    find_numeral_insertion_sites,
    marked_aksara_devanagari,
    ms_records,
    place_ms_marks,
    place_rv_marks,
    preprocess_iso,
    render_rv,
    repair_pua_vocalic_liquids,
    rv_records,
    segment_akshara,
    to_devanagari_accented,
    unterminated_pada_initial_runs,
    vowel_length,
)

# The real corpus line for ``008156.png`` (contains a private-use character
# between ``vy`` and ``dhyate``, kept verbatim as in the corpus metadata).
CORPUS_LINE = (
    "śukraḥ paśúbhir vā́ eṣá vy\ue861dhyate yásya sómaḥ skándati vy àsya yajñáś chidyate"
)

# Upstream ``rom2dev.convert`` output for the same line: the *script* conversion
# this module must reproduce exactly, so the rebuild differs from the corpus images
# in the marks only.
CORPUS_LINE_PLAIN_DEVANAGARI = (
    "शुक्रः पशुभिर्वा एष व्य्\ue861ध्यते यस्य सोमः स्कन्दति व्यस्य यज्ञश्छिद्यते"
)

DANDA_LINE = "rāyás póṣāya suprajās tvā́ya । yajamānasya paśúm"

# A real corpus line (stem 006665) with two sites where a ``grave`` akṣara is
# immediately followed by an ``acute`` akṣara.
CORPUS_LINE_006665 = (
    "kyā̀s tā́ḥ prātár yājyā̀ḥ syus | yā́ḥ prātár anuvākyā̀s tā́ḥ sāyám̐ yājyā̀ḥ"
)


def marks(records: list[Akshara]) -> list[tuple[str, str, str, str]]:
    """``(iso, class, mark, rule)`` for the akṣaras, separators skipped."""
    return [(r.iso, r.accent, r.mark, r.rule) for r in records if not r.is_separator]


def mark_of(records: list[Akshara], iso: str) -> str:
    """The mark on the (unique) akṣara whose ISO surface is ``iso``."""
    hits = [r for r in records if not r.is_separator and r.iso == iso]
    assert len(hits) == 1, f"{iso!r} is not unique in {[r.iso for r in records]}"
    return hits[0].mark


# --------------------------------------------------------------------------- #
# Segmentation and input classes
# --------------------------------------------------------------------------- #


def test_input_classes_come_from_the_iso_text() -> None:
    assert segment_akshara("á")[0].accent == "acute"
    assert segment_akshara("à")[0].accent == "grave"
    assert segment_akshara("a")[0].accent == "none"
    # NFC precomposed and NFD decomposed input must agree.
    assert segment_akshara("paśú")[-1].accent == "acute"


def test_segmentation_is_lossless_and_keeps_separators_in_place() -> None:
    line = "rāyás póṣāya tvā́ya । yò 'gníṃ"
    records = segment_akshara(line)
    assert "".join(r.iso for r in records) == preprocess_iso(line)
    kinds = [r.separator_kind for r in records if r.is_separator]
    assert "danda" in kinds
    assert "avagraha" in kinds
    assert "space" in kinds


def test_viramanta_and_conjunct_segmentation() -> None:
    """A consonant run belongs to the following vowel; a trailing run stands alone."""
    records = segment_akshara("skándati yajñáś")
    assert [r.iso for r in records if not r.is_separator] == [
        "ská",
        "nda",
        "ti",
        "ya",
        "jñá",
        "ś",
    ]
    assert [r.has_vowel for r in records if not r.is_separator][-1] is False


def test_private_use_character_survives_as_separator_when_unrepaired() -> None:
    """Unrepaired input keeps the codepoint rather than guessing at it.

    This is the fallback behaviour for any unknown character; corpus text is
    repaired first (see the repair tests below), so it does not rely on this.
    """
    records = segment_akshara(CORPUS_LINE)
    others = [r for r in records if r.separator_kind == "other"]
    assert [r.iso for r in others] == ["\ue861"]


# --------------------------------------------------------------------------- #
# Corpus repair: private-use codepoints for vocalic liquids
# --------------------------------------------------------------------------- #


def test_repair_table_holds_the_specified_codepoints() -> None:
    assert {
        f"U+{ord(pua):04X}": " ".join(f"U+{ord(c):04X}" for c in replacement)
        for pua, replacement in PUA_VOCALIC_REPAIRS.items()
    } == {
        # Word-level corroboration per slot, which is the only evidence that bears
        # on a specific position: pitr̥̄ṇā́ṃ x6/side, vyr̥̀dhyate x22/side,
        # cīkl̥pat x3/side (cīkr̥pat x0).
        "U+E8C7": "U+0072 U+0325 U+0304",  # r̥̄  long vocalic r
        "U+E861": "U+0072 U+0325 U+0300",  # r̥̀  vocalic r + grave
        # Vocalic L, not r. First read as r̥ from the generic r̥ sequence's 4,589
        # occurrences — a figure irrelevant to this slot. 'cīk?pat is the
        # reduplicated aorist of √kl̥p, which this corpus writes only with l̥.
        "U+E807": "U+006C U+0325",  # l̥  vocalic l
    }


def test_repair_reports_what_it_changed_and_leaves_clean_text_alone() -> None:
    repaired, applied = repair_pua_vocalic_liquids(CORPUS_LINE)
    assert "\ue861" not in repaired
    assert [
        (a.index, a.codepoint, a.replacement, a.replacement_codepoints) for a in applied
    ] == [(26, "U+E861", "r\u0325\u0300", "U+0072 U+0325 U+0300")]
    assert repaired[24:30] == "vyr\u0325\u0300d"

    unchanged, none_applied = repair_pua_vocalic_liquids("dev\u0101\u0301 v\u00e0su")
    assert (unchanged, none_applied) == ("dev\u0101\u0301 v\u00e0su", [])


def test_repaired_private_use_codepoint_becomes_an_accent_bearing_aksara() -> None:
    """The repair changes segmentation *and* marks, which is why it must precede them."""
    repaired, _ = repair_pua_vocalic_liquids(CORPUS_LINE)
    records = rv_records(repaired)

    # The separator row is gone: ``vy`` + ``r\u0325\u0300`` is one ak\u1e63ara, class grave.
    assert not [r for r in records if r.separator_kind == "other"]
    merged = [r for r in records if r.iso == "vyr\u0325\u0300"]
    assert len(merged) == 1
    assert (merged[0].accent, merged[0].mark, merged[0].rule) == (
        "grave",
        STROKE_ABOVE,
        "RV3",
    )
    # Under MS the same ak\u1e63ara takes the underline (MS2), where the unrepaired
    # separator took no mark at all.
    ms = [r for r in ms_records(repaired) if r.iso == "vyr\u0325\u0300"]
    assert (ms[0].mark, ms[0].rule) == (UNDERLINE, "MS2")


def test_repair_of_the_long_vocalic_liquid_line() -> None:
    """Corpus line 009590: U+E8C7 -> ``r\u0325\u0304``, giving ``pitr\u0325\u0304\u1e47\u0101\u0301\u1e43``."""
    stored = "s\u00e1 no dad\u0101tu \u015br\u00e1va\u1e47a\u1e43 pit\ue8c7\u1e47\u0101\u0301\u1e43 t\u00e1smai te deva hav\u00ed\u1e63\u0101 vidhema //"
    repaired, applied = repair_pua_vocalic_liquids(stored)
    assert [a.codepoint for a in applied] == ["U+E8C7"]
    assert "pitr\u0325\u0304\u1e47\u0101\u0301\u1e43" in repaired
    records = rv_records(repaired)
    assert not [r for r in records if r.separator_kind == "other"]
    # The repaired vowel is long and unaccented, so it is a ``none`` ak\u1e63ara.
    vocalic = [r for r in records if r.iso == "tr\u0325\u0304"]
    assert len(vocalic) == 1
    assert vocalic[0].accent == "none"


def test_every_repair_target_segments_as_a_vowel() -> None:
    """Each replacement must be an accent-capable vowel, not another unknown."""
    for replacement in PUA_VOCALIC_REPAIRS.values():
        records = segment_akshara(replacement)
        assert len(records) == 1
        assert records[0].has_vowel is True
        assert records[0].is_separator is False


# --------------------------------------------------------------------------- #
# RV rules, in their evaluation order
# --------------------------------------------------------------------------- #


def test_rv1_acute_is_unmarked() -> None:
    records = rv_records("tvā́ ágne yajámānaḥ")
    assert marks(records)[:2] == [
        ("tvā́", "acute", "", "RV1"),
        ("á", "acute", "", "RV1"),
    ]


def test_rv2_none_after_an_acute_takes_stroke_above() -> None:
    records = rv_records("ágne yajñám ihá")
    assert marks(records) == [
        ("á", "acute", "", "RV1"),
        ("gne", "none", STROKE_ABOVE, "RV2"),
        ("ya", "none", UNDERLINE, "RV4"),
        ("jñá", "acute", "", "RV1"),
        ("mi", "none", STROKE_ABOVE, "RV2"),
        ("há", "acute", "", "RV1"),
    ]


def test_rv2_beats_rv4_so_no_collision_exists() -> None:
    """A ``none`` between an acute and an accent is claimed by RV2, not RV4."""
    for line, aksara in (
        ("yásya sómaḥ", "sya"),  # none between two acutes
        ("sómaḥ vàsu", "maḥ"),  # none between an acute and a grave
    ):
        records = rv_records(line)
        assert mark_of(records, aksara) == STROKE_ABOVE
        assert [r.rule for r in records if r.iso == aksara] == ["RV2"]


def test_rv3_grave_takes_stroke_above_even_after_an_acute() -> None:
    """The case the upstream renderer dropped: RV3 is unconditional."""
    records = rv_records("devā́ vàsu bharanti")
    assert mark_of(records, "vā́") == ""  # RV1
    assert mark_of(records, "và") == STROKE_ABOVE
    assert [r.rule for r in records if r.iso == "và"] == ["RV3"]


def test_rv3_applies_at_pada_start_too() -> None:
    records = rv_records("àsya vīryám")
    assert marks(records) == [
        ("à", "grave", STROKE_ABOVE, "RV3"),
        ("sya", "none", "", "RV6"),
        ("vī", "none", UNDERLINE, "RV4"),
        ("ryá", "acute", "", "RV1"),
        ("m", "none", STROKE_ABOVE, "RV2"),
    ]


def test_rv4_underlines_the_last_none_before_an_acute_or_a_grave() -> None:
    records = rv_records("devā́ bharanti vàsu")
    assert marks(records) == [
        ("de", "none", UNDERLINE, "RV4"),
        ("vā́", "acute", "", "RV1"),
        ("bha", "none", STROKE_ABOVE, "RV2"),
        ("ra", "none", "", "RV6"),
        # Terminator is a grave, not an acute: RV4 still applies.
        ("nti", "none", UNDERLINE, "RV4"),
        ("và", "grave", STROKE_ABOVE, "RV3"),
        ("su", "none", "", "RV6"),
    ]


def test_rv5_underlines_every_member_of_a_pada_initial_none_run() -> None:
    records = rv_records("yajamānasya paśúm")
    assert marks(records) == [
        ("ya", "none", UNDERLINE, "RV5"),
        ("ja", "none", UNDERLINE, "RV5"),
        ("mā", "none", UNDERLINE, "RV5"),
        ("na", "none", UNDERLINE, "RV5"),
        ("sya", "none", UNDERLINE, "RV5"),
        # Last member of the run: RV4 fires first, RV5 also applies (same mark).
        ("pa", "none", UNDERLINE, "RV4"),
        ("śú", "acute", "", "RV1"),
        ("m", "none", STROKE_ABOVE, "RV2"),
    ]
    assert [r.also_applicable for r in records if r.iso == "pa"] == [("RV5",)]


def test_rv5_run_terminated_by_a_grave_is_still_fully_underlined() -> None:
    records = rv_records("agnim idàm")
    assert marks(records) == [
        ("a", "none", UNDERLINE, "RV5"),
        ("gni", "none", UNDERLINE, "RV5"),
        ("mi", "none", UNDERLINE, "RV4"),
        ("dà", "grave", STROKE_ABOVE, "RV3"),
        ("m", "none", "", "RV6"),
    ]


def test_rv5_only_covers_the_run_that_opens_the_pada() -> None:
    """A mid-line run is RV4's business: only its last member is underlined."""
    records = rv_records("ágne suprajās tvā́ya")
    assert marks(records) == [
        ("á", "acute", "", "RV1"),
        ("gne", "none", STROKE_ABOVE, "RV2"),
        ("su", "none", "", "RV6"),
        ("pra", "none", "", "RV6"),
        ("jā", "none", UNDERLINE, "RV4"),
        ("stvā́", "acute", "", "RV1"),
        ("ya", "none", STROKE_ABOVE, "RV2"),
    ]


def test_danda_starts_a_new_pada() -> None:
    records = rv_records(DANDA_LINE)
    assert marks(records)[-8:] == [
        ("ya", "none", UNDERLINE, "RV5"),
        ("ja", "none", UNDERLINE, "RV5"),
        ("mā", "none", UNDERLINE, "RV5"),
        ("na", "none", UNDERLINE, "RV5"),
        ("sya", "none", UNDERLINE, "RV5"),
        ("pa", "none", UNDERLINE, "RV4"),
        ("śú", "acute", "", "RV1"),
        ("m", "none", STROKE_ABOVE, "RV2"),
    ]


def test_without_the_danda_reset_only_rv4_fires_after_the_danda() -> None:
    """Contrast case for the pinned behaviour; ``False`` is not the spec."""
    records = rv_records(DANDA_LINE, danda_starts_new_pada=False)
    assert marks(records)[-8:] == [
        ("ya", "none", "", "RV6"),
        ("ja", "none", "", "RV6"),
        ("mā", "none", "", "RV6"),
        ("na", "none", "", "RV6"),
        ("sya", "none", "", "RV6"),
        ("pa", "none", UNDERLINE, "RV4"),
        ("śú", "acute", "", "RV1"),
        ("m", "none", STROKE_ABOVE, "RV2"),
    ]


def test_rv2_does_not_look_back_across_a_danda() -> None:
    """The acute is in the previous pada, so RV2 cannot see it.

    ``ya`` therefore falls through to RV5 with the rest of the pada-initial run.
    """
    records = rv_records("tvā́ । yajamānasya paśúm")
    assert marks(records) == [
        ("tvā́", "acute", "", "RV1"),
        ("ya", "none", UNDERLINE, "RV5"),
        ("ja", "none", UNDERLINE, "RV5"),
        ("mā", "none", UNDERLINE, "RV5"),
        ("na", "none", UNDERLINE, "RV5"),
        ("sya", "none", UNDERLINE, "RV5"),
        ("pa", "none", UNDERLINE, "RV4"),
        ("śú", "acute", "", "RV1"),
        ("m", "none", STROKE_ABOVE, "RV2"),
    ]


def test_line_without_any_accent_gets_no_marks_at_all() -> None:
    """RV5 inherits RV4's terminator requirement, so nothing is marked.

    A fully unaccented line is one whose source text does not mark accents;
    underlining every akṣara would assert "every syllable is low" instead of
    "unknown".
    """
    records = rv_records("yajamānasya paśum")
    assert {r.mark for r in records} == {""}
    assert {r.rule for r in records if not r.is_separator} == {"RV6"}


def test_pada_initial_run_terminated_by_a_danda_is_not_marked() -> None:
    """The run must meet an accent *inside its own pada*.

    The first pada has no accent at all, so none of it is marked; the second pada
    has one, so it is marked normally.
    """
    records = rv_records("yajamānasya । rāyás póṣāya")
    assert marks(records) == [
        # First pada: run reaches the daṇḍa without meeting an accent.
        ("ya", "none", "", "RV6"),
        ("ja", "none", "", "RV6"),
        ("mā", "none", "", "RV6"),
        ("na", "none", "", "RV6"),
        ("sya", "none", "", "RV6"),
        # Second pada: normal marking.
        ("rā", "none", UNDERLINE, "RV4"),
        ("yá", "acute", "", "RV1"),
        ("spó", "acute", "", "RV1"),
        ("ṣā", "none", STROKE_ABOVE, "RV2"),
        ("ya", "none", "", "RV6"),
    ]


def test_unterminated_pada_initial_runs_reports_what_rv5_declines() -> None:
    """The runs RV5 leaves alone, as inclusive akṣara-index pairs."""
    # Whole line unaccented: one run covering all eight akṣaras.
    assert unterminated_pada_initial_runs(segment_akshara("yajamānasya paśum")) == [
        (0, 7)
    ]
    # First pada ends at the daṇḍa without an accent; second pada is fine.
    assert unterminated_pada_initial_runs(
        segment_akshara("yajamānasya । rāyás póṣāya")
    ) == [(0, 4)]
    # Terminated inside the pada: nothing to report.
    assert unterminated_pada_initial_runs(segment_akshara("yajamānasya paśúm")) == []
    # Pada does not open on a none akṣara: nothing to report.
    assert unterminated_pada_initial_runs(segment_akshara("ágne yajñám ihá")) == []


def test_rv4_does_not_look_forward_across_a_danda() -> None:
    """One boundary notion for every rule: RV4 stops at the daṇḍa too.

    ``na`` ends an unaccented pada. The accent it would have keyed off is in the
    *next* pada, so neither RV4 nor RV5 can mark it.
    """
    records = rv_records("yajamāna । ágne")
    assert marks(records) == [
        ("ya", "none", "", "RV6"),
        ("ja", "none", "", "RV6"),
        ("mā", "none", "", "RV6"),
        ("na", "none", "", "RV6"),
        ("á", "acute", "", "RV1"),
        ("gne", "none", STROKE_ABOVE, "RV2"),
    ]


def test_double_danda_is_a_boundary_too() -> None:
    """``॥`` blocks the same lookups as ``।``; the expert was certain of this."""
    assert marks(rv_records("yajamāna ॥ ágne")) == [
        ("ya", "none", "", "RV6"),
        ("ja", "none", "", "RV6"),
        ("mā", "none", "", "RV6"),
        ("na", "none", "", "RV6"),
        ("á", "acute", "", "RV1"),
        ("gne", "none", STROKE_ABOVE, "RV2"),
    ]
    assert marks(rv_records("tvā́ ॥ yajamānasya paśúm"))[:2] == [
        ("tvā́", "acute", "", "RV1"),
        ("ya", "none", UNDERLINE, "RV5"),
    ]


def test_danda_transparency_contrast_is_not_the_spec() -> None:
    """``danda_starts_new_pada=False`` makes daṇḍas transparent to every rule."""
    transparent = rv_records("yajamāna । ágne", danda_starts_new_pada=False)
    assert marks(transparent)[3] == ("na", "none", UNDERLINE, "RV4")
    transparent = rv_records("tvā́ । yajamānasya paśúm", danda_starts_new_pada=False)
    assert marks(transparent)[1] == ("ya", "none", STROKE_ABOVE, "RV2")


def test_danda_blocked_decisions_reports_the_behavioural_delta() -> None:
    """Both sides of a boundary are reported, with the rule and mark forgone."""
    blocked = danda_blocked_decisions(rv_records("yajamāna । ágne"))
    assert [
        (b.aksara_index, b.separator_kind, b.blocked_rule, b.would_have_been)
        for b in blocked
    ] == [(3, "danda", "RV4", UNDERLINE)]

    blocked = danda_blocked_decisions(rv_records("tvā́ । yajamānasya paśúm"))
    assert [
        (b.aksara_index, b.separator_kind, b.blocked_rule, b.would_have_been)
        for b in blocked
    ] == [(1, "danda", "RV2", STROKE_ABOVE)]

    # A boundary with no accent on either side blocks nothing.
    assert danda_blocked_decisions(rv_records("yajamāna । yajamāna")) == []
    # Double daṇḍa is reported under its own kind.
    assert [
        b.separator_kind for b in danda_blocked_decisions(rv_records("yajamāna ॥ ágne"))
    ] == ["double_danda"]


# --------------------------------------------------------------------------- #
# MS rules
# --------------------------------------------------------------------------- #


def test_ms_is_purely_local() -> None:
    records = ms_records("yajamānasya paśúm")
    assert marks(records) == [
        ("ya", "none", "", "MS3"),
        ("ja", "none", "", "MS3"),
        ("mā", "none", "", "MS3"),
        ("na", "none", "", "MS3"),
        ("sya", "none", "", "MS3"),
        ("pa", "none", "", "MS3"),
        ("śú", "acute", STROKE_ABOVE, "MS1"),
        ("m", "none", "", "MS3"),
    ]


@pytest.mark.parametrize(
    "line", [CORPUS_LINE, DANDA_LINE, "agnim idàm", "àsya vīryám", "yajamānasya paśum"]
)
def test_every_ms_mark_is_a_function_of_that_aksaras_class_alone(line: str) -> None:
    expected = {"none": "", "acute": STROKE_ABOVE, "grave": UNDERLINE}
    for record in ms_records(line):
        if record.is_separator:
            assert record.mark == ""
        else:
            assert record.mark == expected[record.accent]


# --------------------------------------------------------------------------- #
# RV vs MS contrast — the assertion that would have caught the original confusion
# --------------------------------------------------------------------------- #


def test_rv_and_ms_put_the_two_marks_in_opposite_places() -> None:
    line = "devā́ bharanti vàsu"
    rv = rv_records(line)
    ms = ms_records(line)

    # Same segmentation, different assignments.
    assert [r.iso for r in rv] == [r.iso for r in ms]
    assert [r.mark for r in rv] != [r.mark for r in ms]

    # acute: unmarked in RV, STROKE_ABOVE in MS.
    assert mark_of(rv, "vā́") == ""
    assert mark_of(ms, "vā́") == STROKE_ABOVE
    # none right after an acute: STROKE_ABOVE in RV, unmarked in MS.
    assert mark_of(rv, "bha") == STROKE_ABOVE
    assert mark_of(ms, "bha") == ""
    # grave: STROKE_ABOVE in RV, UNDERLINE in MS — same code points, opposite use.
    assert mark_of(rv, "và") == STROKE_ABOVE
    assert mark_of(ms, "và") == UNDERLINE


# --------------------------------------------------------------------------- #
# Numeral insertion: detected as evidence, never emitted
# --------------------------------------------------------------------------- #


def test_numeral_insertion_sites_are_detected_positionally() -> None:
    sites = find_numeral_insertion_sites(rv_records(CORPUS_LINE_006665))
    assert [(s.aksara_index, s.grave_iso, s.vowel_length) for s in sites] == [
        (0, "kyā̀", "long"),
        (14, "kyā̀", "long"),
    ]
    assert {s.numeral_if_inserted for s in sites} == {"३"}


def test_numeral_is_never_emitted_into_the_rendered_string() -> None:
    rendered = render_rv(CORPUS_LINE_006665)
    assert find_numeral_insertion_sites(rv_records(CORPUS_LINE_006665))
    assert "१" not in rendered
    assert "३" not in rendered


def test_a_grave_before_an_acute_takes_stroke_above_only() -> None:
    """The reading implemented for the point the sources disagree on."""
    records = rv_records(CORPUS_LINE_006665)
    first = [r for r in records if not r.is_separator][0]
    assert (first.accent, first.mark, first.rule) == ("grave", STROKE_ABOVE, "RV3")


def test_vowel_length_classification() -> None:
    assert vowel_length("à") == "short"
    assert vowel_length("ì") == "short"
    assert vowel_length("ā̀") == "long"
    assert vowel_length("è") == "long"
    assert vowel_length("aù") == "long"
    assert vowel_length("") == "none"


# --------------------------------------------------------------------------- #
# The real corpus line, and the single-record-list regression
# --------------------------------------------------------------------------- #


EXPECTED_CORPUS_RV = [
    ("śu", "none", UNDERLINE, "RV5"),
    ("kraḥ", "none", UNDERLINE, "RV5"),
    ("pa", "none", UNDERLINE, "RV4"),
    ("śú", "acute", "", "RV1"),
    ("bhi", "none", STROKE_ABOVE, "RV2"),
    ("rvā́", "acute", "", "RV1"),
    ("e", "none", STROKE_ABOVE, "RV2"),
    ("ṣá", "acute", "", "RV1"),
    ("vy", "none", STROKE_ABOVE, "RV2"),
    ("dhya", "none", "", "RV6"),
    ("te", "none", UNDERLINE, "RV4"),
    ("yá", "acute", "", "RV1"),
    ("sya", "none", STROKE_ABOVE, "RV2"),
    ("só", "acute", "", "RV1"),
    ("maḥ", "none", STROKE_ABOVE, "RV2"),
    ("ská", "acute", "", "RV1"),
    ("nda", "none", STROKE_ABOVE, "RV2"),
    ("ti", "none", UNDERLINE, "RV4"),
    ("vyà", "grave", STROKE_ABOVE, "RV3"),
    ("sya", "none", "", "RV6"),
    ("ya", "none", UNDERLINE, "RV4"),
    ("jñá", "acute", "", "RV1"),
    ("śchi", "none", STROKE_ABOVE, "RV2"),
    ("dya", "none", "", "RV6"),
    ("te", "none", "", "RV6"),
]


def test_corpus_line_rv_assignment() -> None:
    assert marks(rv_records(CORPUS_LINE)) == EXPECTED_CORPUS_RV


def test_corpus_line_ms_assignment_marks_only_the_accented_aksaras() -> None:
    records = ms_records(CORPUS_LINE)
    assert [(r.iso, r.mark, r.rule) for r in records if r.mark] == [
        ("śú", STROKE_ABOVE, "MS1"),
        ("rvā́", STROKE_ABOVE, "MS1"),
        ("ṣá", STROKE_ABOVE, "MS1"),
        ("yá", STROKE_ABOVE, "MS1"),
        ("só", STROKE_ABOVE, "MS1"),
        ("ská", STROKE_ABOVE, "MS1"),
        ("vyà", UNDERLINE, "MS2"),
        ("jñá", STROKE_ABOVE, "MS1"),
    ]


def test_marks_do_not_disturb_the_script_conversion() -> None:
    """Stripping the two marks recovers the upstream Devanagari exactly.

    This pins that the rebuild differs from the existing corpus images in the marks
    *only*, not in the script conversion.
    """
    for text in (
        render_rv(CORPUS_LINE),
        to_devanagari_accented(ms_records(CORPUS_LINE)),
    ):
        stripped = "".join(ch for ch in text if ch not in (STROKE_ABOVE, UNDERLINE))
        assert stripped == CORPUS_LINE_PLAIN_DEVANAGARI


@pytest.mark.parametrize(
    "line",
    [
        CORPUS_LINE,
        DANDA_LINE,
        CORPUS_LINE_006665,
        "yajamānasya paśúm",
        "àsya vīryám",
        "sómaḥ vàsu",
        "agnim idàm",
        "yò 'gníṃ cikyānó mányeta",
    ],
)
def test_single_record_list_regression(line: str) -> None:
    """No second array can creep back in: one list, same length, same order.

    The upstream bug was ``mark[i]`` (indexed by the vowel array) attached to
    ``mats[i]`` (indexed by the grapheme-cluster array), 24 vs 65 entries on a real
    line. Here segmentation length must equal the marked length for *both* systems,
    every record must keep its own ISO surface and unit type, and every akṣara must
    have had a rule considered for it (no silent skips).
    """
    records = segment_akshara(line)
    n_aksaras = sum(1 for r in records if not r.is_separator)
    for marked in (place_rv_marks(records), place_ms_marks(records)):
        assert len(marked) == len(records)
        assert [r.iso for r in marked] == [r.iso for r in records]
        assert [r.is_separator for r in marked] == [r.is_separator for r in records]
        assert [r.accent for r in marked] == [r.accent for r in records]
        assert all(r.rule for r in marked)
        assert (
            sum(1 for r in marked if not r.is_separator and r.rule != "SEP")
            == n_aksaras
        )
        assert all(r.mark == "" for r in marked if r.is_separator)
        assert all(r.mark in ("", STROKE_ABOVE, UNDERLINE) for r in marked)


# --------------------------------------------------------------------------- #
# WHERE inside the akṣara the mark goes
#
# The mark must come last in the cluster: that is the only order the Devanagari
# shaping model accepts. Moving it before a trailing visarga / anusvāra — which is
# where it would have to go to sit visually on the syllable — makes the cluster
# ill-formed and HarfBuzz inserts a dotted circle (measured in all five repo fonts).
# The expert's report that the mark lands on the visarga is a font-positioning
# problem, not something the code-point order can fix. It was resolved by rendering
# v2 in Noto, which places the mark inside the akṣara's core on every visarga
# cluster in the corpus; see render_font_decision in the rendered bundle.
# --------------------------------------------------------------------------- #

VISARGA = "ः"  # U+0903, a SPACING mark (category Mc)
ANUSVARA = "ं"  # U+0902
CANDRABINDU = "ँ"  # U+0901
DOTTED_CIRCLE = "◌"  # U+25CC, what a shaper emits for an ill-formed cluster


def codepoints(text: str) -> list[str]:
    return [f"U+{ord(ch):04X}" for ch in text]


def test_cluster_order_is_shapeable_visarga_then_mark() -> None:
    marked = marked_aksara_devanagari(
        Akshara(iso="kraḥ", accent="none", mark=UNDERLINE, rule="RV5")
    )
    assert codepoints(marked) == [
        "U+0915",  # KA
        "U+094D",  # VIRAMA
        "U+0930",  # RA
        "U+0903",  # VISARGA
        "U+0952",  # tone mark last
    ]


@pytest.mark.parametrize("mark", [STROKE_ABOVE, UNDERLINE])
@pytest.mark.parametrize("iso,trailing", [("maḥ", VISARGA), ("maṃ", ANUSVARA)])
def test_trailing_signs_precede_the_mark_for_both_marks(
    mark: str, iso: str, trailing: str
) -> None:
    marked = marked_aksara_devanagari(Akshara(iso=iso, accent="none", mark=mark))
    assert marked.index(trailing) < marked.index(mark)
    assert marked.endswith(mark)


def test_vowel_bearing_core_is_identified_by_unicode_classification() -> None:
    """The core a compositing fix would attach the mark to, kept in one place."""
    from scripts.data.vedic_accent_marks import _carries_the_syllables_vowel

    for char, expected in (
        ("क", True),  # base letter
        ("ा", True),  # dependent vowel sign (Mc)
        ("ृ", True),  # dependent vowel sign (Mn)
        ("्", True),  # virama
        ("़", True),  # nukta
        (VISARGA, False),  # syllable modifier, not part of the vowel
        (ANUSVARA, False),
        (CANDRABINDU, False),
        (STROKE_ABOVE, False),
        (UNDERLINE, False),
    ):
        assert _carries_the_syllables_vowel(char) is expected, char


def test_no_dotted_circle_reaches_the_rendered_string() -> None:
    """An ill-formed cluster would show up as U+25CC once shaped."""
    for line in (CORPUS_LINE, CORPUS_LINE_006665, "śukraḥ sómaḥ vàsu", "vasuṃ sómaṃ"):
        repaired, _ = repair_pua_vocalic_liquids(line)
        rendered = render_rv(repaired)
        assert DOTTED_CIRCLE not in rendered
        assert to_devanagari_accented(ms_records(repaired)).find(DOTTED_CIRCLE) == -1


def test_marks_sit_at_the_end_of_their_own_cluster_on_the_corpus_line() -> None:
    """Each mark still belongs to its own akṣara: it never crosses into the next."""
    repaired, _ = repair_pua_vocalic_liquids(CORPUS_LINE)
    rendered = render_rv(repaired)
    assert "क्र" + VISARGA + UNDERLINE in rendered  # शु॒क्रः॒
    assert "म" + VISARGA + STROKE_ABOVE in rendered  # सोमः॑
    # No mark ever lands on a following base letter's cluster.
    for index, char in enumerate(rendered):
        if char in (STROKE_ABOVE, UNDERLINE):
            assert index > 0
            assert unicodedata.category(rendered[index - 1]).startswith(("L", "M"))


def test_rendered_string_appends_each_mark_to_its_own_aksara() -> None:
    records = rv_records("devā́ vàsu")
    # ``de`` is underlined (RV4) and ``và`` takes the stroke (RV3); both marks sit
    # directly after their own akṣara and the acute akṣara between them is bare.
    assert to_devanagari_accented(records) == (
        "दे" + UNDERLINE + "वा व" + STROKE_ABOVE + "सु"
    )
