"""Build paclic v2 accent error decomposition: PACLIC 2026 v2 release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import sys
import unicodedata
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jiwer

from ..utils.transliteration import convert_text
from . import aggregate_accent_ratio_multiseed as agg
from . import build_paclic_v2_maingrid_analysis as mg
from .accent_metrics import (
    ACUTE,
    GRAVE,
    AccentLabel,
    _charge_orphan_hyp,
    _charge_orphan_ref,
    _charge_paired,
    _coerce_text,
    _new_counts,
    _prepare_for_jiwer,
    _vowel_positions,
    compute_accent_metrics,
)
from .build_paclic_accent_transfer_analysis import direction_of
from .prediction_metrics import (
    DEFAULT_NORMALIZATION,
    Normalization,
    load_prediction_rows,
)

# --------------------------------------------------------------------------- #
# Study identity.
# --------------------------------------------------------------------------- #
#: Study slug, used in log prefixes and in the emitted artifacts.
STUDY: str = "paclic-v2-accent-error-decomposition"
#: The study whose cells are decomposed. Not re-run, not re-trained, not re-decoded.
SOURCE_STUDY: str = mg.STUDY
#: Issue this analysis closes the error-decomposition half of.
ISSUE: int = 125

DEFAULT_STUDY_ROOT = mg.DEFAULT_STUDY_ROOT
DEFAULT_SHARED_STUDY_ROOT = mg.DEFAULT_SHARED_STUDY_ROOT
DEFAULT_SLURM_LOG_DIR = mg.DEFAULT_SLURM_LOG_DIR
DEFAULT_OUT_DIR = Path(
    "results/"
    "paclic-v2-accent-error-decomposition"
)
#: The two test views every cell is evaluated on, in report order.
VIEWS: tuple[str, ...] = ("rv", "ms")
#: Target number of decomposed views: 5 ratios × 3 seeds × 2 views.
TARGET_N_VIEWS: int = len(mg.RV_RATIOS) * len(mg.SEEDS) * len(VIEWS)


# --------------------------------------------------------------------------- #
# Text transform — the runner's, so the committed counts are reproducible.
# --------------------------------------------------------------------------- #
def _no_generic_metrics(_csv_path: Path) -> dict[str, Any]:
    """Metrics stand-in for the shared cell reader: this analysis needs none of them.

    ``read_cell_dir`` re-sources exact-match (and chrF, when missing) from every
    prediction CSV through the study evaluator, whether or not a cross-check was
    asked for. This bundle reports no generic string metric: the committed accent
    counts it reconciles against are read straight out of ``results.json``, and the
    study recompute plus its cross-check belong to — and are published by — the
    main-grid bundle. Injecting this keeps that work from being done twice and keeps
    the module's only heavy dependency the accent aligner it actually uses.
    """
    return {}


def normalize_runner_iso(text: str) -> str:
    """The transform the runner applied before computing the committed metrics.

    ``experiment_accent_ratio.evaluate_on_dataset`` calls
    ``convert_text(text, "iso", "iso")`` (= ``transliteration.normalize_iso``: NFC,
    the diphthong-accent shift ``ái`` → ``aí``, and the precomposed-vocalic-``r̥``
    rewrite) on **both** sides before ``compute_all_accent_aware_metrics``, but saves
    the *raw* strings to ``{rv,ms}_predictions.csv``. Re-reading the CSVs without
    this transform therefore does not reproduce ``results.json``; with it, it
    reproduces the accent counts exactly.
    """
    return convert_text(text, "iso", "iso")


#: Human-readable label for the above, quoted in the artifacts.
RUNNER_TRANSFORM_LABEL: str = (
    "scripts.utils.transliteration.normalize_iso (via convert_text(text, 'iso', "
    "'iso')): NFC + diphthong-accent shift + precomposed-vocalic-r rewrite — the "
    "transform experiment_accent_ratio.evaluate_on_dataset applied before writing "
    "the committed accent counts"
)
#: The study evaluator's convention, reported for comparison and never used here.
ALT_TRANSFORM_LABEL: str = (
    "scripts.evaluation.prediction_metrics.DEFAULT_NORMALIZATION (NFC + strip, no "
    "diphthong shift) — the study evaluator's convention; reported per view as "
    "alt_normalization, not used for the decomposition"
)


# --------------------------------------------------------------------------- #
# Classes.
# --------------------------------------------------------------------------- #
#: The accent-channel classes, in report order. The first four partition the
#: reference's accents; the first two plus the last two partition the hypothesis's.
ERROR_CLASSES: tuple[str, ...] = (
    "matched_same_kind",
    "matched_other_kind",
    "no_mark_at_read_syllable",
    "ref_syllable_misread",
    "extra_at_read_syllable",
    "extra_at_misread_syllable",
)
#: Reference-side partition (sums to ``n_ref_accent_vowels``).
REF_SIDE_CLASSES: tuple[str, ...] = (
    "matched_same_kind",
    "matched_other_kind",
    "no_mark_at_read_syllable",
    "ref_syllable_misread",
)
#: Hypothesis-side partition (sums to ``n_hyp_accent_vowels``).
HYP_SIDE_CLASSES: tuple[str, ...] = (
    "matched_same_kind",
    "matched_other_kind",
    "extra_at_read_syllable",
    "extra_at_misread_syllable",
)
#: Classes that are errors at all (everything but the one non-error class).
ERROR_ONLY_CLASSES: tuple[str, ...] = tuple(
    c for c in ERROR_CLASSES if c != "matched_same_kind"
)

#: Prose definition of each class, rendered to ``tables/class_definitions.md`` and
#: carried in the manifest, so a reader never has to infer a class from its name.
CLASS_DEFINITIONS: dict[str, str] = {
    "matched_same_kind": (
        "The aligned hypothesis syllable carries a mark of the same kind. The only "
        "non-error class; equals the committed accent_tp_labeled."
    ),
    "matched_other_kind": (
        "The aligned hypothesis syllable carries a mark of the OTHER kind (acute "
        "where grave belongs, or the reverse). Counted as a true positive by the "
        "unlabelled accent F1 and as both a false negative and a false positive by "
        "the labelled one — which is the entire difference between accent_f1 and "
        "accent_f1_labeled, lifted out here rather than recomputed."
    ),
    "no_mark_at_read_syllable": (
        "The aligned hypothesis position is the SAME base code point and carries no "
        "mark: the syllable was read, the mark was not written. Named for what is "
        "observed at the position and deliberately NOT called 'omitted': whether the "
        "mark is missing from the output or present elsewhere in it is not decidable "
        "from one position — read hyp_marks_per_ref_accent and the misplacement "
        "refinement for that."
    ),
    "ref_syllable_misread": (
        "The aligned hypothesis position is a different code point, or the reference "
        "character has no counterpart at all (a deletion). The letter itself is "
        "wrong, so no accent judgement about that syllable is meaningful — reported, "
        "never credited."
    ),
    "extra_at_read_syllable": (
        "A hypothesis mark on a syllable whose base code point matches the "
        "reference's and which the reference leaves unaccented: a mark invented on a "
        "correctly read letter."
    ),
    "extra_at_misread_syllable": (
        "A hypothesis mark on a syllable whose base code point differs from the "
        "reference's, or which the hypothesis inserted outright."
    ),
}

#: Diagnostic sub-counts. Each is a SUBSET of a class above (never added to it), so
#: the partitions stay exact; the note says which.
DIAGNOSTIC_COUNTS: dict[str, str] = {
    "hyp_marks_per_ref_accent": (
        "n_hyp_accent_vowels / n_ref_accent_vowels: how many marks the model wrote "
        "per mark the reference has. Not a class — the number that decides whether a "
        "channel with a low F1 is writing FEWER marks (a channel that went quiet) or "
        "about as many in the wrong places (a channel that went random). Reading a "
        "low accent F1 as omission without it is an inference the data need not "
        "support."
    ),
    "matched_on_substituted_base": (
        "Subset of matched_same_kind + matched_other_kind: the mark matched although "
        "the base letter did not, so the accent channel is credited for a syllable "
        "the base channel got wrong."
    ),
    "n_ref_accent_marks_nfd": (
        "Every U+0301 / U+0300 in the NFD reference, whether or not it sits on a "
        "vowel. NOT the decomposition's denominator — see letter_s_acute_note."
    ),
    "n_ref_accent_marks_on_letter_s": (
        "Subset of n_ref_accent_marks_nfd: U+0301 marks that are the second half of "
        "the ISO letter ś (s + U+0301), i.e. consonants, not accents."
    ),
    "n_ref_accent_marks_unanchored_other": (
        "n_ref_accent_marks_nfd minus n_ref_accent_vowels minus "
        "n_ref_accent_marks_on_letter_s: NFD accent code points that are neither "
        "vowel-anchored nor part of an ś. Expected 0; reported so it cannot be zero "
        "by assumption."
    ),
}

#: Why the decomposition's reference denominator is the vowel-anchored count and not
#: the NFD accent-code-point count some earlier artifacts divide by.
LETTER_S_ACUTE_NOTE: str = (
    "U+0301 is both the high-accent mark and the second half of the ISO letter ś "
    "(NFC U+015B decomposes to s + U+0301). A count of 'accent marks' taken over the "
    "NFD reference therefore includes every ś in the corpus, and an accent-vs-base "
    "split that tests membership of the edited code point in an accent set charges "
    "ś edits to the accent channel. The accent channel used here is anchored on "
    "vowel bases (accent_metrics._vowel_positions), so ś is structurally excluded — "
    "which is also why the reference denominator (n_ref_accent_vowels) is smaller "
    "than the NFD accent-mark count. Both are reported per view, with the ś share "
    "measured, so the gap is evidence rather than a claim."
)

#: How far apart, in syllable slots, an unmatched reference mark and an unmatched
#: hypothesis mark may sit and still be paired as one misplacement. 1 = the adjacent
#: syllable, which is the failure the corpus rebuild was about. Pinned as policy:
#: the partition does not depend on it, only the refinement does, and
#: :data:`MISPLACEMENT_SENSITIVITY_WINDOWS` reports the alternatives.
MISPLACEMENT_WINDOW: int = 1
#: Windows the refinement is additionally reported at, so the pinned choice is
#: visibly a choice.
MISPLACEMENT_SENSITIVITY_WINDOWS: tuple[int, ...] = (0, 1, 2, 3)

#: Why the window-0 row of the sensitivity block is degenerate. Stated wherever that
#: block is offered, because its excess comes out large and negative and would
#: otherwise read as a defect in the decomposition.
WINDOW_ZERO_NOTE: str = (
    "Window 0 is a degenerate control, not a smaller version of the others: a "
    "hypothesis mark in the SAME slot as an unmatched reference mark would have been "
    "scored a match, so same-slot pairs are structurally almost impossible in the "
    "observed data while the rotated null is free to produce them. Its excess is "
    "therefore large and negative by construction and carries no information; the "
    "informative windows are 1 and above."
)


# --------------------------------------------------------------------------- #
# The accent alignment, walked here — because accent_metrics must not be edited.
# --------------------------------------------------------------------------- #
# The alignment walk is restated here to preserve the shared metric API. It
# therefore lives here and reuses that module's primitives **unchanged**: its
# normalization (``_prepare_for_jiwer`` / ``_coerce_text``), its channel definition
# (``_vowel_positions``), and — critically — its charging rule (``_charge_paired`` /
# ``_charge_orphan_ref`` / ``_charge_orphan_hyp`` over ``_new_counts``). Only the
# chunk enumeration is restated, and :func:`self_check_counts` verifies per view that
# the restated walk charges *exactly* what ``compute_accent_metrics`` charges, so the
# duplication cannot drift silently. Those five names are private, and reaching for
# them is deliberate coupling: it is the price of not touching a locked file, and it
# is checked rather than trusted.
@dataclass(frozen=True)
class _Chunk:
    """Stand-in for ``jiwer.process.AlignmentChunk`` on the degenerate pairs.

    ``jiwer.process_characters`` rejects empty strings, so a row with an empty
    reference (or an empty hypothesis) never reaches the aligner. Its alignment is
    nonetheless fully determined — one whole-side insert (or delete) — and expressing
    it as a chunk lets those rows travel the same path as the aligned ones, which is
    how ``compute_accent_metrics`` charges them too (it loops the whole side through
    the orphan chargers).
    """

    type: str
    ref_start_idx: int
    ref_end_idx: int
    hyp_start_idx: int
    hyp_end_idx: int


@dataclass(frozen=True)
class AccentEvent:
    """One accent-relevant position of one aligned (reference, hypothesis) pair.

    Emitted for every alignment position where **either** side is a vowel base — the
    exact set of positions ``compute_accent_metrics`` charges TP/FP/FN on, since
    ``_charge_paired`` returns early when neither side is a vowel.

    ``op`` is the position's alignment role: ``paired`` (an ``equal`` or
    ``substitute`` chunk, so both sides have an index), ``orphan_ref`` (a ``delete``
    chunk — the reference character has no counterpart) or ``orphan_hyp`` (an
    ``insert`` chunk — the hypothesis character has none).

    ``same_base`` says whether the two sides' base code points are identical, which
    separates "the letter was read and the mark was not written" from "the letter
    itself is wrong, so no accent judgement about that syllable is meaningful".
    ``hyp_anchor`` is the position's hypothesis-side coordinate — the paired/orphan
    hypothesis index itself, or, for an ``orphan_ref``, the hypothesis index the
    deletion sits at — so locality (a mark written one syllable away) can be read off
    without re-deriving the alignment.
    """

    row: int
    op: str
    chunk_type: str
    ref_idx: Optional[int]
    hyp_idx: Optional[int]
    hyp_anchor: int
    ref_label: AccentLabel
    hyp_label: AccentLabel
    ref_is_vowel: bool
    hyp_is_vowel: bool
    same_base: bool


@dataclass(frozen=True)
class AccentAlignment:
    """One row's accent-channel alignment: its vowel maps and its event stream.

    ``ref_vowels`` / ``hyp_vowels`` are the full ``_vowel_positions`` maps (every
    vowel base, accented or not), so a mark can be located among its neighbouring
    syllables; ``events`` is the charged-position stream of :class:`AccentEvent`.
    """

    row: int
    ref_nfd: str
    hyp_nfd: str
    ref_vowels: dict[int, AccentLabel]
    hyp_vowels: dict[int, AccentLabel]
    events: tuple[AccentEvent, ...]

    @property
    def n_ref_accent_vowels(self) -> int:
        return sum(1 for label in self.ref_vowels.values() if label is not None)

    @property
    def n_hyp_accent_vowels(self) -> int:
        return sum(1 for label in self.hyp_vowels.values() if label is not None)


def _row_chunks(
    pred_list: Sequence[str], ref_list: Sequence[str]
) -> list[tuple[int, str, str, list[Any]]]:
    """``(row, ref_nfd, hyp_nfd, chunks)`` per non-degenerate row, in input order.

    One batched :func:`jiwer.process_characters` call for the whole corpus, exactly as
    ``compute_accent_metrics`` does it, on strings prepared by that module's own
    ``_prepare_for_jiwer``. Rows with two empty sides carry no accent information and
    are skipped (that module skips them too); a row with one empty side gets the
    determined whole-side :class:`_Chunk`.
    """
    prepared = [
        (i, _prepare_for_jiwer(ref), _prepare_for_jiwer(hyp))
        for i, (ref, hyp) in enumerate(zip(ref_list, pred_list))
    ]
    for_jiwer = [(i, r, h) for i, r, h in prepared if r and h]
    alignments: dict[int, list[Any]] = {}
    if for_jiwer:
        out = jiwer.process_characters(
            [r for _, r, _ in for_jiwer], [h for _, _, h in for_jiwer]
        )
        alignments = {
            i: list(chunks) for (i, _, _), chunks in zip(for_jiwer, out.alignments)
        }
    rows: list[tuple[int, str, str, list[Any]]] = []
    for i, ref_nfd, hyp_nfd in prepared:
        if not ref_nfd and not hyp_nfd:
            continue
        if not ref_nfd:
            rows.append(
                (i, ref_nfd, hyp_nfd, [_Chunk("insert", 0, 0, 0, len(hyp_nfd))])
            )
        elif not hyp_nfd:
            rows.append(
                (i, ref_nfd, hyp_nfd, [_Chunk("delete", 0, len(ref_nfd), 0, 0)])
            )
        else:
            rows.append((i, ref_nfd, hyp_nfd, alignments[i]))
    return rows


def _row_events(
    row: int,
    ref_nfd: str,
    hyp_nfd: str,
    chunks: Iterable[Any],
    ref_positions: dict[int, AccentLabel],
    hyp_positions: dict[int, AccentLabel],
) -> tuple[AccentEvent, ...]:
    """Turn one row's alignment chunks into its accent-relevant event stream.

    The chunk enumeration mirrors ``accent_metrics._accumulate_counts``: an
    ``equal``/``substitute`` chunk pairs its reference and hypothesis indices
    position by position (equal-length by construction for character alignment), a
    ``delete`` chunk orphans reference indices and an ``insert`` chunk orphans
    hypothesis ones.
    """
    events: list[AccentEvent] = []
    for chunk in chunks:
        ctype = chunk.type
        r0, r1 = chunk.ref_start_idx, chunk.ref_end_idx
        h0, h1 = chunk.hyp_start_idx, chunk.hyp_end_idx
        if ctype in ("equal", "substitute"):
            for k in range(r1 - r0):
                ri = r0 + k
                hi = h0 + k
                ref_is_vowel = ri in ref_positions
                hyp_is_vowel = hi in hyp_positions
                if not ref_is_vowel and not hyp_is_vowel:
                    continue
                events.append(
                    AccentEvent(
                        row=row,
                        op="paired",
                        chunk_type=ctype,
                        ref_idx=ri,
                        hyp_idx=hi,
                        hyp_anchor=hi,
                        ref_label=ref_positions.get(ri),
                        hyp_label=hyp_positions.get(hi),
                        ref_is_vowel=ref_is_vowel,
                        hyp_is_vowel=hyp_is_vowel,
                        same_base=ref_nfd[ri] == hyp_nfd[hi],
                    )
                )
        elif ctype == "delete":
            for ri in range(r0, r1):
                if ri not in ref_positions:
                    continue
                events.append(
                    AccentEvent(
                        row=row,
                        op="orphan_ref",
                        chunk_type=ctype,
                        ref_idx=ri,
                        hyp_idx=None,
                        hyp_anchor=h0,
                        ref_label=ref_positions.get(ri),
                        hyp_label=None,
                        ref_is_vowel=True,
                        hyp_is_vowel=False,
                        same_base=False,
                    )
                )
        elif ctype == "insert":
            for hi in range(h0, h1):
                if hi not in hyp_positions:
                    continue
                events.append(
                    AccentEvent(
                        row=row,
                        op="orphan_hyp",
                        chunk_type=ctype,
                        ref_idx=None,
                        hyp_idx=hi,
                        hyp_anchor=hi,
                        ref_label=None,
                        hyp_label=hyp_positions.get(hi),
                        ref_is_vowel=False,
                        hyp_is_vowel=True,
                        same_base=False,
                    )
                )
    return tuple(events)


def iter_accent_alignments(
    predictions: Iterable[object],
    references: Iterable[object],
) -> list[AccentAlignment]:
    """The per-row accent alignment underlying ``compute_accent_metrics``.

    Same inputs, same normalization, same aligner, same channel — restated here only
    because ``accent_metrics`` is source-locked (see the section comment above). Any
    divergence is caught by :func:`self_check_counts`, which charges this stream with
    that module's own charging functions and compares the result to
    ``compute_accent_metrics`` on the same text.
    """
    pred_list = [_coerce_text(p) for p in predictions]
    ref_list = [_coerce_text(r) for r in references]
    if len(pred_list) != len(ref_list):
        raise ValueError(
            f"predictions/references length mismatch: {len(pred_list)} vs {len(ref_list)}"
        )
    out: list[AccentAlignment] = []
    for row, ref_nfd, hyp_nfd, chunks in _row_chunks(pred_list, ref_list):
        ref_positions = _vowel_positions(ref_nfd)
        hyp_positions = _vowel_positions(hyp_nfd)
        out.append(
            AccentAlignment(
                row=row,
                ref_nfd=ref_nfd,
                hyp_nfd=hyp_nfd,
                ref_vowels=ref_positions,
                hyp_vowels=hyp_positions,
                events=_row_events(
                    row, ref_nfd, hyp_nfd, chunks, ref_positions, hyp_positions
                ),
            )
        )
    return out


def charge_alignments(
    alignments: Sequence[AccentAlignment],
) -> dict[str, dict[str, int]]:
    """Charge an event stream with ``accent_metrics``' own charging functions.

    The charging *rule* is imported, never restated: this feeds each event to
    ``_charge_paired`` / ``_charge_orphan_ref`` / ``_charge_orphan_hyp`` over that
    module's ``_new_counts`` table. So the only thing this module can get wrong about
    the metric is the chunk walk, and :func:`self_check_counts` tests exactly that.
    """
    counts = _new_counts()
    for alignment in alignments:
        for table in ("unlabeled", "labeled"):
            counts[table]["n_ref_accent_vowels"] += alignment.n_ref_accent_vowels
            counts[table]["n_hyp_accent_vowels"] += alignment.n_hyp_accent_vowels
        for event in alignment.events:
            if event.ref_idx is not None and event.hyp_idx is not None:
                _charge_paired(
                    event.ref_idx,
                    event.hyp_idx,
                    alignment.ref_vowels,
                    alignment.hyp_vowels,
                    counts,
                )
            elif event.ref_idx is not None:
                _charge_orphan_ref(event.ref_idx, alignment.ref_vowels, counts)
            elif event.hyp_idx is not None:
                _charge_orphan_hyp(event.hyp_idx, alignment.hyp_vowels, counts)
    return counts


#: The count keys the self-check compares, in report order.
SELF_CHECK_KEYS: tuple[str, ...] = (
    "tp",
    "fp",
    "fn",
    "n_ref_accent_vowels",
    "n_hyp_accent_vowels",
)


def self_check_counts(
    alignments: Sequence[AccentAlignment],
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, Any]:
    """Does this module's walk charge exactly what ``compute_accent_metrics`` charges?

    Runs the untouched metric function on the same normalized text and compares it,
    key by key and in both the unlabelled and labelled tables, against the same
    events charged by that module's own chargers. This is the guard that replaces the
    shared code path the source lock forbids: it is not a plausibility check but an
    equality, and a failure is publication-blocking.
    """
    derived = charge_alignments(alignments)
    reference = compute_accent_metrics(predictions, references)
    deltas: dict[str, dict[str, int]] = {}
    for table in ("unlabeled", "labeled"):
        for key in SELF_CHECK_KEYS:
            delta = int(derived[table][key]) - int(reference[table][key])
            if delta:
                deltas.setdefault(table, {})[key] = delta
    return {
        "matches_compute_accent_metrics": not deltas,
        "deltas": deltas,
        "note": (
            "accent_metrics.py is source-locked (shared metric implementation), so "
            "the alignment walk is restated in this module while its charging rule is "
            "imported. This equality is what keeps the restatement honest; it is "
            "checked per view and a mismatch blocks publication."
        ),
    }


# --------------------------------------------------------------------------- #
# Per-row classification.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Unmatched:
    """One unmatched accent, reduced to what the misplacement pairing needs.

    ``slot`` is the mark's position in *hypothesis syllable slots*: the number of
    hypothesis vowel bases before its hypothesis-side anchor. Both sides are
    expressed in that one coordinate — a reference mark through the anchor the
    alignment gives it — so a distance of 1 means "the adjacent syllable" for either.
    """

    slot: int
    label: str
    at_read_syllable: bool


def _hyp_slot(hyp_vowel_indices: Sequence[int], anchor: int) -> int:
    """Hypothesis syllable slot of a hypothesis-side code-point ``anchor``.

    For a vowel base this is its rank among the row's vowel bases; for any other
    anchor (a deletion point, a consonant) it is the number of vowel bases before
    it, which places it between the two neighbouring slots.
    """
    return bisect_left(hyp_vowel_indices, anchor)


def classify_row(
    alignment: AccentAlignment,
) -> tuple[dict[str, int], list[_Unmatched], list[_Unmatched]]:
    """Classify one row's accent channel into the window-free partition.

    Returns ``(counts, unmatched_ref, unmatched_hyp)``: the class counts for this
    row, and the two pools the misplacement refinement pairs across. Every event of
    :func:`~scripts.evaluation.accent_metrics.iter_accent_alignments` that carries a
    mark on either side lands in exactly one class, so the two partition sums hold
    row by row and therefore corpus-wide.
    """
    counts: dict[str, int] = {cls: 0 for cls in ERROR_CLASSES}
    counts["matched_on_substituted_base"] = 0
    unmatched_ref: list[_Unmatched] = []
    unmatched_hyp: list[_Unmatched] = []
    hyp_vowel_indices = sorted(alignment.hyp_vowels)

    for event in alignment.events:
        ref_label = event.ref_label
        hyp_label = event.hyp_label
        if ref_label is not None and hyp_label is not None:
            if ref_label == hyp_label:
                counts["matched_same_kind"] += 1
            else:
                counts["matched_other_kind"] += 1
            if not event.same_base:
                counts["matched_on_substituted_base"] += 1
        elif ref_label is not None:
            # The mark belongs here and the hypothesis has none at this position.
            # "Read" means the aligned hypothesis code point is the same letter;
            # anything else (a different letter, or no counterpart at all) makes an
            # accent judgement about this syllable meaningless.
            at_read = event.op == "paired" and event.same_base
            counts[
                "no_mark_at_read_syllable" if at_read else "ref_syllable_misread"
            ] += 1
            unmatched_ref.append(
                _Unmatched(
                    slot=_hyp_slot(hyp_vowel_indices, event.hyp_anchor),
                    label=ref_label,
                    at_read_syllable=at_read,
                )
            )
        elif hyp_label is not None:
            at_read = event.op == "paired" and event.same_base
            counts[
                "extra_at_read_syllable" if at_read else "extra_at_misread_syllable"
            ] += 1
            unmatched_hyp.append(
                _Unmatched(
                    slot=_hyp_slot(hyp_vowel_indices, event.hyp_anchor),
                    label=hyp_label,
                    at_read_syllable=at_read,
                )
            )
    return counts, unmatched_ref, unmatched_hyp


# --------------------------------------------------------------------------- #
# Misplacement refinement (window-dependent, always reported with its null).
# --------------------------------------------------------------------------- #
def pair_misplaced(
    unmatched_ref: Sequence[_Unmatched],
    unmatched_hyp: Sequence[_Unmatched],
    window: int,
) -> list[tuple[int, int, int]]:
    """One-to-one nearest-first pairing of same-kind unmatched marks within ``window``.

    Returns ``(distance, ref_index, hyp_index)`` triples. Candidates are enumerated,
    sorted by ``(distance, ref_index, hyp_index)`` — a total order, so the result is
    deterministic and independent of input order beyond the indices themselves — and
    taken greedily, each mark used at most once. Requiring the **same kind** is what
    makes a pair mean "right kind, wrong syllable"; a nearby pair of different kinds
    is left in ``no_mark_at_read_syllable`` / ``extra_*`` and counted separately as
    ``misplaced_other_kind``.
    """
    candidates = [
        (abs(ref.slot - hyp.slot), i, j)
        for i, ref in enumerate(unmatched_ref)
        for j, hyp in enumerate(unmatched_hyp)
        if ref.label == hyp.label and abs(ref.slot - hyp.slot) <= window
    ]
    candidates.sort()
    used_ref: set[int] = set()
    used_hyp: set[int] = set()
    pairs: list[tuple[int, int, int]] = []
    for distance, i, j in candidates:
        if i in used_ref or j in used_hyp:
            continue
        used_ref.add(i)
        used_hyp.add(j)
        pairs.append((distance, i, j))
    return pairs


def _shift_null_pairs(
    unmatched_ref: Sequence[_Unmatched],
    unmatched_hyp: Sequence[_Unmatched],
    window: int,
    n_slots: int,
) -> int:
    """The same pairing after rotating the hypothesis marks half a row.

    The density control. Each unmatched hypothesis mark's slot is rotated by
    ``n_slots // 2`` modulo the row's slot count, which preserves how many marks
    there are, their kinds, and the number of syllables they can land on, while
    destroying any correspondence with the reference's marks. What it returns is
    therefore how many "misplacements" this row yields from density alone. It is a
    locality control, not a significance test: no p-value is derived from it.
    """
    if n_slots <= 1:
        return 0
    shift = n_slots // 2
    rotated = [
        _Unmatched(
            slot=(hyp.slot + shift) % n_slots,
            label=hyp.label,
            at_read_syllable=hyp.at_read_syllable,
        )
        for hyp in unmatched_hyp
    ]
    return len(pair_misplaced(unmatched_ref, rotated, window))


def misplacement_refinement(
    pools: Sequence[tuple[Sequence[_Unmatched], Sequence[_Unmatched], int]],
    *,
    window: int = MISPLACEMENT_WINDOW,
) -> dict[str, Any]:
    """Corpus-level misplacement refinement over per-row unmatched pools.

    ``pools`` is one ``(unmatched_ref, unmatched_hyp, n_hyp_slots)`` triple per row.
    The refinement never changes the partition: each pair it forms consumes one
    reference-side member and one hypothesis-side member, both of which keep their
    original class.
    """
    n_pairs = 0
    n_null = 0
    n_other_kind = 0
    from_read_syllable = 0
    distances: dict[int, int] = {}
    for unmatched_ref, unmatched_hyp, n_slots in pools:
        pairs = pair_misplaced(unmatched_ref, unmatched_hyp, window)
        n_pairs += len(pairs)
        for distance, i, _j in pairs:
            distances[distance] = distances.get(distance, 0) + 1
            if unmatched_ref[i].at_read_syllable:
                from_read_syllable += 1
        n_null += _shift_null_pairs(unmatched_ref, unmatched_hyp, window, n_slots)
        # Residual pools after the same-kind pairing, paired ignoring the kind: how
        # many nearby unmatched pairs disagree on which mark it should have been.
        used_ref = {i for _, i, _ in pairs}
        used_hyp = {j for _, _, j in pairs}
        rest_ref = [m for i, m in enumerate(unmatched_ref) if i not in used_ref]
        rest_hyp = [m for j, m in enumerate(unmatched_hyp) if j not in used_hyp]
        relabelled = [
            _Unmatched(slot=m.slot, label="*", at_read_syllable=m.at_read_syllable)
            for m in rest_hyp
        ]
        n_other_kind += len(
            pair_misplaced(
                [
                    _Unmatched(
                        slot=m.slot, label="*", at_read_syllable=m.at_read_syllable
                    )
                    for m in rest_ref
                ],
                relabelled,
                window,
            )
        )
    return {
        "window_slots": window,
        "misplaced": n_pairs,
        "misplaced_shift_null": n_null,
        "misplaced_excess_over_null": n_pairs - n_null,
        "misplaced_other_kind": n_other_kind,
        "misplaced_from_read_syllable": from_read_syllable,
        "distance_histogram": {str(k): distances[k] for k in sorted(distances)},
    }


# --------------------------------------------------------------------------- #
# One prediction CSV → its decomposition.
# --------------------------------------------------------------------------- #
def _nfd_accent_mark_census(references: Sequence[str]) -> dict[str, int]:
    """NFD accent code points, and how many of them are the letter ``ś``.

    Walks back from each U+0301 / U+0300 to its base code point; an acute whose base
    is ``s``/``S`` is the letter ś, not an accent (:data:`LETTER_S_ACUTE_NOTE`).
    """
    total = 0
    on_letter_s = 0
    for text in references:
        nfd = unicodedata.normalize("NFD", text)
        for i, ch in enumerate(nfd):
            if ch != ACUTE and ch != GRAVE:
                continue
            total += 1
            j = i - 1
            while j >= 0 and unicodedata.combining(nfd[j]):
                j -= 1
            if j >= 0 and ch == ACUTE and nfd[j] in ("s", "S"):
                on_letter_s += 1
    return {
        "n_ref_accent_marks_nfd": total,
        "n_ref_accent_marks_on_letter_s": on_letter_s,
    }


def decompose_prediction_file(
    csv_path: Path,
    *,
    window: int = MISPLACEMENT_WINDOW,
    sensitivity_windows: Sequence[int] = MISPLACEMENT_SENSITIVITY_WINDOWS,
    alt_normalization: Optional[Normalization] = DEFAULT_NORMALIZATION,
) -> dict[str, Any]:
    """Decompose one committed ``{rv,ms}_predictions.csv`` into the accent classes.

    Rows come from the shared :func:`~scripts.evaluation.prediction_metrics.load_prediction_rows`
    (one loader), then both sides go through :func:`normalize_runner_iso` — the
    runner's own transform, which is what makes the result reconcilable against
    ``results.json``. ``alt_normalization``, when given, additionally reports what
    the study evaluator's NFC-and-strip convention would have counted, so the
    difference between the two conventions is measured here instead of surfacing as
    an unexplained delta somewhere downstream.
    """
    rows = load_prediction_rows(csv_path)
    references = [normalize_runner_iso(row.reference) for row in rows]
    predictions = [normalize_runner_iso(row.prediction) for row in rows]

    counts: dict[str, int] = {cls: 0 for cls in ERROR_CLASSES}
    counts["matched_on_substituted_base"] = 0
    n_ref_accent_vowels = 0
    n_hyp_accent_vowels = 0
    pools: list[tuple[Sequence[_Unmatched], Sequence[_Unmatched], int]] = []
    alignments = iter_accent_alignments(predictions, references)
    for alignment in alignments:
        row_counts, unmatched_ref, unmatched_hyp = classify_row(alignment)
        for key, value in row_counts.items():
            counts[key] += value
        n_ref_accent_vowels += alignment.n_ref_accent_vowels
        n_hyp_accent_vowels += alignment.n_hyp_accent_vowels
        pools.append((unmatched_ref, unmatched_hyp, len(alignment.hyp_vowels)))

    refinement = misplacement_refinement(pools, window=window)
    sensitivity = {
        str(w): {
            k: v
            for k, v in misplacement_refinement(pools, window=w).items()
            if k in ("misplaced", "misplaced_shift_null", "misplaced_excess_over_null")
        }
        for w in sensitivity_windows
    }

    out: dict[str, Any] = {
        "n_samples": len(rows),
        "counts": counts,
        "n_ref_accent_vowels": n_ref_accent_vowels,
        "n_hyp_accent_vowels": n_hyp_accent_vowels,
        "misplacement": refinement,
        "misplacement_sensitivity": sensitivity,
        "self_check": self_check_counts(alignments, predictions, references),
        "reference_sha256": hashlib.sha256(
            chr(0).join(references).encode("utf-8")
        ).hexdigest(),
        **_nfd_accent_mark_census(references),
        "text_transform": RUNNER_TRANSFORM_LABEL,
    }
    out["n_ref_accent_marks_unanchored_other"] = (
        out["n_ref_accent_marks_nfd"]
        - out["n_ref_accent_marks_on_letter_s"]
        - n_ref_accent_vowels
    )
    if alt_normalization is not None:
        out["alt_normalization"] = _alt_normalization_counts(rows, alt_normalization)
    return out


def _alt_normalization_counts(
    rows: Sequence[Any], normalization: Normalization
) -> dict[str, Any]:
    """The same accent counts under the study evaluator's normalization.

    Reported so the gap between the two documented conventions is a measured number
    in this bundle rather than an unexplained residue in the main grid's cross-check.
    """
    from .accent_metrics import compute_accent_metrics

    metrics = compute_accent_metrics(
        [normalization.apply(row.prediction) for row in rows],
        [normalization.apply(row.reference) for row in rows],
    )
    unlabeled = metrics["unlabeled"]
    return {
        "normalization": ALT_TRANSFORM_LABEL,
        "accent_tp": int(unlabeled["tp"]),
        "accent_fp": int(unlabeled["fp"]),
        "accent_fn": int(unlabeled["fn"]),
        "n_ref_accent_vowels": int(unlabeled["n_ref_accent_vowels"]),
        "n_hyp_accent_vowels": int(unlabeled["n_hyp_accent_vowels"]),
    }


# --------------------------------------------------------------------------- #
# Reconciliation against the committed per-cell counts.
# --------------------------------------------------------------------------- #
#: The six identities, as ``(name, decomposition-side expression, committed key)``.
#: Kept declarative so the results page, the CSV and the warning text all read the
#: same list and cannot describe different checks.
RECONCILIATION_IDENTITIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("accent_tp", ("matched_same_kind", "matched_other_kind"), "accent_tp"),
    ("accent_fn", ("no_mark_at_read_syllable", "ref_syllable_misread"), "accent_fn"),
    ("accent_fp", ("extra_at_read_syllable", "extra_at_misread_syllable"), "accent_fp"),
    ("accent_tp_labeled", ("matched_same_kind",), "accent_tp_labeled"),
    (
        "accent_fn_labeled",
        ("no_mark_at_read_syllable", "ref_syllable_misread", "matched_other_kind"),
        "accent_fn_labeled",
    ),
    (
        "accent_fp_labeled",
        ("extra_at_read_syllable", "extra_at_misread_syllable", "matched_other_kind"),
        "accent_fp_labeled",
    ),
    ("n_ref_accent_vowels", REF_SIDE_CLASSES, "n_ref_accent_vowels"),
    ("n_hyp_accent_vowels", HYP_SIDE_CLASSES, "n_hyp_accent_vowels"),
)


def reconcile_view(counts: dict[str, int], committed: dict[str, Any]) -> dict[str, Any]:
    """Check every identity of :data:`RECONCILIATION_IDENTITIES` against ``committed``.

    ``committed`` is the cell's own ``results.json`` view block — the published
    numbers, not a recomputation of them. A key the block does not carry is reported
    as ``committed: None`` and ``checked: False`` rather than silently passing.
    """
    rows: list[dict[str, Any]] = []
    for name, classes, key in RECONCILIATION_IDENTITIES:
        derived = sum(counts.get(cls, 0) for cls in classes)
        raw = committed.get(key)
        # A key the block does not carry stays unchecked: ``ok`` is False and
        # ``reconciles`` therefore cannot be earned by an absent number.
        value = None if raw is None else int(raw)
        rows.append(
            {
                "identity": name,
                "expression": " + ".join(classes),
                "derived": derived,
                "committed": value,
                "delta": None if value is None else derived - value,
                "checked": value is not None,
                "ok": value is not None and derived == value,
            }
        )
    return {
        "identities": rows,
        "n_checked": sum(1 for r in rows if r["checked"]),
        "n_unchecked": sum(1 for r in rows if not r["checked"]),
        "mismatches": [r["identity"] for r in rows if r["checked"] and not r["ok"]],
        "reconciles": all(r["ok"] for r in rows),
        "note": (
            "Right-hand sides are the values the run itself committed in "
            "results.json. Reproducing them from the saved CSVs requires the "
            "runner's text transform (see normalize_runner_iso); the study "
            "evaluator's NFC-and-strip convention does not reproduce them, and the "
            "difference is reported per view as alt_normalization."
        ),
    }


# --------------------------------------------------------------------------- #
# Per-cell / per-view rows.
# --------------------------------------------------------------------------- #
def prediction_csv_path(cell: agg.SeedCell, view: str) -> Optional[Path]:
    """The committed ``<view>_predictions.csv`` beside a present cell's results."""
    if cell.results_path is None:
        return None
    return Path(cell.results_path).parent / f"{view}_predictions.csv"


def _committed_view(study_root: Path, cell: agg.SeedCell, view: str) -> dict[str, Any]:
    """The cell's own ``results.json`` block for one view (``{}`` when unreadable)."""
    path = study_root / cell.run_name / "results.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    block = data.get(f"{view}_test")
    return block if isinstance(block, dict) else {}


def decompose_grid(
    cells: Sequence[agg.SeedCell],
    study_root: Path,
    *,
    window: int = MISPLACEMENT_WINDOW,
    views: Sequence[str] = VIEWS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per present ``(cell, view)`` with a prediction CSV, plus the skips.

    A present cell whose prediction CSV is absent is **skipped and named**, never
    imputed: the runner writes ``results.json`` before the CSVs, so mid-flight this
    is a normal state — and a table that quietly drops it would report a smaller n
    without saying so.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for cell in cells:
        if not cell.present:
            continue
        for view in views:
            csv_path = prediction_csv_path(cell, view)
            if csv_path is None or not csv_path.is_file():
                skipped.append(f"{cell.run_name}:{view}")
                continue
            payload = decompose_prediction_file(csv_path, window=window)
            counts = payload["counts"]
            committed = _committed_view(study_root, cell, view)
            n_ref = payload["n_ref_accent_vowels"]
            row: dict[str, Any] = {
                "rv_ratio": cell.rv_ratio,
                "ms_ratio": cell.ms_ratio,
                "seed": cell.seed,
                "run_name": cell.run_name,
                "view": view,
                "direction": direction_of(cell.rv_ratio, view),
                "trained_on_view": trained_on_view(cell.rv_ratio, view),
                "n_samples": payload["n_samples"],
                "n_ref_accent_vowels": n_ref,
                "n_hyp_accent_vowels": payload["n_hyp_accent_vowels"],
                # Marks written per mark owed: the discriminator between a channel
                # that went quiet and one that went random (see DIAGNOSTIC_COUNTS).
                "hyp_marks_per_ref_accent": (
                    payload["n_hyp_accent_vowels"] / n_ref if n_ref else None
                ),
                "counts": counts,
                "per_ref_accent_vowel": {
                    cls: (counts[cls] / n_ref if n_ref else None)
                    for cls in ERROR_CLASSES
                },
                "misplacement": payload["misplacement"],
                "misplacement_sensitivity": payload["misplacement_sensitivity"],
                "self_check": payload["self_check"],
                "reconciliation": reconcile_view(counts, committed),
                "reference_sha256": payload["reference_sha256"],
                "n_ref_accent_marks_nfd": payload["n_ref_accent_marks_nfd"],
                "n_ref_accent_marks_on_letter_s": payload[
                    "n_ref_accent_marks_on_letter_s"
                ],
                "n_ref_accent_marks_unanchored_other": payload[
                    "n_ref_accent_marks_unanchored_other"
                ],
                "matched_on_substituted_base": counts["matched_on_substituted_base"],
                "alt_normalization": payload.get("alt_normalization"),
                "prediction_csv": str(csv_path),
            }
            rows.append(row)
    return rows, skipped


def trained_on_view(rv_ratio: float, view: str) -> Optional[bool]:
    """Did this cell's training mix include the notation it is being read on?

    ``True`` / ``False`` for the two single-notation endpoints — the only cells where
    "trained notation" versus "untrained notation" is a fact about the mix rather
    than a matter of degree — and ``None`` for the three mixed ratios, which saw
    both. The trained-versus-untrained contrasts use only the endpoints.
    """
    if rv_ratio == 1.0:
        return view == "rv"
    if rv_ratio == 0.0:
        return view == "ms"
    return None


# --------------------------------------------------------------------------- #
# Aggregation across seeds (shared statistics, never re-implemented).
# --------------------------------------------------------------------------- #
def aggregate_by_ratio_view(
    rows: Sequence[dict[str, Any]],
    *,
    rv_ratios: Sequence[float] = mg.RV_RATIOS,
    views: Sequence[str] = VIEWS,
) -> list[dict[str, Any]]:
    """Per ``(rv_ratio, view, class)`` mean ± 95% CI of the per-reference-accent rate.

    Summarized across the seeds present for that cell with
    :func:`~scripts.evaluation.aggregate_accent_ratio_multiseed.summarize`, so ``n``,
    the sample SD and the Student-t interval are the same construction every other v2
    table uses. Rows come back in canonical ratio → view → class order, each carrying
    its own ``n`` and the seeds behind it.
    """
    index: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        index.setdefault((row["rv_ratio"], row["view"]), []).append(row)
    out: list[dict[str, Any]] = []
    for rv_ratio in rv_ratios:
        for view in views:
            group = sorted(index.get((rv_ratio, view), []), key=lambda r: r["seed"])
            if not group:
                continue
            for cls in ERROR_CLASSES:
                stat = agg.summarize(r["per_ref_accent_vowel"][cls] for r in group)
                out.append(
                    {
                        "rv_ratio": rv_ratio,
                        "ms_ratio": round(1.0 - rv_ratio, 2),
                        "view": view,
                        "direction": group[0]["direction"],
                        "trained_on_view": group[0]["trained_on_view"],
                        "error_class": cls,
                        "seeds": [r["seed"] for r in group],
                        "per_ref_accent_vowel": stat.as_dict(),
                        "count_mean": statistics.fmean(
                            [r["counts"][cls] for r in group]
                        ),
                        "n_ref_accent_vowels": statistics.fmean(
                            [r["n_ref_accent_vowels"] for r in group]
                        ),
                        "n_samples": statistics.fmean([r["n_samples"] for r in group]),
                    }
                )
    return out


def aggregate_mark_budget(
    rows: Sequence[dict[str, Any]],
    *,
    rv_ratios: Sequence[float] = mg.RV_RATIOS,
    views: Sequence[str] = VIEWS,
) -> list[dict[str, Any]]:
    """Per ``(rv_ratio, view)``: how many marks the model wrote per mark it owed.

    ``hyp_marks_per_ref_accent`` is deliberately separate from the class table
    because it is what makes the class table interpretable. A collapsed accent F1 is
    consistent with two opposite behaviours — a channel that stopped writing marks and
    a channel that kept writing them somewhere else — and only this ratio tells them
    apart. ``matched_on_substituted_base`` rides along: it is the share of the
    channel's credited matches that sit on a letter the base channel got wrong.
    """
    index: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        index.setdefault((row["rv_ratio"], row["view"]), []).append(row)
    out: list[dict[str, Any]] = []
    for rv_ratio in rv_ratios:
        for view in views:
            group = sorted(index.get((rv_ratio, view), []), key=lambda r: r["seed"])
            if not group:
                continue
            out.append(
                {
                    "rv_ratio": rv_ratio,
                    "ms_ratio": round(1.0 - rv_ratio, 2),
                    "view": view,
                    "direction": group[0]["direction"],
                    "trained_on_view": group[0]["trained_on_view"],
                    "seeds": [r["seed"] for r in group],
                    "hyp_marks_per_ref_accent": agg.summarize(
                        r["hyp_marks_per_ref_accent"] for r in group
                    ).as_dict(),
                    "matched_on_substituted_base_per_ref_accent": agg.summarize(
                        (
                            r["matched_on_substituted_base"] / r["n_ref_accent_vowels"]
                            if r["n_ref_accent_vowels"]
                            else None
                        )
                        for r in group
                    ).as_dict(),
                    "n_ref_accent_vowels": statistics.fmean(
                        [r["n_ref_accent_vowels"] for r in group]
                    ),
                    "n_hyp_accent_vowels": statistics.fmean(
                        [r["n_hyp_accent_vowels"] for r in group]
                    ),
                }
            )
    return out


def aggregate_misplacement_by_ratio_view(
    rows: Sequence[dict[str, Any]],
    *,
    rv_ratios: Sequence[float] = mg.RV_RATIOS,
    views: Sequence[str] = VIEWS,
) -> list[dict[str, Any]]:
    """Per ``(rv_ratio, view)`` misplacement count, its density null, and the excess.

    Rates are per reference accent, summarized across seeds with the same
    :func:`~scripts.evaluation.aggregate_accent_ratio_multiseed.summarize`. The null
    travels with the count in every row, so the count is never available without it.
    """
    index: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        index.setdefault((row["rv_ratio"], row["view"]), []).append(row)
    out: list[dict[str, Any]] = []
    for rv_ratio in rv_ratios:
        for view in views:
            group = sorted(index.get((rv_ratio, view), []), key=lambda r: r["seed"])
            if not group:
                continue

            def _rate(key: str, grp: Sequence[dict[str, Any]] = ()) -> agg.Stat:
                grp = grp or group
                return agg.summarize(
                    (
                        r["misplacement"][key] / r["n_ref_accent_vowels"]
                        if r["n_ref_accent_vowels"]
                        else None
                    )
                    for r in grp
                )

            out.append(
                {
                    "rv_ratio": rv_ratio,
                    "ms_ratio": round(1.0 - rv_ratio, 2),
                    "view": view,
                    "direction": group[0]["direction"],
                    "trained_on_view": group[0]["trained_on_view"],
                    "window_slots": group[0]["misplacement"]["window_slots"],
                    "seeds": [r["seed"] for r in group],
                    "misplaced_per_ref_accent": _rate("misplaced").as_dict(),
                    "shift_null_per_ref_accent": _rate(
                        "misplaced_shift_null"
                    ).as_dict(),
                    "excess_per_ref_accent": _rate(
                        "misplaced_excess_over_null"
                    ).as_dict(),
                    "misplaced_count_mean": statistics.fmean(
                        [r["misplacement"]["misplaced"] for r in group]
                    ),
                    "shift_null_count_mean": statistics.fmean(
                        [r["misplacement"]["misplaced_shift_null"] for r in group]
                    ),
                    "other_kind_count_mean": statistics.fmean(
                        [r["misplacement"]["misplaced_other_kind"] for r in group]
                    ),
                    "n_ref_accent_vowels": statistics.fmean(
                        [r["n_ref_accent_vowels"] for r in group]
                    ),
                }
            )
    return out


#: The two endpoint models, each read on the notation it trained on and the one it
#: did not: ``(label, rv_ratio, trained_view, untrained_view)``.
ENDPOINT_MODELS: tuple[tuple[str, float, str, str], ...] = (
    ("rv_only_model", 1.0, "rv", "ms"),
    ("ms_only_model", 0.0, "ms", "rv"),
)


def _paired_rate(
    rows: Sequence[dict[str, Any]], rv_ratio: float, view: str, cls: str
) -> dict[int, float]:
    """``seed → per-reference-accent rate`` for one (ratio, view, class) arm."""
    out: dict[int, float] = {}
    for row in rows:
        if row["rv_ratio"] != rv_ratio or row["view"] != view:
            continue
        value = row["per_ref_accent_vowel"].get(cls)
        if value is not None:
            out[row["seed"]] = value
    return out


def trained_vs_untrained_contrasts(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per endpoint model, the seed-paired ``untrained − trained`` shift per class.

    Uses :func:`~scripts.evaluation.aggregate_accent_ratio_multiseed._contrast`
    verbatim, so the difference, its Student-t interval, its per-seed values and its
    ``sign_consistent`` flag are constructed exactly as in the main-grid tables. The
    orientation is fixed and stated: **positive means the class is more frequent on
    the notation the model never trained on.** Only the two single-notation endpoints
    are contrasted, because only there is "untrained notation" a property of the mix.
    """
    out: dict[str, dict[str, Any]] = {}
    for label, rv_ratio, trained_view, untrained_view in ENDPOINT_MODELS:
        per_class = {
            cls: agg._contrast(
                _paired_rate(rows, rv_ratio, trained_view, cls),
                _paired_rate(rows, rv_ratio, untrained_view, cls),
            )
            for cls in ERROR_CLASSES
        }
        out[label] = {
            "rv_ratio": rv_ratio,
            "trained_view": trained_view,
            "untrained_view": untrained_view,
            "orientation": "untrained_view - trained_view (positive = worse/more on "
            "the notation the model never saw)",
            "per_class": per_class,
        }
    return out


# --------------------------------------------------------------------------- #
# Coverage / warnings.
# --------------------------------------------------------------------------- #
def view_coverage(
    rows: Sequence[dict[str, Any]],
    skipped: Sequence[str],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Decomposed-view coverage of the target 5 × 3 × 2 grid.

    Two things can be missing and they are not the same: a **cell** that has not
    finished (``coverage`` from the main-grid module) and a finished cell whose
    **prediction CSV** is not there yet. ``complete`` requires both to be whole,
    measured against :data:`TARGET_N_VIEWS` rather than against ``len(rows)``.
    """
    per_ratio: dict[str, Any] = {}
    for rv_ratio in mg.RV_RATIOS:
        group = [r for r in rows if r["rv_ratio"] == rv_ratio]
        per_view = {
            view: sorted(r["seed"] for r in group if r["view"] == view)
            for view in VIEWS
        }
        per_ratio[f"{rv_ratio:g}"] = {
            "rv_ratio": rv_ratio,
            "ms_ratio": round(1.0 - rv_ratio, 2),
            "n_views": len(mg.SEEDS) * len(VIEWS),
            "n_views_decomposed": len(group),
            "seeds_per_view": per_view,
            "complete": len(group) == len(mg.SEEDS) * len(VIEWS),
        }
    return {
        "n_views": TARGET_N_VIEWS,
        "n_views_decomposed": len(rows),
        "complete": len(rows) == TARGET_N_VIEWS and coverage["complete"],
        "cells_complete": coverage["complete"],
        "n_cells_present": coverage["n_present"],
        "n_cells": coverage["n_cells"],
        "skipped_views_missing_predictions": list(skipped),
        "per_ratio": per_ratio,
    }


def status_label(view_cov: dict[str, Any]) -> str:
    """``"COMPLETE"`` or ``"PARTIAL"`` — the word every artifact leads with."""
    return "COMPLETE" if view_cov["complete"] else "PARTIAL"


def partial_banner(view_cov: dict[str, Any]) -> Optional[str]:
    """The PARTIAL banner, or ``None`` when every target view is decomposed."""
    if view_cov["complete"]:
        return None
    missing = view_cov["skipped_views_missing_predictions"]
    detail = (
        f" Present cells whose prediction CSVs are missing: {', '.join(missing)}."
        if missing
        else ""
    )
    return (
        f"PARTIAL — NOT A COMPLETE RESULT: {view_cov['n_views_decomposed']}/"
        f"{view_cov['n_views']} test views decomposed "
        f"({view_cov['n_cells_present']}/{view_cov['n_cells']} cells present)."
        " No number below may be presented as the v2 accent error decomposition, and "
        "no per-ratio mean/CI is a 3-seed number unless its own n says 3." + detail
    )


def reference_set_identity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Do all decomposed views share one reference set, and can it attest the corpus?

    The RV and MS test labels are parallel by text, so a healthy grid has exactly one
    reference sha256 across all 30 views — which is what licenses reading an RV↔MS
    gap as a notation effect rather than a subset-difficulty effect. It is emphatically
    **not** corpus evidence: v1 and v2 re-render the *same* ISO text, so this hash is
    identical across corpora by construction. Corpus identity comes from the Slurm
    header attestation, never from this hash and never from a path name.
    """
    hashes = sorted({row["reference_sha256"] for row in rows})
    return {
        "n_distinct_reference_sequences": len(hashes),
        "reference_sha256": hashes,
        "single_reference_set": len(hashes) == 1,
        "note": (
            "One hash across every (cell, view) is the parallel-by-text property: the "
            "notation contrast is image-only. This hash CANNOT attest the corpus — v2 "
            "re-renders the same ISO text as v1, so it is identical across corpora by "
            "construction. Corpus identity is the Slurm-header attestation."
        ),
    }


#: Publication-blocking substrings, extending the main grid's list with this
#: analysis's own failure modes. Reused by the CLI exit status.
HARD_WARNING_MARKERS: tuple[str, ...] = mg.HARD_WARNING_MARKERS + (
    "does not reconcile",
    "more than one reference set",
    "unanchored accent code points",
    "disagrees with compute_accent_metrics",
)


def hard_warnings(warnings: Sequence[str]) -> list[str]:
    """The publication-blocking subset of :func:`audit_warnings`."""
    return [
        w
        for w in warnings
        if w.startswith("PARTIAL") or any(m in w for m in HARD_WARNING_MARKERS)
    ]


def audit_warnings(
    cells: Sequence[agg.SeedCell],
    coverage: dict[str, Any],
    view_cov: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    reference_identity: dict[str, Any],
    *,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
) -> list[str]:
    """The main grid's warnings plus this analysis's reconciliation checks."""
    warnings: list[str] = []
    banner = partial_banner(view_cov)
    if banner is not None:
        warnings.append(banner)
    # The grid-level checks, minus its own coverage banner (this study's banner is
    # about decomposed views and already leads the list).
    grid_banner = mg.partial_banner(coverage)
    warnings.extend(
        w
        for w in mg.audit_warnings(
            cells, coverage, subset=subset, attestation=attestation, integrity=integrity
        )
        if w != grid_banner
    )
    if grid_banner is not None and banner is None:
        warnings.append(grid_banner)

    drifted = [
        f"{row['run_name']}:{row['view']} ({row['self_check']['deltas']})"
        for row in rows
        if not row["self_check"]["matches_compute_accent_metrics"]
    ]
    if drifted:
        warnings.append(
            "view(s) where this module's restated alignment walk disagrees with "
            "compute_accent_metrics on the same text — accent_metrics.py is "
            "source-locked, so the walk lives here and this equality is what keeps it "
            "honest: " + "; ".join(drifted)
        )
    failed = [
        f"{row['run_name']}:{row['view']} ({', '.join(row['reconciliation']['mismatches'])})"
        for row in rows
        if not row["reconciliation"]["reconciles"]
    ]
    if failed:
        warnings.append(
            "view(s) whose class decomposition does not reconcile against the "
            "committed accent counts in results.json: " + "; ".join(failed)
        )
    unchecked = [
        f"{row['run_name']}:{row['view']}"
        for row in rows
        if row["reconciliation"]["n_unchecked"]
    ]
    if unchecked:
        warnings.append(
            "view(s) whose results.json omits one or more accent count keys, so that "
            "identity could not be checked (reported as checked=False, never as "
            "passing): " + ", ".join(unchecked)
        )
    if rows and not reference_identity["single_reference_set"]:
        warnings.append(
            "decomposed views span more than one reference set "
            f"({reference_identity['n_distinct_reference_sequences']} distinct "
            "sha256), so the RV/MS comparison is not parallel by text and a "
            "cross-notation gap could be a subset-difficulty difference"
        )
    unanchored = [
        f"{row['run_name']}:{row['view']}={row['n_ref_accent_marks_unanchored_other']}"
        for row in rows
        if row["n_ref_accent_marks_unanchored_other"]
    ]
    if unanchored:
        warnings.append(
            "view(s) with unanchored accent code points in the reference that are "
            "neither vowel-anchored nor part of the letter ś, so the accent channel "
            "does not account for every mark in the text: " + ", ".join(unanchored)
        )
    return warnings


# --------------------------------------------------------------------------- #
# Aggregate payload.
# --------------------------------------------------------------------------- #
def build_aggregate(
    cells: Sequence[agg.SeedCell],
    rows: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    view_cov: dict[str, Any],
    *,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Machine-readable payload (→ ``aggregate.json``)."""
    reference_identity = reference_set_identity(rows)
    warnings = audit_warnings(
        cells,
        coverage,
        view_cov,
        rows,
        reference_identity,
        subset=subset,
        attestation=attestation,
        integrity=integrity,
    )
    by_ratio_view = aggregate_by_ratio_view(rows)
    return {
        "study": STUDY,
        "source_study": SOURCE_STUDY,
        "issue": ISSUE,
        "corpus": "v2",
        "analysis_only": True,
        "no_training_no_inference": True,
        "status": status_label(view_cov),
        "complete": view_cov["complete"],
        "partial_banner": partial_banner(view_cov),
        "grid": {
            "rv_ratios": list(mg.RV_RATIOS),
            "views": list(VIEWS),
            "seeds": coverage["seeds_present"],
            "target_seeds": list(mg.SEEDS),
        },
        "text_transform": RUNNER_TRANSFORM_LABEL,
        "alt_text_transform": ALT_TRANSFORM_LABEL,
        "misplacement_window_slots": MISPLACEMENT_WINDOW,
        "misplacement_sensitivity_windows": list(MISPLACEMENT_SENSITIVITY_WINDOWS),
        "misplacement_window_zero_note": WINDOW_ZERO_NOTE,
        "class_definitions": CLASS_DEFINITIONS,
        "diagnostic_counts": DIAGNOSTIC_COUNTS,
        "ref_side_partition": list(REF_SIDE_CLASSES),
        "hyp_side_partition": list(HYP_SIDE_CLASSES),
        "letter_s_acute_note": LETTER_S_ACUTE_NOTE,
        "coverage": view_cov,
        "cell_coverage": coverage,
        "reference_set_identity": reference_identity,
        "per_cell": [
            {k: v for k, v in row.items() if k != "misplacement_sensitivity"}
            for row in rows
        ],
        "per_ratio_view": by_ratio_view,
        "mark_budget_per_ratio_view": aggregate_mark_budget(rows),
        "misplacement_per_ratio_view": aggregate_misplacement_by_ratio_view(rows),
        "trained_vs_untrained": trained_vs_untrained_contrasts(rows),
        "misplacement_sensitivity": {
            f"{row['rv_ratio']:g}:{row['view']}:{row['seed']}": row[
                "misplacement_sensitivity"
            ]
            for row in rows
        },
        "alignment_self_check": {
            "n_views": len(rows),
            "n_matching_compute_accent_metrics": sum(
                1 for row in rows if row["self_check"]["matches_compute_accent_metrics"]
            ),
            "all_match": bool(rows)
            and all(
                row["self_check"]["matches_compute_accent_metrics"] for row in rows
            ),
            "why": (
                "scripts/evaluation/accent_metrics.py is inside the shared OCR evaluation "
                "published artifact's source lock, so the event-emitting hook this "
                "analysis wants cannot be added there. The alignment walk is restated "
                "in this module and its charging rule imported unchanged; this "
                "per-view equality against compute_accent_metrics is the guard that "
                "replaces the shared code path."
            ),
        },
        "reconciliation": {
            "n_views": len(rows),
            "n_reconciling": sum(
                1 for row in rows if row["reconciliation"]["reconciles"]
            ),
            "all_reconcile": all(row["reconciliation"]["reconciles"] for row in rows)
            and bool(rows),
            "identities": [
                {
                    "identity": name,
                    "expression": " + ".join(classes),
                    "committed_key": key,
                }
                for name, classes, key in RECONCILIATION_IDENTITIES
            ],
        },
        "subset_consistency": subset,
        "corpus_attestation": attestation,
        "grid_integrity": integrity,
        "warnings": warnings,
        "hard_warnings": hard_warnings(warnings),
        "publishable": not hard_warnings(warnings)
        and bool((attestation or {}).get("healthy")),
    }


def build_manifest(
    aggregate: dict[str, Any],
    *,
    study_root: Path,
    shared_study_root: Optional[str],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Provenance for the bundle: inputs, method, and what it deliberately is not."""
    return {
        "study": STUDY,
        "issue": ISSUE,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "corpus": "v2 (accent marks corrected)",
        "corpus_claim_basis": (
            "Not asserted by this builder. Cell provenance and corpus identity are "
            "the main-grid module's Slurm-header attestation (carried below); the "
            "reference sha256 in this bundle is a parallel-by-text check and cannot "
            "attest a corpus, because v2 re-renders the same ISO text as v1."
        ),
        "analysis_only": True,
        "no_training_no_inference": True,
        "inputs": {
            "study_root": str(study_root),
            "shared_study_root": shared_study_root,
            "prediction_csvs": [row["prediction_csv"] for row in rows],
            "committed_counts": (
                "each cell's own results.json {rv,ms}_test accent_tp / accent_fp / "
                "accent_fn / accent_*_labeled / n_ref_accent_vowels / "
                "n_hyp_accent_vowels"
            ),
        },
        "method": {
            "aligner": (
                "The accent alignment of scripts.evaluation.accent_metrics, walked in "
                "this module (that file is inside the shared OCR evaluation published "
                "artifact's source lock, so no hook could be added to it). Its "
                "normalization (_prepare_for_jiwer/_coerce_text), channel definition "
                "(_vowel_positions) and charging rule (_charge_paired / "
                "_charge_orphan_ref / _charge_orphan_hyp over _new_counts) are "
                "imported unchanged; only the chunk enumeration is restated, and "
                "alignment_self_check verifies per view that it charges exactly what "
                "compute_accent_metrics charges"
            ),
            "alignment_self_check": aggregate["alignment_self_check"],
            "statistics": (
                "scripts.evaluation.aggregate_accent_ratio_multiseed.summarize (mean, "
                "sample SD, Student-t 95% CI, df=n-1) and _contrast (paired by seed)"
            ),
            "grid_machinery": (
                "scripts.evaluation.build_paclic_v2_maingrid_analysis (cell "
                "discovery, coverage, recipe consistency, corpus attestation, "
                "per-cell integrity)"
            ),
            "text_transform": aggregate["text_transform"],
            "alt_text_transform": aggregate["alt_text_transform"],
            "misplacement": (
                "window-dependent refinement, never a partition member: same-kind "
                f"one-to-one nearest-first pairing within {MISPLACEMENT_WINDOW} "
                "syllable slot(s), always reported beside a density null (the same "
                "pairing after rotating each row's hypothesis marks by half the row) "
                "and at the sensitivity windows "
                f"{list(MISPLACEMENT_SENSITIVITY_WINDOWS)}"
            ),
            "accent_terminology": (
                "acute (high) / grave (falling) / none. Traditional Sanskrit accent "
                "terminology is deliberately not used; scripts/evaluation/"
                "accent_metrics.py records why"
            ),
        },
        "reconciliation": aggregate["reconciliation"],
        "class_definitions": CLASS_DEFINITIONS,
        "diagnostic_counts": DIAGNOSTIC_COUNTS,
        "letter_s_acute_note": LETTER_S_ACUTE_NOTE,
        "status": aggregate["status"],
        "complete": aggregate["complete"],
        "publishable": aggregate["publishable"],
        "warnings": aggregate["warnings"],
        "hard_warnings": aggregate["hard_warnings"],
        "coverage": aggregate["coverage"],
        "corpus_attestation": aggregate["corpus_attestation"],
        "subset_consistency": aggregate["subset_consistency"],
        "grid_integrity": aggregate["grid_integrity"],
        "not_in_this_bundle": {
            "v1_comparison": (
                "None. v1 is abandoned and its corpus is different; no v1 number is "
                "quoted, subtracted or plotted here, and v1's own "
                "error_decomposition.csv is left untouched and reproducible."
            ),
            "figure": (
                "No figure is rendered. The paper's v1 fig5 plotted v1's accent-vs-base "
                "edit-op classes, which are not these classes; a v2 figure would need "
                "its own caption and design decision."
            ),
            "plateau_and_effect_sizes": (
                "Not here. The worst-vs-average plateau table and the endpoint Cohen's "
                "dz / relative effect are the other half of the study and are "
                "unrelated to per-prediction alignment."
            ),
        },
        "generated_by": f"scripts.evaluation.{Path(__file__).stem}",
    }


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float], digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _fmt_count(value: Optional[float]) -> str:
    """A count/denominator mean: bare integer when whole, else 4 dp (never sci-notation)."""
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}"


def _write_rows(fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


PER_CELL_COLUMNS: tuple[str, ...] = (
    "rv_ratio",
    "ms_ratio",
    "seed",
    "view",
    "direction",
    "trained_on_view",
    "error_class",
    "count",
    "per_ref_accent_vowel",
    "n_ref_accent_vowels",
    "n_hyp_accent_vowels",
    "n_samples",
    "reconciles",
)


def render_per_cell_csv(rows: Sequence[dict[str, Any]]) -> str:
    """One row per (cell, view, class): the raw counts every aggregate comes from."""
    out = []
    for row in rows:
        for cls in ERROR_CLASSES:
            out.append(
                {
                    "rv_ratio": row["rv_ratio"],
                    "ms_ratio": row["ms_ratio"],
                    "seed": row["seed"],
                    "view": row["view"],
                    "direction": row["direction"],
                    "trained_on_view": row["trained_on_view"],
                    "error_class": cls,
                    "count": row["counts"][cls],
                    "per_ref_accent_vowel": _fmt(row["per_ref_accent_vowel"][cls]),
                    "n_ref_accent_vowels": row["n_ref_accent_vowels"],
                    "n_hyp_accent_vowels": row["n_hyp_accent_vowels"],
                    "n_samples": row["n_samples"],
                    "reconciles": row["reconciliation"]["reconciles"],
                }
            )
    return _write_rows(PER_CELL_COLUMNS, out)


BY_RATIO_VIEW_COLUMNS: tuple[str, ...] = (
    "rv_ratio",
    "ms_ratio",
    "view",
    "direction",
    "trained_on_view",
    "error_class",
    "n",
    "mean",
    "sd",
    "sem",
    "ci95_lo",
    "ci95_hi",
    "ci95_halfwidth",
    "count_mean",
    "n_ref_accent_vowels",
    "n_samples",
    "seeds",
    "values",
)


def render_by_ratio_view_csv(rows: Sequence[dict[str, Any]]) -> str:
    """Per (ratio, view, class) rate: mean ± Student-t 95% CI across the seeds."""
    out = []
    for row in rows:
        stat = row["per_ref_accent_vowel"]
        out.append(
            {
                "rv_ratio": row["rv_ratio"],
                "ms_ratio": row["ms_ratio"],
                "view": row["view"],
                "direction": row["direction"],
                "trained_on_view": row["trained_on_view"],
                "error_class": row["error_class"],
                "n": stat["n"],
                "mean": _fmt(stat["mean"]),
                "sd": _fmt(stat["sd"]),
                "sem": _fmt(stat["sem"]),
                "ci95_lo": _fmt(stat["ci95_lo"]),
                "ci95_hi": _fmt(stat["ci95_hi"]),
                "ci95_halfwidth": _fmt(stat["ci95_halfwidth"]),
                "count_mean": _fmt(row["count_mean"], 2),
                "n_ref_accent_vowels": _fmt_count(row["n_ref_accent_vowels"]),
                "n_samples": _fmt_count(row["n_samples"]),
                "seeds": ";".join(str(s) for s in row["seeds"]),
                "values": ";".join(f"{v:.6f}" for v in stat["values"]),
            }
        )
    return _write_rows(BY_RATIO_VIEW_COLUMNS, out)


MARK_BUDGET_COLUMNS: tuple[str, ...] = (
    "rv_ratio",
    "ms_ratio",
    "view",
    "direction",
    "trained_on_view",
    "n",
    "hyp_marks_per_ref_accent_mean",
    "hyp_marks_per_ref_accent_ci95_lo",
    "hyp_marks_per_ref_accent_ci95_hi",
    "matched_on_substituted_base_per_ref_accent_mean",
    "n_ref_accent_vowels",
    "n_hyp_accent_vowels",
    "seeds",
    "values",
)


def render_mark_budget_csv(rows: Sequence[dict[str, Any]]) -> str:
    """Marks written per mark owed, per (ratio, view), mean ± Student-t 95% CI."""
    out = []
    for row in rows:
        budget = row["hyp_marks_per_ref_accent"]
        substituted = row["matched_on_substituted_base_per_ref_accent"]
        out.append(
            {
                "rv_ratio": row["rv_ratio"],
                "ms_ratio": row["ms_ratio"],
                "view": row["view"],
                "direction": row["direction"],
                "trained_on_view": row["trained_on_view"],
                "n": budget["n"],
                "hyp_marks_per_ref_accent_mean": _fmt(budget["mean"]),
                "hyp_marks_per_ref_accent_ci95_lo": _fmt(budget["ci95_lo"]),
                "hyp_marks_per_ref_accent_ci95_hi": _fmt(budget["ci95_hi"]),
                "matched_on_substituted_base_per_ref_accent_mean": _fmt(
                    substituted["mean"]
                ),
                "n_ref_accent_vowels": _fmt_count(row["n_ref_accent_vowels"]),
                "n_hyp_accent_vowels": _fmt_count(row["n_hyp_accent_vowels"]),
                "seeds": ";".join(str(s) for s in row["seeds"]),
                "values": ";".join(f"{v:.6f}" for v in budget["values"]),
            }
        )
    return _write_rows(MARK_BUDGET_COLUMNS, out)


MISPLACEMENT_COLUMNS: tuple[str, ...] = (
    "rv_ratio",
    "ms_ratio",
    "view",
    "direction",
    "trained_on_view",
    "window_slots",
    "n",
    "misplaced_mean",
    "misplaced_ci95_halfwidth",
    "shift_null_mean",
    "excess_mean",
    "excess_ci95_lo",
    "excess_ci95_hi",
    "misplaced_count_mean",
    "shift_null_count_mean",
    "other_kind_count_mean",
    "n_ref_accent_vowels",
    "seeds",
)


def render_misplacement_csv(rows: Sequence[dict[str, Any]]) -> str:
    """Misplacement rate, its density null and the excess — never one without the others."""
    out = []
    for row in rows:
        misplaced = row["misplaced_per_ref_accent"]
        null = row["shift_null_per_ref_accent"]
        excess = row["excess_per_ref_accent"]
        out.append(
            {
                "rv_ratio": row["rv_ratio"],
                "ms_ratio": row["ms_ratio"],
                "view": row["view"],
                "direction": row["direction"],
                "trained_on_view": row["trained_on_view"],
                "window_slots": row["window_slots"],
                "n": misplaced["n"],
                "misplaced_mean": _fmt(misplaced["mean"]),
                "misplaced_ci95_halfwidth": _fmt(misplaced["ci95_halfwidth"]),
                "shift_null_mean": _fmt(null["mean"]),
                "excess_mean": _fmt(excess["mean"]),
                "excess_ci95_lo": _fmt(excess["ci95_lo"]),
                "excess_ci95_hi": _fmt(excess["ci95_hi"]),
                "misplaced_count_mean": _fmt(row["misplaced_count_mean"], 2),
                "shift_null_count_mean": _fmt(row["shift_null_count_mean"], 2),
                "other_kind_count_mean": _fmt(row["other_kind_count_mean"], 2),
                "n_ref_accent_vowels": _fmt_count(row["n_ref_accent_vowels"]),
                "seeds": ";".join(str(s) for s in row["seeds"]),
            }
        )
    return _write_rows(MISPLACEMENT_COLUMNS, out)


RECONCILIATION_COLUMNS: tuple[str, ...] = (
    "rv_ratio",
    "ms_ratio",
    "seed",
    "view",
    "identity",
    "expression",
    "derived",
    "committed",
    "delta",
    "checked",
    "ok",
)


def render_reconciliation_csv(rows: Sequence[dict[str, Any]]) -> str:
    """Every identity, per view, with the committed value it was checked against."""
    out = []
    for row in rows:
        for identity in row["reconciliation"]["identities"]:
            out.append(
                {
                    "rv_ratio": row["rv_ratio"],
                    "ms_ratio": row["ms_ratio"],
                    "seed": row["seed"],
                    "view": row["view"],
                    **identity,
                }
            )
    return _write_rows(RECONCILIATION_COLUMNS, out)


CONTRAST_COLUMNS: tuple[str, ...] = (
    "model",
    "rv_ratio",
    "trained_view",
    "untrained_view",
    "error_class",
    "n",
    "trained_mean",
    "untrained_mean",
    "diff_mean",
    "diff_ci95_lo",
    "diff_ci95_hi",
    "sign_consistent",
    "direction",
    "per_seed_diff",
)


def render_contrast_csv(contrasts: dict[str, dict[str, Any]]) -> str:
    """Seed-paired ``untrained − trained`` shift per class, per endpoint model."""
    out = []
    for label, _rv_ratio, _trained, _untrained in ENDPOINT_MODELS:
        block = contrasts.get(label)
        if block is None:
            continue
        for cls in ERROR_CLASSES:
            contrast = block["per_class"][cls]
            diff = contrast["diff"]
            out.append(
                {
                    "model": label,
                    "rv_ratio": block["rv_ratio"],
                    "trained_view": block["trained_view"],
                    "untrained_view": block["untrained_view"],
                    "error_class": cls,
                    "n": diff["n"],
                    "trained_mean": _fmt(contrast["a"]["mean"]),
                    "untrained_mean": _fmt(contrast["b"]["mean"]),
                    "diff_mean": _fmt(diff["mean"]),
                    "diff_ci95_lo": _fmt(diff["ci95_lo"]),
                    "diff_ci95_hi": _fmt(diff["ci95_hi"]),
                    "sign_consistent": contrast["sign_consistent"],
                    "direction": contrast["direction"],
                    "per_seed_diff": ";".join(
                        f"{s}:{d:.6f}"
                        for s, d in sorted(contrast["per_seed_diff"].items())
                    ),
                }
            )
    return _write_rows(CONTRAST_COLUMNS, out)


def render_class_definitions_md() -> str:
    """The class contract: definitions, the two partitions, and the identities."""
    lines = [
        f"# PACLIC v2 accent error decomposition — class definitions (issue #{ISSUE})",
        "",
        "Definitions, not results. Accent kinds are **acute (high)** / **grave "
        "(falling)** / none; traditional Sanskrit accent terminology is deliberately "
        "not used (`scripts/evaluation/accent_metrics.py` records why).",
        "",
        "The accent channel is the set of **vowel-base positions** "
        "(`accent_metrics::_vowel_positions`): a mark counts only where it sits on a "
        "vowel that can carry one, which is also the only reason the reported accent "
        "F1 is well defined.",
        "",
        "| class | reference partition | hypothesis partition | decided by |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for cls in ERROR_CLASSES:
        lines.append(
            f"| `{cls}` | {'yes' if cls in REF_SIDE_CLASSES else '—'} | "
            f"{'yes' if cls in HYP_SIDE_CLASSES else '—'} | {CLASS_DEFINITIONS[cls]} |"
        )
    lines += [
        "",
        "## Diagnostic counts (subsets, never added to a partition)",
        "",
        "| count | meaning |",
        "| :--- | :--- |",
    ]
    for name, meaning in DIAGNOSTIC_COUNTS.items():
        lines.append(f"| `{name}` | {meaning} |")
    lines += [
        "",
        f"> {LETTER_S_ACUTE_NOTE}",
        "",
        "## Misplacement is a refinement, not a class",
        "",
        '"Right kind, wrong syllable" is a relation between two unmatched marks, not a '
        "property of one position, and it depends on how far apart they may sit. The "
        "partition above is therefore window-free, and misplacement is measured on top "
        f"of it: same-kind, one-to-one, nearest-first, within **{MISPLACEMENT_WINDOW} "
        "syllable slot** (`MISPLACEMENT_WINDOW`), each pair consuming one "
        "reference-side and one hypothesis-side member so the partition and its "
        "reconciliation are unchanged. Every misplacement count is reported beside a "
        "**density null** — the identical pairing after each row's hypothesis marks are "
        "rotated by half the row's syllable count — because the pairing saturates when "
        "both pools are dense. Sensitivity windows: "
        f"{list(MISPLACEMENT_SENSITIVITY_WINDOWS)}. {WINDOW_ZERO_NOTE}",
        "",
        "## The identities the decomposition must satisfy",
        "",
        "Checked per view against the values the run itself committed in "
        "`results.json` (not against a recomputation of them):",
        "",
        "| identity | derived as | committed key |",
        "| :--- | :--- | :--- |",
    ]
    for name, classes, key in RECONCILIATION_IDENTITIES:
        lines.append(f"| `{name}` | `{' + '.join(classes)}` | `{key}` |")
    lines += [
        "",
        f"Text transform: {RUNNER_TRANSFORM_LABEL}.",
        "",
        f"Reported for comparison only: {ALT_TRANSFORM_LABEL}.",
        "",
    ]
    return "\n".join(lines)


def _pm_n(stat: dict[str, Any], digits: int = 4) -> str:
    """``mean ± halfwidth (n=k)`` — a value never appears without its own n."""
    if stat["mean"] is None:
        return "—"
    halfwidth = stat["ci95_halfwidth"]
    body = (
        f"{stat['mean']:.{digits}f}"
        if halfwidth is None
        else f"{stat['mean']:.{digits}f} ± {halfwidth:.{digits}f}"
    )
    return f"{body} (n={stat['n']})"


def _stat_for(
    by_ratio_view: Sequence[dict[str, Any]], rv_ratio: float, view: str, cls: str
) -> Optional[dict[str, Any]]:
    for row in by_ratio_view:
        if (
            row["rv_ratio"] == rv_ratio
            and row["view"] == view
            and row["error_class"] == cls
        ):
            return row["per_ref_accent_vowel"]
    return None


def render_results_md(aggregate: dict[str, Any]) -> str:
    """Human-readable results page: status, then the tables, n on every row."""
    view_cov = aggregate["coverage"]
    banner = aggregate["partial_banner"]
    by_ratio_view = aggregate["per_ratio_view"]
    out: list[str] = []
    out.append("# PACLIC v2 accent error decomposition — what kind of accent error")
    out.append("")
    out.append(
        f"**Status:** {aggregate['status']} · **Corpus:** v2 (accent marks "
        f"corrected) · **Views decomposed:** {view_cov['n_views_decomposed']}/"
        f"{view_cov['n_views']} (5 ratios × 3 seeds × 2 test views) · **Seeds "
        f"present:** `{aggregate['grid']['seeds']}` · **Analysis-only** (no "
        "training, no inference, no re-decoding)."
    )
    out.append("")
    if banner is not None:
        out.append(f"> **{banner}**")
        out.append("")
    other_warnings = [w for w in aggregate["warnings"] if w != banner]
    if other_warnings:
        out.append("> **Integrity warnings:**")
        for warning in other_warnings:
            out.append(f"> - {warning}")
        out.append("")
    self_check = aggregate["alignment_self_check"]
    out.append(
        "Every number here is a partition of the accent true/false positives and "
        "false negatives each cell already committed. The alignment is "
        "`accent_metrics`' own — its normalization, its vowel-anchored channel and "
        "its charging functions are imported unchanged — but the chunk walk is "
        "restated in this module, because `scripts/evaluation/accent_metrics.py` is "
        "inside the shared metric implementation's source lock and cannot be given "
        "the hook this analysis would otherwise use. "
        f"**{self_check['n_matching_compute_accent_metrics']}/"
        f"{self_check['n_views']} views charge exactly what "
        "`compute_accent_metrics` charges** on the same text, which is the guard that "
        "replaces the shared code path; §4 then checks the partition against "
        "`results.json` view by view. Statistics (mean, sample SD, "
        "Student-t 95% CI with df=n−1) and the paired contrasts are reused verbatim "
        "from `scripts.evaluation.aggregate_accent_ratio_multiseed`. Class "
        "definitions: `tables/class_definitions.md`. **No v1 number appears here** — "
        "v1 is a different image corpus and is neither quoted nor subtracted."
    )
    out.append("")

    out.append("## 0. Coverage")
    out.append("")
    out.append("| RV:MS | views decomposed | RV-view seeds | MS-view seeds |")
    out.append("| :--- | ---: | :--- | :--- |")
    for blk in view_cov["per_ratio"].values():
        rv_pct = int(round(blk["rv_ratio"] * 100))
        ms_pct = int(round(blk["ms_ratio"] * 100))
        out.append(
            f"| {rv_pct}:{ms_pct} | {blk['n_views_decomposed']}/{blk['n_views']} | "
            f"{blk['seeds_per_view']['rv'] or '—'} | "
            f"{blk['seeds_per_view']['ms'] or '—'} |"
        )
    out.append("")
    if view_cov["skipped_views_missing_predictions"]:
        out.append(
            "Present cells whose prediction CSVs are absent (skipped, never imputed): "
            + ", ".join(
                f"`{name}`" for name in view_cov["skipped_views_missing_predictions"]
            )
        )
        out.append("")

    out.append("## 1. Per mix ratio and test view — share of the reference's accents")
    out.append("")
    out.append(
        "Every cell is rate-normalised by its own `n_ref_accent_vowels` (the "
        "reference accents the channel can score), then summarized across seeds. The "
        "first four classes partition the reference's accents and sum to 1; "
        "`extra_*` are hypothesis-side marks the reference does not have, so they are "
        "rates against the same denominator, not shares of it."
    )
    out.append("")
    header = "| RV:MS | view | " + " | ".join(f"`{c}`" for c in ERROR_CLASSES) + " |"
    out.append(header)
    out.append("| :--- | :--- | " + " | ".join("---:" for _ in ERROR_CLASSES) + " |")
    for rv_ratio in mg.RV_RATIOS:
        for view in VIEWS:
            cells_md = [
                _pm_n(stat, 4)
                if (stat := _stat_for(by_ratio_view, rv_ratio, view, cls)) is not None
                else "—"
                for cls in ERROR_CLASSES
            ]
            if all(value == "—" for value in cells_md):
                continue
            rv_pct = int(round(rv_ratio * 100))
            ms_pct = int(round((1.0 - rv_ratio) * 100))
            out.append(f"| {rv_pct}:{ms_pct} | {view} | " + " | ".join(cells_md) + " |")
    out.append("")
    out.append("Machine-readable: `tables/by_ratio_view.csv`, `tables/per_cell.csv`.")
    out.append("")

    out.append("### 1.1 How many marks were written per mark owed")
    out.append("")
    out.append(
        "`hyp_marks_per_ref_accent` = `n_hyp_accent_vowels` / `n_ref_accent_vowels`. "
        "**Read this before reading §1 as omission.** A collapsed accent F1 is "
        "consistent with two opposite behaviours — a channel that stopped writing "
        "marks, and one that kept writing about as many in unrelated places — and only "
        "this ratio separates them. `matched on substituted base` is the share of the "
        "channel's credited matches that sit on a letter the base channel got wrong "
        "(the accent channel taking credit where the akṣara is not right)."
    )
    out.append("")
    out.append(
        "| RV:MS | view | trained on view | marks written / marks owed | "
        "matched on substituted base |"
    )
    out.append("| :--- | :--- | :--- | ---: | ---: |")
    for row in aggregate["mark_budget_per_ratio_view"]:
        rv_pct = int(round(row["rv_ratio"] * 100))
        ms_pct = int(round(row["ms_ratio"] * 100))
        out.append(
            f"| {rv_pct}:{ms_pct} | {row['view']} | {row['trained_on_view']} | "
            f"{_pm_n(row['hyp_marks_per_ref_accent'], 4)} | "
            f"{_pm_n(row['matched_on_substituted_base_per_ref_accent'], 4)} |"
        )
    out.append("")
    out.append("Machine-readable: `tables/mark_budget.csv`.")
    out.append("")

    out.append(
        "## 2. Trained versus untrained notation (endpoint models, paired by seed)"
    )
    out.append("")
    out.append(
        'The two single-notation endpoints are the only cells where "the notation '
        'this model never trained on" is a property of the mix rather than a matter '
        "of degree. Δ = untrained view − trained view, differenced within a seed then "
        "summarized; **positive means the class is more frequent on the notation the "
        "model never saw**."
    )
    out.append("")
    out.append(
        "| model | class | trained view | untrained view | Δ | sign consistent |"
    )
    out.append("| :--- | :--- | ---: | ---: | ---: | :--- |")
    for label, _rv, _t, _u in ENDPOINT_MODELS:
        block = aggregate["trained_vs_untrained"].get(label)
        if block is None:
            continue
        for cls in ERROR_CLASSES:
            contrast = block["per_class"][cls]
            diff = contrast["diff"]
            if diff["mean"] is None:
                continue
            out.append(
                f"| `{label}` ({block['trained_view']}→{block['untrained_view']}) | "
                f"`{cls}` | {_pm_n(contrast['a'], 4)} | {_pm_n(contrast['b'], 4)} | "
                f"{diff['mean']:+.4f}"
                + (
                    ""
                    if diff["ci95_halfwidth"] is None
                    else f" ± {diff['ci95_halfwidth']:.4f}"
                )
                + f" | {contrast['sign_consistent']} |"
            )
    out.append("")
    out.append("Machine-readable: `tables/trained_vs_untrained.csv`.")
    out.append("")

    out.append("## 3. Misplacement — the refinement, always beside its density null")
    out.append("")
    out.append(
        f"Same-kind one-to-one pairing within {MISPLACEMENT_WINDOW} syllable slot, "
        "as a share of the reference's accents. `null` is the identical pairing after "
        "each row's hypothesis marks are rotated by half the row: what adjacency "
        "yields from density alone. **Only the excess carries information**, and when "
        "the two pools are both large the pairing saturates — a large `misplaced` with "
        "a `null` beside it is not evidence of a placement mechanism. This is a "
        "locality control, not a significance test; no p-value is derived from it."
    )
    out.append("")
    out.append("| RV:MS | view | trained on view | misplaced | null | excess |")
    out.append("| :--- | :--- | :--- | ---: | ---: | ---: |")
    for row in aggregate["misplacement_per_ratio_view"]:
        rv_pct = int(round(row["rv_ratio"] * 100))
        ms_pct = int(round(row["ms_ratio"] * 100))
        out.append(
            f"| {rv_pct}:{ms_pct} | {row['view']} | {row['trained_on_view']} | "
            f"{_pm_n(row['misplaced_per_ref_accent'], 4)} | "
            f"{_pm_n(row['shift_null_per_ref_accent'], 4)} | "
            f"{_pm_n(row['excess_per_ref_accent'], 4)} |"
        )
    out.append("")
    out.append(
        "Sensitivity to the window (counts only, in `aggregate.json` "
        f"`misplacement_sensitivity`): windows {list(MISPLACEMENT_SENSITIVITY_WINDOWS)}. "
        f"{WINDOW_ZERO_NOTE} Machine-readable: `tables/misplacement.csv`."
    )
    out.append("")

    rec = aggregate["reconciliation"]
    out.append("## 4. Reconciliation against the committed per-cell counts")
    out.append("")
    out.append(
        f"**{rec['n_reconciling']}/{rec['n_views']} decomposed views satisfy all "
        f"{len(RECONCILIATION_IDENTITIES)} identities exactly.** The right-hand sides "
        "are the values each run committed in its own `results.json` — not a "
        "recomputation of them — so this checks the decomposition against the "
        "published numbers. A failing view is a publication-blocking warning: a "
        "decomposition that does not add up is worse than none."
    )
    out.append("")
    out.append("| identity | derived as | committed key |")
    out.append("| :--- | :--- | :--- |")
    for identity in rec["identities"]:
        out.append(
            f"| `{identity['identity']}` | `{identity['expression']}` | "
            f"`{identity['committed_key']}` |"
        )
    out.append("")
    out.append(
        "Reproducing the committed counts from the saved CSVs needs the transform the "
        f"runner used — {RUNNER_TRANSFORM_LABEL} — because the CSVs store the raw "
        "decoder output. The study evaluator's NFC-and-strip convention is a different "
        "transform and does not reproduce them; what it would have counted is in "
        "`aggregate.json` `per_cell[].alt_normalization`, which is also the origin of "
        "the main grid's non-zero cross-check |Δaccent F1|. Per-identity detail: "
        "`tables/reconciliation.csv`."
    )
    out.append("")

    identity = aggregate["reference_set_identity"]
    out.append("## 5. What the reference set can and cannot attest")
    out.append("")
    out.append(
        f"`n_distinct_reference_sequences` = {identity['n_distinct_reference_sequences']}"
        f" across all {rec['n_views']} decomposed views "
        f"(sha256 `{', '.join(h[:12] for h in identity['reference_sha256'])}…`). "
        + identity["note"]
    )
    out.append("")
    if aggregate["per_cell"]:
        first = aggregate["per_cell"][0]
        out.append(
            "Accent-mark census on the reference (identical for every view, being one "
            f"reference set): {first['n_ref_accent_marks_nfd']} accent code points in "
            f"NFD, of which {first['n_ref_accent_marks_on_letter_s']} are the second "
            "half of the ISO letter `ś` (`s` + U+0301) and "
            f"{first['n_ref_accent_vowels']} sit on a vowel base; "
            f"{first['n_ref_accent_marks_unanchored_other']} are neither. "
            + LETTER_S_ACUTE_NOTE
        )
        out.append("")
    out.append(
        "Corpus identity is **not** taken from a path name or from this hash: it is "
        "the main-grid module's attestation from the launcher's immutable Slurm `.out` "
        "header, carried in `manifest.json` → `corpus_attestation`."
    )
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Bundle.
# --------------------------------------------------------------------------- #
def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def build_bundle(
    rows: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
    out_dir: Path,
    *,
    study_root: Path,
    shared_study_root: Optional[str] = None,
) -> None:
    """Write ``tables/`` + ``aggregate.json`` + ``manifest.json`` + ``results.md``."""
    tables = out_dir / "tables"
    _write(tables / "per_cell.csv", render_per_cell_csv(rows))
    _write(
        tables / "by_ratio_view.csv",
        render_by_ratio_view_csv(aggregate["per_ratio_view"]),
    )
    _write(
        tables / "mark_budget.csv",
        render_mark_budget_csv(aggregate["mark_budget_per_ratio_view"]),
    )
    _write(
        tables / "misplacement.csv",
        render_misplacement_csv(aggregate["misplacement_per_ratio_view"]),
    )
    _write(tables / "reconciliation.csv", render_reconciliation_csv(rows))
    _write(
        tables / "trained_vs_untrained.csv",
        render_contrast_csv(aggregate["trained_vs_untrained"]),
    )
    _write(tables / "class_definitions.md", render_class_definitions_md())
    _write(out_dir / "results.md", render_results_md(aggregate))
    _write_json(out_dir / "aggregate.json", aggregate)
    _write_json(
        out_dir / "manifest.json",
        build_manifest(
            aggregate,
            study_root=study_root,
            shared_study_root=shared_study_root,
            rows=rows,
        ),
    )


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
        help="The v2 main-grid study root holding the 15 run dirs and their "
        "prediction CSVs.",
    )
    parser.add_argument(
        "--shared-study-root",
        default=DEFAULT_SHARED_STUDY_ROOT,
        help="Optional path recorded in provenance for the v2 cells.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Bundle dir: writes tables/, results.md, aggregate.json, manifest.json. "
            f"Omit to print the results page to stdout. Conventional: {DEFAULT_OUT_DIR}"
        ),
    )
    parser.add_argument(
        "--slurm-log-dir",
        type=Path,
        default=DEFAULT_SLURM_LOG_DIR,
        help=(
            "Dir of immutable Slurm .out logs, used to attest each cell's corpus from "
            "the launcher's header echo. Absent dir → every cell 'unattested' (never "
            "assumed)."
        ),
    )
    parser.add_argument(
        "--no-slurm-attestation",
        action="store_true",
        help="Skip the Slurm-log corpus attestation entirely.",
    )
    parser.add_argument(
        "--misplacement-window",
        type=int,
        default=MISPLACEMENT_WINDOW,
        help=(
            "Syllable slots within which an unmatched reference mark and an unmatched "
            "hypothesis mark of the same kind are paired as one misplacement. The "
            f"pinned policy is {MISPLACEMENT_WINDOW}; the class partition does not "
            "depend on it."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cells, _checks = mg.load_grid(
        args.study_root,
        metrics_fn=_no_generic_metrics,
        cross_check=False,
        shared_study_root=args.shared_study_root,
    )
    coverage = mg.coverage_summary(cells)
    subset = mg.subset_consistency(mg.subset_provenance(args.study_root, cells), cells)
    integrity = mg.grid_integrity(cells, study_root=args.study_root)
    attestation = (
        None
        if args.no_slurm_attestation
        else mg.corpus_attestation_summary(
            mg.attest_corpus_from_slurm_logs(args.slurm_log_dir),
            cells,
            study_root=args.study_root,
        )
    )

    rows, skipped = decompose_grid(
        cells, args.study_root, window=args.misplacement_window
    )
    view_cov = view_coverage(rows, skipped, coverage)
    aggregate = build_aggregate(
        cells,
        rows,
        coverage,
        view_cov,
        subset=subset,
        attestation=attestation,
        integrity=integrity,
    )
    for warning in aggregate["warnings"]:
        print(f"[{STUDY}][warn] {warning}", file=sys.stderr)

    if args.out_dir is not None:
        build_bundle(
            rows,
            aggregate,
            args.out_dir,
            study_root=args.study_root,
            shared_study_root=args.shared_study_root,
        )
        print(f"[{STUDY}] bundle -> {args.out_dir}", file=sys.stderr)
    else:
        print(render_results_md(aggregate))

    return 1 if aggregate["hard_warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
