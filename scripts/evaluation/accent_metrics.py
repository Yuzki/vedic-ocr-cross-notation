"""Accent metrics: PACLIC 2026 v2 release."""

from __future__ import annotations

import unicodedata
from typing import Iterable, Literal, Optional

import jiwer
import pandas as pd

# The two accent marks the romanized ISO corpus actually writes. Named by what
# they are, not by traditional Sanskrit accent terminology: those terms were
# deliberately dropped from this codebase because mapping them onto the two ISO
# marks is ambiguous and caused real errors (the previous comment here called
# GRAVE "anudatta", which is wrong — a domain expert confirmed the ISO grave is
# the falling accent, and the corpus's rendering rules treat it as such).
ACUTE = "́"  # combining acute — HIGH accent
GRAVE = "̀"  # combining grave — FALLING accent

_VOCALIC_BELOW = "̥"  # COMBINING RING BELOW, marks vocalic r̥ / l̥
_MACRON = "̄"  # COMBINING MACRON, marks long vowels in NFD form

# Simple base vowels in ISO 15919 (NFD precomposed bases).
_SIMPLE_VOWEL_BASES = frozenset("aāiīuūeoĀĪŪĒŌ")

AccentLabel = Optional[Literal["acute", "grave"]]


def strip_iso_accents(text: str) -> str:
    """Remove combining acute / grave marks from ISO text."""
    nfd = unicodedata.normalize("NFD", text)
    cleaned = "".join(ch for ch in nfd if ch != ACUTE and ch != GRAVE)
    return unicodedata.normalize("NFC", cleaned)


def _is_vowel_base(text: str, idx: int) -> bool:
    """Return ``True`` if ``text[idx]`` is the base of a vowel grapheme.

    Handles vocalic ``r̥`` / ``l̥`` by peeking ahead for ``\\u0325``: a bare
    consonant ``r`` (or ``l``) without that combining mark is *not* counted
    as a vowel base, since it cannot legitimately carry an accent mark in
    Vedic ISO transliteration.
    """
    ch = text[idx]
    if ch in _SIMPLE_VOWEL_BASES:
        return True
    if ch in ("r", "l", "R", "L"):
        return idx + 1 < len(text) and text[idx + 1] == _VOCALIC_BELOW
    return False


def _accent_label_after(text: str, idx: int) -> AccentLabel:
    """Inspect combining marks following ``text[idx]`` for an accent.

    Walks forward over combining characters until a non-combining character
    is encountered. ``\\u0301`` is the HIGH accent and ``\\u0300`` the FALLING
    accent; other combining marks (e.g. ``\\u0325`` for the
    vocalic ring, ``\\u0304`` for macron) are ignored.
    """
    j = idx + 1
    while j < len(text) and unicodedata.combining(text[j]):
        if text[j] == ACUTE:
            return "acute"
        if text[j] == GRAVE:
            return "grave"
        j += 1
    return None


def _vowel_positions(text_nfd: str) -> dict[int, AccentLabel]:
    """Map code-point index → accent label for every vowel-base position."""
    positions: dict[int, AccentLabel] = {}
    for i, ch in enumerate(text_nfd):
        if unicodedata.combining(ch):
            continue
        if _is_vowel_base(text_nfd, i):
            positions[i] = _accent_label_after(text_nfd, i)
    return positions


def iter_grapheme_clusters(text: str) -> Iterable[str]:
    """Yield base + trailing combining marks as one cluster.

    Implemented in pure stdlib so we do not pull in ``regex``. Equivalent to
    ``regex.findall(r"\\X", text)`` for the alphabets we encounter, since
    ISO 15919 does not use surrogate pairs or RTL controls.
    """
    text = unicodedata.normalize("NFD", text)
    if not text:
        return
    buf = text[0]
    for ch in text[1:]:
        if unicodedata.combining(ch):
            buf += ch
        else:
            yield buf
            buf = ch
    yield buf


def is_iso_vowel_grapheme(grapheme: str) -> bool:
    """Return ``True`` if the grapheme cluster's base is an ISO vowel.

    Normalises the input to NFD so NFC-precomposed forms like ``á`` (U+00E1)
    are recognised via their decomposed base ``a``.
    """
    if not grapheme:
        return False
    nfd = unicodedata.normalize("NFD", grapheme)
    return _is_vowel_base(nfd, 0)


def grapheme_accent_label(grapheme: str) -> AccentLabel:
    """Return the accent label (acute/grave/None) of a grapheme cluster."""
    if not grapheme:
        return None
    nfd = unicodedata.normalize("NFD", grapheme)
    return _accent_label_after(nfd, 0)


def _coerce_text(text: object) -> str:
    """Treat NaN / None as empty so callers can pass raw DataFrame columns."""
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    return str(text)


def _prepare_for_jiwer(text: str) -> str:
    """NFC-normalise then NFD-decompose so accents live as separate marks.

    ``jiwer`` requires non-empty strings; callers must filter empty pairs
    before invoking it.
    """
    return unicodedata.normalize("NFD", unicodedata.normalize("NFC", text))


def _accumulate_counts(
    ref_nfd: str,
    hyp_nfd: str,
    chunks,
    counts: dict[str, dict[str, int]],
) -> None:
    """Walk one aligned pair, charging TP/FP/FN into ``counts``.

    ``counts`` carries two sub-tables: ``"unlabeled"`` (acute vs grave merged)
    and ``"labeled"`` (acute and grave kept distinct).
    """
    ref_positions = _vowel_positions(ref_nfd)
    hyp_positions = _vowel_positions(hyp_nfd)

    counts["unlabeled"]["n_ref_accent_vowels"] += sum(
        1 for label in ref_positions.values() if label is not None
    )
    counts["unlabeled"]["n_hyp_accent_vowels"] += sum(
        1 for label in hyp_positions.values() if label is not None
    )
    counts["labeled"]["n_ref_accent_vowels"] = counts["unlabeled"][
        "n_ref_accent_vowels"
    ]
    counts["labeled"]["n_hyp_accent_vowels"] = counts["unlabeled"][
        "n_hyp_accent_vowels"
    ]

    for chunk in chunks:
        ctype = chunk.type
        r0, r1 = chunk.ref_start_idx, chunk.ref_end_idx
        h0, h1 = chunk.hyp_start_idx, chunk.hyp_end_idx
        if ctype in ("equal", "substitute"):
            for k in range(r1 - r0):
                ri = r0 + k
                hi = h0 + k
                _charge_paired(ri, hi, ref_positions, hyp_positions, counts)
        elif ctype == "delete":
            for ri in range(r0, r1):
                _charge_orphan_ref(ri, ref_positions, counts)
        elif ctype == "insert":
            for hi in range(h0, h1):
                _charge_orphan_hyp(hi, hyp_positions, counts)


def _charge_paired(
    ref_idx: int,
    hyp_idx: int,
    ref_positions: dict[int, AccentLabel],
    hyp_positions: dict[int, AccentLabel],
    counts: dict[str, dict[str, int]],
) -> None:
    ref_is_vowel = ref_idx in ref_positions
    hyp_is_vowel = hyp_idx in hyp_positions
    if not ref_is_vowel and not hyp_is_vowel:
        return
    ref_label = ref_positions.get(ref_idx)
    hyp_label = hyp_positions.get(hyp_idx)

    if ref_label is not None and hyp_label is not None:
        # Unlabeled: any accent on both sides ⇒ TP.
        counts["unlabeled"]["tp"] += 1
        # Labeled: must match exactly; mismatch becomes FN + FP.
        if ref_label == hyp_label:
            counts["labeled"]["tp"] += 1
        else:
            counts["labeled"]["fn"] += 1
            counts["labeled"]["fp"] += 1
    elif ref_label is not None:
        counts["unlabeled"]["fn"] += 1
        counts["labeled"]["fn"] += 1
    elif hyp_label is not None:
        counts["unlabeled"]["fp"] += 1
        counts["labeled"]["fp"] += 1


def _charge_orphan_ref(
    ref_idx: int,
    ref_positions: dict[int, AccentLabel],
    counts: dict[str, dict[str, int]],
) -> None:
    label = ref_positions.get(ref_idx)
    if label is None:
        return
    counts["unlabeled"]["fn"] += 1
    counts["labeled"]["fn"] += 1


def _charge_orphan_hyp(
    hyp_idx: int,
    hyp_positions: dict[int, AccentLabel],
    counts: dict[str, dict[str, int]],
) -> None:
    label = hyp_positions.get(hyp_idx)
    if label is None:
        return
    counts["unlabeled"]["fp"] += 1
    counts["labeled"]["fp"] += 1


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def _finalise_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_ref_accent_vowels": counts["n_ref_accent_vowels"],
        "n_hyp_accent_vowels": counts["n_hyp_accent_vowels"],
    }


def _new_counts() -> dict[str, dict[str, int]]:
    base = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "n_ref_accent_vowels": 0,
        "n_hyp_accent_vowels": 0,
    }
    return {"unlabeled": dict(base), "labeled": dict(base)}


def compute_accent_metrics(
    predictions: Iterable[object],
    references: Iterable[object],
) -> dict[str, dict[str, float]]:
    """Compute alignment-based accent metrics for unlabeled and labeled modes.

    Returns a dict with two sub-dicts (``"unlabeled"``, ``"labeled"``), each
    carrying ``tp``/``fp``/``fn`` along with ``precision``/``recall``/``f1``
    and total accented-vowel counts for the reference and hypothesis sides.
    """
    pred_list = [_coerce_text(p) for p in predictions]
    ref_list = [_coerce_text(r) for r in references]
    if len(pred_list) != len(ref_list):
        raise ValueError(
            f"predictions/references length mismatch: {len(pred_list)} vs {len(ref_list)}"
        )

    counts = _new_counts()

    pairs_for_jiwer: list[tuple[int, str, str]] = []
    for i, (ref, hyp) in enumerate(zip(ref_list, pred_list)):
        ref_nfd = _prepare_for_jiwer(ref)
        hyp_nfd = _prepare_for_jiwer(hyp)
        if not ref_nfd and not hyp_nfd:
            continue
        if not ref_nfd:
            hyp_positions = _vowel_positions(hyp_nfd)
            n_acc_hyp = sum(1 for label in hyp_positions.values() if label is not None)
            counts["unlabeled"]["n_hyp_accent_vowels"] += n_acc_hyp
            counts["labeled"]["n_hyp_accent_vowels"] += n_acc_hyp
            for hi in range(len(hyp_nfd)):
                _charge_orphan_hyp(hi, hyp_positions, counts)
            continue
        if not hyp_nfd:
            ref_positions = _vowel_positions(ref_nfd)
            n_acc_ref = sum(1 for label in ref_positions.values() if label is not None)
            counts["unlabeled"]["n_ref_accent_vowels"] += n_acc_ref
            counts["labeled"]["n_ref_accent_vowels"] += n_acc_ref
            for ri in range(len(ref_nfd)):
                _charge_orphan_ref(ri, ref_positions, counts)
            continue
        pairs_for_jiwer.append((i, ref_nfd, hyp_nfd))

    if pairs_for_jiwer:
        refs_in = [r for _, r, _ in pairs_for_jiwer]
        hyps_in = [h for _, _, h in pairs_for_jiwer]
        out = jiwer.process_characters(refs_in, hyps_in)
        for (_, ref_nfd, hyp_nfd), chunks in zip(pairs_for_jiwer, out.alignments):
            _accumulate_counts(ref_nfd, hyp_nfd, chunks, counts)

    return {
        "unlabeled": _finalise_counts(counts["unlabeled"]),
        "labeled": _finalise_counts(counts["labeled"]),
    }


def _cer(predictions: list[str], references: list[str]) -> float:
    """Compute jiwer CER, dropping only empty *references*; matches OCRMetrics
    behaviour .

    Empty predictions are kept and charged as full deletions (a blank prediction
    is a total failure, not a row to skip). An all-empty-reference corpus reports
    ``1.0`` to mirror :mod:`scripts.training.trocr.metrics`.
    """
    pairs = [
        (p.strip(), r.strip()) for p, r in zip(predictions, references) if r.strip()
    ]
    if not pairs:
        return 1.0
    preds, refs = zip(*pairs)
    return jiwer.cer(list(refs), list(preds))


def compute_all_accent_aware_metrics(
    predictions: Iterable[object],
    references: Iterable[object],
) -> dict[str, float]:
    """Return a flat dict with full CER, accent-stripped CER, and accent F1.

    The keys are flat so callers (``experiment_accent_ratio.py``) can merge
    them directly into ``results.json`` and WandB ``final_metrics``.
    """
    pred_list = [_coerce_text(p) for p in predictions]
    ref_list = [_coerce_text(r) for r in references]

    accent = compute_accent_metrics(pred_list, ref_list)

    full_cer = _cer(pred_list, ref_list)
    stripped_preds = [strip_iso_accents(p) for p in pred_list]
    stripped_refs = [strip_iso_accents(r) for r in ref_list]
    stripped_cer = _cer(stripped_preds, stripped_refs)

    unlab = accent["unlabeled"]
    lab = accent["labeled"]
    return {
        "full_cer": full_cer,
        "accent_stripped_cer": stripped_cer,
        "accent_precision": unlab["precision"],
        "accent_recall": unlab["recall"],
        "accent_f1": unlab["f1"],
        "accent_tp": unlab["tp"],
        "accent_fp": unlab["fp"],
        "accent_fn": unlab["fn"],
        "accent_precision_labeled": lab["precision"],
        "accent_recall_labeled": lab["recall"],
        "accent_f1_labeled": lab["f1"],
        "accent_tp_labeled": lab["tp"],
        "accent_fp_labeled": lab["fp"],
        "accent_fn_labeled": lab["fn"],
        "n_ref_accent_vowels": unlab["n_ref_accent_vowels"],
        "n_hyp_accent_vowels": unlab["n_hyp_accent_vowels"],
    }


def main() -> None:
    """CLI: ``python -m scripts.evaluation.accent_metrics --csv <predictions>``.

    Expects ISO 15919 ``Prediction`` / ``Label`` columns, matching the output
    written by ``scripts.training.trocr.test``.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Predictions CSV path")
    parser.add_argument(
        "--prediction-col",
        default="Prediction",
        help="Column name for predictions (ISO 15919)",
    )
    parser.add_argument(
        "--label-col",
        default="Label",
        help="Column name for references (ISO 15919)",
    )
    parser.add_argument("--json", help="Optional path to write metrics JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.prediction_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(
            f"CSV must contain '{args.prediction_col}' and '{args.label_col}' columns; "
            f"found {list(df.columns)}"
        )

    metrics = compute_all_accent_aware_metrics(
        df[args.prediction_col], df[args.label_col]
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
