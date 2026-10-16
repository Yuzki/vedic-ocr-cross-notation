"""Prediction metrics: PACLIC 2026 v2 release."""

from __future__ import annotations

import csv
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

#: Unicode normalization forms accepted by :class:`Normalization`.
NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]

# --------------------------------------------------------------------------- #
# Versioned prediction schema.
# --------------------------------------------------------------------------- #
#: Current prediction-file schema version. Bump on any breaking column change.
PREDICTION_SCHEMA_VERSION = 1

#: v1 prediction CSV columns, in order (one row per evaluated sample).
PREDICTION_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "sample_id",
    "reference",
    "prediction",
    "font",
    "condition",
    "fraction",
    "seed",
    "view",
    "split",
    "decode_failed",
    "error_type",
    "error_message",
)

#: Legacy study three-column prediction CSV, kept for reader back-compatibility.
LEGACY_PREDICTION_COLUMNS: tuple[str, ...] = ("sample_id", "reference", "prediction")

#: Per-file-constant run axes carried on every row (populated from the runner).
AXIS_FIELDS: tuple[str, ...] = ("condition", "fraction", "seed", "view", "split")


class PredictionSchemaError(Exception):
    """Raised when a prediction file violates the versioned prediction schema."""


@dataclass(frozen=True)
class PredictionRow:
    """One evaluated sample from a prediction file.

    ``reference`` / ``prediction`` are the raw saved strings (never ``None``;
    an empty ``prediction`` is a real empty output). All metadata is optional so
    a legacy three-column file loads with metadata left ``None`` / ``False`` and
    ``is_legacy=True``.
    """

    reference: str
    prediction: str
    sample_id: Optional[str] = None
    font: Optional[str] = None
    condition: Optional[str] = None
    fraction: Optional[int] = None
    seed: Optional[int] = None
    view: Optional[str] = None
    split: Optional[str] = None
    decode_failed: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    schema_version: Optional[int] = None
    is_legacy: bool = False


# --------------------------------------------------------------------------- #
# Normalization + reliability policies (explicit, shared across all conditions).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Normalization:
    """Explicit text normalization applied before every metric.

    ``form`` is a Unicode normalization form (``NFC`` by default — the DESIGN
    ISO 15919 canonical form). ``collapse_whitespace`` folds internal whitespace
    runs to single spaces; ``strip`` removes leading/trailing whitespace. The
    same instance is used for the overall corpus, per-font subsets, and any
    re-evaluation, so metric behaviour is identical across conditions.
    """

    form: NormalizationForm = "NFC"
    strip: bool = True
    collapse_whitespace: bool = False

    def apply(self, text: str) -> str:
        s = unicodedata.normalize(self.form, text)
        if self.collapse_whitespace:
            s = " ".join(s.split())
        if self.strip:
            s = s.strip()
        return s

    def as_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "strip": self.strip,
            "collapse_whitespace": self.collapse_whitespace,
        }


@dataclass(frozen=True)
class OverGenerationPolicy:
    """When a prediction counts as *over-generated* (runaway decoding).

    A **non-empty-reference** row is flagged when
    ``pred_len > length_ratio * ref_len + abs_margin``, where ``ref_len`` and
    ``pred_len`` are the **NFC code-point lengths of the normalized** reference
    and prediction (the strings after :data:`DEFAULT_NORMALIZATION`, i.e. NFC +
    strip — *not* NFD code points and *not* grapheme clusters). Empty references
    (``ref_len <= 0``) are never flagged and are excluded from the rate
    denominator (:func:`_reliability_block`); no special case is made for very
    short references — the ratio test applies uniformly (the study corpus's
    shortest reference is 19 code points, so ``abs_margin`` is a no-op there).

    The **reviewed, final** policy for the study OCR study is
    ``length_ratio=2.0, abs_margin=0`` (:data:`REVIEWED_OVER_GENERATION_POLICY`)
    — numerically the value every committed ``metrics/standard_test.json``
    already reports as ``over_generation_policy``, so finalizing it 
    changes no reported number. Threshold sensitivity is in
    ``the evaluation policy below`` §4.4. The knobs stay configurable so
    the same evaluator can be reused under a different policy.
    """

    length_ratio: float = 2.0
    abs_margin: int = 0

    def is_over_generation(self, ref_len: int, pred_len: int) -> bool:
        if ref_len <= 0:
            return False
        return pred_len > self.length_ratio * ref_len + self.abs_margin

    def as_dict(self) -> dict[str, Any]:
        return {"length_ratio": self.length_ratio, "abs_margin": self.abs_margin}


#: Diacritic inventory as NFD combining-mark code points, per class. A
#: conservative first pass over the Vedic/ISO marks that matter; ``dot_below``
#: deliberately conflates retroflex consonants, anusvāra, and visarga (they all
#: carry U+0323 in NFD). A reviewed, disambiguated inventory for claim tables is
#: an INJECTABLE open decision (see ``docs/experiments/axis-a-preadaptation-staged-learning/contracts/ms_transfer_evaluation.md``).
DEFAULT_DIACRITIC_CLASSES: dict[str, tuple[str, ...]] = {
    "acute_udatta": ("́",),
    "grave_anudatta": ("̀",),
    "macron_long_vowel": ("̄",),
    "dot_below": ("̣",),
    "ring_below_vocalic": ("̥",),
    "candrabindu": ("̐",),
}

DEFAULT_NORMALIZATION = Normalization()
DEFAULT_OVER_GENERATION_POLICY = OverGenerationPolicy()

#: Default cap on the number of substitution pairs recorded in metrics JSON.
DEFAULT_TOP_SUBSTITUTIONS = 20


# --------------------------------------------------------------------------- #
# Reviewed, finalized evaluation policies .
# --------------------------------------------------------------------------- #
# These promote the study provisional evaluator knobs to the reviewed policies
# used for the study OCR paper, versioned here in code (not only prose).
# They are derived + justified in the machine-readable bundle
# ``docs/experiments/axis-a-preadaptation-staged-learning/OCR-evaluation-policy/``
# (regenerated by ``scripts.evaluation.build_evaluation_policy_bundle``) and
# documented in ``the evaluation policy below`` §4.4.

#: Tag for the reviewed evaluation policies pinned in study. Bump on any change
#: to the over-generation criterion or the rare-character rule below.
EVALUATION_POLICY_VERSION = "paclic-2026-v2"

#: Reviewed, final over-generation policy : unit = NFC code points
#: of the normalized strings, ``pred_len > 2.0 * ref_len`` for non-empty
#: references. Numerically identical to the study default every committed metrics
#: JSON already reports, so finalizing it changes no headline or reliability
#: number. Threshold sensitivity (ratios 1.5–3.0 all hold the corrected-grid
#: rate at 1/587, the single shortest reference ``2-1-16``) is committed in the
#: bundle's ``over_generation_sensitivity.csv``; the exact ratio is immaterial.
REVIEWED_OVER_GENERATION_POLICY = OverGenerationPolicy(length_ratio=2.0, abs_margin=0)

#: Operational rarity rule for the reviewed rare-character inventory (study), over
#: the **NFD grapheme clusters** of the normalized reference corpus (the unit
#: :func:`_rare_character_block` matches). A grapheme is *rare* when its corpus
#: occurrence count is ``<=`` this ceiling — calibrated to the natural frequency
#: gap between 8 and 21 in the study MS standard_test corpus (a documented
#: judgment call, not a corpus-independent constant).
RARE_CHARACTER_MAX_COUNT = 10

#: A rare grapheme is *claimable* (may back a recall/precision claim) only with
#: document support ``>=`` this many reference lines; below it the grapheme is
#: still inventoried but flagged non-claimable (single-line anecdote / artifact).
RARE_CHARACTER_MIN_SUPPORT = 2

#: Reviewed rare-character injection set (study): the *claimable* NFD grapheme
#: clusters (rare by :data:`RARE_CHARACTER_MAX_COUNT`, support >=
#: :data:`RARE_CHARACTER_MIN_SUPPORT`, non-artifact) derived from the study MS
#: standard_test reference corpus. Unicode names, linguistic categories, and the
#: per-grapheme inclusion/exclusion rationale live in the machine-readable
#: ``rare_character_inventory.json`` of the evaluation-policy bundle.
#: NOTE: corpus support tops out at 6 lines, so this set is provided for
#: descriptive error analysis only — a headline rare-character recall claim is
#: explicitly scoped out (insufficient corpus support), not asserted.
REVIEWED_RARE_CHARACTERS: tuple[str, ...] = (
    "ì",  # ì   LATIN SMALL LETTER I + COMBINING GRAVE ACCENT
    "ī̀",  # ī̀   + COMBINING MACRON (long vowel) + grave
    "r̥̀",  # r̥̀   vocalic r (COMBINING RING BELOW) + grave
    "r̥̄",  # r̥̄   long vocalic r (ring below + macron)
    "ù",  # ù   LATIN SMALL LETTER U + COMBINING GRAVE ACCENT
)


# --------------------------------------------------------------------------- #
# Loading (schema-aware; legacy-compatible).
# --------------------------------------------------------------------------- #
def _blank_to_none(value: Optional[str]) -> Optional[str]:
    """Blank / whitespace-only → ``None``; otherwise the value unchanged."""
    if value is None:
        return None
    return value if value.strip() else None


def _parse_int(value: Optional[str], field_name: str) -> Optional[int]:
    v = _blank_to_none(value)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError as exc:
        raise PredictionSchemaError(
            f"{field_name!r} must be an integer, got {value!r}"
        ) from exc


_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "n", "f", ""})


def _parse_bool(value: Optional[str], field_name: str) -> bool:
    if value is None:
        return False
    token = value.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise PredictionSchemaError(f"{field_name!r} must be a boolean, got {value!r}")


def _parse_schema_version(value: Optional[str]) -> int:
    """Parse the ``schema_version`` cell of a versioned file, or fail fast.

    A header carrying ``schema_version`` marks a versioned prediction artifact,
    so every row must stamp a *supported* version. A blank cell or a version
    this evaluator does not support raises :class:`PredictionSchemaError` rather
    than silently evaluating under whatever reader semantics happen to exist —
    the saved prediction file is the study artifact contract, so an unreadable or
    unknown version must never quietly produce metrics.
    """
    parsed = _parse_int(value, "schema_version")
    if parsed is None:
        raise PredictionSchemaError(
            "versioned prediction file has a blank 'schema_version'; expected "
            f"{PREDICTION_SCHEMA_VERSION}"
        )
    if parsed != PREDICTION_SCHEMA_VERSION:
        raise PredictionSchemaError(
            f"unsupported prediction schema_version {parsed}; this evaluator "
            f"supports {PREDICTION_SCHEMA_VERSION}"
        )
    return parsed


def _row_from_record(record: Mapping[str, Any], *, legacy: bool) -> PredictionRow:
    reference = record.get("reference")
    prediction = record.get("prediction")
    return PredictionRow(
        reference="" if reference is None else str(reference),
        prediction="" if prediction is None else str(prediction),
        sample_id=_blank_to_none(record.get("sample_id")),
        font=_blank_to_none(record.get("font")),
        condition=_blank_to_none(record.get("condition")),
        fraction=_parse_int(record.get("fraction"), "fraction"),
        seed=_parse_int(record.get("seed"), "seed"),
        view=_blank_to_none(record.get("view")),
        split=_blank_to_none(record.get("split")),
        decode_failed=_parse_bool(record.get("decode_failed"), "decode_failed"),
        error_type=_blank_to_none(record.get("error_type")),
        error_message=_blank_to_none(record.get("error_message")),
        schema_version=(
            None if legacy else _parse_schema_version(record.get("schema_version"))
        ),
        is_legacy=legacy,
    )


def load_prediction_rows(path: str | Path) -> list[PredictionRow]:
    """Load prediction rows from a v1 *or* legacy three-column CSV.

    A header without a ``schema_version`` column is treated as a legacy study
    file (``sample_id,reference,prediction``); its metadata loads as ``None`` /
    ``False`` and every row is marked ``is_legacy=True``. A versioned header must
    stamp every row with a supported ``schema_version`` (see
    :func:`_parse_schema_version`). Either header must carry ``reference`` and
    ``prediction`` columns. Raises :class:`PredictionSchemaError` on an empty
    file, a header with no data rows, a missing required column, an unsupported
    or blank schema version, or an unparseable typed cell.
    """
    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise PredictionSchemaError(
            f"could not read prediction file {path}: {exc}"
        ) from exc
    with handle as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise PredictionSchemaError(f"{path} is empty (no header row)")
        for required in ("reference", "prediction"):
            if required not in fieldnames:
                raise PredictionSchemaError(
                    f"{path} is missing the required {required!r} column; "
                    f"header was {fieldnames}"
                )
        legacy = "schema_version" not in fieldnames
        rows = [_row_from_record(record, legacy=legacy) for record in reader]
    if not rows:
        raise PredictionSchemaError(
            f"{path} has a header but no prediction rows; a prediction file must "
            f"contain at least one evaluated sample"
        )
    return rows


# --------------------------------------------------------------------------- #
# Metric primitives.
# --------------------------------------------------------------------------- #
def _reliability_block(
    rows: Sequence[PredictionRow],
    norm_refs: Sequence[str],
    norm_preds: Sequence[str],
    policy: OverGenerationPolicy,
) -> dict[str, Any]:
    """Empty-output / decode-failure / over-generation rates for a row subset."""
    n = len(rows)
    n_empty = sum(1 for p in norm_preds if not p)
    n_failed = sum(1 for r in rows if r.decode_failed)
    denom = sum(1 for ref in norm_refs if ref)
    n_over = sum(
        1
        for ref, pred in zip(norm_refs, norm_preds)
        if ref and policy.is_over_generation(len(ref), len(pred))
    )
    return {
        "empty_output_rate": n_empty / n if n else 0.0,
        "n_empty_output": n_empty,
        "decode_failure_rate": n_failed / n if n else 0.0,
        "n_decode_failed": n_failed,
        "over_generation_rate": n_over / denom if denom else 0.0,
        "n_over_generation": n_over,
        "over_generation_denominator": denom,
        "over_generation_policy": policy.as_dict(),
    }


def _multiset_presence(
    ref_tokens: Sequence[Sequence[str]],
    hyp_tokens: Sequence[Sequence[str]],
    targets: frozenset[str],
) -> dict[str, Any]:
    """Position-agnostic presence recall/precision of ``targets`` tokens.

    Per row, the matched count is the multiset intersection of target tokens in
    the reference and hypothesis; recall is over reference occurrences,
    precision over hypothesis occurrences. Used for both diacritic combining
    marks (code-point tokens) and rare graphemes (grapheme-cluster tokens).
    """
    ref_count = hyp_count = matched = 0
    for rt, ht in zip(ref_tokens, hyp_tokens):
        rc = Counter(t for t in rt if t in targets)
        hc = Counter(t for t in ht if t in targets)
        ref_count += sum(rc.values())
        hyp_count += sum(hc.values())
        matched += sum((rc & hc).values())
    precision = matched / hyp_count if hyp_count else 0.0
    recall = matched / ref_count if ref_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "ref_count": ref_count,
        "hyp_count": hyp_count,
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_rate": (1.0 - recall) if ref_count else 0.0,
    }


def _diacritic_block(
    norm_refs: Sequence[str],
    norm_preds: Sequence[str],
    classes: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    ref_cp = [list(unicodedata.normalize("NFD", r)) for r in norm_refs]
    hyp_cp = [list(unicodedata.normalize("NFD", p)) for p in norm_preds]
    return {
        name: _multiset_presence(ref_cp, hyp_cp, frozenset(marks))
        for name, marks in classes.items()
    }


def _rare_character_block(
    norm_refs: Sequence[str],
    norm_preds: Sequence[str],
    rare_characters: Iterable[str],
) -> dict[str, Any]:
    from .accent_metrics import iter_grapheme_clusters

    graphemes = frozenset(unicodedata.normalize("NFD", c) for c in rare_characters)
    ref_g = [list(iter_grapheme_clusters(r)) for r in norm_refs]
    hyp_g = [list(iter_grapheme_clusters(p)) for p in norm_preds]
    block = _multiset_presence(ref_g, hyp_g, graphemes)
    block["charset"] = sorted(graphemes)
    return block


def _edit_event_block(
    norm_refs: Sequence[str],
    norm_preds: Sequence[str],
    top_substitutions: int,
) -> dict[str, Any]:
    """Aggregate character-level edit-op counts + top substitution pairs.

    Aligns NFD code points with ``jiwer.process_characters`` (empty-string pairs
    are charged as all-insert / all-delete without calling jiwer, which rejects
    empty inputs). Deterministic tie-break on the substitution ranking.
    """
    import jiwer

    substitutions = deletions = insertions = hits = n_ref_chars = 0
    sub_counter: Counter[tuple[str, str]] = Counter()
    pairs: list[tuple[str, str]] = []
    for ref, pred in zip(norm_refs, norm_preds):
        ref_nfd = unicodedata.normalize("NFD", ref)
        hyp_nfd = unicodedata.normalize("NFD", pred)
        n_ref_chars += len(ref_nfd)
        if not ref_nfd and not hyp_nfd:
            continue
        if not ref_nfd:
            insertions += len(hyp_nfd)
            continue
        if not hyp_nfd:
            deletions += len(ref_nfd)
            continue
        pairs.append((ref_nfd, hyp_nfd))

    if pairs:
        out = jiwer.process_characters([r for r, _ in pairs], [h for _, h in pairs])
        substitutions += out.substitutions
        deletions += out.deletions
        insertions += out.insertions
        hits += out.hits
        for (ref_nfd, hyp_nfd), chunks in zip(pairs, out.alignments):
            for chunk in chunks:
                if chunk.type != "substitute":
                    continue
                for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    ref_char = ref_nfd[chunk.ref_start_idx + k]
                    hyp_char = hyp_nfd[chunk.hyp_start_idx + k]
                    sub_counter[(ref_char, hyp_char)] += 1

    ranked = sorted(sub_counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return {
        "n_ref_chars": n_ref_chars,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "hits": hits,
        "sub_per_ref": substitutions / n_ref_chars if n_ref_chars else 0.0,
        "del_per_ref": deletions / n_ref_chars if n_ref_chars else 0.0,
        "ins_per_ref": insertions / n_ref_chars if n_ref_chars else 0.0,
        "top_substitutions": [
            {"ref": ref_char, "hyp": hyp_char, "count": count}
            for (ref_char, hyp_char), count in ranked[:top_substitutions]
        ],
    }


def _core_block(
    ocr_metrics: Any,
    rows: Sequence[PredictionRow],
    norm_refs: Sequence[str],
    norm_preds: Sequence[str],
    *,
    policy: OverGenerationPolicy,
    classes: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """The generic-OCR + accent + reliability + diacritic block for a row set.

    Shared verbatim by the overall corpus and every per-font subset so a font's
    metrics are computed identically to the whole.
    """
    from .accent_metrics import compute_all_accent_aware_metrics

    block: dict[str, Any] = {
        "num_samples": len(rows),
        "num_scored": sum(1 for ref in norm_refs if ref),
    }
    ocr = ocr_metrics.compute(list(norm_preds), list(norm_refs))
    block["cer"] = ocr.get("cer")
    block["wer"] = ocr.get("wer")
    block["chrf"] = ocr.get("chrf")
    block["exact_match"] = ocr.get("exact_match")
    # Exact-match *count* alongside the rate : the number of scored
    # samples whose normalized prediction equals its normalized reference. Scored
    # over the same basis as ``OCRMetrics.exact_match`` and ``num_scored`` —
    # non-empty references only — so ``exact_match_count / num_scored`` reproduces
    # the ``exact_match`` rate exactly.
    block["exact_match_count"] = sum(
        1 for ref, pred in zip(norm_refs, norm_preds) if ref and ref == pred
    )
    block.update(compute_all_accent_aware_metrics(norm_preds, norm_refs))
    block["reliability"] = _reliability_block(rows, norm_refs, norm_preds, policy)
    block["diacritics"] = _diacritic_block(norm_refs, norm_preds, classes)
    return block


def _constant_axis(rows: Sequence[PredictionRow], name: str) -> Any:
    """The single non-``None`` value of an axis across ``rows``, else ``None``."""
    values = {getattr(row, name) for row in rows}
    values.discard(None)
    return next(iter(values)) if len(values) == 1 else None


def evaluate_prediction_rows(
    rows: Iterable[PredictionRow],
    *,
    normalization: Normalization = DEFAULT_NORMALIZATION,
    rare_characters: Optional[Iterable[str]] = None,
    over_generation_policy: OverGenerationPolicy = DEFAULT_OVER_GENERATION_POLICY,
    diacritic_classes: Mapping[str, Sequence[str]] = DEFAULT_DIACRITIC_CLASSES,
    use_chrf: bool = True,
    top_substitutions: int = DEFAULT_TOP_SUBSTITUTIONS,
) -> dict[str, Any]:
    """Compute the full study metric payload from prediction rows, deterministically.

    Pure function of ``rows`` and the (frozen) policies — no model inference,
    no I/O — so re-evaluating the same file yields byte-identical JSON. The
    result carries the run axes (``condition`` / ``fraction`` / ``seed`` /
    ``view`` / ``split``, each the constant value across rows or ``None``),
    generic OCR + flat accent metrics, grouped reliability / diacritic /
    edit-event / rare-character blocks, and a per-font block, so it aggregates
    directly into the planned main and font-held-out tables.
    """
    from scripts.training.trocr.metrics import OCRMetrics

    rows = list(rows)
    norm = normalization
    norm_refs = [norm.apply(row.reference) for row in rows]
    norm_preds = [norm.apply(row.prediction) for row in rows]

    ocr_metrics = OCRMetrics(use_chrf=use_chrf)

    result: dict[str, Any] = {
        "prediction_schema_version": _constant_axis(rows, "schema_version"),
        "is_legacy": any(row.is_legacy for row in rows),
        "normalization": norm.as_dict(),
    }
    for axis in AXIS_FIELDS:
        result[axis] = _constant_axis(rows, axis)

    result.update(
        _core_block(
            ocr_metrics,
            rows,
            norm_refs,
            norm_preds,
            policy=over_generation_policy,
            classes=diacritic_classes,
        )
    )
    result["edit_events"] = _edit_event_block(norm_refs, norm_preds, top_substitutions)
    result["rare_characters"] = (
        None
        if rare_characters is None
        else _rare_character_block(norm_refs, norm_preds, rare_characters)
    )

    by_font: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if row.font:
            by_font.setdefault(row.font, []).append(i)
    result["per_font"] = {
        font: _core_block(
            ocr_metrics,
            [rows[i] for i in indices],
            [norm_refs[i] for i in indices],
            [norm_preds[i] for i in indices],
            policy=over_generation_policy,
            classes=diacritic_classes,
        )
        for font, indices in sorted(by_font.items())
    }
    return result


def evaluate_prediction_file(
    path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a prediction file and evaluate it (convenience over the two calls)."""
    return evaluate_prediction_rows(load_prediction_rows(path), **kwargs)


def iter_edit_events(
    rows: Iterable[PredictionRow],
    *,
    normalization: Normalization = DEFAULT_NORMALIZATION,
) -> Iterable[dict[str, Any]]:
    """Yield per-sample character-level edit events for error analysis.

    Each event is a substitution / deletion / insertion charged against the NFD
    code-point alignment of the normalized reference and prediction (``equal``
    runs are omitted). ``ref`` / ``hyp`` carry the code point on each side
    (``None`` for the absent side); ``ref_index`` / ``hyp_index`` locate it. The
    ``sample_id`` falls back to the row's ordinal when the file has none.
    Deterministic and inference-free, like :func:`evaluate_prediction_rows`.
    """
    import jiwer

    for ordinal, row in enumerate(rows):
        ref_nfd = unicodedata.normalize("NFD", normalization.apply(row.reference))
        hyp_nfd = unicodedata.normalize("NFD", normalization.apply(row.prediction))
        sample_id = row.sample_id if row.sample_id is not None else str(ordinal)
        if not ref_nfd and not hyp_nfd:
            continue
        if not ref_nfd:
            ops = [("insert", 0, 0, 0, len(hyp_nfd))]
        elif not hyp_nfd:
            ops = [("delete", 0, len(ref_nfd), 0, 0)]
        else:
            out = jiwer.process_characters([ref_nfd], [hyp_nfd])
            ops = [
                (c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx)
                for c in out.alignments[0]
            ]
        for op_type, r0, r1, h0, h1 in ops:
            if op_type == "equal":
                continue
            if op_type == "substitute":
                for k in range(r1 - r0):
                    yield {
                        "sample_id": sample_id,
                        "op": "substitute",
                        "ref": ref_nfd[r0 + k],
                        "hyp": hyp_nfd[h0 + k],
                        "ref_index": r0 + k,
                        "hyp_index": h0 + k,
                    }
            elif op_type == "delete":
                for i in range(r0, r1):
                    yield {
                        "sample_id": sample_id,
                        "op": "delete",
                        "ref": ref_nfd[i],
                        "hyp": None,
                        "ref_index": i,
                        "hyp_index": None,
                    }
            elif op_type == "insert":
                for j in range(h0, h1):
                    yield {
                        "sample_id": sample_id,
                        "op": "insert",
                        "ref": None,
                        "hyp": hyp_nfd[j],
                        "ref_index": None,
                        "hyp_index": j,
                    }
