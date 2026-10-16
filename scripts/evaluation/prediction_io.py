"""Prediction io: PACLIC 2026 v2 release."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Optional
from .prediction_metrics import PREDICTION_COLUMNS, PREDICTION_SCHEMA_VERSION

"""Shared helpers retained for the PACLIC v2 release."""


class PredictionFormatError(Exception):
    """Raised when an input manifest / config violates the study contract."""


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fmt_cell(value: Any) -> str:
    """Blank for ``None``; lowercase for bools (parseable by the evaluator)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_predictions_csv(
    path: str | Path,
    *,
    references: Sequence[str],
    predictions: Sequence[str],
    sample_ids: Sequence[str] | None = None,
    fonts: Sequence[str | None] | None = None,
    condition: str | None = None,
    fraction: int | None = None,
    seed: int | None = None,
    view: str | None = None,
    split: str | None = None,
    decode_failed: Sequence[bool] | None = None,
    error_types: Sequence[str | None] | None = None,
    error_messages: Sequence[str | None] | None = None,
    schema_version: int = PREDICTION_SCHEMA_VERSION,
) -> Path:
    """Write a versioned prediction CSV with the :data:`PREDICTION_COLUMNS` header.

    Per-row columns (``sample_ids`` / ``fonts`` / ``decode_failed`` /
    ``error_types`` / ``error_messages``) are sequences aligned to
    ``references``; the run axes (``condition`` / ``fraction`` / ``seed`` /
    ``view`` / ``split``) and ``schema_version`` are per-file scalars written on
    every row. Any omitted value is left blank so the header stays constant —
    real runs supply the metadata, smoke/back-compat callers may leave it blank.
    All provided sequences must match ``len(references)``.
    """
    path = Path(path)
    _ensure_parent(path)
    n = len(references)

    def _check(name: str, seq: Sequence[Any] | None) -> None:
        if seq is not None and len(seq) != n:
            raise PredictionFormatError(
                f"predictions/references/{name} length mismatch: "
                f"refs={n} {name}={len(seq)}"
            )

    _check("predictions", predictions)
    _check("sample_ids", sample_ids)
    _check("fonts", fonts)
    _check("decode_failed", decode_failed)
    _check("error_types", error_types)
    _check("error_messages", error_messages)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PREDICTION_COLUMNS)
        for i in range(n):
            writer.writerow(
                [
                    schema_version,
                    _fmt_cell(None if sample_ids is None else sample_ids[i]),
                    references[i],
                    predictions[i],
                    _fmt_cell(None if fonts is None else fonts[i]),
                    _fmt_cell(condition),
                    _fmt_cell(fraction),
                    _fmt_cell(seed),
                    _fmt_cell(view),
                    _fmt_cell(split),
                    _fmt_cell(
                        None if decode_failed is None else bool(decode_failed[i])
                    ),
                    _fmt_cell(None if error_types is None else error_types[i]),
                    _fmt_cell(None if error_messages is None else error_messages[i]),
                ]
            )
    return path


def write_metrics_json(path: str | Path, metrics: Mapping[str, Any]) -> Path:
    """Write a metrics JSON file (UTF-8, indent 2, non-ASCII preserved)."""
    path = Path(path)
    _ensure_parent(path)
    path.write_text(
        json.dumps(dict(metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


