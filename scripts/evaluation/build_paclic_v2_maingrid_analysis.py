"""Build paclic v2 maingrid analysis: PACLIC 2026 v2 release."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from . import aggregate_accent_ratio_multiseed as agg

# --------------------------------------------------------------------------- #
# Study identity + canonical grid (the axes are agg's; only the naming differs).
# --------------------------------------------------------------------------- #
#: Study slug, used in log prefixes and the emitted artifacts.
STUDY: str = "paclic-v2-maingrid"
#: The v2 launcher that produced the cells (the record of the recipe/corpus).
LAUNCHER: str = "scripts/reproduce.py maingrid"
#: ``source`` stamped on every cell — one root, one provenance (contrast with the
#: v1 ``prior-study-reference`` / ``multiseed`` split).
CELL_SOURCE: str = "v2-maingrid"

#: Mix ratios and seeds. Same axes as v1 (:data:`agg.RV_RATIOS` / :data:`agg.SEEDS`)
#: so the reused per-ratio / contrast helpers apply unchanged.
RV_RATIOS: tuple[float, ...] = agg.RV_RATIOS
SEEDS: tuple[int, ...] = agg.SEEDS
#: The target grid size (5 ratios × 3 seeds). ``complete`` is measured against
#: THIS, never against ``len(cells)``: a caller that loads a subset of the axes
#: would otherwise get ``complete=True`` for a 10-cell "grid".
TARGET_N_CELLS: int = len(RV_RATIOS) * len(SEEDS)

#: The v2 corpus the launcher points at by default. Every attested cell's echoed
#: ``rv=``/``ms=`` dirs must be these — ``RV_DATA_DIR``/``MS_DATA_DIR`` are
#: documented launcher overrides, so the ``corpus=`` token is a *label*, and only
#: the dirs are evidence.
EXPECTED_RV_DATA_DIR: str = "data/datasets/rv_v2"
EXPECTED_MS_DATA_DIR: str = "data/datasets/ms_v2"
#: The label the launcher derives from those dirs.
EXPECTED_CORPUS: str = "v2"
#: The only data source with a v2 build. ``DATA_SOURCE=hf`` reads a Hugging Face
#: artifact (whose immutable pin must be checked) while the launcher still echoes the *unused* v2 CSV
#: dirs, so an hf cell must never pass as attested-v2 on the strength of those.
EXPECTED_DATA_SOURCE: str = "csv"

DEFAULT_STUDY_ROOT = Path("outputs/experiments_paclic_v2_maingrid")
DEFAULT_SHARED_STUDY_ROOT = (
    "./outputs/experiments_paclic_v2_maingrid"
)
DEFAULT_SLURM_LOG_DIR = Path("outputs/logs")
DEFAULT_OUT_DIR = Path(
    "results/paclic-v2-maingrid"
)
DEFAULT_FIGURES_OUT_DIR = Path("outputs/figures/maingrid")
#: Slurm ``.out`` basenames the launcher writes (``…_<jobid>_<task>.out``).
SLURM_LOG_GLOB: str = "paclic_v2_maingrid_*.out"


class FigureGateError(RuntimeError):
    """Raised when a publication figure is requested from an unpublishable grid.

    Coverage is only one of the gates: an attestation that does not establish the v2
    corpus, or any publication-blocking warning, refuses the figures just as a
    partial grid does. The figures are the artifact most likely to be lifted out of
    context, so they are the strictest consumer of the bundle.
    """


class IncompleteGridError(FigureGateError):
    """Raised when a publication artifact is requested from a partial grid."""


class UnverifiedCorpusError(FigureGateError):
    """Raised when the grid's corpus attestation does not establish v2."""


class GridInvariantError(FigureGateError):
    """Raised when a publication-blocking integrity warning is present."""


# --------------------------------------------------------------------------- #
# Cell naming + loading (single root, uniform seed suffix).
# --------------------------------------------------------------------------- #
def run_name(rv_ratio: float, seed: int) -> str:
    """Run-dir name for a v2 cell — ``rv_<pct>_ms_<pct>_seed_<n>`` for **every** seed.

    Mirrors ``experiment_accent_ratio.build_experiment_name`` with
    ``run_suffix=seed_<n>``, which the launcher passes for all three seeds. This is
    exactly where the v1 contract diverges: ``agg.expected_run_name(rv, 42)`` drops
    the suffix (v1's seed 42 was inherited from the factorial dirs), so the v1
    loader cannot find a v2 seed-42 cell.
    """
    rv_pct = int(round(rv_ratio * 100))
    ms_pct = int(round((1.0 - rv_ratio) * 100))
    return f"rv_{rv_pct}_ms_{ms_pct}_seed_{seed}"


def _results_json_parses(cell_dir: Path) -> bool:
    """Is this cell's ``results.json`` absent (→ ``True``) or readable JSON?

    ``False`` means "on disk but unreadable". Absent is ``True`` so the shared cell
    reader keeps ownership of the normal absent-cell path; only the torn-write case
    is short-circuited here.
    """
    path = cell_dir / "results.json"
    if not path.is_file():
        return True
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def load_grid(
    study_root: Path,
    *,
    metrics_fn: Optional[agg.MetricsFn] = None,
    cross_check: bool = True,
    shared_study_root: Optional[str] = None,
    seeds: Sequence[int] = SEEDS,
    rv_ratios: Sequence[float] = RV_RATIOS,
) -> tuple[list[agg.SeedCell], list[dict[str, Any]]]:
    """Load every ``rv_ratio × seed`` cell from the one study root (ratio-major).

    Ratio-major order matches the launcher's array decomposition, so cell *i* here
    is array task *i*. Absent cells come back ``present=False`` rather than
    raising: the grid is expected to be read mid-flight.

    No ``slurm_log_dir`` is passed to the cell reader on purpose. The study
    training-state recovery it enables is keyed by the **v1** run registry, and v2
    needs none of it: the launcher's ``results.json`` carries ``training_state``
    inline for every cell.

    A ``results.json`` that exists but does not parse yields a ``present=False``
    cell instead of killing the whole aggregation: the shared cell reader's
    ``json.loads`` is unguarded, and this grid is read *while* the launcher writes
    those files, so a torn/partial write is a normal mid-flight state. It is not
    silent — :func:`grid_integrity` re-detects it (absent cell whose
    ``results.json`` is on disk) and it is a hard, publication-blocking warning.
    """
    metrics = agg._MetricsCache(metrics_fn or agg.default_metrics_fn)
    cells: list[agg.SeedCell] = []
    checks: list[dict[str, Any]] = []
    for rv_ratio in rv_ratios:
        for seed in seeds:
            name = run_name(rv_ratio, seed)
            shared_path = (
                f"{shared_study_root.rstrip('/')}/{name}" if shared_study_root else None
            )
            if not _results_json_parses(study_root / name):
                cells.append(
                    agg.SeedCell(
                        rv_ratio=rv_ratio,
                        seed=seed,
                        run_name=name,
                        source=CELL_SOURCE,
                        present=False,
                        shared_path=shared_path,
                    )
                )
                continue
            cell, cell_checks = agg.read_cell_dir(
                study_root / name,
                rv_ratio=rv_ratio,
                seed=seed,
                run_name=name,
                source=CELL_SOURCE,
                shared_path=shared_path,
                metrics=metrics,
                cross_check=cross_check,
            )
            cells.append(cell)
            checks.extend(cell_checks)
    return cells, checks


# --------------------------------------------------------------------------- #
# Coverage / partial-grid reporting.
# --------------------------------------------------------------------------- #
def coverage_summary(cells: Sequence[agg.SeedCell]) -> dict[str, Any]:
    """Cell coverage of the target grid, per ratio and per seed.

    ``complete`` is the only flag any artifact may use to claim a finished grid;
    ``per_ratio[…].n_present`` is the n behind that ratio's mean/CI.

    Completeness is measured against the **target** grid (:data:`TARGET_N_CELLS`
    cells, :data:`SEEDS` per ratio, :data:`RV_RATIOS` per seed), not against
    ``len(cells)``: a caller that loaded only two of the three seeds must not be
    told its 10-cell grid is complete.
    """
    per_ratio: dict[str, Any] = {}
    for rv_ratio in RV_RATIOS:
        group = [c for c in cells if c.rv_ratio == rv_ratio]
        present = [c for c in group if c.present]
        per_ratio[f"{rv_ratio:g}"] = {
            "rv_ratio": rv_ratio,
            "ms_ratio": round(1.0 - rv_ratio, 2),
            "n_cells": len(SEEDS),
            "n_cells_loaded": len(group),
            "n_present": len(present),
            "seeds_present": sorted(c.seed for c in present),
            "seeds_missing": sorted(set(SEEDS) - {c.seed for c in present}),
            "complete": len(present) == len(SEEDS),
        }
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        group = [c for c in cells if c.seed == seed]
        present = [c for c in group if c.present]
        per_seed[str(seed)] = {
            "n_cells": len(RV_RATIOS),
            "n_cells_loaded": len(group),
            "n_present": len(present),
            "complete": len(present) == len(RV_RATIOS),
        }
    n_present = sum(1 for c in cells if c.present)
    return {
        "n_cells": TARGET_N_CELLS,
        "n_cells_loaded": len(cells),
        "n_present": n_present,
        "complete": n_present == TARGET_N_CELLS,
        "seeds_present": sorted({c.seed for c in cells if c.present}),
        "target_seeds": list(SEEDS),
        "target_rv_ratios": list(RV_RATIOS),
        "per_ratio": per_ratio,
        "per_seed": per_seed,
        # Enumerated from the TARGET grid, so a cell that was never even loaded
        # (subset axes) still counts as missing rather than vanishing.
        "missing_cells": [
            name
            for name in (
                run_name(rv_ratio, seed) for rv_ratio in RV_RATIOS for seed in SEEDS
            )
            if name not in {c.run_name for c in cells if c.present}
        ],
    }


def status_label(coverage: dict[str, Any]) -> str:
    """``"COMPLETE"`` or ``"PARTIAL"`` — the word every artifact leads with."""
    return "COMPLETE" if coverage["complete"] else "PARTIAL"


def per_ratio_n_phrase(coverage: dict[str, Any]) -> str:
    """``100:0 n=3/3; 75:25 n=1/3; …`` — the per-ratio n, never a single grid n."""
    return "; ".join(
        f"{int(round(blk['rv_ratio'] * 100))}:{int(round(blk['ms_ratio'] * 100))} "
        f"n={blk['n_present']}/{blk['n_cells']}"
        for blk in coverage["per_ratio"].values()
    )


def partial_banner(coverage: dict[str, Any]) -> Optional[str]:
    """The PARTIAL banner, or ``None`` when the grid really is complete.

    Follows the convention already used elsewhere in ``scripts/evaluation``
    (``build_paclic_accent_ratio_data_efficiency_analysis``,
    ``issue161_retrain_deltas``): the words "PARTIAL", "NOT A COMPLETE RESULT",
    the present/expected counts, and the explicit list of what is still missing.
    """
    if coverage["complete"]:
        return None
    missing = coverage["missing_cells"]
    shown = ", ".join(missing[:15]) + (" …" if len(missing) > 15 else "")
    return (
        f"PARTIAL — NOT A COMPLETE RESULT: {coverage['n_present']}/"
        f"{coverage['n_cells']} cells present (per ratio: "
        f"{per_ratio_n_phrase(coverage)}). The v2 main grid is still running, so no "
        "value below may be presented as the v2 main-grid result, and no per-ratio "
        f"mean/CI is a 3-seed number unless its own n says 3. Still missing: {shown}"
    )


# --------------------------------------------------------------------------- #
# Subset / recipe provenance recorded inside each run dir.
# --------------------------------------------------------------------------- #
#: ``results.json`` keys describing *which training subset* a cell consumed. These
#: pin the shared-parallel text selection — NOT the image corpus (see the module
#: docstring: v1 and v2 share the ISO text byte-for-byte).
SUBSET_FIELDS: tuple[str, ...] = (
    "mix_mode",
    "key_column",
    "selection_algorithm",
    "parallel_keys_total",
    "parallel_keys_used",
    "parallel_subset_seed",
    "key_list_sha256",
    "total_train_samples_requested",
    "rv_train_samples",
    "ms_train_samples",
    "num_train_epochs",
    "model_base",
    "freeze_mode",
    "input_scheme",
)

#: Note attached to every subset-consistency payload so the hash is never read as
#: corpus evidence.
KEY_LIST_SHA_NOTE: str = (
    "key_list_sha256 hashes the ordered shared-parallel ISO *text* keys. The v2 "
    "corpus re-renders the SAME text, so this hash is identical between v1 and v2 "
    "by construction and CANNOT attest which image corpus a cell consumed. Use "
    "corpus_attestation (Slurm .out header) for that."
)

#: Why the hash is checked per seed instead of grid-wide. Attached to the payload
#: and quoted in the results page, because "3 distinct hashes" looks like a defect
#: until you know the selection is seeded.
KEY_LIST_SHA_SEED_NOTE: str = (
    "key_list_sha256 hashes the ORDERED selected key list, and the selection is "
    "seeded: scripts/training/trocr/dataset.py sorts the shared keys and then "
    "applies a seeded shuffle (the launcher leaves parallel_subset_seed=null, so it "
    "follows `seed`). At full N the selected key SET is identical across seeds while "
    "its ORDER — hence the hash — is not. The hash is therefore a WITHIN-SEED "
    "invariant, constant across the five ratios of one seed and EXPECTED to differ "
    "across seeds: a healthy 15-cell grid has exactly 3 distinct values, one per "
    "seed. It is checked that way (key_list_sha256_per_seed) and is NOT a grid-wide "
    "held-fixed field; a cross-seed recipe split is still caught by the grid-wide "
    "fields, which include model_base and total_train_samples_requested."
)


def subset_provenance(
    study_root: Path, cells: Sequence[agg.SeedCell]
) -> dict[str, dict[str, Any]]:
    """``run_name → {SUBSET_FIELDS}`` re-read from each present cell's results.json.

    The shared cell reader keeps only the metrics it aggregates, so the subset /
    recipe fields are read here rather than widening ``agg.SeedCell``.
    """
    out: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not cell.present:
            continue
        path = study_root / cell.run_name / "results.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out[cell.run_name] = {field: data.get(field) for field in SUBSET_FIELDS}
    return out


def expected_mix_counts(total: int, rv_ratio: float) -> tuple[int, int]:
    """``(rv_train_samples, ms_train_samples)`` under the launcher's floor rule."""
    n_rv = int(total * rv_ratio)
    return n_rv, total - n_rv


def subset_consistency(
    prov: dict[str, dict[str, Any]], cells: Sequence[agg.SeedCell]
) -> dict[str, Any]:
    """Do the present cells share one recipe, and did each train the mix it claims?

    Three independent checks:

    * **held-fixed fields** — every present cell must agree, *grid-wide*, on the
      corpus-independent recipe pins (``parallel_keys_total``,
      ``total_train_samples_requested``, ``mix_mode``, ``model_base``, …). More than
      one distinct value means the grid is not a single study.
    * **per-seed subset hash** — ``key_list_sha256`` is deliberately NOT a grid-wide
      pin (see :data:`KEY_LIST_SHA_SEED_NOTE`): the key list is shuffled with the
      run's seed, so the invariant is that it is constant *within* a seed across the
      five ratios, giving three distinct values grid-wide. Checking it grid-wide
      makes a healthy grid report "this is not a single study".
    * **mix realization** — ``rv_train_samples`` / ``ms_train_samples`` must equal
      the floor split of the requested total at that cell's ``rv_ratio``, so a cell
      cannot be silently mislabelled by ratio.
    """
    fixed_fields = (
        "mix_mode",
        "key_column",
        "selection_algorithm",
        "parallel_keys_total",
        "total_train_samples_requested",
        "num_train_epochs",
        "model_base",
        "freeze_mode",
        "input_scheme",
    )
    distinct: dict[str, list[Any]] = {}
    for field in fixed_fields:
        values = sorted(
            {
                json.dumps(rec.get(field), sort_keys=True)
                for rec in prov.values()
                if rec.get(field) is not None
            }
        )
        distinct[field] = [json.loads(v) for v in values]
    inconsistent = sorted(f for f, vals in distinct.items() if len(vals) > 1)

    ratio_by_run = {c.run_name: c.rv_ratio for c in cells}
    seed_by_run = {c.run_name: c.seed for c in cells}
    per_seed: dict[str, dict[str, Any]] = {}
    for name, rec in sorted(prov.items()):
        sha = rec.get("key_list_sha256")
        seed = seed_by_run.get(name)
        if sha is None or seed is None:
            continue
        blk = per_seed.setdefault(
            str(seed), {"seed": seed, "runs": [], "key_list_sha256": []}
        )
        blk["runs"].append(name)
        if sha not in blk["key_list_sha256"]:
            blk["key_list_sha256"].append(sha)
    split_seeds = sorted(
        (blk["seed"] for blk in per_seed.values() if len(blk["key_list_sha256"]) > 1),
    )

    mix_rows: list[dict[str, Any]] = []
    for name, rec in sorted(prov.items()):
        total = rec.get("total_train_samples_requested")
        rv_ratio = ratio_by_run.get(name)
        if total is None or rv_ratio is None:
            continue
        exp_rv, exp_ms = expected_mix_counts(int(total), rv_ratio)
        ok = rec.get("rv_train_samples") == exp_rv and rec.get("ms_train_samples") == (
            exp_ms
        )
        mix_rows.append(
            {
                "run_name": name,
                "rv_ratio": rv_ratio,
                "total_train_samples_requested": total,
                "expected_rv_train_samples": exp_rv,
                "expected_ms_train_samples": exp_ms,
                "rv_train_samples": rec.get("rv_train_samples"),
                "ms_train_samples": rec.get("ms_train_samples"),
                "matches_ratio": bool(ok),
            }
        )
    return {
        "n_cells_checked": len(prov),
        "held_fixed": distinct,
        "held_fixed_scope": (
            "grid-wide across every present cell; key_list_sha256 is excluded on "
            "purpose and checked per seed instead (key_list_sha256_scope)"
        ),
        "inconsistent_fields": inconsistent,
        "grid_wide_consistent": not inconsistent,
        "key_list_sha256_per_seed": {k: per_seed[k] for k in sorted(per_seed)},
        "key_list_sha256_split_seeds": split_seeds,
        "key_list_sha256_distinct_grid_wide": sorted(
            {sha for blk in per_seed.values() for sha in blk["key_list_sha256"]}
        ),
        "key_list_sha256_scope": (
            "constant WITHIN each seed across the five ratios; expected to DIFFER "
            f"across seeds ({len(SEEDS)} distinct values for a healthy grid)"
        ),
        "key_list_sha256_consistent": not split_seeds,
        "consistent": not inconsistent and not split_seeds,
        "mix_realization": mix_rows,
        "mix_mismatches": [r["run_name"] for r in mix_rows if not r["matches_ratio"]],
        "key_list_sha256_note": KEY_LIST_SHA_NOTE,
        "key_list_sha256_seed_note": KEY_LIST_SHA_SEED_NOTE,
    }


# --------------------------------------------------------------------------- #
# Corpus attestation from the immutable Slurm .out header (best-effort).
# --------------------------------------------------------------------------- #
_RUN_NAME_RE = re.compile(r"\brun_name=(\S+)")
_CORPUS_LINE_RE = re.compile(r"^.*\bcorpus=(\S+).*$", re.MULTILINE)
_RV_DIR_RE = re.compile(r"\brv=(\S+)")
_MS_DIR_RE = re.compile(r"\bms=(\S+)")
_NODE_RE = re.compile(r"\bnode=(\S+)")
_OUTPUT_DIR_RE = re.compile(r"\boutput_dir=(\S+)")
_DATA_SOURCE_RE = re.compile(r"\bdata_source=(\S+)")
#: The launcher's CSV presence guard, echoed only on the ``DATA_SOURCE=csv`` path.
#: It is how ``data_source`` is recovered from logs written before the launcher
#: started echoing the token — including the logs of the currently running grid.
_CSV_GUARD_RE = re.compile(
    r"^.*\bcorpus CSVs \+ first images present:.*$", re.MULTILINE
)
#: ``…_<jobid>_<task>.out`` — parsed so logs sort NUMERICALLY. Lexicographic
#: basename order puts job 99 after job 100, which would pick a superseded log as
#: the winner.
_LOG_NAME_RE = re.compile(r"_(\d+)_(\d+)\.out$")


def parse_corpus_attestation(text: str) -> Optional[dict[str, Any]]:
    """Corpus + run identity echoed in a v2 launcher's Slurm ``.out`` header.

    The launcher prints ``run_name=…``, ``output_dir=…`` and a
    ``corpus=<id> rv=<dir> ms=<dir> data_source=<src>`` line before training. Those
    echoes are the only in-repo record of which image corpus a cell consumed, since
    ``results.json`` omits the Hydra data paths. ``output_dir`` is kept because the
    smoke study root reuses the SAME run names, so a run name alone cannot tell a
    smoke log from a real one. Returns ``None`` when the log carries no
    ``run_name=`` (e.g. a truncated or unrelated log) — absence is reported as
    *unattested*, never inferred.

    ``data_source`` is load-bearing, not decoration: under ``data.source=hf`` the
    launcher reads a Hub artifact identified by its revision while still echoing the unused
    v2 CSV dirs, so the dirs alone would attest a possibly-v1 grid as clean v2. Logs
    written before the launcher echoed the token fall back to the CSV presence guard
    (`corpus CSVs + first images present:`), which only the csv path prints; if
    neither is present ``data_source`` stays ``None`` and the cell is treated as
    unverified rather than assumed csv.
    """
    name = _RUN_NAME_RE.search(text)
    if name is None:
        return None
    rec: dict[str, Any] = {
        "run_name": name.group(1),
        "corpus": None,
        "rv_data_dir": None,
        "ms_data_dir": None,
        "data_source": None,
        "data_source_evidence": None,
        "node": None,
        "output_dir": None,
    }
    corpus_line = _CORPUS_LINE_RE.search(text)
    if corpus_line is not None:
        rec["corpus"] = corpus_line.group(1)
        line = corpus_line.group(0)
        rv = _RV_DIR_RE.search(line)
        ms = _MS_DIR_RE.search(line)
        rec["rv_data_dir"] = rv.group(1) if rv else None
        rec["ms_data_dir"] = ms.group(1) if ms else None
    data_source = _DATA_SOURCE_RE.search(text)
    if data_source is not None:
        rec["data_source"] = data_source.group(1)
        rec["data_source_evidence"] = "data_source= echo"
    elif _CSV_GUARD_RE.search(text) is not None:
        rec["data_source"] = "csv"
        rec["data_source_evidence"] = (
            "inferred from the CSV presence guard echo, which only the "
            "DATA_SOURCE=csv path prints"
        )
    node = _NODE_RE.search(text)
    rec["node"] = node.group(1) if node else None
    for field in ("dataset_id", "dataset_revision"):
        match = re.search(r"\b" + field + r"=(\S+)", text)
        if match is not None:
            rec[field] = match.group(1)
    output_dir = _OUTPUT_DIR_RE.search(text)
    rec["output_dir"] = output_dir.group(1) if output_dir else None
    return rec


def _log_sort_key(name: str) -> tuple[int, int, int, str]:
    """``(jobid, task)`` numeric order, with unparseable basenames last."""
    match = _LOG_NAME_RE.search(name)
    if match is None:
        return (1, 0, 0, name)
    return (0, int(match.group(1)), int(match.group(2)), name)


def attest_corpus_from_slurm_logs(
    log_dir: Optional[Path], *, glob: str = SLURM_LOG_GLOB
) -> dict[str, dict[str, Any]]:
    """``run_name → attestation`` over the study's Slurm logs (best-effort).

    Logs are read in **numeric** ``(jobid, task)`` order, so a re-queued cell's later
    log wins and the earlier ones are recorded as ``superseded_logs``. A missing log
    dir yields ``{}``.
    """
    out: dict[str, dict[str, Any]] = {}
    if log_dir is None or not log_dir.is_dir():
        return out
    for path in sorted(log_dir.glob(glob), key=lambda p: _log_sort_key(p.name)):
        try:
            rec = parse_corpus_attestation(path.read_text(errors="replace"))
        except OSError:
            continue
        if rec is None:
            continue
        rec["log"] = path.name
        previous = out.get(rec["run_name"])
        if previous is not None:
            rec["superseded_logs"] = [
                *previous.get("superseded_logs", []),
                previous["log"],
            ]
        out[rec["run_name"]] = rec
    return out


def _norm_data_dir(value: Optional[str]) -> Optional[str]:
    """Normalize an echoed data dir for comparison (``./x/`` → ``x``)."""
    if value is None:
        return None
    text = value.strip()
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def attestation_defects(rec: dict[str, Any]) -> list[str]:
    """Why this log does NOT establish the expected v2 corpus (empty list = it does).

    The ``corpus=`` token is only a label: ``RV_DATA_DIR``/``MS_DATA_DIR`` are
    documented launcher overrides, and the header used to assert ``corpus=v2``
    unconditionally, so a grid trained end to end on v1 pixels attested as clean v2.
    What is checked instead is the evidence: the echoed data dirs against
    :data:`EXPECTED_RV_DATA_DIR`/:data:`EXPECTED_MS_DATA_DIR`, and ``data_source``
    against :data:`EXPECTED_DATA_SOURCE` (an ``hf`` run ignores the CSV dirs it still
    echoes). The label is checked too, so a launcher whose label and dirs disagree is
    caught rather than trusted.
    """
    if rec.get("data_source") == "hf" and rec.get("dataset_id") is not None:
        defects = []
        if rec.get("dataset_id") != "yzk/veda-ocr-rv-ms-noto-v2":
            defects.append("Hub dataset is not the pinned v2 corpus")
        if rec.get("dataset_revision") != "7225f9adb86af6d63316b1e5a60b2b8de46e320c":
            defects.append("Hub dataset revision does not match the immutable v2 pin")
        if rec.get("corpus") != EXPECTED_CORPUS:
            defects.append("Hub corpus label is not v2")
        return defects
    defects: list[str] = []
    for side, echoed, expected in (
        ("rv", rec.get("rv_data_dir"), EXPECTED_RV_DATA_DIR),
        ("ms", rec.get("ms_data_dir"), EXPECTED_MS_DATA_DIR),
    ):
        norm = _norm_data_dir(echoed)
        if norm is None:
            defects.append(f"no {side}= data dir echoed, so the corpus is unverified")
        elif norm != expected:
            defects.append(f"{side}_data_dir={echoed} is not the expected {expected}")
    corpus = rec.get("corpus")
    if corpus is not None and corpus != EXPECTED_CORPUS:
        defects.append(f"corpus label {corpus} is not {EXPECTED_CORPUS}")
    source = rec.get("data_source")
    if source is None:
        defects.append(
            "data_source is not recoverable from this log, so data.source=hf (which "
            "ignores the echoed CSV dirs and needs a pinned artifact) cannot be ruled out"
        )
    elif source != EXPECTED_DATA_SOURCE:
        defects.append(
            f"data_source={source} is not {EXPECTED_DATA_SOURCE}, so the echoed "
            "rv=/ms= CSV dirs were NOT the images this cell consumed"
        )
    return defects


def corpus_attestation_summary(
    attestation: dict[str, dict[str, Any]],
    cells: Sequence[agg.SeedCell],
    *,
    study_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Are all present cells attested, do they name ONE corpus, and is it *the v2 one*?

    ``study_root`` enables the study-root cross-check: a matched log's own
    ``output_dir`` echo must end in the study root being aggregated. Run names are
    reused verbatim by the ``…_smoke`` root, so without this a smoke cell would
    happily borrow the real grid's attestation (and vice versa).

    ``single_corpus`` is necessary but nowhere near sufficient: it catches *mixtures*
    only, so a grid uniformly trained on v1 passes it. Each attested cell is
    therefore also validated against the expected v2 dataset dirs and data source
    (:func:`attestation_defects`), and any disagreement is a hard failure.
    """
    present = [c for c in cells if c.present]
    attested = [c for c in present if c.run_name in attestation]
    expected_root = study_root.name if study_root is not None else None
    root_mismatches = [
        {
            "run_name": c.run_name,
            "log": attestation[c.run_name].get("log"),
            "log_output_dir": attestation[c.run_name].get("output_dir"),
        }
        for c in attested
        if expected_root is not None
        and attestation[c.run_name].get("output_dir") is not None
        and Path(str(attestation[c.run_name]["output_dir"])).name != expected_root
    ]
    corpora = sorted(
        {
            str(attestation[c.run_name].get("corpus"))
            for c in attested
            if attestation[c.run_name].get("corpus") is not None
        }
    )
    data_dirs = sorted(
        {
            (
                str(attestation[c.run_name].get("rv_data_dir")),
                str(attestation[c.run_name].get("ms_data_dir")),
            )
            for c in attested
            if attestation[c.run_name].get("rv_data_dir") is not None
        }
    )
    corpus_mismatches = [
        {
            "run_name": c.run_name,
            "log": attestation[c.run_name].get("log"),
            "corpus": attestation[c.run_name].get("corpus"),
            "rv_data_dir": attestation[c.run_name].get("rv_data_dir"),
            "ms_data_dir": attestation[c.run_name].get("ms_data_dir"),
            "data_source": attestation[c.run_name].get("data_source"),
            "defects": defects,
        }
        for c in attested
        for defects in [attestation_defects(attestation[c.run_name])]
        if defects
    ]
    superseded = [
        {
            "run_name": c.run_name,
            "log": attestation[c.run_name].get("log"),
            "superseded_logs": attestation[c.run_name]["superseded_logs"],
        }
        for c in attested
        if attestation[c.run_name].get("superseded_logs")
    ]
    single_corpus = len(corpora) == 1 and len(data_dirs) <= 1
    return {
        "source": "immutable Slurm .out header echoed by the launcher",
        "n_present_cells": len(present),
        "n_attested": len(attested),
        "n_unattested": len(present) - len(attested),
        "unattested_runs": [
            c.run_name for c in present if c.run_name not in attestation
        ],
        "corpora": corpora,
        "data_dirs": [list(pair) for pair in data_dirs],
        "single_corpus": single_corpus,
        "expected_corpus": {
            "corpus": EXPECTED_CORPUS,
            "rv_data_dir": EXPECTED_RV_DATA_DIR,
            "ms_data_dir": EXPECTED_MS_DATA_DIR,
            "data_source": EXPECTED_DATA_SOURCE,
        },
        "corpus_mismatches": corpus_mismatches,
        "expected_study_root": expected_root,
        "study_root_mismatches": root_mismatches,
        "superseded": superseded,
        # The one flag a publication artifact may read: every present cell attested,
        # one corpus, that corpus validated as v2, and every log this study's own.
        "healthy": bool(present)
        and len(attested) == len(present)
        and single_corpus
        and not corpus_mismatches
        and not root_mismatches,
        "per_run": {name: attestation[name] for name in sorted(attestation)},
        "note": (
            "Absence of a log means UNATTESTED, not v1. results.json records no "
            "corpus path, so an unattested cell's corpus is established only from "
            "its W&B config / Hydra snapshot. Attestation matches a log to a cell by "
            "run name, which the _smoke study root reuses verbatim — "
            "study_root_mismatches lists any matched log whose own output_dir echo "
            "belongs to a different study root. The corpus= token is a LABEL, not "
            "evidence (RV_DATA_DIR/MS_DATA_DIR are launcher overrides): "
            "corpus_mismatches lists every attested cell whose echoed data dirs or "
            "data_source do not establish the expected v2 corpus, which is what "
            "single_corpus cannot catch — it sees mixtures, not a uniformly wrong "
            "corpus."
        ),
    }


# --------------------------------------------------------------------------- #
# Per-cell invariants that are neither coverage nor recipe.
# --------------------------------------------------------------------------- #
#: ``…_seed_<n>`` suffix of a v2 run dir, so the seed a cell was *requested* under
#: can be compared against the seed its ``results.json`` actually recorded.
_SEED_SUFFIX_RE = re.compile(r"_seed_(\d+)$")


def grid_integrity(
    cells: Sequence[agg.SeedCell], *, study_root: Optional[Path] = None
) -> dict[str, Any]:
    """Does each cell's own ``results.json`` agree with the dir it was read from?

    Three cell-level invariants the coverage and recipe checks cannot see:

    * **seed ↔ dir agreement** — the shared cell reader takes the seed from
      ``results.json`` (``data.get("seed", seed)``) while the dir name encodes the
      *requested* seed. A copied or mis-overridden run therefore lets one seed appear
      twice under two dir names, and the ratio publishes n=3 over 2 distinct seeds.
    * **distinct seeds per ratio** — the direct statement of the same hazard on the
      axis that matters: a ratio's mean/CI is only an n-seed number if its present
      cells carry n *different* seeds.
    * **unreadable ``results.json``** — a cell whose file is on disk but did not
      parse (see :func:`_results_json_parses`). Reported here rather than crashing
      the aggregation, but never silently: it is publication-blocking, because that
      cell's numbers are simply missing from every table.

    ``study_root`` is required for the last check only; without it that check is
    reported as not performed rather than as passing.
    """
    seed_mismatches = [
        {
            "run_name": c.run_name,
            "dir_seed": int(match.group(1)),
            "results_json_seed": c.seed,
        }
        for c in cells
        if c.present
        for match in [_SEED_SUFFIX_RE.search(c.run_name)]
        if match is not None and int(match.group(1)) != c.seed
    ]
    ratio_duplicates = []
    for rv_ratio in RV_RATIOS:
        seeds = [c.seed for c in cells if c.present and c.rv_ratio == rv_ratio]
        if len(seeds) != len(set(seeds)):
            ratio_duplicates.append(
                {
                    "rv_ratio": rv_ratio,
                    "n_present": len(seeds),
                    "seeds": sorted(seeds),
                    "n_distinct_seeds": len(set(seeds)),
                }
            )
    unreadable = (
        [
            c.run_name
            for c in cells
            if not c.present
            and (study_root / c.run_name / "results.json").is_file()
            and not _results_json_parses(study_root / c.run_name)
        ]
        if study_root is not None
        else []
    )
    return {
        "seed_mismatches": seed_mismatches,
        "ratio_seed_duplicates": ratio_duplicates,
        "unreadable_results_json": unreadable,
        "unreadable_check_performed": study_root is not None,
        "ok": not seed_mismatches and not ratio_duplicates and not unreadable,
        "note": (
            "read_cell_dir takes a cell's seed from its results.json while the run "
            "dir name encodes the requested seed, so these are two independent "
            "records of the same fact and must agree. A ratio whose present cells do "
            "not carry distinct seeds is not an n-seed result no matter what n the "
            "table prints."
        ),
    }


# --------------------------------------------------------------------------- #
# Warnings + aggregate payload.
# --------------------------------------------------------------------------- #
#: Substrings marking a warning as **publication-blocking**: these fail the CLI
#: (exit 1) *and* refuse the paper figures. Everything else is a warning that is
#: reported and carried into the artifacts but does not block, so a merely
#: ambiguous condition cannot make the gate un-greenable.
HARD_WARNING_MARKERS: tuple[str, ...] = (
    "not held fixed",
    "freeze_mode",
    "not a single study",
    "not constant within seed",
    "do not match their mix ratio",
    "different study root",
    "does not establish the expected v2 corpus",
    "no Slurm-log corpus attestation",
    "more than one corpus",
    "disagrees with its run dir",
    "do not carry distinct seeds",
    "results.json that did not parse",
)


def hard_warnings(warnings: Sequence[str]) -> list[str]:
    """The publication-blocking subset of :func:`audit_warnings`.

    One predicate shared by the CLI exit status and the figure gate, so "the figures
    rendered" and "the run exited 0" can never disagree — before this, the CLI wrote
    the figures to disk and *then* returned 1, leaving a publishable bundle behind a
    hard failure.
    """
    return [
        w
        for w in warnings
        if w.startswith("PARTIAL") or any(m in w for m in HARD_WARNING_MARKERS)
    ]


def audit_warnings(
    cells: Sequence[agg.SeedCell],
    coverage: dict[str, Any],
    *,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
) -> list[str]:
    """The v1 integrity checks plus the v2 recipe/corpus/cell ones, in report order."""
    warnings: list[str] = []
    banner = partial_banner(coverage)
    if banner is not None:
        warnings.append(banner)
    warnings.extend(agg.audit_warnings(cells))
    if integrity is not None:
        if integrity["seed_mismatches"]:
            warnings.append(
                "cell(s) whose results.json seed disagrees with its run dir: "
                + ", ".join(
                    f"{m['run_name']} (dir seed={m['dir_seed']}, "
                    f"results.json seed={m['results_json_seed']})"
                    for m in integrity["seed_mismatches"]
                )
            )
        if integrity["ratio_seed_duplicates"]:
            warnings.append(
                "ratio(s) whose present cells do not carry distinct seeds, so their "
                "n overstates the number of seeds behind the mean/CI: "
                + ", ".join(
                    f"{int(round(d['rv_ratio'] * 100))}:"
                    f"{int(round((1 - d['rv_ratio']) * 100))} "
                    f"n={d['n_present']} over {d['n_distinct_seeds']} distinct "
                    f"seed(s) {d['seeds']}"
                    for d in integrity["ratio_seed_duplicates"]
                )
            )
        if integrity["unreadable_results_json"]:
            warnings.append(
                "cell(s) with a results.json that did not parse (torn/partial write) "
                "and are therefore counted as ABSENT: "
                + ", ".join(integrity["unreadable_results_json"])
            )
    if subset is not None:
        if subset["inconsistent_fields"]:
            warnings.append(
                "present cells disagree on held-fixed recipe field(s) "
                f"{subset['inconsistent_fields']} — this is not a single study"
            )
        if subset["key_list_sha256_split_seeds"]:
            warnings.append(
                "key_list_sha256 is not constant within seed(s) "
                f"{subset['key_list_sha256_split_seeds']} — the ratio arms of a seed "
                "did not train on the same text subset (the hash is expected to "
                "differ ACROSS seeds, never within one)"
            )
        if subset["mix_mismatches"]:
            warnings.append(
                "cell(s) whose rv/ms train counts do not match their mix ratio: "
                + ", ".join(subset["mix_mismatches"])
            )
    if attestation is not None:
        if attestation["n_unattested"]:
            warnings.append(
                f"{attestation['n_unattested']}/{attestation['n_present_cells']} "
                "present cell(s) have no Slurm-log corpus attestation (corpus "
                "unattested, NOT assumed): " + ", ".join(attestation["unattested_runs"])
            )
        if attestation["n_attested"] and not attestation["single_corpus"]:
            warnings.append(
                "attested cells name more than one corpus / data dir pair: "
                f"corpora={attestation['corpora']} dirs={attestation['data_dirs']}"
            )
        if attestation.get("corpus_mismatches"):
            warnings.append(
                "attested log(s) whose echoed corpus evidence does not establish the "
                f"expected v2 corpus (rv={EXPECTED_RV_DATA_DIR} "
                f"ms={EXPECTED_MS_DATA_DIR} data_source={EXPECTED_DATA_SOURCE}) — the "
                "corpus= label alone is not evidence, and single_corpus cannot catch a "
                "uniformly wrong corpus: "
                + "; ".join(
                    f"{m['run_name']}←{m['log']} ({'; '.join(m['defects'])})"
                    for m in attestation["corpus_mismatches"]
                )
            )
        if attestation.get("study_root_mismatches"):
            warnings.append(
                "attested log(s) belong to a different study root than the one being "
                "aggregated (run names are shared with the _smoke root, so this "
                "attestation is not this study's): "
                + ", ".join(
                    f"{m['run_name']}←{m['log']} (output_dir={m['log_output_dir']})"
                    for m in attestation["study_root_mismatches"]
                )
            )
        if attestation.get("superseded"):
            # Not publication-blocking: the winning log is the newest by (jobid,
            # task) and is validated like any other. But it IS ambiguous — a smoke
            # run and a real run share run names — so it must not be silent.
            warnings.append(
                "cell(s) matched by several Slurm logs; the highest (jobid, task) "
                "wins and the rest are recorded as superseded_logs — check the "
                "winner is the intended run: "
                + "; ".join(
                    f"{s['run_name']}←{s['log']} (superseded: "
                    f"{', '.join(s['superseded_logs'])})"
                    for s in attestation["superseded"]
                )
            )
    return warnings


def build_aggregate(
    cells: Sequence[agg.SeedCell],
    checks: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    *,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Machine-readable v2 payload (→ ``aggregate.json``).

    Schema-compatible with the study ``aggregate.json`` for the blocks the figure
    layer reads (``grid``, ``per_ratio``, ``asymmetry``, ``retention_adaptation``),
    so ``build_paclic_paper_figures`` renders from it unchanged, plus the v2-only
    ``status`` / ``complete`` / ``coverage`` / provenance blocks.

    ``grid.seeds`` holds the seeds **actually present** (``target_seeds`` holds the
    3 planned ones) so a caption derived from it can never overstate n.
    """
    by_ratio = agg.aggregate_by_ratio(cells)
    warnings = audit_warnings(
        cells, coverage, subset=subset, attestation=attestation, integrity=integrity
    )
    return {
        "study": STUDY,
        "corpus": "v2",
        "single_root": True,
        "analysis_only": True,
        "status": status_label(coverage),
        "complete": coverage["complete"],
        "partial_banner": partial_banner(coverage),
        "grid": {
            "rv_ratios": list(RV_RATIOS),
            "seeds": coverage["seeds_present"],
            "target_seeds": list(SEEDS),
        },
        "n_cells": coverage["n_cells"],
        "n_present": coverage["n_present"],
        "coverage": coverage,
        "per_ratio": {
            f"{rv:g}": {metric: stat.as_dict() for metric, stat in metrics.items()}
            for rv, metrics in by_ratio.items()
        },
        "asymmetry": agg.asymmetry(cells),
        "retention_adaptation": agg.retention_adaptation(cells),
        "cross_check": agg.cross_check_summary(checks),
        "subset_consistency": subset,
        "corpus_attestation": attestation,
        "grid_integrity": integrity,
        "warnings": warnings,
        "hard_warnings": hard_warnings(warnings),
        "publishable": not hard_warnings(warnings)
        and bool((attestation or {}).get("healthy")),
    }


# --------------------------------------------------------------------------- #
# Rendering — Markdown (CSV tables are agg's, verbatim).
# --------------------------------------------------------------------------- #
def _pm_n(stat: agg.Stat, digits: int = 4) -> str:
    """``mean ± halfwidth (n=k)`` — a value never appears without its own n."""
    if stat.mean is None:
        return "—"
    return f"{agg._pm(stat, digits)} (n={stat.n})"


def render_coverage_table(coverage: dict[str, Any]) -> list[str]:
    """Per-ratio n table — the honest denominator behind every mean/CI below."""
    out = [
        "| RV:MS | cells present | seeds present | seeds missing |",
        "| :--- | ---: | :--- | :--- |",
    ]
    for blk in coverage["per_ratio"].values():
        rv_pct = int(round(blk["rv_ratio"] * 100))
        ms_pct = int(round(blk["ms_ratio"] * 100))
        out.append(
            f"| {rv_pct}:{ms_pct} | {blk['n_present']}/{blk['n_cells']} | "
            f"{blk['seeds_present'] or '—'} | {blk['seeds_missing'] or '—'} |"
        )
    return out


def render_results_md(
    cells: Sequence[agg.SeedCell],
    by_ratio: dict[float, dict[str, agg.Stat]],
    asym: dict[str, Any],
    ra: dict[str, Any],
    checks: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    *,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
) -> str:
    """Human-readable results page: status first, then the tables, n on every row."""
    status = status_label(coverage)
    banner = partial_banner(coverage)
    out: list[str] = []
    out.append("# PACLIC v2 main grid — RV/MS mix ratios × seeds (corrected corpus)")
    out.append("")
    out.append(
        f"**Status:** {status} · **Corpus:** v2 (accent marks corrected) · "
        f"**Cells:** {coverage['n_present']}/{coverage['n_cells']} · "
        f"**Seeds present:** `{coverage['seeds_present']}` · **Analysis-only** "
        "(no training, no inference)."
    )
    out.append("")
    if banner is not None:
        out.append(f"> **{banner}**")
        out.append("")
    # Every non-banner integrity warning is surfaced in the page itself, not only on
    # stderr — a reader of the committed doc must see them too.
    other_warnings = [
        w
        for w in audit_warnings(
            cells,
            coverage,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
        )
        if w != banner
    ]
    if other_warnings:
        out.append("> **Integrity warnings:**")
        for warning in other_warnings:
            out.append(f"> - {warning}")
        out.append("")
    out.append(
        "All 15 cells are trained by "
        f"`{LAUNCHER}` into ONE study root with uniform "
        "`rv_<pct>_ms_<pct>_seed_<n>` names — there is no inherited seed-42 arm, so "
        "no cell here is reused from v1 and the v1 aggregator's factorial/multiseed "
        "reconciliation does not apply. Statistics (mean, sample SD, Student-t 95% "
        "CI with df=n−1), the paired contrasts, and the exact-match recompute are "
        "reused verbatim from `scripts.evaluation.aggregate_accent_ratio_multiseed`. "
        "**v2 numbers are never pooled with, or subtracted from, v1 numbers**: the "
        "two are different image corpora."
    )
    out.append("")
    out.append("## 0. Coverage")
    out.append("")
    out.extend(render_coverage_table(coverage))
    out.append("")

    out.append("## 1. Per mix ratio — mean ± 95% CI across the present seeds")
    out.append("")
    out.append(
        "`cells` is that ratio's present-cell count. **Every metric then carries its "
        "own `(n=k)`, because the denominators differ**: the runner writes "
        "`results.json` before the prediction CSVs, and exact-match is re-sourced "
        "from those CSVs (study), so mid-flight a cell can contribute CER but not EM. "
        "A metric with n<3 is not a 3-seed number; with n<2 it has no CI at all. "
        "`cells` is coverage, never a metric's n."
    )
    out.append("")
    out.append(
        "| RV:MS | cells | RV CER | MS CER | avg CER | RV chrF | MS chrF | "
        "RV EM | MS EM | RV accF1 | MS accF1 |"
    )
    out.append(
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for rv_ratio in RV_RATIOS:
        m = by_ratio[rv_ratio]
        rv_pct = int(round(rv_ratio * 100))
        ms_pct = int(round((1.0 - rv_ratio) * 100))
        blk = coverage["per_ratio"][f"{rv_ratio:g}"]
        out.append(
            f"| {rv_pct}:{ms_pct} | {blk['n_present']}/{blk['n_cells']} | "
            f"{_pm_n(m['rv_cer'])} | {_pm_n(m['ms_cer'])} | {_pm_n(m['avg_cer'])} | "
            f"{_pm_n(m['rv_chrf'], 2)} | {_pm_n(m['ms_chrf'], 2)} | "
            f"{_pm_n(m['rv_exact_match'], 3)} | {_pm_n(m['ms_exact_match'], 3)} | "
            f"{_pm_n(m['rv_accent_f1'], 3)} | {_pm_n(m['ms_accent_f1'], 3)} |"
        )
    out.append("")

    out.append("## 2. Zero-shot asymmetry (endpoints, paired by seed)")
    out.append("")
    out.append(
        "RV→MS = the RV-only model (`rv_ratio=1.0`) read on **MS** test; MS→RV = the "
        "MS-only model (`rv_ratio=0.0`) read on **RV** test. Each arm carries its own "
        "n; `n paired` is the number of seeds present at **both** endpoints, and only "
        "that n backs the difference column."
    )
    out.append("")
    out.append("| metric | n paired | RV→MS | MS→RV | diff (MS→RV − RV→MS) |")
    out.append("| :--- | ---: | ---: | ---: | :--- |")
    for metric, digits in (("cer", 4), ("chrf", 2), ("accent_f1", 3)):
        block = asym["zero_shot"][metric]
        rv2ms = agg.Stat(**agg._stat_from_dict(block["rv_to_ms"]))
        ms2rv = agg.Stat(**agg._stat_from_dict(block["ms_to_rv"]))
        out.append(
            f"| {metric} | {block['diff']['n']} | {_pm_n(rv2ms, digits)} | "
            f"{_pm_n(ms2rv, digits)} | {agg._diff_phrase(block, digits)} |"
        )
    out.append("")
    idc = asym["in_domain_cer"]
    rv_own = agg.Stat(**agg._stat_from_dict(idc["rv_own"]))
    ms_own = agg.Stat(**agg._stat_from_dict(idc["ms_own"]))
    out.append(
        f"In-domain endpoint CER (own notation): RV {_pm_n(rv_own)} vs "
        f"MS {_pm_n(ms_own)}; diff (MS−RV) {agg._diff_phrase(idc)}."
    )
    out.append("")

    out.append("## 3. Retention–adaptation with a 25% minority share (paired by seed)")
    out.append("")
    out.append("| scenario | quantity | n | Δ (added − baseline) |")
    out.append("| :--- | :--- | ---: | :--- |")
    for group, quantity, label, digits in (
        ("rv_primary_add_ms", "rv_retention_cer", "RV retention (Δ RV CER)", 4),
        (
            "rv_primary_add_ms",
            "ms_adaptation_accent_f1",
            "MS adaptation (Δ MS accF1)",
            3,
        ),
        ("ms_primary_add_rv", "ms_retention_cer", "MS retention (Δ MS CER)", 4),
        (
            "ms_primary_add_rv",
            "rv_adaptation_accent_f1",
            "RV adaptation (Δ RV accF1)",
            3,
        ),
    ):
        contrast = ra[group][quantity]
        scenario = (
            "RV-primary +25% MS"
            if group == "rv_primary_add_ms"
            else "MS-primary +25% RV"
        )
        out.append(
            f"| {scenario} | {label} | {contrast['diff']['n']} | "
            f"{agg._diff_phrase(contrast, digits)} |"
        )
    out.append("")

    out.append("## 4. Per-cell training state + corpus attestation")
    out.append("")
    out.append(
        "`best epoch`/`best step`/`global step` come from each cell's own "
        "`results.json` `training_state` (the v2 launcher writes it inline, so no "
        "checkpoint or Slurm-log recovery is needed). `corpus` is attested from the "
        "immutable Slurm `.out` header — `results.json` records no corpus path, and "
        "`key_list_sha256` cannot discriminate v1 from v2."
    )
    out.append("")
    out.append(
        "| RV:MS | seed | present | best epoch | best step | global step | "
        "state source | corpus (attested) |"
    )
    out.append("| :--- | ---: | :--- | ---: | ---: | ---: | :--- | :--- |")
    per_run = (attestation or {}).get("per_run", {})
    for cell in cells:
        ts = cell.training_state or {}
        be = ts.get("best_epoch")
        bs = ts.get("best_step")
        gs = ts.get("global_step")
        att = per_run.get(cell.run_name) or {}
        corpus = att.get("corpus") or "unattested"
        if not cell.present and att.get("corpus"):
            # A log without a results.json: the cell is running or failed. The
            # attestation is about the log, so say so instead of implying a result.
            corpus = f"{corpus} (log only)"
        rv_pct = int(round(cell.rv_ratio * 100))
        ms_pct = int(round(cell.ms_ratio * 100))
        out.append(
            f"| {rv_pct}:{ms_pct} | {cell.seed} | {'yes' if cell.present else 'NO'} | "
            f"{'—' if be is None else f'{be:g}'} | {'—' if bs is None else bs} | "
            f"{'—' if gs is None else gs} | "
            f"{cell.training_state_source if cell.present else '—'} | {corpus} |"
        )
    out.append("")

    if subset is not None:
        out.append("## 5. Recipe / subset consistency")
        out.append("")
        # The verdict must name the SCOPE it was checked at. Reporting the per-seed
        # subset hash as a grid-wide held-fixed field made a healthy grid read
        # "INCONSISTENT — this is not a single study".
        grid_verdict = (
            "consistent" if subset["grid_wide_consistent"] else "INCONSISTENT"
        )
        out.append(
            f"**Grid-wide held-fixed fields** — checked across all "
            f"{subset['n_cells_checked']} present cell(s), one distinct value required "
            f"per field ({', '.join(sorted(subset['held_fixed']))}): "
            f"**{grid_verdict}**"
            + (
                ""
                if subset["grid_wide_consistent"]
                else f" — differing: {subset['inconsistent_fields']}"
            )
            + "."
        )
        out.append("")
        sha_verdict = (
            "consistent"
            if subset["key_list_sha256_consistent"]
            else "INCONSISTENT WITHIN A SEED"
        )
        per_seed_phrase = "; ".join(
            f"seed {blk['seed']}: {len(blk['key_list_sha256'])} hash(es) over "
            f"{len(blk['runs'])} cell(s)"
            for blk in subset["key_list_sha256_per_seed"].values()
        )
        out.append(
            f"**`key_list_sha256`** — checked WITHIN each seed across the five ratios, "
            f"NOT grid-wide: **{sha_verdict}**"
            + (
                ""
                if subset["key_list_sha256_consistent"]
                else f" — split seed(s): {subset['key_list_sha256_split_seeds']}"
            )
            + f". Per seed: {per_seed_phrase or 'no cell recorded a hash'}. "
            f"{len(subset['key_list_sha256_distinct_grid_wide'])} distinct value(s) "
            f"grid-wide, {len(SEEDS)} expected (one per seed). "
            f"{KEY_LIST_SHA_SEED_NOTE}"
        )
        out.append("")
        out.append(
            "**Mix realization** (`rv/ms_train_samples` == floor split of the "
            "requested total at that ratio): "
            + (
                "all cells match"
                if not subset["mix_mismatches"]
                else f"MISMATCH in {subset['mix_mismatches']}"
            )
            + f". {KEY_LIST_SHA_NOTE}"
        )
        out.append("")

    if integrity is not None:
        out.append("## 6. Per-cell invariants (seed ↔ dir, distinct seeds, parse)")
        out.append("")
        out.append(
            "Each cell's `results.json` `seed` must equal the seed its dir name "
            "encodes (the loader reads one from the file and the other from the "
            "request, so they are two independent records), each ratio's present "
            "cells must carry **distinct** seeds, and a `results.json` on disk must "
            "parse. Verdict: **"
            + ("all hold" if integrity["ok"] else "VIOLATED")
            + "** — seed↔dir mismatches: "
            + (
                "none"
                if not integrity["seed_mismatches"]
                else str([m["run_name"] for m in integrity["seed_mismatches"]])
            )
            + "; ratios without distinct seeds: "
            + (
                "none"
                if not integrity["ratio_seed_duplicates"]
                else str([d["rv_ratio"] for d in integrity["ratio_seed_duplicates"]])
            )
            + "; unreadable `results.json`: "
            + (
                (
                    "none"
                    if integrity["unreadable_check_performed"]
                    else "not checked (no study root given)"
                )
                if not integrity["unreadable_results_json"]
                else str(integrity["unreadable_results_json"])
            )
            + "."
        )
        out.append("")

    cc = agg.cross_check_summary(checks)
    if cc["n_views_checked"]:
        parts = []
        for key, label in (
            ("delta_cer", "CER"),
            ("delta_wer", "WER"),
            ("delta_accent_f1", "accent F1"),
            ("delta_exact_match", "exact-match"),
        ):
            blk = cc[key]
            if blk["max"] is not None:
                parts.append(f"max |Δ{label}|={blk['max']:.5f}")
        out.append(
            "**Provenance cross-check** — over "
            f"{cc['n_views_checked']} re-evaluated views, the runner `results.json` "
            "metrics reproduce under the study inference-free evaluator to "
            + (", ".join(parts) if parts else "no comparable metric")
            + " (exact-match is *sourced* from that evaluator for every cell, study)."
        )
        out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Provenance manifest.
# --------------------------------------------------------------------------- #
def build_manifest(
    cells: Sequence[agg.SeedCell],
    checks: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
    *,
    study_root: Path,
    shared_study_root: Optional[str],
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Data-provenance manifest for the v2 bundle (no v1 notes, no v1 roots)."""
    per_run = (attestation or {}).get("per_run", {})
    return {
        "study": STUDY,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        "corpus": "v2 (accent marks corrected; same ISO text as v1, different pixels)",
        "analysis_only": True,
        "no_training_no_inference": True,
        "status": status_label(coverage),
        "complete": coverage["complete"],
        "partial_banner": partial_banner(coverage),
        "coverage": coverage,
        "launcher": LAUNCHER,
        "study_root": str(study_root),
        "shared_study_root": shared_study_root,
        "grid": {
            "rv_ratios": list(RV_RATIOS),
            "target_seeds": list(SEEDS),
            "seeds_present": coverage["seeds_present"],
            "single_root": True,
            "run_name_pattern": "rv_<pct>_ms_<pct>_seed_<n> (uniform, incl. seed 42)",
        },
        "why_not_the_v1_aggregator": (
            "scripts.evaluation.aggregate_accent_ratio_multiseed encodes v1's "
            "provenance split: expected_run_name() drops the _seed_42 suffix for "
            "REUSED_SEED=42, _cell_dir() routes seed 42 to --factorial-root, "
            "load_cell() stamps source=prior-study-reference, and "
            "build_manifest() emits issue 123 with the seed-42 chrF / training-state "
            "recovery notes and the v1 pins manifest. Pointed at the v2 root it "
            "would read all five seed-42 cells as ABSENT (wrong dir name) and label "
            "the rest with v1 provenance, so v2 gets its own loader/manifest here "
            "and reuses only the study-independent reader, statistics, contrasts, "
            "and table renderers."
        ),
        "reuse": {
            "cell_reader": "scripts.evaluation.aggregate_accent_ratio_multiseed.read_cell_dir",
            "statistics": "scripts.evaluation.aggregate_accent_ratio_multiseed.summarize (mean / sample SD / Student-t 95% CI, df=n-1)",
            "contrasts": "aggregate_by_ratio / asymmetry / retention_adaptation / cross_check_summary",
            "tables": "render_per_seed_csv / render_per_ratio_csv / render_asymmetry_csv / render_retention_csv",
        },
        "cells": [
            {
                "rv_ratio": c.rv_ratio,
                "ms_ratio": c.ms_ratio,
                "seed": c.seed,
                "run_name": c.run_name,
                "source": c.source,
                "present": c.present,
                "results_path": c.results_path,
                "shared_path": c.shared_path,
                "rv_chrf_source": c.rv_chrf_source,
                "ms_chrf_source": c.ms_chrf_source,
                "rv_exact_match_source": c.rv_exact_match_source,
                "ms_exact_match_source": c.ms_exact_match_source,
                "training_state_source": c.training_state_source,
                "training_state_recovery_method": c.training_state_recovery_method,
                "model_base": c.model_base,
                "corpus_attestation": per_run.get(c.run_name),
            }
            for c in cells
        ],
        "subset_consistency": subset,
        "corpus_attestation": attestation,
        "grid_integrity": integrity,
        "metric_sourcing_note": (
            "CER / WER / chrF / accent metrics are the runner's canonical "
            "training-eval numbers from each cell's own results.json — the v2 runner "
            "records chrF for every cell, so nothing is recomputed to fill a gap "
            "(rv_/ms_chrf_source=results.json). exact_match (rate + count) is "
            "recomputed inference-free from the committed {rv,ms}_predictions.csv "
            "via scripts.evaluation.prediction_metrics under the pinned NFC+strip "
            "normalization (study/study), so it regenerates from saved predictions "
            "alone; the runner value feeds only the cross_check delta."
        ),
        "training_state_note": (
            "Every v2 cell's results.json carries training_state inline "
            "(recovery_method=results_json). The study Slurm-log recovery is NOT "
            "wired in here: it resolves logs through the v1 RUN_REGISTRY, whose "
            "run ids include names that collide with v2's seed-43/44 dirs."
        ),
        "evaluator": {
            "module": "scripts.evaluation.prediction_metrics",
            "generic_ocr_backend": "scripts.training.trocr.metrics.OCRMetrics",
            "normalization": {
                "form": "NFC",
                "strip": True,
                "collapse_whitespace": False,
            },
            "dependency_pin": agg.evaluator_dependency_pin(),
        },
        "cross_check": agg.cross_check_summary(checks),
        "limitations": [
            "v1 and v2 are different image corpora: a v1→v2 delta confounds the "
            "corpus change with nothing else only if the recipe is identical, and "
            "it is never reported as a result here. No v1 cell is pooled in.",
            "results.json records no corpus path; corpus identity is attested only "
            "from the Slurm .out header (corpus_attestation), and an unattested "
            "cell is reported as such. The header's corpus= token is a LABEL derived "
            "by the launcher, so the echoed data dirs + data_source are what is "
            "validated against the expected v2 dataset dirs.",
            "key_list_sha256 hashes ISO text keys, identical between v1 and v2 by "
            "construction, so it is not corpus evidence. It is also seeded, hence a "
            "within-seed rather than a grid-wide invariant.",
            "Paper figures are gated on complete==true AND a healthy corpus "
            "attestation AND no publication-blocking warning; a partial, unattested, "
            "wrong-corpus or invariant-violating grid renders no publication figure.",
        ],
        "reproduce": (
            "uv run python -m scripts.evaluation.build_paclic_v2_maingrid_analysis "
            f"--study-root {DEFAULT_STUDY_ROOT} --out-dir {DEFAULT_OUT_DIR}"
        ),
        "generated_by": "scripts.evaluation.build_paclic_v2_maingrid_analysis",
    }


# --------------------------------------------------------------------------- #
# Bundle orchestration.
# --------------------------------------------------------------------------- #
def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[{STUDY}] wrote {path}", file=sys.stderr)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write(
        path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def build_bundle(
    cells: Sequence[agg.SeedCell],
    checks: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    study_root: Path,
    shared_study_root: Optional[str] = None,
    subset: Optional[dict[str, Any]] = None,
    attestation: Optional[dict[str, Any]] = None,
    integrity: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write ``tables/`` + ``aggregate.json`` + ``manifest.json`` + ``results.md``.

    Returns the aggregate payload (also the input to :func:`build_figures`).

    ``integrity`` defaults to a freshly computed :func:`grid_integrity` over
    ``study_root`` rather than to "not checked", so a caller cannot get a bundle
    whose per-cell invariants were silently skipped.
    """
    if integrity is None:
        integrity = grid_integrity(cells, study_root=study_root)
    coverage = coverage_summary(cells)
    by_ratio = agg.aggregate_by_ratio(cells)
    asym = agg.asymmetry(cells)
    ra = agg.retention_adaptation(cells)

    tables = out_dir / "tables"
    _write(tables / "per_seed.csv", agg.render_per_seed_csv(cells))
    _write(tables / "per_ratio.csv", agg.render_per_ratio_csv(by_ratio))
    _write(tables / "asymmetry.csv", agg.render_asymmetry_csv(asym))
    _write(tables / "retention_adaptation.csv", agg.render_retention_csv(ra))
    _write(
        out_dir / "results.md",
        render_results_md(
            cells,
            by_ratio,
            asym,
            ra,
            checks,
            coverage,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
        ),
    )
    aggregate = build_aggregate(
        cells,
        checks,
        coverage,
        subset=subset,
        attestation=attestation,
        integrity=integrity,
    )
    _write_json(out_dir / "aggregate.json", aggregate)
    _write_json(
        out_dir / "manifest.json",
        build_manifest(
            cells,
            checks,
            coverage,
            study_root=study_root,
            shared_study_root=shared_study_root,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
        ),
    )
    return aggregate


# --------------------------------------------------------------------------- #
# Figures — reuse the PACLIC paper figure layer, gated on a complete grid.
# --------------------------------------------------------------------------- #
#: The paper figures whose input is exactly a study-shaped aggregate payload
#: (``per_ratio`` / ``grid`` / ``retention_adaptation``), so they render from the
#: v2 aggregate with no change to the figure code. The other four paper figures
#: read the study transfer-analysis bundle (permutation test, plateau, error
#: decomposition), which has no v2 counterpart yet.
V2_FIGURE_STEMS: tuple[str, ...] = (
    "fig1_per_ratio_performance",
    "fig4_retention_adaptation",
)

#: v2-safe replacements for the v1 figure captions. The v1 ``shows`` prose asserts
#: v1 *findings* (e.g. "the primary notation's CER barely moves", "accent F1 jumps
#: sharply"); repeating them over v2 data would be an unverified claim, so the v2
#: manifest/README describe what is plotted and nothing more.
V2_FIGURE_SHOWS: dict[str, str] = {
    "fig1_per_ratio_performance": (
        "CER, WER and accent F1 on the RV-test and MS-test sets across the five "
        "RV:MS training mixes (100:0 … 0:100) of the v2 (corrected-accent) corpus, "
        "mean ± 95% CI over the 3 seeds. Descriptive: the direction of the v2 "
        "trade-off is read off the plotted values, not asserted here."
    ),
    "fig4_retention_adaptation": (
        "Paired before/after when a 25% minority notation is added to a "
        "single-notation v2 baseline: (a) the primary notation's test CER, (b) the "
        "added minority's accent F1, with Δ the seed-paired mean difference; "
        "mean ± 95% CI over the 3 seeds. Descriptive: no retention/adaptation "
        "verdict is asserted in this caption — read it from the plotted Δ and CIs."
    ),
}


def _retarget_sources(
    figure: dict[str, Any], bundle_dir: str, *, v1_dir: str
) -> dict[str, Any]:
    """Repoint a v1 figure record's ``sources`` at the v2 bundle.

    The figure builders return provenance strings hard-coded to the study bundle
    path; a v2 figure that cited them would misattribute its own data.
    """
    sources = [
        {**s, "file": str(s["file"]).replace(v1_dir, bundle_dir)}
        for s in figure.get("sources", [])
    ]
    return {**figure, "sources": sources}


def build_figures(
    aggregate: dict[str, Any],
    out_dir: Path,
    *,
    bundle_dir: str = str(DEFAULT_OUT_DIR),
    make_pdf: bool = True,
    dpi: int = 300,
) -> list[dict[str, Any]]:
    """Render the v2 main-grid paper figures from a **complete** v2 aggregate.

    Reuses ``build_paclic_paper_figures``' figure functions verbatim — same
    colorblind-safe palette, same deterministic Agg rendering, same ≥300-dpi PNG +
    vector PDF — because the v2 aggregate is schema-compatible with the payload
    they already read.

    Three gates, all of which must pass before anything is written:

    * **coverage** — :class:`IncompleteGridError` unless every cell is present: the
      figures caption themselves "n=3 seeds", which a partial grid cannot honour.
    * **corpus** — :class:`UnverifiedCorpusError` unless the attestation is
      *healthy*: every present cell attested, one corpus, and that corpus validated
      as v2 by its data dirs and data source. Figures labelled "v2 (corrected
      accent)" plotted from unverified pixels are the whole hazard, and a skipped
      attestation (``--no-slurm-attestation``) counts as unverified — not checking is
      not the same as passing.
    * **invariants** — :class:`GridInvariantError` on any publication-blocking
      warning (:func:`hard_warnings`), the same predicate the CLI exits non-zero on,
      so figures on disk and a zero exit status can never disagree.
    """
    if not aggregate.get("complete"):
        coverage = aggregate.get("coverage") or {}
        raise IncompleteGridError(
            "refusing to render v2 paper figures from a partial grid: "
            f"{coverage.get('n_present')}/{coverage.get('n_cells')} cells present. "
            f"{aggregate.get('partial_banner')}"
        )
    attestation = aggregate.get("corpus_attestation")
    if attestation is None:
        raise UnverifiedCorpusError(
            "refusing to render v2 paper figures without a corpus attestation: the "
            "grid's corpus was not checked (--no-slurm-attestation, or no attestation "
            "passed to build_aggregate). results.json records no corpus path, so "
            "nothing else in the bundle can tell v1 pixels from v2 pixels, and these "
            "figures are labelled v2."
        )
    if not attestation.get("healthy"):
        raise UnverifiedCorpusError(
            "refusing to render v2 paper figures from an unhealthy corpus "
            "attestation: "
            f"{attestation.get('n_attested')}/{attestation.get('n_present_cells')} "
            f"cells attested, single_corpus={attestation.get('single_corpus')}, "
            f"corpus_mismatches="
            f"{[m['run_name'] for m in attestation.get('corpus_mismatches') or []]}, "
            "study_root_mismatches="
            f"{[m['run_name'] for m in attestation.get('study_root_mismatches') or []]}"
            f". Expected corpus: {attestation.get('expected_corpus')}"
        )
    blocking = hard_warnings(aggregate.get("warnings") or [])
    if blocking:
        raise GridInvariantError(
            "refusing to render v2 paper figures while publication-blocking "
            f"warning(s) stand: {blocking}"
        )
    from . import build_paclic_paper_figures as bpf

    figures = [
        bpf.fig_per_ratio_performance(aggregate, out_dir, make_pdf=make_pdf, dpi=dpi),
        bpf.fig_retention_adaptation(aggregate, out_dir, make_pdf=make_pdf, dpi=dpi),
    ]
    figures = [
        {
            **_retarget_sources(fig, bundle_dir, v1_dir=bpf.MULTISEED_DIR),
            "shows": V2_FIGURE_SHOWS.get(fig["stem"], fig["shows"]),
        }
        for fig in figures
    ]
    stems = tuple(f["stem"] for f in figures)
    if stems != V2_FIGURE_STEMS:
        raise RuntimeError(f"unexpected v2 figure set {stems!r}")
    leaked = [
        s["file"]
        for f in figures
        for s in f["sources"]
        if bpf.MULTISEED_DIR in str(s["file"])
    ]
    if leaked:
        raise RuntimeError(
            f"v1 bundle paths leaked into v2 figure provenance: {leaked}"
        )

    _write_json(
        out_dir / "manifest.json",
        build_figures_manifest(
            figures, aggregate, bundle_dir=bundle_dir, make_pdf=make_pdf, dpi=dpi
        ),
    )
    _write(
        out_dir / "README.md",
        render_figures_readme(
            figures, aggregate, dpi=dpi, out_dir=out_dir, bundle_dir=bundle_dir
        ),
    )
    return figures


def build_figures_manifest(
    figures: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    bundle_dir: str,
    make_pdf: bool,
    dpi: int,
) -> dict[str, Any]:
    """Provenance for the v2 figure set (v2 bundle only; v1 caveats not inherited).

    The corpus claim and the bundle's warnings are **carried, not restated**: this
    manifest used to republish ``corpus: "v2 (accent marks corrected)"`` and
    ``status: COMPLETE`` with the attestation and the warning list stripped out, so
    the figure bundle asserted more than the data bundle it came from.
    """
    attestation = aggregate.get("corpus_attestation")
    return {
        "study": STUDY,
        "paper": "PACLIC 2026 — Cross-Notation Transfer in Vedic OCR",
        # The claim, then the evidence for it, in the same payload.
        "corpus": "v2 (accent marks corrected)",
        "corpus_claim_basis": (
            "Not asserted by this builder: the figures are gated on a HEALTHY "
            "corpus_attestation (below) — every present cell attested from its "
            "immutable Slurm .out header, one corpus, and that corpus validated "
            "against the expected v2 dataset dirs and data_source. Without that the "
            "figures are refused, so this label is only ever written next to the "
            "evidence that earned it."
        ),
        "corpus_attestation": attestation,
        "analysis_only": True,
        "no_training_no_inference": True,
        "plot_only_committed_numbers": True,
        "status": aggregate["status"],
        "complete": aggregate["complete"],
        "publishable": aggregate.get("publishable"),
        "warnings": aggregate.get("warnings") or [],
        "hard_warnings": aggregate.get("hard_warnings") or [],
        "grid_integrity": aggregate.get("grid_integrity"),
        "subset_consistency": aggregate.get("subset_consistency"),
        "grid": aggregate["grid"],
        "coverage": aggregate["coverage"],
        "input_bundles": {"v2_maingrid": f"{bundle_dir}/aggregate.json"},
        "renders_with": "scripts.evaluation.build_paclic_paper_figures (fig1, fig4)",
        "not_rendered": {
            "reason": (
                'This builder renders main-grid figures only. The data-efficiency figure is regenerated separately; effect-size and accent-error analyses are released as tables.'            ),
            "stems": [
                "fig2_cross_notation_asymmetry",
                "fig3_plateau_worst_vs_average",
                "fig5_error_decomposition",
                "fig6_data_efficiency",
            ],
        },
        "caption_note": (
            "The v1 figure captions asserted v1 findings; the v2 records here "
            "describe only what is plotted (see V2_FIGURE_SHOWS)."
        ),
        "rendering": {
            "backend": "matplotlib Agg",
            "png_dpi": dpi,
            "pdf": "vector" if make_pdf else "skipped",
            "deterministic": (
                "no RNG, no timestamp; PDF CreationDate cleared and PNG Software "
                "pinned so figures regenerate byte-identically"
            ),
            "palette": "inherited from scripts.evaluation.build_paclic_paper_figures",
        },
        "figures": [
            {
                "stem": f["stem"],
                "title": f["title"],
                "shows": f["shows"],
                "files": f["files"],
                "sources": f["sources"],
            }
            for f in figures
        ],
        "generated_by": "scripts.evaluation.build_paclic_v2_maingrid_analysis",
    }


def render_figures_readme(
    figures: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    dpi: int,
    out_dir: Path,
    bundle_dir: str,
) -> str:
    """README for the v2 figure dir — status, provenance, and what is NOT here."""
    cov = aggregate["coverage"]
    att = aggregate.get("corpus_attestation") or {}
    lines = [
        "# PACLIC 2026 figures — v2 main grid (corrected-accent corpus)",
        "",
        f"**Status:** {aggregate['status']} · **Cells:** "
        f"{cov['n_present']}/{cov['n_cells']} · **Seeds:** "
        f"`{aggregate['grid']['seeds']}`.",
        "",
        f"**Corpus attested:** {att.get('n_attested')}/"
        f"{att.get('n_present_cells')} present cells, from the immutable Slurm `.out` "
        f"headers; corpora `{att.get('corpora')}`, data dirs `{att.get('data_dirs')}`, "
        f"validated against `{att.get('expected_corpus')}`. These figures are refused "
        "unless that attestation is healthy — the v2 label is never asserted by the "
        "figure builder itself, because `results.json` records no corpus path.",
        "",
    ]
    warnings = aggregate.get("warnings") or []
    if warnings:
        lines.append(
            "> **Warnings carried from the data bundle** (none of them "
            "publication-blocking, or these figures would not exist):"
        )
        lines.extend(f"> - {w}" for w in warnings)
        lines.append("")
    lines += [
        "Data-viz only — no training or inference. Every value is read from the v2 "
        "bundle's `aggregate.json`; the figure code, palette and determinism are "
        "reused from `scripts.evaluation.build_paclic_paper_figures`. Regenerate:",
        "",
        "```bash",
        "uv run python -m scripts.evaluation.build_paclic_v2_maingrid_analysis \\",
        f"  --out-dir {bundle_dir} --figures-out-dir {out_dir}",
        "```",
        "",
        f"Each figure is a ≥{dpi}-dpi PNG plus a vector PDF (matplotlib Agg, "
        "deterministic). Machine-readable provenance: `manifest.json`.",
        "",
        "| # | figure | shows | source data |",
        "| :-- | :--- | :--- | :--- |",
    ]
    for i, fig in enumerate(figures, start=1):
        files = ", ".join(f"`{name}`" for name in fig["files"])
        srcs = "; ".join(f"`{s['file']}`" for s in fig["sources"])
        lines.append(f"| {i} | {files} | {fig['shows']} | {srcs} |")
    lines.append("")
    lines.append(
        '> These figures use the corrected v2 corpus and the released numeric tables.'
    )
    lines.append("")
    lines.append(
        '> This builder renders the two main-grid figures. The data-efficiency figure is regenerated separately; effect-size and accent-error analyses are released as tables.'    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
        help="The single v2 main-grid study root holding all 15 run dirs.",
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
            "Dir of immutable Slurm .out logs, used to attest each cell's corpus "
            "from the launcher's header echo. Absent dir → every cell reported "
            "'unattested' (never assumed)."
        ),
    )
    parser.add_argument(
        "--no-slurm-attestation",
        action="store_true",
        help="Skip the Slurm-log corpus attestation entirely.",
    )
    parser.add_argument(
        "--no-cross-check",
        action="store_true",
        help="Skip re-evaluating every view against results.json.",
    )
    parser.add_argument(
        "--figures-out-dir",
        type=Path,
        default=None,
        help=(
            "Render the v2 paper figures here. Requires --out-dir (the figures cite "
            "the bundle as their source, so the bundle must actually be written). "
            "Refused, with a note, while the grid is incomplete, its corpus "
            "attestation unhealthy, or any publication-blocking warning stands. "
            f"Conventional: {DEFAULT_FIGURES_OUT_DIR}"
        ),
    )
    parser.add_argument(
        "--no-pdf", action="store_true", help="Skip the vector PDF copies."
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (>=300 for publication)."
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # The figures' own manifest/README cite the bundle as their source, so rendering
    # them without writing that bundle publishes a provenance pointer at a path that
    # does not exist. Refuse before doing any work.
    if args.figures_out_dir is not None and args.out_dir is None:
        print(
            f"[{STUDY}][fatal] --figures-out-dir requires --out-dir: the figure "
            "manifest and README cite the bundle's aggregate.json as their data "
            "source, and without --out-dir no bundle is written, so they would cite a "
            "path this run never produced.",
            file=sys.stderr,
        )
        return 2

    cells, checks = load_grid(
        args.study_root,
        cross_check=not args.no_cross_check,
        shared_study_root=args.shared_study_root,
    )
    coverage = coverage_summary(cells)
    subset = subset_consistency(subset_provenance(args.study_root, cells), cells)
    integrity = grid_integrity(cells, study_root=args.study_root)
    attestation = (
        None
        if args.no_slurm_attestation
        else corpus_attestation_summary(
            attest_corpus_from_slurm_logs(args.slurm_log_dir),
            cells,
            study_root=args.study_root,
        )
    )

    warnings = audit_warnings(
        cells, coverage, subset=subset, attestation=attestation, integrity=integrity
    )
    for warning in warnings:
        print(f"[{STUDY}][warn] {warning}", file=sys.stderr)
    # Computed BEFORE the figures: a hard failure must not leave a publishable
    # figure bundle on disk just because the exit code came last.
    hard = hard_warnings(warnings)

    if args.out_dir is not None:
        aggregate = build_bundle(
            cells,
            checks,
            args.out_dir,
            study_root=args.study_root,
            shared_study_root=args.shared_study_root,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
        )
    else:
        aggregate = build_aggregate(
            cells,
            checks,
            coverage,
            subset=subset,
            attestation=attestation,
            integrity=integrity,
        )
        print(
            render_results_md(
                cells,
                agg.aggregate_by_ratio(cells),
                agg.asymmetry(cells),
                agg.retention_adaptation(cells),
                checks,
                coverage,
                subset=subset,
                attestation=attestation,
                integrity=integrity,
            )
        )

    if args.figures_out_dir is not None:
        if hard:
            print(
                f"[{STUDY}][warn] refusing to render v2 paper figures: this run is a "
                f"hard failure ({len(hard)} publication-blocking warning(s)), so no "
                "figure bundle is written. First: "
                f"{hard[0]}",
                file=sys.stderr,
            )
        else:
            try:
                figures = build_figures(
                    aggregate,
                    args.figures_out_dir,
                    bundle_dir=str(args.out_dir),
                    make_pdf=not args.no_pdf,
                    dpi=args.dpi,
                )
            except FigureGateError as exc:
                print(f"[{STUDY}][warn] {exc}", file=sys.stderr)
            else:
                print(
                    f"[{STUDY}] {len(figures)} figures -> {args.figures_out_dir}",
                    file=sys.stderr,
                )

    # Non-zero while the grid is incomplete or a present-cell invariant is broken,
    # so a caller can gate publication on the exit status alone. Same predicate the
    # figure gate uses, so "figures on disk" and "exit 0" cannot disagree.
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
