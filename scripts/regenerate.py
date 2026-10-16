"""Regenerate paper statistics and figures from the small released result tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.evaluation import build_paclic_v2_maingrid_analysis as main_grid
from scripts.evaluation import build_paclic_v2_data_efficiency_analysis as efficiency
from scripts.evaluation import build_paclic_v2_effect_sizes_and_plateau as effects
from scripts.evaluation import aggregate_accent_ratio_multiseed as agg
from scripts.evaluation import build_paclic_v2_accent_error_decomposition as decomposition

ROOT = Path(__file__).resolve().parents[1]


def read_cells(path: Path) -> list[agg.SeedCell]:
    """Reconstruct metric cells from the released, full-precision per-seed table."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cells = []
    for row in rows:
        views = {}
        for view in ("rv", "ms"):
            views[view] = {
                metric: float(row[f"{view}_{metric}"])
                for metric in agg.VIEW_METRICS
                if row.get(f"{view}_{metric}") not in (None, "")
            }
        cells.append(agg.SeedCell(
            rv_ratio=float(row["rv_ratio"]), seed=int(row["seed"]),
            run_name=row["run_name"], source=row["source"],
            present=row["present"].lower() == "true", rv=views["rv"], ms=views["ms"],
            avg_cer=float(row["avg_cer"]),
        ))
    return cells


def write_table(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def regenerate_tables(results: Path, out: Path) -> None:
    main_cells = read_cells(results / "paclic-v2-maingrid/tables/per_seed.csv")
    dest = out / "maingrid/tables"
    for name, content in {
        "per_ratio.csv": agg.render_per_ratio_csv(agg.aggregate_by_ratio(main_cells)),
        "asymmetry.csv": agg.render_asymmetry_csv(agg.asymmetry(main_cells)),
        "retention_adaptation.csv": agg.render_retention_csv(agg.retention_adaptation(main_cells)),
    }.items():
        write_table(dest / name, content)
    curve = {
        total: read_cells(results / f"paclic-v2-data-efficiency/tables/per_seed_n{total}.csv")
        for total in efficiency.TOTALS
    }
    dest = out / "data-efficiency/tables"
    for name, content in {
        "curve.csv": efficiency.render_curve_csv(efficiency.aggregate_curve(curve)),
        "curve_deltas.csv": efficiency.render_delta_csv(efficiency.curve_deltas(curve)),
        "step_separability.csv": efficiency.render_step_separability_csv(efficiency.step_separability(curve)),
    }.items():
        write_table(dest / name, content)
    # Aggregate count snapshots suffice for summary tables; new alignments require
    # the predictions saved by training and the decomposition CLI.
    payload = json.loads((results / "paclic-v2-accent-error-decomposition/aggregate.json").read_text())
    dest = out / "accent-error-decomposition/tables"
    for name, content in {
        "by_ratio_view.csv": decomposition.render_by_ratio_view_csv(payload["per_ratio_view"]),
        "mark_budget.csv": decomposition.render_mark_budget_csv(payload["mark_budget_per_ratio_view"]),
        "misplacement.csv": decomposition.render_misplacement_csv(payload["misplacement_per_ratio_view"]),
        "trained_vs_untrained.csv": decomposition.render_contrast_csv(payload["trained_vs_untrained"]),
    }.items():
        write_table(dest / name, content)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/paper")
    args = parser.parse_args(argv)
    regenerate_tables(args.results_dir, args.out_dir)
    main_dir = args.results_dir / "paclic-v2-maingrid"
    de_dir = args.results_dir / "paclic-v2-data-efficiency"
    selection_dir = args.results_dir / "paclic-v2-checkpoint-selection-sensitivity"
    for source, renderer, dest in (
        (main_dir, main_grid.build_figures, "maingrid"),
        (de_dir, efficiency.build_figures, "data-efficiency"),
    ):
        payload = json.loads((source / "aggregate.json").read_text())
        figs = renderer(payload, args.out_dir / "figures" / dest, bundle_dir=str(source))
        print(f"{dest}: {len(figs)} figures (PNG and PDF)")
    cells, aggregate = effects.analyze(main_dir, selection_dir)
    if aggregate["hard_warnings"]:
        raise RuntimeError(aggregate["hard_warnings"])
    effects.build_bundle(cells, aggregate, args.out_dir / "effect-sizes-and-plateau")
    print(f"Tables and figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
