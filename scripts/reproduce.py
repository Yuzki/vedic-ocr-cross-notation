"""Portable PACLIC v2 recipes, transcribed from the two source launchers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = "yzk/trocr-large-printed-sanskrit-iso-preadapt"
MODEL_REVISION = "6dc79fc60cb0f8149b234478b46062c7b8ac732f"
DATASET = "yzk/veda-ocr-rv-ms-noto-v2"
DATASET_REVISION = "7225f9adb86af6d63316b1e5a60b2b8de46e320c"
MAIN_RATIOS = (1.0, 0.75, 0.5, 0.25, 0.0)
LOW_RATIOS = (1.0, 0.5, 0.0)
SEEDS = (42, 43, 44)
LOW_TOTALS = (1042, 2605, 5210)


def cells(study: str):
    totals = (10420,) if study == "maingrid" else LOW_TOTALS
    ratios = MAIN_RATIOS if study == "maingrid" else LOW_RATIOS
    return [(n, r, s) for n in totals for r in ratios for s in SEEDS]


def overrides(study: str, cell: int) -> list[str]:
    n, ratio, seed = cells(study)[cell]
    suffix = f"seed_{seed}" if study == "maingrid" else f"n{n}_seed_{seed}"
    tags = "[paclic,corpus-v2,accent-marks-corrected,v2-maingrid]" if study == "maingrid" else f"[paclic,corpus-v2,accent-marks-corrected,v2-data-efficiency,n{n}]"
    return [
        f"model.base={MODEL}", f"model.revision={MODEL_REVISION}",
        f"rv_ratio={ratio}", "model.freeze=none", f"seed={seed}",
        f"run_suffix={suffix}", f"total_train_samples={n}",
        "params.num_train_epochs=20", "wandb_mode=disabled", f"wandb.tags={tags}",
        f"output_dir=./outputs/experiments_paclic_v2_{study if study == 'maingrid' else 'data_efficiency'}",
        "data.source=hf", f"data.hf.dataset_id={DATASET}",
        f"data.hf.revision={DATASET_REVISION}",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("maingrid", "data-efficiency"))
    parser.add_argument("--cell", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Run every cell sequentially.")
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without imports or downloads.")
    parser.add_argument("--resolve", action="store_true", help="Resolve Hydra config on CPU; no data/model download.")
    args = parser.parse_args(argv)
    study = args.study
    grid = cells(study)
    if not 0 <= args.cell < len(grid):
        parser.error(f"--cell must be in 0..{len(grid)-1}")
    for index in range(len(grid)) if args.all else [args.cell]:
        opts = overrides(study, index)
        cmd = [sys.executable, "-m", "scripts.training.trocr.experiment_accent_ratio", *opts]
        print(shlex.join(cmd), flush=True)
        if args.resolve:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf
            with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "scripts/training/trocr/conf")):
                cfg = compose(config_name="experiment", overrides=opts)
                print(OmegaConf.to_yaml(cfg, resolve=True))
        elif not args.dry_run:
            import torch
            if torch.cuda.device_count() != 2:
                parser.error("The published recipe requires exactly two visible GPUs (effective batch 32). Set CUDA_VISIBLE_DEVICES accordingly.")
            env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false")
            # Preserve the original log format for the analysis parsers. The Hub pin
            # in results.json is the primary attestation for new release runs.
            n, ratio, seed = grid[index]
            run = f"rv_{int(ratio*100)}_ms_{int((1-ratio)*100)}_" + (f"seed_{seed}" if study == "maingrid" else f"n{n}_seed_{seed}")
            log_dir = ROOT / "outputs/logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            prefix = "paclic_v2_maingrid" if study == "maingrid" else "paclic_v2_data_efficiency"
            with (log_dir / f"{prefix}_0_{index}.out").open("w") as log:
                log.write(f"run_name={run} output_dir={opts[10].split('=', 1)[1]}\n")
                log.write(f"corpus=v2 data_source=hf dataset_id={DATASET} dataset_revision={DATASET_REVISION}\n")
                log.flush()
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
