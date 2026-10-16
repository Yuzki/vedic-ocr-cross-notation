# Transfer Between Accent Notations in Vedic OCR

Code and small numerical result tables for the PACLIC 2026 camera-ready paper,
**Transfer Between Accent Notations in Vedic OCR: Learning Visual Conventions from Parallel Text**.
The study fine-tunes one Sanskrit/ISO preadapted TrOCR initialization on paired
synthetic RV/MS lines, varying the notation mix and training-set size.

## Install

Use Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run pytest -q
```

Direct dependencies match the source lock exactly, including PyTorch 2.9.1,
Transformers 4.57.3, datasets 4.4.1, and the pinned SanskritTransliteration Git
revision. The lock retains the source CUDA 12.8 wheels. CPU tests do not require a
GPU; for an explicit CPU run, use `CUDA_VISIBLE_DEVICES='' uv run pytest -q`.
The `evaluate` CER/WER/chrF plugins download on first use; subsequent runs can use
the local cache. Training and inference also need the Hub artifacts below.

## Data

The default is `data.source=hf`, loading
[`yzk/veda-ocr-rv-ms-noto-v2`](https://huggingface.co/datasets/yzk/veda-ocr-rv-ms-noto-v2)
at **`7225f9adb86af6d63316b1e5a60b2b8de46e320c`**. The six splits are
`rv_train`, `rv_val`, `rv_test`, `ms_train`, `ms_val`, and `ms_test`.
Each notation has 10,420 training lines and 578 validation/test lines each.
RV/MS share parallel ISO text; the mix routes each selected text to one image.

The romanized texts are the Maitrāyaṇī Saṃhitā texts released with
[`yzk/veda-ocr-ms`](https://huggingface.co/datasets/yzk/veda-ocr-ms)
(Tsukagoshi et al., 2025); the line splits come from the upstream rendering
pipeline that produced the original RV/MS CSVs.
`scripts/data/build_v2_corpus.py` implements the v2 text repairs and rendering;
`scripts/data/vedic_accent_marks.py` implements the accent placement rules.
To rebuild images, supply the upstream `file_name,text` CSVs at
`<upstream-root>/data/datasets/{rv,ms}/{train,val,test}_dataset.csv` and place
**NotoSansDevanagari-Regular.ttf**, version 2.006, at
`data/resources/fonts/NotoSansDevanagari-Regular.ttf`.
Its SHA256 must be
`084a94d89eb54aafb93a056e15425c34fd859f6342875165d304837b3bcfc2d2`.
Obtain the font from the [Noto Devanagari project](https://github.com/notofonts/devanagari)
under SIL OFL 1.1; the binary is not bundled.

```bash
uv run python -m scripts.data.build_v2_corpus \
  --source-root /path/to/upstream-root --out-root . --workers 8
```

Rendering uses 32 px type, grayscale images, ink crop and 20 px padding, with no
noise layer. The Hub dataset is the recommended reproduction input: rebuilding
also requires the original upstream split files and exact font. The corpus is
synthetic; MS underlines use a substitute Unicode glyph and RV numeral insertion
is deliberately omitted. See rule descriptions in `vedic_accent_marks.py`.

## Models

Initialization:
[`yzk/trocr-large-printed-sanskrit-iso-preadapt`](https://huggingface.co/yzk/trocr-large-printed-sanskrit-iso-preadapt)
at **`6dc79fc60cb0f8149b234478b46062c7b8ac732f`**.
The 15 trained main-grid models (5 RV/MS shares × 3 seeds) are in
[`yzk/trocr-vedic-cross-notation-rv-ms`](https://huggingface.co/yzk/trocr-vedic-cross-notation-rv-ms),
one subfolder per condition (e.g. `rv_50_ms_50/seed_42`). Weights are not bundled here. Both Hub revisions are defaults in the Hydra configuration
and are recorded in training results.

## Reproduce training

A main-grid cell uses 10,420 parallel texts, 20 maximum epochs, seeds 42/43/44,
full fine-tuning, learning rate 2e-5, weight decay 0.01, warmup 0.1, a linear
schedule, FP16, train/eval batch sizes 16 per device, and early stopping patience 3.
Checkpoint selection minimizes validation CER on the majority notation; 50:50
uses average validation CER. Training consumes ISO directly with augmentation off.
Generation uses 4 beams, length penalty 2, no-repeat 3-grams and maximum length 256,
preserving the checkpoint's decoder start token. The original no-shift loss is retained.

The published recipe used **two visible GPUs**, with Trainer DataParallel and an
effective training batch of 32 (326 optimizer steps per full-data epoch).
The launcher checks for exactly two GPUs. W&B logging is disabled by default.

```bash
# CPU-only inspection: 50:50, seed 42; no downloads or model allocation.
uv run python -m scripts.reproduce maingrid --cell 6 --resolve

# Train that cell; outputs and its log are written below outputs/.
CUDA_VISIBLE_DEVICES=0,1 uv run python -m scripts.reproduce maingrid --cell 6

# All 15 main-grid cells, run sequentially.
CUDA_VISIBLE_DEVICES=0,1 uv run python -m scripts.reproduce maingrid --all

# All 27 low-data cells, run sequentially.
CUDA_VISIBLE_DEVICES=0,1 uv run python -m scripts.reproduce data-efficiency --all
```

Main-grid indexing is ratio-major over `{1.0,0.75,0.5,0.25,0.0}` × `{42,43,44}`.
Low-data indexing is total-major over `{1042,2605,5210}` × `{1.0,0.5,0.0}` ×
`{42,43,44}`. The 100% curve points reuse main-grid runs.
`parallel_subset_seed=null` follows each training seed, preserving the nested
subset ladder. `--dry-run` prints the exact overrides for any cell or grid.

## Evaluate

Training evaluates both held-out test views and saves `results.json`,
`rv_predictions.csv`, `ms_predictions.csv` and `final_model/` under each run.
To evaluate the initialization independently:

```bash
uv run python -m scripts.evaluation.evaluate_initial_checkpoint \
  --out-dir outputs/evaluation/initial

# Evaluate a trained cell, preserving its saved generation configuration.
uv run python -m scripts.evaluation.evaluate_initial_checkpoint \
  --model-id outputs/experiments_paclic_v2_maingrid/rv_50_ms_50_seed_42/final_model \
  --fine-tuned --out-dir outputs/evaluation/rv50-seed42
```

These use the pinned dataset by default. Metrics include CER, WER, chrF, exact
match, alignment-based accent precision/recall/F1 and accent-stripped CER.
Normalization is NFC and strip, without whitespace collapse. Empty predictions
count as deletions. Accent metrics use the shared authoritative implementation.

## Regenerate tables and figures without GPUs

The `results/` directory contains five small bundles of CSV/JSON summaries, under
5 MB total, with no raw line predictions or images. Numerical measurements are
unchanged; infrastructure paths and private operational labels have been removed.

```bash
CUDA_VISIBLE_DEVICES='' uv run python -m scripts.regenerate
```

This writes regenerated main-grid/curve/error-summary tables, effect sizes and
plateau statistics, and three figures as PNG/PDF to `outputs/paper/`.
The figures are `fig1_per_ratio_performance`, `fig4_retention_adaptation`, and
`fig6_data_efficiency`. All regenerated CSV cells match the released tables.
Statistics use sample SD and Student-t 95% intervals; paired contrasts preserve
seed pairing. At n=3 the exact two-sided sign-flip test has a p-value floor of 0.25.

To aggregate newly trained runs, use the included v2 builders:

```bash
uv run python -m scripts.evaluation.build_paclic_v2_maingrid_analysis \
  --study-root outputs/experiments_paclic_v2_maingrid \
  --slurm-log-dir outputs/logs --out-dir outputs/analysis/maingrid \
  --figures-out-dir outputs/figures/maingrid

uv run python -m scripts.evaluation.build_paclic_v2_data_efficiency_analysis \
  --low-study-root outputs/experiments_paclic_v2_data_efficiency \
  --full-study-root outputs/experiments_paclic_v2_maingrid \
  --slurm-log-dir outputs/logs \
  --maingrid-aggregate outputs/analysis/maingrid/aggregate.json \
  --out-dir outputs/analysis/data-efficiency \
  --figures-out-dir outputs/figures/data-efficiency

uv run python -m scripts.evaluation.build_paclic_v2_accent_error_decomposition \
  --study-root outputs/experiments_paclic_v2_maingrid \
  --slurm-log-dir outputs/logs --out-dir outputs/analysis/accent-errors
```

The historical `--slurm-log-dir` argument now accepts the portable launcher's local
logs; Slurm is not required. Figure gates retain completeness and corpus checks,
including the exact Hub pin for new runs. Raw predictions and training trajectories
are needed to redo alignment diagnostics or checkpoint-selection sensitivity;
released summaries suffice for the paper figures and numerical summary tables.
The checkpoint-sensitivity module is included, but missing alternate weights
cannot be recovered from summaries.

## Citation

```bibtex
@inproceedings{TODO-paclic2026,
  title     = {Transfer Between Accent Notations in Vedic OCR: Learning Visual Conventions from Parallel Text},
  author    = {TODO},
  booktitle = {Proceedings of the 40th Pacific Asia Conference on Language, Information and Computation (PACLIC 2026)},
  year      = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE).
The models and the dataset are released under CC BY-NC-SA 4.0; see their Hub
cards for training-data provenance and attribution.
