"""Test experiment accent ratio subset manifest: PACLIC 2026 v2 release."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.training.trocr.dataset import (
    OCRDataset,
    create_parallel_mixed_dataset,
    hash_key_list,
)
from scripts.training.trocr.experiment_accent_ratio import (
    build_subset_metadata,
    write_subset_manifest,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


def _ocr_dataset(texts: list[str]) -> OCRDataset:
    df = pd.DataFrame(
        {"file_name": [f"{i}.png" for i in range(len(texts))], "text": texts}
    )
    return OCRDataset(
        root_dir="/does-not-exist",
        df=df,
        processor=_FakeProcessor(),
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
    )


def _mixed(rv_ratio=0.75, total=1042, seed=43, subset_seed=42):
    texts = [f"t{i}" for i in range(2000)]
    a, b = _ocr_dataset(texts), _ocr_dataset(texts)
    return create_parallel_mixed_dataset(
        a, b, ratio_a=rv_ratio, total_samples=total, seed=seed, subset_seed=subset_seed
    )


def test_metadata_keeps_seed_and_subset_seed_distinct():
    mixed = _mixed(seed=43, subset_seed=42)
    meta = build_subset_metadata(
        mixed,
        experiment_name="rv_75_ms_25_n1042_seed_43",
        rv_ratio=0.75,
        ms_ratio=0.25,
        seed=43,
        parallel_subset_seed_explicit=42,
        total_train_samples_requested=1042,
    )
    # The whole point of study: the Trainer seed and the data-selection seed are
    # separate axes and must never be conflated in the record.
    assert meta["seed"] == 43
    assert meta["parallel_subset_seed"] == 42
    assert meta["parallel_subset_seed_explicit"] == 42
    assert meta["seed"] != meta["parallel_subset_seed"]


def test_metadata_records_counts_hash_and_routing():
    mixed = _mixed(rv_ratio=0.75, total=1042, subset_seed=42)
    meta = build_subset_metadata(
        mixed,
        experiment_name="rv_75_ms_25_n1042_seed_43",
        rv_ratio=0.75,
        ms_ratio=0.25,
        seed=43,
        parallel_subset_seed_explicit=42,
        total_train_samples_requested=1042,
    )
    assert meta["parallel_keys_used"] == 1042
    assert meta["parallel_keys_total"] == 2000
    # int(1042*0.75) = 781 RV, 261 MS.
    assert (meta["n_a"], meta["n_b"]) == (781, 261)
    assert meta["rv_train_samples"] == 781 and meta["ms_train_samples"] == 261
    assert meta["keys_a_count"] == 781 and meta["keys_b_count"] == 261
    assert meta["key_list_sha256"] == mixed.key_list_sha256
    assert meta["keys_a_sha256"] == hash_key_list(mixed.keys_a)
    assert meta["keys_b_sha256"] == hash_key_list(mixed.keys_b)
    assert meta["mix_mode"] == "parallel_by_text"
    assert meta["key_column"] == "text"
    assert meta["selection_algorithm"] == "sorted_common_keys+seeded_shuffle+prefix"


def test_metadata_explicit_none_falls_back_to_seed():
    # No explicit parallel_subset_seed ⇒ mixer selected with seed; the record shows
    # explicit=None but the resolved parallel_subset_seed == seed.
    texts = [f"t{i}" for i in range(2000)]
    a, b = _ocr_dataset(texts), _ocr_dataset(texts)
    mixed = create_parallel_mixed_dataset(
        a, b, ratio_a=0.5, total_samples=1042, seed=44
    )
    meta = build_subset_metadata(
        mixed,
        experiment_name="rv_50_ms_50_n1042_seed_44",
        rv_ratio=0.5,
        ms_ratio=0.5,
        seed=44,
        parallel_subset_seed_explicit=None,
        total_train_samples_requested=1042,
    )
    assert meta["parallel_subset_seed_explicit"] is None
    assert meta["parallel_subset_seed"] == 44
    assert meta["seed"] == 44


def test_write_subset_manifest_creates_both_files(tmp_path: Path):
    mixed = _mixed(rv_ratio=0.75, total=1042, seed=43, subset_seed=42)
    meta = build_subset_metadata(
        mixed,
        experiment_name="rv_75_ms_25_n1042_seed_43",
        rv_ratio=0.75,
        ms_ratio=0.25,
        seed=43,
        parallel_subset_seed_explicit=42,
        total_train_samples_requested=1042,
    )
    out = tmp_path / "rv_75_ms_25_n1042_seed_43"
    write_subset_manifest(out, mixed, meta)

    manifest_path = out / "subset_manifest.json"
    keys_path = out / "parallel_keys.txt"
    assert manifest_path.is_file() and keys_path.is_file()

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["seed"] == 43 and loaded["parallel_subset_seed"] == 42
    assert loaded["key_list_sha256"] == mixed.key_list_sha256

    # parallel_keys.txt is the full ordered (ratio-invariant) selection, and its
    # body (sans trailing newline) hashes back to key_list_sha256.
    lines = keys_path.read_text(encoding="utf-8").splitlines()
    assert lines == mixed.selected_parallel_keys
    assert len(lines) == 1042
    assert hash_key_list(lines) == mixed.key_list_sha256


def test_parallel_keys_txt_is_ratio_invariant(tmp_path: Path):
    """The persisted key list is identical across ratios (only routing differs)."""
    texts = [f"t{i}" for i in range(2000)]
    a, b = _ocr_dataset(texts), _ocr_dataset(texts)
    written = []
    for rv_ratio in (1.0, 0.5, 0.0):
        mixed = create_parallel_mixed_dataset(
            a, b, ratio_a=rv_ratio, total_samples=1042, subset_seed=42
        )
        meta = build_subset_metadata(
            mixed,
            experiment_name=f"rv_{int(rv_ratio * 100)}",
            rv_ratio=rv_ratio,
            ms_ratio=1.0 - rv_ratio,
            seed=42,
            parallel_subset_seed_explicit=42,
            total_train_samples_requested=1042,
        )
        out = tmp_path / f"r{rv_ratio}"
        write_subset_manifest(out, mixed, meta)
        written.append((out / "parallel_keys.txt").read_text(encoding="utf-8"))
    assert len(set(written)) == 1
