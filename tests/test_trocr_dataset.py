"""Test trocr dataset: PACLIC 2026 v2 release."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from scripts.training.trocr import experiment_accent_ratio
from scripts.training.trocr.dataset import (
    MixedDataset,
    OCRDataset,
    create_parallel_mixed_dataset,
    hash_key_list,
)
from scripts.training.trocr.experiment_accent_ratio import build_train_transform


# ---------- fakes ----------


class _FakeTokenizerOutput:
    """Stand-in for the ``BatchEncoding`` returned by HF tokenizers."""

    def __init__(self, ids: list[int], pad_id: int, max_length: int) -> None:
        padded = ids + [pad_id] * (max_length - len(ids))
        # Tokenizers return ``(1, max_length)`` tensors when ``return_tensors="pt"``.
        self.input_ids = torch.tensor([padded])
        self.attention_mask = torch.tensor(
            [[1] * len(ids) + [0] * (max_length - len(ids))]
        )


class _FakeTokenizer:
    """Minimal tokenizer that returns deterministic ids for any input string."""

    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(
        self,
        text,
        padding=None,
        max_length: int = 8,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> _FakeTokenizerOutput:
        # Two non-pad tokens followed by padding — exact ids don't matter.
        return _FakeTokenizerOutput(
            ids=[1, 2], pad_id=self.pad_token_id, max_length=max_length
        )


class _FakeProcessorResult:
    def __init__(self, pixel_values) -> None:
        self.pixel_values = pixel_values


class _FakeProcessor:
    """Records the image it received and returns a fixed-shape pixel tensor."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.last_image: Image.Image | None = None
        self.call_count = 0

    def __call__(self, image: Image.Image, return_tensors: str = "pt"):
        self.last_image = image
        self.call_count += 1
        # Shape ``(1, 3, 4, 4)`` — squeeze(0) yields ``(3, 4, 4)``.
        return _FakeProcessorResult(torch.zeros((1, 3, 4, 4)))


# ---------- fixtures ----------


def _write_dummy_image(
    path: Path, color: tuple[int, int, int] = (255, 255, 255)
) -> None:
    Image.new("RGB", (16, 8), color=color).save(path)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> tuple[Path, pd.DataFrame, _FakeProcessor]:
    """Create a 1-row dataset on disk so ``__getitem__`` can exercise the full path."""
    image_name = "img.png"
    _write_dummy_image(tmp_path / image_name)
    df = pd.DataFrame({"file_name": [image_name], "text": ["agnim"]})
    return tmp_path, df, _FakeProcessor()


# ---------- OCRDataset.image_transform contract ----------


def test_default_image_transform_is_none(tiny_dataset):
    root, df, processor = tiny_dataset
    ds = OCRDataset(
        root_dir=root,
        df=df,
        processor=processor,
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
    )
    # Backwards-compatible default: legacy callers don't have to pass anything.
    assert ds.image_transform is None


def test_legacy_path_unchanged_when_transform_is_none(tiny_dataset):
    root, df, processor = tiny_dataset
    ds = OCRDataset(
        root_dir=root,
        df=df,
        processor=processor,
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
        image_transform=None,
    )
    sample = ds[0]
    # Tensor shape / dtype / key structure must match the pre-augmentation contract.
    assert set(sample) == {"pixel_values", "labels", "attention_mask"}
    assert sample["pixel_values"].shape == (3, 4, 4)
    assert sample["labels"].shape == (8,)
    assert sample["attention_mask"].shape == (8,)
    assert processor.call_count == 1
    # The processor must have seen the freshly-decoded RGB image, not a transformed one.
    assert isinstance(processor.last_image, Image.Image)
    assert processor.last_image.mode == "RGB"


def test_image_transform_is_applied_before_processor(tiny_dataset):
    root, df, processor = tiny_dataset
    sentinel = Image.new("RGB", (16, 8), color=(123, 45, 67))
    calls = {"n": 0, "received_rgb": False}

    def deterministic_transform(image: Image.Image) -> Image.Image:
        calls["n"] += 1
        # The hook must run after the ``.convert("RGB")`` step.
        calls["received_rgb"] = isinstance(image, Image.Image) and image.mode == "RGB"
        return sentinel

    ds = OCRDataset(
        root_dir=root,
        df=df,
        processor=processor,
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
        image_transform=deterministic_transform,
    )

    _ = ds[0]
    _ = ds[0]

    # Hook ran on every ``__getitem__`` invocation, before the processor saw the image.
    assert calls["n"] == 2
    assert calls["received_rgb"] is True
    assert processor.call_count == 2
    assert processor.last_image is sentinel


# ---------- build_train_transform ----------


def test_build_train_transform_returns_none_when_section_missing():
    cfg = OmegaConf.create({})
    assert build_train_transform(cfg) is None


def test_build_train_transform_returns_none_when_disabled():
    cfg = OmegaConf.create({"augmentation": {"enabled": False}})
    assert build_train_transform(cfg) is None


def test_build_train_transform_returns_callable_when_enabled():
    cfg = OmegaConf.create(
        {
            "augmentation": {
                "enabled": True,
                "affine_degrees": 2,
                "affine_translate": 0.02,
                "affine_scale_min": 0.95,
                "affine_scale_max": 1.02,
                "color_jitter_brightness": 0.15,
                "color_jitter_contrast": 0.15,
                "blur_kernel": 3,
                "blur_sigma_min": 0.1,
                "blur_sigma_max": 1.0,
            }
        }
    )
    transform = build_train_transform(cfg)
    assert callable(transform)

    # Apply once to make sure the pipeline accepts a PIL image and returns one.
    out = transform(Image.new("RGB", (32, 16), color=(255, 255, 255)))
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    # Output dimensions are preserved (affine fills with white, no crop/resize).
    assert out.size == (32, 16)


def test_build_train_transform_uses_overrides(monkeypatch):
    """The factory must read every documented knob — not silently fall back to defaults."""
    captured: dict = {}

    class _RecordingAffine:
        def __init__(self, **kwargs):
            captured["affine"] = kwargs

        def __call__(self, image):
            return image

    class _RecordingColorJitter:
        def __init__(self, **kwargs):
            captured["color_jitter"] = kwargs

        def __call__(self, image):
            return image

    class _RecordingBlur:
        def __init__(self, **kwargs):
            captured["blur"] = kwargs

        def __call__(self, image):
            return image

    # ``experiment_accent_ratio.transforms`` is the same module object as
    # ``torchvision.transforms`` (shared reference via ``from torchvision import``),
    # so patching attributes here is what ``build_train_transform`` will see.
    monkeypatch.setattr(
        experiment_accent_ratio.transforms, "RandomAffine", _RecordingAffine
    )
    monkeypatch.setattr(
        experiment_accent_ratio.transforms, "ColorJitter", _RecordingColorJitter
    )
    monkeypatch.setattr(
        experiment_accent_ratio.transforms, "GaussianBlur", _RecordingBlur
    )

    cfg = OmegaConf.create(
        {
            "augmentation": {
                "enabled": True,
                "affine_degrees": 5,
                "affine_translate": 0.03,
                "affine_scale_min": 0.9,
                "affine_scale_max": 1.05,
                "color_jitter_brightness": 0.2,
                "color_jitter_contrast": 0.25,
                "blur_kernel": 5,
                "blur_sigma_min": 0.2,
                "blur_sigma_max": 1.5,
            }
        }
    )
    build_train_transform(cfg)

    assert captured["affine"]["degrees"] == 5
    assert captured["affine"]["translate"] == (0.03, 0.03)
    assert captured["affine"]["scale"] == (0.9, 1.05)
    assert captured["affine"]["fill"] == 255
    assert captured["color_jitter"]["brightness"] == 0.2
    assert captured["color_jitter"]["contrast"] == 0.25
    assert captured["blur"]["kernel_size"] == 5
    assert captured["blur"]["sigma"] == (0.2, 1.5)


# ---------- pre-flight validation ----------


@pytest.mark.parametrize(
    "override, expected_field",
    [
        ({"blur_kernel": 4}, "augmentation.blur_kernel"),
        ({"blur_kernel": 0}, "augmentation.blur_kernel"),
        ({"blur_kernel": -1}, "augmentation.blur_kernel"),
        (
            {"affine_scale_min": 1.1, "affine_scale_max": 0.9},
            "augmentation.affine_scale_min",
        ),
        (
            {"blur_sigma_min": 1.5, "blur_sigma_max": 0.2},
            "augmentation.blur_sigma_min",
        ),
        ({"affine_translate": 1.5}, "augmentation.affine_translate"),
        ({"affine_translate": -0.1}, "augmentation.affine_translate"),
    ],
)
def test_build_train_transform_rejects_invalid_knob(override, expected_field):
    """Each user-facing knob with a known torchvision failure mode must raise
    ``ValueError`` *before* we reach ``transforms.Compose`` — and the message
    must name the offending YAML field so the user can fix the config without
    digging through a stack trace."""
    cfg = OmegaConf.create({"augmentation": {"enabled": True, **override}})
    with pytest.raises(ValueError) as exc_info:
        build_train_transform(cfg)
    assert expected_field in str(exc_info.value)


def test_build_train_transform_batches_multiple_errors():
    """A misconfigured yaml should surface *all* problems at once, not one per
    re-run, so the user can fix the config in a single round-trip."""
    cfg = OmegaConf.create(
        {
            "augmentation": {
                "enabled": True,
                "blur_kernel": 4,
                "affine_scale_min": 1.1,
                "affine_scale_max": 0.9,
                "blur_sigma_min": 1.5,
                "blur_sigma_max": 0.2,
                "affine_translate": 1.5,
            }
        }
    )
    with pytest.raises(ValueError) as exc_info:
        build_train_transform(cfg)
    msg = str(exc_info.value)
    assert "augmentation.blur_kernel" in msg
    assert "augmentation.affine_scale_min" in msg
    assert "augmentation.blur_sigma_min" in msg
    assert "augmentation.affine_translate" in msg


def test_build_train_transform_accepts_equal_min_max():
    """``min == max`` is a constant range and torchvision accepts it; treat it
    as valid so users can pin a fixed scale/sigma without a workaround."""
    cfg = OmegaConf.create(
        {
            "augmentation": {
                "enabled": True,
                "affine_scale_min": 1.0,
                "affine_scale_max": 1.0,
                "blur_sigma_min": 0.5,
                "blur_sigma_max": 0.5,
            }
        }
    )
    transform = build_train_transform(cfg)
    assert callable(transform)


# ---------- end-to-end: real torchvision pipeline applied through OCRDataset ----------


def test_real_torchvision_pipeline_runs_through_dataset(tmp_path: Path):
    """Sanity check: the actual augmentation pipeline + ``OCRDataset`` co-operate."""
    image_name = "img.png"
    # Use a non-uniform image so blur / affine produce visibly different pixel arrays.
    arr = np.tile(np.arange(32, dtype=np.uint8), (16, 3, 1)).transpose(0, 2, 1)
    Image.fromarray(arr, mode="RGB").save(tmp_path / image_name)
    df = pd.DataFrame({"file_name": [image_name], "text": ["agnim"]})

    cfg = OmegaConf.create({"augmentation": {"enabled": True}})
    transform = build_train_transform(cfg)
    assert transform is not None

    processor = _FakeProcessor()
    ds = OCRDataset(
        root_dir=tmp_path,
        df=df,
        processor=processor,
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
        image_transform=transform,
    )

    sample = ds[0]
    assert sample["pixel_values"].shape == (3, 4, 4)
    # Processor must have received a PIL image (the augmentation output).
    assert isinstance(processor.last_image, Image.Image)
    assert processor.last_image.mode == "RGB"


# ---------- guard rails ----------


def test_image_transform_module_namespace():
    # Smoke check: the public API surface stays minimal.
    import scripts.training.trocr.experiment_accent_ratio as exp

    assert callable(exp.build_train_transform)
    # Avoid accidental name collisions with the existing module-level helpers.
    assert isinstance(exp.build_train_transform, types.FunctionType)


def test_parallel_mix_is_exported_at_package_level():
    # Matches the existing package-level export of ``create_mixed_dataset`` so the
    # parallel sibling is importable the same way.
    import scripts.training.trocr as pkg

    assert pkg.create_parallel_mixed_dataset is create_parallel_mixed_dataset


# ---------- create_parallel_mixed_dataset  ----------
#
# RV/MS are parallel corpora: identical text labels, only the image differs.
# create_parallel_mixed_dataset partitions the *shared* texts by ratio so the
# mix has a fixed size across the sweep and no text appears as both an RV and an
# MS image. These build real OCRDataset objects (the fake processor means no
# disk/tokenizer work) and only inspect ``.df`` / length / the Subset indices —
# __getitem__ is never called.


def _ocr_dataset(texts: list[str], file_names: list[str] | None = None) -> OCRDataset:
    fnames = file_names or [f"{idx}.png" for idx in range(len(texts))]
    df = pd.DataFrame({"file_name": fnames, "text": texts})
    return OCRDataset(
        root_dir="/does-not-exist",
        df=df,
        processor=_FakeProcessor(),
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
    )


def _texts_in_mix(mixed) -> list[str]:
    """Recover the label of every row actually selected into the mix."""
    out: list[str] = []
    for subset in mixed.datasets:  # ConcatDataset.datasets -> list[Subset]
        df = subset.dataset.df  # Subset.dataset -> OCRDataset
        for idx in subset.indices:  # Subset.indices -> positional row indices
            out.append(df.iloc[idx]["text"])
    return out


def test_parallel_mix_none_uses_common_key_count():
    # total_samples=None ⇒ all 4 shared texts, not len(a)+len(b)=8.
    a = _ocr_dataset(["t1", "t2", "t3", "t4"])
    b = _ocr_dataset(["t1", "t2", "t3", "t4"])
    mixed = create_parallel_mixed_dataset(a, b, ratio_a=0.5)
    assert len(mixed) == 4
    assert mixed.n_a == 2
    assert mixed.n_b == 2
    assert mixed.parallel_keys_total == 4
    assert mixed.parallel_keys_used == 4
    assert mixed.mix_mode == "parallel_by_text"


def test_parallel_mix_no_text_from_both_sides():
    a = _ocr_dataset(["t1", "t2", "t3", "t4"])
    b = _ocr_dataset(["t1", "t2", "t3", "t4"])
    mixed = create_parallel_mixed_dataset(a, b, ratio_a=0.5)
    texts = _texts_in_mix(mixed)
    # Every selected text is unique: none enters as both an RV and an MS image.
    assert len(texts) == 4
    assert len(set(texts)) == 4


@pytest.mark.parametrize("ratio", [1.0, 0.5, 0.0])
def test_parallel_mix_extremes_keep_same_n(ratio):
    a = _ocr_dataset(["t1", "t2", "t3", "t4"])
    b = _ocr_dataset(["t1", "t2", "t3", "t4"])
    mixed = create_parallel_mixed_dataset(a, b, ratio_a=ratio)
    assert len(mixed) == 4


def test_parallel_mix_extreme_ratios_route_all_to_one_side():
    a = _ocr_dataset(["t1", "t2", "t3", "t4"])
    b = _ocr_dataset(["t1", "t2", "t3", "t4"])
    all_a = create_parallel_mixed_dataset(a, b, ratio_a=1.0)
    assert (all_a.n_a, all_a.n_b) == (4, 0)
    all_b = create_parallel_mixed_dataset(a, b, ratio_a=0.0)
    assert (all_b.n_a, all_b.n_b) == (0, 4)


def test_parallel_mix_selected_text_set_is_ratio_invariant():
    # For a fixed (seed, total_samples) the *selected* text set must be identical
    # across ratios — only the RV/MS split of those texts changes.
    a = _ocr_dataset([f"t{i}" for i in range(10)])
    b = _ocr_dataset([f"t{i}" for i in range(10)])
    selected = set()
    for ratio in (1.0, 0.7, 0.5, 0.3, 0.0):
        mixed = create_parallel_mixed_dataset(
            a, b, ratio_a=ratio, total_samples=6, seed=42
        )
        assert len(mixed) == 6
        selected.add(frozenset(_texts_in_mix(mixed)))
    assert len(selected) == 1


def test_parallel_mix_ignores_non_common_keys():
    a = _ocr_dataset(["t1", "t2", "t3", "only_a"])
    b = _ocr_dataset(["t1", "t2", "t3", "only_b"])
    mixed = create_parallel_mixed_dataset(a, b, ratio_a=0.5)
    assert mixed.parallel_keys_total == 3  # only t1, t2, t3 are shared
    texts = set(_texts_in_mix(mixed))
    assert "only_a" not in texts
    assert "only_b" not in texts


def test_parallel_mix_dedups_duplicate_text_within_source():
    # "t1" appears twice in A; it must still surface at most once in the mix.
    a = _ocr_dataset(["t1", "t1", "t2", "t3"])
    b = _ocr_dataset(["t1", "t2", "t3", "t3"])
    mixed = create_parallel_mixed_dataset(a, b, ratio_a=0.5)
    assert mixed.parallel_keys_total == 3
    texts = _texts_in_mix(mixed)
    assert len(texts) == len(set(texts))


def test_parallel_mix_total_samples_exceeding_common_fails():
    a = _ocr_dataset(["t1", "t2"])
    b = _ocr_dataset(["t1", "t2"])
    with pytest.raises(ValueError):
        create_parallel_mixed_dataset(a, b, ratio_a=0.5, total_samples=3)


@pytest.mark.parametrize("total_samples", [-1, 0, 2.0, True])
def test_parallel_mix_invalid_total_samples_fails(total_samples):
    # ``total_train_samples`` is part of the controlled-sweep contract: a bad
    # Hydra override (negative, zero, float, bool) must fail loudly instead of
    # slicing with a junk value and recording inconsistent N/metadata. ``bool``
    # is an ``int`` subclass, so it is rejected explicitly. A float would
    # otherwise slip the size guard and only blow up later at list slicing.
    a = _ocr_dataset(["t1", "t2", "t3"])
    b = _ocr_dataset(["t1", "t2", "t3"])
    with pytest.raises(ValueError):
        create_parallel_mixed_dataset(a, b, ratio_a=0.5, total_samples=total_samples)


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_parallel_mix_invalid_ratio_fails(ratio):
    a = _ocr_dataset(["t1", "t2"])
    b = _ocr_dataset(["t1", "t2"])
    with pytest.raises(ValueError):
        create_parallel_mixed_dataset(a, b, ratio_a=ratio)


def test_parallel_mix_returns_mixeddataset_instance():
    a = _ocr_dataset(["t1", "t2"])
    b = _ocr_dataset(["t1", "t2"])
    mixed = create_parallel_mixed_dataset(a, b, ratio_a=0.5)
    assert isinstance(mixed, MixedDataset)


# ---------- study data-efficiency subset contract ----------
#
# The data-efficiency curve  needs the parallel-subset selection to
# be a *deterministic function of (parallel_subset_seed, total_samples)* and
# decoupled from the Trainer seed, so all five mix ratios at a given total share
# the exact same texts (only the RV/MS routing changes) and the low-data totals
# nest (10% ⊂ 25% ⊂ 50% ⊂ 100%). These tests are the safety net for that
# contract and for the exact per-cell RV/MS routed counts in the plan's table.

#: Exact (n_a, n_b) = (int(total*rv_ratio), total-n_a) for every curve cell,
#: transcribed from the parallel-subset specification. n_a = RV images, n_b = MS images.
EXACT_COUNTS: dict[int, dict[float, tuple[int, int]]] = {
    1042: {
        1.0: (1042, 0),
        0.75: (781, 261),
        0.5: (521, 521),
        0.25: (260, 782),
        0.0: (0, 1042),
    },
    2605: {
        1.0: (2605, 0),
        0.75: (1953, 652),
        0.5: (1302, 1303),
        0.25: (651, 1954),
        0.0: (0, 2605),
    },
    5210: {
        1.0: (5210, 0),
        0.75: (3907, 1303),
        0.5: (2605, 2605),
        0.25: (1302, 3908),
        0.0: (0, 5210),
    },
    10420: {
        1.0: (10420, 0),
        0.75: (7815, 2605),
        0.5: (5210, 5210),
        0.25: (2605, 7815),
        0.0: (0, 10420),
    },
}


@pytest.fixture(scope="module")
def parallel_10420() -> tuple[OCRDataset, OCRDataset]:
    """Two parallel corpora sharing all 10420 texts (the full study curve N).

    Built once for the module: only ``.df`` / ``Subset`` indices are inspected, so
    no image or tokenizer work happens.
    """
    texts = [f"t{i}" for i in range(10420)]
    return _ocr_dataset(texts), _ocr_dataset(texts)


@pytest.mark.parametrize("total", sorted(EXACT_COUNTS))
@pytest.mark.parametrize("rv_ratio", [1.0, 0.75, 0.5, 0.25, 0.0])
def test_parallel_mix_exact_routed_counts_match_plan_table(
    parallel_10420, total, rv_ratio
):
    """Every (total, rv_ratio) cell routes the exact RV/MS counts the plan pins.

    The floor rule ``n_a = int(total * rv_ratio)`` with odd totals is the whole
    reason the table is asserted rather than trusted (e.g. 1042×0.75 → 781/261,
    2605×0.5 → 1302/1303).
    """
    a, b = parallel_10420
    expected = EXACT_COUNTS[total][rv_ratio]
    mixed = create_parallel_mixed_dataset(
        a, b, ratio_a=rv_ratio, total_samples=total, subset_seed=42
    )
    assert (mixed.n_a, mixed.n_b) == expected
    # The routed key lists have exactly those lengths and never overlap.
    assert mixed.keys_a is not None and mixed.keys_b is not None
    assert len(mixed.keys_a) == expected[0]
    assert len(mixed.keys_b) == expected[1]
    assert set(mixed.keys_a).isdisjoint(mixed.keys_b)
    # total_samples is preserved (no silent capping); mix size == total.
    assert mixed.parallel_keys_used == total
    assert len(mixed) == total


def test_parallel_mix_selection_ratio_invariant_and_hash_stable(parallel_10420):
    """For fixed (subset_seed, total) the ordered selection + hash are identical
    across all five ratios; only the RV/MS routing differs."""
    a, b = parallel_10420
    selections = []
    hashes = set()
    routings = set()
    for rv_ratio in (1.0, 0.75, 0.5, 0.25, 0.0):
        m = create_parallel_mixed_dataset(
            a, b, ratio_a=rv_ratio, total_samples=2605, subset_seed=42
        )
        assert m.selected_parallel_keys is not None
        assert m.keys_a is not None and m.keys_b is not None
        selections.append(tuple(m.selected_parallel_keys))
        hashes.add(m.key_list_sha256)
        routings.add((tuple(m.keys_a), tuple(m.keys_b)))
    # Selected set + order + hash are the SAME for every ratio.
    assert len(set(selections)) == 1
    assert len(hashes) == 1
    # ...but the routing genuinely differs across ratios (this is the only knob).
    assert len(routings) == 5
    # The hash is the newline-join digest of the selected list.
    assert next(iter(hashes)) == hash_key_list(selections[0])


def test_parallel_mix_nested_low_data_ladder(parallel_10420):
    """1042 ⊂ 2605 ⊂ 5210 ⊂ 10420 as *ordered prefixes* for a fixed subset seed."""
    a, b = parallel_10420
    sel = {}
    for total in (1042, 2605, 5210, 10420):
        m = create_parallel_mixed_dataset(
            a, b, ratio_a=0.5, total_samples=total, subset_seed=42
        )
        sel[total] = m.selected_parallel_keys
    assert sel[1042] == sel[2605][:1042]
    assert sel[2605] == sel[5210][:2605]
    assert sel[5210] == sel[10420][:5210]
    # Prefix ⇒ set-subset too.
    assert set(sel[1042]) < set(sel[2605]) < set(sel[5210]) < set(sel[10420])


def test_parallel_mix_subset_seed_decoupled_from_trainer_seed(parallel_10420):
    """subset_seed drives the selection; the Trainer ``seed`` does not touch it."""
    a, b = parallel_10420
    m1 = create_parallel_mixed_dataset(
        a, b, ratio_a=0.5, total_samples=1042, seed=1, subset_seed=42
    )
    m2 = create_parallel_mixed_dataset(
        a, b, ratio_a=0.5, total_samples=1042, seed=999, subset_seed=42
    )
    assert m1.selected_parallel_keys == m2.selected_parallel_keys
    assert m1.key_list_sha256 == m2.key_list_sha256
    assert m1.subset_seed == m2.subset_seed == 42


def test_parallel_mix_subset_seed_defaults_to_seed_backward_compat(parallel_10420):
    """Without subset_seed the selection uses ``seed`` — exact previous behaviour."""
    a, b = parallel_10420
    # No subset_seed ⇒ selects with seed=42, identical to an explicit subset_seed=42.
    legacy = create_parallel_mixed_dataset(
        a, b, ratio_a=0.5, total_samples=1042, seed=42
    )
    explicit = create_parallel_mixed_dataset(
        a, b, ratio_a=0.5, total_samples=1042, seed=42, subset_seed=42
    )
    assert legacy.subset_seed == 42
    assert legacy.selected_parallel_keys == explicit.selected_parallel_keys
    assert legacy.key_list_sha256 == explicit.key_list_sha256
    # A different seed (no subset_seed) selects a genuinely different subset.
    other = create_parallel_mixed_dataset(a, b, ratio_a=0.5, total_samples=1042, seed=7)
    assert other.subset_seed == 7
    assert other.key_list_sha256 != legacy.key_list_sha256


def test_parallel_mix_metadata_fields_populated(parallel_10420):
    """The study provenance fields are all populated on the parallel path."""
    a, b = parallel_10420
    m = create_parallel_mixed_dataset(
        a, b, ratio_a=0.75, total_samples=1042, subset_seed=42
    )
    assert m.key_column == "text"
    assert m.selection_algorithm == "sorted_common_keys+seeded_shuffle+prefix"
    assert m.subset_seed == 42
    assert m.key_list_sha256 and len(m.key_list_sha256) == 64
    # keys_a/keys_b are the ratio's prefix split of the selection.
    assert m.selected_parallel_keys is not None
    assert m.keys_a == m.selected_parallel_keys[: m.n_a]
    assert m.keys_b == m.selected_parallel_keys[m.n_a :]


def test_hash_key_list_is_order_sensitive_and_stable():
    assert hash_key_list(["a", "b", "c"]) == hash_key_list(["a", "b", "c"])
    assert hash_key_list(["a", "b", "c"]) != hash_key_list(["c", "b", "a"])
    assert hash_key_list([]) == hash_key_list([])
