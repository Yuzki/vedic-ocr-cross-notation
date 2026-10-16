"""Test trocr hf dataset: PACLIC 2026 v2 release."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset as HFDataset
from datasets import Image as HFImage
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from scripts.training.trocr.dataset import (
    HFOCRDataset,
    OCRDataset,
    _coerce_to_pil_rgb,
    create_parallel_mixed_dataset,
    load_ocr_dataset,
    load_ocr_dataset_from_config,
)
from scripts.utils.transliteration import convert_text


# ---------- fakes (content-sensitive so equivalence is a real assertion) ----------


class _FakeTokenizerOutput:
    def __init__(self, ids: list[int], pad_id: int, max_length: int) -> None:
        padded = ids[:max_length] + [pad_id] * max(0, max_length - len(ids))
        self.input_ids = torch.tensor([padded])
        n = min(len(ids), max_length)
        self.attention_mask = torch.tensor([[1] * n + [0] * (max_length - n)])


class _CharTokenizer:
    """Deterministic char-level tokenizer: ids depend on the *text* content so the
    ``input_scheme`` → ``model_scheme`` conversion is observable in the labels."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.last_text: str | None = None

    def __call__(
        self,
        text,
        padding=None,
        max_length: int = 8,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> _FakeTokenizerOutput:
        self.last_text = text
        ids = [(ord(c) % 100) + 1 for c in text]
        return _FakeTokenizerOutput(ids, self.pad_token_id, max_length)


class _FakeProcessorResult:
    def __init__(self, pixel_values) -> None:
        self.pixel_values = pixel_values


class _PixelProcessor:
    """Deterministic processor: pixel_values are a function of the *image* content
    (resized to a fixed 4x4 RGB), so identical images yield identical tensors."""

    def __init__(self) -> None:
        self.tokenizer = _CharTokenizer()
        self.last_image: Image.Image | None = None
        self.call_count = 0

    def __call__(self, image: Image.Image, return_tensors: str = "pt"):
        self.last_image = image
        self.call_count += 1
        small = image.convert("RGB").resize((4, 4))
        arr = np.asarray(small, dtype=np.float32) / 255.0  # (4, 4, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 4, 4)
        return _FakeProcessorResult(tensor)


# ---------- helpers ----------


def _write_rgb_image(path: Path, color: tuple[int, int, int] = (200, 120, 60)) -> None:
    Image.new("RGB", (16, 8), color=color).save(path)


def _make_hf_dataset(texts: list[str], image_path: Path) -> HFDataset:
    """Tiny in-memory HF dataset: every row reuses ``image_path`` (cast to the
    ``datasets.Image`` feature so access yields a decoded PIL image)."""
    ds = HFDataset.from_dict(
        {"image": [str(image_path)] * len(texts), "text": list(texts)}
    )
    return ds.cast_column("image", HFImage())


def _hf_ocr(texts: list[str], image_path: Path) -> HFOCRDataset:
    return HFOCRDataset(
        _make_hf_dataset(texts, image_path),
        processor=_PixelProcessor(),
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
    )


def _texts_in_mix(mixed) -> list[str]:
    """Recover the label of every row actually selected into the mix."""
    out: list[str] = []
    for subset in mixed.datasets:
        df = subset.dataset.df
        for idx in subset.indices:
            out.append(df.iloc[idx]["text"])
    return out


@pytest.fixture
def white_png(tmp_path: Path) -> Path:
    path = tmp_path / "white.png"
    _write_rgb_image(path, (255, 255, 255))
    return path


# ---------- _coerce_to_pil_rgb ----------


def test_coerce_pil_image_to_rgb():
    out = _coerce_to_pil_rgb(Image.new("L", (4, 4), color=128))
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"


def test_coerce_path_str(tmp_path: Path):
    p = tmp_path / "a.png"
    _write_rgb_image(p)
    out = _coerce_to_pil_rgb(str(p))
    assert out.mode == "RGB"
    assert out.size == (16, 8)


def test_coerce_path_object(tmp_path: Path):
    p = tmp_path / "a.png"
    _write_rgb_image(p)
    assert _coerce_to_pil_rgb(p).mode == "RGB"


def test_coerce_bytes_dict(tmp_path: Path):
    p = tmp_path / "a.png"
    _write_rgb_image(p)
    data = p.read_bytes()
    out = _coerce_to_pil_rgb({"bytes": data, "path": None})
    assert out.mode == "RGB"


def test_coerce_path_dict(tmp_path: Path):
    p = tmp_path / "a.png"
    _write_rgb_image(p)
    out = _coerce_to_pil_rgb({"bytes": None, "path": str(p)})
    assert out.mode == "RGB"


def test_coerce_ndarray():
    arr = np.full((8, 8, 3), 50, dtype=np.uint8)
    assert _coerce_to_pil_rgb(arr).mode == "RGB"


def test_coerce_unknown_type_raises():
    with pytest.raises(ValueError, match="Unsupported HF image value"):
        _coerce_to_pil_rgb(42)


def test_coerce_empty_dict_raises():
    with pytest.raises(ValueError, match="neither 'bytes' nor 'path'"):
        _coerce_to_pil_rgb({})


# ---------- HFOCRDataset structure / contract ----------


def test_hf_missing_text_column_raises(white_png: Path):
    ds = HFDataset.from_dict({"image": [str(white_png)]}).cast_column(
        "image", HFImage()
    )
    with pytest.raises(ValueError, match="missing required column"):
        HFOCRDataset(ds, processor=_PixelProcessor())


def test_hf_missing_image_column_raises():
    ds = HFDataset.from_dict({"text": ["agni"]})
    with pytest.raises(ValueError, match="missing required column"):
        HFOCRDataset(ds, processor=_PixelProcessor())


def test_hf_custom_column_names(tmp_path: Path, white_png: Path):
    ds = HFDataset.from_dict({"img": [str(white_png)], "label": ["agni"]}).cast_column(
        "img", HFImage()
    )
    hf_ds = HFOCRDataset(
        ds,
        processor=_PixelProcessor(),
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
        image_column="img",
        text_column="label",
    )
    assert list(hf_ds.df["text"]) == ["agni"]
    sample = hf_ds[0]
    assert set(sample) == {"pixel_values", "labels", "attention_mask"}


def test_hf_df_carries_only_text_column(white_png: Path):
    hf_ds = _hf_ocr(["t1", "t2", "t3"], white_png)
    # The text column is materialised (in HF row order); images stay lazy.
    assert list(hf_ds.df.columns) == ["text"]
    assert list(hf_ds.df["text"]) == ["t1", "t2", "t3"]
    assert len(hf_ds) == 3


def test_hf_getitem_structure(white_png: Path):
    hf_ds = _hf_ocr(["agni"], white_png)
    sample = hf_ds[0]
    assert set(sample) == {"pixel_values", "labels", "attention_mask"}
    assert sample["pixel_values"].shape == (3, 4, 4)
    assert sample["labels"].shape == (8,)
    assert sample["attention_mask"].shape == (8,)


def test_hf_image_transform_applied_before_processor(white_png: Path):
    sentinel = Image.new("RGB", (16, 8), color=(1, 2, 3))
    calls = {"n": 0, "received_rgb": False}

    def transform(image: Image.Image) -> Image.Image:
        calls["n"] += 1
        calls["received_rgb"] = isinstance(image, Image.Image) and image.mode == "RGB"
        return sentinel

    processor = _PixelProcessor()
    hf_ds = HFOCRDataset(
        _make_hf_dataset(["agni"], white_png),
        processor=processor,
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
        image_transform=transform,
    )
    _ = hf_ds[0]
    assert calls["n"] == 1
    assert calls["received_rgb"] is True
    assert processor.last_image is sentinel


def test_hf_applies_input_to_model_scheme_conversion(white_png: Path):
    """The HF path must run the same convert_text(input→model) as the CSV path:
    the tokenizer should receive the slp1_accented form, not the raw ISO text."""
    raw = "agním"  # ISO with a combining acute
    processor = _PixelProcessor()
    hf_ds = HFOCRDataset(
        _make_hf_dataset([raw], white_png),
        processor=processor,
        max_target_length=16,
        input_scheme="iso",
        model_scheme="slp1_accented",
    )
    _ = hf_ds[0]
    expected = convert_text(raw, "iso", "slp1_accented")
    assert processor.tokenizer.last_text == expected
    assert expected != raw  # guard: the conversion is actually exercised


# ---------- CSV vs HF equivalence ----------


def test_csv_and_hf_paths_produce_identical_sample(tmp_path: Path):
    img = tmp_path / "img.png"
    _write_rgb_image(img, (200, 120, 60))
    text = "agním"

    csv_ds = OCRDataset(
        root_dir=tmp_path,
        df=pd.DataFrame({"file_name": ["img.png"], "text": [text]}),
        processor=_PixelProcessor(),
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
    )
    hf_ds = HFOCRDataset(
        _make_hf_dataset([text], img),
        processor=_PixelProcessor(),
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
    )

    a, b = csv_ds[0], hf_ds[0]
    assert torch.equal(a["pixel_values"], b["pixel_values"])
    assert torch.equal(a["labels"], b["labels"])
    assert torch.equal(a["attention_mask"], b["attention_mask"])


# ---------- create_parallel_mixed_dataset on HF-backed corpora ----------


def test_parallel_mix_works_on_hf_datasets(white_png: Path):
    a = _hf_ocr(["t1", "t2", "t3", "t4"], white_png)
    b = _hf_ocr(["t1", "t2", "t3", "t4"], white_png)
    mixed = create_parallel_mixed_dataset(a, b, ratio_a=0.5)
    assert len(mixed) == 4
    assert mixed.n_a == 2
    assert mixed.n_b == 2
    assert mixed.parallel_keys_total == 4
    assert mixed.mix_mode == "parallel_by_text"
    texts = _texts_in_mix(mixed)
    # No text enters as both an RV and an MS image.
    assert len(texts) == 4
    assert len(set(texts)) == 4


@pytest.mark.parametrize("ratio,expected", [(1.0, (4, 0)), (0.0, (0, 4))])
def test_parallel_mix_hf_extreme_ratios(white_png: Path, ratio, expected):
    a = _hf_ocr(["t1", "t2", "t3", "t4"], white_png)
    b = _hf_ocr(["t1", "t2", "t3", "t4"], white_png)
    mixed = create_parallel_mixed_dataset(a, b, ratio_a=ratio)
    assert (mixed.n_a, mixed.n_b) == expected


def test_parallel_mix_mixed_sources_pair_by_text(tmp_path: Path, white_png: Path):
    """A CSV-backed RV corpus and an HF-backed MS corpus with matching text still
    pair by shared text — the mixer only relies on the shared ``.df`` contract."""
    csv_rv = OCRDataset(
        root_dir=tmp_path,
        df=pd.DataFrame({"file_name": ["x.png"] * 4, "text": ["t1", "t2", "t3", "t4"]}),
        processor=_PixelProcessor(),
        max_target_length=8,
        input_scheme="iso",
        model_scheme="iso",
    )
    hf_ms = _hf_ocr(["t1", "t2", "t3", "t4"], white_png)
    mixed = create_parallel_mixed_dataset(csv_rv, hf_ms, ratio_a=0.5)
    assert len(mixed) == 4
    assert mixed.parallel_keys_total == 4


# ---------- load_ocr_dataset (source-aware workhorse) ----------


def test_load_ocr_dataset_unsupported_source_raises():
    with pytest.raises(ValueError, match="Unsupported data source"):
        load_ocr_dataset(source="parquet", processor=_PixelProcessor())


def test_load_ocr_dataset_csv_requires_paths():
    with pytest.raises(ValueError, match="requires both csv_path and image_root"):
        load_ocr_dataset(source="csv", processor=_PixelProcessor())


def test_load_ocr_dataset_hf_requires_id_and_split():
    with pytest.raises(ValueError, match="requires both hf_dataset_id and hf_split"):
        load_ocr_dataset(source="hf", processor=_PixelProcessor(), hf_split="x")
    with pytest.raises(ValueError, match="requires both hf_dataset_id and hf_split"):
        load_ocr_dataset(source="hf", processor=_PixelProcessor(), hf_dataset_id="o/r")


def test_load_ocr_dataset_csv_applies_max_samples(tmp_path: Path):
    img = tmp_path / "x.png"
    _write_rgb_image(img)
    csv = tmp_path / "rows.csv"
    pd.DataFrame({"file_name": ["x.png"] * 3, "text": ["a", "b", "c"]}).to_csv(
        csv, index=False
    )
    ds = load_ocr_dataset(
        source="csv",
        processor=_PixelProcessor(),
        csv_path=csv,
        image_root=tmp_path,
        input_scheme="iso",
        model_scheme="iso",
        max_samples=2,
    )
    assert isinstance(ds, OCRDataset)
    assert len(ds) == 2


def test_load_ocr_dataset_hf_forwards_args_and_selects(monkeypatch, white_png: Path):
    import datasets

    captured: dict = {}
    fixture = _make_hf_dataset(["t1", "t2", "t3"], white_png)

    def fake_load_dataset(dataset_id, split=None, revision=None, token=None):
        captured.update(
            dataset_id=dataset_id, split=split, revision=revision, token=token
        )
        return fixture

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    ds = load_ocr_dataset(
        source="hf",
        processor=_PixelProcessor(),
        hf_dataset_id="org/repo",
        hf_split="rv_train",
        hf_revision="abc123",
        hf_token="tok",
        input_scheme="iso",
        model_scheme="iso",
        max_samples=2,
    )
    assert isinstance(ds, HFOCRDataset)
    assert captured == {
        "dataset_id": "org/repo",
        "split": "rv_train",
        "revision": "abc123",
        "token": "tok",
    }
    assert len(ds) == 2  # max_samples → .select(range(2))


# ---------- load_ocr_dataset_from_config (Hydra resolution) ----------


def _csv_data_cfg_nested(tmp_path: Path) -> tuple[DictConfig, Path]:
    """Build a nested rv/ms CSV config plus the files it points at."""
    data_dir = tmp_path / "data"
    (data_dir / "datasets" / "rv").mkdir(parents=True)
    _write_rgb_image(data_dir / "x.png")
    pd.DataFrame({"file_name": ["x.png"], "text": ["t1"]}).to_csv(
        data_dir / "datasets" / "rv" / "train.csv", index=False
    )
    cfg = OmegaConf.create(
        {
            "source": "csv",
            "image_root": "data",
            "rv": {"train": "data/datasets/rv/train.csv"},
        }
    )
    return cfg, tmp_path


def test_from_config_csv_nested_resolves_paths(tmp_path: Path):
    cfg, root = _csv_data_cfg_nested(tmp_path)
    ds = load_ocr_dataset_from_config(
        cfg,
        split="train",
        domain="rv",
        root_dir=root,
        processor=_PixelProcessor(),
        input_scheme="iso",
        model_scheme="iso",
    )
    assert isinstance(ds, OCRDataset)
    assert ds.root_dir == root / "data"
    # __getitem__ resolves file_name under image_root without error.
    assert set(ds[0]) == {"pixel_values", "labels", "attention_mask"}


def test_from_config_defaults_to_csv_when_source_absent(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_rgb_image(data_dir / "x.png")
    pd.DataFrame({"file_name": ["x.png"], "text": ["t1"]}).to_csv(
        data_dir / "t.csv", index=False
    )
    cfg = OmegaConf.create({"train": "data/t.csv"})  # flat, no `source`
    ds = load_ocr_dataset_from_config(
        cfg,
        split="train",
        root_dir=tmp_path,
        processor=_PixelProcessor(),
        input_scheme="iso",
        model_scheme="iso",
    )
    assert isinstance(ds, OCRDataset)
    assert len(ds) == 1


def test_from_config_hf_nested_forwards_split_and_repo(monkeypatch, white_png: Path):
    import datasets

    captured: dict = {}
    fixture = _make_hf_dataset(["t1", "t2"], white_png)

    def fake_load_dataset(dataset_id, split=None, revision=None, token=None):
        captured.update(
            dataset_id=dataset_id, split=split, revision=revision, token=token
        )
        return fixture

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    cfg = OmegaConf.create(
        {
            "source": "hf",
            "hf": {
                "dataset_id": "o/r",
                "revision": "v1",
                "token": None,
                "image_column": "image",
                "text_column": "text",
                "rv": {"train": "rv_train", "val": "rv_val", "test": "rv_test"},
                "ms": {"train": "ms_train", "val": "ms_val", "test": "ms_test"},
            },
        }
    )
    ds = load_ocr_dataset_from_config(
        cfg,
        split="val",
        domain="ms",
        root_dir="/does-not-matter",
        processor=_PixelProcessor(),
        input_scheme="iso",
        model_scheme="iso",
    )
    assert isinstance(ds, HFOCRDataset)
    assert captured["dataset_id"] == "o/r"
    assert captured["split"] == "ms_val"
    assert captured["revision"] == "v1"


def test_from_config_hf_flat_forwards_split(monkeypatch, white_png: Path):
    import datasets

    captured: dict = {}
    fixture = _make_hf_dataset(["t1"], white_png)

    def fake_load_dataset(dataset_id, split=None, revision=None, token=None):
        captured.update(dataset_id=dataset_id, split=split)
        return fixture

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    cfg = OmegaConf.create(
        {
            "source": "hf",
            "hf": {
                "dataset_id": "o/r",
                "train": "train",
                "val": "validation",
                "test": "test",
            },
        }
    )
    ds = load_ocr_dataset_from_config(
        cfg,
        split="test",
        root_dir="/x",
        processor=_PixelProcessor(),
        input_scheme="iso",
        model_scheme="iso",
    )
    assert isinstance(ds, HFOCRDataset)
    assert captured["split"] == "test"


def test_from_config_hf_requires_hf_block():
    cfg = OmegaConf.create({"source": "hf"})
    with pytest.raises(ValueError, match="requires a data.hf block"):
        load_ocr_dataset_from_config(
            cfg,
            split="train",
            root_dir="/x",
            processor=_PixelProcessor(),
        )


# ---------- package exports ----------


def test_package_level_exports():
    import scripts.training.trocr as pkg

    assert pkg.HFOCRDataset is HFOCRDataset
    assert pkg.load_ocr_dataset is load_ocr_dataset
    assert pkg.load_ocr_dataset_from_config is load_ocr_dataset_from_config
