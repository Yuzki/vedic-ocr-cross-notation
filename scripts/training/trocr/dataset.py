"""Dataset: PACLIC 2026 v2 release."""

import hashlib
import io
import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset

from scripts.utils.transliteration import convert_text

#: How the parallel-subset key list is serialised for hashing and for
#: ``parallel_keys.txt``: one key per line, newline-joined. The hash is over the
#: *ordered* selected key list, so it is a fingerprint of the deterministic
#: selection (data, ``subset_seed``, ``total_samples``) — and, being taken before
#: the ratio split, is identical across the five mix ratios at a fixed subset.
_KEY_LIST_SEP = "\n"


def hash_key_list(keys: Sequence[str]) -> str:
    """SHA256 of an *ordered* key list, newline-joined .

    Order-sensitive so a reshuffle changes the digest, and stable across calls /
    processes so the data-efficiency ladder's subsets can be compared by digest.
    ``parallel_keys.txt`` is written with the same ``"\\n".join(keys)`` body, so
    ``sha256`` of that file (sans trailing newline) reproduces this value.
    """
    return hashlib.sha256(_KEY_LIST_SEP.join(keys).encode("utf-8")).hexdigest()


class _OCRDatasetBase(Dataset):
    """Shared behaviour for OCR datasets regardless of the image source.

    Subclasses supply the *image source* by implementing :meth:`_load_image`;
    everything downstream of the decoded ``RGB`` image — optional train-only
    augmentation, the ``input_scheme`` → ``model_scheme`` text conversion, image
    processing, and tokenisation — lives here so the CSV-backed and HF-backed
    paths feed *identical* pixels and labels to the Trainer .

    ``self.df`` always carries a positional ``text`` column so the parallel-corpus
    mixer (:func:`create_parallel_mixed_dataset`) can pair RV/MS rows by shared
    text on either source. The frame index is reset so positional ``__getitem__``
    keys (the integers ``Subset`` passes back) line up with ``df.iloc`` rows.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        max_target_length: int = 256,
        tokenizer=None,
        accent_system: Optional[str] = None,
        input_scheme: str = "iso",
        model_scheme: str = "slp1_accented",
        image_transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length
        self.accent_system = accent_system
        self.input_scheme = input_scheme
        self.model_scheme = model_scheme
        self.image_transform = image_transform

        if tokenizer:
            self.tokenizer = tokenizer
        elif hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            raise ValueError(
                "No tokenizer provided and processor has no tokenizer attribute"
            )

    def __len__(self):
        return len(self.df)

    def _load_image(self, idx: int) -> Image.Image:
        """Return the decoded ``RGB`` PIL image for positional row ``idx``."""
        raise NotImplementedError

    def __getitem__(self, idx):  # ty: ignore[invalid-method-override]
        text = self.df.iloc[idx]["text"]
        text = convert_text(text, self.input_scheme, self.model_scheme)

        # Load and process image (source-specific) then run the shared pipeline.
        image = self._load_image(idx)
        if self.image_transform is not None:
            image = self.image_transform(image)
        pixel_values = self.processor(
            image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        # Encode the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        # Replace padding token IDs with -100 for loss calculation
        labels = torch.where(
            input_ids != self.tokenizer.pad_token_id, input_ids, torch.tensor(-100)
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class OCRDataset(_OCRDatasetBase):
    """Generic OCR dataset backed by a local image tree.

    Loads from a ``file_name,text`` table where ``file_name`` resolves against
    ``root_dir``. This is the migration / smoke-test / debug compatibility path;
    the published Hugging Face Dataset (:class:`HFOCRDataset`) is the preferred
    long-term input — see the README.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        df: pd.DataFrame,
        processor,
        max_target_length: int = 256,
        tokenizer=None,
        accent_system: Optional[str] = None,
        input_scheme: str = "iso",
        model_scheme: str = "slp1_accented",
        image_transform: Optional[Callable] = None,
    ):
        """Initialize dataset.

        Args:
            root_dir: Root directory for image paths
            df: DataFrame with 'file_name' and 'text' columns
            processor: Image processor (TrOCRProcessor or ViTImageProcessor)
            max_target_length: Maximum length for tokenized text
            tokenizer: Optional separate tokenizer
            accent_system: Optional accent system label (e.g., 'rv', 'ms')
            input_scheme: Text scheme of the CSV ``text`` column.
            model_scheme: Text scheme fed to the tokenizer.
            image_transform: Optional callable applied to the PIL ``Image`` after
                ``Image.open(...).convert("RGB")`` and before ``processor(...)``.
                Intended for *train-only* pixel-level augmentation; leave ``None``
                for val/test to preserve the evaluation input distribution.
        """
        super().__init__(
            df=df,
            processor=processor,
            max_target_length=max_target_length,
            tokenizer=tokenizer,
            accent_system=accent_system,
            input_scheme=input_scheme,
            model_scheme=model_scheme,
            image_transform=image_transform,
        )
        self.root_dir = Path(root_dir)

    def _load_image(self, idx: int) -> Image.Image:
        file_name = self.df.iloc[idx]["file_name"]
        return Image.open(self.root_dir / file_name).convert("RGB")


def _coerce_to_pil_rgb(value: Any) -> Image.Image:
    """Coerce a Hugging Face image-column value to an ``RGB`` PIL image.

    The expected shape is a decoded :class:`PIL.Image.Image` from the
    ``datasets.Image`` feature. We also accept the common non-decoded shapes — a
    ``{"bytes" | "path"}`` dict (``datasets.Image(decode=False)``), a filesystem
    path, or a raw ``numpy`` array — and raise a clear ``ValueError`` on anything
    else rather than letting an opaque error surface deep inside the image
    processor (the study risk note: image columns may arrive in several shapes).
    """
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is not None:
            return Image.open(io.BytesIO(data)).convert("RGB")
        path = value.get("path")
        if path:
            return Image.open(path).convert("RGB")
        raise ValueError(
            "HF image dict carries neither 'bytes' nor 'path'; "
            f"keys present: {sorted(value)}"
        )
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    raise ValueError(
        f"Unsupported HF image value of type {type(value).__name__!r}; expected a "
        "decoded PIL image (datasets.Image), a {'bytes' | 'path'} dict, a path, or "
        "a numpy array"
    )


class HFOCRDataset(_OCRDatasetBase):
    """OCR dataset backed by an already-loaded :class:`datasets.Dataset`.

    Each HF row carries the OCR target ``text`` and a decoded image in
    ``image_column`` (the ``datasets.Image`` feature yields a PIL image on
    access). Images are decoded lazily per ``__getitem__`` so a large split is not
    held in memory; only the small ``text`` column is materialised into
    ``self.df`` (in HF row order) so the parallel-corpus mixer can pair RV/MS rows
    by shared text without forcing every sample through a physical CSV.

    The constructor takes an in-memory dataset object (not a repo id) so it is
    unit-testable without network access; :func:`load_ocr_dataset` performs the
    actual ``datasets.load_dataset`` call.
    """

    def __init__(
        self,
        hf_dataset,
        processor,
        max_target_length: int = 256,
        tokenizer=None,
        accent_system: Optional[str] = None,
        input_scheme: str = "iso",
        model_scheme: str = "slp1_accented",
        image_transform: Optional[Callable] = None,
        image_column: str = "image",
        text_column: str = "text",
    ):
        column_names = list(getattr(hf_dataset, "column_names", []) or [])
        missing = [c for c in (image_column, text_column) if c not in column_names]
        if missing:
            raise ValueError(
                f"HF dataset is missing required column(s) {missing}; available "
                f"columns: {column_names}. Set data.hf.image_column / "
                "data.hf.text_column to match the published dataset."
            )
        # Materialise only the (small) text column; images stay lazy in hf_dataset.
        df = pd.DataFrame({"text": list(hf_dataset[text_column])})
        super().__init__(
            df=df,
            processor=processor,
            max_target_length=max_target_length,
            tokenizer=tokenizer,
            accent_system=accent_system,
            input_scheme=input_scheme,
            model_scheme=model_scheme,
            image_transform=image_transform,
        )
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.text_column = text_column

    def _load_image(self, idx: int) -> Image.Image:
        value = self.hf_dataset[int(idx)][self.image_column]
        return _coerce_to_pil_rgb(value)


# Alias for backward compatibility
IAMDataset = OCRDataset


class MixedDataset(ConcatDataset):
    """``ConcatDataset`` of two sources that records how the mix was built.

    The ``create_*_mixed_dataset`` factories attach these attributes so the
    training script can log the realised composition without re-deriving it from
    the underlying frames. ``parallel_keys_*`` / ``mix_mode`` describe the
    parallel-corpus path ; the generic independent-pool mix leaves the
    key counts as ``None`` and reports ``mix_mode='independent'``.

    The ``selected_parallel_keys`` / ``keys_a`` / ``keys_b`` / ``subset_seed`` /
    ``key_list_sha256`` fields expose the deterministic parallel-subset selection
    so the training script can persist a per-run ``subset_manifest.json`` +
    ``parallel_keys.txt`` and the data-efficiency ladder  is auditable
    and reproducible. They stay ``None`` on the independent mix and on any
    non-parallel path.
    """

    n_a: int
    n_b: int
    ratio_a: float
    parallel_keys_total: Optional[int] = None
    parallel_keys_used: Optional[int] = None
    mix_mode: str = "independent"
    #: Ordered key list after the deterministic shuffle + prefix. Ratio-invariant
    #: for a fixed ``(subset_seed, total_samples)``; length == ``parallel_keys_used``.
    selected_parallel_keys: Optional[list[str]] = None
    #: Keys routed to ``dataset_a`` (RV) / ``dataset_b`` (MS) for *this* ratio;
    #: ``keys_a = selected_parallel_keys[:n_a]``, ``keys_b = the remainder``.
    keys_a: Optional[list[str]] = None
    keys_b: Optional[list[str]] = None
    #: The ``.df`` column the corpora were paired on (defaults to ``text``).
    key_column: Optional[str] = None
    #: The seed that actually drove the selection/partition (the resolved
    #: ``subset_seed``; falls back to ``seed`` when no explicit subset seed given).
    subset_seed: Optional[int] = None
    #: Human-readable label of the selection algorithm (for the manifest).
    selection_algorithm: Optional[str] = None
    #: :func:`hash_key_list` of ``selected_parallel_keys`` (ratio-invariant).
    key_list_sha256: Optional[str] = None


def create_mixed_dataset(
    dataset_a: Dataset,
    dataset_b: Dataset,
    ratio_a: float,
    total_samples: Optional[int] = None,
    seed: int = 42,
) -> MixedDataset:
    """Create a mixed dataset with specified ratio between two datasets.

    Args:
        dataset_a: First dataset
        dataset_b: Second dataset
        ratio_a: Ratio of samples from dataset_a (0.0 to 1.0)
        total_samples: Total number of samples (None = use all available)
        seed: Random seed for reproducibility

    Returns:
        ConcatDataset with samples from both datasets
    """
    random.seed(seed)
    np.random.seed(seed)

    size_a = len(dataset_a)  # ty: ignore[invalid-argument-type]
    size_b = len(dataset_b)  # ty: ignore[invalid-argument-type]

    if total_samples is None:
        total_samples = size_a + size_b

    # Calculate samples from each dataset
    n_a = int(total_samples * ratio_a)
    n_b = total_samples - n_a

    # Ensure we don't exceed available data
    n_a = min(n_a, size_a)
    n_b = min(n_b, size_b)

    # Sample indices
    indices_a = random.sample(range(size_a), n_a) if n_a > 0 else []
    indices_b = random.sample(range(size_b), n_b) if n_b > 0 else []

    # Create subsets
    datasets = []
    if indices_a:
        datasets.append(Subset(dataset_a, indices_a))
    if indices_b:
        datasets.append(Subset(dataset_b, indices_b))

    if not datasets:
        raise ValueError("No samples selected for dataset")

    mixed = MixedDataset(datasets)

    # Store metadata
    mixed.n_a = n_a
    mixed.n_b = n_b
    mixed.ratio_a = ratio_a

    return mixed


def create_parallel_mixed_dataset(
    dataset_a: _OCRDatasetBase,
    dataset_b: _OCRDatasetBase,
    ratio_a: float,
    total_samples: Optional[int] = None,
    seed: int = 42,
    key_column: str = "text",
    subset_seed: Optional[int] = None,
) -> MixedDataset:
    """Mix two *parallel corpora* by routing each shared text to one image style.

    Both corpora may be CSV-backed (:class:`OCRDataset`) or HF-backed
    (:class:`HFOCRDataset`); the mixer only relies on the shared ``.df``
    ``key_column`` view, so the two sources can even be mixed .

    ``dataset_a`` and ``dataset_b`` are assumed to be parallel: the same set of
    ``key_column`` values (e.g. the Sanskrit2003 RV and MS renders of the same
    sentences), differing only in the image. Unlike :func:`create_mixed_dataset`,
    which samples the two as independent pools, this partitions the *shared* texts
    so that:

    - the train mix size is a single fixed ``N`` across the whole ratio sweep
      (``ratio_a`` only changes which image style a text is drawn from, never how
      many texts there are), and
    - no text enters the mix from both sides at once (each text contributes at
      most one image).

    Args:
        dataset_a: First parallel corpus (assigned the ``ratio_a`` share).
        dataset_b: Second parallel corpus (assigned ``1 - ratio_a``).
        ratio_a: Fraction of the selected texts drawn from ``dataset_a``'s images
            (``0.0``–``1.0``).
        total_samples: Number of shared texts to use. ``None`` uses *all* texts
            common to both corpora. An explicit value must be a positive integer
            no greater than the shared count; a non-int, ``< 1``, or value
            exceeding the shared count raises ``ValueError`` rather than silently
            capping or mangling N, so an N change in a controlled sweep cannot
            pass unnoticed.
        seed: Seed for the (process-stable) text selection / partition. Used as
            the selection seed only when ``subset_seed`` is ``None`` (the historical
            behaviour), so existing callers are unaffected.
        key_column: ``.df`` column used to pair the corpora. Defaults to ``text``.
        subset_seed: When given, drives the deterministic text selection / partition
            *instead of* ``seed`` . This decouples the data-efficiency
            subset ladder from the Trainer RNG: fixing ``subset_seed`` across several
            ``seed`` values makes every seed train on the *same* sampled subset, so
            seed variation is training stochasticity rather than a re-draw of the
            low-data sample. ``None`` (default) preserves the previous behaviour of
            selecting with ``seed``.

    Returns:
        :class:`MixedDataset` with ``mix_mode='parallel_by_text'``, the
        ``n_a`` / ``n_b`` / ``parallel_keys_total`` / ``parallel_keys_used``
        counts, and the ``selected_parallel_keys`` / ``keys_a`` / ``keys_b`` /
        ``subset_seed`` / ``key_list_sha256`` provenance populated.

    Note:
        This intentionally does *not* address train/test text leakage; that is a
        separate split-cleaning concern .
    """
    if not 0.0 <= ratio_a <= 1.0:
        raise ValueError(f"ratio_a must be in [0.0, 1.0], got {ratio_a}")

    # ``subset_seed`` (when given) drives the selection so the data subset ladder
    # can be held fixed across Trainer seeds; otherwise fall back to ``seed`` for
    # exact previous behaviour. Only this resolved seed touches the selection RNG.
    resolved_subset_seed = seed if subset_seed is None else subset_seed
    rng = random.Random(resolved_subset_seed)

    def _representatives(df: pd.DataFrame) -> dict:
        """Map each ``key_column`` value to a single positional row index.

        ``OCRDataset`` resets its frame index, so positional indices double as
        the integer keys ``Subset`` -> ``OCRDataset.__getitem__`` expects. When a
        text repeats within one corpus we seeded-shuffle the duplicates and keep
        the first, so each text contributes at most one image to the mix.
        """
        groups: dict = {}
        for pos, key in enumerate(df[key_column].tolist()):
            groups.setdefault(key, []).append(pos)
        reps: dict = {}
        for key, positions in groups.items():
            if len(positions) > 1:
                shuffled = list(positions)
                rng.shuffle(shuffled)
                reps[key] = shuffled[0]
            else:
                reps[key] = positions[0]
        return reps

    reps_a = _representatives(dataset_a.df)
    reps_b = _representatives(dataset_b.df)

    common_keys = set(reps_a) & set(reps_b)
    parallel_keys_total = len(common_keys)

    if total_samples is None:
        total_samples = parallel_keys_total
    else:
        # ``total_train_samples`` is part of the controlled-sweep contract, so a
        # bad Hydra override (negative, float, bool) must fail clearly here
        # rather than slice with a junk value and record inconsistent metadata.
        # ``bool`` is an ``int`` subclass, hence the explicit reject.
        if isinstance(total_samples, bool) or not isinstance(total_samples, int):
            raise ValueError(
                "total_samples must be a positive int or None, got "
                f"{total_samples!r} ({type(total_samples).__name__})"
            )
        if total_samples < 1:
            raise ValueError(f"total_samples must be >= 1, got {total_samples}")
        if total_samples > parallel_keys_total:
            raise ValueError(
                f"total_samples={total_samples} exceeds the {parallel_keys_total} "
                "texts shared by both corpora; refusing to silently cap (this would "
                "change N across a ratio sweep)"
            )

    # ``sorted`` first so the order is deterministic across processes (Python's
    # per-process str hash randomisation makes raw ``set`` iteration order vary,
    # which would desync the selection between the sweep's separate runs); then a
    # seeded shuffle. The result depends only on (data, seed, total_samples) —
    # not on ``ratio_a`` — so the *selected* text set is identical across ratios.
    ordered_keys = sorted(common_keys)
    rng.shuffle(ordered_keys)
    selected_keys = ordered_keys[:total_samples]

    # Partition the already-shuffled selection by a simple prefix split. Because
    # ``selected_keys`` is ratio-independent, the assignment is nested: a text
    # routed to A at ratio 0.5 is also routed to A at ratio 1.0.
    n_a = int(total_samples * ratio_a)
    n_b = total_samples - n_a
    keys_a = selected_keys[:n_a]
    keys_b = selected_keys[n_a:]

    indices_a = [reps_a[k] for k in keys_a]
    indices_b = [reps_b[k] for k in keys_b]

    subsets: list = []
    if indices_a:
        subsets.append(Subset(dataset_a, indices_a))
    if indices_b:
        subsets.append(Subset(dataset_b, indices_b))

    if not subsets:
        raise ValueError("No samples selected for dataset")

    mixed = MixedDataset(subsets)
    mixed.n_a = n_a
    mixed.n_b = n_b
    mixed.ratio_a = ratio_a
    mixed.parallel_keys_total = parallel_keys_total
    mixed.parallel_keys_used = total_samples
    mixed.mix_mode = "parallel_by_text"
    # Deterministic-subset provenance . ``selected_keys`` is the
    # ratio-invariant ordered selection; ``keys_a``/``keys_b`` are its prefix
    # split for this ratio; the hash fingerprints the selection independent of
    # ratio. Store copies so a later mutation of the local lists can't alias in.
    mixed.selected_parallel_keys = list(selected_keys)
    mixed.keys_a = list(keys_a)
    mixed.keys_b = list(keys_b)
    mixed.key_column = key_column
    mixed.subset_seed = resolved_subset_seed
    mixed.selection_algorithm = "sorted_common_keys+seeded_shuffle+prefix"
    mixed.key_list_sha256 = hash_key_list(selected_keys)

    return mixed


def load_dataset_from_csv(
    csv_path: Union[str, Path],
    root_dir: Union[str, Path],
    processor,
    max_target_length: int = 256,
    tokenizer=None,
    accent_system: Optional[str] = None,
    input_scheme: str = "iso",
    model_scheme: str = "slp1_accented",
    image_transform: Optional[Callable] = None,
) -> OCRDataset:
    """Convenience function to load dataset from CSV file.

    Args:
        csv_path: Path to CSV file with 'file_name' and 'text' columns
        root_dir: Root directory for image paths
        processor: Image processor
        max_target_length: Maximum length for tokenized text
        tokenizer: Optional separate tokenizer
        accent_system: Optional accent system label
        input_scheme: Text scheme of the CSV ``text`` column.
        model_scheme: Text scheme fed to the tokenizer.
        image_transform: Optional PIL-image transform; forwarded to ``OCRDataset``.

    Returns:
        OCRDataset instance
    """
    df = pd.read_csv(csv_path)
    return OCRDataset(
        root_dir=root_dir,
        df=df,
        processor=processor,
        max_target_length=max_target_length,
        tokenizer=tokenizer,
        accent_system=accent_system,
        input_scheme=input_scheme,
        model_scheme=model_scheme,
        image_transform=image_transform,
    )


def load_ocr_dataset(
    *,
    source: str,
    processor,
    tokenizer=None,
    input_scheme: str = "iso",
    model_scheme: str = "slp1_accented",
    max_target_length: int = 256,
    image_transform: Optional[Callable] = None,
    accent_system: Optional[str] = None,
    max_samples: Optional[int] = None,
    csv_path: Optional[Union[str, Path]] = None,
    image_root: Optional[Union[str, Path]] = None,
    hf_dataset_id: Optional[str] = None,
    hf_split: Optional[str] = None,
    hf_revision: Optional[str] = None,
    hf_token: Optional[Union[str, bool]] = None,
    hf_image_column: str = "image",
    hf_text_column: str = "text",
) -> _OCRDatasetBase:
    """Build an ``OCRDataset``-compatible dataset from a CSV or HF source.

    This is the source-aware workhorse behind :func:`load_ocr_dataset_from_config`
    and is intentionally fully explicit (no Hydra coupling) so it is easy to unit
    test:

    - ``source='csv'`` reads a local ``file_name,text`` table (``csv_path``) whose
      images resolve under ``image_root`` — the compatibility / debug path.
    - ``source='hf'`` loads ``hf_dataset_id`` (optionally pinned to
      ``hf_revision``) at ``hf_split`` via ``datasets.load_dataset`` and wraps it
      in :class:`HFOCRDataset` — the preferred long-term input.

    Both return objects expose ``.df`` with a positional ``text`` column, so the
    parallel-corpus mixer and the rest of the pipeline are source-agnostic.
    ``max_samples`` keeps the first N rows (CSV ``head`` / HF ``select``) for the
    debug sample-limit knobs; ``None`` uses the whole split. Unknown ``source``
    values, or a CSV/HF call missing its required inputs, raise ``ValueError``.
    """
    source_normalized = (source or "csv").lower()

    if source_normalized == "csv":
        if csv_path is None or image_root is None:
            raise ValueError("source='csv' requires both csv_path and image_root")
        df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples < len(df):
            df = df.head(max_samples)
        return OCRDataset(
            root_dir=image_root,
            df=df,
            processor=processor,
            max_target_length=max_target_length,
            tokenizer=tokenizer,
            accent_system=accent_system,
            input_scheme=input_scheme,
            model_scheme=model_scheme,
            image_transform=image_transform,
        )

    if source_normalized == "hf":
        if not hf_dataset_id or not hf_split:
            raise ValueError(
                "source='hf' requires both hf_dataset_id and hf_split (the split "
                "name in the published dataset)"
            )
        # Imported lazily so the common CSV path (and importing this module)
        # doesn't pay the heavy ``datasets`` import cost.
        from datasets import Dataset as HFDatasetType
        from datasets import load_dataset

        hf_dataset = load_dataset(
            hf_dataset_id,
            split=hf_split,
            revision=hf_revision,
            token=hf_token,
        )
        # ``load_dataset`` is typed as a broad union; passing a concrete ``split``
        # returns a single non-streaming ``Dataset``. Assert it so the row count /
        # select below type-check and so a streaming/dict input fails loudly
        # (streaming is intentionally out of scope for the first cut — the study).
        if not isinstance(hf_dataset, HFDatasetType):
            raise TypeError(
                f"Expected a single datasets.Dataset for split {hf_split!r}, got "
                f"{type(hf_dataset).__name__}; streaming/multi-split inputs are not "
                "supported yet"
            )
        if max_samples is not None and max_samples < len(hf_dataset):
            hf_dataset = hf_dataset.select(range(max_samples))
        return HFOCRDataset(
            hf_dataset=hf_dataset,
            processor=processor,
            max_target_length=max_target_length,
            tokenizer=tokenizer,
            accent_system=accent_system,
            input_scheme=input_scheme,
            model_scheme=model_scheme,
            image_transform=image_transform,
            image_column=hf_image_column,
            text_column=hf_text_column,
        )

    raise ValueError(f"Unsupported data source {source!r}; expected 'csv' or 'hf'")


def load_ocr_dataset_from_config(
    data_cfg,
    *,
    split: str,
    root_dir: Union[str, Path],
    processor,
    tokenizer=None,
    input_scheme: str = "iso",
    model_scheme: str = "slp1_accented",
    max_target_length: int = 256,
    image_transform: Optional[Callable] = None,
    max_samples: Optional[int] = None,
    domain: Optional[str] = None,
    accent_system: Optional[str] = None,
) -> _OCRDatasetBase:
    """Resolve a Hydra ``data`` config into a :func:`load_ocr_dataset` call.

    ``data_cfg.source`` defaults to ``'csv'`` when absent, so existing CSV configs
    keep working untouched. ``domain`` selects the RV/MS sub-block of the nested
    experiment config (``data.rv`` / ``data.ms``); ``domain=None`` reads a flat
    single-dataset config (``data.train`` / ``data.val`` / ``data.test``) for the
    generic train/test paths.

    CSV paths and ``image_root`` (default ``data``) resolve against ``root_dir``
    (the project root). For HF, the split name comes from ``data.hf[domain][split]``
    (or ``data.hf[split]`` when flat) and the repo-level fields — ``dataset_id``,
    ``revision``, ``token``, ``image_column``, ``text_column`` — from ``data.hf``.
    """
    source = str(data_cfg.get("source", "csv") or "csv").lower()
    root_dir = Path(root_dir)

    if source == "csv":
        node = data_cfg[domain] if domain is not None else data_cfg
        csv_rel = node[split]
        image_root_rel = data_cfg.get("image_root", "data") or "data"
        return load_ocr_dataset(
            source="csv",
            processor=processor,
            tokenizer=tokenizer,
            input_scheme=input_scheme,
            model_scheme=model_scheme,
            max_target_length=max_target_length,
            image_transform=image_transform,
            accent_system=accent_system,
            max_samples=max_samples,
            csv_path=root_dir / csv_rel,
            image_root=root_dir / image_root_rel,
        )

    if source == "hf":
        hf_cfg = data_cfg.get("hf")
        if hf_cfg is None:
            raise ValueError(
                "source='hf' requires a data.hf block (dataset_id, columns, and "
                "the split mapping)"
            )
        split_node = hf_cfg[domain] if domain is not None else hf_cfg
        hf_split = split_node[split]
        return load_ocr_dataset(
            source="hf",
            processor=processor,
            tokenizer=tokenizer,
            input_scheme=input_scheme,
            model_scheme=model_scheme,
            max_target_length=max_target_length,
            image_transform=image_transform,
            accent_system=accent_system,
            max_samples=max_samples,
            hf_dataset_id=hf_cfg.get("dataset_id"),
            hf_split=hf_split,
            hf_revision=hf_cfg.get("revision"),
            hf_token=hf_cfg.get("token"),
            hf_image_column=hf_cfg.get("image_column", "image"),
            hf_text_column=hf_cfg.get("text_column", "text"),
        )

    raise ValueError(f"Unsupported data source {source!r}; expected 'csv' or 'hf'")
