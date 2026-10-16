"""Test trocr seq2seq loss: PACLIC 2026 v2 release."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers.loss.loss_utils import ForCausalLMLoss

from scripts.training.trocr import experiment_accent_ratio as ear
from scripts.training.trocr.dataset import hash_key_list
from scripts.training.trocr.seq2seq_loss import (
    install_no_shift_loss,
    no_shift_cross_entropy_loss,
)

V, T = 5, 4


def _peaky_logits(labels):
    """Logits where position i puts all mass on labels[i] (perfect no-shift)."""
    logits = torch.full((1, T, V), -10.0)
    for i, tok in enumerate(labels[0].tolist()):
        if tok != -100:
            logits[0, i, tok] = 10.0
    return logits


def test_supervises_logits_i_to_labels_i():
    labels = torch.tensor([[1, 2, 3, 4]])
    logits = _peaky_logits(labels)
    loss = no_shift_cross_entropy_loss(logits, labels, V)
    # logits[i] already peaks at labels[i] -> loss ~ 0 only if there is NO shift.
    assert loss.item() < 1e-3
    # Matches plain (no-shift) cross-entropy exactly.
    ref = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
    assert torch.allclose(loss, ref)


def test_differs_from_shifting_causal_loss():
    """Same logits that are perfect under no-shift must be *bad* under the
    shifting ForCausalLMLoss -- this is what the fix removes."""
    labels = torch.tensor([[1, 2, 3, 4]])
    logits = _peaky_logits(labels)
    no_shift = no_shift_cross_entropy_loss(logits, labels, V)
    causal = ForCausalLMLoss(logits, labels, V)
    assert no_shift.item() < 1e-3
    assert causal.item() > 1.0  # the second shift mis-aligns the targets
    assert no_shift.item() < causal.item()


def test_respects_ignore_index():
    labels = torch.tensor([[1, 2, -100, 4]])
    logits = _peaky_logits(labels)  # position 2 left unpeaked, then ignored
    loss = no_shift_cross_entropy_loss(logits, labels, V)
    assert loss.item() < 1e-3


def test_num_items_in_batch_uses_sum_over_n():
    labels = torch.tensor([[1, 2, 3, 4]])
    logits = torch.randn(1, T, V)
    n = torch.tensor(2)
    got = no_shift_cross_entropy_loss(logits, labels, V, num_items_in_batch=n)
    ref = (
        F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="sum") / 2
    )
    assert torch.allclose(got, ref)


def test_install_overrides_loss_function():
    class Dummy:
        pass

    model = Dummy()
    returned = install_no_shift_loss(model)
    assert returned is model
    assert model.loss_function is no_shift_cross_entropy_loss


class _StopBeforeTrainer(Exception):
    """Trip wire raised by the fake trainer to halt run_experiment at the point
    the model is handed to MultiEvalSeq2SeqTrainer."""


class _FakeDataset:
    """Stand-in for both the per-domain OCRDatasets and the mixed train set.

    Carries the parallel-corpus metadata (the study: ``mix_mode`` /
    ``parallel_keys_*``) plus the study deterministic-subset provenance
    (``subset_seed`` / ``selected_parallel_keys`` / routed ``keys_a`` / ``keys_b``
    / ``key_list_sha256`` …) that ``run_experiment`` reads for its dataset logging
    and the per-run ``subset_manifest.json`` + ``parallel_keys.txt`` it now writes.
    """

    def __init__(self, *args, **kwargs):
        self.n_a = 2
        self.n_b = 2
        self.parallel_keys_total = 4
        self.parallel_keys_used = 4
        self.mix_mode = "parallel_by_text"
        # study subset provenance: a coherent 4-key selection split keys_a | keys_b,
        # so the real build_subset_metadata / write_subset_manifest run end-to-end.
        self.selected_parallel_keys = ["k0", "k1", "k2", "k3"]
        self.keys_a = ["k0", "k1"]
        self.keys_b = ["k2", "k3"]
        self.key_column = "text"
        self.subset_seed = 42
        self.selection_algorithm = "sorted_common_keys+seeded_shuffle+prefix"
        self.key_list_sha256 = hash_key_list(self.selected_parallel_keys)

    def __len__(self):
        return 4


def _drive_run_experiment_to_trainer(
    tmp_path, monkeypatch, *, mixed_factory=None, cfg_extra=None
):
    """Drive the real ``run_experiment`` ordering with every heavyweight
    dependency between model load and trainer construction stubbed out, tripping
    a fake trainer the moment the model is handed to ``MultiEvalSeq2SeqTrainer``.

    Returns the captured dict (``loss_function`` at trainer-build time). Pass
    ``mixed_factory`` to override the ``create_parallel_mixed_dataset`` stub (e.g.
    to record its call args); ``cfg_extra`` is merged into the base Hydra config.
    """

    class _FakeModel:
        def __init__(self):
            # A non-no-shift sentinel: the loss assertion only passes if the
            # production install_no_shift_loss call actually ran on this model.
            self.loss_function = "DEFAULT_SHIFTING_LOSS"
            # run_experiment now calls apply_freeze_mode(model, ...) before the
            # trainer ; the helper reads .encoder/.decoder/.parameters()
            # to count trainable params. Empty param iterables keep this fake
            # light while exercising the real freeze code path.
            self.encoder = SimpleNamespace(parameters=lambda: iter(()))
            self.decoder = SimpleNamespace(parameters=lambda: iter(()))

        def parameters(self):
            return iter(())

        def to(self, *args, **kwargs):
            return self

    captured = {}

    class _FakeTrainer:
        def __init__(self, *, model, **kwargs):
            captured["loss_function"] = model.loss_function
            raise _StopBeforeTrainer

    monkeypatch.setattr(
        ear, "wandb", SimpleNamespace(init=lambda **k: None, log=lambda *a, **k: None)
    )
    monkeypatch.setattr(
        ear, "load_model_and_processor", lambda cfg: (_FakeModel(), object(), object())
    )
    monkeypatch.setattr(ear, "configure_model_for_generation", lambda *a, **k: None)
    monkeypatch.setattr(ear, "build_train_transform", lambda cfg: None)
    # The six per-domain RV/MS splits now load through this single source-aware
    # helper  instead of pd.read_csv + OCRDataset; each call returns a
    # fake carrying the parallel-corpus metadata the dataset logging reads.
    monkeypatch.setattr(
        ear, "load_ocr_dataset_from_config", lambda *a, **k: _FakeDataset()
    )
    monkeypatch.setattr(
        ear,
        "create_parallel_mixed_dataset",
        mixed_factory or (lambda *a, **k: _FakeDataset()),
    )
    monkeypatch.setattr(
        ear, "Seq2SeqTrainingArguments", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        ear, "create_compute_metrics_fn", lambda **kwargs: lambda *a, **k: {}
    )
    monkeypatch.setattr(ear, "MultiEvalSeq2SeqTrainer", _FakeTrainer)

    cfg_dict = {
        "rv_ratio": 0.5,
        "wandb": {"project": "test", "entity": None},
        "wandb_mode": "disabled",
        "output_dir": str(tmp_path),
        "seed": 42,
        "eval_targets": ["rv", "ms"],
        "params": {
            "max_length": 8,
            "num_train_epochs": 1,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.0,
        },
        "augmentation": {"enabled": False},
        "data": {
            "rv": {"train": "rv_train.csv", "val": "rv_val.csv", "test": "rv_test.csv"},
            "ms": {"train": "ms_train.csv", "val": "ms_val.csv", "test": "ms_test.csv"},
        },
    }
    if cfg_extra:
        cfg_dict.update(cfg_extra)

    cfg = OmegaConf.create(cfg_dict)
    with pytest.raises(_StopBeforeTrainer):
        ear.run_experiment(cfg)
    return captured


def test_run_experiment_installs_no_shift_loss_before_trainer(tmp_path, monkeypatch):
    """The active experiment entrypoint (used by the RV/MS and preadapt sbatch
    jobs) must install the no-shift loss on the model *before* constructing
    MultiEvalSeq2SeqTrainer. Otherwise the trainer optimizes the default
    double-shifting ForCausalLMLoss and free-running generate() degrades even
    though eval_loss looks healthy (see seq2seq_loss.py).
    """
    captured = _drive_run_experiment_to_trainer(tmp_path, monkeypatch)
    assert captured["loss_function"] is no_shift_cross_entropy_loss


def test_run_experiment_uses_parallel_mixed_dataset(tmp_path, monkeypatch):
    """run_experiment must build the train mix via create_parallel_mixed_dataset
    , forwarding ``total_train_samples`` and ``seed``; the parallel
    metadata it returns must flow through the dataset logging without error."""
    calls = []

    def _recording_mix(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeDataset()

    _drive_run_experiment_to_trainer(
        tmp_path,
        monkeypatch,
        mixed_factory=_recording_mix,
        cfg_extra={"total_train_samples": 4},
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["total_samples"] == 4
    assert kwargs["seed"] == 42
    assert kwargs["ratio_a"] == 0.5
