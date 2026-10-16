"""Test trocr early stopping: PACLIC 2026 v2 release."""

from types import SimpleNamespace

from omegaconf import OmegaConf
from transformers import EarlyStoppingCallback, Seq2SeqTrainer

from scripts.training.trocr.experiment_accent_ratio import MultiEvalSeq2SeqTrainer
from scripts.training.trocr.train import early_stopping_callback_from_params


def test_default_patience_enabled():
    cb = early_stopping_callback_from_params(OmegaConf.create({}))
    assert isinstance(cb, EarlyStoppingCallback)
    assert cb.early_stopping_patience == 3


def test_custom_patience():
    cb = early_stopping_callback_from_params(
        OmegaConf.create({"early_stopping_patience": 5})
    )
    assert isinstance(cb, EarlyStoppingCallback)
    assert cb.early_stopping_patience == 5


def test_disabled_with_zero():
    assert (
        early_stopping_callback_from_params(
            OmegaConf.create({"early_stopping_patience": 0})
        )
        is None
    )


def test_disabled_with_negative():
    assert (
        early_stopping_callback_from_params(
            OmegaConf.create({"early_stopping_patience": -1})
        )
        is None
    )


# ---- multi-eval early stopping wiring  ----------------------------


class _FakeCallbackHandler:
    """Minimal stand-in for HF's CallbackHandler list management."""

    def __init__(self, callbacks):
        self.callbacks = list(callbacks)

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def remove_callback(self, cb):
        self.callbacks = [c for c in self.callbacks if c is not cb]


def test_multi_eval_early_stopping_sees_eval_avg_cer(monkeypatch):
    """Dict eval must hand EarlyStoppingCallback the merged metrics (with
    ``eval_avg_cer``), and must NOT fire it on the avg-less inner per-target
    evals ."""
    seen_metrics: list[dict] = []
    inner_prefixes: list[str] = []

    class RecordingEarlyStopping(EarlyStoppingCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            seen_metrics.append(dict(metrics or {}))
            return control

    def fake_super_evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kw
    ):
        # Mimic HF's inner (non-dict) evaluate: fire whatever callbacks remain
        # attached, with per-target metrics that have no eval_avg_*.
        inner_prefixes.append(metric_key_prefix)
        inner = {"eval_rv_cer": 0.4} if metric_key_prefix == "eval_rv" else None
        inner = {"eval_ms_cer": 0.6} if metric_key_prefix == "eval_ms" else inner
        assert inner is not None, metric_key_prefix
        for cb in self.callback_handler.callbacks:
            cb.on_evaluate(self.args, self.state, self.control, metrics=inner)
        return inner

    monkeypatch.setattr(Seq2SeqTrainer, "evaluate", fake_super_evaluate)

    trainer = object.__new__(MultiEvalSeq2SeqTrainer)
    early_stopping = RecordingEarlyStopping(early_stopping_patience=3)
    trainer.callback_handler = _FakeCallbackHandler([early_stopping])
    trainer.eval_dataset = {"rv": object(), "ms": object()}
    trainer.args = SimpleNamespace(metric_for_best_model="avg_cer")
    trainer.state = SimpleNamespace()
    trainer.control = SimpleNamespace()
    trainer.log = lambda *a, **k: None

    metrics = trainer.evaluate()

    # avg injected from the two inner targets.
    assert metrics["eval_avg_cer"] == 0.5
    # both per-target inner evals ran, in order.
    assert inner_prefixes == ["eval_rv", "eval_ms"]
    # early stopping fired exactly once, on the merged dict that has eval_avg_cer
    # (and never on the avg-less inner metrics).
    assert len(seen_metrics) == 1
    assert "eval_avg_cer" in seen_metrics[0]
    # the callback is back on the handler afterwards (detach was temporary).
    assert early_stopping in trainer.callback_handler.callbacks
