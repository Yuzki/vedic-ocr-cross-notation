"""Train: PACLIC 2026 v2 release."""
from transformers import EarlyStoppingCallback

def early_stopping_callback_from_params(params):
    """Return an ``EarlyStoppingCallback`` from training params, or ``None``.

    Enabled by default (patience=3); set ``params.early_stopping_patience=0``
    (or negative) to disable -- e.g. overfit-sanity runs that intentionally
    train a fixed number of epochs. Assumes ``load_best_model_at_end=True`` plus
    a ``metric_for_best_model`` and a non-"no" ``eval_strategy`` (all set in the
    training args of both training entrypoints).
    """
    patience = params.get("early_stopping_patience", 3)
    if patience and patience > 0:
        return EarlyStoppingCallback(early_stopping_patience=patience)
    return None

