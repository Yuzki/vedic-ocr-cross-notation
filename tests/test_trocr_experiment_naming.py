"""Test trocr experiment naming: PACLIC 2026 v2 release."""

from scripts.training.trocr.experiment_accent_ratio import build_experiment_name


def test_no_suffix_preserves_historical_name():
    assert build_experiment_name(1.0, 0.0) == "rv_100_ms_0"
    assert build_experiment_name(0.5, 0.5) == "rv_50_ms_50"
    assert build_experiment_name(0.0, 1.0) == "rv_0_ms_100"


def test_none_suffix_is_explicit_no_op():
    assert build_experiment_name(0.5, 0.5, None) == "rv_50_ms_50"


def test_suffix_appended():
    assert build_experiment_name(1.0, 0.0, "iso") == "rv_100_ms_0_iso"
    assert (
        build_experiment_name(1.0, 0.0, "slp1_accented") == "rv_100_ms_0_slp1_accented"
    )


def test_matched_pair_at_same_ratio_is_distinct():
    """The whole point of study: same ratio, different scheme ⇒ different names."""
    iso = build_experiment_name(0.5, 0.5, "iso")
    slp1 = build_experiment_name(0.5, 0.5, "slp1_accented")
    assert iso != slp1
    assert iso == "rv_50_ms_50_iso"
    assert slp1 == "rv_50_ms_50_slp1_accented"


def test_blank_or_whitespace_suffix_is_dropped():
    assert build_experiment_name(0.5, 0.5, "") == "rv_50_ms_50"
    assert build_experiment_name(0.5, 0.5, "   ") == "rv_50_ms_50"


def test_suffix_is_stripped():
    assert build_experiment_name(1.0, 0.0, "  iso  ") == "rv_100_ms_0_iso"


# --- Freeze-mode naming  ----------------------------------------
#
# Non-default freeze modes must get a distinct run name so encoder/decoder-frozen
# conditions at the same RV/MS ratio don't overwrite the full-fine-tune outputs.
# ``freeze=none`` preserves the historical ``rv_<n>_ms_<n>`` names, and an
# explicit ``run_suffix`` keeps precedence (the user owns naming).


def test_freeze_none_preserves_historical_name():
    assert build_experiment_name(1.0, 0.0, freeze="none") == "rv_100_ms_0"
    assert build_experiment_name(0.5, 0.5, None, "none") == "rv_50_ms_50"
    # ``None`` freeze is treated as the default ``none``.
    assert build_experiment_name(0.5, 0.5, freeze=None) == "rv_50_ms_50"


def test_non_default_freeze_appended_when_no_suffix():
    assert (
        build_experiment_name(1.0, 0.0, freeze="encoder")
        == "rv_100_ms_0_freeze_encoder"
    )
    assert (
        build_experiment_name(0.0, 1.0, freeze="decoder")
        == "rv_0_ms_100_freeze_decoder"
    )


def test_freeze_modes_are_distinct_at_same_ratio():
    full = build_experiment_name(0.5, 0.5, freeze="none")
    enc = build_experiment_name(0.5, 0.5, freeze="encoder")
    dec = build_experiment_name(0.5, 0.5, freeze="decoder")
    assert len({full, enc, dec}) == 3
    assert full == "rv_50_ms_50"
    assert enc == "rv_50_ms_50_freeze_encoder"
    assert dec == "rv_50_ms_50_freeze_decoder"


def test_explicit_suffix_takes_precedence_over_freeze():
    """An explicit run_suffix owns the name; the freeze mode is recorded in
    WandB / results.json instead of being appended to the dir name."""
    assert build_experiment_name(0.5, 0.5, "iso", freeze="encoder") == "rv_50_ms_50_iso"


def test_freeze_mode_is_normalized_in_name():
    assert (
        build_experiment_name(1.0, 0.0, freeze="  ENCODER ")
        == "rv_100_ms_0_freeze_encoder"
    )
