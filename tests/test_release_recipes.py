"""Public recipe axes and the immutable corpus attestation."""
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import reproduce
from scripts.evaluation import build_paclic_v2_maingrid_analysis as analysis


def test_recipe_axes_and_counts():
    main, low = reproduce.cells("maingrid"), reproduce.cells("data-efficiency")
    assert len(main) == len(set(main)) == 15
    assert len(low) == len(set(low)) == 27
    assert {n for n, _, _ in low} == {1042, 2605, 5210}
    assert not set(main) & set(low)


@pytest.mark.parametrize("study,cell", [("maingrid", 6), ("data-efficiency", 0), ("data-efficiency", 26)])
def test_recipe_resolves_to_paper_settings(study, cell):
    with initialize_config_dir(version_base="1.3", config_dir=str(reproduce.ROOT / "scripts/training/trocr/conf")):
        cfg = compose(config_name="experiment", overrides=reproduce.overrides(study, cell))
        OmegaConf.resolve(cfg)
    n, ratio, seed = reproduce.cells(study)[cell]
    assert (cfg.total_train_samples, cfg.rv_ratio, cfg.seed) == (n, ratio, seed)
    assert cfg.parallel_subset_seed is None
    assert cfg.model.revision == reproduce.MODEL_REVISION
    assert cfg.data.source == "hf"
    assert cfg.data.hf.revision == reproduce.DATASET_REVISION
    assert cfg.params.num_train_epochs == 20
    assert cfg.params.train_batch_size == cfg.params.eval_batch_size == 16
    assert cfg.params.learning_rate == 2e-5
    assert cfg.params.weight_decay == 0.01
    assert cfg.params.warmup_ratio == 0.1
    assert cfg.params.lr_scheduler_type == "linear"
    assert cfg.best_model_policy == "dominant_cer"
    assert cfg.best_model_tie_breaker == "avg_cer"
    assert cfg.generation.decoder_start_token_id is None
    assert cfg.generation.num_beams == 4
    assert cfg.generation.length_penalty == 2.0
    assert cfg.generation.no_repeat_ngram_size == 3
    assert cfg.augmentation.enabled is False
    assert cfg.text.model_scheme == "iso"


def test_pinned_hub_corpus_passes_attestation_and_changed_revision_fails():
    text = f"run_name=rv_50_ms_50_seed_42\ncorpus=v2 data_source=hf dataset_id={reproduce.DATASET} dataset_revision={reproduce.DATASET_REVISION}\n"
    rec = analysis.parse_corpus_attestation(text)
    assert analysis.attestation_defects(rec) == []
    rec["dataset_revision"] = "main"
    assert analysis.attestation_defects(rec)
