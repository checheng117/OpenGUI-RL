"""Tests for configuration loading."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from gui_grounding.constants import CONFIGS_DIR
from gui_grounding.utils.config import load_config


class TestConfigLoading:
    def test_load_sft_config(self):
        cfg = load_config("train/sft_baseline.yaml")
        assert "experiment" in cfg
        assert cfg.experiment.name == "sft_baseline"

    def test_load_rerank_config(self):
        cfg = load_config("train/rerank_reward.yaml")
        assert cfg.experiment.stage == "B"

    def test_load_eval_config(self):
        cfg = load_config("eval/grounding_eval.yaml")
        assert "evaluation" in cfg

    def test_load_demo_config(self):
        cfg = load_config("demo/demo.yaml")
        assert "demo" in cfg

    def test_load_absolute_path(self):
        cfg = load_config(CONFIGS_DIR / "train" / "sft_baseline.yaml")
        assert cfg.experiment.name == "sft_baseline"

    def test_nonexistent_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_overrides(self):
        cfg = load_config(
            "train/sft_baseline.yaml",
            overrides=["training.learning_rate=1e-3", "training.num_epochs=10"],
        )
        assert cfg.training.learning_rate == 1e-3
        assert cfg.training.num_epochs == 10

    def test_all_data_configs_load(self):
        for name in ["mind2web", "screenspot_v2", "visualwebbench", "screenspot_pro"]:
            cfg = load_config(f"data/{name}.yaml")
            assert "dataset" in cfg

    def test_all_model_configs_load(self):
        for name in ["qwen2_vl_2b", "qwen2_vl_7b"]:
            cfg = load_config(f"model/{name}.yaml")
            assert "model" in cfg
