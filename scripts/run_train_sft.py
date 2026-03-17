#!/usr/bin/env python3
"""Stage A: Supervised fine-tuning for GUI grounding.

Usage:
    python scripts/run_train_sft.py --config configs/train/sft_baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.config import load_config
from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed

logger = get_logger("run_train_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for GUI grounding")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Load config and exit without training")
    parser.add_argument("overrides", nargs="*", help="Config overrides (dot-list format)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    logger.info("Loaded config: %s", args.config)
    logger.info("Experiment: %s", cfg.get("experiment", {}).get("name", "unknown"))

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info("Random seed set to %d", seed)

    if args.dry_run:
        logger.info("Dry run — config loaded successfully. Exiting.")
        return

    logger.info("=" * 60)
    logger.info("SFT Training Pipeline")
    logger.info("=" * 60)

    # --- Data loading ---
    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    logger.info("Train data config: %s (split=%s)", train_cfg.get("config", "N/A"), train_cfg.get("split", "N/A"))

    # --- Model loading ---
    model_cfg = cfg.get("model", {})
    logger.info("Model config: %s", model_cfg.get("config", "N/A"))
    logger.warning(
        "[Scaffold] Model loading not yet implemented. "
        "Please download model weights and update the config."
    )

    # --- Trainer ---
    from gui_grounding.training.trainer_sft import SFTTrainer

    training_cfg = cfg.get("training", {})
    trainer = SFTTrainer(
        output_dir=training_cfg.get("output_dir", "outputs/sft"),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        num_epochs=training_cfg.get("num_epochs", 3),
        batch_size=training_cfg.get("batch_size", 4),
        seed=seed,
    )

    result = trainer.train()
    logger.info("Training result: %s", result)
    logger.info("Done.")


if __name__ == "__main__":
    main()
