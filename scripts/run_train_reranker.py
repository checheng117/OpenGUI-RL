#!/usr/bin/env python3
"""Stage B: Train the reward-based reranker.

Usage:
    python scripts/run_train_reranker.py --config configs/train/rerank_reward.yaml
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

logger = get_logger("run_train_reranker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reward-based reranker")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    logger.info("Loaded config: %s", args.config)

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    if args.dry_run:
        logger.info("Dry run — config loaded successfully. Exiting.")
        return

    logger.info("=" * 60)
    logger.info("Reranker Training Pipeline")
    logger.info("=" * 60)

    from gui_grounding.models.candidate_scorer import CandidateScorer
    from gui_grounding.training.trainer_reranker import RerankerTrainer

    scorer = CandidateScorer(scoring_mode="learned")

    training_cfg = cfg.get("training", {})
    trainer = RerankerTrainer(
        scorer=scorer,
        output_dir=training_cfg.get("output_dir", "outputs/reranker"),
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        num_epochs=training_cfg.get("num_epochs", 5),
        seed=seed,
    )

    result = trainer.train()
    logger.info("Training result: %s", result)
    logger.info("Done.")


if __name__ == "__main__":
    main()
