#!/usr/bin/env python3
"""Stage B: Generate candidate actions for reranking.

Usage:
    python scripts/run_generate_candidates.py --config configs/train/rerank_reward.yaml
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

logger = get_logger("run_generate_candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate generation for reranking")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples")
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
    logger.info("Candidate Generation Pipeline")
    logger.info("=" * 60)

    cand_cfg = cfg.get("candidate_generation", {})
    mode = cand_cfg.get("mode", "dummy")
    num_candidates = cand_cfg.get("num_candidates", 8)
    logger.info("Candidate mode: %s, K=%d", mode, num_candidates)

    from gui_grounding.reward.candidate_generator import CandidateGenerator

    generator = CandidateGenerator(mode="dummy", num_candidates=num_candidates, seed=seed)

    from gui_grounding.data.schemas import BBox, GroundingSample

    dummy_sample = GroundingSample(
        sample_id="demo_001",
        dataset_name="mind2web",
        split="train",
        image_path="data/raw/mind2web/screenshots/demo.png",
        instruction="Click the search button",
        target_bbox=BBox(x1=100, y1=200, x2=250, y2=240),
        action_type="click",
    )

    candidates = generator.generate(dummy_sample)
    logger.info("Generated %d candidates for sample '%s':", len(candidates), dummy_sample.sample_id)
    for c in candidates:
        logger.info("  %s: bbox=%s, action=%s", c.candidate_id, c.bbox, c.action_type)

    from gui_grounding.reward.verifiable_reward import VerifiableRewardCalculator

    reward_weights = cfg.get("reward", {}).get("weights", None)
    calculator = VerifiableRewardCalculator(weights=reward_weights)

    for c in candidates:
        result = calculator.compute(
            sample_id=c.candidate_id,
            pred_bbox=c.bbox.as_tuple() if c.bbox else None,
            gt_bbox=dummy_sample.target_bbox.as_tuple() if dummy_sample.target_bbox else None,
            pred_click=c.click_point,
            pred_action=c.action_type,
            gt_action=dummy_sample.action_type,
            pred_element_id=c.element_id,
            gt_element_id=dummy_sample.target_element_id,
        )
        logger.info("  %s reward=%.4f (elem=%.1f, iou=%.3f, click=%.1f, act=%.1f)",
                     c.candidate_id, result.total_reward,
                     result.components.element_correct, result.components.iou,
                     result.components.click_inside_target, result.components.action_type_correct)

    logger.info("Done. In a real run, results would be saved for reranker training.")


if __name__ == "__main__":
    main()
