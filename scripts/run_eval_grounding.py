#!/usr/bin/env python3
"""Evaluate GUI grounding quality on a held-out dataset.

Usage:
    python scripts/run_eval_grounding.py --config configs/eval/grounding_eval.yaml
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

logger = get_logger("run_eval_grounding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GUI grounding")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    logger.info("Loaded config: %s", args.config)

    set_seed(42)

    if args.dry_run:
        logger.info("Dry run — config loaded successfully. Exiting.")
        return

    logger.info("=" * 60)
    logger.info("Grounding Evaluation Pipeline")
    logger.info("=" * 60)

    from gui_grounding.evaluation.evaluator_grounding import GroundingEvaluator
    from gui_grounding.evaluation.metrics import compute_all_metrics

    output_dir = cfg.get("output", {}).get("output_dir", "outputs/eval/grounding")

    evaluator = GroundingEvaluator(output_dir=output_dir)
    results = evaluator.run()

    logger.info("Evaluation results:")
    for k, v in results.items():
        logger.info("  %s: %s", k, v)

    logger.info("Running a quick metrics sanity check with dummy data...")
    metrics = compute_all_metrics(
        pred_element_ids=["btn_1", "btn_2", "btn_3"],
        gt_element_ids=["btn_1", "btn_2", "btn_4"],
        pred_bboxes=[(10, 20, 100, 80), (50, 50, 150, 150), (0, 0, 50, 50)],
        gt_bboxes=[(10, 20, 100, 80), (60, 60, 140, 140), (200, 200, 300, 300)],
        pred_points=[(55, 50), (100, 100), (25, 25)],
        pred_actions=["click", "type", "click"],
        gt_actions=["click", "type", "select"],
    )
    logger.info("Sanity check metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    logger.info("Done.")


if __name__ == "__main__":
    main()
