#!/usr/bin/env python3
"""Cross-website transfer evaluation.

Usage:
    python scripts/run_eval_transfer.py --config configs/eval/transfer_eval.yaml
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

logger = get_logger("run_eval_transfer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-website transfer evaluation")
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
    logger.info("Transfer Evaluation Pipeline")
    logger.info("=" * 60)

    from gui_grounding.evaluation.evaluator_transfer import TransferEvaluator

    output_dir = cfg.get("output", {}).get("output_dir", "outputs/eval/transfer")
    evaluator = TransferEvaluator(output_dir=output_dir)
    results = evaluator.run()

    logger.info("Transfer evaluation results:")
    for split_name, split_results in results.items():
        logger.info("  [%s] %s", split_name, split_results)

    logger.info("Done.")


if __name__ == "__main__":
    main()
