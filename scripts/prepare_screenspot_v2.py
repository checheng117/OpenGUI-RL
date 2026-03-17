#!/usr/bin/env python3
"""Download and preprocess the ScreenSpot-v2 dataset.

Usage:
    python scripts/prepare_screenspot_v2.py [--output-dir data/raw/screenspot_v2]

TODO(stage-2): Implement actual download and preprocessing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.logger import get_logger

logger = get_logger("prepare_screenspot_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ScreenSpot-v2 dataset")
    parser.add_argument("--output-dir", type=str, default="data/raw/screenspot_v2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("ScreenSpot-v2 Data Preparation")
    logger.info("=" * 60)
    logger.info("Output directory: %s", output_dir)

    if args.dry_run:
        logger.info("Dry run. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    logger.warning(
        "[Scaffold] Actual download not yet implemented.\n"
        "To proceed manually:\n"
        "  1. Download ScreenSpot-v2 from the official release\n"
        "  2. Place images in: %s/images/\n"
        "  3. Create annotation file: %s/test.jsonl",
        output_dir,
        output_dir,
    )

    logger.info("Done (scaffold mode).")


if __name__ == "__main__":
    main()
