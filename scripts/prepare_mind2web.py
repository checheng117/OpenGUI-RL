#!/usr/bin/env python3
"""Download and preprocess the Multimodal-Mind2Web dataset.

Usage:
    python scripts/prepare_mind2web.py [--output-dir data/raw/mind2web]

TODO(stage-2): Implement actual download and preprocessing:
- Download from HuggingFace (osunlp/Multimodal-Mind2Web)
- Extract screenshots and annotations
- Convert to canonical JSONL format expected by Mind2WebDataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.logger import get_logger

logger = get_logger("prepare_mind2web")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Mind2Web dataset")
    parser.add_argument("--output-dir", type=str, default="data/raw/mind2web")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Mind2Web Data Preparation")
    logger.info("=" * 60)
    logger.info("Output directory: %s", output_dir)

    if args.dry_run:
        logger.info("Dry run — would prepare data at %s. Exiting.", output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screenshots").mkdir(exist_ok=True)

    logger.warning(
        "[Scaffold] Actual download not yet implemented.\n"
        "To proceed manually:\n"
        "  1. Install: pip install datasets\n"
        "  2. Download: datasets.load_dataset('osunlp/Multimodal-Mind2Web')\n"
        "  3. Place screenshots in: %s/screenshots/\n"
        "  4. Create split files: train.jsonl, test_task.jsonl, etc.",
        output_dir,
    )

    logger.info("Done (scaffold mode).")


if __name__ == "__main__":
    main()
