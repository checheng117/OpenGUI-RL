#!/usr/bin/env python3
"""Launch the interactive Gradio demo.

Usage:
    python scripts/run_demo.py --config configs/demo/demo.yaml
    python scripts/run_demo.py --share  # publicly accessible link
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.config import load_config
from gui_grounding.utils.logger import get_logger

logger = get_logger("run_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch GUI grounding demo")
    parser.add_argument("--config", type=str, default="configs/demo/demo.yaml")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger.info("Loaded config: %s", args.config)

    if args.dry_run:
        logger.info("Dry run — config loaded successfully. Exiting.")
        return

    server_cfg = cfg.get("demo", {}).get("server", cfg.get("server", {}))
    port = args.port or server_cfg.get("port", 7860)
    share = args.share or server_cfg.get("share", False)

    logger.info("Launching demo on port %d (share=%s)", port, share)

    try:
        from gui_grounding.demo.app import launch_demo
        launch_demo(share=share, port=port)
    except ImportError:
        logger.error(
            "Gradio is not installed. Please install it:\n"
            "  pip install gradio>=4.20.0"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
