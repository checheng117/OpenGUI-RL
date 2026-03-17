"""Preprocessing utilities for images and text."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


def load_screenshot(path: str | Path, max_size: int = 1344) -> Optional[Image.Image]:
    """Load and optionally resize a screenshot.

    Parameters
    ----------
    path : str | Path
        Path to the image file.
    max_size : int
        If the longer side exceeds this, resize proportionally.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Screenshot not found: %s", path)
        return None

    img = Image.open(path).convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def normalize_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Normalize pixel coordinates to [0, 1] range."""
    x1, y1, x2, y2 = bbox_xyxy
    return (
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height,
    )


def denormalize_bbox(
    bbox_norm: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert [0, 1] coordinates back to pixel space."""
    x1, y1, x2, y2 = bbox_norm
    return (
        x1 * image_width,
        y1 * image_height,
        x2 * image_width,
        y2 * image_height,
    )


def format_instruction_prompt(
    instruction: str,
    action_type: Optional[str] = None,
    include_candidates: bool = False,
    candidate_texts: Optional[list[str]] = None,
) -> str:
    """Build a text prompt for the VLM.

    TODO(stage-2): Finalize prompt templates after baseline experiments.
    """
    parts = [
        "Given the screenshot, locate the UI element described by the instruction "
        "and predict the next action.",
        f"\nInstruction: {instruction}",
    ]

    if include_candidates and candidate_texts:
        parts.append("\nCandidate elements:")
        for i, text in enumerate(candidate_texts):
            parts.append(f"  [{i}] {text}")

    parts.append("\nOutput the bounding box [x1, y1, x2, y2] and action type.")
    return "\n".join(parts)
