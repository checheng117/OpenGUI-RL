"""Visualization helpers for screenshots, bounding boxes, and click points."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bbox(
    image: Image.Image,
    bbox_xyxy: Sequence[float],
    color: str = "red",
    width: int = 3,
    label: Optional[str] = None,
) -> Image.Image:
    """Draw a bounding box on a PIL Image (returns a copy)."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        draw.text((x1, max(0, y1 - 16)), label, fill=color, font=font)
    return img


def draw_click_point(
    image: Image.Image,
    point_xy: Sequence[float],
    color: str = "lime",
    radius: int = 8,
) -> Image.Image:
    """Draw a click-point marker on a PIL Image (returns a copy)."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y = point_xy
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline="black")
    return img


def draw_prediction(
    image: Image.Image,
    pred_bbox: Optional[Sequence[float]] = None,
    gt_bbox: Optional[Sequence[float]] = None,
    pred_point: Optional[Sequence[float]] = None,
    gt_point: Optional[Sequence[float]] = None,
    action_type: Optional[str] = None,
) -> Image.Image:
    """Overlay prediction and ground-truth annotations on a screenshot."""
    img = image.copy()
    if gt_bbox is not None:
        img = draw_bbox(img, gt_bbox, color="green", width=2, label="GT")
    if pred_bbox is not None:
        img = draw_bbox(img, pred_bbox, color="red", width=2, label="Pred")
    if gt_point is not None:
        img = draw_click_point(img, gt_point, color="lime", radius=6)
    if pred_point is not None:
        img = draw_click_point(img, pred_point, color="red", radius=6)
    if action_type:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Action: {action_type}", fill="yellow")
    return img


def save_visualization(image: Image.Image, path: str | Path) -> None:
    """Save a PIL Image to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
