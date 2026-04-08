"""Collapse diagnostics for Stage-A localization predictions."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def _rounded_tuple(values: list[float] | tuple[float, ...], decimals: int = 4) -> tuple[float, ...]:
    return tuple(round(float(v), decimals) for v in values)


def _top_counter_entry(counter: Counter[tuple[float, ...]], total: int) -> dict[str, Any] | None:
    if not counter or total <= 0:
        return None
    value, count = counter.most_common(1)[0]
    return {
        "value": list(value),
        "count": int(count),
        "fraction": float(count) / float(total),
    }


def _basic_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _bbox_area_ratio(
    bbox: list[float] | tuple[float, float, float, float] | None,
    image_size: tuple[int, int] | None,
) -> float | None:
    if bbox is None or image_size is None:
        return None
    width = max(float(image_size[0]), 1.0)
    height = max(float(image_size[1]), 1.0)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(width * height, 1.0)


def _dominant_click_band_fraction(
    clicks: list[tuple[float, float]],
    dominant_click: tuple[float, float] | None,
    tolerance: float = 20.0,
) -> float:
    if not clicks or dominant_click is None:
        return 0.0
    hits = 0
    for click_x, click_y in clicks:
        if abs(click_x - dominant_click[0]) <= tolerance and abs(click_y - dominant_click[1]) <= tolerance:
            hits += 1
    return float(hits) / float(len(clicks))


def compute_prediction_collapse_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated-template and tiny-box localization collapse patterns."""

    image_size_cache: dict[str, tuple[int, int] | None] = {}
    pred_bbox_counter: Counter[tuple[float, ...]] = Counter()
    pred_click_counter: Counter[tuple[float, ...]] = Counter()
    pred_clicks: list[tuple[float, float]] = []
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    pred_area_ratios: list[float] = []
    gt_area_ratios: list[float] = []

    for row in records:
        image_path = row.get("image_path")
        image_size = image_size_cache.get(str(image_path))
        if image_path is not None and str(image_path) not in image_size_cache:
            size: tuple[int, int] | None = None
            image_file = Path(str(image_path))
            if image_file.exists():
                with Image.open(image_file) as image:
                    size = image.size
            image_size_cache[str(image_path)] = size
            image_size = size

        pred_bbox = row.get("predicted_bbox")
        if isinstance(pred_bbox, list) and len(pred_bbox) == 4:
            rounded_bbox = _rounded_tuple(pred_bbox)
            pred_bbox_counter[rounded_bbox] += 1
            bbox_widths.append(float(pred_bbox[2]) - float(pred_bbox[0]))
            bbox_heights.append(float(pred_bbox[3]) - float(pred_bbox[1]))
            pred_area_ratio = _bbox_area_ratio(pred_bbox, image_size)
            if pred_area_ratio is not None:
                pred_area_ratios.append(pred_area_ratio)

        gt_bbox = row.get("target_bbox")
        if isinstance(gt_bbox, list) and len(gt_bbox) == 4:
            gt_area_ratio = _bbox_area_ratio(gt_bbox, image_size)
            if gt_area_ratio is not None:
                gt_area_ratios.append(gt_area_ratio)

        pred_click = row.get("predicted_click_point")
        if isinstance(pred_click, list) and len(pred_click) == 2:
            rounded_click = _rounded_tuple(pred_click)
            pred_click_counter[rounded_click] += 1
            pred_clicks.append((float(pred_click[0]), float(pred_click[1])))

    top_bbox = _top_counter_entry(pred_bbox_counter, max(sum(pred_bbox_counter.values()), 1))
    top_click = _top_counter_entry(pred_click_counter, max(sum(pred_click_counter.values()), 1))
    dominant_click = tuple(top_click["value"]) if top_click is not None else None
    dominant_bbox_fraction = float(top_bbox["fraction"]) if top_bbox is not None else 0.0
    dominant_click_fraction = float(top_click["fraction"]) if top_click is not None else 0.0

    return {
        "num_records": len(records),
        "num_valid_bbox": int(sum(pred_bbox_counter.values())),
        "num_valid_click_point": int(sum(pred_click_counter.values())),
        "unique_bbox_count": int(len(pred_bbox_counter)),
        "unique_click_point_count": int(len(pred_click_counter)),
        "dominant_bbox": top_bbox,
        "dominant_click_point": top_click,
        "dominant_bbox_fraction": dominant_bbox_fraction,
        "dominant_click_point_fraction": dominant_click_fraction,
        "dominant_click_point_band_fraction_20px": _dominant_click_band_fraction(
            pred_clicks,
            dominant_click=dominant_click if dominant_click is None else (float(dominant_click[0]), float(dominant_click[1])),
            tolerance=20.0,
        ),
        "predicted_bbox_width": _basic_stats(bbox_widths),
        "predicted_bbox_height": _basic_stats(bbox_heights),
        "predicted_bbox_area_ratio": _basic_stats(pred_area_ratios),
        "target_bbox_area_ratio": _basic_stats(gt_area_ratios),
        "collapse_score": max(dominant_bbox_fraction, dominant_click_fraction),
    }
