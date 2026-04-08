#!/usr/bin/env python3
"""Analyze ScreenSpot-v2 predictions and build a focused failure taxonomy."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ScreenSpot-v2 held-out predictions")
    parser.add_argument(
        "--predictions-jsonl",
        type=str,
        default="outputs/screenspot_v2_eval_qwen2_5_vl_3b/predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/screenspot_v2_eval_qwen2_5_vl_3b_analysis",
    )
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--subset-per-platform", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _point_inside_bbox(click_point, bbox) -> bool:
    if click_point is None or bbox is None:
        return False
    cx, cy = click_point
    x1, y1, x2, y2 = bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _bbox_iou(box_a, box_b) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _compute_processor_rescale(record: dict, min_pixels: int, max_pixels: int):
    with Image.open(record["image_path"]) as image:
        image_width, image_height = image.size

    resized_height, resized_width = smart_resize(
        height=image_height,
        width=image_width,
        factor=28,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    scale_x = image_width / resized_width
    scale_y = image_height / resized_height

    bbox = record.get("bbox_proposal")
    click = record.get("click_point")
    scaled_bbox = None
    scaled_click = None

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        scaled_bbox = [
            max(0.0, min(float(image_width), float(x1) * scale_x)),
            max(0.0, min(float(image_height), float(y1) * scale_y)),
            max(0.0, min(float(image_width), float(x2) * scale_x)),
            max(0.0, min(float(image_height), float(y2) * scale_y)),
        ]
        if scaled_bbox[2] < scaled_bbox[0]:
            scaled_bbox[0], scaled_bbox[2] = scaled_bbox[2], scaled_bbox[0]
        if scaled_bbox[3] < scaled_bbox[1]:
            scaled_bbox[1], scaled_bbox[3] = scaled_bbox[3], scaled_bbox[1]

    if click is not None:
        scaled_click = [
            max(0.0, min(float(image_width), float(click[0]) * scale_x)),
            max(0.0, min(float(image_height), float(click[1]) * scale_y)),
        ]

    return {
        "original_image_width": image_width,
        "original_image_height": image_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scaled_bbox": scaled_bbox,
        "scaled_click": scaled_click,
    }


def _compute_metrics(records: list[dict]) -> dict[str, float | int]:
    point_hits = 0
    iou_hits = 0
    iou_sum = 0.0
    action_valid = 0
    parseable = 0
    valid_bbox = 0
    valid_click = 0

    for record in records:
        point_hits += int(bool(record.get("point_in_box")))
        iou_hits += int(bool(record.get("iou_at_0_5")))
        iou_sum += float(record.get("iou") or 0.0)
        action_valid += int(bool(record.get("action_type_valid")))
        parseable += int(bool(record.get("json_parse_success")))
        valid_bbox += int(record.get("bbox_proposal") is not None)
        valid_click += int(record.get("click_point") is not None)

    count = len(records)
    return {
        "count": count,
        "point_accuracy": _safe_rate(point_hits, count),
        "iou@0.5": _safe_rate(iou_hits, count),
        "mean_iou": _safe_rate(iou_sum, count),
        "action_type_valid_rate": _safe_rate(action_valid, count),
        "parseable_output_rate": _safe_rate(parseable, count),
        "valid_bbox_rate": _safe_rate(valid_bbox, count),
        "valid_click_point_rate": _safe_rate(valid_click, count),
    }


def _render_markdown(report: dict) -> str:
    taxonomy = report["failure_taxonomy"]
    subset = report["diagnostic_subset"]
    lines = [
        "# ScreenSpot-v2 Failure Analysis",
        "",
        "## Dominant Findings",
        "",
        f"- Evaluated samples analyzed: `{taxonomy['evaluated_samples']}`",
        f"- Parse failures: `{taxonomy['parse_failure_count']}`",
        f"- BBox-from-click fallbacks: `{taxonomy['bbox_from_click_fallback_count']}`",
        f"- Malformed bbox payloads: `{taxonomy['malformed_bbox_payload_count']}`",
        f"- Parseable-but-spatially-wrong outputs: `{taxonomy['parseable_but_spatially_wrong_count']}`",
        f"- Click action rate: `{taxonomy['click_action_rate']:.4f}`",
        "",
        "## Coordinate-Scale Pattern",
        "",
    ]

    for platform, stats in taxonomy["coordinate_scale_by_platform"].items():
        lines.append(
            f"- `{platform}`: median click x ratio `{stats['median_click_ratio_x']:.4f}`, "
            f"median click y ratio `{stats['median_click_ratio_y']:.4f}`, "
            f"median bbox width ratio `{stats['median_bbox_width_ratio']:.4f}`, "
            f"median bbox height ratio `{stats['median_bbox_height_ratio']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Counterfactual Processor-Rescale Diagnostic",
            "",
            f"- Overall point accuracy would move from `{taxonomy['baseline_metrics']['point_accuracy']:.4f}` to "
            f"`{taxonomy['processor_rescale_counterfactual']['overall']['point_accuracy']:.4f}`",
            f"- Overall IoU@0.5 would move from `{taxonomy['baseline_metrics']['iou@0.5']:.4f}` to "
            f"`{taxonomy['processor_rescale_counterfactual']['overall']['iou@0.5']:.4f}`",
            "",
            "## Diagnostic Subset",
            "",
            f"- Subset size: `{subset['subset_size']}`",
            f"- Per-platform count: `{subset['subset_per_platform']}`",
            f"- Baseline subset point accuracy: `{subset['baseline_metrics']['overall']['point_accuracy']:.4f}`",
            f"- Baseline subset IoU@0.5: `{subset['baseline_metrics']['overall']['iou@0.5']:.4f}`",
            f"- Baseline subset mean IoU: `{subset['baseline_metrics']['overall']['mean_iou']:.4f}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_jsonl(Path(args.predictions_jsonl))
    if not predictions:
        raise ValueError("No predictions found to analyze.")

    parse_failure_count = 0
    bbox_from_click_fallback_count = 0
    malformed_bbox_payload_count = 0
    malformed_click_payload_count = 0
    parseable_but_spatially_wrong_count = 0
    action_histogram = Counter()
    image_sizes_by_platform = defaultdict(lambda: {"widths": [], "heights": []})
    scale_stats = defaultdict(lambda: {"click_ratio_x": [], "click_ratio_y": [], "bbox_ratio_w": [], "bbox_ratio_h": []})
    counterfactual_by_platform = defaultdict(list)
    improved_examples: list[dict] = []
    desktop_success_examples: list[dict] = []

    for record in predictions:
        platform = str(record.get("platform") or "__unknown__")
        element_type = str(record.get("element_type") or "__unknown__")
        parsed_payload = record.get("parsed_model_payload") or {}
        bbox_payload = parsed_payload.get("bbox_proposal", parsed_payload.get("predicted_bbox"))
        click_payload = parsed_payload.get("click_point", parsed_payload.get("predicted_click_point"))

        action_histogram[str(record.get("action_type") or "__invalid_or_missing__")] += 1
        if not record.get("json_parse_success"):
            parse_failure_count += 1
        if bbox_payload is None and click_payload is not None and record.get("bbox_proposal") is not None:
            bbox_from_click_fallback_count += 1
        if bbox_payload is not None and not (isinstance(bbox_payload, list) and len(bbox_payload) == 4):
            malformed_bbox_payload_count += 1
        if click_payload is not None and not (isinstance(click_payload, list) and len(click_payload) == 2):
            malformed_click_payload_count += 1
        if record.get("json_parse_success") and not record.get("point_in_box") and not record.get("iou_at_0_5"):
            parseable_but_spatially_wrong_count += 1

        rescale = _compute_processor_rescale(record, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        gt_bbox = record.get("target_bbox_xyxy")
        gt_click = record.get("target_click_point")
        if gt_click and record.get("click_point"):
            tx, ty = gt_click
            px, py = record["click_point"]
            if tx > 0 and ty > 0:
                scale_stats[platform]["click_ratio_x"].append(float(px) / float(tx))
                scale_stats[platform]["click_ratio_y"].append(float(py) / float(ty))
        if gt_bbox and record.get("bbox_proposal"):
            pred_bbox = record["bbox_proposal"]
            pred_w = max(1e-6, float(pred_bbox[2]) - float(pred_bbox[0]))
            pred_h = max(1e-6, float(pred_bbox[3]) - float(pred_bbox[1]))
            gt_w = max(1e-6, float(gt_bbox[2]) - float(gt_bbox[0]))
            gt_h = max(1e-6, float(gt_bbox[3]) - float(gt_bbox[1]))
            scale_stats[platform]["bbox_ratio_w"].append(pred_w / gt_w)
            scale_stats[platform]["bbox_ratio_h"].append(pred_h / gt_h)

        image_sizes_by_platform[platform]["widths"].append(rescale["original_image_width"])
        image_sizes_by_platform[platform]["heights"].append(rescale["original_image_height"])

        counterfactual_record = dict(record)
        counterfactual_record["bbox_proposal"] = rescale["scaled_bbox"]
        counterfactual_record["click_point"] = rescale["scaled_click"]
        counterfactual_record["point_in_box"] = _point_inside_bbox(rescale["scaled_click"], gt_bbox)
        counterfactual_record["iou"] = _bbox_iou(rescale["scaled_bbox"], gt_bbox)
        counterfactual_record["iou_at_0_5"] = counterfactual_record["iou"] >= 0.5
        counterfactual_by_platform[platform].append(counterfactual_record)
        counterfactual_by_platform["__all__"].append(counterfactual_record)

        if platform in {"mobile", "web"} and not record.get("point_in_box") and counterfactual_record["point_in_box"]:
            if len(improved_examples) < 12:
                improved_examples.append(
                    {
                        "dataset_index": record.get("dataset_index"),
                        "sample_id": record.get("sample_id"),
                        "platform": platform,
                        "element_type": element_type,
                        "instruction": record.get("instruction"),
                        "original_click_point": record.get("click_point"),
                        "counterfactual_click_point": rescale["scaled_click"],
                        "target_click_point": record.get("target_click_point"),
                        "scale_x": rescale["scale_x"],
                        "scale_y": rescale["scale_y"],
                    }
                )

        if platform == "desktop" and record.get("point_in_box") and len(desktop_success_examples) < 8:
            desktop_success_examples.append(
                {
                    "dataset_index": record.get("dataset_index"),
                    "sample_id": record.get("sample_id"),
                    "instruction": record.get("instruction"),
                    "click_point": record.get("click_point"),
                    "target_click_point": record.get("target_click_point"),
                    "iou": record.get("iou"),
                    "element_type": element_type,
                    "data_source": record.get("data_source"),
                }
            )

    baseline_metrics = _compute_metrics(predictions)
    processor_rescale_counterfactual = {
        "overall": _compute_metrics(counterfactual_by_platform["__all__"]),
        "platform": {
            platform: _compute_metrics(platform_records)
            for platform, platform_records in counterfactual_by_platform.items()
            if platform != "__all__"
        },
    }

    coordinate_scale_by_platform = {}
    for platform, stats in scale_stats.items():
        coordinate_scale_by_platform[platform] = {
            "median_click_ratio_x": _median_or_none(stats["click_ratio_x"]),
            "median_click_ratio_y": _median_or_none(stats["click_ratio_y"]),
            "median_bbox_width_ratio": _median_or_none(stats["bbox_ratio_w"]),
            "median_bbox_height_ratio": _median_or_none(stats["bbox_ratio_h"]),
            "median_image_width": _median_or_none(image_sizes_by_platform[platform]["widths"]),
            "median_image_height": _median_or_none(image_sizes_by_platform[platform]["heights"]),
        }

    rng = random.Random(args.seed)
    records_by_platform = defaultdict(list)
    for record in predictions:
        records_by_platform[str(record.get("platform") or "__unknown__")].append(record)

    diagnostic_subset_indices: list[int] = []
    diagnostic_subset_platform_counts: dict[str, int] = {}
    for platform in ("desktop", "web", "mobile"):
        platform_records = records_by_platform[platform]
        sample_count = min(args.subset_per_platform, len(platform_records))
        sampled = rng.sample(platform_records, sample_count)
        diagnostic_subset_indices.extend(int(record["dataset_index"]) for record in sampled)
        diagnostic_subset_platform_counts[platform] = sample_count
    diagnostic_subset_indices.sort()

    diagnostic_subset_records = [
        record for record in predictions if int(record["dataset_index"]) in set(diagnostic_subset_indices)
    ]
    diagnostic_subset_baseline = {
        "overall": _compute_metrics(diagnostic_subset_records),
        "platform": {
            platform: _compute_metrics([record for record in diagnostic_subset_records if record["platform"] == platform])
            for platform in ("desktop", "web", "mobile")
        },
    }

    report = {
        "failure_taxonomy": {
            "evaluated_samples": len(predictions),
            "baseline_metrics": baseline_metrics,
            "parse_failure_count": parse_failure_count,
            "bbox_from_click_fallback_count": bbox_from_click_fallback_count,
            "malformed_bbox_payload_count": malformed_bbox_payload_count,
            "malformed_click_payload_count": malformed_click_payload_count,
            "parseable_but_spatially_wrong_count": parseable_but_spatially_wrong_count,
            "click_action_rate": _safe_rate(action_histogram["click"], len(predictions)),
            "action_type_distribution": dict(action_histogram),
            "coordinate_scale_by_platform": coordinate_scale_by_platform,
            "processor_rescale_counterfactual": processor_rescale_counterfactual,
            "desktop_success_examples": desktop_success_examples,
            "scale_mismatch_examples": improved_examples,
        },
        "diagnostic_subset": {
            "subset_size": len(diagnostic_subset_indices),
            "subset_per_platform": args.subset_per_platform,
            "platform_counts": diagnostic_subset_platform_counts,
            "indices_path": str(output_dir / "diagnostic_subset_balanced_indices.json"),
            "baseline_metrics": diagnostic_subset_baseline,
            "selected_indices": diagnostic_subset_indices,
        },
    }

    (output_dir / "diagnostic_subset_balanced_indices.json").write_text(
        json.dumps(diagnostic_subset_indices, indent=2),
        encoding="utf-8",
    )
    (output_dir / "failure_taxonomy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "failure_taxonomy.md").write_text(_render_markdown(report), encoding="utf-8")
    (output_dir / "diagnostic_subset_baseline_summary.json").write_text(
        json.dumps(diagnostic_subset_baseline, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
