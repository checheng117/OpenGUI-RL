"""VisualWebBench grounding evaluation helpers."""

from __future__ import annotations

import math
from typing import Any, Sequence

from gui_grounding.constants import ACTION_TYPES
from gui_grounding.reward.verifiable_reward import bbox_iou

CHOICE_LABELS = tuple("ABCDEFGH")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or len(value) != 2:
        return None
    x = _safe_float(value[0])
    y = _safe_float(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def _as_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or len(value) != 4:
        return None
    coords = [_safe_float(v) for v in value]
    if any(v is None for v in coords):
        return None
    x1, y1, x2, y2 = coords  # type: ignore[misc]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _point_inside_bbox(
    point: tuple[float, float] | None,
    bbox: tuple[float, float, float, float] | None,
) -> bool:
    if point is None or bbox is None:
        return False
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def _choice_label(index: int | None) -> str | None:
    if index is None or not (0 <= index < len(CHOICE_LABELS)):
        return None
    return CHOICE_LABELS[index]


def _target_area_bucket(area_ratio: float) -> str:
    if area_ratio < 0.005:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.08:
        return "medium"
    return "large"


def _distractor_overlap_bucket(max_distractor_iou: float) -> str:
    if max_distractor_iou < 0.1:
        return "low"
    if max_distractor_iou < 0.3:
        return "medium"
    return "high"


def map_prediction_to_choice(
    *,
    predicted_bbox: Any,
    predicted_click_point: Any,
    candidate_boxes: Sequence[Any] | None,
) -> tuple[int | None, dict[str, Any]]:
    candidates = [_as_bbox(candidate_box) for candidate_box in list(candidate_boxes or [])]
    valid_candidates = [(idx, bbox) for idx, bbox in enumerate(candidates) if bbox is not None]
    candidate_lookup = {idx: bbox for idx, bbox in valid_candidates}
    if not valid_candidates:
        return None, {"mode": "missing_candidates"}

    pred_click = _as_point(predicted_click_point)
    pred_bbox = _as_bbox(predicted_bbox)

    if pred_click is not None:
        containing = [
            idx
            for idx, candidate_bbox in valid_candidates
            if _point_inside_bbox(pred_click, candidate_bbox)
        ]
        if len(containing) == 1:
            return containing[0], {"mode": "point_inside_single", "containing_candidates": containing}
        if len(containing) > 1:
            if pred_bbox is not None:
                best_idx = max(
                    containing,
                    key=lambda idx: bbox_iou(pred_bbox, candidate_lookup[idx]),
                )
                return best_idx, {
                    "mode": "point_inside_multiple_iou_tiebreak",
                    "containing_candidates": containing,
                }
            best_idx = min(
                containing,
                key=lambda idx: _bbox_area(candidate_lookup[idx]),
            )
            return best_idx, {
                "mode": "point_inside_multiple_smallest_area",
                "containing_candidates": containing,
            }

    if pred_bbox is not None:
        candidate_ious = [
            (idx, bbox_iou(pred_bbox, candidate_bbox))
            for idx, candidate_bbox in valid_candidates
        ]
        best_idx, best_iou = max(candidate_ious, key=lambda item: item[1])
        if best_iou > 0.0:
            return best_idx, {"mode": "bbox_max_iou", "max_iou": best_iou}

    reference_point = pred_click
    if reference_point is None and pred_bbox is not None:
        reference_point = _bbox_center(pred_bbox)
    if reference_point is not None:
        distances = [
            (idx, math.dist(reference_point, _bbox_center(candidate_bbox)))
            for idx, candidate_bbox in valid_candidates
        ]
        best_idx, best_distance = min(distances, key=lambda item: item[1])
        return best_idx, {"mode": "nearest_center", "center_distance": best_distance}

    return None, {"mode": "missing_prediction"}


def score_visualwebbench_grounding(
    *,
    predicted_bbox: Any,
    predicted_click_point: Any,
    predicted_action_type: Any,
    candidate_boxes: Sequence[Any] | None,
    target_choice_index: int | None,
    image_size: Sequence[Any] | None,
    task_type: str | None,
    website: str | None,
    predicted_candidate_slot: Any = None,
    candidate_slot_grounded: bool = False,
) -> dict[str, Any]:
    pred_bbox = _as_bbox(predicted_bbox)
    pred_click = _as_point(predicted_click_point)
    candidates = [_as_bbox(candidate_box) for candidate_box in list(candidate_boxes or [])]
    target_bbox = (
        candidates[target_choice_index]
        if target_choice_index is not None and 0 <= target_choice_index < len(candidates)
        else None
    )
    predicted_choice_index, mapping_details = map_prediction_to_choice(
        predicted_bbox=pred_bbox,
        predicted_click_point=pred_click,
        candidate_boxes=candidates,
    )

    iou = bbox_iou(pred_bbox, target_bbox) if pred_bbox is not None and target_bbox is not None else 0.0
    point_hit = _point_inside_bbox(pred_click, target_bbox)
    action = str(predicted_action_type).strip().lower() if predicted_action_type is not None else None
    action_valid = action in ACTION_TYPES if action is not None else False

    image_width = _safe_float(image_size[0]) if isinstance(image_size, Sequence) and len(image_size) >= 1 else None
    image_height = _safe_float(image_size[1]) if isinstance(image_size, Sequence) and len(image_size) >= 2 else None
    if target_bbox is not None and image_width and image_height:
        area_ratio = _bbox_area(target_bbox) / max(image_width * image_height, 1.0)
    else:
        area_ratio = 0.0
    distractor_ious = [
        bbox_iou(target_bbox, candidate_bbox)
        for idx, candidate_bbox in enumerate(candidates)
        if candidate_bbox is not None and target_bbox is not None and idx != target_choice_index
    ]
    max_distractor_iou = max(distractor_ious) if distractor_ious else 0.0

    return {
        "task_type": task_type,
        "website": website,
        "target_choice_index": target_choice_index,
        "target_choice_label": _choice_label(target_choice_index),
        "predicted_choice_index": predicted_choice_index,
        "predicted_choice_label": _choice_label(predicted_choice_index),
        "official_choice_correct": (
            predicted_choice_index is not None
            and target_choice_index is not None
            and predicted_choice_index == target_choice_index
        ),
        "choice_mapping_mode": mapping_details.get("mode"),
        "choice_mapping_details": mapping_details,
        "target_bbox_xyxy": list(target_bbox) if target_bbox is not None else None,
        "point_in_box": point_hit,
        "iou": iou,
        "iou_at_0_5": iou >= 0.5,
        "action_type_valid": action_valid,
        "target_area_ratio": area_ratio,
        "target_area_bucket": _target_area_bucket(area_ratio),
        "max_distractor_iou": max_distractor_iou,
        "distractor_overlap_bucket": _distractor_overlap_bucket(max_distractor_iou),
        "predicted_choice_valid": predicted_choice_index is not None,
        "predicted_candidate_slot": predicted_candidate_slot,
        "candidate_slot_predicted": predicted_candidate_slot is not None,
        "candidate_slot_grounded": bool(candidate_slot_grounded),
    }


def empty_visualwebbench_metrics_bucket() -> dict[str, int | float]:
    return {
        "count": 0,
        "run_success_count": 0,
        "valid_bbox_count": 0,
        "valid_click_point_count": 0,
        "valid_action_type_count": 0,
        "parseable_output_count": 0,
        "predicted_choice_count": 0,
        "official_choice_hits": 0,
        "point_accuracy_hits": 0,
        "iou_at_0_5_hits": 0,
        "iou_sum": 0.0,
        "candidate_slot_prediction_count": 0,
        "candidate_slot_grounding_count": 0,
    }


def update_visualwebbench_metrics_bucket(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    bucket["count"] += 1
    bucket["run_success_count"] += int(record.get("status") == "ok")
    bucket["valid_bbox_count"] += int(record.get("bbox_proposal") is not None)
    bucket["valid_click_point_count"] += int(record.get("click_point") is not None)
    bucket["valid_action_type_count"] += int(bool(record.get("action_type_valid")))
    bucket["parseable_output_count"] += int(bool(record.get("json_parse_success")))
    bucket["predicted_choice_count"] += int(bool(record.get("predicted_choice_valid")))
    bucket["official_choice_hits"] += int(bool(record.get("official_choice_correct")))
    bucket["point_accuracy_hits"] += int(bool(record.get("point_in_box")))
    bucket["iou_at_0_5_hits"] += int(bool(record.get("iou_at_0_5")))
    bucket["iou_sum"] += float(record.get("iou") or 0.0)
    bucket["candidate_slot_prediction_count"] += int(bool(record.get("candidate_slot_predicted")))
    bucket["candidate_slot_grounding_count"] += int(bool(record.get("candidate_slot_grounded")))


def finalize_visualwebbench_metrics(bucket: dict[str, Any]) -> dict[str, int | float]:
    count = int(bucket["count"])

    def _safe_rate(num: int | float, den: int | float) -> float:
        return float(num) / float(den) if den else 0.0

    return {
        "count": count,
        "run_success_rate": _safe_rate(bucket["run_success_count"], count),
        "valid_bbox_rate": _safe_rate(bucket["valid_bbox_count"], count),
        "valid_click_point_rate": _safe_rate(bucket["valid_click_point_count"], count),
        "action_type_valid_rate": _safe_rate(bucket["valid_action_type_count"], count),
        "parseable_output_rate": _safe_rate(bucket["parseable_output_count"], count),
        "predicted_choice_rate": _safe_rate(bucket["predicted_choice_count"], count),
        "official_choice_accuracy": _safe_rate(bucket["official_choice_hits"], count),
        "point_accuracy": _safe_rate(bucket["point_accuracy_hits"], count),
        "iou@0.5": _safe_rate(bucket["iou_at_0_5_hits"], count),
        "mean_iou": _safe_rate(bucket["iou_sum"], count),
        "candidate_slot_prediction_rate": _safe_rate(bucket["candidate_slot_prediction_count"], count),
        "candidate_slot_grounding_rate": _safe_rate(bucket["candidate_slot_grounding_count"], count),
    }


def aggregate_visualwebbench_records(
    records: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    overall_bucket = empty_visualwebbench_metrics_bucket()
    subgroup_buckets: dict[str, dict[str, dict[str, Any]]] = {
        field: {} for field in group_fields
    }

    for record in records:
        update_visualwebbench_metrics_bucket(overall_bucket, record)
        for field in group_fields:
            field_value = record.get(field)
            field_key = "unknown" if field_value in {None, ""} else str(field_value)
            bucket = subgroup_buckets[field].setdefault(
                field_key,
                empty_visualwebbench_metrics_bucket(),
            )
            update_visualwebbench_metrics_bucket(bucket, record)

    overall = finalize_visualwebbench_metrics(overall_bucket)
    subgroup_metrics = {
        field: {
            field_value: finalize_visualwebbench_metrics(bucket)
            for field_value, bucket in value_to_bucket.items()
        }
        for field, value_to_bucket in subgroup_buckets.items()
    }
    return overall, subgroup_metrics


def render_visualwebbench_summary_table(
    *,
    title: str,
    overall: dict[str, Any],
    subgroup_metrics: dict[str, dict[str, Any]],
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Evaluated samples | {overall['count']} |",
        f"| Official choice accuracy | {overall['official_choice_accuracy']:.4f} |",
        f"| Point accuracy | {overall['point_accuracy']:.4f} |",
        f"| IoU@0.5 | {overall['iou@0.5']:.4f} |",
        f"| Mean IoU | {overall['mean_iou']:.4f} |",
        f"| Predicted choice rate | {overall['predicted_choice_rate']:.4f} |",
        f"| Parseable output rate | {overall['parseable_output_rate']:.4f} |",
        f"| Valid bbox rate | {overall['valid_bbox_rate']:.4f} |",
        f"| Valid click point rate | {overall['valid_click_point_rate']:.4f} |",
        f"| Action type valid rate | {overall['action_type_valid_rate']:.4f} |",
        f"| Candidate-slot prediction rate | {overall['candidate_slot_prediction_rate']:.4f} |",
        f"| Candidate-slot grounding rate | {overall['candidate_slot_grounding_rate']:.4f} |",
        "",
    ]

    for field, metrics_by_value in subgroup_metrics.items():
        lines.extend(
            [
                f"## {field.replace('_', ' ').title()}",
                "",
                "| Group | Count | Choice Acc | Point Acc | IoU@0.5 | Mean IoU |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for field_value, metrics in sorted(metrics_by_value.items()):
            lines.append(
                f"| {field_value} | {metrics['count']} | {metrics['official_choice_accuracy']:.4f} | "
                f"{metrics['point_accuracy']:.4f} | {metrics['iou@0.5']:.4f} | {metrics['mean_iou']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)
