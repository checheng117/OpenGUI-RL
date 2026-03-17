"""Error analysis utilities for GUI grounding predictions.

Categorizes errors and produces summaries useful for ablation studies
and qualitative case analysis.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from gui_grounding.reward.verifiable_reward import bbox_iou
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


def categorize_error(
    pred_bbox: Optional[tuple[float, float, float, float]],
    gt_bbox: Optional[tuple[float, float, float, float]],
    pred_action: Optional[str],
    gt_action: Optional[str],
    pred_element_id: Optional[str] = None,
    gt_element_id: Optional[str] = None,
    iou_threshold: float = 0.5,
) -> str:
    """Classify a prediction into an error category.

    Returns one of:
    - ``"correct"``
    - ``"wrong_element"``
    - ``"low_iou"``
    - ``"wrong_action"``
    - ``"missing_prediction"``
    - ``"multiple_errors"``
    """
    issues = []

    if pred_bbox is None and gt_bbox is not None:
        return "missing_prediction"

    if gt_element_id and pred_element_id and pred_element_id != gt_element_id:
        issues.append("wrong_element")

    if pred_bbox is not None and gt_bbox is not None:
        if bbox_iou(pred_bbox, gt_bbox) < iou_threshold:
            issues.append("low_iou")

    if pred_action and gt_action and pred_action.lower() != gt_action.lower():
        issues.append("wrong_action")

    if not issues:
        return "correct"
    if len(issues) == 1:
        return issues[0]
    return "multiple_errors"


def error_summary(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce an error distribution summary.

    TODO(stage-2): Add per-website, per-element-type, text-vs-icon breakdowns.
    """
    categories = []
    for pred, gt in zip(predictions, ground_truths):
        cat = categorize_error(
            pred_bbox=pred.get("bbox"),
            gt_bbox=gt.get("bbox"),
            pred_action=pred.get("action_type"),
            gt_action=gt.get("action_type"),
            pred_element_id=pred.get("element_id"),
            gt_element_id=gt.get("element_id"),
        )
        categories.append(cat)

    counts = Counter(categories)
    total = len(categories) or 1

    return {
        "total_samples": len(categories),
        "category_counts": dict(counts),
        "category_rates": {k: v / total for k, v in counts.items()},
        "accuracy": counts.get("correct", 0) / total,
    }
