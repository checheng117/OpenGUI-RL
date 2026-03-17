"""Verifiable reward calculator for GUI grounding.

Implements the composite reward function:

    r = λ1·element_correct + λ2·IoU(b, b*) + λ3·click_inside_target
        + λ4·action_type_correct − λ5·invalid_format

All component functions are deterministic and verifiable against
ground-truth annotations.
"""

from __future__ import annotations

from typing import Optional

from gui_grounding.constants import DEFAULT_REWARD_WEIGHTS
from gui_grounding.reward.reward_schema import RewardComponents, RewardResult


# -----------------------------------------------------------------------
# Individual reward component functions
# -----------------------------------------------------------------------

def element_correct(
    pred_element_id: Optional[str],
    gt_element_id: Optional[str],
) -> float:
    """Binary reward: 1.0 if predicted element matches ground truth."""
    if pred_element_id is None or gt_element_id is None:
        return 0.0
    return 1.0 if pred_element_id == gt_element_id else 0.0


def bbox_iou(
    pred_bbox: tuple[float, float, float, float],
    gt_bbox: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union between two bounding boxes.

    Both boxes are in (x1, y1, x2, y2) format.
    """
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union_area = pred_area + gt_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def click_inside_target(
    pred_click: tuple[float, float],
    gt_bbox: tuple[float, float, float, float],
) -> float:
    """Binary reward: 1.0 if predicted click point is inside the GT box."""
    cx, cy = pred_click
    x1, y1, x2, y2 = gt_bbox
    return 1.0 if (x1 <= cx <= x2 and y1 <= cy <= y2) else 0.0


def action_type_correct(
    pred_action: Optional[str],
    gt_action: Optional[str],
) -> float:
    """Binary reward: 1.0 if predicted action type matches ground truth."""
    if pred_action is None or gt_action is None:
        return 0.0
    return 1.0 if pred_action.lower() == gt_action.lower() else 0.0


def invalid_format_penalty(
    pred_bbox: Optional[tuple[float, float, float, float]] = None,
    pred_click: Optional[tuple[float, float]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> float:
    """Penalty for invalid prediction format.

    Returns 1.0 (penalty) if the prediction is malformed:
    - bbox has x2 < x1 or y2 < y1
    - coordinates are negative
    - coordinates exceed image dimensions (if provided)
    - both bbox and click are missing
    """
    if pred_bbox is None and pred_click is None:
        return 1.0

    if pred_bbox is not None:
        x1, y1, x2, y2 = pred_bbox
        if x2 < x1 or y2 < y1:
            return 1.0
        if x1 < 0 or y1 < 0:
            return 1.0
        if image_width and (x2 > image_width * 1.05):
            return 1.0
        if image_height and (y2 > image_height * 1.05):
            return 1.0

    if pred_click is not None:
        cx, cy = pred_click
        if cx < 0 or cy < 0:
            return 1.0
        if image_width and cx > image_width * 1.05:
            return 1.0
        if image_height and cy > image_height * 1.05:
            return 1.0

    return 0.0


# -----------------------------------------------------------------------
# Composite reward calculator
# -----------------------------------------------------------------------

class VerifiableRewardCalculator:
    """Compute the composite verifiable reward for GUI grounding.

    Parameters
    ----------
    weights : dict, optional
        Custom weights for each reward term.  Keys should match
        :data:`DEFAULT_REWARD_WEIGHTS`.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}

    def compute(
        self,
        sample_id: str,
        pred_element_id: Optional[str] = None,
        gt_element_id: Optional[str] = None,
        pred_bbox: Optional[tuple[float, float, float, float]] = None,
        gt_bbox: Optional[tuple[float, float, float, float]] = None,
        pred_click: Optional[tuple[float, float]] = None,
        pred_action: Optional[str] = None,
        gt_action: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> RewardResult:
        """Compute the full composite reward for a single prediction.

        Each component is computed independently, then combined:

            r = λ1·elem + λ2·iou + λ3·click + λ4·act − λ5·penalty
        """
        # Individual components
        elem = element_correct(pred_element_id, gt_element_id)

        iou = 0.0
        if pred_bbox is not None and gt_bbox is not None:
            iou = bbox_iou(pred_bbox, gt_bbox)

        click = 0.0
        if pred_click is not None and gt_bbox is not None:
            click = click_inside_target(pred_click, gt_bbox)

        act = action_type_correct(pred_action, gt_action)

        penalty = invalid_format_penalty(
            pred_bbox, pred_click, image_width, image_height
        )

        components = RewardComponents(
            element_correct=elem,
            iou=iou,
            click_inside_target=click,
            action_type_correct=act,
            invalid_format_penalty=penalty,
        )

        # Weighted combination
        total = (
            self.weights["element_correct"] * elem
            + self.weights["iou"] * iou
            + self.weights["click_inside_target"] * click
            + self.weights["action_type_correct"] * act
            - self.weights["invalid_format_penalty"] * penalty
        )

        return RewardResult(
            sample_id=sample_id,
            total_reward=total,
            components=components,
            weights_used=dict(self.weights),
            is_valid_format=(penalty == 0.0),
        )

    def compute_batch(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
    ) -> list[RewardResult]:
        """Compute rewards for a batch of predictions.

        Each item in ``predictions`` and ``ground_truths`` should be a
        dict with keys matching :meth:`compute` parameters.
        """
        results = []
        for pred, gt in zip(predictions, ground_truths):
            result = self.compute(
                sample_id=pred.get("sample_id", gt.get("sample_id", "")),
                pred_element_id=pred.get("element_id"),
                gt_element_id=gt.get("element_id"),
                pred_bbox=pred.get("bbox"),
                gt_bbox=gt.get("bbox"),
                pred_click=pred.get("click_point"),
                pred_action=pred.get("action_type"),
                gt_action=gt.get("action_type"),
                image_width=gt.get("image_width"),
                image_height=gt.get("image_height"),
            )
            results.append(result)
        return results
