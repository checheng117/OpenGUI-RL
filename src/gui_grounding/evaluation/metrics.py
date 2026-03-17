"""Evaluation metrics for GUI grounding and action prediction.

All metric functions operate on plain Python types so they can be
unit-tested without PyTorch.
"""

from __future__ import annotations

from typing import Optional

from gui_grounding.reward.verifiable_reward import bbox_iou


# -----------------------------------------------------------------------
# Core metrics
# -----------------------------------------------------------------------

def element_accuracy(
    pred_ids: list[Optional[str]],
    gt_ids: list[Optional[str]],
) -> float:
    """Fraction of samples where predicted element matches GT."""
    if not pred_ids:
        return 0.0
    correct = sum(
        1 for p, g in zip(pred_ids, gt_ids)
        if p is not None and g is not None and p == g
    )
    return correct / len(pred_ids)


def point_accuracy(
    pred_points: list[Optional[tuple[float, float]]],
    gt_bboxes: list[Optional[tuple[float, float, float, float]]],
) -> float:
    """Fraction of samples where predicted click falls inside GT box."""
    if not pred_points:
        return 0.0
    correct = 0
    total = 0
    for pt, box in zip(pred_points, gt_bboxes):
        if pt is None or box is None:
            continue
        total += 1
        cx, cy = pt
        x1, y1, x2, y2 = box
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            correct += 1
    return correct / max(total, 1)


def mean_iou(
    pred_bboxes: list[Optional[tuple[float, float, float, float]]],
    gt_bboxes: list[Optional[tuple[float, float, float, float]]],
) -> float:
    """Mean IoU over all valid bbox pairs."""
    ious: list[float] = []
    for p, g in zip(pred_bboxes, gt_bboxes):
        if p is not None and g is not None:
            ious.append(bbox_iou(p, g))
    return sum(ious) / max(len(ious), 1)


def iou_at_threshold(
    pred_bboxes: list[Optional[tuple[float, float, float, float]]],
    gt_bboxes: list[Optional[tuple[float, float, float, float]]],
    threshold: float = 0.5,
) -> float:
    """Fraction of samples with IoU >= threshold."""
    if not pred_bboxes:
        return 0.0
    passed = 0
    total = 0
    for p, g in zip(pred_bboxes, gt_bboxes):
        if p is None or g is None:
            continue
        total += 1
        if bbox_iou(p, g) >= threshold:
            passed += 1
    return passed / max(total, 1)


def action_type_accuracy(
    pred_actions: list[Optional[str]],
    gt_actions: list[Optional[str]],
) -> float:
    """Fraction of samples where predicted action type matches GT."""
    if not pred_actions:
        return 0.0
    correct = sum(
        1 for p, g in zip(pred_actions, gt_actions)
        if p is not None and g is not None and p.lower() == g.lower()
    )
    return correct / len(pred_actions)


# -----------------------------------------------------------------------
# Reranking-specific metrics
# -----------------------------------------------------------------------

def best_of_k_improvement(
    first_choice_correct: list[bool],
    best_of_k_correct: list[bool],
) -> float:
    """Improvement from best-of-k selection over first-choice.

    Returns the absolute accuracy gain.
    """
    n = len(first_choice_correct)
    if n == 0:
        return 0.0
    fc_acc = sum(first_choice_correct) / n
    bok_acc = sum(best_of_k_correct) / n
    return bok_acc - fc_acc


def reranked_gain(
    first_choice_rewards: list[float],
    reranked_rewards: list[float],
) -> float:
    """Mean reward improvement from reranking."""
    if not first_choice_rewards:
        return 0.0
    gains = [r - f for r, f in zip(reranked_rewards, first_choice_rewards)]
    return sum(gains) / len(gains)


# -----------------------------------------------------------------------
# Aggregate metric computation
# -----------------------------------------------------------------------

def compute_all_metrics(
    pred_element_ids: list[Optional[str]],
    gt_element_ids: list[Optional[str]],
    pred_bboxes: list[Optional[tuple[float, float, float, float]]],
    gt_bboxes: list[Optional[tuple[float, float, float, float]]],
    pred_points: list[Optional[tuple[float, float]]],
    pred_actions: list[Optional[str]],
    gt_actions: list[Optional[str]],
) -> dict[str, float]:
    """Compute all standard metrics in one call.

    Returns a dict suitable for logging or display.
    """
    return {
        "element_accuracy": element_accuracy(pred_element_ids, gt_element_ids),
        "point_accuracy": point_accuracy(pred_points, gt_bboxes),
        "mean_iou": mean_iou(pred_bboxes, gt_bboxes),
        "iou@0.5": iou_at_threshold(pred_bboxes, gt_bboxes, threshold=0.5),
        "iou@0.75": iou_at_threshold(pred_bboxes, gt_bboxes, threshold=0.75),
        "action_type_accuracy": action_type_accuracy(pred_actions, gt_actions),
    }
