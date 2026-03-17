"""Loss functions for GUI grounding tasks.

Provides composable losses for bounding-box regression, click-point
prediction, action-type classification, and contrastive reranking.
"""

from __future__ import annotations

import math
from typing import Optional


def smooth_l1_loss(pred: list[float], target: list[float], beta: float = 1.0) -> float:
    """Element-wise smooth L1 loss (Huber loss), averaged over coordinates.

    This is a pure-Python reference implementation.
    TODO(stage-2): Replace with ``torch.nn.functional.smooth_l1_loss``.
    """
    total = 0.0
    for p, t in zip(pred, target):
        diff = abs(p - t)
        if diff < beta:
            total += 0.5 * diff * diff / beta
        else:
            total += diff - 0.5 * beta
    return total / max(len(pred), 1)


def bbox_regression_loss(
    pred_bbox: list[float],
    target_bbox: list[float],
    loss_type: str = "smooth_l1",
) -> float:
    """Loss for bounding-box coordinate regression.

    Parameters
    ----------
    pred_bbox, target_bbox : list of 4 floats [x1, y1, x2, y2]
    loss_type : str
        ``"smooth_l1"`` or ``"l2"``.
    """
    assert len(pred_bbox) == 4 and len(target_bbox) == 4
    if loss_type == "smooth_l1":
        return smooth_l1_loss(pred_bbox, target_bbox)
    elif loss_type == "l2":
        return sum((p - t) ** 2 for p, t in zip(pred_bbox, target_bbox)) / 4
    raise ValueError(f"Unknown loss_type: {loss_type}")


def click_point_loss(
    pred_point: list[float],
    target_point: list[float],
) -> float:
    """Euclidean distance loss for click-point prediction."""
    assert len(pred_point) == 2 and len(target_point) == 2
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred_point, target_point)))


def cross_entropy_loss(logits: list[float], target_idx: int) -> float:
    """Softmax cross-entropy loss (pure-Python reference).

    TODO(stage-2): Replace with ``torch.nn.functional.cross_entropy``.
    """
    max_logit = max(logits)
    shifted = [l - max_logit for l in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    return -(shifted[target_idx] - log_sum_exp)


def pairwise_ranking_loss(
    score_preferred: float,
    score_dispreferred: float,
    margin: float = 0.0,
) -> float:
    """Margin-based pairwise ranking loss for reranking.

    loss = max(0, margin - (score_preferred - score_dispreferred))
    """
    return max(0.0, margin - (score_preferred - score_dispreferred))


def dpo_loss(
    log_ratio_preferred: float,
    log_ratio_dispreferred: float,
    beta: float = 0.1,
) -> float:
    """Simplified DPO loss (scalar version for reference).

    loss = -log(sigmoid(beta * (log_ratio_preferred - log_ratio_dispreferred)))

    TODO(stage-2): Implement batched tensor version.
    """
    diff = beta * (log_ratio_preferred - log_ratio_dispreferred)
    sigmoid_val = 1.0 / (1.0 + math.exp(-diff))
    return -math.log(max(sigmoid_val, 1e-10))
