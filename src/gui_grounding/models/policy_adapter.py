"""Policy adapter for RL-style training interfaces.

Provides a thin abstraction so that different policy improvement
strategies (reward reranking, DPO-style preference, lightweight GRPO)
can interact with the base model through a uniform API.
"""

from __future__ import annotations

from typing import Any

from gui_grounding.data.schemas import CandidateAction, GroundingSample, PredictionResult
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class PolicyAdapter:
    """Adapts a grounding model for policy-level operations.

    Methods correspond to the three improvement stages:
    - :meth:`select_best` — reward-based reranking
    - :meth:`compute_preference_pair` — pairwise preference construction
    - :meth:`compute_group_reward` — GRPO-style group reward

    TODO(stage-2): Wire to actual model and optimizer.
    """

    def __init__(self, model: Any = None) -> None:
        self.model = model
        logger.info("[Scaffold] PolicyAdapter initialized.")

    def select_best(
        self,
        sample: GroundingSample,
        candidates: list[CandidateAction],
        reward_scores: list[float],
    ) -> CandidateAction:
        """Select the highest-reward candidate (reranking)."""
        if not candidates:
            raise ValueError("Empty candidate list")
        best_idx = max(range(len(reward_scores)), key=lambda i: reward_scores[i])
        return candidates[best_idx]

    def compute_preference_pair(
        self,
        sample: GroundingSample,
        candidates: list[CandidateAction],
        reward_scores: list[float],
    ) -> tuple[CandidateAction, CandidateAction]:
        """Construct a (preferred, dispreferred) pair for DPO-style training.

        TODO(stage-2): Add margin-based selection and diversity filtering.
        """
        if len(candidates) < 2:
            raise ValueError("Need at least 2 candidates for preference pairs")
        sorted_indices = sorted(range(len(reward_scores)), key=lambda i: reward_scores[i], reverse=True)
        return candidates[sorted_indices[0]], candidates[sorted_indices[-1]]

    def compute_group_reward(
        self,
        sample: GroundingSample,
        candidates: list[CandidateAction],
        reward_scores: list[float],
    ) -> dict[str, Any]:
        """Compute group-level reward statistics for GRPO-style optimization.

        Returns a dict with advantage estimates relative to the group mean.

        TODO(stage-2): Implement proper GRPO advantage normalization.
        """
        if not reward_scores:
            return {"advantages": [], "mean_reward": 0.0}

        mean_r = sum(reward_scores) / len(reward_scores)
        advantages = [r - mean_r for r in reward_scores]
        return {
            "advantages": advantages,
            "mean_reward": mean_r,
            "max_reward": max(reward_scores),
            "min_reward": min(reward_scores),
        }
