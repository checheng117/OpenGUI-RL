"""Data structures for reward computation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewardComponents(BaseModel):
    """Breakdown of individual reward terms before weighting."""

    element_correct: float = 0.0
    iou: float = 0.0
    click_inside_target: float = 0.0
    action_type_correct: float = 0.0
    invalid_format_penalty: float = 0.0


class RewardResult(BaseModel):
    """Complete reward output for a single prediction."""

    sample_id: str
    total_reward: float
    components: RewardComponents
    weights_used: dict[str, float] = Field(default_factory=dict)
    is_valid_format: bool = True


class CandidateRewardSet(BaseModel):
    """Reward results for all candidates of a single sample."""

    sample_id: str
    rewards: list[RewardResult]
    best_candidate_idx: int = 0
    mean_reward: float = 0.0
    max_reward: float = 0.0
