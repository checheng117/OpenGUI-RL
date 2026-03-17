"""Candidate scorer: ranks top-k candidate actions.

Used in the reranking pipeline (Stage B/C) to select the best
candidate from a set of model-generated or heuristically produced
action proposals.
"""

from __future__ import annotations

from typing import Any

from gui_grounding.data.schemas import CandidateAction
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class CandidateScorer:
    """Score and rank candidate actions for a given sample.

    The scorer can be:
    - A learned model (MLP/cross-encoder on VLM features + candidate features)
    - A reward-based scorer that uses verifiable reward signals

    TODO(stage-2): Implement as an ``nn.Module`` that takes VLM features
    and candidate features and outputs a scalar score per candidate.
    """

    def __init__(self, scoring_mode: str = "learned", hidden_dim: int = 768) -> None:
        assert scoring_mode in ("learned", "reward_based"), f"Unknown mode: {scoring_mode}"
        self.scoring_mode = scoring_mode
        self.hidden_dim = hidden_dim
        logger.info("[Scaffold] CandidateScorer created (mode=%s)", scoring_mode)

    def score(
        self,
        hidden_states: Any,
        candidates: list[CandidateAction],
    ) -> list[float]:
        """Assign a scalar score to each candidate.

        Parameters
        ----------
        hidden_states
            VLM output features for the current sample.
        candidates
            List of candidate actions to score.

        Returns
        -------
        list[float]
            One score per candidate (higher = better).
        """
        logger.debug("[Scaffold] CandidateScorer.score — returning uniform scores.")
        return [1.0 / max(len(candidates), 1)] * len(candidates)

    def rank(
        self,
        hidden_states: Any,
        candidates: list[CandidateAction],
    ) -> list[CandidateAction]:
        """Score and return candidates sorted best-first."""
        scores = self.score(hidden_states, candidates)
        for cand, s in zip(candidates, scores):
            cand.score = s
        return sorted(candidates, key=lambda c: c.score or 0, reverse=True)
