"""Candidate scorer model used for learned reranking."""

from __future__ import annotations

import torch
import torch.nn as nn

from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class CandidateScorer(nn.Module):
    """Simple MLP scorer over engineered candidate features."""

    def __init__(
        self,
        scoring_mode: str = "learned",
        input_dim: int = 15,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert scoring_mode in ("learned", "reward_based"), f"Unknown mode: {scoring_mode}"
        self.scoring_mode = scoring_mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            # LazyLinear keeps configs backward-compatible when feature dimension evolves.
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        logger.info(
            "CandidateScorer initialized (mode=%s input_dim=%d hidden_dim=%d)",
            scoring_mode,
            input_dim,
            hidden_dim,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return scalar score per candidate feature row."""
        if features.ndim >= 2 and self.input_dim != int(features.shape[-1]):
            self.input_dim = int(features.shape[-1])
        return self.net(features).squeeze(-1)
