"""Abstract base model for GUI grounding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from gui_grounding.data.schemas import GroundingSample, PredictionResult


class BaseGroundingModel(ABC):
    """Interface that all grounding model implementations must follow.

    Downstream trainers and evaluators program against this interface
    so that swapping backbones only requires a new concrete subclass.
    """

    @abstractmethod
    def predict(
        self,
        sample: GroundingSample,
        **kwargs,
    ) -> PredictionResult:
        """Run inference on a single sample."""

    @abstractmethod
    def predict_batch(
        self,
        batch: dict[str, Any],
        **kwargs,
    ) -> list[PredictionResult]:
        """Run inference on a collated batch."""

    @abstractmethod
    def generate_candidates(
        self,
        sample: GroundingSample,
        num_candidates: int = 5,
        **kwargs,
    ) -> list[PredictionResult]:
        """Generate multiple candidate predictions for reranking."""

    def save(self, path: str) -> None:
        """Persist model weights / adapter to disk."""
        raise NotImplementedError("TODO(stage-2): implement checkpoint saving")

    def load(self, path: str) -> None:
        """Load model weights / adapter from disk."""
        raise NotImplementedError("TODO(stage-2): implement checkpoint loading")
