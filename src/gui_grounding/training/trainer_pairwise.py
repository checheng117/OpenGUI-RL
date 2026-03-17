"""Pairwise preference optimization trainer (DPO-style).

Stage C option 1: Improve the policy using pairwise preferences
constructed from verifiable reward comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class PairwisePreferenceTrainer:
    """DPO-style preference optimization for GUI grounding.

    Given (preferred_action, dispreferred_action) pairs constructed
    from verifiable rewards, optimize the model to prefer higher-reward
    actions.

    TODO(stage-2): Implement with TRL's DPOTrainer or a custom loop.
    """

    def __init__(
        self,
        model: Any = None,
        ref_model: Any = None,
        preference_data_path: Optional[str | Path] = None,
        output_dir: str | Path = "outputs/pairwise",
        learning_rate: float = 5e-6,
        beta: float = 0.1,
        num_epochs: int = 1,
        batch_size: int = 4,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.preference_data_path = preference_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.beta = beta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        logger.info("[Scaffold] PairwisePreferenceTrainer initialized (beta=%.2f).", beta)

    def train(self) -> dict[str, Any]:
        """Run the DPO-style training loop.

        TODO(stage-2): Implement:
        - Load preference pairs
        - Compute policy and reference log-probabilities
        - Apply DPO loss
        """
        logger.info("=" * 60)
        logger.info("Pairwise Preference Training — SCAFFOLD MODE")
        logger.info("=" * 60)

        if self.model is None:
            logger.warning("No model provided. Skipping.")
            return {"status": "scaffold"}
        return {"status": "scaffold", "beta": self.beta}

    def evaluate(self) -> dict[str, float]:
        logger.info("Pairwise Evaluation — SCAFFOLD MODE")
        return {"status": "scaffold"}
