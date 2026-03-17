"""Lightweight GRPO / contextual-bandit trainer.

Stage C option 2: Group Relative Policy Optimization on top-k
candidate actions.  Operates as a contextual bandit rather than
a full trajectory-level RL method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class GRPOLightTrainer:
    """Lightweight GRPO-style policy improvement.

    For each sample:
    1. Generate a group of K candidate actions.
    2. Compute verifiable rewards for each.
    3. Estimate advantages relative to the group mean.
    4. Update the policy to increase probability of above-average actions.

    This avoids full PPO complexity while still leveraging reward signals
    for policy improvement.

    TODO(stage-2): Implement the actual GRPO update step.
    """

    def __init__(
        self,
        model: Any = None,
        output_dir: str | Path = "outputs/grpo",
        learning_rate: float = 1e-5,
        group_size: int = 8,
        num_iterations: int = 100,
        batch_size: int = 4,
        clip_eps: float = 0.2,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.group_size = group_size
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.temperature = temperature
        self.seed = seed
        logger.info(
            "[Scaffold] GRPOLightTrainer initialized (K=%d, clip=%.2f).",
            group_size,
            clip_eps,
        )

    def train(self) -> dict[str, Any]:
        """Run the lightweight GRPO training loop.

        TODO(stage-2): Implement:
        - Sample generation loop (K candidates per input)
        - Reward computation for each candidate
        - Advantage estimation (reward - group_mean)
        - Clipped policy gradient update
        """
        logger.info("=" * 60)
        logger.info("GRPO-Light Training — SCAFFOLD MODE")
        logger.info("=" * 60)

        if self.model is None:
            logger.warning("No model provided. Skipping.")
            return {"status": "scaffold"}

        return {
            "status": "scaffold",
            "group_size": self.group_size,
            "iterations": self.num_iterations,
        }

    def evaluate(self) -> dict[str, float]:
        logger.info("GRPO-Light Evaluation — SCAFFOLD MODE")
        return {"status": "scaffold"}
