"""Supervised fine-tuning (SFT) trainer for GUI grounding.

Stage A: Train on screenshot-instruction pairs to predict the target
element and action type using standard supervised objectives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class SFTTrainer:
    """Supervised fine-tuning trainer.

    Responsible for:
    - Loading model and data
    - Running the SFT training loop
    - Logging metrics and saving checkpoints

    TODO(stage-2): Integrate with HuggingFace Trainer or Accelerate
    for distributed training, mixed precision, and gradient accumulation.
    """

    def __init__(
        self,
        model: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        output_dir: str | Path = "outputs/sft",
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.1,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 50,
        seed: int = 42,
        use_wandb: bool = False,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.seed = seed
        self.use_wandb = use_wandb

        self._global_step = 0
        logger.info("[Scaffold] SFTTrainer initialized. output_dir=%s", self.output_dir)

    def train(self) -> dict[str, float]:
        """Run the SFT training loop.

        Returns a summary dict of training metrics.

        TODO(stage-2): Implement real training loop with:
        - DataLoader construction
        - Optimizer / scheduler setup
        - Forward / backward / step
        - Evaluation at intervals
        - Checkpoint saving
        """
        logger.info("=" * 60)
        logger.info("SFT Training — SCAFFOLD MODE")
        logger.info("=" * 60)
        logger.info("Config: lr=%.2e, epochs=%d, batch=%d", self.learning_rate, self.num_epochs, self.batch_size)

        if self.model is None:
            logger.warning("No model provided. Skipping training. (scaffold mode)")
            return {"status": "scaffold", "note": "No model loaded — training skipped"}

        if self.train_dataset is None or len(self.train_dataset) == 0:
            logger.warning("No training data. Skipping training. (scaffold mode)")
            return {"status": "scaffold", "note": "No data loaded — training skipped"}

        logger.info("Would train on %d samples for %d epochs.", len(self.train_dataset), self.num_epochs)
        return {
            "status": "scaffold",
            "train_samples": len(self.train_dataset),
            "epochs": self.num_epochs,
        }

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval dataset.

        TODO(stage-2): Implement with metrics from evaluation/metrics.py.
        """
        logger.info("SFT Evaluation — SCAFFOLD MODE")
        if self.eval_dataset is None:
            logger.warning("No eval data. Returning dummy metrics.")
            return {"status": "scaffold"}
        return {"status": "scaffold", "eval_samples": len(self.eval_dataset)}

    def save_checkpoint(self, tag: str = "latest") -> Path:
        """Save a checkpoint to disk.

        TODO(stage-2): Save model state_dict, optimizer, scheduler, and config.
        """
        ckpt_dir = self.output_dir / f"checkpoint-{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Scaffold] Would save checkpoint to %s", ckpt_dir)
        return ckpt_dir
