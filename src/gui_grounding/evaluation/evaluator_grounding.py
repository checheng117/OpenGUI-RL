"""Grounding evaluator — runs evaluation on a single dataset split."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from gui_grounding.evaluation.metrics import compute_all_metrics
from gui_grounding.utils.io import save_json
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class GroundingEvaluator:
    """Evaluate a grounding model on a single dataset split.

    Collects predictions, computes metrics, and optionally saves
    results to disk.

    TODO(stage-2): Integrate with real model inference loop.
    """

    def __init__(
        self,
        dataset: Any = None,
        model: Any = None,
        output_dir: str | Path = "outputs/eval",
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, float]:
        """Execute evaluation and return metrics.

        TODO(stage-2): Implement:
        - Iterate over dataset
        - Run model.predict() or model.predict_batch()
        - Collect predictions
        - Compute metrics via compute_all_metrics()
        """
        logger.info("=" * 60)
        logger.info("Grounding Evaluation — SCAFFOLD MODE")
        logger.info("=" * 60)

        if self.dataset is None:
            logger.warning("No dataset provided.")
            return {"status": "scaffold"}

        if self.model is None:
            logger.warning("No model provided. Running with dummy predictions.")

        sample_count = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        logger.info("Would evaluate on %d samples.", sample_count)

        metrics = {
            "status": "scaffold",
            "num_samples": sample_count,
            "element_accuracy": None,
            "point_accuracy": None,
            "mean_iou": None,
            "iou@0.5": None,
            "action_type_accuracy": None,
        }

        save_json(metrics, self.output_dir / "eval_results.json")
        logger.info("Results saved to %s", self.output_dir / "eval_results.json")
        return metrics
