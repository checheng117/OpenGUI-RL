"""Cross-website / cross-domain transfer evaluator.

Evaluates generalization by running the model on multiple splits
(test_task, test_website, test_domain) and comparing performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gui_grounding.evaluation.evaluator_grounding import GroundingEvaluator
from gui_grounding.utils.io import save_json
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class TransferEvaluator:
    """Evaluate cross-website generalization across multiple splits.

    TODO(stage-2): Implement full transfer evaluation with per-split
    and per-domain breakdowns.
    """

    def __init__(
        self,
        datasets: dict[str, Any] | None = None,
        model: Any = None,
        output_dir: str | Path = "outputs/transfer_eval",
    ) -> None:
        self.datasets = datasets or {}
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, dict[str, float]]:
        """Run evaluation across all provided splits."""
        logger.info("=" * 60)
        logger.info("Transfer Evaluation — SCAFFOLD MODE")
        logger.info("=" * 60)

        all_results = {}
        for split_name, dataset in self.datasets.items():
            logger.info("Evaluating split: %s", split_name)
            evaluator = GroundingEvaluator(
                dataset=dataset,
                model=self.model,
                output_dir=self.output_dir / split_name,
            )
            all_results[split_name] = evaluator.run()

        save_json(all_results, self.output_dir / "transfer_results.json")
        logger.info("Transfer results saved to %s", self.output_dir)
        return all_results
