"""Candidate action generator.

Produces multiple candidate actions per sample for reranking.
Supports both heuristic generation (for bootstrapping) and
model-based generation (for the full pipeline).
"""

from __future__ import annotations

import random
from typing import Optional

from gui_grounding.data.schemas import BBox, CandidateAction, GroundingSample
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class CandidateGenerator:
    """Generate candidate actions for a grounding sample.

    Modes:
    - ``"dummy"``: Generate random perturbations of the GT for testing.
    - ``"heuristic"``: Use DOM candidates + spatial heuristics.
    - ``"model"``: Use VLM to generate multiple outputs via sampling.

    TODO(stage-2): Implement model-based candidate generation.
    """

    def __init__(
        self,
        mode: str = "dummy",
        num_candidates: int = 8,
        seed: int = 42,
    ) -> None:
        assert mode in ("dummy", "heuristic", "model")
        self.mode = mode
        self.num_candidates = num_candidates
        self._rng = random.Random(seed)
        logger.info("CandidateGenerator created (mode=%s, K=%d)", mode, num_candidates)

    def generate(self, sample: GroundingSample) -> list[CandidateAction]:
        """Generate candidates for the given sample."""
        if self.mode == "dummy":
            return self._generate_dummy(sample)
        elif self.mode == "heuristic":
            return self._generate_heuristic(sample)
        elif self.mode == "model":
            return self._generate_model(sample)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _generate_dummy(self, sample: GroundingSample) -> list[CandidateAction]:
        """Create dummy candidates by perturbing ground truth.

        Useful for testing the reward and reranking pipeline before
        a real model is available.
        """
        candidates: list[CandidateAction] = []

        gt_bbox = sample.target_bbox
        base_x1 = gt_bbox.x1 if gt_bbox else 100.0
        base_y1 = gt_bbox.y1 if gt_bbox else 100.0
        base_x2 = gt_bbox.x2 if gt_bbox else 200.0
        base_y2 = gt_bbox.y2 if gt_bbox else 200.0

        for i in range(self.num_candidates):
            noise_scale = i * 10.0
            nx1 = max(0, base_x1 + self._rng.uniform(-noise_scale, noise_scale))
            ny1 = max(0, base_y1 + self._rng.uniform(-noise_scale, noise_scale))
            nx2 = max(0, base_x2 + self._rng.uniform(-noise_scale, noise_scale))
            ny2 = max(0, base_y2 + self._rng.uniform(-noise_scale, noise_scale))
            bbox = BBox(
                x1=min(nx1, nx2), y1=min(ny1, ny2),
                x2=max(nx1, nx2), y2=max(ny1, ny2),
            )

            cx, cy = bbox.center
            action_choices = ["click", "type", "select", "hover"]
            action = sample.action_type if (i == 0 and sample.action_type) else self._rng.choice(action_choices)

            candidates.append(
                CandidateAction(
                    candidate_id=f"{sample.sample_id}_cand_{i}",
                    action_type=action,
                    bbox=bbox,
                    click_point=(cx, cy),
                    element_id=sample.target_element_id if i == 0 else None,
                    source="dummy",
                )
            )

        return candidates

    def _generate_heuristic(self, sample: GroundingSample) -> list[CandidateAction]:
        """Generate candidates from DOM candidates if available.

        TODO(stage-2): Use OCR text matching, spatial proximity, and
        element-type filtering to produce better heuristic candidates.
        """
        candidates: list[CandidateAction] = []

        if sample.dom_candidates:
            for i, elem in enumerate(sample.dom_candidates[: self.num_candidates]):
                candidates.append(
                    CandidateAction(
                        candidate_id=f"{sample.sample_id}_dom_{i}",
                        element_id=elem.element_id,
                        bbox=elem.bbox,
                        click_point=elem.bbox.center if elem.bbox else None,
                        source="heuristic_dom",
                    )
                )

        while len(candidates) < self.num_candidates:
            candidates.extend(self._generate_dummy(sample))

        return candidates[: self.num_candidates]

    def _generate_model(self, sample: GroundingSample) -> list[CandidateAction]:
        """Generate candidates using VLM with temperature sampling.

        TODO(stage-2): Implement:
        - VLM.generate() with temperature > 0, num_return_sequences=K
        - Parse each output into a CandidateAction
        """
        logger.warning(
            "[Scaffold] Model-based candidate generation not yet implemented. "
            "Falling back to dummy mode."
        )
        return self._generate_dummy(sample)
