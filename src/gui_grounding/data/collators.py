"""Data collators for batching GUI grounding samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from gui_grounding.data.schemas import GroundingSample


@dataclass
class GroundingCollator:
    """Collate a list of :class:`GroundingSample` into a batch dict.

    This collator produces plain Python structures.  A model-specific
    collator should extend this to add tokenization and image transforms.

    TODO(stage-2): Integrate with a VLM processor (e.g., Qwen2-VL
    processor) for producing model-ready tensors.
    """

    max_candidates: int = 32

    def __call__(self, samples: list[GroundingSample]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "sample_ids": [s.sample_id for s in samples],
            "image_paths": [s.image_path for s in samples],
            "instructions": [s.instruction for s in samples],
            "action_types": [s.action_type for s in samples],
            "target_bboxes": [s.target_bbox.as_tuple() if s.target_bbox else None for s in samples],
            "click_points": [s.click_point for s in samples],
        }

        if any(s.dom_candidates for s in samples):
            batch["candidates"] = [
                (s.dom_candidates or [])[:self.max_candidates] for s in samples
            ]

        return batch
