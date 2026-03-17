"""VisualWebBench dataset adapter (supplementary evaluation).

Reference
---------
Wang et al., "VisualWebBench: How Far Have Multimodal LLMs Evolved in
Web Page Understanding and Grounding?", 2024.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from gui_grounding.data.base_dataset import BaseGroundingDataset
from gui_grounding.data.schemas import BBox, GroundingSample
from gui_grounding.utils.io import load_jsonl
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class VisualWebBenchDataset(BaseGroundingDataset):
    """Adapter for VisualWebBench (grounding-related subsets).

    TODO(stage-2): Determine which subsets are relevant (element grounding,
    action grounding, action prediction) and adapt accordingly.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/raw/visualwebbench",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__(data_dir=data_dir, split=split, max_samples=max_samples)

    @property
    def name(self) -> str:
        return "visualwebbench"

    def _load_raw(self) -> list:
        split_file = self.data_dir / f"{self.split}.jsonl"
        if not split_file.exists():
            logger.warning("Split file not found: %s", split_file)
            return []
        return load_jsonl(split_file)

    def _to_sample(self, raw_item: dict) -> GroundingSample:
        bbox_raw = raw_item.get("bbox")
        bbox = BBox(x1=bbox_raw[0], y1=bbox_raw[1], x2=bbox_raw[2], y2=bbox_raw[3]) if bbox_raw else None

        return GroundingSample(
            sample_id=str(raw_item.get("id", "")),
            dataset_name=self.name,
            split=self.split,
            image_path=str(self.data_dir / "images" / raw_item.get("image", "")),
            instruction=raw_item.get("instruction", ""),
            target_bbox=bbox,
            website=raw_item.get("website"),
        )
