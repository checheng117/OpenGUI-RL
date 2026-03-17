"""ScreenSpot-v2 dataset adapter.

ScreenSpot-v2 provides instruction-grounding pairs with target bounding
boxes across desktop, mobile, and web interfaces.

Reference
---------
ScreenSpot-v2 dataset release, 2025.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from gui_grounding.data.base_dataset import BaseGroundingDataset
from gui_grounding.data.schemas import BBox, GroundingSample
from gui_grounding.utils.io import load_json, load_jsonl
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class ScreenSpotV2Dataset(BaseGroundingDataset):
    """Adapter for ScreenSpot-v2 evaluation set.

    Expected directory layout::

        screenspot_v2/
        ├── test.jsonl
        └── images/

    TODO(stage-2): Verify exact file format from the ScreenSpot-v2 release.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/raw/screenspot_v2",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__(data_dir=data_dir, split=split, max_samples=max_samples)

    @property
    def name(self) -> str:
        return "screenspot_v2"

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
            platform=raw_item.get("platform"),
            metadata={"element_type": raw_item.get("element_type", "")},
        )
