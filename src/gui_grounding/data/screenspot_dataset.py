"""ScreenSpot-v2 dataset adapter backed by Hugging Face datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from gui_grounding.constants import PROCESSED_DATA_DIR
from gui_grounding.data.base_dataset import BaseGroundingDataset
from gui_grounding.data.schemas import BBox, GroundingSample
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)

HF_DATASET_ID = "lscpku/ScreenSpot-v2"
VALID_SPLITS = ("test",)


def _bootstrap_hf_environment() -> Optional[str]:
    """Load local HF settings and return the best available token."""
    try:
        from dotenv import load_dotenv
        from gui_grounding.constants import PROJECT_ROOT

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("GUI_GROUNDING_HF_FALLBACK_ENDPOINT", "https://hf-mirror.com")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    try:
        from huggingface_hub import get_token

        token = get_token()
    except ImportError:
        token = None

    if token:
        os.environ.setdefault("HF_TOKEN", token)
    return token


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_screenshot(image: Optional[Image.Image]) -> Optional[Image.Image]:
    if image is None:
        return None
    image.load()
    normalized = image.convert("RGB")
    close = getattr(image, "close", None)
    if callable(close):
        close()
    return normalized


class ScreenSpotV2Dataset(BaseGroundingDataset):
    """Adapter for ScreenSpot-v2 evaluation data from Hugging Face.

    Source format:
    - `image`: PIL image
    - `instruction`: natural-language grounding instruction
    - `bbox`: absolute `[x, y, width, height]`
    - `split`: platform subgroup (`mobile`, `web`, `desktop`)
    - `data_type`: element subtype (`text`, `icon`)
    - `data_source`: source domain / operating system
    """

    def __init__(
        self,
        data_dir: str | Path = "data/raw/screenspot_v2",
        split: str = "test",
        max_samples: Optional[int] = None,
        hf_dataset_id: str = HF_DATASET_ID,
        screenshot_dir: Optional[str | Path] = None,
        cache_screenshots: bool = True,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from {VALID_SPLITS}")

        self.hf_dataset_id = hf_dataset_id
        self.cache_screenshots = cache_screenshots
        self._sample_counter = 0
        self.gt_bbox_clipped_count = 0

        if screenshot_dir is None:
            screenshot_dir = PROCESSED_DATA_DIR / "screenspot_v2_screenshots" / split
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(data_dir=data_dir, split=split, max_samples=max_samples)

    @property
    def name(self) -> str:
        return "screenspot_v2"

    def _load(self) -> None:
        raw_items = self._load_raw()
        total_rows = len(raw_items)
        if self.max_samples is not None:
            total_rows = min(total_rows, self.max_samples)

        self._samples = [self._to_sample(raw_items[idx]) for idx in range(total_rows)]
        logger.info(
            "Loaded %d samples from %s (split=%s) with %d ground-truth bbox clipping corrections",
            len(self._samples),
            self.name,
            self.split,
            self.gt_bbox_clipped_count,
        )

    def _load_raw(self):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("ScreenSpot-v2 adapter requires `datasets`.") from exc

        token = _bootstrap_hf_environment()
        logger.info(
            "Loading ScreenSpot-v2 split='%s' from HuggingFace dataset=%s",
            self.split,
            self.hf_dataset_id,
        )
        return load_dataset(
            self.hf_dataset_id,
            split=self.split,
            token=token,
        )

    def _cache_image(self, raw_item: dict[str, Any]) -> tuple[str, tuple[int, int]]:
        image_name = str(raw_item.get("img_filename", f"{self._sample_counter:06d}.png"))
        image_path = self.screenshot_dir / image_name

        if image_path.exists():
            with Image.open(image_path) as cached_image:
                return str(image_path), cached_image.size

        image = _normalize_screenshot(raw_item.get("image"))
        if image is None:
            raise ValueError(f"Missing image for ScreenSpot sample: {image_name}")

        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        image.close()
        return str(image_path), raw_item["image"].size

    def _bbox_xywh_to_xyxy(
        self,
        bbox_raw: Any,
        image_width: int,
        image_height: int,
    ) -> tuple[Optional[BBox], bool]:
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            return None, False
        try:
            x, y, w, h = [float(v) for v in bbox_raw]
        except (TypeError, ValueError):
            return None, False
        if w <= 0 or h <= 0:
            return None, False

        raw_x2 = x + w
        raw_y2 = y + h
        x1 = _clamp(x, 0.0, float(image_width))
        y1 = _clamp(y, 0.0, float(image_height))
        x2 = _clamp(raw_x2, 0.0, float(image_width))
        y2 = _clamp(raw_y2, 0.0, float(image_height))
        clipped = abs(raw_x2 - x2) > 1e-6 or abs(raw_y2 - y2) > 1e-6 or abs(x - x1) > 1e-6 or abs(y - y1) > 1e-6

        if x2 <= x1 or y2 <= y1:
            return None, clipped
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2), clipped

    def _to_sample(self, raw_item) -> GroundingSample:
        idx = self._sample_counter
        self._sample_counter += 1

        image_path, image_size = self._cache_image(raw_item)
        image_width = int(raw_item.get("image_width") or image_size[0])
        image_height = int(raw_item.get("image_height") or image_size[1])

        bbox, clipped = self._bbox_xywh_to_xyxy(
            bbox_raw=raw_item.get("bbox"),
            image_width=image_width,
            image_height=image_height,
        )
        if clipped:
            self.gt_bbox_clipped_count += 1

        filename = str(raw_item.get("img_filename", f"{idx:06d}.png"))
        stem = Path(filename).stem
        sample_id = f"screenspot_v2_{self.split}_{idx:05d}_{stem}"

        return GroundingSample(
            sample_id=sample_id,
            dataset_name=self.name,
            split=self.split,
            image_path=image_path,
            instruction=str(raw_item.get("instruction", "")),
            target_bbox=bbox,
            click_point=bbox.center if bbox else None,
            platform=str(raw_item.get("split", "")) or None,
            domain=str(raw_item.get("data_source", "")) or None,
            metadata={
                "hf_dataset_id": self.hf_dataset_id,
                "img_filename": filename,
                "raw_bbox_xywh": raw_item.get("bbox"),
                "bbox_format": "xywh_absolute",
                "element_type": str(raw_item.get("data_type", "")) or None,
                "data_source": str(raw_item.get("data_source", "")) or None,
                "platform": str(raw_item.get("split", "")) or None,
                "image_width": image_width,
                "image_height": image_height,
                "gt_bbox_clipped": clipped,
            },
        )
