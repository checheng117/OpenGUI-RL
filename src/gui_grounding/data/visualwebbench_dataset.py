"""VisualWebBench grounding dataset adapter.

This adapter supports the two grounding-compatible supplementary subsets:

- ``element_ground``
- ``action_ground``

VisualWebBench packages each subset as a single parquet file on Hugging Face.
We download those parquet files directly, cache the raw screenshots locally,
and convert each row into the canonical :class:`GroundingSample` schema.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

from gui_grounding.constants import PROCESSED_DATA_DIR, PROJECT_ROOT
from gui_grounding.data.base_dataset import BaseGroundingDataset
from gui_grounding.data.schemas import BBox, CandidateElement, GroundingSample
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)

HF_DATASET_ID = "visualwebbench/VisualWebBench"
VALID_SPLITS = ("test",)
VALID_TASK_TYPES = ("element_ground", "action_ground")


def _bootstrap_hf_environment() -> Optional[str]:
    """Load local HF settings and return the best available token."""
    try:
        from dotenv import load_dotenv

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


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_bbox_to_absolute(
    bbox_raw: Sequence[Any] | None,
    *,
    image_width: int,
    image_height: int,
) -> BBox | None:
    if not isinstance(bbox_raw, Sequence) or len(bbox_raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_raw]
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(float(image_width), x1 * float(image_width)))
    y1 = max(0.0, min(float(image_height), y1 * float(image_height)))
    x2 = max(0.0, min(float(image_width), x2 * float(image_width)))
    y2 = max(0.0, min(float(image_height), y2 * float(image_height)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _save_image_struct(image_struct: Any, output_path: Path) -> tuple[str | None, tuple[int, int] | None]:
    if output_path.exists():
        with Image.open(output_path) as image:
            return str(output_path), image.size

    if not isinstance(image_struct, dict):
        return None, None

    image_bytes = image_struct.get("bytes")
    if not isinstance(image_bytes, (bytes, bytearray)):
        return None, None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(image_bytes)) as image:
        image_rgb = image.convert("RGB")
        size = image_rgb.size
        image_rgb.save(output_path)
        image_rgb.close()
    return str(output_path), size


def _candidate_elements(
    task_type: str,
    option_boxes: Sequence[BBox],
) -> list[CandidateElement]:
    candidates: list[CandidateElement] = []
    for idx, bbox in enumerate(option_boxes, start=1):
        candidates.append(
            CandidateElement(
                element_id=f"{task_type}_option_{idx}",
                bbox=bbox,
                tag="candidate",
                attributes={"option_index": str(idx - 1)},
            )
        )
    return candidates


class VisualWebBenchDataset(BaseGroundingDataset):
    """Adapter for the grounding-compatible VisualWebBench subsets."""

    def __init__(
        self,
        data_dir: str | Path = "data/raw/visualwebbench",
        split: str = "test",
        task_types: Sequence[str] | None = None,
        max_samples: Optional[int] = None,
        hf_dataset_id: str = HF_DATASET_ID,
        image_variant: str = "raw",
        screenshot_dir: str | Path | None = None,
        annotated_screenshot_dir: str | Path | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from {VALID_SPLITS}")

        resolved_task_types = tuple(task_types or VALID_TASK_TYPES)
        invalid_task_types = sorted(set(resolved_task_types) - set(VALID_TASK_TYPES))
        if invalid_task_types:
            raise ValueError(
                "VisualWebBench grounding adapter only supports "
                f"{VALID_TASK_TYPES}; got {tuple(invalid_task_types)}"
            )
        if image_variant not in {"raw", "annotated"}:
            raise ValueError("image_variant must be either 'raw' or 'annotated'.")

        self.task_types = resolved_task_types
        self.hf_dataset_id = hf_dataset_id
        self.image_variant = image_variant
        self._sample_counter = 0

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if screenshot_dir is None:
            screenshot_dir = PROCESSED_DATA_DIR / "visualwebbench_screenshots" / split / "raw"
        if annotated_screenshot_dir is None:
            annotated_screenshot_dir = PROCESSED_DATA_DIR / "visualwebbench_screenshots" / split / "annotated"
        self.screenshot_dir = Path(screenshot_dir)
        self.annotated_screenshot_dir = Path(annotated_screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_screenshot_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(data_dir=data_dir, split=split, max_samples=max_samples)

    @property
    def name(self) -> str:
        return "visualwebbench"

    def _load(self) -> None:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "VisualWebBench adapter requires `pyarrow` and `huggingface_hub`."
            ) from exc

        token = _bootstrap_hf_environment()
        samples: list[GroundingSample] = []
        remaining = self.max_samples

        for task_type in self.task_types:
            if remaining is not None and remaining <= 0:
                break

            filename = f"{task_type}/{self.split}-00000-of-00001.parquet"
            logger.info(
                "Loading VisualWebBench subset=%s split=%s from dataset=%s",
                task_type,
                self.split,
                self.hf_dataset_id,
            )
            parquet_path = hf_hub_download(
                repo_id=self.hf_dataset_id,
                repo_type="dataset",
                filename=filename,
                token=token,
            )

            parquet_file = pq.ParquetFile(parquet_path)
            subset_count = 0
            for batch in parquet_file.iter_batches(batch_size=16):
                for raw_item in batch.to_pylist():
                    raw_item["_task_type"] = task_type
                    samples.append(self._to_sample(raw_item))
                    subset_count += 1
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            break
                if remaining is not None and remaining <= 0:
                    break

            logger.info(
                "Loaded %d samples from VisualWebBench subset=%s",
                subset_count,
                task_type,
            )

        self._samples = samples
        logger.info(
            "Loaded %d samples from %s (split=%s subsets=%s image_variant=%s)",
            len(self._samples),
            self.name,
            self.split,
            list(self.task_types),
            self.image_variant,
        )

    def _load_raw(self) -> list:
        return []

    def _resolve_image_paths(
        self,
        raw_item: dict[str, Any],
        *,
        sample_id: str,
        task_type: str,
    ) -> tuple[str, tuple[int, int], str | None]:
        raw_output_path = self.screenshot_dir / task_type / f"{sample_id}.png"
        annotated_output_path = self.annotated_screenshot_dir / task_type / f"{sample_id}.png"

        raw_image_path, raw_size = _save_image_struct(raw_item.get("raw_image"), raw_output_path)
        annotated_image_path, annotated_size = _save_image_struct(raw_item.get("image"), annotated_output_path)

        if self.image_variant == "raw":
            primary_image_path = raw_image_path or annotated_image_path
            primary_size = raw_size or annotated_size
        else:
            primary_image_path = annotated_image_path or raw_image_path
            primary_size = annotated_size or raw_size

        if primary_image_path is None or primary_size is None:
            raise ValueError(f"Failed to cache screenshot for VisualWebBench sample {sample_id}")

        return primary_image_path, primary_size, annotated_image_path

    def _to_sample(self, raw_item: dict[str, Any]) -> GroundingSample:
        task_type = str(raw_item.get("_task_type") or raw_item.get("task_type") or "").strip()
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported VisualWebBench task_type={task_type}")

        sample_id = str(raw_item.get("id") or f"{task_type}_{self._sample_counter:05d}")
        self._sample_counter += 1

        image_path, actual_size, annotated_image_path = self._resolve_image_paths(
            raw_item,
            sample_id=sample_id,
            task_type=task_type,
        )
        image_size_raw = raw_item.get("image_size") or []
        image_width = _safe_int(image_size_raw[0]) if len(image_size_raw) >= 1 else actual_size[0]
        image_height = _safe_int(image_size_raw[1]) if len(image_size_raw) >= 2 else actual_size[1]
        image_width = image_width or actual_size[0]
        image_height = image_height or actual_size[1]

        options_raw = raw_item.get("options") or []
        option_boxes = [
            bbox
            for bbox in (
                _normalized_bbox_to_absolute(
                    option_raw,
                    image_width=image_width,
                    image_height=image_height,
                )
                for option_raw in options_raw
            )
            if bbox is not None
        ]

        answer_idx = _safe_int(raw_item.get("answer"))
        target_bbox = (
            option_boxes[answer_idx]
            if answer_idx is not None and 0 <= answer_idx < len(option_boxes)
            else None
        )

        elem_desc = str(raw_item.get("elem_desc") or "").strip()
        if task_type == "element_ground":
            instruction = f"Locate the element that matches this description: {elem_desc}"
        else:
            instruction = str(raw_item.get("instruction") or "").strip()

        metadata = {
            "hf_dataset_id": self.hf_dataset_id,
            "task_type": task_type,
            "website": str(raw_item.get("website") or "").strip() or None,
            "image_width": image_width,
            "image_height": image_height,
            "image_variant_used": self.image_variant,
            "annotated_image_path": annotated_image_path,
            "candidate_options_normalized": options_raw,
            "candidate_options_absolute": [list(bbox.as_tuple()) for bbox in option_boxes],
            "answer_index": answer_idx,
        }
        if elem_desc:
            metadata["elem_desc"] = elem_desc

        return GroundingSample(
            sample_id=sample_id,
            dataset_name=self.name,
            split=self.split,
            image_path=image_path,
            instruction=instruction,
            target_bbox=target_bbox,
            click_point=target_bbox.center if target_bbox else None,
            dom_candidates=_candidate_elements(task_type, option_boxes),
            website=str(raw_item.get("website") or "").strip() or None,
            platform="web",
            metadata=metadata,
        )
