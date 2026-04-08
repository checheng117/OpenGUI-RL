"""Multimodal-Mind2Web dataset adapter.

Loads the ``osunlp/Multimodal-Mind2Web`` dataset from HuggingFace and
converts each action row into the canonical :class:`GroundingSample`.

Each row in the HF dataset represents a **single action step** within a
multi-step web task.  Key raw fields:

    screenshot          PIL Image   — webpage screenshot
    confirmed_task      str         — natural-language task instruction
    operation           JSON str    — {"op": "CLICK", "value": "..."}
    pos_candidates      list[str]   — JSON-encoded positive elements with bbox
    neg_candidates      list[str]   — JSON-encoded negative candidate elements
    target_action_reprs str         — "[tag] text -> OP" summary
    target_action_index str         — which step this action is in the task
    action_uid          str         — unique ID for this action
    annotation_id       str         — annotation session ID
    website             str         — website name (e.g. "united")
    domain              str         — domain category (e.g. "Travel")
    subdomain           str         — subcategory (e.g. "Airlines")

Bounding boxes in ``pos_candidates`` / ``neg_candidates`` are stored as
``"x,y,width,height"`` (**XYWH absolute pixels**) inside a nested JSON
string.  This adapter converts them to **(x1, y1, x2, y2)** format.

Reference
---------
Deng et al., "Mind2Web: Towards a Generalist Agent for the Web", NeurIPS 2023.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from gui_grounding.data.schemas import (
    ActionType,
    BBox,
    CandidateElement,
    GroundingSample,
)
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# Action-type mapping  (Mind2Web uppercase → our enum)
# -----------------------------------------------------------------------
_OP_MAP: dict[str, ActionType] = {
    "CLICK": ActionType.CLICK,
    "TYPE": ActionType.TYPE,
    "SELECT": ActionType.SELECT,
    "HOVER": ActionType.HOVER,
}

# HuggingFace dataset identifier
HF_DATASET_ID = "osunlp/Multimodal-Mind2Web"

# Official splits
VALID_SPLITS = ("train", "test_task", "test_website", "test_domain")


# -----------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------

def _parse_bbox_xywh(bbox_str: str) -> Optional[BBox]:
    """Parse ``"x,y,w,h"`` string into a :class:`BBox` (x1,y1,x2,y2).

    Returns *None* on any parse failure rather than crashing.
    """
    try:
        parts = [float(p) for p in bbox_str.split(",")]
        if len(parts) != 4:
            return None
        x, y, w, h = parts
        if w < 0 or h < 0:
            logger.debug("Negative w/h in bbox: %s", bbox_str)
            return None
        return BBox(x1=x, y1=y, x2=x + w, y2=y + h)
    except (ValueError, TypeError) as exc:
        logger.debug("Failed to parse bbox '%s': %s", bbox_str, exc)
        return None


def _parse_candidate(raw_json: str) -> Optional[CandidateElement]:
    """Parse a single JSON-encoded candidate element."""
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    tag = obj.get("tag", "")
    node_id = str(obj.get("backend_node_id", ""))

    attrs_str = obj.get("attributes", "{}")
    try:
        attrs = json.loads(attrs_str) if isinstance(attrs_str, str) else attrs_str
    except json.JSONDecodeError:
        attrs = {}

    bbox = _parse_bbox_xywh(attrs.get("bounding_box_rect", ""))

    text_parts = []
    for key in ("aria_label", "placeholder", "title", "alt", "value"):
        if key in attrs and attrs[key]:
            text_parts.append(str(attrs[key]))
    text = " | ".join(text_parts) if text_parts else ""

    clean_attrs = {
        k: str(v) for k, v in attrs.items()
        if k not in ("bounding_box_rect", "backend_node_id", "data_pw_testid_buckeye_candidate")
    }

    return CandidateElement(
        element_id=node_id,
        bbox=bbox,
        text=text,
        tag=tag,
        attributes=clean_attrs,
    )


def _parse_operation(op_str: str) -> tuple[Optional[ActionType], str]:
    """Parse the ``operation`` JSON string.

    Returns ``(action_type, typed_value)``.
    """
    try:
        op = json.loads(op_str)
    except json.JSONDecodeError:
        return None, ""
    raw_op = op.get("op", op.get("original_op", ""))
    action_type = _OP_MAP.get(raw_op.upper())
    if action_type is None and raw_op:
        logger.debug("Unknown operation type: '%s'", raw_op)
    value = op.get("value", "")
    return action_type, value


def _bootstrap_hf_environment() -> Optional[str]:
    """Load local HF settings and return the best available token."""
    try:
        from dotenv import load_dotenv
        from gui_grounding.constants import PROJECT_ROOT

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    try:
        from huggingface_hub import get_token

        token = get_token()
    except ImportError:
        token = None

    if token:
        # Make the cached token visible to downstream HF calls that read env vars.
        os.environ.setdefault("HF_TOKEN", token)
    return token


def _normalize_screenshot(image: Optional[Image.Image | dict]) -> Optional[Image.Image]:
    """Detach the screenshot from any lazy file handle and normalize to RGB."""
    if image is None:
        return None

    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray)):
            with Image.open(BytesIO(image_bytes)) as pil_image:
                return pil_image.convert("RGB")
        return None

    image.load()
    normalized = image.convert("RGB")

    close = getattr(image, "close", None)
    if callable(close):
        close()

    return normalized


# -----------------------------------------------------------------------
# Public dataset class
# -----------------------------------------------------------------------

class Mind2WebDataset:
    """Adapter for the Multimodal-Mind2Web dataset.

    Parameters
    ----------
    split : str
        One of ``train``, ``test_task``, ``test_website``, ``test_domain``.
    max_samples : int, optional
        Limit the number of loaded samples (useful for debugging).
    cache_screenshots : bool
        If *True*, save screenshots to ``screenshot_dir`` on first load
        so that subsequent runs don't need to re-download.
    screenshot_dir : Path, optional
        Where to save/load cached screenshots.  Defaults to
        ``data/processed/mind2web_screenshots/{split}/``.
    max_candidates : int
        Maximum number of DOM candidates to keep per sample.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_screenshots: bool = True,
        screenshot_dir: Optional[str | Path] = None,
        max_candidates: int = 32,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from {VALID_SPLITS}")

        self.split = split
        self.max_samples = max_samples
        self.cache_screenshots = cache_screenshots
        self.max_candidates = max_candidates

        if screenshot_dir is None:
            from gui_grounding.constants import PROCESSED_DATA_DIR
            self.screenshot_dir = PROCESSED_DATA_DIR / "mind2web_screenshots" / split
        else:
            self.screenshot_dir = Path(screenshot_dir)

        self._samples: list[GroundingSample] = []
        self._screenshots: dict[str, Image.Image] = {}

        self._load()

    @property
    def name(self) -> str:
        return "mind2web"

    # -------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------

    def _load(self) -> None:
        token = _bootstrap_hf_environment()
        if self.cache_screenshots:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        local_parquets = self._resolve_local_parquet_files(token=token)
        if local_parquets:
            self._load_from_local_parquets(local_parquets)
            return

        self._load_from_hf_streaming(token=token)

    def _resolve_local_parquet_files(self, token: Optional[str]) -> list[Path]:
        if self.split != "train":
            return []
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return []

        try:
            snapshot_dir = Path(
                snapshot_download(
                    repo_id=HF_DATASET_ID,
                    repo_type="dataset",
                    token=token,
                    local_files_only=True,
                )
            )
        except Exception:
            return []

        data_dir = snapshot_dir / "data"
        files = sorted(data_dir.glob(f"{self.split}-*.parquet"))
        if not files:
            return []
        logger.info(
            "Using local Mind2Web parquet cache for split='%s' (%d shard files).",
            self.split,
            len(files),
        )
        return files

    def _load_from_hf_streaming(self, token: Optional[str]) -> None:
        """Load from HuggingFace datasets with streaming."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("Install HuggingFace datasets: pip install datasets")
            raise

        logger.info("Loading Mind2Web split='%s' from HuggingFace (streaming)...", self.split)
        ds = load_dataset(
            HF_DATASET_ID,
            split=self.split,
            streaming=True,
            token=token,
        )

        count = 0
        skipped = 0
        try:
            for row in ds:
                if self.max_samples is not None and count >= self.max_samples:
                    break
                try:
                    sample, screenshot = self._row_to_sample(row)
                    self._samples.append(sample)
                    if screenshot is not None:
                        self._handle_screenshot(sample.sample_id, screenshot)
                    count += 1
                except Exception as exc:
                    skipped += 1
                    uid = row.get("action_uid", "?")
                    logger.warning("Skipped row action_uid=%s: %s", uid, exc)
                    if skipped > 50:
                        logger.error("Too many skipped rows (>50). Aborting load.")
                        break
        finally:
            del ds

        logger.info(
            "Mind2Web loaded: %d samples, %d skipped (split=%s)",
            len(self._samples), skipped, self.split,
        )

    def _load_from_local_parquets(self, parquet_files: list[Path]) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Local Mind2Web parquet loading requires pyarrow.") from exc

        columns = [
            "screenshot",
            "confirmed_task",
            "operation",
            "pos_candidates",
            "neg_candidates",
            "target_action_reprs",
            "target_action_index",
            "action_uid",
            "annotation_id",
            "website",
            "domain",
            "subdomain",
        ]
        count = 0
        skipped = 0
        for parquet_path in parquet_files:
            if self.max_samples is not None and count >= self.max_samples:
                break
            logger.info("Reading local Mind2Web shard: %s", parquet_path.name)
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=16, columns=columns):
                for row in batch.to_pylist():
                    if self.max_samples is not None and count >= self.max_samples:
                        break
                    try:
                        sample, screenshot = self._row_to_sample(row)
                        self._samples.append(sample)
                        if screenshot is not None:
                            self._handle_screenshot(sample.sample_id, screenshot)
                        count += 1
                    except Exception as exc:
                        skipped += 1
                        uid = row.get("action_uid", "?")
                        logger.warning("Skipped local row action_uid=%s: %s", uid, exc)
                        if skipped > 50:
                            logger.error("Too many skipped local rows (>50). Aborting load.")
                            break
                if skipped > 50 or (self.max_samples is not None and count >= self.max_samples):
                    break
            if skipped > 50:
                break

        logger.info(
            "Mind2Web loaded from local parquet cache: %d samples, %d skipped (split=%s)",
            len(self._samples),
            skipped,
            self.split,
        )

    def _row_to_sample(self, row: dict) -> tuple[GroundingSample, Optional[Image.Image]]:
        """Convert a single HF dataset row to :class:`GroundingSample`."""
        action_uid = row.get("action_uid", row.get("annotation_id", "unknown"))
        sample_id = f"mind2web_{self.split}_{action_uid}"

        instruction = row.get("confirmed_task", "")
        action_type, typed_value = _parse_operation(row.get("operation", "{}"))

        # --- Target element bbox from pos_candidates ---
        target_bbox: Optional[BBox] = None
        target_element_id: Optional[str] = None
        pos_candidates_raw = row.get("pos_candidates", [])
        for c_str in pos_candidates_raw:
            cand = _parse_candidate(c_str)
            if cand is not None and cand.bbox is not None:
                target_bbox = cand.bbox
                target_element_id = cand.element_id
                break

        click_point = target_bbox.center if target_bbox else None

        # --- DOM candidates (pos + neg, up to max_candidates) ---
        dom_candidates: list[CandidateElement] = []
        for c_str in pos_candidates_raw:
            cand = _parse_candidate(c_str)
            if cand is not None:
                dom_candidates.append(cand)
        for c_str in row.get("neg_candidates", []):
            if len(dom_candidates) >= self.max_candidates:
                break
            cand = _parse_candidate(c_str)
            if cand is not None:
                dom_candidates.append(cand)

        # --- Screenshot ---
        screenshot = _normalize_screenshot(row.get("screenshot"))
        img_path = ""
        if screenshot is not None:
            img_path = str(self.screenshot_dir / f"{action_uid}.jpg")
        elif self.cache_screenshots:
            cached = self.screenshot_dir / f"{action_uid}.jpg"
            if cached.exists():
                img_path = str(cached)

        sample = GroundingSample(
            sample_id=sample_id,
            dataset_name="mind2web",
            split=self.split,
            image_path=img_path,
            instruction=instruction,
            action_type=action_type,
            target_element_id=target_element_id,
            target_bbox=target_bbox,
            click_point=click_point,
            dom_candidates=dom_candidates if dom_candidates else None,
            website=row.get("website"),
            domain=row.get("domain"),
            platform="web",
            metadata={
                "annotation_id": row.get("annotation_id", ""),
                "action_uid": action_uid,
                "target_action_index": row.get("target_action_index", ""),
                "target_action_reprs": row.get("target_action_reprs", ""),
                "subdomain": row.get("subdomain", ""),
                "typed_value": typed_value,
                "num_pos_candidates": len(pos_candidates_raw),
                "num_neg_candidates": len(row.get("neg_candidates", [])),
            },
        )
        return sample, screenshot

    def _handle_screenshot(
        self, sample_id: str, screenshot: Image.Image
    ) -> None:
        """Cache screenshot to disk or keep in memory."""
        action_uid = sample_id.replace(f"mind2web_{self.split}_", "")
        if self.cache_screenshots:
            out_path = self.screenshot_dir / f"{action_uid}.jpg"
            if not out_path.exists():
                screenshot.save(out_path, "JPEG", quality=90)
        else:
            self._screenshots[sample_id] = screenshot

    def get_screenshot(self, sample: GroundingSample) -> Optional[Image.Image]:
        """Return the PIL Image for a sample (from cache or disk)."""
        if sample.sample_id in self._screenshots:
            return self._screenshots[sample.sample_id]
        path = Path(sample.image_path)
        if path.exists():
            return Image.open(path).convert("RGB")
        logger.warning("Screenshot not found for %s at %s", sample.sample_id, path)
        return None

    # -------------------------------------------------------------------
    # Sequence interface
    # -------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GroundingSample:
        return self._samples[idx]

    def __iter__(self):
        return iter(self._samples)
