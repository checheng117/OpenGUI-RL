"""Qwen-VL based real single-step GUI grounding model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from gui_grounding.data.schemas import BBox, GroundingSample, PredictionResult
from gui_grounding.models.base_model import BaseGroundingModel
from gui_grounding.models.vlm_backbone import VLMBackbone
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)

_ACTION_TYPES = {"click", "type", "select", "hover"}
_POINT_BIN_KEYS = ("point_bin", "point_bins", "click_point_bin")
_BBOX_BIN_KEYS = ("bbox_bin", "bbox_bins", "bbox_proposal_bin")
_BBOX_KEYS = _BBOX_BIN_KEYS + ("bbox_proposal", "predicted_bbox")
_CLICK_KEYS = _POINT_BIN_KEYS + ("click_point", "predicted_click_point")
_ELEMENT_ID_KEYS = ("element_hint_id", "predicted_element_id")
_POINT_ONLY_CLICK_KEYS = _POINT_BIN_KEYS + (
    "point_2d",
    "click_point",
    "predicted_click_point",
    "click_point_normalized",
    "point",
)
_POINT_ONLY_BBOX_KEYS = _BBOX_BIN_KEYS + ("bbox_2d", "bbox_proposal", "predicted_bbox", "bbox", "bbox_normalized")

_STRUCTURED_CANDIDATE_SCHEMA = {
    "version": "bbox_click_action_v1",
    "primary_fields": ["bbox_proposal", "click_point", "action_type"],
    "legacy_fields": ["legacy_metadata.grid_id"],
}
_POINT_PRIMARY_CANDIDATE_SCHEMA = {
    "version": "bbox_click_action_v2_point_primary",
    "primary_fields": ["bbox_proposal", "click_point", "action_type"],
    "primary_prediction_field": "click_point",
    "secondary_fields": [
        "element_hint_id",
        "confidence",
        "click_point_provenance",
        "bbox_provenance",
        "action_type_provenance",
        "point_pass_confidence",
        "structure_pass_confidence",
    ],
    "legacy_fields": ["legacy_metadata.grid_id"],
}


def _clamp(v: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, v))


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in keys:
        if key in data:
            return key, data.get(key)
    return None, None


def _normalize_action_type(value: Any, response_text: str = "") -> tuple[str | None, str | None]:
    if value is not None:
        action = str(value).strip().lower()
        if action in _ACTION_TYPES:
            return action, "json_field"

    lower_text = response_text.lower()
    for action in _ACTION_TYPES:
        if action in lower_text:
            return action, "text_fallback"
    return None, None


def _resolve_point(
    click_data: Any,
    coord_size: tuple[int, int],
) -> tuple[tuple[float, float] | None, str | None]:
    width, height = coord_size
    if not isinstance(click_data, list) or len(click_data) != 2:
        return None, None

    cx = _safe_float(click_data[0])
    cy = _safe_float(click_data[1])
    if cx is None or cy is None:
        return None, None

    if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
        return (
            (_clamp(cx * width, 0.0, float(width)), _clamp(cy * height, 0.0, float(height))),
            "normalized",
        )

    if 0.0 <= cx <= float(width) and 0.0 <= cy <= float(height):
        return ((_clamp(cx, 0.0, float(width)), _clamp(cy, 0.0, float(height))), "absolute")

    return None, None


def _resolve_quantized_point(
    click_data: Any,
    coord_size: tuple[int, int],
    bins: int,
) -> tuple[tuple[float, float] | None, str | None]:
    width, height = coord_size
    if bins < 2 or not isinstance(click_data, list) or len(click_data) != 2:
        return None, None

    cx = _safe_float(click_data[0])
    cy = _safe_float(click_data[1])
    if cx is None or cy is None:
        return None, None
    if not (0.0 <= cx <= float(bins - 1) and 0.0 <= cy <= float(bins - 1)):
        return None, None

    return (
        (
            _clamp((cx / float(bins - 1)) * float(width), 0.0, float(width)),
            _clamp((cy / float(bins - 1)) * float(height), 0.0, float(height)),
        ),
        "quantized_bin",
    )


def _resolve_bbox(
    bbox_data: Any,
    coord_size: tuple[int, int],
) -> tuple[BBox | None, str | None]:
    width, height = coord_size
    if not isinstance(bbox_data, list) or len(bbox_data) != 4:
        return None, None

    raw_vals = [_safe_float(v) for v in bbox_data]
    if any(v is None for v in raw_vals):
        return None, None
    x1, y1, x2, y2 = raw_vals  # type: ignore[misc]

    if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
        x1, x2 = x1 * float(width), x2 * float(width)
        y1, y2 = y1 * float(height), y2 * float(height)
        mode = "normalized"
    else:
        mode = "absolute"

    x1 = _clamp(x1, 0.0, float(width))
    y1 = _clamp(y1, 0.0, float(height))
    x2 = _clamp(x2, 0.0, float(width))
    y2 = _clamp(y2, 0.0, float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None, None

    return BBox(x1=x1, y1=y1, x2=x2, y2=y2), mode


def _resolve_quantized_bbox(
    bbox_data: Any,
    coord_size: tuple[int, int],
    bins: int,
) -> tuple[BBox | None, str | None]:
    width, height = coord_size
    if bins < 2 or not isinstance(bbox_data, list) or len(bbox_data) != 4:
        return None, None

    raw_vals = [_safe_float(v) for v in bbox_data]
    if any(v is None for v in raw_vals):
        return None, None
    x1, y1, x2, y2 = raw_vals  # type: ignore[misc]
    if any(not (0.0 <= v <= float(bins - 1)) for v in (x1, y1, x2, y2)):
        return None, None

    x1 = (x1 / float(bins - 1)) * float(width)
    y1 = (y1 / float(bins - 1)) * float(height)
    x2 = (x2 / float(bins - 1)) * float(width)
    y2 = (y2 / float(bins - 1)) * float(height)
    x1 = _clamp(x1, 0.0, float(width))
    y1 = _clamp(y1, 0.0, float(height))
    x2 = _clamp(x2, 0.0, float(width))
    y2 = _clamp(y2, 0.0, float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None, None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2), "quantized_bin"


def _derive_bbox_from_click(
    pred_click: tuple[float, float],
    coord_size: tuple[int, int],
    delta: float = 12.0,
) -> BBox:
    width, height = coord_size
    cx, cy = pred_click
    return BBox(
        x1=_clamp(cx - delta, 0.0, float(width)),
        y1=_clamp(cy - delta, 0.0, float(height)),
        x2=_clamp(cx + delta, 0.0, float(width)),
        y2=_clamp(cy + delta, 0.0, float(height)),
    )


def _bbox_contains_click(pred_bbox: BBox | None, pred_click: tuple[float, float] | None) -> bool:
    if pred_bbox is None or pred_click is None:
        return False
    cx, cy = pred_click
    return pred_bbox.x1 <= cx <= pred_bbox.x2 and pred_bbox.y1 <= cy <= pred_bbox.y2


def _ensure_bbox_contains_click(
    pred_bbox: BBox | None,
    pred_click: tuple[float, float] | None,
) -> tuple[BBox | None, dict[str, Any] | None]:
    if pred_bbox is None or pred_click is None:
        return pred_bbox, None
    if _bbox_contains_click(pred_bbox, pred_click):
        return pred_bbox, {
            "applied": False,
            "reason": "bbox_already_contains_primary_click",
        }

    cx, cy = pred_click
    reconciled = BBox(
        x1=min(pred_bbox.x1, cx),
        y1=min(pred_bbox.y1, cy),
        x2=max(pred_bbox.x2, cx),
        y2=max(pred_bbox.y2, cy),
    )
    return reconciled, {
        "applied": True,
        "reason": "expanded_bbox_to_include_primary_click",
        "original_bbox": list(pred_bbox.as_tuple()),
        "primary_click_point": [cx, cy],
        "reconciled_bbox": list(reconciled.as_tuple()),
    }


def _map_prediction_to_original(
    pred_bbox: BBox | None,
    pred_click: tuple[float, float] | None,
    image_size: tuple[int, int],
    coord_size: tuple[int, int],
) -> tuple[BBox | None, tuple[float, float] | None]:
    width, height = image_size
    coord_width, coord_height = coord_size
    if (coord_width, coord_height) == (width, height):
        return pred_bbox, pred_click

    scale_x = float(width) / float(coord_width)
    scale_y = float(height) / float(coord_height)

    mapped_bbox = pred_bbox
    if pred_bbox is not None:
        mapped_bbox = BBox(
            x1=_clamp(pred_bbox.x1 * scale_x, 0.0, float(width)),
            y1=_clamp(pred_bbox.y1 * scale_y, 0.0, float(height)),
            x2=_clamp(pred_bbox.x2 * scale_x, 0.0, float(width)),
            y2=_clamp(pred_bbox.y2 * scale_y, 0.0, float(height)),
        )

    mapped_click = pred_click
    if pred_click is not None:
        mapped_click = (
            _clamp(pred_click[0] * scale_x, 0.0, float(width)),
            _clamp(pred_click[1] * scale_y, 0.0, float(height)),
        )

    return mapped_bbox, mapped_click


def _refine_edge_click_in_bbox(
    pred_click: tuple[float, float] | None,
    pred_bbox: BBox | None,
    threshold: float,
    interior_position: float,
) -> tuple[tuple[float, float] | None, dict[str, float | bool | str] | None]:
    if pred_click is None or pred_bbox is None or threshold <= 0.0:
        return pred_click, None

    width = max(pred_bbox.x2 - pred_bbox.x1, 1e-6)
    height = max(pred_bbox.y2 - pred_bbox.y1, 1e-6)
    rel_x = (pred_click[0] - pred_bbox.x1) / width
    rel_y = (pred_click[1] - pred_bbox.y1) / height
    if rel_x > threshold or rel_y > threshold:
        return pred_click, {
            "applied": False,
            "reason": "click_not_in_top_left_band",
            "threshold": threshold,
            "interior_position": interior_position,
            "relative_x": rel_x,
            "relative_y": rel_y,
        }

    refined_click = (
        _clamp(pred_bbox.x1 + interior_position * width, pred_bbox.x1, pred_bbox.x2),
        _clamp(pred_bbox.y1 + interior_position * height, pred_bbox.y1, pred_bbox.y2),
    )
    return refined_click, {
        "applied": True,
        "reason": "top_left_edge_click_moved_inward",
        "threshold": threshold,
        "interior_position": interior_position,
        "relative_x": rel_x,
        "relative_y": rel_y,
        "original_x": pred_click[0],
        "original_y": pred_click[1],
        "refined_x": refined_click[0],
        "refined_y": refined_click[1],
    }


def _build_web_mobile_hotspot_instruction(sample: GroundingSample) -> str:
    platform = str(sample.platform or "").strip().lower()
    if platform not in {"web", "mobile"}:
        return ""

    instruction = (
        "Web/mobile hotspot rule: click the center of the actual clickable or tappable hotspot.\n"
        "Avoid the left or top edge, borders, row/container edges, and nearby whitespace.\n"
    )
    element_type = str(sample.metadata.get("element_type") or "").strip().lower()
    if element_type == "text":
        instruction += (
            "For text targets, click the middle of the intended label or button text region, "
            "not the first character or left edge.\n"
        )
    elif element_type == "icon":
        instruction += (
            "For icon targets, click the visual center of the icon button or glyph itself, "
            "not the surrounding padding, row, or card area.\n"
        )
    return instruction


class QwenVLGroundingModel(BaseGroundingModel):
    """Qwen-VL wrapper that predicts structured GUI actions."""

    def __init__(
        self,
        model_name: str = "qwen2_5_vl_3b",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        attn_implementation: str = "sdpa",
        gpu_memory_utilization: float = 0.9,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        log_generate_diagnostics: bool = False,
        coordinate_frame: str = "original",
        coordinate_format: str = "absolute",
        point_first_prompt: bool = False,
        web_mobile_hotspot_prompt: bool = False,
        decoupled_point_native_decode: bool = False,
        coordinate_quantization_bins: int | None = None,
        point_native_secondary_bbox_only: bool = False,
        edge_click_interior_threshold: float = 0.0,
        edge_click_interior_position: float = 0.45,
        adapter_path: str | None = None,
    ) -> None:
        if coordinate_frame not in {"original", "model_resized"}:
            raise ValueError("coordinate_frame must be one of: original, model_resized")
        if coordinate_format not in {"absolute", "normalized"}:
            raise ValueError("coordinate_format must be one of: absolute, normalized")
        if not 0.0 <= edge_click_interior_threshold <= 1.0:
            raise ValueError("edge_click_interior_threshold must be in [0, 1].")
        if not 0.0 <= edge_click_interior_position <= 1.0:
            raise ValueError("edge_click_interior_position must be in [0, 1].")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.coordinate_frame = coordinate_frame
        self.coordinate_format = coordinate_format
        self.point_first_prompt = point_first_prompt
        self.web_mobile_hotspot_prompt = web_mobile_hotspot_prompt
        self.decoupled_point_native_decode = decoupled_point_native_decode
        self.coordinate_quantization_bins = (
            int(coordinate_quantization_bins) if coordinate_quantization_bins is not None else None
        )
        self.point_native_secondary_bbox_only = bool(point_native_secondary_bbox_only)
        self.edge_click_interior_threshold = edge_click_interior_threshold
        self.edge_click_interior_position = edge_click_interior_position
        self.candidate_schema = (
            _POINT_PRIMARY_CANDIDATE_SCHEMA if decoupled_point_native_decode else _STRUCTURED_CANDIDATE_SCHEMA
        )
        self.candidate_semantics = (
            "click_point_primary_bbox_proposal_action_type"
            if decoupled_point_native_decode
            else "bbox_proposal_click_point_action_type"
        )
        self.backbone = VLMBackbone(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            load_model=True,
            attn_implementation=attn_implementation,
            gpu_memory_utilization=gpu_memory_utilization,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            log_generate_diagnostics=log_generate_diagnostics,
            adapter_path=adapter_path,
        )

    def _get_model_coordinate_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        width, height = image_size
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            factor=28,
            min_pixels=self.backbone.min_pixels or (56 * 56),
            max_pixels=self.backbone.max_pixels or (14 * 14 * 4 * 1280),
        )
        return resized_width, resized_height

    def _get_coordinate_instruction(self, image_size: tuple[int, int]) -> tuple[str, tuple[int, int]]:
        width, height = image_size
        coord_size = image_size
        if self.coordinate_quantization_bins is not None:
            bins = self.coordinate_quantization_bins
            if self.coordinate_frame == "model_resized":
                model_width, model_height = self._get_model_coordinate_size(image_size)
                coord_size = (model_width, model_height)
                coordinate_instruction = (
                    f"Original screenshot size: width={width}, height={height}.\n"
                    f"Model-view resized screenshot size: width={model_width}, height={model_height}.\n"
                    f"Coordinates must be integer bins in [0, {bins - 1}] relative to the model-view resized screenshot.\n"
                    f"Interpret each bin by dividing by {bins - 1} to recover normalized coordinates.\n"
                    "These coordinates will be mapped back to the original screenshot after parsing.\n"
                )
            else:
                coordinate_instruction = (
                    f"Screenshot size: width={width}, height={height}.\n"
                    f"Coordinates must be integer bins in [0, {bins - 1}] relative to the screenshot.\n"
                    f"Interpret each bin by dividing by {bins - 1} to recover normalized coordinates.\n"
                )
            return coordinate_instruction, coord_size

        if self.coordinate_format == "absolute":
            coordinate_instruction = (
                f"Screenshot size: width={width}, height={height}.\n"
                "Coordinates must be absolute pixel coordinates within the screenshot bounds.\n"
            )
        else:
            coordinate_instruction = (
                f"Screenshot size: width={width}, height={height}.\n"
                "Coordinates must be normalized floats in [0, 1].\n"
                "Use x values relative to screenshot width and y values relative to screenshot height.\n"
            )
        if self.coordinate_frame == "model_resized":
            model_width, model_height = self._get_model_coordinate_size(image_size)
            coord_size = (model_width, model_height)
            if self.coordinate_format == "absolute":
                coordinate_instruction = (
                    f"Original screenshot size: width={width}, height={height}.\n"
                    f"Model-view resized screenshot size: width={model_width}, height={model_height}.\n"
                    "Output coordinates in the model-view resized screenshot coordinate system.\n"
                    "The coordinates will be mapped back to the original screenshot after parsing.\n"
                )
            else:
                coordinate_instruction = (
                    f"Original screenshot size: width={width}, height={height}.\n"
                    f"Model-view resized screenshot size: width={model_width}, height={model_height}.\n"
                    "Coordinates must be normalized floats in [0, 1] relative to the model-view resized screenshot.\n"
                    "They will be mapped back to the original screenshot after parsing.\n"
                )
        return coordinate_instruction, coord_size

    def _build_prompt(
        self,
        sample: GroundingSample,
        image_size: tuple[int, int],
        *,
        point_first_prompt: bool | None = None,
        web_mobile_hotspot_prompt: bool | None = None,
    ) -> str:
        coordinate_instruction, _ = self._get_coordinate_instruction(image_size)
        resolved_point_first_prompt = self.point_first_prompt if point_first_prompt is None else bool(point_first_prompt)
        resolved_web_mobile_hotspot_prompt = (
            self.web_mobile_hotspot_prompt
            if web_mobile_hotspot_prompt is None
            else bool(web_mobile_hotspot_prompt)
        )
        point_priority_instruction = ""
        if resolved_point_first_prompt:
            if self.coordinate_quantization_bins is not None:
                point_priority_instruction = (
                    "Primary goal: place point_bin on the exact spot a user should click.\n"
                    "The point_bin must lie inside the actionable interior of the target, preferably near its center.\n"
                    "Do not place point_bin on the bbox edge unless the target is extremely tiny.\n"
                    "Choose point_bin first, then make bbox_bin enclose that clicked target region.\n"
                )
            else:
                point_priority_instruction = (
                    "Primary goal: place predicted_click_point on the exact spot a user should click.\n"
                    "The click point must lie inside the actionable interior of the target, preferably near its center.\n"
                    "Do not place the click point on the bbox corner or border unless the target is extremely tiny.\n"
                    "Choose the click point first, then make predicted_bbox enclose that clicked target region.\n"
                )
        hotspot_instruction = ""
        if resolved_web_mobile_hotspot_prompt:
            hotspot_instruction = _build_web_mobile_hotspot_instruction(sample)
        if self.coordinate_quantization_bins is not None:
            schema_lines = [
                '  "action_type": "click|type|select|hover",',
                '  "bbox_bin": [x1_bin, y1_bin, x2_bin, y2_bin],',
                '  "point_bin": [x_bin, y_bin]',
            ]
            if resolved_point_first_prompt:
                schema_lines = [
                    '  "point_bin": [x_bin, y_bin],',
                    '  "action_type": "click|type|select|hover",',
                    '  "bbox_bin": [x1_bin, y1_bin, x2_bin, y2_bin]',
                ]
        else:
            schema_lines = [
                '  "action_type": "click|type|select|hover",',
                '  "predicted_element_id": "string or null",',
                '  "predicted_bbox": [x1, y1, x2, y2],',
                '  "predicted_click_point": [x, y],',
                '  "confidence": 0.0',
            ]
            if resolved_point_first_prompt:
                schema_lines = [
                    '  "predicted_click_point": [x, y],',
                    '  "predicted_bbox": [x1, y1, x2, y2],',
                    '  "action_type": "click|type|select|hover",',
                    '  "predicted_element_id": "string or null",',
                    '  "confidence": 0.0',
                ]
        schema_text = "{\n" + "\n".join(schema_lines) + "\n}\n\n"
        return (
            "You are a GUI grounding model.\n"
            "Given ONE screenshot and ONE instruction, predict the target element and next action.\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            f"{schema_text}"
            f"{coordinate_instruction}"
            f"{point_priority_instruction}"
            f"{hotspot_instruction}"
            "If uncertain, still provide your best grounded guess.\n\n"
            f"Instruction: {sample.instruction}\n"
        )

    def _build_point_native_prompt(self, sample: GroundingSample, image_size: tuple[int, int]) -> str:
        coordinate_instruction, _ = self._get_coordinate_instruction(image_size)
        if self.coordinate_quantization_bins is not None:
            return (
                "You are a GUI grounding model.\n"
                "Given one screenshot and one instruction, identify the target UI element.\n\n"
                "Return ONLY valid JSON with this exact schema:\n"
                "{\n"
                '  "point_bin": [x_bin, y_bin],\n'
                '  "action_type": "click|type|select|hover"\n'
                "}\n\n"
                f"{coordinate_instruction}"
                "Primary goal: predict the single best click point a user should use.\n"
                "Do not output a bounding box in this pass.\n\n"
                f"Instruction: {sample.instruction}\n"
            )
        return (
            "You are a GUI grounding model.\n"
            "Given one screenshot and one instruction, identify the target UI element.\n\n"
            "Return ONLY valid JSON in Qwen grounding style with this exact schema:\n"
            "{\n"
            '  "point_2d": [x, y],\n'
            '  "label": "target",\n'
            '  "action_type": "click|type|select|hover",\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            f"{coordinate_instruction}"
            "Primary goal: predict the single best click point a user should use.\n"
            "Do not output a bounding box in this pass.\n"
            "- do not output explanation\n\n"
            f"Instruction: {sample.instruction}\n"
        )

    def _build_secondary_structure_prompt(
        self,
        sample: GroundingSample,
        image_size: tuple[int, int],
        primary_click: tuple[float, float],
    ) -> str:
        coordinate_instruction, coord_size = self._get_coordinate_instruction(image_size)
        click_x, click_y = primary_click
        if self.coordinate_quantization_bins is not None:
            bins = self.coordinate_quantization_bins
            click_x = int(round((float(click_x) / max(float(coord_size[0]), 1.0)) * float(bins - 1)))
            click_y = int(round((float(click_y) / max(float(coord_size[1]), 1.0)) * float(bins - 1)))
            if self.point_native_secondary_bbox_only:
                return (
                    "You are a GUI grounding model.\n"
                    "A primary click point has already been chosen for this screenshot and instruction.\n"
                    "Keep that click point fixed and produce the supporting bbox only.\n\n"
                    "Return ONLY valid JSON with this exact schema:\n"
                    "{\n"
                    '  "bbox_bin": [x1_bin, y1_bin, x2_bin, y2_bin]\n'
                    "}\n\n"
                    f"{coordinate_instruction}"
                    f"Primary click point already selected: point_bin=[{click_x}, {click_y}].\n"
                    "Predict only the target bbox_bin that encloses the clicked UI element.\n"
                    "The bbox_bin must contain the selected point_bin.\n\n"
                    f"Instruction: {sample.instruction}\n"
                )
        elif self.coordinate_format == "normalized":
            click_x = round(float(click_x) / max(float(coord_size[0]), 1.0), 4)
            click_y = round(float(click_y) / max(float(coord_size[1]), 1.0), 4)
        else:
            click_x = int(round(primary_click[0]))
            click_y = int(round(primary_click[1]))
        return (
            "You are a GUI grounding model.\n"
            "A primary click point has already been chosen for this screenshot and instruction.\n"
            "Keep that click point fixed and produce the supporting structured action fields.\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  "action_type": "click|type|select|hover",\n'
            '  "predicted_element_id": "string or null",\n'
            '  "predicted_bbox": [x1, y1, x2, y2],\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            f"{coordinate_instruction}"
            f"Primary click point already selected: [{click_x}, {click_y}].\n"
            "Predict the target bbox that best encloses the clicked UI element.\n"
            "The bbox must contain the selected click point.\n"
            "Do not revise the click point in this pass.\n\n"
            f"Instruction: {sample.instruction}\n"
        )

    def _extract_json_dict(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and len(parsed) == 2:
                return {"point_2d": parsed}
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            block = text[start : end + 1]
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            block = text[start : end + 1]
            try:
                parsed = json.loads(block)
                if isinstance(parsed, list) and len(parsed) == 2:
                    return {"point_2d": parsed}
            except json.JSONDecodeError:
                return {}
        return {}

    def _parse_prediction(
        self,
        response_text: str,
        sample_id: str,
        image_size: tuple[int, int],
        *,
        point_first_prompt: bool | None = None,
        web_mobile_hotspot_prompt: bool | None = None,
    ) -> tuple[PredictionResult, dict[str, Any]]:
        _, coord_size = self._get_coordinate_instruction(image_size)
        data = self._extract_json_dict(response_text)
        action, action_source = _normalize_action_type(data.get("action_type"), response_text=response_text)

        _, element_id = _first_present(data, _ELEMENT_ID_KEYS)
        if element_id is not None:
            element_id = str(element_id)

        bbox_key, bbox_data = _first_present(data, _BBOX_KEYS)
        if bbox_key in _BBOX_BIN_KEYS and self.coordinate_quantization_bins is not None:
            pred_bbox, bbox_mode = _resolve_quantized_bbox(
                bbox_data,
                coord_size,
                self.coordinate_quantization_bins,
            )
        else:
            pred_bbox, bbox_mode = _resolve_bbox(bbox_data, coord_size)

        click_key, click_data = _first_present(data, _CLICK_KEYS)
        if click_key in _POINT_BIN_KEYS and self.coordinate_quantization_bins is not None:
            pred_click, click_mode = _resolve_quantized_point(
                click_data,
                coord_size,
                self.coordinate_quantization_bins,
            )
        else:
            pred_click, click_mode = _resolve_point(click_data, coord_size)

        if pred_click is None and pred_bbox is not None:
            pred_click = pred_bbox.center
            click_mode = "derived_from_bbox"

        if pred_bbox is None and pred_click is not None:
            pred_bbox = _derive_bbox_from_click(pred_click, coord_size)
            bbox_mode = "derived_from_click"

        pred_bbox, pred_click = _map_prediction_to_original(
            pred_bbox=pred_bbox,
            pred_click=pred_click,
            image_size=image_size,
            coord_size=coord_size,
        )

        point_refinement = None
        pred_click, point_refinement = _refine_edge_click_in_bbox(
            pred_click=pred_click,
            pred_bbox=pred_bbox,
            threshold=self.edge_click_interior_threshold,
            interior_position=self.edge_click_interior_position,
        )

        confidence = _safe_float(data.get("confidence"))
        if confidence is not None:
            confidence = _clamp(confidence, 0.0, 1.0)

        if data:
            data = dict(data)
            data["_resolved_coordinate_frame"] = self.coordinate_frame
            data["_resolved_coordinate_format"] = self.coordinate_format
            data["_resolved_action_source"] = action_source
            data["_resolved_click_mode"] = click_mode
            data["_resolved_bbox_mode"] = bbox_mode
            data["_point_first_prompt"] = (
                self.point_first_prompt if point_first_prompt is None else bool(point_first_prompt)
            )
            data["_web_mobile_hotspot_prompt"] = (
                self.web_mobile_hotspot_prompt
                if web_mobile_hotspot_prompt is None
                else bool(web_mobile_hotspot_prompt)
            )
            if point_refinement is not None:
                data["_click_point_refinement"] = point_refinement

        result = PredictionResult(
            sample_id=sample_id,
            predicted_action_type=action,
            predicted_bbox=pred_bbox,
            predicted_click_point=pred_click,
            predicted_element_id=element_id,
            confidence=confidence,
        )
        return result, data

    def _parse_point_native_pass(
        self,
        response_text: str,
        image_size: tuple[int, int],
    ) -> tuple[dict[str, Any], tuple[int, int]]:
        _, coord_size = self._get_coordinate_instruction(image_size)
        data = self._extract_json_dict(response_text)
        click_key, click_data = _first_present(data, _POINT_ONLY_CLICK_KEYS)
        if click_key in _POINT_BIN_KEYS and self.coordinate_quantization_bins is not None:
            pred_click, click_mode = _resolve_quantized_point(
                click_data,
                coord_size,
                self.coordinate_quantization_bins,
            )
        else:
            pred_click, click_mode = _resolve_point(click_data, coord_size)
        bbox_key, bbox_data = _first_present(data, _POINT_ONLY_BBOX_KEYS)
        if bbox_key in _BBOX_BIN_KEYS and self.coordinate_quantization_bins is not None:
            pred_bbox, bbox_mode = _resolve_quantized_bbox(
                bbox_data,
                coord_size,
                self.coordinate_quantization_bins,
            )
        else:
            pred_bbox, bbox_mode = _resolve_bbox(bbox_data, coord_size)
        action, action_source = _normalize_action_type(data.get("action_type"), response_text=response_text)
        confidence = _safe_float(data.get("confidence"))
        if confidence is not None:
            confidence = _clamp(confidence, 0.0, 1.0)
        return (
            {
                "parsed_payload": data,
                "click_point": pred_click,
                "click_mode": click_mode,
                "click_key": click_key,
                "bbox": pred_bbox,
                "bbox_mode": bbox_mode,
                "bbox_key": bbox_key,
                "action_type": action,
                "action_source": action_source,
                "confidence": confidence,
            },
            coord_size,
        )

    def _parse_structure_only_pass(
        self,
        response_text: str,
        image_size: tuple[int, int],
    ) -> tuple[dict[str, Any], tuple[int, int]]:
        _, coord_size = self._get_coordinate_instruction(image_size)
        data = self._extract_json_dict(response_text)
        action, action_source = _normalize_action_type(data.get("action_type"), response_text=response_text)
        _, element_id = _first_present(data, _ELEMENT_ID_KEYS)
        if element_id is not None:
            element_id = str(element_id)
        bbox_key, bbox_data = _first_present(data, _BBOX_KEYS)
        if bbox_key in _BBOX_BIN_KEYS and self.coordinate_quantization_bins is not None:
            pred_bbox, bbox_mode = _resolve_quantized_bbox(
                bbox_data,
                coord_size,
                self.coordinate_quantization_bins,
            )
        else:
            pred_bbox, bbox_mode = _resolve_bbox(bbox_data, coord_size)
        confidence = _safe_float(data.get("confidence"))
        if confidence is not None:
            confidence = _clamp(confidence, 0.0, 1.0)
        return (
            {
                "parsed_payload": data,
                "bbox": pred_bbox,
                "bbox_mode": bbox_mode,
                "bbox_key": bbox_key,
                "action_type": action,
                "action_source": action_source,
                "element_id": element_id,
                "confidence": confidence,
            },
            coord_size,
        )

    def _predict_decoupled_point_native(
        self,
        sample: GroundingSample,
        image: Image.Image,
        temperature: float | None = None,
        *,
        point_first_prompt: bool | None = None,
        web_mobile_hotspot_prompt: bool | None = None,
    ) -> tuple[PredictionResult, str, dict[str, Any]]:
        generation_temperature = self.temperature if temperature is None else float(temperature)
        resolved_point_first_prompt = self.point_first_prompt if point_first_prompt is None else bool(point_first_prompt)
        resolved_web_mobile_hotspot_prompt = (
            self.web_mobile_hotspot_prompt
            if web_mobile_hotspot_prompt is None
            else bool(web_mobile_hotspot_prompt)
        )
        point_prompt = self._build_point_native_prompt(sample=sample, image_size=image.size)
        point_outputs = self.backbone.generate(
            images=[image],
            prompts=[point_prompt],
            max_new_tokens=self.max_new_tokens,
            temperature=generation_temperature,
            num_return_sequences=1,
        )
        point_raw_text = point_outputs[0] if point_outputs else ""
        point_pass, coord_size = self._parse_point_native_pass(point_raw_text, image.size)

        primary_click = point_pass["click_point"]
        if primary_click is None:
            fallback_prompt = self._build_prompt(
                sample=sample,
                image_size=image.size,
                point_first_prompt=resolved_point_first_prompt,
                web_mobile_hotspot_prompt=resolved_web_mobile_hotspot_prompt,
            )
            fallback_outputs = self.backbone.generate(
                images=[image],
                prompts=[fallback_prompt],
                max_new_tokens=self.max_new_tokens,
                temperature=generation_temperature,
                num_return_sequences=1,
            )
            fallback_raw_text = fallback_outputs[0] if fallback_outputs else ""
            fallback_pred, fallback_parsed = self._parse_prediction(
                response_text=fallback_raw_text,
                sample_id=sample.sample_id,
                image_size=image.size,
                point_first_prompt=resolved_point_first_prompt,
                web_mobile_hotspot_prompt=resolved_web_mobile_hotspot_prompt,
            )
            if fallback_parsed:
                fallback_parsed = dict(fallback_parsed)
                fallback_parsed["_decoupled_point_native_decode"] = True
                fallback_parsed["_point_first_prompt"] = resolved_point_first_prompt
                fallback_parsed["_web_mobile_hotspot_prompt"] = resolved_web_mobile_hotspot_prompt
                fallback_parsed["_resolved_click_provenance"] = "structured_single_pass_fallback"
                fallback_parsed["_resolved_bbox_provenance"] = "structured_single_pass_fallback"
                fallback_parsed["_resolved_action_provenance"] = "structured_single_pass_fallback"
                fallback_parsed["_primary_point_pass"] = {
                    "raw_response": point_raw_text,
                    "parsed_payload": point_pass["parsed_payload"],
                    "click_key": point_pass["click_key"],
                    "click_mode": point_pass["click_mode"],
                    "click_point": list(point_pass["click_point"]) if point_pass["click_point"] is not None else None,
                    "bbox_key": point_pass["bbox_key"],
                    "bbox_mode": point_pass["bbox_mode"],
                    "bbox": list(point_pass["bbox"].as_tuple()) if point_pass["bbox"] is not None else None,
                    "action_type": point_pass["action_type"],
                    "action_source": point_pass["action_source"],
                    "confidence": point_pass["confidence"],
                }
            combined_raw_text = (
                "[point_native_primary_pass]\n"
                f"{point_raw_text}\n\n"
                "[structured_single_pass_fallback]\n"
                f"{fallback_raw_text}"
            )
            return fallback_pred, combined_raw_text, fallback_parsed

        structure_prompt = self._build_secondary_structure_prompt(
            sample=sample,
            image_size=image.size,
            primary_click=primary_click,
        )
        structure_outputs = self.backbone.generate(
            images=[image],
            prompts=[structure_prompt],
            max_new_tokens=self.max_new_tokens,
            temperature=generation_temperature,
            num_return_sequences=1,
        )
        structure_raw_text = structure_outputs[0] if structure_outputs else ""
        structure_pass, _ = self._parse_structure_only_pass(structure_raw_text, image.size)

        pred_bbox = structure_pass["bbox"]
        bbox_provenance = "structured_secondary_pass"
        if pred_bbox is None:
            pred_bbox = point_pass["bbox"]
            if pred_bbox is not None:
                bbox_provenance = "point_pass_bbox_fallback"
        if pred_bbox is None:
            pred_bbox = _derive_bbox_from_click(primary_click, coord_size)
            bbox_provenance = "derived_from_primary_click"

        pred_bbox, bbox_reconciliation = _ensure_bbox_contains_click(pred_bbox, primary_click)
        pred_action = structure_pass["action_type"]
        action_provenance = "structured_secondary_pass"
        if pred_action is None:
            pred_action = point_pass["action_type"]
            action_provenance = "point_native_primary_pass"
        if pred_action is None:
            pred_action = "click"
            action_provenance = "default_click_fallback"

        pred_element_id = structure_pass["element_id"]
        if pred_element_id is None:
            _, point_element_id = _first_present(point_pass["parsed_payload"], _ELEMENT_ID_KEYS)
            pred_element_id = str(point_element_id) if point_element_id is not None else None

        mapped_bbox, mapped_click = _map_prediction_to_original(
            pred_bbox=pred_bbox,
            pred_click=primary_click,
            image_size=image.size,
            coord_size=coord_size,
        )
        mapped_bbox, bbox_reconciliation_after_map = _ensure_bbox_contains_click(mapped_bbox, mapped_click)

        final_confidence = structure_pass["confidence"]
        if final_confidence is None:
            final_confidence = point_pass["confidence"]

        parsed_payload = {
            "_decoupled_point_native_decode": True,
            "_point_first_prompt": resolved_point_first_prompt,
            "_web_mobile_hotspot_prompt": resolved_web_mobile_hotspot_prompt,
            "_resolved_coordinate_frame": self.coordinate_frame,
            "_resolved_coordinate_format": self.coordinate_format,
            "_resolved_click_provenance": "point_native_primary_pass",
            "_resolved_bbox_provenance": bbox_provenance,
            "_resolved_action_provenance": action_provenance,
            "_primary_point_pass": {
                "parsed_payload": point_pass["parsed_payload"],
                "click_key": point_pass["click_key"],
                "click_mode": point_pass["click_mode"],
                "click_point": list(primary_click),
                "bbox_key": point_pass["bbox_key"],
                "bbox_mode": point_pass["bbox_mode"],
                "bbox": list(point_pass["bbox"].as_tuple()) if point_pass["bbox"] is not None else None,
                "action_type": point_pass["action_type"],
                "action_source": point_pass["action_source"],
                "confidence": point_pass["confidence"],
            },
            "_secondary_structure_pass": {
                "parsed_payload": structure_pass["parsed_payload"],
                "bbox_key": structure_pass["bbox_key"],
                "bbox_mode": structure_pass["bbox_mode"],
                "bbox": list(structure_pass["bbox"].as_tuple()) if structure_pass["bbox"] is not None else None,
                "action_type": structure_pass["action_type"],
                "action_source": structure_pass["action_source"],
                "element_id": structure_pass["element_id"],
                "confidence": structure_pass["confidence"],
            },
            "_bbox_click_reconciliation_before_map": bbox_reconciliation,
            "_bbox_click_reconciliation_after_map": bbox_reconciliation_after_map,
            "action_type": pred_action,
            "predicted_element_id": pred_element_id,
            "predicted_bbox": list(mapped_bbox.as_tuple()) if mapped_bbox is not None else None,
            "predicted_click_point": list(mapped_click) if mapped_click is not None else None,
            "confidence": final_confidence,
        }

        combined_raw_text = (
            "[point_native_primary_pass]\n"
            f"{point_raw_text}\n\n"
            "[structured_support_pass]\n"
            f"{structure_raw_text}"
        )
        result = PredictionResult(
            sample_id=sample.sample_id,
            predicted_action_type=pred_action,
            predicted_bbox=mapped_bbox,
            predicted_click_point=mapped_click,
            predicted_element_id=pred_element_id,
            confidence=final_confidence,
        )
        return result, combined_raw_text, parsed_payload

    def predict_with_details(
        self,
        sample: GroundingSample,
        temperature: float | None = None,
        *,
        point_first_prompt: bool | None = None,
        web_mobile_hotspot_prompt: bool | None = None,
        decoupled_point_native_decode: bool | None = None,
    ) -> tuple[PredictionResult, str, dict[str, Any]]:
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Sample image does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        resolved_point_first_prompt = self.point_first_prompt if point_first_prompt is None else bool(point_first_prompt)
        resolved_web_mobile_hotspot_prompt = (
            self.web_mobile_hotspot_prompt
            if web_mobile_hotspot_prompt is None
            else bool(web_mobile_hotspot_prompt)
        )
        resolved_decoupled = (
            self.decoupled_point_native_decode
            if decoupled_point_native_decode is None
            else bool(decoupled_point_native_decode)
        )
        if resolved_decoupled:
            return self._predict_decoupled_point_native(
                sample=sample,
                image=image,
                temperature=temperature,
                point_first_prompt=resolved_point_first_prompt,
                web_mobile_hotspot_prompt=resolved_web_mobile_hotspot_prompt,
            )

        prompt = self._build_prompt(
            sample=sample,
            image_size=image.size,
            point_first_prompt=resolved_point_first_prompt,
            web_mobile_hotspot_prompt=resolved_web_mobile_hotspot_prompt,
        )
        outputs = self.backbone.generate(
            images=[image],
            prompts=[prompt],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if temperature is None else float(temperature),
            num_return_sequences=1,
        )
        raw_text = outputs[0] if outputs else ""
        pred, parsed = self._parse_prediction(
            response_text=raw_text,
            sample_id=sample.sample_id,
            image_size=image.size,
            point_first_prompt=resolved_point_first_prompt,
            web_mobile_hotspot_prompt=resolved_web_mobile_hotspot_prompt,
        )
        return pred, raw_text, parsed

    def predict(
        self,
        sample: GroundingSample,
        temperature: float | None = None,
        **kwargs,
    ) -> PredictionResult:
        pred, _, _ = self.predict_with_details(sample, temperature=temperature, **kwargs)
        return pred

    def predict_batch(
        self,
        batch: dict[str, Any],
        **kwargs,
    ) -> list[PredictionResult]:
        samples = batch.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("batch['samples'] must be a list[GroundingSample].")
        preds = []
        for sample in samples:
            if not isinstance(sample, GroundingSample):
                raise TypeError("predict_batch expects GroundingSample objects.")
            preds.append(self.predict(sample))
        return preds

    def generate_candidates(
        self,
        sample: GroundingSample,
        num_candidates: int = 5,
        **kwargs,
    ) -> list[PredictionResult]:
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Sample image does not exist: {image_path}")
        preds: list[PredictionResult] = []
        if self.decoupled_point_native_decode:
            for _ in range(num_candidates):
                preds.append(self.predict(sample))
        else:
            image = Image.open(image_path).convert("RGB")
            prompt = self._build_prompt(sample=sample, image_size=image.size)
            outputs = self.backbone.generate(
                images=[image] * num_candidates,
                prompts=[prompt] * num_candidates,
                max_new_tokens=self.max_new_tokens,
                temperature=max(self.temperature, 0.3),
                num_return_sequences=1,
            )
            for out in outputs:
                pred, _ = self._parse_prediction(out, sample.sample_id, image.size)
                preds.append(pred)
        return preds


# Backward-compatible alias: prefer QwenVLGroundingModel in new code.
Qwen2VLGroundingModel = QwenVLGroundingModel
