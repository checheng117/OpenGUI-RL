"""Plain public Qwen2.5-VL point baseline for ScreenSpot-style evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from gui_grounding.data.schemas import BBox, GroundingSample, PredictionResult
from gui_grounding.models.base_model import BaseGroundingModel
from gui_grounding.models.vlm_backbone import VLMBackbone

_ACTION_TYPES = {"click", "type", "select", "hover"}
_CLICK_KEYS = ("point_2d", "click_point", "predicted_click_point", "click_point_normalized", "point")
_BBOX_KEYS = ("bbox_2d", "bbox_proposal", "predicted_bbox", "bbox", "bbox_normalized")
_ELEMENT_ID_KEYS = ("element_hint_id", "predicted_element_id")


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


class QwenVLPublicPointBaselineModel(BaseGroundingModel):
    """Minimal public-model adapter centered on point prediction."""

    candidate_schema = {
        "version": "public_qwen_point_json_v1",
        "primary_fields": ["point_2d", "action_type"],
        "derived_fields": ["bbox_proposal_from_click_delta_12px"],
    }
    candidate_semantics = "point_2d_action_type"

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
        coordinate_frame: str = "original",
    ) -> None:
        if coordinate_frame not in {"original", "model_resized"}:
            raise ValueError("coordinate_frame must be one of: original, model_resized")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.coordinate_frame = coordinate_frame
        self.backbone = VLMBackbone(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            load_model=True,
            attn_implementation=attn_implementation,
            gpu_memory_utilization=gpu_memory_utilization,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            log_generate_diagnostics=False,
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

    def _build_prompt(self, sample: GroundingSample, image_size: tuple[int, int]) -> str:
        width, height = image_size
        coordinate_instruction = (
            f"Screenshot size: width={width}, height={height}.\n"
            "The point_2d coordinates must be absolute pixel coordinates in the screenshot.\n"
        )
        if self.coordinate_frame == "model_resized":
            model_width, model_height = self._get_model_coordinate_size(image_size)
            coordinate_instruction = (
                f"Original screenshot size: width={width}, height={height}.\n"
                f"Model-view resized screenshot size: width={model_width}, height={model_height}.\n"
                "The point_2d coordinates must be absolute pixel coordinates in the model-view resized screenshot.\n"
                "These coordinates will be mapped back to the original screenshot after parsing.\n"
            )
        return (
            "You are a GUI grounding model.\n"
            "Given one screenshot and one instruction, identify the target UI element.\n\n"
            "Return ONLY valid JSON in Qwen grounding style with this exact schema:\n"
            "{\n"
            '  "point_2d": [x, y],\n'
            '  "label": "target",\n'
            '  "action_type": "click",\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            f"{coordinate_instruction}"
            "- do not output explanation\n\n"
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

    def _normalize_action_type(self, value: Any) -> str | None:
        if value is None:
            return None
        action = str(value).strip().lower()
        if action in _ACTION_TYPES:
            return action
        return None

    def _resolve_point(
        self,
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
            return ((_clamp(cx * width, 0.0, float(width)), _clamp(cy * height, 0.0, float(height))), "normalized")

        if 0.0 <= cx <= float(width) and 0.0 <= cy <= float(height):
            return ((_clamp(cx, 0.0, float(width)), _clamp(cy, 0.0, float(height))), "absolute")

        return None, None

    def _resolve_bbox(
        self,
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
            scale_x = float(width)
            scale_y = float(height)
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
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

    def _parse_prediction(
        self,
        response_text: str,
        sample_id: str,
        image_size: tuple[int, int],
    ) -> tuple[PredictionResult, dict[str, Any]]:
        width, height = image_size
        coord_size = image_size
        if self.coordinate_frame == "model_resized":
            coord_size = self._get_model_coordinate_size(image_size)
        data = self._extract_json_dict(response_text)
        action = self._normalize_action_type(data.get("action_type"))
        _, element_id = _first_present(data, _ELEMENT_ID_KEYS)
        if element_id is not None:
            element_id = str(element_id)

        bbox_key, bbox_data = _first_present(data, _BBOX_KEYS)
        pred_bbox, bbox_mode = self._resolve_bbox(bbox_data, coord_size)

        click_key, click_data = _first_present(data, _CLICK_KEYS)
        pred_click, click_mode = self._resolve_point(click_data, coord_size)

        if pred_click is None and pred_bbox is not None:
            pred_click = pred_bbox.center
            click_mode = "derived_from_bbox"

        if pred_bbox is None and pred_click is not None:
            cx, cy = pred_click
            delta = 12.0
            coord_width, coord_height = coord_size
            pred_bbox = BBox(
                x1=_clamp(cx - delta, 0.0, float(coord_width)),
                y1=_clamp(cy - delta, 0.0, float(coord_height)),
                x2=_clamp(cx + delta, 0.0, float(coord_width)),
                y2=_clamp(cy + delta, 0.0, float(coord_height)),
            )
            bbox_mode = "derived_from_click"

        if self.coordinate_frame == "model_resized":
            coord_width, coord_height = coord_size
            scale_x = float(width) / float(coord_width)
            scale_y = float(height) / float(coord_height)
            if pred_bbox is not None:
                pred_bbox = BBox(
                    x1=_clamp(pred_bbox.x1 * scale_x, 0.0, float(width)),
                    y1=_clamp(pred_bbox.y1 * scale_y, 0.0, float(height)),
                    x2=_clamp(pred_bbox.x2 * scale_x, 0.0, float(width)),
                    y2=_clamp(pred_bbox.y2 * scale_y, 0.0, float(height)),
                )
            if pred_click is not None:
                pred_click = (
                    _clamp(pred_click[0] * scale_x, 0.0, float(width)),
                    _clamp(pred_click[1] * scale_y, 0.0, float(height)),
                )

        confidence = _safe_float(data.get("confidence"))
        if confidence is not None:
            confidence = _clamp(confidence, 0.0, 1.0)

        if data:
            data = dict(data)
            data["_resolved_click_key"] = click_key
            data["_resolved_click_mode"] = click_mode
            data["_resolved_bbox_key"] = bbox_key
            data["_resolved_bbox_mode"] = bbox_mode
            data["_resolved_coordinate_frame"] = self.coordinate_frame

        result = PredictionResult(
            sample_id=sample_id,
            predicted_action_type=action,
            predicted_bbox=pred_bbox,
            predicted_click_point=pred_click,
            predicted_element_id=element_id,
            confidence=confidence,
        )
        return result, data

    def predict_with_details(
        self,
        sample: GroundingSample,
    ) -> tuple[PredictionResult, str, dict[str, Any]]:
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Sample image does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt = self._build_prompt(sample=sample, image_size=image.size)
        outputs = self.backbone.generate(
            images=[image],
            prompts=[prompt],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
        )
        raw_text = outputs[0] if outputs else ""
        pred, parsed = self._parse_prediction(
            response_text=raw_text,
            sample_id=sample.sample_id,
            image_size=image.size,
        )
        return pred, raw_text, parsed

    def predict(
        self,
        sample: GroundingSample,
        **kwargs,
    ) -> PredictionResult:
        pred, _, _ = self.predict_with_details(sample)
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
        image = Image.open(image_path).convert("RGB")
        prompt = self._build_prompt(sample=sample, image_size=image.size)
        outputs = self.backbone.generate(
            images=[image] * num_candidates,
            prompts=[prompt] * num_candidates,
            max_new_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 0.3),
            num_return_sequences=1,
        )
        preds: list[PredictionResult] = []
        for out in outputs:
            pred, _ = self._parse_prediction(out, sample.sample_id, image.size)
            preds.append(pred)
        return preds
