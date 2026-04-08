"""Real Qwen-first Stage-A supervised fine-tuning for Mind2Web."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, get_scheduler
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from gui_grounding.data.schemas import GroundingSample
from gui_grounding.models.vlm_backbone import VLMBackbone
from gui_grounding.utils.io import save_json, save_jsonl
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _clamp(v: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, v))


def _coordinate_size_for_supervision(
    image_size: tuple[int, int],
    coordinate_frame: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[int, int]:
    if coordinate_frame == "original":
        return image_size
    if coordinate_frame != "model_resized":
        raise ValueError(f"Unsupported coordinate_frame={coordinate_frame}")

    width, height = image_size
    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=28,
        min_pixels=min_pixels or (56 * 56),
        max_pixels=max_pixels or (14 * 14 * 4 * 1280),
    )
    return resized_width, resized_height


def _project_point_to_frame(
    point: tuple[float, float],
    image_size: tuple[int, int],
    coord_size: tuple[int, int],
) -> tuple[float, float]:
    image_width, image_height = image_size
    coord_width, coord_height = coord_size
    scale_x = float(coord_width) / max(float(image_width), 1.0)
    scale_y = float(coord_height) / max(float(image_height), 1.0)
    return (float(point[0]) * scale_x, float(point[1]) * scale_y)


def _project_bbox_to_frame(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    coord_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    image_width, image_height = image_size
    coord_width, coord_height = coord_size
    scale_x = float(coord_width) / max(float(image_width), 1.0)
    scale_y = float(coord_height) / max(float(image_height), 1.0)
    return (
        float(bbox[0]) * scale_x,
        float(bbox[1]) * scale_y,
        float(bbox[2]) * scale_x,
        float(bbox[3]) * scale_y,
    )


def _serialize_point(
    point: tuple[float, float],
    coord_size: tuple[int, int],
    coordinate_format: str,
) -> list[float]:
    x, y = float(point[0]), float(point[1])
    if coordinate_format == "absolute":
        return [round(x, 4), round(y, 4)]
    if coordinate_format == "normalized":
        width, height = coord_size
        return [
            round(x / max(float(width), 1.0), 4),
            round(y / max(float(height), 1.0), 4),
        ]
    raise ValueError(f"Unsupported coordinate_format={coordinate_format}")


def _serialize_bbox(
    bbox: tuple[float, float, float, float],
    coord_size: tuple[int, int],
    coordinate_format: str,
) -> list[float]:
    if coordinate_format == "absolute":
        return [round(float(v), 4) for v in bbox]
    if coordinate_format == "normalized":
        width, height = coord_size
        return [
            round(float(bbox[0]) / max(float(width), 1.0), 4),
            round(float(bbox[1]) / max(float(height), 1.0), 4),
            round(float(bbox[2]) / max(float(width), 1.0), 4),
            round(float(bbox[3]) / max(float(height), 1.0), 4),
        ]
    raise ValueError(f"Unsupported coordinate_format={coordinate_format}")


def _quantize_coordinate(value: float, size: float, bins: int) -> int:
    if bins < 2:
        raise ValueError("coordinate_quantization_bins must be >= 2")
    normalized = _clamp(float(value) / max(float(size), 1.0), 0.0, 1.0)
    return int(round(normalized * float(bins - 1)))


def _serialize_quantized_point(
    point: tuple[float, float],
    coord_size: tuple[int, int],
    bins: int,
) -> list[int]:
    width, height = coord_size
    return [
        _quantize_coordinate(point[0], width, bins),
        _quantize_coordinate(point[1], height, bins),
    ]


def _serialize_quantized_bbox(
    bbox: tuple[float, float, float, float],
    coord_size: tuple[int, int],
    bins: int,
) -> list[int]:
    width, height = coord_size
    return [
        _quantize_coordinate(bbox[0], width, bins),
        _quantize_coordinate(bbox[1], height, bins),
        _quantize_coordinate(bbox[2], width, bins),
        _quantize_coordinate(bbox[3], height, bins),
    ]


def _resolve_serialization_targets(
    sample: GroundingSample,
    image_size: tuple[int, int],
    *,
    coordinate_frame: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[tuple[int, int], tuple[float, float, float, float], tuple[float, float], str]:
    coord_size = _coordinate_size_for_supervision(
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    target_bbox = sample.target_bbox.as_tuple() if sample.target_bbox else (0.0, 0.0, 0.0, 0.0)
    click = sample.click_point
    if click is None and sample.target_bbox is not None:
        click = sample.target_bbox.center
    if click is None:
        click = (0.0, 0.0)

    if coordinate_frame == "model_resized":
        target_bbox = _project_bbox_to_frame(target_bbox, image_size=image_size, coord_size=coord_size)
        click = _project_point_to_frame(click, image_size=image_size, coord_size=coord_size)

    action_type = str(sample.action_type) if sample.action_type is not None else "click"
    return coord_size, target_bbox, click, action_type


def _build_quantized_coordinate_instruction(
    image_size: tuple[int, int],
    *,
    coordinate_frame: str,
    bins: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[str, tuple[int, int]]:
    width, height = image_size
    coord_size = _coordinate_size_for_supervision(
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    coord_width, coord_height = coord_size
    if coordinate_frame == "model_resized":
        coordinate_instruction = (
            f"Original screenshot size: width={width}, height={height}.\n"
            f"Model-view resized screenshot size: width={coord_width}, height={coord_height}.\n"
            f"Coordinates must be integer bins in [0, {bins - 1}] relative to the model-view resized screenshot.\n"
            f"Interpret each bin value by dividing by {bins - 1} to recover the normalized coordinate.\n"
            "These coordinates will be mapped back to the original screenshot after parsing.\n"
        )
    else:
        coordinate_instruction = (
            f"Screenshot size: width={width}, height={height}.\n"
            f"Coordinates must be integer bins in [0, {bins - 1}] relative to the screenshot.\n"
            f"Interpret each bin value by dividing by {bins - 1} to recover the normalized coordinate.\n"
        )
    return coordinate_instruction, coord_size


def _build_training_prompt(
    sample: GroundingSample,
    image_size: tuple[int, int],
    *,
    coordinate_frame: str = "original",
    coordinate_format: str = "absolute",
    point_first_target: bool = False,
    coordinate_quantization_bins: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> str:
    if coordinate_quantization_bins is not None:
        coordinate_instruction, _ = _build_quantized_coordinate_instruction(
            image_size=image_size,
            coordinate_frame=coordinate_frame,
            bins=coordinate_quantization_bins,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        point_priority_instruction = ""
        schema_lines = [
            '  "action_type": "click|type|select|hover",',
            '  "bbox_bin": [x1_bin, y1_bin, x2_bin, y2_bin],',
            '  "point_bin": [x_bin, y_bin]',
        ]
        if point_first_target:
            point_priority_instruction = (
                "Primary goal: predict point_bin first.\n"
                "Place point_bin on the exact user click location inside the actionable interior of the target.\n"
                "Then make bbox_bin enclose that clicked target region.\n"
            )
            schema_lines = [
                '  "point_bin": [x_bin, y_bin],',
                '  "action_type": "click|type|select|hover",',
                '  "bbox_bin": [x1_bin, y1_bin, x2_bin, y2_bin]',
            ]
        schema_text = "{\n" + "\n".join(schema_lines) + "\n}\n\n"
        return (
            "You are a GUI grounding model.\n"
            "Given ONE screenshot and ONE instruction, predict the grounded action.\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            f"{schema_text}"
            f"{coordinate_instruction}"
            f"{point_priority_instruction}"
            "If uncertain, still provide the best grounded action.\n\n"
            f"Instruction: {sample.instruction}\n"
        )

    width, height = image_size
    coord_size = _coordinate_size_for_supervision(
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    coord_width, coord_height = coord_size
    if coordinate_format == "absolute":
        coordinate_instruction = (
            f"Screenshot size: width={width}, height={height}.\n"
            "Coordinates must be absolute pixel coordinates within the screenshot bounds.\n"
        )
        if coordinate_frame == "model_resized":
            coordinate_instruction = (
                f"Original screenshot size: width={width}, height={height}.\n"
                f"Model-view resized screenshot size: width={coord_width}, height={coord_height}.\n"
                "Output coordinates in the model-view resized screenshot coordinate system.\n"
                "These coordinates will be mapped back to the original screenshot after parsing.\n"
            )
    elif coordinate_format == "normalized":
        coordinate_instruction = (
            f"Screenshot size: width={width}, height={height}.\n"
            "Coordinates must be normalized floats in [0, 1].\n"
            "Use x values relative to screenshot width and y values relative to screenshot height.\n"
        )
        if coordinate_frame == "model_resized":
            coordinate_instruction = (
                f"Original screenshot size: width={width}, height={height}.\n"
                f"Model-view resized screenshot size: width={coord_width}, height={coord_height}.\n"
                "Coordinates must be normalized floats in [0, 1] relative to the model-view resized screenshot.\n"
                "They will be mapped back to the original screenshot after parsing.\n"
            )
    else:
        raise ValueError(f"Unsupported coordinate_format={coordinate_format}")

    point_priority_instruction = ""
    schema_lines = [
        '  "action_type": "click|type|select|hover",',
        '  "predicted_element_id": "string or null",',
        '  "predicted_bbox": [x1, y1, x2, y2],',
        '  "predicted_click_point": [x, y],',
        '  "confidence": 0.0',
    ]
    if point_first_target:
        point_priority_instruction = (
            "Primary goal: predict predicted_click_point first.\n"
            "Place the click inside the actionable interior of the target, preferably near its center.\n"
            "Then make predicted_bbox enclose that clicked target region.\n"
            "If you are unsure about predicted_element_id, use null.\n"
        )
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
        "Given ONE screenshot and ONE instruction, predict the grounded action.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        f"{schema_text}"
        f"{coordinate_instruction}"
        f"{point_priority_instruction}"
        "If uncertain, still provide the best grounded action.\n\n"
        f"Instruction: {sample.instruction}\n"
    )


def _build_target_text(
    sample: GroundingSample,
    image_size: tuple[int, int],
    *,
    coordinate_frame: str = "original",
    coordinate_format: str = "absolute",
    point_first_target: bool = False,
    coordinate_quantization_bins: int | None = None,
    supervise_element_id: bool = True,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> str:
    coord_size, target_bbox, click, action_type = _resolve_serialization_targets(
        sample=sample,
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if coordinate_quantization_bins is not None:
        serialized_click = _serialize_quantized_point(click, coord_size=coord_size, bins=coordinate_quantization_bins)
        serialized_bbox = _serialize_quantized_bbox(
            target_bbox,
            coord_size=coord_size,
            bins=coordinate_quantization_bins,
        )
    else:
        serialized_click = _serialize_point(click, coord_size=coord_size, coordinate_format=coordinate_format)
        serialized_bbox = _serialize_bbox(target_bbox, coord_size=coord_size, coordinate_format=coordinate_format)
    predicted_element_id = sample.target_element_id if supervise_element_id else None

    if coordinate_quantization_bins is not None:
        if point_first_target:
            payload = {
                "point_bin": serialized_click,
                "action_type": action_type,
                "bbox_bin": serialized_bbox,
            }
        else:
            payload = {
                "action_type": action_type,
                "bbox_bin": serialized_bbox,
                "point_bin": serialized_click,
            }
    else:
        if point_first_target:
            payload = {
                "predicted_click_point": serialized_click,
                "predicted_bbox": serialized_bbox,
                "action_type": action_type,
                "predicted_element_id": predicted_element_id,
                "confidence": 1.0,
            }
        else:
            payload = {
                "action_type": action_type,
                "predicted_element_id": predicted_element_id,
                "predicted_bbox": serialized_bbox,
                "predicted_click_point": serialized_click,
                "confidence": 1.0,
            }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _build_point_native_primary_prompt(
    sample: GroundingSample,
    image_size: tuple[int, int],
    *,
    coordinate_frame: str,
    coordinate_quantization_bins: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> str:
    coordinate_instruction, _ = _build_quantized_coordinate_instruction(
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        bins=coordinate_quantization_bins,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return (
        "You are a GUI grounding model.\n"
        "Given ONE screenshot and ONE instruction, predict the primary click target.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "point_bin": [x_bin, y_bin],\n'
        '  "action_type": "click|type|select|hover"\n'
        "}\n\n"
        f"{coordinate_instruction}"
        "Primary goal: place point_bin on the exact user click location.\n"
        "Do not output a bounding box in this pass.\n\n"
        f"Instruction: {sample.instruction}\n"
    )


def _build_point_native_primary_target_text(
    sample: GroundingSample,
    image_size: tuple[int, int],
    *,
    coordinate_frame: str,
    coordinate_quantization_bins: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> str:
    coord_size, _, click, action_type = _resolve_serialization_targets(
        sample=sample,
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    payload = {
        "point_bin": _serialize_quantized_point(click, coord_size=coord_size, bins=coordinate_quantization_bins),
        "action_type": action_type,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _build_bbox_support_prompt(
    sample: GroundingSample,
    image_size: tuple[int, int],
    *,
    coordinate_frame: str,
    coordinate_quantization_bins: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> str:
    coordinate_instruction, coord_size = _build_quantized_coordinate_instruction(
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        bins=coordinate_quantization_bins,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    _, _, click, _ = _resolve_serialization_targets(
        sample=sample,
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    click_bin = _serialize_quantized_point(click, coord_size=coord_size, bins=coordinate_quantization_bins)
    return (
        "You are a GUI grounding model.\n"
        "A primary click point has already been chosen for this screenshot and instruction.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "bbox_bin": [x1_bin, y1_bin, x2_bin, y2_bin]\n'
        "}\n\n"
        f"{coordinate_instruction}"
        f"Keep the primary click fixed at point_bin={click_bin}.\n"
        "Predict only the bbox_bin that encloses the clicked target.\n"
        "The bbox_bin must contain the provided point_bin.\n\n"
        f"Instruction: {sample.instruction}\n"
    )


def _build_bbox_support_target_text(
    sample: GroundingSample,
    image_size: tuple[int, int],
    *,
    coordinate_frame: str,
    coordinate_quantization_bins: int,
    min_pixels: int | None,
    max_pixels: int | None,
) -> str:
    coord_size, target_bbox, _, _ = _resolve_serialization_targets(
        sample=sample,
        image_size=image_size,
        coordinate_frame=coordinate_frame,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    payload = {
        "bbox_bin": _serialize_quantized_bbox(target_bbox, coord_size=coord_size, bins=coordinate_quantization_bins),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class QwenSFTDataset(Dataset):
    """Convert GroundingSample rows into prompt/target supervision."""

    def __init__(
        self,
        samples: list[GroundingSample],
        *,
        coordinate_frame: str = "original",
        coordinate_format: str = "absolute",
        point_first_target: bool = False,
        supervise_element_id: bool = True,
        supervision_mode: str = "structured",
        coordinate_quantization_bins: int | None = None,
        bbox_support_fraction: float = 0.0,
        seed: int = 42,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.samples = samples
        self.coordinate_frame = coordinate_frame
        self.coordinate_format = coordinate_format
        self.point_first_target = bool(point_first_target)
        self.supervise_element_id = bool(supervise_element_id)
        self.supervision_mode = str(supervision_mode)
        self.coordinate_quantization_bins = (
            int(coordinate_quantization_bins) if coordinate_quantization_bins is not None else None
        )
        self.bbox_support_fraction = float(bbox_support_fraction)
        self.seed = int(seed)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.examples = self._build_examples()
        self.example_stats = self._summarize_examples()

        if self.supervision_mode == "point_native_point_then_bbox" and self.coordinate_quantization_bins is None:
            raise ValueError(
                "point_native_point_then_bbox supervision requires coordinate_quantization_bins."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def _stable_order_key(self, sample_id: str, namespace: str) -> str:
        payload = f"{self.seed}:{namespace}:{sample_id}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _build_examples(self) -> list[dict[str, Any]]:
        if self.supervision_mode == "structured":
            return [
                {
                    "sample": sample,
                    "example_type": "structured",
                    "example_sample_id": sample.sample_id,
                }
                for sample in self.samples
            ]

        if self.supervision_mode != "point_native_point_then_bbox":
            raise ValueError(f"Unsupported supervision_mode={self.supervision_mode}")

        examples: list[dict[str, Any]] = []
        for sample in self.samples:
            examples.append(
                {
                    "sample": sample,
                    "example_type": "point_primary",
                    "example_sample_id": f"{sample.sample_id}::point_primary",
                }
            )

        support_count = int(round(len(self.samples) * self.bbox_support_fraction))
        if support_count > 0:
            ordered_samples = sorted(
                self.samples,
                key=lambda sample: self._stable_order_key(sample.sample_id, "bbox_support"),
            )
            for sample in ordered_samples[:support_count]:
                examples.append(
                    {
                        "sample": sample,
                        "example_type": "bbox_support",
                        "example_sample_id": f"{sample.sample_id}::bbox_support",
                    }
                )
        return examples

    def _summarize_examples(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for entry in self.examples:
            key = str(entry["example_type"])
            counts[key] = counts.get(key, 0) + 1
        return {
            "supervision_mode": self.supervision_mode,
            "num_original_samples": len(self.samples),
            "num_supervision_examples": len(self.examples),
            "example_type_counts": counts,
            "coordinate_quantization_bins": self.coordinate_quantization_bins,
            "bbox_support_fraction": self.bbox_support_fraction,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.examples[idx]
        sample = entry["sample"]
        example_type = str(entry["example_type"])
        image = Image.open(sample.image_path).convert("RGB")
        if example_type == "structured":
            prompt = _build_training_prompt(
                sample=sample,
                image_size=image.size,
                coordinate_frame=self.coordinate_frame,
                coordinate_format=self.coordinate_format,
                point_first_target=self.point_first_target,
                coordinate_quantization_bins=self.coordinate_quantization_bins,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            target_text = _build_target_text(
                sample=sample,
                image_size=image.size,
                coordinate_frame=self.coordinate_frame,
                coordinate_format=self.coordinate_format,
                point_first_target=self.point_first_target,
                coordinate_quantization_bins=self.coordinate_quantization_bins,
                supervise_element_id=self.supervise_element_id,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        elif example_type == "point_primary":
            prompt = _build_point_native_primary_prompt(
                sample=sample,
                image_size=image.size,
                coordinate_frame=self.coordinate_frame,
                coordinate_quantization_bins=int(self.coordinate_quantization_bins or 0),
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            target_text = _build_point_native_primary_target_text(
                sample=sample,
                image_size=image.size,
                coordinate_frame=self.coordinate_frame,
                coordinate_quantization_bins=int(self.coordinate_quantization_bins or 0),
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        elif example_type == "bbox_support":
            prompt = _build_bbox_support_prompt(
                sample=sample,
                image_size=image.size,
                coordinate_frame=self.coordinate_frame,
                coordinate_quantization_bins=int(self.coordinate_quantization_bins or 0),
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            target_text = _build_bbox_support_target_text(
                sample=sample,
                image_size=image.size,
                coordinate_frame=self.coordinate_frame,
                coordinate_quantization_bins=int(self.coordinate_quantization_bins or 0),
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        else:
            raise ValueError(f"Unsupported example_type={example_type}")
        return {
            "sample_id": str(entry["example_sample_id"]),
            "image": image,
            "prompt": prompt,
            "target_text": target_text,
        }


class QwenSFTCollator:
    """Build masked multimodal chat batches for supervised Qwen training."""

    def __init__(
        self,
        processor: Any,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def _encode_messages(self, messages: list[list[dict[str, Any]]]) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = int(self.max_pixels)
        return self.processor(**processor_kwargs)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        full_messages: list[list[dict[str, Any]]] = []
        prompt_messages: list[list[dict[str, Any]]] = []
        for item in batch:
            prompt_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": item["prompt"]},
                    ],
                }
            ]
            full_messages.append(
                prompt_message
                + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": item["target_text"]}],
                    }
                ]
            )
            prompt_messages.append(prompt_message)

        model_inputs = self._encode_messages(full_messages)

        from qwen_vl_utils import process_vision_info

        prompt_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in prompt_messages
        ]
        prompt_images, prompt_videos = process_vision_info(prompt_messages)
        prompt_kwargs: dict[str, Any] = {
            "text": prompt_texts,
            "images": prompt_images,
            "videos": prompt_videos,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.min_pixels is not None:
            prompt_kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels is not None:
            prompt_kwargs["max_pixels"] = int(self.max_pixels)
        prompt_inputs = self.processor(**prompt_kwargs)

        labels = model_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        for row_idx, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[row_idx, : int(prompt_len)] = -100
        model_inputs["labels"] = labels
        model_inputs["sample_ids"] = [item["sample_id"] for item in batch]
        return model_inputs


class QwenSFTTrainer:
    """LoRA-based Qwen SFT trainer with validation and checkpointing."""

    def __init__(
        self,
        model_name: str,
        train_samples: list[GroundingSample],
        eval_samples: list[GroundingSample],
        output_dir: str | Path,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 1,
        max_steps: int | None = None,
        warmup_ratio: float = 0.05,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        eval_steps: int = 50,
        save_steps: int = 200,
        torch_dtype: str = "auto",
        device: str = "auto",
        attn_implementation: str = "sdpa",
        gpu_memory_utilization: float = 0.9,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        num_workers: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | tuple[str, ...] | None = None,
        gradient_checkpointing: bool = True,
        target_coordinate_frame: str = "original",
        target_coordinate_format: str = "absolute",
        point_first_target: bool = False,
        supervise_element_id: bool = True,
        supervision_mode: str = "structured",
        coordinate_quantization_bins: int | None = None,
        bbox_support_fraction: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
        self.num_epochs = max(int(num_epochs), 1)
        self.max_steps = int(max_steps) if max_steps is not None and int(max_steps) > 0 else None
        self.warmup_ratio = float(warmup_ratio)
        self.max_grad_norm = float(max_grad_norm)
        self.logging_steps = max(int(logging_steps), 1)
        self.eval_steps = max(int(eval_steps), 1)
        self.save_steps = max(int(save_steps), 1)
        self.num_workers = int(num_workers)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.target_coordinate_frame = str(target_coordinate_frame)
        self.target_coordinate_format = str(target_coordinate_format)
        self.point_first_target = bool(point_first_target)
        self.supervise_element_id = bool(supervise_element_id)
        self.supervision_mode = str(supervision_mode)
        self.coordinate_quantization_bins = (
            int(coordinate_quantization_bins) if coordinate_quantization_bins is not None else None
        )
        self.bbox_support_fraction = float(bbox_support_fraction)
        self.seed = int(seed)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype_map = {
            "auto": torch.bfloat16 if self.device == "cuda" else torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(torch_dtype, torch.float32)

        hf_model_id = VLMBackbone.BACKBONE_REGISTRY.get(model_name, model_name)
        logger.info(
            "Loading Qwen Stage-A model: %s device=%s dtype=%s attn=%s",
            hf_model_id,
            self.device,
            self.dtype,
            attn_implementation,
        )
        self.processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": self.dtype,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(hf_model_id, **model_kwargs)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if self.gradient_checkpointing and hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        target_modules = list(lora_target_modules or DEFAULT_LORA_TARGET_MODULES)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model = self.model.to(self.device)
        self.model.train()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Qwen Stage-A trainable params=%d total_params=%d ratio=%.6f",
            trainable_params,
            total_params,
            trainable_params / max(total_params, 1),
        )

        self.train_dataset = QwenSFTDataset(
            train_samples,
            coordinate_frame=self.target_coordinate_frame,
            coordinate_format=self.target_coordinate_format,
            point_first_target=self.point_first_target,
            supervise_element_id=self.supervise_element_id,
            supervision_mode=self.supervision_mode,
            coordinate_quantization_bins=self.coordinate_quantization_bins,
            bbox_support_fraction=self.bbox_support_fraction,
            seed=self.seed,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.eval_dataset = QwenSFTDataset(
            eval_samples,
            coordinate_frame=self.target_coordinate_frame,
            coordinate_format=self.target_coordinate_format,
            point_first_target=self.point_first_target,
            supervise_element_id=self.supervise_element_id,
            supervision_mode=self.supervision_mode,
            coordinate_quantization_bins=self.coordinate_quantization_bins,
            bbox_support_fraction=self.bbox_support_fraction,
            seed=self.seed + 1009,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.collator = QwenSFTCollator(
            processor=self.processor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device == "cuda",
            collate_fn=self.collator,
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device == "cuda",
            collate_fn=self.collator,
        )

        self.total_train_steps = max(
            1,
            math.ceil(len(self.train_loader) / self.gradient_accumulation_steps) * self.num_epochs,
        )
        if self.max_steps is not None:
            self.total_train_steps = min(self.total_train_steps, self.max_steps)
        self.warmup_steps = int(self.total_train_steps * self.warmup_ratio)

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_train_steps,
        )

        self._global_step = 0
        self._best_eval_loss = float("inf")
        self._best_checkpoint_dir: Path | None = None
        self.history: list[dict[str, float]] = []

        self._save_dataset_manifest(train_samples, "train")
        self._save_dataset_manifest(eval_samples, "eval")

    def _save_dataset_manifest(self, samples: list[GroundingSample], split_name: str) -> None:
        rows = []
        for sample in samples:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "instruction": sample.instruction,
                    "image_path": sample.image_path,
                    "action_type": sample.action_type,
                    "target_element_id": sample.target_element_id,
                    "target_bbox": list(sample.target_bbox.as_tuple()) if sample.target_bbox else None,
                    "click_point": list(sample.click_point) if sample.click_point else None,
                    "website": sample.website,
                    "domain": sample.domain,
                }
            )
        save_jsonl(rows, self.output_dir / f"{split_name}_samples_manifest.jsonl")

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if key == "sample_ids":
                moved[key] = value
                continue
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _evaluate_loss(self) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for batch in self.eval_loader:
                batch = self._move_batch_to_device(batch)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch.get("image_grid_thw"),
                    labels=batch["labels"],
                )
                losses.append(float(outputs.loss.detach().cpu().item()))
        self.model.train()
        eval_loss = sum(losses) / max(len(losses), 1)
        return {"eval_loss": eval_loss}

    def _save_checkpoint(self, checkpoint_name: str, eval_loss: float) -> Path:
        ckpt_dir = self.output_dir / checkpoint_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)
        save_json(
            {
                "global_step": self._global_step,
                "eval_loss": eval_loss,
                "device": self.device,
                "dtype": str(self.dtype),
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "target_coordinate_frame": self.target_coordinate_frame,
                "target_coordinate_format": self.target_coordinate_format,
                "point_first_target": self.point_first_target,
                "supervise_element_id": self.supervise_element_id,
                "supervision_mode": self.supervision_mode,
                "coordinate_quantization_bins": self.coordinate_quantization_bins,
                "bbox_support_fraction": self.bbox_support_fraction,
            },
            ckpt_dir / "trainer_state.json",
        )
        return ckpt_dir

    def train(self) -> dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Qwen Stage-A Supervised Training")
        logger.info("=" * 60)
        logger.info(
            "train=%d eval=%d batch=%d grad_acc=%d epochs=%d max_steps=%s",
            len(self.train_dataset),
            len(self.eval_dataset),
            self.batch_size,
            self.gradient_accumulation_steps,
            self.num_epochs,
            self.max_steps,
        )

        running_loss = 0.0
        optimizer_steps = 0
        stop_training = False
        latest_checkpoint_dir: Path | None = None

        for epoch in range(1, self.num_epochs + 1):
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                batch = self._move_batch_to_device(batch)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch.get("image_grid_thw"),
                    labels=batch["labels"],
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                running_loss += float(outputs.loss.detach().cpu().item())

                should_step = (
                    batch_idx % self.gradient_accumulation_steps == 0
                    or batch_idx == len(self.train_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    self._global_step += 1

                    if self._global_step % self.logging_steps == 0:
                        avg_loss = running_loss / self.logging_steps
                        self.history.append(
                            {
                                "epoch": float(epoch),
                                "step": float(self._global_step),
                                "train_loss": avg_loss,
                                "lr": float(self.scheduler.get_last_lr()[0]),
                            }
                        )
                        logger.info(
                            "epoch=%d step=%d train_loss=%.6f lr=%.8f",
                            epoch,
                            self._global_step,
                            avg_loss,
                            self.scheduler.get_last_lr()[0],
                        )
                        running_loss = 0.0

                    if self._global_step % self.eval_steps == 0:
                        metrics = self._evaluate_loss()
                        metrics.update({"epoch": float(epoch), "step": float(self._global_step)})
                        self.history.append(metrics)
                        logger.info(
                            "epoch=%d step=%d eval_loss=%.6f",
                            epoch,
                            self._global_step,
                            metrics["eval_loss"],
                        )
                        if metrics["eval_loss"] < self._best_eval_loss:
                            self._best_eval_loss = metrics["eval_loss"]
                            self._best_checkpoint_dir = self._save_checkpoint("checkpoint-best", metrics["eval_loss"])
                            logger.info("Saved best checkpoint: %s", self._best_checkpoint_dir)

                    if self._global_step % self.save_steps == 0:
                        latest_checkpoint_dir = self._save_checkpoint("checkpoint-latest", float("nan"))

                    if self.max_steps is not None and optimizer_steps >= self.max_steps:
                        stop_training = True
                        break

            epoch_metrics = self._evaluate_loss()
            epoch_metrics.update({"epoch": float(epoch), "step": float(self._global_step)})
            self.history.append(epoch_metrics)
            logger.info(
                "epoch=%d completed eval_loss=%.6f",
                epoch,
                epoch_metrics["eval_loss"],
            )
            if epoch_metrics["eval_loss"] < self._best_eval_loss:
                self._best_eval_loss = epoch_metrics["eval_loss"]
                self._best_checkpoint_dir = self._save_checkpoint("checkpoint-best", epoch_metrics["eval_loss"])
                logger.info("Saved best checkpoint: %s", self._best_checkpoint_dir)
            latest_checkpoint_dir = self._save_checkpoint("checkpoint-latest", epoch_metrics["eval_loss"])

            if stop_training:
                break

        history_path = self.output_dir / "training_history.json"
        save_json(self.history, history_path)

        summary = {
            "status": "ok",
            "model_name": self.model.peft_config["default"].base_model_name_or_path,  # type: ignore[index]
            "num_train_samples": len(self.train_dataset),
            "num_eval_samples": len(self.eval_dataset),
            "num_epochs": self.num_epochs,
            "optimizer_steps": optimizer_steps,
            "global_steps": self._global_step,
            "planned_total_train_steps": self.total_train_steps,
            "warmup_steps": self.warmup_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "target_coordinate_frame": self.target_coordinate_frame,
            "target_coordinate_format": self.target_coordinate_format,
            "point_first_target": self.point_first_target,
            "supervise_element_id": self.supervise_element_id,
            "supervision_mode": self.supervision_mode,
            "coordinate_quantization_bins": self.coordinate_quantization_bins,
            "bbox_support_fraction": self.bbox_support_fraction,
            "train_supervision_stats": self.train_dataset.example_stats,
            "eval_supervision_stats": self.eval_dataset.example_stats,
            "best_eval_loss": self._best_eval_loss,
            "best_checkpoint_dir": str(self._best_checkpoint_dir) if self._best_checkpoint_dir else None,
            "latest_checkpoint_dir": str(latest_checkpoint_dir) if latest_checkpoint_dir else None,
            "last_checkpoint_dir": str(latest_checkpoint_dir) if latest_checkpoint_dir else None,
            "history_path": str(history_path),
            "train_manifest_path": str(self.output_dir / "train_samples_manifest.jsonl"),
            "eval_manifest_path": str(self.output_dir / "eval_samples_manifest.jsonl"),
        }
        save_json(summary, self.output_dir / "train_summary.json")
        logger.info("Saved Stage-A summary: %s", self.output_dir / "train_summary.json")
        return summary
