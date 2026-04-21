#!/usr/bin/env python3
"""Stage B: Candidate generation with Qwen-primary / CLIP legacy backends.

Usage:
    python scripts/run_generate_candidates.py --config configs/train/generate_candidates_clip_grid.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.data.mind2web_dataset import Mind2WebDataset
from gui_grounding.data.schemas import BBox, PredictionResult
from gui_grounding.models.qwen2_vl_grounding import QwenVLGroundingModel
from gui_grounding.models.sft_clip_grid_model import SFTCLIPGridModel
from gui_grounding.reward.verifiable_reward import VerifiableRewardCalculator, bbox_iou
from gui_grounding.utils.config import load_config
from gui_grounding.utils.io import save_jsonl
from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed

logger = get_logger("run_generate_candidates")
ACTION_TYPES = ("click", "type", "select", "hover")
PRIMARY_CANDIDATE_SCHEMA = {
    "version": "bbox_click_action_v1",
    "primary_fields": ["bbox_proposal", "click_point", "action_type"],
    "secondary_fields": ["element_hint_id", "score", "confidence", "reward"],
    "legacy_fields": ["legacy_metadata.grid_id"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model-based candidate generation for reranking")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("GUI_GROUNDING_HF_FALLBACK_ENDPOINT", "https://hf-mirror.com")
    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        logger.warning("HF token not found in process env; relying on HF local cache/session.")


def _grid_to_bbox(
    grid_id: int,
    image_size: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
) -> tuple[float, float, float, float]:
    width, height = image_size
    row = grid_id // grid_cols
    col = grid_id % grid_cols
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    return (
        float(col * cell_w),
        float(row * cell_h),
        float((col + 1) * cell_w),
        float((row + 1) * cell_h),
    )


def _resolve_checkpoint_model_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        model_file = path / "model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Checkpoint directory missing model.pt: {path}")
        return model_file
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint model file not found: {path}")
    return path


def _make_candidate_hypotheses(
    action_logits: torch.Tensor,
    grid_logits: torch.Tensor,
    top_k: int,
    top_action_k: int,
    top_grid_k: int,
    selection_strategy: str = "score_only",
    max_per_action: int = 9999,
) -> list[dict]:
    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    grid_log_probs = torch.log_softmax(grid_logits, dim=-1)
    a_k = min(top_action_k, int(action_log_probs.shape[0]))
    g_k = min(top_grid_k, int(grid_log_probs.shape[0]))
    top_actions = torch.topk(action_log_probs, k=a_k)
    top_grids = torch.topk(grid_log_probs, k=g_k)

    combos: list[dict] = []
    for a_lp, a_idx in zip(top_actions.values.tolist(), top_actions.indices.tolist()):
        for g_lp, g_idx in zip(top_grids.values.tolist(), top_grids.indices.tolist()):
            combos.append(
                {
                    "action_id": int(a_idx),
                    "grid_id": int(g_idx),
                    "action_log_prob": float(a_lp),
                    "grid_log_prob": float(g_lp),
                    "joint_log_prob": float(a_lp + g_lp),
                }
            )
    combos.sort(key=lambda x: x["joint_log_prob"], reverse=True)
    if not combos:
        return []

    selected: list[dict] = []
    if selection_strategy == "score_only":
        selected = combos[:top_k]
    elif selection_strategy == "diverse":
        action_count = {a: 0 for a in range(4)}
        used_pairs: set[tuple[int, int]] = set()
        used_grids: set[int] = set()
        # Pass 1: ensure action coverage when possible.
        for action_id in range(4):
            picked = None
            for c in combos:
                pair = (c["action_id"], c["grid_id"])
                if pair in used_pairs:
                    continue
                if c["action_id"] != action_id:
                    continue
                picked = c
                break
            if picked is not None:
                selected.append(picked)
                used_pairs.add((picked["action_id"], picked["grid_id"]))
                used_grids.add(picked["grid_id"])
                action_count[picked["action_id"]] += 1
                if len(selected) >= top_k:
                    break

        # Pass 2: prefer unseen grids and respect per-action cap.
        for c in combos:
            if len(selected) >= top_k:
                break
            pair = (c["action_id"], c["grid_id"])
            if pair in used_pairs:
                continue
            if action_count[c["action_id"]] >= max_per_action:
                continue
            if c["grid_id"] in used_grids:
                continue
            selected.append(c)
            used_pairs.add(pair)
            used_grids.add(c["grid_id"])
            action_count[c["action_id"]] += 1

        # Pass 3: fill remaining purely by score (still unique pairs).
        for c in combos:
            if len(selected) >= top_k:
                break
            pair = (c["action_id"], c["grid_id"])
            if pair in used_pairs:
                continue
            selected.append(c)
            used_pairs.add(pair)
    else:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

    denom = math.log(sum(math.exp(c["joint_log_prob"]) for c in selected))
    for rank, c in enumerate(selected, start=1):
        c["rank"] = rank
        c["score"] = math.exp(c["joint_log_prob"])
        c["confidence"] = math.exp(c["joint_log_prob"] - denom)
    return selected


def _exit_cleanly(exit_code: int = 0) -> NoReturn:
    """Legacy workaround for Python 3.13 finalization crashes."""
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cuda_memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device_idx = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    return {
        "cuda_available": True,
        "device_index": int(device_idx),
        "device_name": torch.cuda.get_device_name(device_idx),
        "allocated_bytes": int(torch.cuda.memory_allocated(device_idx)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device_idx)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device_idx)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device_idx)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _update_peak_cuda_stats(peak: dict, current: dict) -> None:
    if not current.get("cuda_available", False):
        return
    peak["peak_allocated_bytes"] = max(int(peak.get("peak_allocated_bytes", 0)), int(current.get("allocated_bytes", 0)))
    peak["peak_reserved_bytes"] = max(int(peak.get("peak_reserved_bytes", 0)), int(current.get("reserved_bytes", 0)))
    peak["peak_max_allocated_bytes"] = max(
        int(peak.get("peak_max_allocated_bytes", 0)),
        int(current.get("max_allocated_bytes", 0)),
    )
    peak["peak_max_reserved_bytes"] = max(
        int(peak.get("peak_max_reserved_bytes", 0)),
        int(current.get("max_reserved_bytes", 0)),
    )


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        try:
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        except Exception:
            return str(value)
    return str(value)


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _candidate_signature(
    *,
    action_type: str | None,
    bbox: list[float] | None,
    click: list[float] | None,
    element_hint_id: str | None,
) -> tuple:
    def _round_list(values: list[float] | None) -> tuple | None:
        if values is None:
            return None
        return tuple(round(float(v), 2) for v in values)

    return (
        (action_type or "").strip().lower(),
        _round_list(bbox),
        _round_list(click),
        element_hint_id,
    )


def _normalize_source_name(source: str | None) -> str:
    source = str(source or "")
    for prefix in (
        "structured_sampled_t0p6",
        "point_first_sampled_t0p7",
        "point_first_structured",
        "point_native_primary",
        "hybrid_point_structured",
        "stagea_first_choice",
    ):
        if source.startswith(prefix):
            return prefix
    return source or "unknown"


def _bbox_iou_lists(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _click_distance_lists(click_a: list[float] | None, click_b: list[float] | None) -> float:
    if not click_a or not click_b or len(click_a) != 2 or len(click_b) != 2:
        return 0.0
    dx = float(click_a[0]) - float(click_b[0])
    dy = float(click_a[1]) - float(click_b[1])
    return math.sqrt(dx * dx + dy * dy)


def _tokenize_text(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _point_inside_bbox(click: tuple[float, float] | None, bbox: tuple[float, float, float, float] | None) -> bool:
    if click is None or bbox is None:
        return False
    cx, cy = click
    x1, y1, x2, y2 = bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _build_dom_match_metadata(
    sample,
    *,
    bbox: tuple[float, float, float, float] | None,
    click: tuple[float, float] | None,
    element_hint_id: str | None,
) -> dict:
    dom_candidates = sample.dom_candidates or []
    if not dom_candidates:
        return {
            "available": False,
            "best_iou": 0.0,
            "click_inside_best_match": False,
            "instruction_text_overlap": 0.0,
            "exact_element_id_match": False,
        }

    instruction_tokens = _tokenize_text(sample.instruction)
    best_match = None
    best_score = -1e9
    for elem in dom_candidates:
        elem_bbox = elem.bbox.as_tuple() if elem.bbox is not None else None
        exact_id = bool(element_hint_id and elem.element_id == element_hint_id)
        match_iou = 0.0
        if bbox is not None and elem_bbox is not None:
            match_iou = bbox_iou(bbox, elem_bbox)
        click_inside = _point_inside_bbox(click, elem_bbox)
        score = match_iou + (0.25 if click_inside else 0.0) + (1.0 if exact_id else 0.0)
        if score > best_score:
            best_score = score
            best_match = (elem, elem_bbox, exact_id, match_iou, click_inside)

    if best_match is None:
        return {
            "available": True,
            "best_iou": 0.0,
            "click_inside_best_match": False,
            "instruction_text_overlap": 0.0,
            "exact_element_id_match": False,
        }

    elem, elem_bbox, exact_id, match_iou, click_inside = best_match
    text_tokens = _tokenize_text(elem.text)
    text_overlap = len(instruction_tokens & text_tokens) / max(len(text_tokens), 1)
    area = None
    if elem_bbox is not None:
        area = max(0.0, elem_bbox[2] - elem_bbox[0]) * max(0.0, elem_bbox[3] - elem_bbox[1])
    return {
        "available": True,
        "element_id": elem.element_id,
        "tag": elem.tag,
        "text": elem.text,
        "bbox": [float(v) for v in elem_bbox] if elem_bbox is not None else None,
        "best_iou": float(match_iou),
        "click_inside_best_match": bool(click_inside),
        "instruction_text_overlap": float(text_overlap),
        "exact_element_id_match": bool(exact_id),
        "area": float(area) if area is not None else None,
    }


def _build_parser_metadata(parsed_payload: dict | None) -> dict:
    parsed_payload = parsed_payload or {}
    primary_pass = parsed_payload.get("_primary_point_pass") or {}
    secondary_pass = parsed_payload.get("_secondary_structure_pass") or {}
    return {
        "resolved_coordinate_frame": parsed_payload.get("_resolved_coordinate_frame"),
        "resolved_action_source": parsed_payload.get("_resolved_action_source"),
        "resolved_click_mode": parsed_payload.get("_resolved_click_mode"),
        "resolved_bbox_mode": parsed_payload.get("_resolved_bbox_mode"),
        "resolved_click_provenance": parsed_payload.get("_resolved_click_provenance"),
        "resolved_bbox_provenance": parsed_payload.get("_resolved_bbox_provenance"),
        "resolved_action_provenance": parsed_payload.get("_resolved_action_provenance"),
        "point_pass_confidence": primary_pass.get("confidence"),
        "structure_pass_confidence": secondary_pass.get("confidence"),
    }


def _prediction_bbox_list(pred: PredictionResult | None) -> list[float] | None:
    if pred is None or pred.predicted_bbox is None:
        return None
    return [float(v) for v in pred.predicted_bbox.as_tuple()]


def _prediction_click_list(pred: PredictionResult | None) -> list[float] | None:
    if pred is None or pred.predicted_click_point is None:
        return None
    return [float(pred.predicted_click_point[0]), float(pred.predicted_click_point[1])]


def _expand_bbox_to_include_click(
    *,
    bbox: tuple[float, float, float, float] | None,
    click: tuple[float, float] | None,
    image_size: tuple[int, int],
    delta: float = 12.0,
) -> tuple[tuple[float, float, float, float] | None, dict]:
    width, height = float(image_size[0]), float(image_size[1])
    if bbox is None and click is None:
        return None, {"applied": False, "reason": "missing_bbox_and_click"}
    if bbox is None and click is not None:
        cx, cy = click
        derived_bbox = (
            max(0.0, cx - delta),
            max(0.0, cy - delta),
            min(width, cx + delta),
            min(height, cy + delta),
        )
        return derived_bbox, {
            "applied": True,
            "reason": "derived_bbox_from_click",
            "delta": delta,
        }
    if click is None:
        return bbox, {"applied": False, "reason": "missing_click"}
    if _point_inside_bbox(click, bbox):
        return bbox, {"applied": False, "reason": "bbox_already_contains_click"}
    cx, cy = click
    x1, y1, x2, y2 = bbox
    expanded_bbox = (
        max(0.0, min(x1, cx)),
        max(0.0, min(y1, cy)),
        min(width, max(x2, cx)),
        min(height, max(y2, cy)),
    )
    return expanded_bbox, {
        "applied": True,
        "reason": "expanded_bbox_to_include_click",
        "original_bbox": [float(v) for v in bbox],
        "primary_click_point": [float(cx), float(cy)],
        "reconciled_bbox": [float(v) for v in expanded_bbox],
    }


def _build_qwen_generation_detail(
    *,
    pred: PredictionResult,
    raw_text: str,
    parsed_payload: dict | None,
    generation_mode: str,
    generation_temperature: float | None,
    attempt_index: int,
    point_first_prompt: bool = False,
    web_mobile_hotspot_prompt: bool = False,
    decoupled_point_native_decode: bool = False,
    source_family: str = "structured",
    extra_provenance: dict | None = None,
) -> dict:
    parsed_payload = dict(parsed_payload or {})
    parsed_payload["_point_first_prompt"] = bool(point_first_prompt)
    parsed_payload["_web_mobile_hotspot_prompt"] = bool(web_mobile_hotspot_prompt)
    parsed_payload["_decoupled_point_native_decode"] = bool(decoupled_point_native_decode)
    return {
        "pred": pred,
        "raw_text": raw_text,
        "parsed_payload": parsed_payload,
        "generation_mode": generation_mode,
        "generation_temperature": generation_temperature,
        "attempt_index": attempt_index,
        "point_first_prompt": bool(point_first_prompt),
        "web_mobile_hotspot_prompt": bool(web_mobile_hotspot_prompt),
        "decoupled_point_native_decode": bool(decoupled_point_native_decode),
        "source_family": source_family,
        "extra_provenance": extra_provenance or {},
    }


def _build_hybrid_qwen_detail(
    *,
    sample,
    image_size: tuple[int, int],
    generation_mode: str,
    click_detail: dict,
    support_detail: dict,
    attempt_index: int,
) -> dict | None:
    click_pred = click_detail.get("pred")
    support_pred = support_detail.get("pred")
    if not isinstance(click_pred, PredictionResult) or not isinstance(support_pred, PredictionResult):
        return None

    click_point = click_pred.predicted_click_point or support_pred.predicted_click_point
    bbox = support_pred.predicted_bbox.as_tuple() if support_pred.predicted_bbox is not None else None
    reconciled_bbox, bbox_reconciliation = _expand_bbox_to_include_click(
        bbox=bbox,
        click=click_point,
        image_size=image_size,
    )
    if click_point is None and reconciled_bbox is None:
        return None

    pred_bbox = None
    if reconciled_bbox is not None:
        pred_bbox = BBox(x1=reconciled_bbox[0], y1=reconciled_bbox[1], x2=reconciled_bbox[2], y2=reconciled_bbox[3])
    action_type = (
        support_pred.predicted_action_type
        or click_pred.predicted_action_type
        or "click"
    )
    element_hint_id = support_pred.predicted_element_id or click_pred.predicted_element_id
    confidences = [c for c in (click_pred.confidence, support_pred.confidence) if c is not None]
    confidence = min(confidences) if confidences else None

    click_parsed = dict(click_detail.get("parsed_payload") or {})
    support_parsed = dict(support_detail.get("parsed_payload") or {})
    parsed_payload = {
        "_hybrid_candidate": True,
        "_point_first_prompt": bool(click_detail.get("point_first_prompt") or support_detail.get("point_first_prompt")),
        "_web_mobile_hotspot_prompt": bool(
            click_detail.get("web_mobile_hotspot_prompt") or support_detail.get("web_mobile_hotspot_prompt")
        ),
        "_decoupled_point_native_decode": bool(
            click_detail.get("decoupled_point_native_decode") or support_detail.get("decoupled_point_native_decode")
        ),
        "_resolved_coordinate_frame": (
            support_parsed.get("_resolved_coordinate_frame")
            or click_parsed.get("_resolved_coordinate_frame")
        ),
        "_resolved_click_provenance": (
            click_parsed.get("_resolved_click_provenance")
            or click_detail.get("generation_mode")
        ),
        "_resolved_bbox_provenance": (
            support_parsed.get("_resolved_bbox_provenance")
            or support_detail.get("generation_mode")
        ),
        "_resolved_action_provenance": (
            support_parsed.get("_resolved_action_provenance")
            or support_detail.get("generation_mode")
        ),
        "_primary_point_pass": click_parsed.get("_primary_point_pass")
        or {
            "click_point": list(click_point) if click_point is not None else None,
            "confidence": click_pred.confidence,
            "source_generation_mode": click_detail.get("generation_mode"),
        },
        "_secondary_structure_pass": support_parsed.get("_secondary_structure_pass")
        or {
            "bbox": list(reconciled_bbox) if reconciled_bbox is not None else None,
            "action_type": support_pred.predicted_action_type,
            "element_id": support_pred.predicted_element_id,
            "confidence": support_pred.confidence,
            "source_generation_mode": support_detail.get("generation_mode"),
        },
        "_bbox_click_reconciliation_after_map": bbox_reconciliation,
        "_hybrid_sources": {
            "click_source": click_detail.get("generation_mode"),
            "support_source": support_detail.get("generation_mode"),
            "click_candidate_id": click_detail.get("pred").sample_id if click_detail.get("pred") else None,
            "support_candidate_id": support_detail.get("pred").sample_id if support_detail.get("pred") else None,
        },
        "action_type": action_type,
        "predicted_element_id": element_hint_id,
        "predicted_bbox": list(reconciled_bbox) if reconciled_bbox is not None else None,
        "predicted_click_point": list(click_point) if click_point is not None else None,
        "confidence": confidence,
    }
    raw_text = (
        f"[hybrid_click_source:{click_detail.get('generation_mode')}]\n"
        f"{click_detail.get('raw_text', '')}\n\n"
        f"[hybrid_support_source:{support_detail.get('generation_mode')}]\n"
        f"{support_detail.get('raw_text', '')}"
    )
    pred = PredictionResult(
        sample_id=sample.sample_id,
        predicted_action_type=action_type,
        predicted_bbox=pred_bbox,
        predicted_click_point=click_point,
        predicted_element_id=element_hint_id,
        confidence=confidence,
    )
    return _build_qwen_generation_detail(
        pred=pred,
        raw_text=raw_text,
        parsed_payload=parsed_payload,
        generation_mode=generation_mode,
        generation_temperature=None,
        attempt_index=attempt_index,
        point_first_prompt=bool(parsed_payload["_point_first_prompt"]),
        web_mobile_hotspot_prompt=bool(parsed_payload["_web_mobile_hotspot_prompt"]),
        decoupled_point_native_decode=bool(parsed_payload["_decoupled_point_native_decode"]),
        source_family="hybrid",
        extra_provenance={
            "click_source": click_detail.get("generation_mode"),
            "support_source": support_detail.get("generation_mode"),
            "bbox_reconciliation": bbox_reconciliation,
        },
    )


def _normalize_action_type(value) -> str | None:
    if value is None:
        return None
    action = str(value).strip().lower()
    if action in ACTION_TYPES:
        return action
    return None


def _extract_numeric_list(value, expected_len: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected_len:
        return None
    numbers: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        numbers.append(number)
    return numbers


def _is_valid_bbox_list(value) -> bool:
    bbox = _extract_numeric_list(value, expected_len=4)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    return x2 >= x1 and y2 >= y1


def _is_valid_click_list(value) -> bool:
    return _extract_numeric_list(value, expected_len=2) is not None


def _count_bbox_corrections(raw_bbox: list[float] | None, final_bbox: list[float] | None) -> dict[str, bool]:
    if raw_bbox is None or final_bbox is None:
        return {
            "bbox_clipped": False,
            "bbox_axis_reordered": False,
        }
    raw_x1, raw_y1, raw_x2, raw_y2 = raw_bbox
    final_x1, final_y1, final_x2, final_y2 = final_bbox
    bbox_clipped = any(
        abs(raw - final) > 1e-6
        for raw, final in zip(raw_bbox, final_bbox)
    )
    bbox_axis_reordered = raw_x2 < raw_x1 or raw_y2 < raw_y1
    return {
        "bbox_clipped": bbox_clipped,
        "bbox_axis_reordered": bbox_axis_reordered,
    }


def _click_was_clipped(raw_click: list[float] | None, final_click: list[float] | None) -> bool:
    if raw_click is None or final_click is None:
        return False
    return any(abs(raw - final) > 1e-6 for raw, final in zip(raw_click, final_click))


def _build_qwen_structured_output_diagnostics(
    raw_text: str,
    parsed_payload: dict,
    pred,
    final_bbox: list[float] | None,
    final_click: list[float] | None,
) -> dict:
    bbox_field_used = None
    raw_bbox = None
    for key in ("bbox_proposal", "predicted_bbox"):
        if key in parsed_payload:
            bbox_field_used = key
            raw_bbox = _extract_numeric_list(parsed_payload.get(key), expected_len=4)
            break

    click_field_used = None
    raw_click = None
    for key in ("click_point", "predicted_click_point"):
        if key in parsed_payload:
            click_field_used = key
            raw_click = _extract_numeric_list(parsed_payload.get(key), expected_len=2)
            break

    element_field_used = None
    for key in ("element_hint_id", "predicted_element_id"):
        if key in parsed_payload:
            element_field_used = key
            break

    raw_action = parsed_payload.get("action_type")
    valid_raw_action = _normalize_action_type(raw_action)
    final_action = _normalize_action_type(pred.predicted_action_type)
    action_source = None
    if valid_raw_action is not None:
        action_source = "json_field"
    elif final_action is not None:
        action_source = "text_fallback"

    bbox_corrections = _count_bbox_corrections(raw_bbox=raw_bbox, final_bbox=final_bbox)
    click_clipped = _click_was_clipped(raw_click=raw_click, final_click=final_click)
    bbox_from_click_fallback = raw_bbox is None and final_bbox is not None and raw_click is not None
    click_from_bbox_fallback = raw_click is None and final_click is not None and raw_bbox is not None

    failure_tags: list[str] = []
    if not raw_text.strip():
        failure_tags.append("empty_raw_response")
    if not parsed_payload:
        failure_tags.append("unparseable_json")
    if raw_action is not None and valid_raw_action is None:
        failure_tags.append("invalid_action_type")
    if final_action is None:
        failure_tags.append("missing_action_type")
    if bbox_field_used is not None and raw_bbox is None:
        failure_tags.append("malformed_bbox")
    if final_bbox is None:
        failure_tags.append("missing_bbox")
    if click_field_used is not None and raw_click is None:
        failure_tags.append("malformed_click_point")
    if final_click is None:
        failure_tags.append("missing_click_point")
    if bbox_corrections["bbox_clipped"]:
        failure_tags.append("bbox_clipped")
    if bbox_corrections["bbox_axis_reordered"]:
        failure_tags.append("bbox_axis_reordered")
    if click_clipped:
        failure_tags.append("click_point_clipped")
    if bbox_from_click_fallback:
        failure_tags.append("bbox_from_click_fallback")
    if click_from_bbox_fallback:
        failure_tags.append("click_from_bbox_fallback")

    confidence_present = parsed_payload.get("confidence") is not None
    return {
        "details_available": True,
        "raw_response_nonempty": bool(raw_text.strip()),
        "json_parse_success": bool(parsed_payload),
        "parsed_top_level_keys": sorted(str(k) for k in parsed_payload.keys()),
        "action_type_valid": final_action is not None,
        "action_source": action_source,
        "bbox_valid": final_bbox is not None and _is_valid_bbox_list(final_bbox),
        "bbox_field_used": bbox_field_used,
        "click_point_valid": final_click is not None and _is_valid_click_list(final_click),
        "click_field_used": click_field_used,
        "element_field_used": element_field_used,
        "bbox_from_click_fallback": bbox_from_click_fallback,
        "click_from_bbox_fallback": click_from_bbox_fallback,
        "bbox_clipped": bbox_corrections["bbox_clipped"],
        "bbox_axis_reordered": bbox_corrections["bbox_axis_reordered"],
        "click_point_clipped": click_clipped,
        "confidence_source": "model_json" if confidence_present else "uniform_fallback",
        "failure_tags": failure_tags,
    }


def _default_candidate_diagnostics() -> dict:
    return {
        "details_available": False,
        "raw_response_nonempty": None,
        "json_parse_success": None,
        "parsed_top_level_keys": [],
        "action_type_valid": None,
        "action_source": None,
        "bbox_valid": None,
        "bbox_field_used": None,
        "click_point_valid": None,
        "click_field_used": None,
        "element_field_used": None,
        "bbox_from_click_fallback": False,
        "click_from_bbox_fallback": False,
        "bbox_clipped": False,
        "bbox_axis_reordered": False,
        "click_point_clipped": False,
        "confidence_source": None,
        "failure_tags": ["structured_output_details_unavailable"],
    }


def _build_quality_summary(
    *,
    config_path: str,
    backend: str,
    model_name: str,
    split: str,
    sample_rows: list[dict],
    flat_rows: list[dict],
    failure_rows: list[dict],
    summary: dict,
    runtime_details: dict,
) -> dict:
    action_hist: Counter[str] = Counter()
    source_hist: Counter[str] = Counter()
    failure_hist: Counter[str] = Counter()
    bbox_field_hist: Counter[str] = Counter()
    click_field_hist: Counter[str] = Counter()
    confidence_source_hist: Counter[str] = Counter()

    parseable_outputs = 0
    valid_bbox_count = 0
    valid_click_count = 0
    valid_action_count = 0
    malformed_json_count = 0
    malformed_coordinate_count = 0
    bbox_clipping_corrections = 0
    click_clipping_corrections = 0
    bbox_axis_reorder_corrections = 0

    for row in flat_rows:
        action = _normalize_action_type(row.get("action_type"))
        source_hist[str(row.get("source", "__missing_source__"))] += 1
        if action is not None:
            valid_action_count += 1
            action_hist[action] += 1
        else:
            action_hist["__invalid_or_missing__"] += 1

        if _is_valid_bbox_list(row.get("bbox_proposal")):
            valid_bbox_count += 1
        if _is_valid_click_list(row.get("click_point")):
            valid_click_count += 1

        diag = row.get("structured_output_diagnostics") or {}
        if diag.get("json_parse_success") is True:
            parseable_outputs += 1
        if diag.get("bbox_clipped"):
            bbox_clipping_corrections += 1
        if diag.get("click_point_clipped"):
            click_clipping_corrections += 1
        if diag.get("bbox_axis_reordered"):
            bbox_axis_reorder_corrections += 1
        if diag.get("bbox_field_used"):
            bbox_field_hist[str(diag["bbox_field_used"])] += 1
        if diag.get("click_field_used"):
            click_field_hist[str(diag["click_field_used"])] += 1
        if diag.get("confidence_source"):
            confidence_source_hist[str(diag["confidence_source"])] += 1

        tags = diag.get("failure_tags") or []
        for tag in tags:
            failure_hist[str(tag)] += 1
        if "unparseable_json" in tags:
            malformed_json_count += 1
        if "malformed_bbox" in tags or "malformed_click_point" in tags:
            malformed_coordinate_count += 1

    for failure in failure_rows:
        failure_hist[f"inference_exception:{failure.get('error_type', 'unknown')}"] += 1

    empty_candidate_samples = sum(1 for row in sample_rows if not row.get("candidates"))
    successful_samples = int(summary.get("successful_samples", 0))
    total_candidates = len(flat_rows)
    attempted_samples = int(summary.get("attempted_samples", 0))

    return {
        "schema": {
            "version": PRIMARY_CANDIDATE_SCHEMA["version"],
            "primary_fields": PRIMARY_CANDIDATE_SCHEMA["primary_fields"],
            "legacy_fields": PRIMARY_CANDIDATE_SCHEMA["legacy_fields"],
        },
        "run_context": {
            "config_path": config_path,
            "backend": backend,
            "model_backbone": model_name,
            "split": split,
            "candidates_jsonl": summary.get("candidates_jsonl", ""),
            "flat_candidates_jsonl": summary.get("flat_candidates_jsonl", ""),
            "runtime_settings": runtime_details.get("summary", {}).get("runtime_settings", {}),
        },
        "sample_stats": {
            "attempted_samples": attempted_samples,
            "successful_end_to_end_runs": successful_samples,
            "failed_runs": int(summary.get("failed_samples", 0)),
            "exported_sample_rows": len(sample_rows),
            "empty_candidate_samples": empty_candidate_samples,
            "empty_candidate_rate": _safe_rate(empty_candidate_samples, attempted_samples),
        },
        "candidate_stats": {
            "total_candidates": total_candidates,
            "avg_candidates_per_sample": _safe_rate(total_candidates, successful_samples),
            "parseable_structured_outputs": parseable_outputs,
            "parseable_structured_output_rate": _safe_rate(parseable_outputs, total_candidates),
            "valid_bbox_proposal_count": valid_bbox_count,
            "valid_bbox_proposal_rate": _safe_rate(valid_bbox_count, total_candidates),
            "valid_click_point_count": valid_click_count,
            "valid_click_point_rate": _safe_rate(valid_click_count, total_candidates),
            "valid_action_type_count": valid_action_count,
            "valid_action_type_rate": _safe_rate(valid_action_count, total_candidates),
            "malformed_json_count": malformed_json_count,
            "malformed_json_rate": _safe_rate(malformed_json_count, total_candidates),
            "malformed_coordinate_count": malformed_coordinate_count,
            "malformed_coordinate_rate": _safe_rate(malformed_coordinate_count, total_candidates),
            "bbox_clipping_corrections": bbox_clipping_corrections,
            "click_point_clipping_corrections": click_clipping_corrections,
            "bbox_axis_reorder_corrections": bbox_axis_reorder_corrections,
            "bbox_field_histogram": dict(bbox_field_hist),
            "click_field_histogram": dict(click_field_hist),
            "confidence_source_histogram": dict(confidence_source_hist),
            "source_histogram": dict(source_hist),
            "action_type_distribution": dict(action_hist),
            "common_parse_failure_categories": dict(failure_hist),
        },
    }


def main() -> None:
    _bootstrap_env()
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    logger.info("Loaded config: %s", args.config)

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    if args.dry_run:
        logger.info("Dry run — config loaded successfully. Exiting.")
        return

    logger.info("=" * 60)
    logger.info("Candidate Generation Pipeline")
    logger.info("=" * 60)

    cand_cfg = cfg.get("candidate_generation", {})
    backend = cand_cfg.get("backend", "qwen_vl")
    top_k = int(cand_cfg.get("top_k", 6))
    if top_k < 1:
        raise ValueError("candidate_generation.top_k must be >= 1")
    if backend not in {"clip_grid_stagea", "qwen_vl"}:
        raise NotImplementedError(f"Unsupported backend: {backend}")
    top_action_k = int(cand_cfg.get("top_action_k", 4))
    top_grid_k = int(cand_cfg.get("top_grid_k", max(top_k, 8)))
    selection_strategy = cand_cfg.get("selection_strategy", "score_only")
    max_per_action = int(cand_cfg.get("max_per_action", 9999))
    first_choice_temperature = float(cand_cfg.get("first_choice_temperature", 0.0))
    sampled_temperature = float(cand_cfg.get("sampled_temperature", cfg.get("model", {}).get("temperature", 0.3)))
    deduplicate_candidates = bool(cand_cfg.get("deduplicate_candidates", True))
    max_generation_attempts = int(cand_cfg.get("max_generation_attempts", max(top_k * 4, 8)))
    generation_strategies = list(cand_cfg.get("generation_strategies", []) or [])
    hybrid_candidate_recipes = list(cand_cfg.get("hybrid_candidate_recipes", []) or [])
    source_gating_cfg = cand_cfg.get("source_gating", {}) or {}
    source_gating_enabled = bool(source_gating_cfg.get("enabled", False))
    drop_export_sources = {_normalize_source_name(s) for s in source_gating_cfg.get("drop_export_sources", [])}
    disagreement_required_sources = {
        _normalize_source_name(s) for s in source_gating_cfg.get("disagreement_required_sources", [])
    }
    min_disagreement_click_distance = float(source_gating_cfg.get("min_disagreement_click_distance", 96.0))
    max_disagreement_bbox_iou = float(source_gating_cfg.get("max_disagreement_bbox_iou", 0.35))
    near_duplicate_click_distance = float(source_gating_cfg.get("near_duplicate_click_distance", 48.0))
    near_duplicate_bbox_iou = float(source_gating_cfg.get("near_duplicate_bbox_iou", 0.75))
    max_per_normalized_source = {
        _normalize_source_name(k): int(v)
        for k, v in (source_gating_cfg.get("max_per_normalized_source", {}) or {}).items()
    }
    source_priority = {
        _normalize_source_name(k): float(v)
        for k, v in (source_gating_cfg.get("source_priority", {}) or {}).items()
    }
    safe_cfg = cand_cfg.get("safe_run", {})
    safe_run_enabled = bool(safe_cfg.get("enabled", False))
    if safe_run_enabled and bool(safe_cfg.get("cuda_launch_blocking", False)):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger.warning("Safe-run compatibility mode enabled: CUDA_LAUNCH_BLOCKING=1")
    log_cuda_memory_per_sample = bool(safe_cfg.get("log_cuda_memory_per_sample", False))
    clear_cuda_cache_on_error = bool(safe_cfg.get("clear_cuda_cache_on_error", True))
    incremental_save = bool(safe_cfg.get("incremental_save", True))
    force_serial_qwen = bool(safe_cfg.get("force_serial_qwen", True))
    log_qwen_generate_diagnostics = bool(safe_cfg.get("log_qwen_generate_diagnostics", False))
    logger.info(
        "Candidate backend=%s, top_k=%d, top_action_k=%d, top_grid_k=%d, selection=%s",
        backend,
        top_k,
        top_action_k,
        top_grid_k,
        selection_strategy,
    )
    logger.info(
        "Qwen candidate temps: first_choice=%.3f sampled=%.3f deduplicate=%s max_attempts=%d",
        first_choice_temperature,
        sampled_temperature,
        deduplicate_candidates,
        max_generation_attempts,
    )
    if generation_strategies:
        logger.info(
            "Qwen diversity strategies=%s hybrids=%s",
            [str(s.get("name", f"strategy_{idx}")) for idx, s in enumerate(generation_strategies)],
            [str(r.get("name", f"hybrid_{idx}")) for idx, r in enumerate(hybrid_candidate_recipes)],
        )
    if source_gating_enabled:
        logger.info(
            "Source gating enabled: drop=%s disagreement_required=%s max_per_source=%s",
            sorted(drop_export_sources),
            sorted(disagreement_required_sources),
            {k: int(v) for k, v in sorted(max_per_normalized_source.items())},
        )
    logger.info(
        "Safe-run enabled=%s force_serial_qwen=%s incremental_save=%s log_cuda_memory_per_sample=%s",
        safe_run_enabled,
        force_serial_qwen,
        incremental_save,
        log_cuda_memory_per_sample,
    )

    data_cfg = cfg.get("data", {})
    train_cfg = data_cfg.get("train", {})
    split = train_cfg.get("split", "train")
    max_samples = args.num_samples or train_cfg.get("max_samples", 20)
    ds = Mind2WebDataset(
        split=split,
        max_samples=max_samples,
        cache_screenshots=train_cfg.get("cache_screenshots", True),
    )
    samples = [s for s in ds if s.target_bbox is not None and s.action_type is not None and s.image_path]
    if not samples:
        raise RuntimeError("No valid supervised samples found for candidate export.")
    logger.info("Loaded %d valid samples from Mind2Web split=%s", len(samples), split)

    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("backbone", "qwen2_5_vl_3b")
    adapter_path = model_cfg.get("adapter_path")
    grid_rows = int(model_cfg.get("grid_rows", 4))
    grid_cols = int(model_cfg.get("grid_cols", 6))
    device = model_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_model_path = None
    processor = None
    model = None
    qwen_model = None
    logger.info("Inference device: %s", device)
    if backend == "clip_grid_stagea":
        checkpoint_model_path = _resolve_checkpoint_model_path(model_cfg.get("checkpoint"))
        logger.info("Using CLIP legacy checkpoint model: %s", checkpoint_model_path)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = SFTCLIPGridModel(
            model_name=model_name,
            num_actions=4,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        )
        state_dict = torch.load(checkpoint_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()
    else:
        qwen_model = QwenVLGroundingModel(
            model_name=model_name,
            device=device,
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 256)),
            temperature=float(model_cfg.get("temperature", 0.3)),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
            gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.9)),
            min_pixels=model_cfg.get("min_pixels"),
            max_pixels=model_cfg.get("max_pixels"),
            log_generate_diagnostics=log_qwen_generate_diagnostics,
            coordinate_frame=model_cfg.get("coordinate_frame", "original"),
            coordinate_format=model_cfg.get("coordinate_format", "absolute"),
            point_first_prompt=bool(model_cfg.get("point_first_prompt", False)),
            target_field_order=model_cfg.get("target_field_order"),
            point_primary_bbox_anchored_prompt=bool(model_cfg.get("point_primary_bbox_anchored_prompt", False)),
            use_candidate_anchors=bool(model_cfg.get("use_candidate_anchors", False)),
            max_prompt_candidates=int(model_cfg.get("max_prompt_candidates", 32)),
            candidate_grounding_from_slot=bool(model_cfg.get("candidate_grounding_from_slot", True)),
            web_mobile_hotspot_prompt=bool(model_cfg.get("web_mobile_hotspot_prompt", False)),
            decoupled_point_native_decode=bool(model_cfg.get("decoupled_point_native_decode", False)),
            coordinate_quantization_bins=model_cfg.get("coordinate_quantization_bins"),
            point_native_secondary_bbox_only=bool(model_cfg.get("point_native_secondary_bbox_only", False)),
            edge_click_interior_threshold=float(model_cfg.get("edge_click_interior_threshold", 0.0)),
            edge_click_interior_position=float(model_cfg.get("edge_click_interior_position", 0.45)),
            adapter_path=adapter_path,
        )
        logger.info("Using Qwen primary candidate backend with model=%s", model_name)
        logger.info(
            "Qwen runtime settings: dtype=%s attn_implementation=%s min_pixels=%s max_pixels=%s adapter=%s candidate_anchors=%s slot_grounding=%s coordinate_format=%s",
            qwen_model.backbone.resolved_torch_dtype,
            qwen_model.backbone.attn_implementation,
            qwen_model.backbone.min_pixels,
            qwen_model.backbone.max_pixels,
            adapter_path,
            qwen_model.use_candidate_anchors,
            qwen_model.candidate_grounding_from_slot,
            qwen_model.coordinate_format,
        )

    reward_calc = VerifiableRewardCalculator(weights=cfg.get("reward", {}).get("weights"))
    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("output_dir", "outputs/candidate_generation_clip_grid"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_jsonl = out_dir / out_cfg.get("candidates_jsonl", f"candidates_{split}.jsonl")
    flat_jsonl = out_dir / out_cfg.get("flat_candidates_jsonl", f"candidates_{split}_flat.jsonl")
    summary_json = out_dir / out_cfg.get("summary_json", f"summary_{split}.json")
    failure_json = out_dir / out_cfg.get("failure_json", f"failures_{split}.json")
    runtime_json = out_dir / out_cfg.get("runtime_json", f"runtime_{split}.json")
    structured_quality_json = out_dir / out_cfg.get("structured_quality_json", "structured_quality_summary.json")
    structured_quality_md = out_dir / out_cfg.get("structured_quality_md", "structured_quality_summary.md")
    action_histogram_json = out_dir / out_cfg.get("action_histogram_json", "action_type_histogram.json")

    sample_rows = []
    flat_rows = []
    total_candidates = 0
    reward_sum = 0.0
    best_reward_sum = 0.0
    failure_rows: list[dict] = []
    attempted_samples = 0
    successful_samples = 0
    failed_samples = 0
    last_processed_sample_id = None
    last_failed_sample_id = None
    source_gate_accept_hist: Counter[str] = Counter()
    source_gate_reject_hist: Counter[str] = Counter()
    source_gate_reject_reason_hist: Counter[str] = Counter()
    peak_cuda = {
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "peak_max_allocated_bytes": 0,
        "peak_max_reserved_bytes": 0,
    }

    sample_jsonl.write_text("", encoding="utf-8")
    flat_jsonl.write_text("", encoding="utf-8")

    for sample in samples:
        attempted_samples += 1
        last_processed_sample_id = sample.sample_id
        if log_cuda_memory_per_sample:
            pre_mem = _cuda_memory_stats()
            _update_peak_cuda_stats(peak_cuda, pre_mem)
            logger.info("sample_id=%s pre_cuda_mem=%s", sample.sample_id, pre_mem)
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as exc:
            failure_row = {
                "sample_id": sample.sample_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "cuda_memory": _cuda_memory_stats(),
                "utc_time": datetime.now(timezone.utc).isoformat(),
            }
            _update_peak_cuda_stats(peak_cuda, failure_row["cuda_memory"])
            failure_rows.append(failure_row)
            failed_samples += 1
            last_failed_sample_id = sample.sample_id
            logger.error("Skipping sample_id=%s image load failed: %s", sample.sample_id, exc)
            continue

        try:
            hypotheses: list[dict] = []
            qwen_pred_details: list[dict] = []
            if backend == "clip_grid_stagea":
                with torch.inference_mode():
                    encoded = processor(
                        text=[sample.instruction],
                        images=[image],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    outputs = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        pixel_values=encoded["pixel_values"],
                    )
                    hypotheses = _make_candidate_hypotheses(
                        action_logits=outputs["action_logits"][0].detach().cpu(),
                        grid_logits=outputs["grid_logits"][0].detach().cpu(),
                        top_k=top_k,
                        top_action_k=top_action_k,
                        top_grid_k=top_grid_k,
                        selection_strategy=selection_strategy,
                        max_per_action=max_per_action,
                    )
            else:
                qwen_pred_details = []
                duplicate_details: list[dict] = []
                seen_signatures: set[tuple] = set()
                strategy_bank: dict[str, list[dict]] = {}
                source_gate_counts: Counter[str] = Counter()
                total_attempts = 0

                def _detail_core(detail: dict) -> tuple[str, list[float] | None, list[float] | None, str | None]:
                    pred = detail["pred"]
                    return (
                        _normalize_source_name(detail.get("generation_mode")),
                        _prediction_bbox_list(pred),
                        _prediction_click_list(pred),
                        pred.predicted_action_type,
                    )

                def _stagea_anchor_detail() -> dict | None:
                    anchors = strategy_bank.get("stagea_first_choice", [])
                    return anchors[0] if anchors else None

                def _maybe_gate_qwen_detail(detail: dict) -> tuple[bool, list[str], dict]:
                    normalized_source, bbox, click, action_type = _detail_core(detail)
                    gate_meta = {
                        "normalized_source": normalized_source,
                        "source_priority": float(source_priority.get(normalized_source, 0.0)),
                    }
                    reasons: list[str] = []

                    if normalized_source in drop_export_sources:
                        reasons.append("drop_export_source")

                    max_allowed = max_per_normalized_source.get(normalized_source)
                    if max_allowed is not None and int(source_gate_counts[normalized_source]) >= int(max_allowed):
                        reasons.append("max_per_source")

                    anchor = _stagea_anchor_detail()
                    if anchor is not None and normalized_source != "stagea_first_choice":
                        _, anchor_bbox, anchor_click, anchor_action = _detail_core(anchor)
                        gate_meta["click_distance_to_first"] = _click_distance_lists(click, anchor_click)
                        gate_meta["bbox_iou_to_first"] = _bbox_iou_lists(bbox, anchor_bbox)
                        if normalized_source in disagreement_required_sources:
                            same_action = action_type is not None and anchor_action == action_type
                            low_disagreement = (
                                gate_meta["click_distance_to_first"] < min_disagreement_click_distance
                                and gate_meta["bbox_iou_to_first"] > max_disagreement_bbox_iou
                            )
                            if same_action and low_disagreement:
                                reasons.append("insufficient_disagreement_vs_first")

                    if qwen_pred_details:
                        for kept_detail in qwen_pred_details:
                            kept_source, kept_bbox, kept_click, kept_action = _detail_core(kept_detail)
                            same_action = action_type is not None and kept_action == action_type
                            if not same_action:
                                continue
                            click_dist = _click_distance_lists(click, kept_click)
                            bbox_overlap = _bbox_iou_lists(bbox, kept_bbox)
                            if click_dist <= near_duplicate_click_distance and bbox_overlap >= near_duplicate_bbox_iou:
                                gate_meta["near_duplicate_to_source"] = kept_source
                                gate_meta["near_duplicate_click_distance"] = click_dist
                                gate_meta["near_duplicate_bbox_iou"] = bbox_overlap
                                reasons.append("near_duplicate_to_kept_candidate")
                                break

                    return len(reasons) == 0, reasons, gate_meta

                def _maybe_add_qwen_detail(detail: dict) -> None:
                    pred = detail["pred"]
                    bbox = _prediction_bbox_list(pred)
                    click = _prediction_click_list(pred)
                    signature = _candidate_signature(
                        action_type=pred.predicted_action_type,
                        bbox=bbox,
                        click=click,
                        element_hint_id=pred.predicted_element_id,
                    )
                    if deduplicate_candidates and signature in seen_signatures:
                        duplicate_details.append(detail)
                        return
                    normalized_source = _normalize_source_name(detail.get("generation_mode"))
                    if source_gating_enabled:
                        keep_detail, reasons, gate_meta = _maybe_gate_qwen_detail(detail)
                        detail["gate_metadata"] = gate_meta
                        if not keep_detail:
                            source_gate_reject_hist[normalized_source] += 1
                            for reason in reasons:
                                source_gate_reject_reason_hist[reason] += 1
                            return
                    seen_signatures.add(signature)
                    source_gate_counts[normalized_source] += 1
                    source_gate_accept_hist[normalized_source] += 1
                    qwen_pred_details.append(detail)

                def _run_qwen_strategy(strategy: dict, *, repeat_index: int = 0) -> dict:
                    nonlocal total_attempts
                    total_attempts += 1
                    strategy_name = str(strategy.get("name", f"strategy_{total_attempts}"))
                    point_first_prompt = bool(strategy.get("point_first_prompt", False))
                    web_mobile_hotspot_prompt = bool(strategy.get("web_mobile_hotspot_prompt", False))
                    decoupled_point_native_decode = bool(strategy.get("decoupled_point_native_decode", False))
                    generation_temperature = float(strategy.get("temperature", sampled_temperature))
                    pred, raw_text, parsed_payload = qwen_model.predict_with_details(
                        sample,
                        temperature=generation_temperature,
                        point_first_prompt=point_first_prompt,
                        web_mobile_hotspot_prompt=web_mobile_hotspot_prompt,
                        decoupled_point_native_decode=decoupled_point_native_decode,
                    )
                    detail = _build_qwen_generation_detail(
                        pred=pred,
                        raw_text=raw_text,
                        parsed_payload=parsed_payload,
                        generation_mode=strategy_name,
                        generation_temperature=generation_temperature,
                        attempt_index=total_attempts,
                        point_first_prompt=point_first_prompt,
                        web_mobile_hotspot_prompt=web_mobile_hotspot_prompt,
                        decoupled_point_native_decode=decoupled_point_native_decode,
                        source_family="point_native" if decoupled_point_native_decode else "structured",
                    )
                    strategy_bank.setdefault(strategy_name, []).append(detail)
                    return detail

                if generation_strategies:
                    for strategy in generation_strategies:
                        repeat = max(int(strategy.get("repeat", 1)), 1)
                        for repeat_index in range(repeat):
                            if len(qwen_pred_details) >= top_k:
                                break
                            detail = _run_qwen_strategy(strategy, repeat_index=repeat_index)
                            _maybe_add_qwen_detail(detail)
                        if len(qwen_pred_details) >= top_k:
                            break

                    for recipe in hybrid_candidate_recipes:
                        if len(qwen_pred_details) >= top_k:
                            break
                        click_source_name = str(recipe.get("click_source", ""))
                        support_source_name = str(recipe.get("support_source", ""))
                        click_candidates = strategy_bank.get(click_source_name, [])
                        support_candidates = strategy_bank.get(support_source_name, [])
                        if not click_candidates or not support_candidates:
                            continue
                        hybrid_detail = _build_hybrid_qwen_detail(
                            sample=sample,
                            image_size=image.size,
                            generation_mode=str(recipe.get("name", "hybrid_point_structured")),
                            click_detail=click_candidates[0],
                            support_detail=support_candidates[0],
                            attempt_index=total_attempts + 1,
                        )
                        if hybrid_detail is None:
                            continue
                        total_attempts = int(hybrid_detail["attempt_index"])
                        strategy_bank.setdefault(str(recipe.get("name", "hybrid_point_structured")), []).append(
                            hybrid_detail
                        )
                        _maybe_add_qwen_detail(hybrid_detail)
                else:
                    baseline_detail = _run_qwen_strategy(
                        {
                            "name": "stagea_first_choice",
                            "temperature": first_choice_temperature,
                            "point_first_prompt": False,
                            "web_mobile_hotspot_prompt": False,
                            "decoupled_point_native_decode": False,
                        }
                    )
                    _maybe_add_qwen_detail(baseline_detail)

                if top_k > 1:
                    fallback_strategies = [
                        dict(strategy)
                        for strategy in generation_strategies
                        if float(strategy.get("temperature", 0.0)) > 0.0
                    ]
                    if not fallback_strategies:
                        fallback_strategies = [
                            {
                                "name": "sampled_candidate",
                                "temperature": sampled_temperature,
                                "point_first_prompt": False,
                                "web_mobile_hotspot_prompt": False,
                                "decoupled_point_native_decode": False,
                            }
                        ]
                    fallback_index = 0
                    while len(qwen_pred_details) < top_k and total_attempts < max_generation_attempts:
                        strategy = fallback_strategies[fallback_index % len(fallback_strategies)]
                        fallback_index += 1
                        strategy_copy = dict(strategy)
                        if generation_strategies:
                            strategy_copy["name"] = f"{strategy_copy.get('name', 'sampled_candidate')}_fallback"
                        detail = _run_qwen_strategy(strategy_copy)
                        _maybe_add_qwen_detail(detail)

                    while len(qwen_pred_details) < top_k and duplicate_details:
                        detail = dict(duplicate_details.pop(0))
                        detail["generation_mode"] = f"{detail['generation_mode']}_duplicate_fill"
                        qwen_pred_details.append(detail)

            gt_bbox = sample.target_bbox.as_tuple() if sample.target_bbox else None
            gt_action = str(sample.action_type) if sample.action_type is not None else None
            candidates = []
            if backend == "clip_grid_stagea":
                for hyp in hypotheses:
                    bbox = _grid_to_bbox(hyp["grid_id"], image.size, grid_rows, grid_cols)
                    click = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    action_type = ACTION_TYPES[hyp["action_id"]]
                    reward_result = reward_calc.compute(
                        sample_id=sample.sample_id,
                        pred_element_id=None,
                        gt_element_id=sample.target_element_id,
                        pred_bbox=bbox,
                        gt_bbox=gt_bbox,
                        pred_click=click,
                        pred_action=action_type,
                        gt_action=gt_action,
                        image_width=image.size[0],
                        image_height=image.size[1],
                    )
                    candidates.append(
                        {
                            "candidate_id": f"{sample.sample_id}_model_{hyp['rank']}",
                            "proposal_id": f"{sample.sample_id}_bbox_{hyp['rank']}",
                            "rank": hyp["rank"],
                            "action_type": action_type,
                            "bbox_proposal": [float(v) for v in bbox],
                            "click_point": [float(click[0]), float(click[1])],
                            "element_hint_id": None,
                            "score": hyp["score"],
                            "confidence": hyp["confidence"],
                            "joint_log_prob": hyp["joint_log_prob"],
                            "action_log_prob": hyp["action_log_prob"],
                            "grid_log_prob": hyp["grid_log_prob"],
                            "source": "legacy_clip_grid_stagea",
                            "legacy_metadata": {
                                "grid_id": hyp["grid_id"],
                                "legacy_element_id": f"grid_{hyp['grid_id']}",
                            },
                            "structured_output_diagnostics": _default_candidate_diagnostics(),
                            "reward": {
                                "total_reward": reward_result.total_reward,
                                "is_valid_format": reward_result.is_valid_format,
                                "components": reward_result.components.model_dump(),
                            },
                        }
                    )
            else:
                if safe_run_enabled and log_qwen_generate_diagnostics:
                    logger.info(
                        "sample_id=%s image_size=%s qwen_generate_stats=%s",
                        sample.sample_id,
                        image.size,
                        qwen_model.backbone.last_generate_stats,
                    )
                for rank, detail in enumerate(qwen_pred_details, start=1):
                    pred = detail["pred"]
                    bbox_tuple = pred.predicted_bbox.as_tuple() if pred.predicted_bbox else None
                    click_tuple = pred.predicted_click_point
                    if bbox_tuple is None and click_tuple is not None:
                        cx, cy = click_tuple
                        delta = 12.0
                        bbox_tuple = (
                            max(0.0, cx - delta),
                            max(0.0, cy - delta),
                            min(float(image.size[0]), cx + delta),
                            min(float(image.size[1]), cy + delta),
                        )
                    action_type = pred.predicted_action_type
                    reward_result = reward_calc.compute(
                        sample_id=sample.sample_id,
                        pred_element_id=pred.predicted_element_id,
                        gt_element_id=sample.target_element_id,
                        pred_bbox=bbox_tuple,
                        gt_bbox=gt_bbox,
                        pred_click=click_tuple,
                        pred_action=action_type,
                        gt_action=gt_action,
                        image_width=image.size[0],
                        image_height=image.size[1],
                    )
                    conf = float(pred.confidence) if pred.confidence is not None else (1.0 / max(top_k, 1))
                    final_bbox = [float(v) for v in bbox_tuple] if bbox_tuple else None
                    final_click = [float(click_tuple[0]), float(click_tuple[1])] if click_tuple else None
                    if detail["raw_text"] or detail["parsed_payload"]:
                        structured_output_diagnostics = _build_qwen_structured_output_diagnostics(
                            raw_text=detail["raw_text"],
                            parsed_payload=detail["parsed_payload"],
                            pred=pred,
                            final_bbox=final_bbox,
                            final_click=final_click,
                        )
                    else:
                        structured_output_diagnostics = _default_candidate_diagnostics()
                    dom_match = _build_dom_match_metadata(
                        sample,
                        bbox=bbox_tuple,
                        click=click_tuple,
                        element_hint_id=pred.predicted_element_id,
                    )
                    candidates.append(
                        {
                            "candidate_id": f"{sample.sample_id}_qwen_{rank}",
                            "proposal_id": f"{sample.sample_id}_bbox_{rank}",
                            "rank": rank,
                            "action_type": action_type,
                            "bbox_proposal": final_bbox,
                            "click_point": final_click,
                            "element_hint_id": pred.predicted_element_id,
                            "score": conf,
                            "confidence": conf,
                            "joint_log_prob": None,
                            "action_log_prob": None,
                            "grid_log_prob": None,
                            "source": detail.get("generation_mode", "qwen_vl_generation"),
                            "provenance": {
                                "source_model": model_name,
                                "adapter_path": adapter_path,
                                "stagea_representation": (
                                    "hybrid_candidate_anchored"
                                    if qwen_model.use_candidate_anchors
                                    else "visual_only"
                                ),
                                "generation_mode": detail.get("generation_mode", "qwen_vl_generation"),
                                "generation_temperature": detail.get("generation_temperature"),
                                "attempt_index": detail.get("attempt_index"),
                                "source_family": detail.get("source_family", "structured"),
                                "point_first_prompt": bool(detail.get("point_first_prompt", False)),
                                "web_mobile_hotspot_prompt": bool(detail.get("web_mobile_hotspot_prompt", False)),
                                "decoupled_point_native_decode": bool(
                                    detail.get("decoupled_point_native_decode", False)
                                ),
                                "coordinate_frame": qwen_model.coordinate_frame,
                                "coordinate_format": qwen_model.coordinate_format,
                                "target_field_order": qwen_model.target_field_order,
                                "point_primary_bbox_anchored_prompt": qwen_model.point_primary_bbox_anchored_prompt,
                                "use_candidate_anchors": qwen_model.use_candidate_anchors,
                                "max_prompt_candidates": qwen_model.max_prompt_candidates,
                                "candidate_grounding_from_slot": qwen_model.candidate_grounding_from_slot,
                                "extra_provenance": detail.get("extra_provenance", {}),
                            },
                            "gating_metadata": detail.get("gate_metadata"),
                            "parser_metadata": _build_parser_metadata(detail.get("parsed_payload")),
                            "dom_match": dom_match,
                            "legacy_metadata": {},
                            "structured_output_diagnostics": structured_output_diagnostics,
                            "reward": {
                                "total_reward": reward_result.total_reward,
                                "is_valid_format": reward_result.is_valid_format,
                                "components": reward_result.components.model_dump(),
                            },
                        }
                    )

            sample_row = {
                "sample_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "split": sample.split,
                "instruction": sample.instruction,
                "image_path": sample.image_path,
                "image_width": image.size[0],
                "image_height": image.size[1],
                "target_action_type": gt_action,
                "target_bbox": [float(v) for v in gt_bbox] if gt_bbox else None,
                "target_click_point": list(sample.click_point) if sample.click_point else None,
                "target_element_id": sample.target_element_id,
                "candidate_semantics": "bbox_proposal_click_point_action_type",
                "candidate_schema": PRIMARY_CANDIDATE_SCHEMA,
                "ocr_text_hint": sample.ocr_text,
                "dom_candidates_available": bool(sample.dom_candidates),
                "dom_candidates_count": len(sample.dom_candidates) if sample.dom_candidates else 0,
                "backend": backend,
                "stagea_representation": (
                    "hybrid_candidate_anchored"
                    if qwen_model and qwen_model.use_candidate_anchors
                    else "visual_only"
                ),
                "checkpoint_model_path": str(checkpoint_model_path) if checkpoint_model_path else None,
                "adapter_path": adapter_path,
                "top_k": top_k,
                "candidates": candidates,
            }
            sample_rows.append(sample_row)
            if incremental_save:
                _append_jsonl(sample_jsonl, sample_row)

            for candidate in candidates:
                flat_row = {
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "instruction": sample.instruction,
                    "image_path": sample.image_path,
                    "image_width": image.size[0],
                    "image_height": image.size[1],
                    **candidate,
                }
                flat_rows.append(flat_row)
                if incremental_save:
                    _append_jsonl(flat_jsonl, flat_row)

            if candidates:
                reward_values = [c["reward"]["total_reward"] for c in candidates]
                total_candidates += len(candidates)
                reward_sum += sum(reward_values)
                best_reward_sum += max(reward_values)
            successful_samples += 1
            if log_cuda_memory_per_sample:
                post_mem = _cuda_memory_stats()
                _update_peak_cuda_stats(peak_cuda, post_mem)
                logger.info("sample_id=%s post_cuda_mem=%s", sample.sample_id, post_mem)
        except Exception as exc:  # noqa: BLE001
            failed_samples += 1
            failure_row = {
                "sample_id": sample.sample_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "cuda_memory": _cuda_memory_stats(),
                "utc_time": datetime.now(timezone.utc).isoformat(),
            }
            _update_peak_cuda_stats(peak_cuda, failure_row["cuda_memory"])
            failure_rows.append(failure_row)
            last_failed_sample_id = sample.sample_id
            logger.error("Sample failed sample_id=%s error=%s", sample.sample_id, exc, exc_info=True)
            logger.error("Sample failure CUDA memory snapshot: %s", failure_row["cuda_memory"])
            if clear_cuda_cache_on_error and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception as cuda_exc:  # noqa: BLE001
                    logger.warning("CUDA cleanup failed after sample_id=%s: %s", sample.sample_id, cuda_exc)
            continue

    if not incremental_save:
        save_jsonl(sample_rows, sample_jsonl)
        save_jsonl(flat_rows, flat_jsonl)

    run_id = cfg.get("experiment", {}).get("run_id") if cfg.get("experiment") else None
    config_signature = {
        "backend": backend,
        "dtype": model_cfg.get("torch_dtype", "auto"),
        "attention_backend": model_cfg.get("attn_implementation", "sdpa"),
        "top_k": top_k,
        "max_new_tokens": int(model_cfg.get("max_new_tokens", 256)),
        "num_samples": len(samples),
        "adapter_path": adapter_path,
        "stagea_representation": (
            "hybrid_candidate_anchored"
            if qwen_model and qwen_model.use_candidate_anchors
            else "visual_only"
        ),
        "first_choice_temperature": first_choice_temperature,
        "sampled_temperature": sampled_temperature,
        "generation_strategies": _to_jsonable(generation_strategies),
        "hybrid_candidate_recipes": _to_jsonable(hybrid_candidate_recipes),
        "source_gating": _to_jsonable(source_gating_cfg),
        "coordinate_frame": model_cfg.get("coordinate_frame", "original"),
        "coordinate_format": model_cfg.get("coordinate_format", "absolute"),
        "target_field_order": model_cfg.get("target_field_order"),
        "point_primary_bbox_anchored_prompt": bool(model_cfg.get("point_primary_bbox_anchored_prompt", False)),
        "use_candidate_anchors": bool(model_cfg.get("use_candidate_anchors", False)),
        "max_prompt_candidates": int(model_cfg.get("max_prompt_candidates", 32)),
        "candidate_grounding_from_slot": bool(model_cfg.get("candidate_grounding_from_slot", True)),
    }
    final_failure_type = failure_rows[-1]["error_type"] if failure_rows else None
    summary = {
        "run_id": run_id,
        "config_signature": config_signature,
        "candidate_schema": PRIMARY_CANDIDATE_SCHEMA,
        "backend": backend,
        "checkpoint_model_path": str(checkpoint_model_path) if checkpoint_model_path else None,
        "model_backbone": model_name,
        "adapter_path": adapter_path,
        "device": device,
        "split": split,
        "num_samples": len(sample_rows),
        "attempted_samples": attempted_samples,
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "run_survived": True,
        "last_processed_sample_id": last_processed_sample_id,
        "last_failed_sample_id": last_failed_sample_id,
        "final_failure_type": final_failure_type,
        "peak_gpu_memory": peak_cuda,
        "top_k": top_k,
        "num_candidates_total": total_candidates,
        "avg_reward": reward_sum / max(total_candidates, 1),
        "avg_best_reward_per_sample": best_reward_sum / max(len(sample_rows), 1),
        "selection_strategy": selection_strategy,
        "max_per_action": max_per_action,
        "deduplicate_candidates": deduplicate_candidates,
        "max_generation_attempts": max_generation_attempts,
        "generation_strategies": _to_jsonable(generation_strategies),
        "hybrid_candidate_recipes": _to_jsonable(hybrid_candidate_recipes),
        "source_gating": _to_jsonable(source_gating_cfg),
        "safe_run": _to_jsonable(safe_cfg),
        "source_gate_accept_histogram": dict(source_gate_accept_hist),
        "source_gate_reject_histogram": dict(source_gate_reject_hist),
        "source_gate_reject_reason_histogram": dict(source_gate_reject_reason_hist),
        "runtime_settings": {
            "torch_dtype": model_cfg.get("torch_dtype", "auto"),
            "resolved_torch_dtype": str(qwen_model.backbone.resolved_torch_dtype) if qwen_model else None,
            "attn_implementation": qwen_model.backbone.attn_implementation if qwen_model else None,
            "gpu_memory_utilization": model_cfg.get("gpu_memory_utilization", 0.9),
            "max_new_tokens": model_cfg.get("max_new_tokens", 256),
            "temperature": model_cfg.get("temperature", 0.3),
            "first_choice_temperature": first_choice_temperature,
            "sampled_temperature": sampled_temperature,
            "generation_strategies": _to_jsonable(generation_strategies),
            "hybrid_candidate_recipes": _to_jsonable(hybrid_candidate_recipes),
            "source_gating": _to_jsonable(source_gating_cfg),
            "coordinate_frame": model_cfg.get("coordinate_frame", "original"),
            "coordinate_format": model_cfg.get("coordinate_format", "absolute"),
            "target_field_order": model_cfg.get("target_field_order"),
            "point_primary_bbox_anchored_prompt": bool(model_cfg.get("point_primary_bbox_anchored_prompt", False)),
            "use_candidate_anchors": bool(model_cfg.get("use_candidate_anchors", False)),
            "max_prompt_candidates": int(model_cfg.get("max_prompt_candidates", 32)),
            "candidate_grounding_from_slot": bool(model_cfg.get("candidate_grounding_from_slot", True)),
            "min_pixels": model_cfg.get("min_pixels"),
            "max_pixels": model_cfg.get("max_pixels"),
            "cuda_launch_blocking": os.getenv("CUDA_LAUNCH_BLOCKING", "0"),
        },
        "candidates_jsonl": str(sample_jsonl),
        "flat_candidates_jsonl": str(flat_jsonl),
        "failure_json": str(failure_json),
        "structured_quality_json": str(structured_quality_json),
        "action_histogram_json": str(action_histogram_json),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(failure_json, "w", encoding="utf-8") as f:
        json.dump(failure_rows, f, indent=2, ensure_ascii=False)
    runtime_details = {
        "config_path": args.config,
        "overrides": args.overrides,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "failures": failure_rows,
    }
    with open(runtime_json, "w", encoding="utf-8") as f:
        json.dump(runtime_details, f, indent=2, ensure_ascii=False)

    quality_summary = _build_quality_summary(
        config_path=args.config,
        backend=backend,
        model_name=model_name,
        split=split,
        sample_rows=sample_rows,
        flat_rows=flat_rows,
        failure_rows=failure_rows,
        summary=summary,
        runtime_details=runtime_details,
    )
    with open(structured_quality_json, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, indent=2, ensure_ascii=False)
    with open(action_histogram_json, "w", encoding="utf-8") as f:
        json.dump(
            quality_summary["candidate_stats"]["action_type_distribution"],
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(structured_quality_md, "w", encoding="utf-8") as f:
        f.write("# Structured Candidate Export Quality Summary\n\n")
        f.write(f"- config_path: `{args.config}`\n")
        f.write(f"- backend: `{backend}`\n")
        f.write(f"- model_backbone: `{model_name}`\n")
        f.write(f"- split: `{split}`\n")
        f.write(f"- attempted_samples: {quality_summary['sample_stats']['attempted_samples']}\n")
        f.write(f"- successful_end_to_end_runs: {quality_summary['sample_stats']['successful_end_to_end_runs']}\n")
        f.write(f"- failed_runs: {quality_summary['sample_stats']['failed_runs']}\n")
        f.write(f"- total_candidates: {quality_summary['candidate_stats']['total_candidates']}\n")
        f.write(
            f"- parseable_structured_outputs: {quality_summary['candidate_stats']['parseable_structured_outputs']} "
            f"({quality_summary['candidate_stats']['parseable_structured_output_rate']:.4f})\n"
        )
        f.write(
            f"- valid_bbox_proposal_count: {quality_summary['candidate_stats']['valid_bbox_proposal_count']} "
            f"({quality_summary['candidate_stats']['valid_bbox_proposal_rate']:.4f})\n"
        )
        f.write(
            f"- valid_click_point_count: {quality_summary['candidate_stats']['valid_click_point_count']} "
            f"({quality_summary['candidate_stats']['valid_click_point_rate']:.4f})\n"
        )
        f.write(
            f"- valid_action_type_count: {quality_summary['candidate_stats']['valid_action_type_count']} "
            f"({quality_summary['candidate_stats']['valid_action_type_rate']:.4f})\n"
        )
        f.write(
            f"- avg_candidates_per_sample: {quality_summary['candidate_stats']['avg_candidates_per_sample']:.4f}\n"
        )
        f.write(
            f"- empty_candidate_rate: {quality_summary['sample_stats']['empty_candidate_rate']:.4f}\n"
        )
        f.write(
            f"- malformed_json_rate: {quality_summary['candidate_stats']['malformed_json_rate']:.4f}\n"
        )
        f.write(
            f"- malformed_coordinate_rate: {quality_summary['candidate_stats']['malformed_coordinate_rate']:.4f}\n"
        )
        f.write("\n## Action Type Distribution\n\n")
        for key, value in quality_summary["candidate_stats"]["action_type_distribution"].items():
            f.write(f"- {key}: {value}\n")
        f.write("\n## Common Parse Failure Categories\n\n")
        for key, value in quality_summary["candidate_stats"]["common_parse_failure_categories"].items():
            f.write(f"- {key}: {value}\n")

    logger.info("Saved sample-level candidates JSONL: %s", sample_jsonl)
    logger.info("Saved flat candidates JSONL: %s", flat_jsonl)
    logger.info("Saved summary JSON: %s", summary_json)
    logger.info("Saved failure JSON: %s", failure_json)
    logger.info("Saved runtime JSON: %s", runtime_json)
    logger.info("Saved structured quality JSON: %s", structured_quality_json)
    logger.info("Saved structured quality MD: %s", structured_quality_md)
    logger.info("Saved action histogram JSON: %s", action_histogram_json)
    logger.info(
        "Done. attempted=%d success=%d failed=%d candidates=%d",
        attempted_samples,
        successful_samples,
        failed_samples,
        total_candidates,
    )


if __name__ == "__main__":
    main()
    if os.getenv("GUI_GROUNDING_LEGACY_HARD_EXIT", "0") == "1":
        _exit_cleanly(0)
