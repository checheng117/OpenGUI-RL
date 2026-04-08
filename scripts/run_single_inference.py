#!/usr/bin/env python3
"""Run real single-sample GUI grounding inference with Qwen-VL.

Usage:
    python scripts/run_single_inference.py --config configs/demo/single_inference.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.constants import OUTPUTS_DIR
from gui_grounding.data.mind2web_dataset import Mind2WebDataset
from gui_grounding.data.schemas import GroundingSample
from gui_grounding.models.clip_grid_grounding import CLIPGridGroundingModel
from gui_grounding.models.qwen2_vl_grounding import QwenVLGroundingModel
from gui_grounding.utils import load_config
from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed
from gui_grounding.utils.visualization import draw_prediction, save_visualization

logger = get_logger("run_single_inference")


def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("GUI_GROUNDING_HF_FALLBACK_ENDPOINT", "https://hf-mirror.com")

    has_token = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
    logger.info("HF token loaded: %s", has_token)
    logger.info("HF endpoint: %s", os.getenv("HF_ENDPOINT", "https://huggingface.co"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single Mind2Web sample real inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/demo/single_inference.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--split", type=str, default=None, help="Override dataset split")
    parser.add_argument("--sample-index", type=int, default=None, help="Override sample index")
    parser.add_argument("--max-samples", type=int, default=None, help="Override load cap")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--model-id", type=str, default=None, help="Override HF model id")
    parser.add_argument("--device", type=str, default=None, help="Override device: auto/cuda/cpu")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["qwen2_5_vl_3b", "qwen3_vl_2b", "clip_grid_legacy"],
        default=None,
        help="Model backend to run.",
    )
    parser.add_argument("--image-path", type=str, default=None, help="Direct screenshot path override")
    parser.add_argument("--instruction", type=str, default=None, help="Direct instruction override")
    parser.add_argument("--sample-id", type=str, default=None, help="Direct sample_id override")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config load")
    return parser.parse_args()


def _resolve_from_cfg(args: argparse.Namespace, cfg):
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})
    output_cfg = cfg.get("output", {})

    split = args.split or data_cfg.get("split", "train")
    sample_index = args.sample_index if args.sample_index is not None else data_cfg.get("sample_index", 0)
    max_samples = args.max_samples if args.max_samples is not None else data_cfg.get("max_samples", 8)
    output_dir = args.output_dir or output_cfg.get("output_dir", str(OUTPUTS_DIR / "single_inference"))
    backend = args.backend or model_cfg.get("backend", "qwen2_5_vl_3b")
    model_id = args.model_id or model_cfg.get("backbone", "qwen2_5_vl_3b")
    if args.model_id is None and backend in {"qwen2_5_vl_3b", "qwen3_vl_2b"}:
        model_id = backend
    device = args.device or model_cfg.get("device", "auto")

    return {
        "split": split,
        "sample_index": int(sample_index),
        "max_samples": int(max_samples),
        "cache_screenshots": bool(data_cfg.get("cache_screenshots", True)),
        "output_dir": output_dir,
        "model_id": model_id,
        "device": device,
        "backend": backend,
        "torch_dtype": model_cfg.get("torch_dtype", "auto"),
        "max_new_tokens": int(model_cfg.get("max_new_tokens", 256)),
        "temperature": float(model_cfg.get("temperature", 0.0)),
        "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
        "gpu_memory_utilization": float(model_cfg.get("gpu_memory_utilization", 0.9)),
        "grid_cols": int(model_cfg.get("grid_cols", 6)),
        "grid_rows": int(model_cfg.get("grid_rows", 4)),
        "seed": int(runtime_cfg.get("seed", 42)),
        "legacy_hard_exit_on_success": bool(runtime_cfg.get("legacy_hard_exit_on_success", False)),
        "image_path": args.image_path or data_cfg.get("image_path", None),
        "instruction": args.instruction or data_cfg.get("instruction", None),
        "sample_id": args.sample_id or data_cfg.get("sample_id", "mind2web_local_override"),
    }


def _render_instruction_overlay(
    image: Image.Image,
    instruction: str,
    action_type: str | None,
    confidence: float | None,
) -> Image.Image:
    """Add a small top overlay with instruction/action metadata."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    conf_text = f"{confidence:.3f}" if confidence is not None else "n/a"
    action_text = action_type or "unknown"
    text = (
        f"Instruction: {instruction}\n"
        f"Predicted action: {action_text} | confidence: {conf_text}"
    )
    pad = 8
    line_height = 18
    lines = text.split("\n")
    box_height = pad * 2 + line_height * len(lines)
    draw.rectangle([0, 0, img.width, box_height], fill=(0, 0, 0, 160))
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill="white", font=font)
        y += line_height
    return img


def _exit_cleanly(exit_code: int = 0) -> NoReturn:
    """Legacy workaround for Python 3.13 finalization crashes."""
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def main() -> None:
    _bootstrap_env()
    args = parse_args()
    cfg = load_config(args.config)
    resolved = _resolve_from_cfg(args, cfg)
    logger.info("Loaded config from %s", args.config)
    logger.info("Resolved runtime config: %s", resolved)

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    set_seed(resolved["seed"])
    out_dir = Path(resolved["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if resolved["image_path"] and resolved["instruction"]:
        image_path = str(Path(resolved["image_path"]).resolve())
        sample = GroundingSample(
            sample_id=resolved["sample_id"],
            dataset_name="mind2web",
            split=resolved["split"],
            image_path=image_path,
            instruction=resolved["instruction"],
            metadata={"source": "local_override"},
        )
        logger.info("Using direct local sample override.")
    else:
        # Ensure we load enough rows so that sample_index exists.
        required_samples = max(resolved["sample_index"] + 1, resolved["max_samples"])
        logger.info(
            "Loading Mind2Web split=%s with max_samples=%d",
            resolved["split"],
            required_samples,
        )
        ds = Mind2WebDataset(
            split=resolved["split"],
            max_samples=required_samples,
            cache_screenshots=resolved["cache_screenshots"],
        )
        if len(ds) == 0:
            raise RuntimeError("No samples loaded from Mind2Web.")
        if resolved["sample_index"] >= len(ds):
            raise IndexError(
                f"sample_index={resolved['sample_index']} out of range for loaded dataset size={len(ds)}"
            )
        sample = ds[resolved["sample_index"]]

    logger.info("Selected sample_id=%s", sample.sample_id)
    logger.info("Instruction: %s", sample.instruction)
    logger.info("Screenshot path: %s", sample.image_path)

    logger.info("Loading model backend=%s id=%s", resolved["backend"], resolved["model_id"])
    if resolved["backend"] in {"qwen2_5_vl_3b", "qwen3_vl_2b"}:
        model = QwenVLGroundingModel(
            model_name=resolved["backend"] if args.model_id is None else resolved["model_id"],
            device=resolved["device"],
            torch_dtype=resolved["torch_dtype"],
            max_new_tokens=resolved["max_new_tokens"],
            temperature=resolved["temperature"],
            attn_implementation=resolved["attn_implementation"],
            gpu_memory_utilization=resolved["gpu_memory_utilization"],
        )
    else:
        logger.warning("Using legacy surrogate backend: clip_grid_legacy")
        model = CLIPGridGroundingModel(
            model_name=resolved["model_id"],
            device=resolved["device"],
            grid_cols=resolved["grid_cols"],
            grid_rows=resolved["grid_rows"],
        )

    logger.info("Running real single-sample inference...")
    pred, raw_text, parsed_payload = model.predict_with_details(sample)
    logger.info("Inference done. action=%s confidence=%s", pred.predicted_action_type, pred.confidence)
    logger.info("Raw model response: %s", raw_text)

    try:
        image = Image.open(sample.image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Cannot open screenshot for visualization: {sample.image_path}") from exc

    pred_bbox = pred.predicted_bbox.as_tuple() if pred.predicted_bbox else None
    vis = draw_prediction(
        image=image,
        pred_bbox=pred_bbox,
        pred_point=pred.predicted_click_point,
        action_type=pred.predicted_action_type,
    )
    vis = _render_instruction_overlay(
        image=vis,
        instruction=sample.instruction,
        action_type=pred.predicted_action_type,
        confidence=pred.confidence,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{sample.sample_id}_{timestamp}"
    vis_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"

    save_visualization(vis, vis_path)
    logger.info("Saved qualitative artifact: %s", vis_path)

    payload = {
        "pipeline": "single_real_inference",
        "backend": resolved["backend"],
        "model_id": resolved["model_id"],
        "device": getattr(model, "device", resolved["device"]),
        "sample": sample.model_dump(),
        "prediction": pred.model_dump(),
        "parsed_model_payload": parsed_payload,
        "raw_model_response": raw_text,
        "artifact_path": str(vis_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved structured prediction artifact: %s", json_path)
    logger.info("Single-sample real inference completed successfully.")

    if resolved["legacy_hard_exit_on_success"]:
        _exit_cleanly(0)


if __name__ == "__main__":
    main()
