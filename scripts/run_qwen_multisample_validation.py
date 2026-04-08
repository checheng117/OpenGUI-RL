#!/usr/bin/env python3
"""Run Qwen-first inference sweep on multiple real Mind2Web samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.data.mind2web_dataset import Mind2WebDataset
from gui_grounding.models.qwen2_vl_grounding import QwenVLGroundingModel
from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed
from gui_grounding.utils.visualization import draw_prediction, save_visualization

logger = get_logger("run_qwen_multisample_validation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen multisample validation on Mind2Web")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=12)
    p.add_argument("--backend", type=str, default="qwen2_5_vl_3b")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--torch-dtype", type=str, default="auto")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--attn-implementation", type=str, default="sdpa")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--output-dir", type=str, default="outputs/qwen_multisample_validation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _failure_tags(raw_text: str, parsed: dict, pred) -> list[str]:
    tags: list[str] = []
    if not raw_text.strip():
        tags.append("empty_raw_response")
    if not parsed:
        tags.append("unparseable_json")
    if pred.predicted_action_type is None:
        tags.append("missing_action_type")
    if pred.predicted_bbox is None:
        tags.append("missing_bbox")
    if pred.predicted_click_point is None:
        tags.append("missing_click_point")
    return tags


def main() -> None:
    _bootstrap_env()
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    per_sample_dir = out_dir / "per_sample_json"
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Mind2Web split=%s max_samples=%d", args.split, args.max_samples)
    ds = Mind2WebDataset(split=args.split, max_samples=args.max_samples, cache_screenshots=True)
    samples = [s for s in ds if s.image_path and Path(s.image_path).exists()]
    if not samples:
        raise RuntimeError("No valid samples with existing screenshots.")
    logger.info("Prepared %d samples for sweep.", len(samples))

    model = QwenVLGroundingModel(
        model_name=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        attn_implementation=args.attn_implementation,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    attempted = len(samples)
    success_runs = 0
    parseable = 0
    valid_bbox = 0
    valid_click_point = 0
    action_hist: Counter[str] = Counter()
    failure_hist: Counter[str] = Counter()
    rows = []

    for idx, sample in enumerate(samples):
        record = {
            "index": idx,
            "sample_id": sample.sample_id,
            "split": sample.split,
            "instruction": sample.instruction,
            "image_path": sample.image_path,
            "status": "failed",
            "failure_tags": [],
        }
        try:
            pred, raw_text, parsed_payload = model.predict_with_details(sample)
            success_runs += 1
            if parsed_payload:
                parseable += 1
            if pred.predicted_bbox is not None:
                valid_bbox += 1
            if pred.predicted_click_point is not None:
                valid_click_point += 1
            action_hist[str(pred.predicted_action_type) if pred.predicted_action_type else "__none__"] += 1

            tags = _failure_tags(raw_text=raw_text, parsed=parsed_payload, pred=pred)
            for t in tags:
                failure_hist[t] += 1

            record.update(
                {
                    "status": "ok",
                    "prediction": pred.model_dump(),
                    "raw_model_response": raw_text,
                    "parsed_model_payload": parsed_payload,
                    "failure_tags": tags,
                }
            )

            try:
                image = Image.open(sample.image_path).convert("RGB")
                pred_bbox = pred.predicted_bbox.as_tuple() if pred.predicted_bbox else None
                vis = draw_prediction(
                    image=image,
                    pred_bbox=pred_bbox,
                    pred_point=pred.predicted_click_point,
                    action_type=pred.predicted_action_type,
                )
                vis_path = vis_dir / f"{sample.sample_id}.png"
                save_visualization(vis, vis_path)
                record["visualization_path"] = str(vis_path)
            except (FileNotFoundError, UnidentifiedImageError) as exc:
                failure_hist["visualization_error"] += 1
                record["visualization_error"] = str(exc)
        except Exception as exc:
            tag = f"inference_exception:{type(exc).__name__}"
            record["failure_tags"] = [tag]
            record["error"] = str(exc)
            failure_hist[tag] += 1

        row_path = per_sample_dir / f"{sample.sample_id}.json"
        with open(row_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        rows.append(record)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_backend": args.backend,
        "attempted_samples": attempted,
        "successful_runs": success_runs,
        "parseable_outputs": parseable,
        "valid_bbox_outputs": valid_bbox,
        "valid_click_point_outputs": valid_click_point,
        "success_rate": _safe_rate(success_runs, attempted),
        "parseability_rate": _safe_rate(parseable, attempted),
        "valid_bbox_rate": _safe_rate(valid_bbox, attempted),
        "valid_click_point_rate": _safe_rate(valid_click_point, attempted),
        "action_type_histogram": dict(action_hist),
        "failure_category_histogram": dict(failure_hist),
        "output_dir": str(out_dir),
        "per_sample_json_dir": str(per_sample_dir),
        "visualization_dir": str(vis_dir),
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    summary_md = out_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Qwen Multisample Validation Summary\n\n")
        f.write(f"- attempted_samples: {summary['attempted_samples']}\n")
        f.write(f"- successful_runs: {summary['successful_runs']}\n")
        f.write(f"- parseable_outputs: {summary['parseable_outputs']}\n")
        f.write(f"- valid_bbox_outputs: {summary['valid_bbox_outputs']}\n")
        f.write(f"- valid_click_point_outputs: {summary['valid_click_point_outputs']}\n")
        f.write(f"- success_rate: {summary['success_rate']:.4f}\n")
        f.write(f"- parseability_rate: {summary['parseability_rate']:.4f}\n")
        f.write(f"- valid_bbox_rate: {summary['valid_bbox_rate']:.4f}\n")
        f.write(f"- valid_click_point_rate: {summary['valid_click_point_rate']:.4f}\n\n")
        f.write("## Action Type Histogram\n\n")
        for k, v in summary["action_type_histogram"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Failure Categories\n\n")
        for k, v in summary["failure_category_histogram"].items():
            f.write(f"- {k}: {v}\n")

    logger.info("Sweep done. Summary JSON: %s", summary_json)
    logger.info("Sweep done. Summary MD: %s", summary_md)


if __name__ == "__main__":
    main()
