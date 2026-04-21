#!/usr/bin/env python3
"""Run VisualWebBench supplementary grounding evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.constants import ACTION_TYPES
from gui_grounding.data.visualwebbench_dataset import VisualWebBenchDataset
from gui_grounding.evaluation.visualwebbench_metrics import (
    aggregate_visualwebbench_records,
    render_visualwebbench_summary_table,
    score_visualwebbench_grounding,
)
from gui_grounding.models.qwen2_vl_grounding import QwenVLGroundingModel
from gui_grounding.utils import load_config
from gui_grounding.utils.io import save_json
from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed

logger = get_logger("run_eval_visualwebbench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisualWebBench grounding evaluation")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_action_type(value: Any) -> str | None:
    if value is None:
        return None
    action = str(value).strip().lower()
    return action if action in ACTION_TYPES else None


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_config(args.config)
    evaluation_cfg = cfg.get("evaluation", {})
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})
    runtime_cfg = cfg.get("runtime", {})

    return argparse.Namespace(
        config=args.config,
        evaluation_name=evaluation_cfg.get("name", "visualwebbench_eval"),
        evaluation_description=evaluation_cfg.get("description"),
        hf_dataset_id=dataset_cfg.get("hf_dataset_id", "visualwebbench/VisualWebBench"),
        split=dataset_cfg.get("split", "test"),
        task_types=list(dataset_cfg.get("task_types", ["element_ground", "action_ground"])),
        max_samples=dataset_cfg.get("max_samples"),
        image_variant=dataset_cfg.get("image_variant", "raw"),
        backbone=model_cfg.get("backbone", "qwen2_5_vl_3b"),
        model_adapter=model_cfg.get("adapter", "qwen_structured_json"),
        adapter_path=model_cfg.get("adapter_path"),
        device=model_cfg.get("device", "cuda"),
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.7),
        max_new_tokens=model_cfg.get("max_new_tokens", 192),
        temperature=model_cfg.get("temperature", 0.0),
        min_pixels=model_cfg.get("min_pixels", 65536),
        max_pixels=model_cfg.get("max_pixels", 524288),
        coordinate_frame=model_cfg.get("coordinate_frame", "original"),
        coordinate_format=model_cfg.get("coordinate_format", "absolute"),
        point_first_prompt=bool(model_cfg.get("point_first_prompt", False)),
        target_field_order=model_cfg.get("target_field_order"),
        point_primary_bbox_anchored_prompt=bool(model_cfg.get("point_primary_bbox_anchored_prompt", False)),
        use_candidate_anchors=bool(model_cfg.get("use_candidate_anchors", False)),
        max_prompt_candidates=model_cfg.get("max_prompt_candidates", 8),
        candidate_grounding_from_slot=bool(model_cfg.get("candidate_grounding_from_slot", True)),
        web_mobile_hotspot_prompt=bool(model_cfg.get("web_mobile_hotspot_prompt", False)),
        decoupled_point_native_decode=bool(model_cfg.get("decoupled_point_native_decode", False)),
        edge_click_interior_threshold=float(model_cfg.get("edge_click_interior_threshold", 0.0)),
        edge_click_interior_position=float(model_cfg.get("edge_click_interior_position", 0.45)),
        output_dir=output_cfg.get("output_dir", "outputs/visualwebbench_eval"),
        seed=int(runtime_cfg.get("seed", 42)),
        log_every=int(runtime_cfg.get("log_every", 25)),
    )


def _build_model(args: argparse.Namespace) -> tuple[QwenVLGroundingModel, dict[str, Any], str]:
    if args.model_adapter != "qwen_structured_json":
        raise ValueError(f"Unsupported model_adapter={args.model_adapter}")

    model = QwenVLGroundingModel(
        model_name=args.backbone,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        attn_implementation=args.attn_implementation,
        gpu_memory_utilization=args.gpu_memory_utilization,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        coordinate_frame=args.coordinate_frame,
        coordinate_format=args.coordinate_format,
        point_first_prompt=args.point_first_prompt,
        target_field_order=args.target_field_order,
        point_primary_bbox_anchored_prompt=args.point_primary_bbox_anchored_prompt,
        use_candidate_anchors=args.use_candidate_anchors,
        max_prompt_candidates=args.max_prompt_candidates,
        candidate_grounding_from_slot=args.candidate_grounding_from_slot,
        web_mobile_hotspot_prompt=args.web_mobile_hotspot_prompt,
        decoupled_point_native_decode=args.decoupled_point_native_decode,
        edge_click_interior_threshold=args.edge_click_interior_threshold,
        edge_click_interior_position=args.edge_click_interior_position,
        adapter_path=args.adapter_path,
    )
    return model, model.candidate_schema, model.candidate_semantics


def _build_record(
    *,
    sample,
    dataset_index: int,
    pred,
    raw_text: str,
    parsed_payload: dict[str, Any],
    candidate_schema: dict[str, Any],
    candidate_semantics: str,
) -> dict[str, Any]:
    action_type = _normalize_action_type(pred.predicted_action_type)
    candidate_options = sample.metadata.get("candidate_options_absolute") or []
    target_choice_index = sample.metadata.get("answer_index")
    image_size = [sample.metadata.get("image_width"), sample.metadata.get("image_height")]
    candidate_slot_grounded = bool(parsed_payload.get("_candidate_slot_used_for_grounding"))

    score_fields = score_visualwebbench_grounding(
        predicted_bbox=list(pred.predicted_bbox.as_tuple()) if pred.predicted_bbox is not None else None,
        predicted_click_point=list(pred.predicted_click_point) if pred.predicted_click_point is not None else None,
        predicted_action_type=action_type,
        candidate_boxes=candidate_options,
        target_choice_index=target_choice_index,
        image_size=image_size,
        task_type=sample.metadata.get("task_type"),
        website=sample.website,
        predicted_candidate_slot=pred.predicted_candidate_slot,
        candidate_slot_grounded=candidate_slot_grounded,
    )

    return {
        "sample_id": sample.sample_id,
        "dataset_name": sample.dataset_name,
        "split": sample.split,
        "dataset_index": dataset_index,
        "task_type": sample.metadata.get("task_type"),
        "website": sample.website,
        "platform": sample.platform,
        "image_path": sample.image_path,
        "annotated_image_path": sample.metadata.get("annotated_image_path"),
        "image_size": image_size,
        "instruction": sample.instruction,
        "candidate_options_xyxy": candidate_options,
        "status": "ok",
        "raw_model_response": raw_text,
        "raw_response_nonempty": bool(str(raw_text or "").strip()),
        "json_parse_success": bool(parsed_payload),
        "parsed_model_payload": parsed_payload,
        "candidate_schema": candidate_schema,
        "candidate_semantics": candidate_semantics,
        "bbox_proposal": list(pred.predicted_bbox.as_tuple()) if pred.predicted_bbox is not None else None,
        "click_point": list(pred.predicted_click_point) if pred.predicted_click_point is not None else None,
        "action_type": action_type,
        "action_type_valid": action_type is not None,
        "confidence": pred.confidence,
        "element_hint_id": pred.predicted_element_id,
        **score_fields,
    }


def main() -> None:
    _bootstrap_env()
    args = _resolve_args(parse_args())
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_jsonl = output_dir / "predictions.jsonl"
    evaluation_summary_json = output_dir / "evaluation_summary.json"
    subgroup_metrics_json = output_dir / "subgroup_metrics.json"
    summary_table_md = output_dir / "summary_table.md"
    failures_json = output_dir / "failures.json"
    predictions_jsonl.write_text("", encoding="utf-8")

    logger.info("=" * 60)
    logger.info("VisualWebBench Supplementary Grounding Evaluation")
    logger.info("=" * 60)
    logger.info("Eval config: %s", args.config)
    logger.info("Task types: %s", args.task_types)
    logger.info("Image variant: %s", args.image_variant)
    logger.info("Model adapter: %s", args.model_adapter)
    logger.info("Backbone: %s", args.backbone)
    if args.adapter_path:
        logger.info("Adapter path: %s", args.adapter_path)
    logger.info("Use candidate anchors: %s", args.use_candidate_anchors)
    logger.info("Decoupled point-native decode: %s", args.decoupled_point_native_decode)

    dataset = VisualWebBenchDataset(
        split=args.split,
        task_types=args.task_types,
        max_samples=args.max_samples,
        hf_dataset_id=args.hf_dataset_id,
        image_variant=args.image_variant,
    )
    task_counter = Counter(sample.metadata.get("task_type") for sample in dataset)
    logger.info("Loaded %d VisualWebBench samples: %s", len(dataset), dict(task_counter))

    model, candidate_schema, candidate_semantics = _build_model(args)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset_index, sample in enumerate(dataset):
        try:
            pred, raw_text, parsed_payload = model.predict_with_details(sample, temperature=args.temperature)
            record = _build_record(
                sample=sample,
                dataset_index=dataset_index,
                pred=pred,
                raw_text=raw_text,
                parsed_payload=parsed_payload,
                candidate_schema=candidate_schema,
                candidate_semantics=candidate_semantics,
            )
        except Exception as exc:
            failure = {
                "sample_id": sample.sample_id,
                "dataset_index": dataset_index,
                "task_type": sample.metadata.get("task_type"),
                "website": sample.website,
                "instruction": sample.instruction,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            record = {
                "sample_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "split": sample.split,
                "dataset_index": dataset_index,
                "task_type": sample.metadata.get("task_type"),
                "website": sample.website,
                "platform": sample.platform,
                "image_path": sample.image_path,
                "annotated_image_path": sample.metadata.get("annotated_image_path"),
                "image_size": [sample.metadata.get("image_width"), sample.metadata.get("image_height")],
                "instruction": sample.instruction,
                "candidate_options_xyxy": sample.metadata.get("candidate_options_absolute") or [],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_model_response": "",
                "raw_response_nonempty": False,
                "json_parse_success": False,
                "parsed_model_payload": {},
                "candidate_schema": candidate_schema,
                "candidate_semantics": candidate_semantics,
                "bbox_proposal": None,
                "click_point": None,
                "action_type": None,
                "action_type_valid": False,
                "confidence": None,
                "element_hint_id": None,
                **score_visualwebbench_grounding(
                    predicted_bbox=None,
                    predicted_click_point=None,
                    predicted_action_type=None,
                    candidate_boxes=sample.metadata.get("candidate_options_absolute") or [],
                    target_choice_index=sample.metadata.get("answer_index"),
                    image_size=[sample.metadata.get("image_width"), sample.metadata.get("image_height")],
                    task_type=sample.metadata.get("task_type"),
                    website=sample.website,
                ),
            }

        records.append(record)
        _append_jsonl(predictions_jsonl, record)

        if (dataset_index + 1) % args.log_every == 0 or dataset_index + 1 == len(dataset):
            ok_records = [row for row in records if row["status"] == "ok"]
            choice_acc = (
                sum(int(row["official_choice_correct"]) for row in ok_records) / max(len(records), 1)
            )
            logger.info(
                "[%d/%d] running official choice accuracy=%.4f failures=%d",
                dataset_index + 1,
                len(dataset),
                choice_acc,
                len(failures),
            )

    overall_metrics, subgroup_metrics = aggregate_visualwebbench_records(
        records,
        group_fields=["task_type", "target_area_bucket", "distractor_overlap_bucket"],
    )
    evaluation_summary = {
        "evaluation_name": args.evaluation_name,
        "evaluation_description": args.evaluation_description,
        "dataset_name": "visualwebbench",
        "task_types": args.task_types,
        "image_variant": args.image_variant,
        "model": {
            "backbone": args.backbone,
            "adapter": args.model_adapter,
            "adapter_path": args.adapter_path,
            "coordinate_frame": args.coordinate_frame,
            "coordinate_format": args.coordinate_format,
            "use_candidate_anchors": args.use_candidate_anchors,
            "decoupled_point_native_decode": args.decoupled_point_native_decode,
        },
        "task_counts": dict(task_counter),
        "evaluated_samples": overall_metrics["count"],
        "successful_runs": sum(int(row["status"] == "ok") for row in records),
        "failed_runs": sum(int(row["status"] != "ok") for row in records),
        **overall_metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_table_md.write_text(
        render_visualwebbench_summary_table(
            title="VisualWebBench Supplementary Grounding Summary",
            overall=evaluation_summary,
            subgroup_metrics=subgroup_metrics,
        ),
        encoding="utf-8",
    )
    save_json(evaluation_summary, evaluation_summary_json)
    save_json(subgroup_metrics, subgroup_metrics_json)
    save_json(failures, failures_json)

    logger.info("Official choice accuracy: %.4f", evaluation_summary["official_choice_accuracy"])
    logger.info("Point accuracy: %.4f", evaluation_summary["point_accuracy"])
    logger.info("Mean IoU: %.4f", evaluation_summary["mean_iou"])
    logger.info("Saved outputs to %s", output_dir)


if __name__ == "__main__":
    main()
