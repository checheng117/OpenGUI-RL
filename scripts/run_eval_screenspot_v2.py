#!/usr/bin/env python3
"""Run clean Qwen-first held-out evaluation on ScreenSpot-v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.constants import ACTION_TYPES
from gui_grounding.data.screenspot_dataset import ScreenSpotV2Dataset
from gui_grounding.models.qwen2_vl_public_point_baseline import QwenVLPublicPointBaselineModel
from gui_grounding.models.qwen2_vl_grounding import QwenVLGroundingModel
from gui_grounding.reward.verifiable_reward import bbox_iou
from gui_grounding.utils import load_config
from gui_grounding.utils.io import save_json
from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed

logger = get_logger("run_eval_screenspot_v2")

PRIMARY_CANDIDATE_SCHEMA = {
    "version": "bbox_click_action_v1",
    "primary_fields": ["bbox_proposal", "click_point", "action_type"],
    "legacy_fields": ["legacy_metadata.grid_id"],
}
PUBLIC_POINT_CANDIDATE_SCHEMA = QwenVLPublicPointBaselineModel.candidate_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-first ScreenSpot-v2 held-out evaluation")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--hf-dataset-id", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--indices-json", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--model-adapter", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--coordinate-frame", type=str, default=None)
    parser.add_argument("--point-first-prompt", action="store_true", default=None)
    parser.add_argument("--web-mobile-hotspot-prompt", action="store_true", default=None)
    parser.add_argument("--decoupled-point-native-decode", action="store_true", default=None)
    parser.add_argument("--edge-click-interior-threshold", type=float, default=None)
    parser.add_argument("--edge-click-interior-position", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
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


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _normalize_action_type(value) -> str | None:
    if value is None:
        return None
    action = str(value).strip().lower()
    if action in ACTION_TYPES:
        return action
    return None


def _point_inside_bbox(click_point, bbox) -> bool:
    if click_point is None or bbox is None:
        return False
    cx, cy = click_point
    x1, y1, x2, y2 = bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _empty_metrics_dict() -> dict[str, int | float]:
    return {
        "count": 0,
        "run_success_count": 0,
        "valid_bbox_count": 0,
        "valid_click_point_count": 0,
        "valid_action_type_count": 0,
        "parseable_output_count": 0,
        "point_accuracy_hits": 0,
        "iou_at_0_5_hits": 0,
        "iou_sum": 0.0,
    }


def _update_metrics_bucket(bucket: dict, record: dict) -> None:
    bucket["count"] += 1
    bucket["run_success_count"] += int(record["status"] == "ok")
    bucket["valid_bbox_count"] += int(record["bbox_proposal"] is not None)
    bucket["valid_click_point_count"] += int(record["click_point"] is not None)
    bucket["valid_action_type_count"] += int(record["action_type_valid"])
    bucket["parseable_output_count"] += int(record["json_parse_success"])
    bucket["point_accuracy_hits"] += int(record["point_in_box"])
    bucket["iou_at_0_5_hits"] += int(record["iou_at_0_5"])
    bucket["iou_sum"] += float(record["iou"])


def _finalize_metrics(bucket: dict) -> dict[str, int | float]:
    count = int(bucket["count"])
    return {
        "count": count,
        "run_success_rate": _safe_rate(bucket["run_success_count"], count),
        "valid_bbox_rate": _safe_rate(bucket["valid_bbox_count"], count),
        "valid_click_point_rate": _safe_rate(bucket["valid_click_point_count"], count),
        "action_type_valid_rate": _safe_rate(bucket["valid_action_type_count"], count),
        "parseable_output_rate": _safe_rate(bucket["parseable_output_count"], count),
        "point_accuracy": _safe_rate(bucket["point_accuracy_hits"], count),
        "iou@0.5": _safe_rate(bucket["iou_at_0_5_hits"], count),
        "mean_iou": _safe_rate(bucket["iou_sum"], count),
    }


def _render_summary_table(overall: dict, subgroup_metrics: dict) -> str:
    lines = [
        "# ScreenSpot-v2 Held-Out Evaluation Summary",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Evaluated samples | {overall['evaluated_samples']} |",
        f"| Successful runs | {overall['successful_runs']} |",
        f"| Failed runs | {overall['failed_runs']} |",
        f"| Point accuracy | {overall['point_accuracy']:.4f} |",
        f"| IoU@0.5 | {overall['iou@0.5']:.4f} |",
        f"| Mean IoU | {overall['mean_iou']:.4f} |",
        f"| Action type validity | {overall['action_type_valid_rate']:.4f} |",
        f"| Parseable output rate | {overall['parseable_output_rate']:.4f} |",
        f"| Valid bbox rate | {overall['valid_bbox_rate']:.4f} |",
        f"| Valid click_point rate | {overall['valid_click_point_rate']:.4f} |",
        "",
    ]

    for group_name, metrics_by_value in subgroup_metrics.items():
        lines.extend(
            [
                f"## {group_name.replace('_', ' ').title()}",
                "",
                "| Group | Count | Point Acc | IoU@0.5 | Mean IoU | Action Valid |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for subgroup_value, metrics in sorted(metrics_by_value.items()):
            lines.append(
                f"| {subgroup_value} | {metrics['count']} | {metrics['point_accuracy']:.4f} | "
                f"{metrics['iou@0.5']:.4f} | {metrics['mean_iou']:.4f} | {metrics['action_type_valid_rate']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_config(args.config) if args.config else {}
    evaluation_cfg = cfg.get("evaluation", {})
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})
    runtime_cfg = cfg.get("runtime", {})

    def pick(cli_value, cfg_value, default):
        return cli_value if cli_value is not None else (cfg_value if cfg_value is not None else default)

    return argparse.Namespace(
        config=args.config,
        evaluation_name=evaluation_cfg.get("name", "screenspot_v2_eval"),
        evaluation_description=evaluation_cfg.get("description", None),
        hf_dataset_id=pick(args.hf_dataset_id, dataset_cfg.get("hf_dataset_id"), "lscpku/ScreenSpot-v2"),
        split=pick(args.split, dataset_cfg.get("split"), "test"),
        max_samples=pick(args.max_samples, dataset_cfg.get("max_samples"), None),
        start_index=pick(args.start_index, dataset_cfg.get("start_index"), 0),
        end_index=pick(args.end_index, dataset_cfg.get("end_index"), None),
        indices_json=pick(args.indices_json, dataset_cfg.get("indices_json"), None),
        backbone=pick(args.backbone, model_cfg.get("backbone"), "qwen2_5_vl_3b"),
        model_adapter=pick(args.model_adapter, model_cfg.get("adapter"), "qwen_structured_json"),
        device=pick(args.device, model_cfg.get("device"), "cuda"),
        torch_dtype=pick(args.torch_dtype, model_cfg.get("torch_dtype"), "bfloat16"),
        attn_implementation=pick(args.attn_implementation, model_cfg.get("attn_implementation"), "sdpa"),
        gpu_memory_utilization=pick(
            args.gpu_memory_utilization,
            model_cfg.get("gpu_memory_utilization"),
            0.7,
        ),
        max_new_tokens=pick(args.max_new_tokens, model_cfg.get("max_new_tokens"), 256),
        temperature=pick(args.temperature, model_cfg.get("temperature"), 0.2),
        min_pixels=pick(args.min_pixels, model_cfg.get("min_pixels"), 65536),
        max_pixels=pick(args.max_pixels, model_cfg.get("max_pixels"), 524288),
        coordinate_frame=pick(args.coordinate_frame, model_cfg.get("coordinate_frame"), "original"),
        point_first_prompt=pick(args.point_first_prompt, model_cfg.get("point_first_prompt"), False),
        web_mobile_hotspot_prompt=pick(
            args.web_mobile_hotspot_prompt,
            model_cfg.get("web_mobile_hotspot_prompt"),
            False,
        ),
        decoupled_point_native_decode=pick(
            args.decoupled_point_native_decode,
            model_cfg.get("decoupled_point_native_decode"),
            False,
        ),
        edge_click_interior_threshold=pick(
            args.edge_click_interior_threshold,
            model_cfg.get("edge_click_interior_threshold"),
            0.0,
        ),
        edge_click_interior_position=pick(
            args.edge_click_interior_position,
            model_cfg.get("edge_click_interior_position"),
            0.45,
        ),
        output_dir=pick(
            args.output_dir,
            output_cfg.get("output_dir"),
            "outputs/screenspot_v2_eval_qwen2_5_vl_3b",
        ),
        seed=pick(args.seed, runtime_cfg.get("seed"), 42),
        log_every=pick(args.log_every, runtime_cfg.get("log_every"), 50),
    )


def _build_model(args: argparse.Namespace):
    common_kwargs = {
        "model_name": args.backbone,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "attn_implementation": args.attn_implementation,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
    }

    if args.model_adapter == "qwen_public_point_json":
        model = QwenVLPublicPointBaselineModel(
            **common_kwargs,
            coordinate_frame=args.coordinate_frame,
        )
    elif args.model_adapter == "qwen_structured_json":
        model = QwenVLGroundingModel(
            **common_kwargs,
            log_generate_diagnostics=False,
            coordinate_frame=args.coordinate_frame,
            point_first_prompt=args.point_first_prompt,
            web_mobile_hotspot_prompt=args.web_mobile_hotspot_prompt,
            decoupled_point_native_decode=args.decoupled_point_native_decode,
            edge_click_interior_threshold=args.edge_click_interior_threshold,
            edge_click_interior_position=args.edge_click_interior_position,
        )
    else:
        raise ValueError(f"Unsupported model adapter: {args.model_adapter}")

    candidate_schema = getattr(model, "candidate_schema", PRIMARY_CANDIDATE_SCHEMA)
    candidate_semantics = getattr(model, "candidate_semantics", "bbox_proposal_click_point_action_type")
    return model, candidate_schema, candidate_semantics


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
    logger.info("ScreenSpot-v2 Qwen-First Held-Out Evaluation")
    logger.info("=" * 60)
    if args.config:
        logger.info("Eval config: %s", args.config)
        logger.info("Eval name: %s", args.evaluation_name)
    logger.info("HF dataset source: %s", args.hf_dataset_id)
    logger.info("Model adapter: %s", args.model_adapter)
    logger.info("Coordinate frame: %s", args.coordinate_frame)

    dataset = ScreenSpotV2Dataset(
        split=args.split,
        max_samples=args.max_samples,
        hf_dataset_id=args.hf_dataset_id,
        cache_screenshots=True,
    )
    dataset_total_samples = len(dataset)
    start_index = max(args.start_index, 0)
    end_index = dataset_total_samples if args.end_index is None else min(args.end_index, dataset_total_samples)
    if args.indices_json:
        selected_indices = json.loads(Path(args.indices_json).read_text(encoding="utf-8"))
        if not isinstance(selected_indices, list) or not all(isinstance(idx, int) for idx in selected_indices):
            raise ValueError("--indices-json must point to a JSON list[int].")
        selected_indices = [idx for idx in selected_indices if 0 <= idx < dataset_total_samples]
        if not selected_indices:
            raise ValueError("No valid indices remained after applying --indices-json.")
        eval_samples = [dataset[idx] for idx in selected_indices]
        range_description = {
            "mode": "explicit_indices",
            "count": len(selected_indices),
            "min_index": min(selected_indices),
            "max_index": max(selected_indices),
            "indices_json": args.indices_json,
        }
        logger.info(
            "Prepared %d ScreenSpot-v2 samples from explicit index list %s.",
            len(eval_samples),
            args.indices_json,
        )
    else:
        if start_index >= end_index:
            raise ValueError(
                f"Invalid evaluation range: start_index={start_index}, end_index={end_index}, "
                f"dataset_total_samples={dataset_total_samples}"
            )

        selected_indices = list(range(start_index, end_index))
        eval_samples = [dataset[idx] for idx in selected_indices]
        range_description = {
            "mode": "contiguous_range",
            "start_index": start_index,
            "end_index_exclusive": end_index,
        }

    logger.info(
        "Prepared %d ScreenSpot-v2 samples for evaluation selection out of %d total.",
        len(eval_samples),
        dataset_total_samples,
    )
    logger.info("Ground-truth bbox clipping corrections applied during load: %d", dataset.gt_bbox_clipped_count)

    model, candidate_schema, candidate_semantics = _build_model(args)

    failures: list[dict] = []
    action_hist = Counter()
    platform_buckets = defaultdict(_empty_metrics_dict)
    element_type_buckets = defaultdict(_empty_metrics_dict)
    data_source_buckets = defaultdict(_empty_metrics_dict)
    overall_bucket = _empty_metrics_dict()

    started_at = datetime.now(timezone.utc)

    for idx, (dataset_index, sample) in enumerate(zip(selected_indices, eval_samples), start=1):
        gt_bbox = sample.target_bbox.as_tuple() if sample.target_bbox else None
        platform = sample.platform or "__unknown__"
        element_type = str(sample.metadata.get("element_type") or "__unknown__")
        data_source = str(sample.metadata.get("data_source") or "__unknown__")

        record = {
            "dataset_index": dataset_index,
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "hf_dataset_id": args.hf_dataset_id,
            "split": args.split,
            "instruction": sample.instruction,
            "image_path": sample.image_path,
            "platform": platform,
            "element_type": element_type,
            "data_source": data_source,
            "target_bbox_xyxy": list(gt_bbox) if gt_bbox else None,
            "target_click_point": list(sample.click_point) if sample.click_point else None,
            "target_action_type": None,
            "candidate_semantics": candidate_semantics,
            "candidate_schema": candidate_schema,
            "status": "failed",
        }

        try:
            pred, raw_text, parsed_payload = model.predict_with_details(sample)
            pred_bbox = pred.predicted_bbox.as_tuple() if pred.predicted_bbox else None
            pred_click = pred.predicted_click_point
            action_type = _normalize_action_type(pred.predicted_action_type)
            iou = bbox_iou(pred_bbox, gt_bbox) if pred_bbox is not None and gt_bbox is not None else 0.0
            point_in_box = _point_inside_bbox(pred_click, gt_bbox)

            action_hist[action_type or "__invalid_or_missing__"] += 1

            record.update(
                {
                    "status": "ok",
                    "bbox_proposal": [float(v) for v in pred_bbox] if pred_bbox else None,
                    "click_point": [float(pred_click[0]), float(pred_click[1])] if pred_click else None,
                    "action_type": action_type,
                    "action_type_valid": action_type is not None,
                    "element_hint_id": pred.predicted_element_id,
                    "confidence": float(pred.confidence) if pred.confidence is not None else None,
                    "raw_model_response": raw_text,
                    "parsed_model_payload": parsed_payload,
                    "raw_response_nonempty": bool(raw_text.strip()),
                    "json_parse_success": bool(parsed_payload),
                    "iou": float(iou),
                    "iou_at_0_5": bool(iou >= 0.5),
                    "point_in_box": bool(point_in_box),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "sample_id": sample.sample_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "utc_time": datetime.now(timezone.utc).isoformat(),
            }
            failures.append(failure)
            record.update(
                {
                    "bbox_proposal": None,
                    "click_point": None,
                    "action_type": None,
                    "action_type_valid": False,
                    "element_hint_id": None,
                    "confidence": None,
                    "raw_model_response": "",
                    "parsed_model_payload": {},
                    "raw_response_nonempty": False,
                    "json_parse_success": False,
                    "iou": 0.0,
                    "iou_at_0_5": False,
                    "point_in_box": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            action_hist["__inference_exception__"] += 1

        _append_jsonl(predictions_jsonl, record)
        _update_metrics_bucket(overall_bucket, record)
        _update_metrics_bucket(platform_buckets[platform], record)
        _update_metrics_bucket(element_type_buckets[element_type], record)
        _update_metrics_bucket(data_source_buckets[data_source], record)

        if idx % max(args.log_every, 1) == 0:
            logger.info(
                "Processed %d / %d ScreenSpot-v2 samples",
                idx,
                len(eval_samples),
            )

    finished_at = datetime.now(timezone.utc)
    evaluated_samples = len(eval_samples)
    overall_metrics = _finalize_metrics(overall_bucket)
    subgroup_metrics = {
        "platform": {k: _finalize_metrics(v) for k, v in sorted(platform_buckets.items())},
        "element_type": {k: _finalize_metrics(v) for k, v in sorted(element_type_buckets.items())},
        "data_source": {k: _finalize_metrics(v) for k, v in sorted(data_source_buckets.items())},
    }

    summary = {
        "dataset_name": "screenspot_v2",
        "evaluation_name": args.evaluation_name,
        "evaluation_description": args.evaluation_description,
        "config_path": args.config,
        "dataset_source": args.hf_dataset_id,
        "dataset_split": args.split,
        "dataset_total_samples": dataset_total_samples,
        "evaluation_selection": range_description,
        "candidate_schema": candidate_schema,
        "model_adapter": args.model_adapter,
        "model_backbone": args.backbone,
        "evaluated_samples": evaluated_samples,
        "successful_runs": int(overall_bucket["run_success_count"]),
        "failed_runs": len(failures),
        "point_accuracy": overall_metrics["point_accuracy"],
        "iou@0.5": overall_metrics["iou@0.5"],
        "mean_iou": overall_metrics["mean_iou"],
        "action_type_valid_rate": overall_metrics["action_type_valid_rate"],
        "parseable_output_rate": overall_metrics["parseable_output_rate"],
        "valid_bbox_rate": overall_metrics["valid_bbox_rate"],
        "valid_click_point_rate": overall_metrics["valid_click_point_rate"],
        "action_type_distribution": dict(action_hist),
        "gt_bbox_clipped_count": dataset.gt_bbox_clipped_count,
        "runtime": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
        },
        "runtime_settings": {
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "coordinate_frame": args.coordinate_frame,
            "point_first_prompt": args.point_first_prompt,
            "web_mobile_hotspot_prompt": args.web_mobile_hotspot_prompt,
            "decoupled_point_native_decode": args.decoupled_point_native_decode,
            "edge_click_interior_threshold": args.edge_click_interior_threshold,
            "edge_click_interior_position": args.edge_click_interior_position,
            "model_adapter": args.model_adapter,
            "hf_endpoint": os.getenv("HF_ENDPOINT"),
        },
        "artifacts": {
            "predictions_jsonl": str(predictions_jsonl),
            "evaluation_summary_json": str(evaluation_summary_json),
            "subgroup_metrics_json": str(subgroup_metrics_json),
            "summary_table_md": str(summary_table_md),
            "failures_json": str(failures_json),
        },
    }

    save_json(summary, evaluation_summary_json)
    save_json(subgroup_metrics, subgroup_metrics_json)
    save_json(failures, failures_json)
    summary_table_md.write_text(_render_summary_table(summary, subgroup_metrics), encoding="utf-8")

    logger.info("Saved predictions: %s", predictions_jsonl)
    logger.info("Saved evaluation summary: %s", evaluation_summary_json)
    logger.info("Saved subgroup metrics: %s", subgroup_metrics_json)
    logger.info("Saved summary table: %s", summary_table_md)
    logger.info("Saved failures: %s", failures_json)
    logger.info(
        "Done. evaluated=%d success=%d failed=%d point_acc=%.4f iou@0.5=%.4f mean_iou=%.4f",
        evaluated_samples,
        summary["successful_runs"],
        summary["failed_runs"],
        summary["point_accuracy"],
        summary["iou@0.5"],
        summary["mean_iou"],
    )


if __name__ == "__main__":
    main()
