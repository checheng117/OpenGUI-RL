#!/usr/bin/env python3
"""Merge ScreenSpot-v2 evaluation shard predictions into one summary."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ScreenSpot-v2 eval shard predictions")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction JSONL files to merge")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/screenspot_v2_eval_qwen2_5_vl_3b",
    )
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--dataset-source", type=str, default="lscpku/ScreenSpot-v2")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--model-backbone", type=str, default="qwen2_5_vl_3b")
    return parser.parse_args()


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


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
    bucket["run_success_count"] += int(record.get("status") == "ok")
    bucket["valid_bbox_count"] += int(record.get("bbox_proposal") is not None)
    bucket["valid_click_point_count"] += int(record.get("click_point") is not None)
    bucket["valid_action_type_count"] += int(bool(record.get("action_type_valid")))
    bucket["parseable_output_count"] += int(bool(record.get("json_parse_success")))
    bucket["point_accuracy_hits"] += int(bool(record.get("point_in_box")))
    bucket["iou_at_0_5_hits"] += int(bool(record.get("iou_at_0_5")))
    bucket["iou_sum"] += float(record.get("iou") or 0.0)


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


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_dataset_index(record: dict) -> int | None:
    dataset_index = record.get("dataset_index")
    if isinstance(dataset_index, int):
        return dataset_index

    sample_id = str(record.get("sample_id") or "")
    match = re.search(r"_(\d{5})_", sample_id)
    if not match:
        return None
    return int(match.group(1))


def _record_sort_key(record: dict) -> tuple[int, str]:
    dataset_index = _infer_dataset_index(record)
    if dataset_index is not None:
        return (dataset_index, str(record.get("sample_id", "")))
    return (10**9, str(record.get("sample_id", "")))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_predictions_path = output_dir / "predictions.jsonl"
    evaluation_summary_path = output_dir / "evaluation_summary.json"
    subgroup_metrics_path = output_dir / "subgroup_metrics.json"
    summary_table_path = output_dir / "summary_table.md"
    merge_manifest_path = output_dir / "merge_manifest.json"
    failures_path = output_dir / "failures.json"

    all_records: list[dict] = []
    seen_sample_ids: set[str] = set()
    shard_summaries: list[dict] = []

    for pred_path_str in args.predictions:
        pred_path = Path(pred_path_str)
        records = _load_jsonl(pred_path)
        shard_summary = _load_json(pred_path.with_name("evaluation_summary.json"))
        if shard_summary is not None:
            shard_summaries.append(shard_summary)
        for record in records:
            inferred_dataset_index = _infer_dataset_index(record)
            if inferred_dataset_index is not None:
                record["dataset_index"] = inferred_dataset_index
            sample_id = str(record.get("sample_id", ""))
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id during merge: {sample_id}")
            seen_sample_ids.add(sample_id)
            all_records.append(record)

    all_records.sort(key=_record_sort_key)

    if args.expected_samples is not None and len(all_records) != args.expected_samples:
        raise ValueError(
            f"Merged sample count mismatch: expected {args.expected_samples}, got {len(all_records)}"
        )

    action_hist = Counter()
    overall_bucket = _empty_metrics_dict()
    platform_buckets = defaultdict(_empty_metrics_dict)
    element_type_buckets = defaultdict(_empty_metrics_dict)
    data_source_buckets = defaultdict(_empty_metrics_dict)

    for record in all_records:
        action_hist[str(record.get("action_type") or "__invalid_or_missing__")] += 1
        _update_metrics_bucket(overall_bucket, record)
        _update_metrics_bucket(platform_buckets[str(record.get("platform") or "__unknown__")], record)
        _update_metrics_bucket(
            element_type_buckets[str(record.get("element_type") or "__unknown__")],
            record,
        )
        _update_metrics_bucket(data_source_buckets[str(record.get("data_source") or "__unknown__")], record)

    overall_metrics = _finalize_metrics(overall_bucket)
    subgroup_metrics = {
        "platform": {k: _finalize_metrics(v) for k, v in sorted(platform_buckets.items())},
        "element_type": {k: _finalize_metrics(v) for k, v in sorted(element_type_buckets.items())},
        "data_source": {k: _finalize_metrics(v) for k, v in sorted(data_source_buckets.items())},
    }

    candidate_schema = all_records[0].get("candidate_schema") if all_records else {}
    reference_summary = shard_summaries[0] if shard_summaries else {}
    dataset_indices = [
        dataset_index
        for record in all_records
        if (dataset_index := _infer_dataset_index(record)) is not None
    ]

    summary = {
        "dataset_name": "screenspot_v2",
        "evaluation_name": reference_summary.get("evaluation_name"),
        "evaluation_description": reference_summary.get("evaluation_description"),
        "config_path": reference_summary.get("config_path"),
        "dataset_source": args.dataset_source,
        "dataset_split": args.dataset_split,
        "dataset_total_samples": reference_summary.get("dataset_total_samples"),
        "evaluation_selection": {
            "mode": "merged_shards",
            "expected_samples": args.expected_samples,
            "dataset_index_coverage": {
                "min": min(dataset_indices) if dataset_indices else None,
                "max": max(dataset_indices) if dataset_indices else None,
            },
        },
        "candidate_schema": candidate_schema,
        "model_backbone": args.model_backbone,
        "evaluated_samples": len(all_records),
        "successful_runs": int(overall_bucket["run_success_count"]),
        "failed_runs": len(all_records) - int(overall_bucket["run_success_count"]),
        "point_accuracy": overall_metrics["point_accuracy"],
        "iou@0.5": overall_metrics["iou@0.5"],
        "mean_iou": overall_metrics["mean_iou"],
        "action_type_valid_rate": overall_metrics["action_type_valid_rate"],
        "parseable_output_rate": overall_metrics["parseable_output_rate"],
        "valid_bbox_rate": overall_metrics["valid_bbox_rate"],
        "valid_click_point_rate": overall_metrics["valid_click_point_rate"],
        "action_type_distribution": dict(action_hist),
        "merged_prediction_inputs": args.predictions,
        "merged_shard_summaries": [
            str(Path(path).with_name("evaluation_summary.json")) for path in args.predictions
        ],
        "gt_bbox_clipped_count": reference_summary.get("gt_bbox_clipped_count"),
        "runtime_settings": reference_summary.get("runtime_settings"),
        "runtime": {
            "merged_duration_seconds": sum(
                float(summary.get("runtime", {}).get("duration_seconds") or 0.0)
                for summary in shard_summaries
            ),
            "merged_from_shards": len(shard_summaries),
        },
        "dataset_index_coverage": {
            "min": min(dataset_indices) if dataset_indices else None,
            "max": max(dataset_indices) if dataset_indices else None,
        },
        "merge_time_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "predictions_jsonl": str(merged_predictions_path),
            "evaluation_summary_json": str(evaluation_summary_path),
            "subgroup_metrics_json": str(subgroup_metrics_path),
            "summary_table_md": str(summary_table_path),
            "merge_manifest_json": str(merge_manifest_path),
            "failures_json": str(failures_path),
        },
    }

    with open(merged_predictions_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    save_json(summary, evaluation_summary_path)
    save_json(subgroup_metrics, subgroup_metrics_path)
    save_json(
        {
            "prediction_inputs": args.predictions,
            "merged_sample_count": len(all_records),
            "expected_samples": args.expected_samples,
        },
        merge_manifest_path,
    )
    save_json(
        [record for record in all_records if record.get("status") != "ok"],
        failures_path,
    )
    summary_table_path.write_text(_render_summary_table(summary, subgroup_metrics), encoding="utf-8")


if __name__ == "__main__":
    main()
