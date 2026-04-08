#!/usr/bin/env python3
"""Build dual-path candidates from saved runs and evaluate a lightweight verifier."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.constants import ACTION_TYPES
from gui_grounding.reward.lightweight_verifier import (
    DEFAULT_LIGHTWEIGHT_VERIFIER_CONFIG,
    DUAL_PATH_CANDIDATE_SCHEMA,
    LIGHTWEIGHT_VERIFIER_SCHEMA,
    build_dual_path_candidates,
    score_dual_path_candidates,
)
from gui_grounding.reward.verifiable_reward import bbox_iou
from gui_grounding.utils import load_config
from gui_grounding.utils.io import save_json
from gui_grounding.utils.logger import get_logger

logger = get_logger("run_eval_dual_path_verifier")

SELECTED_PREDICTION_SCHEMA = {
    "version": "bbox_click_action_v3_dual_path_verifier_selected",
    "primary_fields": ["bbox_proposal", "click_point", "action_type"],
    "candidate_pool_schema": DUAL_PATH_CANDIDATE_SCHEMA["version"],
    "verifier_schema": LIGHTWEIGHT_VERIFIER_SCHEMA["version"],
    "candidate_sources": DUAL_PATH_CANDIDATE_SCHEMA["candidate_sources"],
}
OVERALL_METRICS = [
    "evaluated_samples",
    "point_accuracy",
    "iou@0.5",
    "mean_iou",
    "action_type_valid_rate",
    "parseable_output_rate",
    "valid_bbox_rate",
    "valid_click_point_rate",
]
GROUP_METRICS = [
    "count",
    "point_accuracy",
    "iou@0.5",
    "mean_iou",
    "action_type_valid_rate",
    "parseable_output_rate",
    "valid_bbox_rate",
    "valid_click_point_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a dual-path lightweight verifier on ScreenSpot-v2")
    parser.add_argument("--config", type=str, required=True, help="YAML config for the dual-path verifier run.")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def _normalize_action_type(value: Any) -> str | None:
    if value is None:
        return None
    action = str(value).strip().lower()
    if action in ACTION_TYPES:
        return action
    return None


def _point_inside_bbox(click_point, bbox) -> bool:
    if click_point is None or bbox is None:
        return False
    click_x, click_y = click_point
    x1, y1, x2, y2 = bbox
    return x1 <= click_x <= x2 and y1 <= click_y <= y2


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


def _update_metrics_bucket(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    bucket["count"] += 1
    bucket["run_success_count"] += int(record["status"] == "ok")
    bucket["valid_bbox_count"] += int(record["bbox_proposal"] is not None)
    bucket["valid_click_point_count"] += int(record["click_point"] is not None)
    bucket["valid_action_type_count"] += int(record["action_type_valid"])
    bucket["parseable_output_count"] += int(record["json_parse_success"])
    bucket["point_accuracy_hits"] += int(record["point_in_box"])
    bucket["iou_at_0_5_hits"] += int(record["iou_at_0_5"])
    bucket["iou_sum"] += float(record["iou"])


def _finalize_metrics(bucket: dict[str, Any]) -> dict[str, int | float]:
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


def _render_summary_table(overall: dict[str, Any], subgroup_metrics: dict[str, Any]) -> str:
    lines = [
        "# ScreenSpot-v2 Dual-Path Verifier Summary",
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
        "## Verifier",
        "",
        f"- Oracle best-of-two point accuracy: `{overall['oracle_best_of_two_point_accuracy']:.4f}`",
        f"- Gain vs point-native: `{overall['gain_vs_point_native']:+.4f}`",
        f"- Gain vs structured: `{overall['gain_vs_structured']:+.4f}`",
        "",
    ]

    for source_path, count in sorted(overall["verifier_selection_distribution"].items()):
        lines.append(f"- Selected `{source_path}`: `{count}`")
    lines.append("")

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


def _compute_metric_triplet(before: dict[str, Any], after: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    triplet: dict[str, Any] = {}
    for key in keys:
        before_value = before.get(key)
        after_value = after.get(key)
        delta = None
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            delta = after_value - before_value
        triplet[key] = {
            "before": before_value,
            "after": after_value,
            "delta": delta,
        }
    return triplet


def _compare_group_section(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    compared: dict[str, Any] = {}
    for group_name in sorted(set(before) | set(after)):
        compared[group_name] = _compute_metric_triplet(
            before.get(group_name, {}),
            after.get(group_name, {}),
            GROUP_METRICS,
        )
    return compared


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_pairwise_comparison_md(comparison: dict[str, Any], before_label: str, after_label: str) -> str:
    lines = [
        "# ScreenSpot-v2 Comparison",
        "",
        f"- Before: `{before_label}`",
        f"- After: `{after_label}`",
        "",
        "## Overall",
        "",
        f"| Metric | {before_label} | {after_label} | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in OVERALL_METRICS:
        triplet = comparison["overall"][metric]
        lines.append(
            f"| {metric} | {_fmt(triplet['before'])} | {_fmt(triplet['after'])} | {_fmt(triplet['delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Platform",
            "",
            f"| Group | Metric | {before_label} | {after_label} | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for group_name, metrics in comparison["platform"].items():
        for metric in GROUP_METRICS:
            triplet = metrics[metric]
            lines.append(
                f"| {group_name} | {metric} | {_fmt(triplet['before'])} | {_fmt(triplet['after'])} | {_fmt(triplet['delta'])} |"
            )
    return "\n".join(lines)


def _write_pairwise_comparison(
    *,
    before_summary: dict[str, Any],
    before_subgroups: dict[str, Any],
    after_summary: dict[str, Any],
    after_subgroups: dict[str, Any],
    before_label: str,
    after_label: str,
    output_json: Path,
    output_md: Path,
) -> None:
    comparison = {
        "labels": {
            "before": before_label,
            "after": after_label,
        },
        "overall": _compute_metric_triplet(before_summary, after_summary, OVERALL_METRICS),
        "platform": _compare_group_section(
            before_subgroups.get("platform", {}),
            after_subgroups.get("platform", {}),
        ),
        "element_type": _compare_group_section(
            before_subgroups.get("element_type", {}),
            after_subgroups.get("element_type", {}),
        ),
        "data_source": _compare_group_section(
            before_subgroups.get("data_source", {}),
            after_subgroups.get("data_source", {}),
        ),
    }
    save_json(comparison, output_json)
    output_md.write_text(
        _render_pairwise_comparison_md(comparison, before_label=before_label, after_label=after_label),
        encoding="utf-8",
    )


def _build_core_method_comparison(
    method_rows: dict[str, dict[str, Any]],
    platform_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    comparison = {
        "methods": method_rows,
        "platform": platform_rows,
    }
    lines = [
        "# ScreenSpot-v2 Core Method Comparison",
        "",
        "## Overall",
        "",
        "| Method | Point Acc | Desktop | Web | Mobile | IoU@0.5 | Mean IoU | Action Valid | Parseable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method_name, row in method_rows.items():
        lines.append(
            f"| {method_name} | {row['point_accuracy']:.4f} | {row['desktop_point_accuracy']:.4f} | "
            f"{row['web_point_accuracy']:.4f} | {row['mobile_point_accuracy']:.4f} | "
            f"{row['iou@0.5']:.4f} | {row['mean_iou']:.4f} | {row['action_type_valid_rate']:.4f} | "
            f"{row['parseable_output_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Verifier Selection",
            "",
            "| Platform | point_native_primary | structured_single_pass | hybrid_point_structured |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for platform_name, counts in platform_rows.items():
        lines.append(
            f"| {platform_name} | {counts.get('point_native_primary', 0)} | "
            f"{counts.get('structured_single_pass', 0)} | {counts.get('hybrid_point_structured', 0)} |"
        )
    return comparison, "\n".join(lines)


def _resolve_args(cfg_path: str) -> argparse.Namespace:
    cfg = load_config(cfg_path)
    evaluation_cfg = cfg.get("evaluation", {})
    input_cfg = cfg.get("inputs", {})
    output_cfg = cfg.get("output", {})
    verifier_cfg = cfg.get("verifier", {})
    runtime_cfg = cfg.get("runtime", {})

    return argparse.Namespace(
        config=cfg_path,
        evaluation_name=evaluation_cfg.get("name", "screenspot_v2_dual_path_verifier"),
        evaluation_description=evaluation_cfg.get("description"),
        point_native_predictions=input_cfg.get("point_native_predictions"),
        structured_predictions=input_cfg.get("structured_predictions"),
        point_native_summary=input_cfg.get("point_native_summary"),
        point_native_subgroups=input_cfg.get("point_native_subgroups"),
        structured_summary=input_cfg.get("structured_summary"),
        structured_subgroups=input_cfg.get("structured_subgroups"),
        public_baseline_summary=input_cfg.get("public_baseline_summary"),
        public_baseline_subgroups=input_cfg.get("public_baseline_subgroups"),
        output_dir=output_cfg.get("output_dir", "outputs/screenspot_v2_eval_qwen2_5_vl_3b_dual_path_verifier"),
        verifier_config={**DEFAULT_LIGHTWEIGHT_VERIFIER_CONFIG, **dict(verifier_cfg)},
        log_every=int(runtime_cfg.get("log_every", 100)),
    )


def _validate_input_records(
    point_records: list[dict[str, Any]],
    structured_records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    point_by_id = {row["sample_id"]: row for row in point_records}
    structured_by_id = {row["sample_id"]: row for row in structured_records}
    if set(point_by_id) != set(structured_by_id):
        missing_from_structured = sorted(set(point_by_id) - set(structured_by_id))
        missing_from_point = sorted(set(structured_by_id) - set(point_by_id))
        raise ValueError(
            "Point-native and structured records do not align. "
            f"missing_from_structured={missing_from_structured[:5]} "
            f"missing_from_point={missing_from_point[:5]}"
        )
    ordered_pairs = []
    for record in point_records:
        structured = structured_by_id[record["sample_id"]]
        if record.get("instruction") != structured.get("instruction"):
            raise ValueError(f"Instruction mismatch for sample_id={record['sample_id']}")
        ordered_pairs.append((record, structured))
    return ordered_pairs


def _build_selected_record(
    *,
    point_record: dict[str, Any],
    structured_record: dict[str, Any],
    verifier_result: dict[str, Any],
) -> dict[str, Any]:
    selected_candidate = verifier_result["selected_candidate"]
    bbox_proposal = selected_candidate.get("bbox_proposal")
    click_point = selected_candidate.get("click_point")
    action_type = _normalize_action_type(selected_candidate.get("action_type"))
    gt_bbox = point_record.get("target_bbox_xyxy")
    iou = bbox_iou(tuple(bbox_proposal), tuple(gt_bbox)) if bbox_proposal is not None and gt_bbox is not None else 0.0
    point_in_box = _point_inside_bbox(click_point, gt_bbox)
    parsed_payload = {
        "_dual_path_lightweight_verifier": True,
        "_selected_candidate_id": selected_candidate["candidate_id"],
        "_selected_source_path": selected_candidate["source_path"],
        "_candidate_pool_schema": DUAL_PATH_CANDIDATE_SCHEMA["version"],
        "_verifier_schema": LIGHTWEIGHT_VERIFIER_SCHEMA["version"],
        "_verifier_total_score": selected_candidate["verifier_total_score"],
        "_verifier_score_components": selected_candidate["verifier_score_components"],
        "_verifier_features": selected_candidate["verifier_features"],
        "_verifier_cross_path_context": verifier_result["cross_path_context"],
        "_point_native_source_sample_id": point_record["sample_id"],
        "_structured_source_sample_id": structured_record["sample_id"],
        "_point_native_candidate_schema": point_record.get("candidate_schema"),
        "_structured_candidate_schema": structured_record.get("candidate_schema"),
        "_click_provenance": selected_candidate.get("click_provenance"),
        "_bbox_provenance": selected_candidate.get("bbox_provenance"),
        "_action_provenance": selected_candidate.get("action_provenance"),
        "action_type": action_type,
        "predicted_element_id": selected_candidate.get("element_hint_id"),
        "predicted_bbox": bbox_proposal,
        "predicted_click_point": click_point,
        "confidence": selected_candidate.get("confidence"),
    }
    raw_model_response = (
        "[point_native_source]\n"
        f"{point_record.get('raw_model_response', '')}\n\n"
        "[structured_source]\n"
        f"{structured_record.get('raw_model_response', '')}\n\n"
        "[verifier_selection]\n"
        f"{json.dumps({'selected_candidate_id': selected_candidate['candidate_id'], 'selected_source_path': selected_candidate['source_path']}, ensure_ascii=False)}"
    )

    return {
        "dataset_index": point_record.get("dataset_index"),
        "sample_id": point_record["sample_id"],
        "dataset_name": point_record.get("dataset_name"),
        "hf_dataset_id": point_record.get("hf_dataset_id"),
        "split": point_record.get("split"),
        "instruction": point_record.get("instruction"),
        "image_path": point_record.get("image_path"),
        "platform": point_record.get("platform"),
        "element_type": point_record.get("element_type"),
        "data_source": point_record.get("data_source"),
        "target_bbox_xyxy": gt_bbox,
        "target_click_point": point_record.get("target_click_point"),
        "target_action_type": point_record.get("target_action_type"),
        "candidate_semantics": "dual_path_candidate_selection_bbox_proposal_click_point_action_type",
        "candidate_schema": SELECTED_PREDICTION_SCHEMA,
        "status": "ok",
        "bbox_proposal": bbox_proposal,
        "click_point": click_point,
        "action_type": action_type,
        "action_type_valid": action_type is not None,
        "element_hint_id": selected_candidate.get("element_hint_id"),
        "confidence": selected_candidate.get("confidence"),
        "raw_model_response": raw_model_response,
        "parsed_model_payload": parsed_payload,
        "raw_response_nonempty": bool(point_record.get("raw_response_nonempty")) or bool(structured_record.get("raw_response_nonempty")),
        "json_parse_success": bool((selected_candidate.get("parser_metadata") or {}).get("json_parse_success")),
        "iou": float(iou),
        "iou_at_0_5": bool(iou >= 0.5),
        "point_in_box": bool(point_in_box),
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_source_path": selected_candidate["source_path"],
        "verifier_total_score": selected_candidate["verifier_total_score"],
        "verifier_score_components": selected_candidate["verifier_score_components"],
        "verifier_cross_path_context": verifier_result["cross_path_context"],
    }


def main() -> None:
    args = _resolve_args(parse_args().config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_jsonl = output_dir / "candidate_artifacts.jsonl"
    verifier_outputs_jsonl = output_dir / "verifier_outputs.jsonl"
    predictions_jsonl = output_dir / "predictions.jsonl"
    evaluation_summary_json = output_dir / "evaluation_summary.json"
    subgroup_metrics_json = output_dir / "subgroup_metrics.json"
    summary_table_md = output_dir / "summary_table.md"
    verifier_summary_json = output_dir / "verifier_selection_summary.json"
    verifier_summary_md = output_dir / "verifier_selection_summary.md"
    core_method_comparison_json = output_dir / "core_method_comparison.json"
    core_method_comparison_md = output_dir / "core_method_comparison.md"

    for path in (candidates_jsonl, verifier_outputs_jsonl, predictions_jsonl):
        path.write_text("", encoding="utf-8")

    logger.info("Loading same-protocol source artifacts.")
    point_records = _load_jsonl(args.point_native_predictions)
    structured_records = _load_jsonl(args.structured_predictions)
    pairs = _validate_input_records(point_records, structured_records)
    logger.info("Loaded %d aligned samples.", len(pairs))

    point_summary = _load_json(args.point_native_summary)
    point_subgroups = _load_json(args.point_native_subgroups)
    structured_summary = _load_json(args.structured_summary)
    structured_subgroups = _load_json(args.structured_subgroups)
    public_summary = _load_json(args.public_baseline_summary)
    public_subgroups = _load_json(args.public_baseline_subgroups)

    overall_bucket = _empty_metrics_dict()
    platform_buckets = defaultdict(_empty_metrics_dict)
    element_type_buckets = defaultdict(_empty_metrics_dict)
    data_source_buckets = defaultdict(_empty_metrics_dict)
    selection_hist = Counter()
    selection_by_platform = defaultdict(Counter)
    action_hist = Counter()

    oracle_best_of_two_hits = 0
    started_at = datetime.now(timezone.utc)

    for idx, (point_record, structured_record) in enumerate(pairs, start=1):
        candidates = build_dual_path_candidates(
            point_record,
            structured_record,
            point_artifact_label=args.point_native_predictions,
            structured_artifact_label=args.structured_predictions,
        )
        verifier_result = score_dual_path_candidates(candidates, verifier_config=args.verifier_config)
        selected_record = _build_selected_record(
            point_record=point_record,
            structured_record=structured_record,
            verifier_result=verifier_result,
        )

        sample_row = {
            "dataset_index": point_record.get("dataset_index"),
            "sample_id": point_record["sample_id"],
            "instruction": point_record.get("instruction"),
            "image_path": point_record.get("image_path"),
            "platform": point_record.get("platform"),
            "element_type": point_record.get("element_type"),
            "data_source": point_record.get("data_source"),
            "target_bbox_xyxy": point_record.get("target_bbox_xyxy"),
            "target_click_point": point_record.get("target_click_point"),
            "candidate_schema": DUAL_PATH_CANDIDATE_SCHEMA,
            "verifier_schema": LIGHTWEIGHT_VERIFIER_SCHEMA,
            "candidates": candidates,
        }
        verifier_row = {
            "sample_id": point_record["sample_id"],
            "selected_candidate_id": verifier_result["selected_candidate_id"],
            "selected_source_path": verifier_result["selected_source_path"],
            "cross_path_context": verifier_result["cross_path_context"],
            "candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "source_path": row["source_path"],
                    "verifier_total_score": row["verifier_total_score"],
                    "verifier_score_components": row["verifier_score_components"],
                    "verifier_features": row["verifier_features"],
                }
                for row in verifier_result["candidates"]
            ],
            "oracle_best_point_source": (
                "point_native_primary"
                if point_record.get("point_in_box") and not structured_record.get("point_in_box")
                else (
                    "structured_single_pass"
                    if structured_record.get("point_in_box") and not point_record.get("point_in_box")
                    else "tie_or_miss"
                )
            ),
        }

        _append_jsonl(candidates_jsonl, sample_row)
        _append_jsonl(verifier_outputs_jsonl, verifier_row)
        _append_jsonl(predictions_jsonl, selected_record)

        _update_metrics_bucket(overall_bucket, selected_record)
        _update_metrics_bucket(platform_buckets[str(selected_record.get("platform") or "__unknown__")], selected_record)
        _update_metrics_bucket(element_type_buckets[str(selected_record.get("element_type") or "__unknown__")], selected_record)
        _update_metrics_bucket(data_source_buckets[str(selected_record.get("data_source") or "__unknown__")], selected_record)

        selected_source = str(selected_record["selected_source_path"])
        selection_hist[selected_source] += 1
        selection_by_platform[str(selected_record.get("platform") or "__unknown__")][selected_source] += 1
        action_hist[selected_record["action_type"] or "__invalid_or_missing__"] += 1
        oracle_best_of_two_hits += int(bool(point_record.get("point_in_box")) or bool(structured_record.get("point_in_box")))

        if idx % max(args.log_every, 1) == 0:
            logger.info("Processed %d / %d samples", idx, len(pairs))

    finished_at = datetime.now(timezone.utc)
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
        "candidate_schema": SELECTED_PREDICTION_SCHEMA,
        "candidate_pool_schema": DUAL_PATH_CANDIDATE_SCHEMA,
        "verifier_schema": LIGHTWEIGHT_VERIFIER_SCHEMA,
        "input_artifacts": {
            "point_native_predictions": args.point_native_predictions,
            "structured_predictions": args.structured_predictions,
            "point_native_summary": args.point_native_summary,
            "structured_summary": args.structured_summary,
            "public_baseline_summary": args.public_baseline_summary,
        },
        "evaluated_samples": len(pairs),
        "successful_runs": len(pairs),
        "failed_runs": 0,
        "point_accuracy": overall_metrics["point_accuracy"],
        "iou@0.5": overall_metrics["iou@0.5"],
        "mean_iou": overall_metrics["mean_iou"],
        "action_type_valid_rate": overall_metrics["action_type_valid_rate"],
        "parseable_output_rate": overall_metrics["parseable_output_rate"],
        "valid_bbox_rate": overall_metrics["valid_bbox_rate"],
        "valid_click_point_rate": overall_metrics["valid_click_point_rate"],
        "action_type_distribution": dict(action_hist),
        "verifier_selection_distribution": dict(selection_hist),
        "oracle_best_of_two_point_accuracy": _safe_rate(oracle_best_of_two_hits, len(pairs)),
        "gain_vs_point_native": overall_metrics["point_accuracy"] - float(point_summary["point_accuracy"]),
        "gain_vs_structured": overall_metrics["point_accuracy"] - float(structured_summary["point_accuracy"]),
        "gain_vs_public_baseline": overall_metrics["point_accuracy"] - float(public_summary["point_accuracy"]),
        "runtime": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
        },
        "runtime_settings": {
            "verifier_config": args.verifier_config,
        },
        "artifacts": {
            "candidate_artifacts_jsonl": str(candidates_jsonl),
            "verifier_outputs_jsonl": str(verifier_outputs_jsonl),
            "predictions_jsonl": str(predictions_jsonl),
            "evaluation_summary_json": str(evaluation_summary_json),
            "subgroup_metrics_json": str(subgroup_metrics_json),
            "summary_table_md": str(summary_table_md),
            "verifier_summary_json": str(verifier_summary_json),
            "verifier_summary_md": str(verifier_summary_md),
            "core_method_comparison_json": str(core_method_comparison_json),
            "core_method_comparison_md": str(core_method_comparison_md),
        },
    }

    save_json(summary, evaluation_summary_json)
    save_json(subgroup_metrics, subgroup_metrics_json)
    summary_table_md.write_text(_render_summary_table(summary, subgroup_metrics), encoding="utf-8")

    verifier_summary = {
        "selection_distribution": dict(selection_hist),
        "selection_by_platform": {
            platform: dict(counter) for platform, counter in sorted(selection_by_platform.items())
        },
        "oracle_best_of_two_point_accuracy": summary["oracle_best_of_two_point_accuracy"],
        "actual_point_accuracy": summary["point_accuracy"],
        "gain_vs_point_native": summary["gain_vs_point_native"],
        "gain_vs_structured": summary["gain_vs_structured"],
        "gain_vs_public_baseline": summary["gain_vs_public_baseline"],
    }
    save_json(verifier_summary, verifier_summary_json)
    verifier_summary_md.write_text(
        "\n".join(
            [
                "# Dual-Path Verifier Selection Summary",
                "",
                f"- Oracle best-of-two point accuracy: `{summary['oracle_best_of_two_point_accuracy']:.4f}`",
                f"- Actual point accuracy: `{summary['point_accuracy']:.4f}`",
                f"- Gain vs point-native: `{summary['gain_vs_point_native']:+.4f}`",
                f"- Gain vs structured: `{summary['gain_vs_structured']:+.4f}`",
                f"- Gain vs public baseline: `{summary['gain_vs_public_baseline']:+.4f}`",
                "",
            ]
            + [f"- Selected `{source}`: `{count}`" for source, count in sorted(selection_hist.items())]
        ),
        encoding="utf-8",
    )

    method_rows = {
        "public_baseline": {
            **{k: float(public_summary[k]) for k in ["point_accuracy", "iou@0.5", "mean_iou", "action_type_valid_rate", "parseable_output_rate"]},
            "desktop_point_accuracy": float(public_subgroups["platform"]["desktop"]["point_accuracy"]),
            "web_point_accuracy": float(public_subgroups["platform"]["web"]["point_accuracy"]),
            "mobile_point_accuracy": float(public_subgroups["platform"]["mobile"]["point_accuracy"]),
        },
        "structured_only": {
            **{k: float(structured_summary[k]) for k in ["point_accuracy", "iou@0.5", "mean_iou", "action_type_valid_rate", "parseable_output_rate"]},
            "desktop_point_accuracy": float(structured_subgroups["platform"]["desktop"]["point_accuracy"]),
            "web_point_accuracy": float(structured_subgroups["platform"]["web"]["point_accuracy"]),
            "mobile_point_accuracy": float(structured_subgroups["platform"]["mobile"]["point_accuracy"]),
        },
        "point_native_only": {
            **{k: float(point_summary[k]) for k in ["point_accuracy", "iou@0.5", "mean_iou", "action_type_valid_rate", "parseable_output_rate"]},
            "desktop_point_accuracy": float(point_subgroups["platform"]["desktop"]["point_accuracy"]),
            "web_point_accuracy": float(point_subgroups["platform"]["web"]["point_accuracy"]),
            "mobile_point_accuracy": float(point_subgroups["platform"]["mobile"]["point_accuracy"]),
        },
        "dual_path_verifier": {
            **{k: float(summary[k]) for k in ["point_accuracy", "iou@0.5", "mean_iou", "action_type_valid_rate", "parseable_output_rate"]},
            "desktop_point_accuracy": float(subgroup_metrics["platform"]["desktop"]["point_accuracy"]),
            "web_point_accuracy": float(subgroup_metrics["platform"]["web"]["point_accuracy"]),
            "mobile_point_accuracy": float(subgroup_metrics["platform"]["mobile"]["point_accuracy"]),
        },
    }
    core_comparison, core_md = _build_core_method_comparison(
        method_rows=method_rows,
        platform_rows={platform: dict(counter) for platform, counter in sorted(selection_by_platform.items())},
    )
    save_json(core_comparison, core_method_comparison_json)
    core_method_comparison_md.write_text(core_md, encoding="utf-8")

    _write_pairwise_comparison(
        before_summary=public_summary,
        before_subgroups=public_subgroups,
        after_summary=summary,
        after_subgroups=subgroup_metrics,
        before_label="public_baseline",
        after_label="dual_path_verifier",
        output_json=output_dir / "comparison_vs_public_baseline.json",
        output_md=output_dir / "comparison_vs_public_baseline.md",
    )
    _write_pairwise_comparison(
        before_summary=point_summary,
        before_subgroups=point_subgroups,
        after_summary=summary,
        after_subgroups=subgroup_metrics,
        before_label="point_native_only",
        after_label="dual_path_verifier",
        output_json=output_dir / "comparison_vs_point_native.json",
        output_md=output_dir / "comparison_vs_point_native.md",
    )
    _write_pairwise_comparison(
        before_summary=structured_summary,
        before_subgroups=structured_subgroups,
        after_summary=summary,
        after_subgroups=subgroup_metrics,
        before_label="structured_only",
        after_label="dual_path_verifier",
        output_json=output_dir / "comparison_vs_structured.json",
        output_md=output_dir / "comparison_vs_structured.md",
    )

    logger.info(
        "Done. point_acc=%.4f iou@0.5=%.4f mean_iou=%.4f gain_vs_point_native=%+.4f",
        summary["point_accuracy"],
        summary["iou@0.5"],
        summary["mean_iou"],
        summary["gain_vs_point_native"],
    )


if __name__ == "__main__":
    main()
