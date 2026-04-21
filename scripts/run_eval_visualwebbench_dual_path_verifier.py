#!/usr/bin/env python3
"""Evaluate the dual-path lightweight verifier on VisualWebBench grounding."""

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

from gui_grounding.evaluation.visualwebbench_metrics import (
    aggregate_visualwebbench_records,
    render_visualwebbench_summary_table,
    score_visualwebbench_grounding,
)
from gui_grounding.reward.lightweight_verifier import (
    DEFAULT_LIGHTWEIGHT_VERIFIER_CONFIG,
    DUAL_PATH_CANDIDATE_SCHEMA,
    LIGHTWEIGHT_VERIFIER_SCHEMA,
    build_dual_path_candidates,
    score_dual_path_candidates,
)
from gui_grounding.utils import load_config
from gui_grounding.utils.io import save_json
from gui_grounding.utils.logger import get_logger

logger = get_logger("run_eval_visualwebbench_dual_path_verifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisualWebBench dual-path lightweight verifier")
    parser.add_argument("--config", type=str, required=True)
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    inputs_cfg = cfg.get("inputs", {})
    verifier_cfg = {**DEFAULT_LIGHTWEIGHT_VERIFIER_CONFIG, **dict(cfg.get("verifier", {}))}
    output_cfg = cfg.get("output", {})

    output_dir = Path(output_cfg.get("output_dir", "outputs/visualwebbench_dual_path_verifier"))
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_jsonl = output_dir / "predictions.jsonl"
    evaluation_summary_json = output_dir / "evaluation_summary.json"
    subgroup_metrics_json = output_dir / "subgroup_metrics.json"
    summary_table_md = output_dir / "summary_table.md"
    predictions_jsonl.write_text("", encoding="utf-8")

    point_records = {
        row["sample_id"]: row
        for row in _load_jsonl(inputs_cfg["point_native_predictions"])
    }
    structured_records = {
        row["sample_id"]: row
        for row in _load_jsonl(inputs_cfg["structured_predictions"])
    }
    sample_ids = sorted(set(point_records) & set(structured_records))
    if not sample_ids:
        raise RuntimeError("No overlapping sample IDs between point-native and structured predictions.")

    point_summary = _load_json(inputs_cfg["point_native_summary"])
    structured_summary = _load_json(inputs_cfg["structured_summary"])

    records: list[dict[str, Any]] = []
    selection_distribution: Counter[str] = Counter()
    selection_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    oracle_choice_hits = 0
    oracle_point_hits = 0

    for sample_id in sample_ids:
        point_record = point_records[sample_id]
        structured_record = structured_records[sample_id]

        candidates = build_dual_path_candidates(
            point_record,
            structured_record,
            point_artifact_label=Path(inputs_cfg["point_native_predictions"]).name,
            structured_artifact_label=Path(inputs_cfg["structured_predictions"]).name,
        )
        verifier_output = score_dual_path_candidates(candidates, verifier_cfg)
        selected = verifier_output["selected_candidate"]

        sample_task_type = point_record.get("task_type")
        selection_distribution[selected["source_path"]] += 1
        selection_by_task[str(sample_task_type)][selected["source_path"]] += 1

        oracle_choice_hits += int(
            bool(point_record.get("official_choice_correct")) or bool(structured_record.get("official_choice_correct"))
        )
        oracle_point_hits += int(
            bool(point_record.get("point_in_box")) or bool(structured_record.get("point_in_box"))
        )

        score_fields = score_visualwebbench_grounding(
            predicted_bbox=selected.get("bbox_proposal"),
            predicted_click_point=selected.get("click_point"),
            predicted_action_type=selected.get("action_type"),
            candidate_boxes=point_record.get("candidate_options_xyxy") or structured_record.get("candidate_options_xyxy") or [],
            target_choice_index=point_record.get("target_choice_index"),
            image_size=point_record.get("image_size"),
            task_type=sample_task_type,
            website=point_record.get("website"),
            predicted_candidate_slot=point_record.get("predicted_candidate_slot"),
            candidate_slot_grounded=False,
        )
        record = {
            "sample_id": sample_id,
            "dataset_name": point_record.get("dataset_name"),
            "split": point_record.get("split"),
            "dataset_index": point_record.get("dataset_index"),
            "task_type": sample_task_type,
            "website": point_record.get("website"),
            "platform": point_record.get("platform"),
            "image_path": point_record.get("image_path"),
            "annotated_image_path": point_record.get("annotated_image_path"),
            "image_size": point_record.get("image_size"),
            "instruction": point_record.get("instruction"),
            "candidate_options_xyxy": point_record.get("candidate_options_xyxy") or [],
            "status": "ok",
            "raw_model_response": None,
            "raw_response_nonempty": True,
            "json_parse_success": bool(
                selected.get("parser_metadata", {}).get("json_parse_success")
            ),
            "parsed_model_payload": {
                "verifier_output": verifier_output,
                "selected_source_path": selected.get("source_path"),
            },
            "candidate_schema": {
                "selected_prediction_schema": {
                    "version": "bbox_click_action_v1_visualwebbench_dual_path_selected",
                    "primary_fields": ["bbox_proposal", "click_point", "action_type"],
                    "candidate_pool_schema": DUAL_PATH_CANDIDATE_SCHEMA["version"],
                    "verifier_schema": LIGHTWEIGHT_VERIFIER_SCHEMA["version"],
                }
            },
            "candidate_semantics": "dual_path_verifier_selected_prediction",
            "bbox_proposal": selected.get("bbox_proposal"),
            "click_point": selected.get("click_point"),
            "action_type": selected.get("action_type"),
            "action_type_valid": bool(selected.get("action_type_valid")),
            "confidence": selected.get("confidence"),
            "element_hint_id": selected.get("element_hint_id"),
            "selected_source_path": selected.get("source_path"),
            "verifier_total_score": selected.get("verifier_total_score"),
            **score_fields,
        }
        records.append(record)
        _append_jsonl(predictions_jsonl, record)

    overall_metrics, subgroup_metrics = aggregate_visualwebbench_records(
        records,
        group_fields=["task_type", "target_area_bucket", "distractor_overlap_bucket"],
    )

    evaluation_summary = {
        "evaluation_name": cfg.get("evaluation", {}).get("name", "visualwebbench_dual_path_verifier"),
        "evaluation_description": cfg.get("evaluation", {}).get("description"),
        "dataset_name": "visualwebbench",
        "task_types": sorted({row.get("task_type") for row in records}),
        "evaluated_samples": overall_metrics["count"],
        "successful_runs": overall_metrics["count"],
        "failed_runs": 0,
        **overall_metrics,
        "oracle_best_of_two_choice_accuracy": oracle_choice_hits / max(len(records), 1),
        "oracle_best_of_two_point_accuracy": oracle_point_hits / max(len(records), 1),
        "gain_vs_point_native_choice_accuracy": (
            overall_metrics["official_choice_accuracy"] - float(point_summary["official_choice_accuracy"])
        ),
        "gain_vs_structured_choice_accuracy": (
            overall_metrics["official_choice_accuracy"] - float(structured_summary["official_choice_accuracy"])
        ),
        "gain_vs_point_native_point_accuracy": (
            overall_metrics["point_accuracy"] - float(point_summary["point_accuracy"])
        ),
        "gain_vs_structured_point_accuracy": (
            overall_metrics["point_accuracy"] - float(structured_summary["point_accuracy"])
        ),
        "verifier_selection_distribution": dict(selection_distribution),
        "verifier_selection_distribution_by_task": {
            task_type: dict(counter) for task_type, counter in selection_by_task.items()
        },
        "verifier_config_used": verifier_cfg,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_table_md.write_text(
        render_visualwebbench_summary_table(
            title="VisualWebBench Dual-Path Verifier Summary",
            overall=evaluation_summary,
            subgroup_metrics=subgroup_metrics,
        ),
        encoding="utf-8",
    )
    save_json(evaluation_summary, evaluation_summary_json)
    save_json(subgroup_metrics, subgroup_metrics_json)

    logger.info("Dual-path official choice accuracy: %.4f", evaluation_summary["official_choice_accuracy"])
    logger.info(
        "Dual-path gain vs point-native: %+0.4f",
        evaluation_summary["gain_vs_point_native_choice_accuracy"],
    )
    logger.info(
        "Dual-path gain vs structured: %+0.4f",
        evaluation_summary["gain_vs_structured_choice_accuracy"],
    )
    logger.info("Saved outputs to %s", output_dir)


if __name__ == "__main__":
    main()
