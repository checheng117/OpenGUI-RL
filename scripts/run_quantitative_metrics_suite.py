#!/usr/bin/env python3
"""Recompute the project's quantitative metrics from saved evaluation artifacts.

This script does not launch new model inference. Instead, it reruns the metric
computation layer over the repository's real prediction / candidate artifacts so
the reported numbers are derived from executable code rather than copied tables.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.evaluation.metrics import (
    compute_all_metrics,
    invalid_format_rate,
)
from gui_grounding.utils.io import save_json


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "quantitative_metrics_suite"


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_box(value: Any):
    if value is None:
        return None
    return tuple(float(x) for x in value)


def _as_point(value: Any):
    if value is None:
        return None
    return (float(value[0]), float(value[1]))


def _pct(value: float) -> float:
    return round(value * 100.0, 2)


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _sec_per_image(duration_seconds: float | None, evaluated_samples: int | None) -> float | None:
    if duration_seconds is None or not evaluated_samples:
        return None
    return duration_seconds / float(evaluated_samples)


def _recompute_mind2web_prediction_metrics(
    predictions_path: Path,
) -> dict[str, float]:
    rows = _load_jsonl(predictions_path)
    pred_bboxes = []
    gt_bboxes = []
    pred_points = []
    pred_actions = []
    gt_actions = []

    for row in rows:
        pred_bboxes.append(_as_box(row.get("predicted_bbox")))
        gt_bboxes.append(_as_box(row.get("target_bbox")))
        pred_points.append(_as_point(row.get("predicted_click_point")))
        pred_actions.append(row.get("predicted_action_type"))
        gt_actions.append(row.get("target_action_type"))

    metrics = compute_all_metrics(
        pred_element_ids=[None] * len(rows),
        gt_element_ids=[None] * len(rows),
        pred_bboxes=pred_bboxes,
        gt_bboxes=gt_bboxes,
        pred_points=pred_points,
        pred_actions=pred_actions,
        gt_actions=gt_actions,
    )
    metrics["invalid_format_rate"] = invalid_format_rate(
        pred_bboxes=pred_bboxes,
        pred_points=pred_points,
    )
    return metrics


def _mind2web_stagea_section(output_dir_name: str) -> dict[str, Any]:
    root = PROJECT_ROOT / "outputs" / output_dir_name
    summary = _load_json(root / "eval_summary.json")

    official = {}
    for split_name in ("test_task", "test_website", "test_domain"):
        recomputed = _recompute_mind2web_prediction_metrics(
            root / f"eval_predictions_{split_name}.jsonl",
        )
        summary_metrics = summary["official_cached_subset"][split_name]
        official[split_name] = {
            "element_accuracy": float(summary_metrics["element_accuracy"]),
            "click_point_accuracy": recomputed["point_accuracy"],
            "iou@0.5": recomputed["iou@0.5"],
            "action_type_accuracy": recomputed["action_type_accuracy"],
            "invalid_format_rate": recomputed["invalid_format_rate"],
            "num_samples": summary_metrics["num_samples"],
            "prediction_path": summary_metrics["prediction_path"],
        }

    official_avg = {
        "element_accuracy": _mean([official[s]["element_accuracy"] for s in official]),
        "click_point_accuracy": _mean([official[s]["click_point_accuracy"] for s in official]),
        "iou@0.5": _mean([official[s]["iou@0.5"] for s in official]),
        "action_type_accuracy": _mean([official[s]["action_type_accuracy"] for s in official]),
        "invalid_format_rate": _mean([official[s]["invalid_format_rate"] for s in official]),
    }

    internal_eval = summary["internal_eval"]
    return {
        "artifact_root": str(root),
        "official_splits": official,
        "official_average": official_avg,
        "internal_eval": {
            "element_accuracy": internal_eval["element_accuracy"],
            "click_point_accuracy": internal_eval["point_accuracy"],
            "iou@0.5": internal_eval["iou@0.5"],
            "action_type_accuracy": internal_eval["action_type_accuracy"],
        },
    }


def _delta_section(newer: dict[str, Any], older: dict[str, Any]) -> dict[str, float]:
    return {
        key: newer["official_average"][key] - older["official_average"][key]
        for key in (
            "element_accuracy",
            "click_point_accuracy",
            "iou@0.5",
            "action_type_accuracy",
            "invalid_format_rate",
        )
    }


def _candidate_pool_best_of_k_section(root_name: str) -> dict[str, Any]:
    root = PROJECT_ROOT / "outputs" / root_name
    first_choice_point_hits: list[bool] = []
    oracle_point_hits: list[bool] = []
    first_choice_element_hits: list[bool] = []
    oracle_element_hits: list[bool] = []
    first_choice_rewards: list[float] = []
    oracle_rewards: list[float] = []
    top_k_values: list[int] = []

    per_split: dict[str, dict[str, float]] = {}
    for split_name in ("test_task", "test_website", "test_domain"):
        rows = _load_jsonl(root / split_name / f"candidates_{split_name}.jsonl")
        split_first_point: list[bool] = []
        split_oracle_point: list[bool] = []
        split_first_reward: list[float] = []
        split_oracle_reward: list[float] = []
        for row in rows:
            candidates = sorted(row["candidates"], key=lambda cand: cand.get("rank", 10**9))
            first = candidates[0]
            oracle = max(candidates, key=lambda cand: float(cand["reward"]["total_reward"]))
            top_k_values.append(int(row.get("top_k", len(candidates))))

            fc_point = bool(first["reward"]["components"]["click_inside_target"] > 0.5)
            or_point = bool(oracle["reward"]["components"]["click_inside_target"] > 0.5)
            fc_elem = bool(first["reward"]["components"]["element_correct"] > 0.5)
            or_elem = bool(oracle["reward"]["components"]["element_correct"] > 0.5)
            fc_reward = float(first["reward"]["total_reward"])
            or_reward = float(oracle["reward"]["total_reward"])

            first_choice_point_hits.append(fc_point)
            oracle_point_hits.append(or_point)
            first_choice_element_hits.append(fc_elem)
            oracle_element_hits.append(or_elem)
            first_choice_rewards.append(fc_reward)
            oracle_rewards.append(or_reward)

            split_first_point.append(fc_point)
            split_oracle_point.append(or_point)
            split_first_reward.append(fc_reward)
            split_oracle_reward.append(or_reward)

        per_split[split_name] = {
            "first_choice_point_accuracy": _mean([float(v) for v in split_first_point]),
            "oracle_best_of_k_point_accuracy": _mean([float(v) for v in split_oracle_point]),
            "oracle_point_gain": _mean([float(b) - float(a) for a, b in zip(split_first_point, split_oracle_point)]),
            "first_choice_mean_reward": _mean(split_first_reward),
            "oracle_best_of_k_mean_reward": _mean(split_oracle_reward),
            "oracle_reward_gain": _mean([b - a for a, b in zip(split_first_reward, split_oracle_reward)]),
            "num_samples": len(rows),
        }

    return {
        "artifact_root": str(root),
        "top_k": int(round(_mean([float(v) for v in top_k_values]))),
        "aggregate": {
            "first_choice_point_accuracy": _mean([float(v) for v in first_choice_point_hits]),
            "oracle_best_of_k_point_accuracy": _mean([float(v) for v in oracle_point_hits]),
            "oracle_point_gain": _mean(
                [float(b) - float(a) for a, b in zip(first_choice_point_hits, oracle_point_hits)]
            ),
            "first_choice_element_accuracy": _mean([float(v) for v in first_choice_element_hits]),
            "oracle_best_of_k_element_accuracy": _mean([float(v) for v in oracle_element_hits]),
            "oracle_element_gain": _mean(
                [float(b) - float(a) for a, b in zip(first_choice_element_hits, oracle_element_hits)]
            ),
            "first_choice_mean_reward": _mean(first_choice_rewards),
            "oracle_best_of_k_mean_reward": _mean(oracle_rewards),
            "oracle_reward_gain": _mean([b - a for a, b in zip(first_choice_rewards, oracle_rewards)]),
            "num_samples": len(first_choice_point_hits),
        },
        "per_split": per_split,
    }


def _screenspot_section() -> dict[str, Any]:
    point_native_summary = _load_json(
        PROJECT_ROOT / "outputs" / "screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled" / "evaluation_summary.json"
    )
    dual_path_summary = _load_json(
        PROJECT_ROOT / "outputs" / "screenspot_v2_eval_qwen2_5_vl_3b_dual_path_verifier" / "evaluation_summary.json"
    )
    dual_path_subgroups = _load_json(
        PROJECT_ROOT / "outputs" / "screenspot_v2_eval_qwen2_5_vl_3b_dual_path_verifier" / "subgroup_metrics.json"
    )

    structured_shard_paths = sorted(
        (PROJECT_ROOT / "outputs").glob("screenspot_v2_eval_qwen2_5_vl_3b_model_resized_shard_*/evaluation_summary.json")
    )
    structured_duration = sum(
        float((_load_json(path).get("runtime") or {}).get("duration_seconds") or 0.0) for path in structured_shard_paths
    )
    structured_samples = sum(int(_load_json(path).get("evaluated_samples") or 0) for path in structured_shard_paths)

    point_native_sec_per_image = _sec_per_image(
        float(point_native_summary["runtime"]["duration_seconds"]),
        int(point_native_summary["evaluated_samples"]),
    )
    structured_sec_per_image = _sec_per_image(structured_duration, structured_samples)
    verifier_sec_per_image = _sec_per_image(
        float(dual_path_summary["runtime"]["duration_seconds"]),
        int(dual_path_summary["evaluated_samples"]),
    )

    web_point = float(dual_path_subgroups["platform"]["web"]["point_accuracy"])
    desktop_point = float(dual_path_subgroups["platform"]["desktop"]["point_accuracy"])
    mobile_point = float(dual_path_subgroups["platform"]["mobile"]["point_accuracy"])
    icon_point = float(dual_path_subgroups["element_type"]["icon"]["point_accuracy"])
    text_point = float(dual_path_subgroups["element_type"]["text"]["point_accuracy"])

    return {
        "point_native": {
            "point_accuracy": point_native_summary["point_accuracy"],
            "iou@0.5": point_native_summary["iou@0.5"],
            "mean_iou": point_native_summary["mean_iou"],
            "action_type_valid_rate": point_native_summary["action_type_valid_rate"],
            "parseable_output_rate": point_native_summary["parseable_output_rate"],
        },
        "dual_path_verifier": {
            "point_accuracy": dual_path_summary["point_accuracy"],
            "iou@0.5": dual_path_summary["iou@0.5"],
            "mean_iou": dual_path_summary["mean_iou"],
            "action_type_valid_rate": dual_path_summary["action_type_valid_rate"],
            "parseable_output_rate": dual_path_summary["parseable_output_rate"],
            "oracle_best_of_two_point_accuracy": dual_path_summary["oracle_best_of_two_point_accuracy"],
            "gain_vs_point_native": dual_path_summary["gain_vs_point_native"],
        },
        "subgroup_point_accuracy": {
            "text_minus_icon": text_point - icon_point,
            "desktop_minus_web": desktop_point - web_point,
            "mobile_minus_web": mobile_point - web_point,
            "text": text_point,
            "icon": icon_point,
            "desktop": desktop_point,
            "mobile": mobile_point,
            "web": web_point,
        },
        "runtime_seconds_per_image": {
            "point_native_first_choice": point_native_sec_per_image,
            "structured_support_path": structured_sec_per_image,
            "dual_path_verifier_only": verifier_sec_per_image,
            "dual_path_end_to_end": (
                (point_native_sec_per_image or 0.0)
                + (structured_sec_per_image or 0.0)
                + (verifier_sec_per_image or 0.0)
            ),
        },
        "artifact_paths": {
            "point_native_summary": str(
                PROJECT_ROOT / "outputs" / "screenspot_v2_eval_qwen2_5_vl_3b_point_native_decoupled" / "evaluation_summary.json"
            ),
            "dual_path_summary": str(
                PROJECT_ROOT / "outputs" / "screenspot_v2_eval_qwen2_5_vl_3b_dual_path_verifier" / "evaluation_summary.json"
            ),
        },
    }


def _visualwebbench_section() -> dict[str, Any]:
    point_native = _load_json(
        PROJECT_ROOT / "outputs" / "visualwebbench_eval_qwen2_5_vl_3b_point_native_decoupled" / "evaluation_summary.json"
    )
    dual_path = _load_json(
        PROJECT_ROOT / "outputs" / "visualwebbench_eval_qwen2_5_vl_3b_dual_path_verifier" / "evaluation_summary.json"
    )
    return {
        "point_native": {
            "official_choice_accuracy": point_native["official_choice_accuracy"],
            "point_accuracy": point_native["point_accuracy"],
            "iou@0.5": point_native["iou@0.5"],
            "mean_iou": point_native["mean_iou"],
        },
        "dual_path_verifier": {
            "official_choice_accuracy": dual_path["official_choice_accuracy"],
            "point_accuracy": dual_path["point_accuracy"],
            "iou@0.5": dual_path["iou@0.5"],
            "mean_iou": dual_path["mean_iou"],
        },
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    hybrid = summary["mind2web_stagea"]["hybrid"]["official_splits"]
    pure_delta = summary["mind2web_stagea"]["official_average_delta_hybrid_minus_pure_visual"]
    stageb_k4 = summary["mind2web_stageb_best_of_k"]["historical_k4"]["aggregate"]
    stageb_k8 = summary["mind2web_stageb_best_of_k"]["historical_k8"]["aggregate"]
    stageb_hybrid_k4 = summary["mind2web_stageb_best_of_k"]["hybrid_rebuild_k4"]["aggregate"]
    screenspot = summary["screenspot_v2"]
    visualwebbench = summary["visualwebbench"]

    return "\n".join(
        [
            "# Quantitative Metrics Suite",
            "",
            f"- Generated at: `{summary['generated_at_utc']}`",
            "",
            "## Mind2Web Hybrid Stage A Official Cached Subset",
            "",
            "| Split | Element Acc | Click-Point Acc | IoU@0.5 | Action-Type Acc | Invalid Format |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| test_task | {_pct(hybrid['test_task']['element_accuracy']):.2f}% | {_pct(hybrid['test_task']['click_point_accuracy']):.2f}% | {_pct(hybrid['test_task']['iou@0.5']):.2f}% | {_pct(hybrid['test_task']['action_type_accuracy']):.2f}% | {_pct(hybrid['test_task']['invalid_format_rate']):.2f}% |",
            f"| test_website | {_pct(hybrid['test_website']['element_accuracy']):.2f}% | {_pct(hybrid['test_website']['click_point_accuracy']):.2f}% | {_pct(hybrid['test_website']['iou@0.5']):.2f}% | {_pct(hybrid['test_website']['action_type_accuracy']):.2f}% | {_pct(hybrid['test_website']['invalid_format_rate']):.2f}% |",
            f"| test_domain | {_pct(hybrid['test_domain']['element_accuracy']):.2f}% | {_pct(hybrid['test_domain']['click_point_accuracy']):.2f}% | {_pct(hybrid['test_domain']['iou@0.5']):.2f}% | {_pct(hybrid['test_domain']['action_type_accuracy']):.2f}% | {_pct(hybrid['test_domain']['invalid_format_rate']):.2f}% |",
            "",
            "## OCR/DOM Hybrid Minus Pure Visual",
            "",
            f"- Official-split average element gain: `{_pct(pure_delta['element_accuracy']):.2f} pts`",
            f"- Official-split average click-point gain: `{_pct(pure_delta['click_point_accuracy']):.2f} pts`",
            f"- Official-split average IoU@0.5 gain: `{_pct(pure_delta['iou@0.5']):.2f} pts`",
            f"- Official-split average action-type gain: `{_pct(pure_delta['action_type_accuracy']):.2f} pts`",
            "",
            "## Mind2Web Stage B Oracle Best-of-k Headroom",
            "",
            f"- Historical k=4 pool: point gain `{_pct(stageb_k4['oracle_point_gain']):.2f} pts`, reward gain `{stageb_k4['oracle_reward_gain']:.4f}`",
            f"- Historical k=8 expanded pool: point gain `{_pct(stageb_k8['oracle_point_gain']):.2f} pts`, reward gain `{stageb_k8['oracle_reward_gain']:.4f}`",
            f"- Final hybrid-rebuild k=4 pool: point gain `{_pct(stageb_hybrid_k4['oracle_point_gain']):.2f} pts`, reward gain `{stageb_hybrid_k4['oracle_reward_gain']:.4f}`",
            "",
            "## ScreenSpot-v2",
            "",
            f"- Point-native: point acc `{_pct(screenspot['point_native']['point_accuracy']):.2f}%`, IoU@0.5 `{_pct(screenspot['point_native']['iou@0.5']):.2f}%`, mean IoU `{_pct(screenspot['point_native']['mean_iou']):.2f}%`",
            f"- Dual-path verifier: point acc `{_pct(screenspot['dual_path_verifier']['point_accuracy']):.2f}%`, IoU@0.5 `{_pct(screenspot['dual_path_verifier']['iou@0.5']):.2f}%`, mean IoU `{_pct(screenspot['dual_path_verifier']['mean_iou']):.2f}%`",
            f"- Text minus icon point-accuracy gap: `{_pct(screenspot['subgroup_point_accuracy']['text_minus_icon']):.2f} pts`",
            f"- Desktop minus web point-accuracy gap: `{_pct(screenspot['subgroup_point_accuracy']['desktop_minus_web']):.2f} pts`",
            f"- Mobile minus web point-accuracy gap: `{_pct(screenspot['subgroup_point_accuracy']['mobile_minus_web']):.2f} pts`",
            "",
            "## Runtime",
            "",
            f"- Point-native first choice: `{screenspot['runtime_seconds_per_image']['point_native_first_choice']:.4f} s / image`",
            f"- Structured support path: `{screenspot['runtime_seconds_per_image']['structured_support_path']:.4f} s / image`",
            f"- Dual-path verifier only: `{screenspot['runtime_seconds_per_image']['dual_path_verifier_only']:.6f} s / image`",
            f"- Dual-path end-to-end: `{screenspot['runtime_seconds_per_image']['dual_path_end_to_end']:.4f} s / image`",
            "",
            "## VisualWebBench",
            "",
            f"- Point-native official choice accuracy: `{_pct(visualwebbench['point_native']['official_choice_accuracy']):.2f}%`",
            f"- Dual-path official choice accuracy: `{_pct(visualwebbench['dual_path_verifier']['official_choice_accuracy']):.2f}%`",
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pure_visual = _mind2web_stagea_section("mind2web_stageA_sft_localization_fixed")
    hybrid = _mind2web_stagea_section("mind2web_stageA_sft_hybrid_candidates")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mind2web_stagea": {
            "pure_visual": pure_visual,
            "hybrid": hybrid,
            "official_average_delta_hybrid_minus_pure_visual": _delta_section(hybrid, pure_visual),
        },
        "mind2web_stageb_best_of_k": {
            "historical_k4": _candidate_pool_best_of_k_section("mind2web_stageB_candidates"),
            "historical_k8": _candidate_pool_best_of_k_section("mind2web_stageB_candidates_headroom_expanded"),
            "hybrid_rebuild_k4": _candidate_pool_best_of_k_section("mind2web_stageB_candidates_hybrid_stagea"),
        },
        "screenspot_v2": _screenspot_section(),
        "visualwebbench": _visualwebbench_section(),
    }

    save_json(summary, OUTPUT_DIR / "summary.json")
    (OUTPUT_DIR / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
