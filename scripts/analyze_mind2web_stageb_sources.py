#!/usr/bin/env python3
"""Per-source attribution analysis for Mind2Web Stage-B candidate pools."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.io import load_jsonl, save_json
from gui_grounding.utils.logger import get_logger

logger = get_logger("analyze_mind2web_stageb_sources")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-source Mind2Web Stage-B attribution.")
    parser.add_argument("--candidate-dir", type=str, required=True, help="Directory with split subdirs and candidate JSONL")
    parser.add_argument(
        "--reranker-dir",
        type=str,
        default=None,
        help="Optional reranker output dir with evaluation_<split>_per_sample.jsonl files",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["test_task", "test_website", "test_domain"],
        help="Official split names to analyze",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _normalize_source_name(source: str | None) -> str:
    source = str(source or "")
    if source.startswith("structured_sampled_t0p6"):
        return "structured_sampled_t0p6"
    if source.startswith("point_first_sampled_t0p7"):
        return "point_first_sampled_t0p7"
    if source.startswith("point_first_structured"):
        return "point_first_structured"
    if source.startswith("point_native_primary"):
        return "point_native_primary"
    if source.startswith("hybrid_point_structured"):
        return "hybrid_point_structured"
    if source.startswith("stagea_first_choice"):
        return "stagea_first_choice"
    return source or "unknown"


def _bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
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


def _click_distance(click_a: list[float] | None, click_b: list[float] | None) -> float:
    if not click_a or not click_b or len(click_a) != 2 or len(click_b) != 2:
        return 0.0
    dx = float(click_a[0]) - float(click_b[0])
    dy = float(click_a[1]) - float(click_b[1])
    return math.sqrt(dx * dx + dy * dy)


def _candidate_rows_by_id(pool: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c.get("candidate_id")): c for c in pool.get("candidates", [])}


def _summarize_split(
    split: str,
    candidate_path: Path,
    reranker_path: Path | None,
) -> dict[str, Any]:
    pools = load_jsonl(candidate_path)
    reranker_rows = load_jsonl(reranker_path) if reranker_path and reranker_path.exists() else []
    reranker_by_sample = {str(row.get("sample_id")): row for row in reranker_rows}

    source_stats: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    split_summary: dict[str, Any] = {
        "split": split,
        "num_pools": len(pools),
        "avg_candidates_per_pool": 0.0,
        "first_choice_mean_reward": 0.0,
        "oracle_mean_reward": 0.0,
        "oracle_point_accuracy": 0.0,
        "parseable_rate": 0.0,
    }
    candidate_count_total = 0
    first_reward_total = 0.0
    oracle_reward_total = 0.0
    oracle_click_total = 0.0
    parseable_total = 0.0

    for pool in pools:
        sample_id = str(pool.get("sample_id"))
        candidates = list(pool.get("candidates", []))
        if not candidates:
            continue

        candidate_count_total += len(candidates)
        first_candidate = candidates[0]
        first_reward = float(first_candidate.get("reward", {}).get("total_reward", 0.0))
        first_reward_total += first_reward
        parseable_total += 1.0 if (first_candidate.get("structured_output_diagnostics") or {}).get("json_parse_success") is not False else 0.0

        rewards = [float(c.get("reward", {}).get("total_reward", 0.0)) for c in candidates]
        oracle_reward = max(rewards)
        oracle_reward_total += oracle_reward
        oracle_candidates = [c for c in candidates if abs(float(c.get("reward", {}).get("total_reward", 0.0)) - oracle_reward) <= 1e-9]
        oracle_click_total += max(float((c.get("reward", {}).get("components", {}) or {}).get("click_inside_target", 0.0)) for c in oracle_candidates)

        first_click = first_candidate.get("click_point")
        first_bbox = first_candidate.get("bbox_proposal")
        first_action = first_candidate.get("action_type")

        rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            rows_by_source[_normalize_source_name(candidate.get("source"))].append(candidate)

        better_than_first_sources = {
            source
            for source, rows in rows_by_source.items()
            if max(float(r.get("reward", {}).get("total_reward", 0.0)) for r in rows) > first_reward + 1e-9
        }
        oracle_sources = {
            _normalize_source_name(candidate.get("source"))
            for candidate in oracle_candidates
        }

        for source, rows in rows_by_source.items():
            rewards_for_source = [float(r.get("reward", {}).get("total_reward", 0.0)) for r in rows]
            click_for_source = [
                float((r.get("reward", {}).get("components", {}) or {}).get("click_inside_target", 0.0))
                for r in rows
            ]
            point_acc_for_source = click_for_source
            parseable_for_source = [
                1.0 if (r.get("structured_output_diagnostics") or {}).get("json_parse_success") is not False else 0.0
                for r in rows
            ]
            near_dup_to_first = 0.0
            for row in rows:
                same_action = row.get("action_type") == first_action and row.get("action_type") is not None
                click_near = _click_distance(row.get("click_point"), first_click) <= 96.0
                bbox_near = _bbox_iou(row.get("bbox_proposal"), first_bbox) >= 0.5
                if same_action and click_near and bbox_near:
                    near_dup_to_first = 1.0
                    break

            st = source_stats[source]
            st["source"] = source
            st["pools_present"] += 1
            st["candidate_count"] += len(rows)
            st["reward_sum"] += sum(rewards_for_source)
            st["point_sum"] += sum(point_acc_for_source)
            st["parseable_sum"] += sum(parseable_for_source)
            st["source_best_reward_sum"] += max(rewards_for_source)
            st["source_best_gain_vs_first_sum"] += max(rewards_for_source) - first_reward
            st["near_duplicate_to_first_pool_count"] += near_dup_to_first
            if source in oracle_sources:
                st["oracle_best_pool_count"] += 1
            if source in better_than_first_sources:
                st["better_than_first_pool_count"] += 1
            else:
                st["no_gain_pool_count"] += 1

        reranker_row = reranker_by_sample.get(sample_id)
        if reranker_row:
            id_to_candidate = _candidate_rows_by_id(pool)
            selected_candidate = id_to_candidate.get(str(reranker_row.get("reranked_candidate_id")))
            baseline_candidate = id_to_candidate.get(str(reranker_row.get("baseline_candidate_id")))
            oracle_candidate = id_to_candidate.get(str(reranker_row.get("oracle_candidate_id")))

            if selected_candidate is not None:
                selected_source = _normalize_source_name(selected_candidate.get("source"))
                st = source_stats[selected_source]
                st["selected_count"] += 1
                st["selected_reward_sum"] += float(reranker_row.get("reranked_reward", 0.0))
                st["selected_point_sum"] += float(reranker_row.get("reranked_click", 0.0))
                st["selected_parseable_sum"] += float(reranker_row.get("reranked_parseable", 0.0))
                baseline_reward = float(reranker_row.get("baseline_reward", 0.0))
                reranked_reward = float(reranker_row.get("reranked_reward", 0.0))
                if reranked_reward > baseline_reward + 1e-9:
                    st["selected_win_count"] += 1
                if reranked_reward < baseline_reward - 1e-9:
                    st["selected_harm_count"] += 1

            if baseline_candidate is not None:
                baseline_source = _normalize_source_name(baseline_candidate.get("source"))
                st = source_stats[baseline_source]
                st["baseline_selected_count"] += 1
                st["baseline_selected_reward_sum"] += float(reranker_row.get("baseline_reward", 0.0))

            if oracle_candidate is not None:
                oracle_source = _normalize_source_name(oracle_candidate.get("source"))
                source_stats[oracle_source]["oracle_selected_count"] += 1

    num_pools = max(len(pools), 1)
    split_summary["avg_candidates_per_pool"] = candidate_count_total / num_pools
    split_summary["first_choice_mean_reward"] = first_reward_total / num_pools
    split_summary["oracle_mean_reward"] = oracle_reward_total / num_pools
    split_summary["oracle_point_accuracy"] = oracle_click_total / num_pools
    split_summary["parseable_rate"] = parseable_total / num_pools

    source_rows: list[dict[str, Any]] = []
    for source, st in source_stats.items():
        pools_present = max(int(st["pools_present"]), 1)
        candidate_count = max(int(st["candidate_count"]), 1)
        selected_count = max(int(st["selected_count"]), 1)
        source_rows.append(
            {
                "source": source,
                "avg_candidates_per_pool": _safe_div(st["candidate_count"], num_pools),
                "avg_candidates_when_present": _safe_div(st["candidate_count"], pools_present),
                "pool_presence_rate": _safe_div(st["pools_present"], num_pools),
                "oracle_contribution_rate": _safe_div(st["oracle_best_pool_count"], num_pools),
                "better_than_first_rate": _safe_div(st["better_than_first_pool_count"], num_pools),
                "no_gain_when_present_rate": _safe_div(st["no_gain_pool_count"], st["pools_present"]),
                "near_duplicate_to_first_rate": _safe_div(st["near_duplicate_to_first_pool_count"], st["pools_present"]),
                "mean_candidate_reward": _safe_div(st["reward_sum"], candidate_count),
                "mean_candidate_point_accuracy": _safe_div(st["point_sum"], candidate_count),
                "mean_candidate_parseable_rate": _safe_div(st["parseable_sum"], candidate_count),
                "mean_source_best_reward": _safe_div(st["source_best_reward_sum"], st["pools_present"]),
                "mean_source_best_gain_vs_first": _safe_div(st["source_best_gain_vs_first_sum"], st["pools_present"]),
                "reranker_selected_rate": _safe_div(st["selected_count"], num_pools),
                "selected_source_reward": _safe_div(st["selected_reward_sum"], selected_count),
                "selected_source_point_accuracy": _safe_div(st["selected_point_sum"], selected_count),
                "selected_source_parseable_rate": _safe_div(st["selected_parseable_sum"], selected_count),
                "selected_source_win_rate": _safe_div(st["selected_win_count"], st["selected_count"]),
                "selected_source_harm_rate": _safe_div(st["selected_harm_count"], st["selected_count"]),
                "oracle_selected_rate": _safe_div(st["oracle_selected_count"], num_pools),
                "baseline_selected_rate": _safe_div(st["baseline_selected_count"], num_pools),
            }
        )

    source_rows.sort(
        key=lambda row: (
            -float(row["oracle_contribution_rate"]),
            -float(row["better_than_first_rate"]),
            -float(row["reranker_selected_rate"]),
            row["source"],
        )
    )

    return {
        "split_summary": split_summary,
        "source_rows": source_rows,
    }


def _write_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = [
        "# Mind2Web Stage-B Per-Source Attribution",
        "",
    ]
    for split in analysis["splits"]:
        split_name = split["split_summary"]["split"]
        summary = split["split_summary"]
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- num_pools: {summary['num_pools']}",
                f"- avg_candidates_per_pool: {summary['avg_candidates_per_pool']:.4f}",
                f"- first_choice_mean_reward: {summary['first_choice_mean_reward']:.4f}",
                f"- oracle_mean_reward: {summary['oracle_mean_reward']:.4f}",
                f"- oracle_point_accuracy: {summary['oracle_point_accuracy']:.4f}",
                f"- parseable_rate: {summary['parseable_rate']:.4f}",
                "",
                "| Source | Avg cand/pool | Oracle contrib | Better than first | No-gain when present | Near-dup to first | Selected | Selected reward | Selected harm | Source-best gain |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in split["source_rows"]:
            lines.append(
                "| {source} | {avg_candidates_per_pool:.2f} | {oracle_contribution_rate:.4f} | {better_than_first_rate:.4f} | {no_gain_when_present_rate:.4f} | {near_duplicate_to_first_rate:.4f} | {reranker_selected_rate:.4f} | {selected_source_reward:.4f} | {selected_source_harm_rate:.4f} | {mean_source_best_gain_vs_first:+.4f} |".format(
                    **row
                )
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    reranker_dir = Path(args.reranker_dir) if args.reranker_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_analyses = []
    for split in args.splits:
        candidate_path = candidate_dir / split / f"candidates_{split}.jsonl"
        reranker_path = reranker_dir / f"evaluation_{split}_per_sample.jsonl" if reranker_dir else None
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing candidate file for split={split}: {candidate_path}")
        split_analyses.append(_summarize_split(split, candidate_path, reranker_path))

    analysis = {"candidate_dir": str(candidate_dir), "reranker_dir": str(reranker_dir) if reranker_dir else None, "splits": split_analyses}
    json_path = output_dir / "source_attribution_summary.json"
    md_path = output_dir / "source_attribution_summary.md"
    save_json(analysis, json_path)
    _write_markdown(analysis, md_path)
    logger.info("Saved source attribution JSON: %s", json_path)
    logger.info("Saved source attribution Markdown: %s", md_path)


if __name__ == "__main__":
    main()
