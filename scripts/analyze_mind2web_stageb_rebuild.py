#!/usr/bin/env python3
"""Summarize rebuilt Mind2Web Stage-B candidate pools and reranker results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.io import load_json, load_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rebuilt Mind2Web Stage-B hybrid-Stage-A runs")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate pool path in split=path form. May be passed multiple times.",
    )
    parser.add_argument(
        "--old-candidate",
        action="append",
        default=[],
        help="Old candidate pool path in split=path form. May be passed multiple times.",
    )
    parser.add_argument("--reranker-evals", type=str, default=None)
    parser.add_argument("--old-reranker-evals", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    parser.add_argument("--new-label", type=str, default="Hybrid Stage A Rebuild")
    parser.add_argument("--old-label", type=str, default="Old Stage A Stage B")
    return parser.parse_args()


def _parse_keyed_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected split=path, got: {value}")
        split, path = value.split("=", 1)
        result[split] = Path(path)
    return result


def _candidate_summary(path: Path) -> dict[str, Any]:
    pools = load_jsonl(path)
    if not pools:
        raise RuntimeError(f"No candidate pools found in {path}")

    rows: list[dict[str, Any]] = []
    oracle_sources: Counter[str] = Counter()
    headroom_oracle_sources: Counter[str] = Counter()
    total_candidates = 0
    total_unique_sources = 0

    for pool in pools:
        candidates = list(pool.get("candidates", []))
        if not candidates:
            continue
        total_candidates += len(candidates)
        total_unique_sources += len({str(c.get("source", "")) for c in candidates})

        rewards = [float(c.get("reward", {}).get("total_reward", 0.0)) for c in candidates]
        oracle_idx = max(range(len(candidates)), key=lambda idx: rewards[idx])
        first = candidates[0]
        oracle = candidates[oracle_idx]
        first_components = first.get("reward", {}).get("components", {}) or {}
        oracle_components = oracle.get("reward", {}).get("components", {}) or {}
        first_diag = first.get("structured_output_diagnostics") or {}
        oracle_diag = oracle.get("structured_output_diagnostics") or {}
        oracle_source = str(oracle.get("source", "unknown"))
        oracle_sources[oracle_source] += 1
        headroom = rewards[oracle_idx] > rewards[0] + 1e-9
        if headroom:
            headroom_oracle_sources[oracle_source] += 1
        rows.append(
            {
                "num_candidates": len(candidates),
                "num_unique_sources": len({str(c.get("source", "")) for c in candidates}),
                "baseline_reward": rewards[0],
                "oracle_reward": rewards[oracle_idx],
                "baseline_iou": float(first_components.get("iou", 0.0)),
                "oracle_iou": float(oracle_components.get("iou", 0.0)),
                "baseline_click": float(first_components.get("click_inside_target", 0.0)),
                "oracle_click": float(oracle_components.get("click_inside_target", 0.0)),
                "baseline_action": float(first_components.get("action_type_correct", 0.0)),
                "oracle_action": float(oracle_components.get("action_type_correct", 0.0)),
                "baseline_parseable": 1.0 if first_diag.get("json_parse_success") is not False else 0.0,
                "oracle_parseable": 1.0 if oracle_diag.get("json_parse_success") is not False else 0.0,
                "headroom": 1.0 if headroom else 0.0,
            }
        )

    n = max(len(rows), 1)
    headroom_rows = [row for row in rows if row["headroom"] > 0.0]
    headroom_count = len(headroom_rows)
    result = {
        "candidate_path": str(path),
        "num_samples": len(rows),
        "avg_candidates_per_sample": total_candidates / n,
        "avg_unique_sources_per_sample": total_unique_sources / n,
        "baseline_mean_reward": sum(r["baseline_reward"] for r in rows) / n,
        "oracle_mean_reward": sum(r["oracle_reward"] for r in rows) / n,
        "baseline_mean_iou": sum(r["baseline_iou"] for r in rows) / n,
        "oracle_mean_iou": sum(r["oracle_iou"] for r in rows) / n,
        "baseline_point_accuracy": sum(r["baseline_click"] for r in rows) / n,
        "oracle_point_accuracy": sum(r["oracle_click"] for r in rows) / n,
        "baseline_action_accuracy": sum(r["baseline_action"] for r in rows) / n,
        "oracle_action_accuracy": sum(r["oracle_action"] for r in rows) / n,
        "baseline_parseable_rate": sum(r["baseline_parseable"] for r in rows) / n,
        "oracle_parseable_rate": sum(r["oracle_parseable"] for r in rows) / n,
        "first_to_oracle_reward_gap": sum(r["oracle_reward"] - r["baseline_reward"] for r in rows) / n,
        "first_to_oracle_point_gap": sum(r["oracle_click"] - r["baseline_click"] for r in rows) / n,
        "headroom_pool_fraction": headroom_count / n,
        "headroom_pool_count": headroom_count,
        "headroom_mean_reward_gap": (
            sum(r["oracle_reward"] - r["baseline_reward"] for r in headroom_rows) / max(headroom_count, 1)
        ),
        "headroom_mean_point_gap": (
            sum(r["oracle_click"] - r["baseline_click"] for r in headroom_rows) / max(headroom_count, 1)
        ),
        "oracle_source_histogram": dict(oracle_sources),
        "headroom_oracle_source_histogram": dict(headroom_oracle_sources),
    }
    return result


def _reranker_summary(path: Path) -> dict[str, Any]:
    raw = load_json(path)
    result: dict[str, Any] = {}
    for split, row in raw.items():
        metrics = row.get("metrics", {})
        baseline_reward = float(metrics.get("full_pool_baseline_mean_reward", 0.0))
        reranked_reward = float(metrics.get("full_pool_reranked_mean_reward", 0.0))
        oracle_reward = float(metrics.get("full_pool_oracle_mean_reward", 0.0))
        upper_bound = oracle_reward - baseline_reward
        gain = reranked_reward - baseline_reward
        result[split] = {
            "candidate_path": row.get("candidate_path"),
            "num_samples": row.get("num_samples"),
            "baseline_reward": baseline_reward,
            "reranked_reward": reranked_reward,
            "oracle_reward": oracle_reward,
            "reward_gain": gain,
            "recovery_toward_oracle": gain / upper_bound if upper_bound > 1e-9 else 0.0,
            "baseline_point_accuracy": float(metrics.get("full_pool_baseline_point_accuracy", 0.0)),
            "reranked_point_accuracy": float(metrics.get("full_pool_reranked_point_accuracy", 0.0)),
            "oracle_point_accuracy": float(metrics.get("full_pool_oracle_point_accuracy", 0.0)),
            "baseline_parseable_rate": float(metrics.get("full_pool_baseline_parseable_rate", 0.0)),
            "reranked_parseable_rate": float(metrics.get("full_pool_reranked_parseable_rate", 0.0)),
            "oracle_parseable_rate": float(metrics.get("full_pool_oracle_parseable_rate", 0.0)),
            "baseline_to_oracle_gap": float(metrics.get("full_pool_baseline_to_oracle_gap", 0.0)),
            "reranked_to_oracle_gap": float(metrics.get("full_pool_reranked_to_oracle_gap", 0.0)),
            "baseline_action_accuracy": float(metrics.get("full_pool_baseline_action_type_correct", 0.0)),
            "reranked_action_accuracy": float(metrics.get("full_pool_reranked_action_type_correct", 0.0)),
            "oracle_action_accuracy": float(metrics.get("full_pool_oracle_action_type_correct", 0.0)),
        }
    return result


def _build_headroom_markdown(
    new_results: dict[str, dict[str, Any]],
    old_results: dict[str, dict[str, Any]],
    new_label: str,
    old_label: str,
) -> str:
    lines = [
        "# Mind2Web Stage-B Rebuild Headroom Summary",
        "",
        f"- new_label: {new_label}",
        f"- old_label: {old_label}",
        "",
        "| Split | Avg cand | Oracle reward | Baseline reward | Reward gap | Oracle point | Baseline point | Headroom frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in new_results.items():
        lines.append(
            f"| {split} | {row['avg_candidates_per_sample']:.2f} | {row['oracle_mean_reward']:.4f} | "
            f"{row['baseline_mean_reward']:.4f} | {row['first_to_oracle_reward_gap']:.4f} | "
            f"{row['oracle_point_accuracy']:.4f} | {row['baseline_point_accuracy']:.4f} | "
            f"{row['headroom_pool_fraction']:.4f} |"
        )
    if old_results:
        lines.extend(
            [
                "",
                "## Old vs New",
                "",
                "| Split | Avg cand old -> new | Oracle reward old -> new | Baseline reward old -> new | Headroom frac old -> new | Oracle point old -> new |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for split, new_row in new_results.items():
            old_row = old_results.get(split)
            if old_row is None:
                continue
            lines.append(
                f"| {split} | {old_row['avg_candidates_per_sample']:.2f} -> {new_row['avg_candidates_per_sample']:.2f} | "
                f"{old_row['oracle_mean_reward']:.4f} -> {new_row['oracle_mean_reward']:.4f} | "
                f"{old_row['baseline_mean_reward']:.4f} -> {new_row['baseline_mean_reward']:.4f} | "
                f"{old_row['headroom_pool_fraction']:.4f} -> {new_row['headroom_pool_fraction']:.4f} | "
                f"{old_row['oracle_point_accuracy']:.4f} -> {new_row['oracle_point_accuracy']:.4f} |"
            )
    return "\n".join(lines)


def _build_reranker_markdown(
    new_results: dict[str, dict[str, Any]],
    old_results: dict[str, dict[str, Any]],
    new_label: str,
    old_label: str,
) -> str:
    lines = [
        "# Mind2Web Stage-B Rebuild Reranker Summary",
        "",
        f"- new_label: {new_label}",
        f"- old_label: {old_label}",
        "",
        "| Split | Baseline reward | Reranked reward | Oracle reward | Gain | Recovery | Baseline point | Reranked point | Oracle point | Parseable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in new_results.items():
        lines.append(
            f"| {split} | {row['baseline_reward']:.4f} | {row['reranked_reward']:.4f} | {row['oracle_reward']:.4f} | "
            f"{row['reward_gain']:+.4f} | {row['recovery_toward_oracle']:.4f} | "
            f"{row['baseline_point_accuracy']:.4f} | {row['reranked_point_accuracy']:.4f} | "
            f"{row['oracle_point_accuracy']:.4f} | {row['reranked_parseable_rate']:.4f} |"
        )
    if old_results:
        lines.extend(
            [
                "",
                "## Old vs New",
                "",
                "| Split | Gain old -> new | Recovery old -> new | Baseline point old -> new | Reranked point old -> new |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for split, new_row in new_results.items():
            old_row = old_results.get(split)
            if old_row is None:
                continue
            lines.append(
                f"| {split} | {old_row['reward_gain']:+.4f} -> {new_row['reward_gain']:+.4f} | "
                f"{old_row['recovery_toward_oracle']:.4f} -> {new_row['recovery_toward_oracle']:.4f} | "
                f"{old_row['baseline_point_accuracy']:.4f} -> {new_row['baseline_point_accuracy']:.4f} | "
                f"{old_row['reranked_point_accuracy']:.4f} -> {new_row['reranked_point_accuracy']:.4f} |"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_candidate_paths = _parse_keyed_paths(args.candidate)
    old_candidate_paths = _parse_keyed_paths(args.old_candidate)

    new_headroom = {
        split: _candidate_summary(path)
        for split, path in sorted(new_candidate_paths.items())
    }
    old_headroom = {
        split: _candidate_summary(path)
        for split, path in sorted(old_candidate_paths.items())
    }
    save_json(new_headroom, output_dir / "headroom_summary_new.json")
    if old_headroom:
        save_json(old_headroom, output_dir / "headroom_summary_old.json")

    headroom_md = _build_headroom_markdown(
        new_results=new_headroom,
        old_results=old_headroom,
        new_label=args.new_label,
        old_label=args.old_label,
    )
    (output_dir / "headroom_summary.md").write_text(headroom_md, encoding="utf-8")

    if args.reranker_evals:
        new_reranker = _reranker_summary(Path(args.reranker_evals))
        old_reranker = _reranker_summary(Path(args.old_reranker_evals)) if args.old_reranker_evals else {}
        save_json(new_reranker, output_dir / "reranker_summary_new.json")
        if old_reranker:
            save_json(old_reranker, output_dir / "reranker_summary_old.json")
        reranker_md = _build_reranker_markdown(
            new_results=new_reranker,
            old_results=old_reranker,
            new_label=args.new_label,
            old_label=args.old_label,
        )
        (output_dir / "reranker_summary.md").write_text(reranker_md, encoding="utf-8")

        combined = {
            "new_label": args.new_label,
            "old_label": args.old_label,
            "headroom_new": new_headroom,
            "headroom_old": old_headroom,
            "reranker_new": new_reranker,
            "reranker_old": old_reranker,
        }
        save_json(combined, output_dir / "combined_summary.json")


if __name__ == "__main__":
    main()
