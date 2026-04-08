#!/usr/bin/env python3
"""Analyze genuine Mind2Web Stage-B recovery pools and reranker misses."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage-B genuine recovery cases.")
    parser.add_argument("--candidate-root", type=str, required=True)
    parser.add_argument("--reranker-eval-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--splits", nargs="*", default=["train", "test_task", "test_website", "test_domain"])
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_source_name(source: str | None) -> str:
    source = str(source or "")
    if source.startswith("stagea_first_choice"):
        return "stagea_first_choice"
    if source.startswith("structured_sampled_t0p6"):
        return "structured_sampled_t0p6"
    if source.startswith("point_first_structured"):
        return "point_first_structured"
    if source.startswith("point_native_primary"):
        return "point_native_primary"
    if source.startswith("point_first_sampled_t0p7"):
        return "point_first_sampled_t0p7"
    if source.startswith("hybrid_point_structured"):
        return "hybrid_point_structured"
    if source.startswith("legacy_clip_grid"):
        return "legacy_clip_grid"
    return "other"


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _sample_recovery_record(sample: dict[str, Any]) -> dict[str, Any] | None:
    candidates = sample.get("candidates", [])
    if len(candidates) < 2:
        return None
    rewards = [float(c.get("reward", {}).get("total_reward", 0.0)) for c in candidates]
    first_reward = rewards[0]
    positive_indices = [idx for idx, reward in enumerate(rewards) if idx != 0 and reward > first_reward + 1e-9]
    if not positive_indices:
        return None
    best_idx = max(positive_indices, key=lambda idx: rewards[idx])
    positive_sources = [_normalize_source_name(candidates[idx].get("source")) for idx in positive_indices]
    record = {
        "sample_id": sample.get("sample_id"),
        "first_reward": first_reward,
        "oracle_reward": rewards[best_idx],
        "oracle_gap": rewards[best_idx] - first_reward,
        "oracle_candidate_id": candidates[best_idx].get("candidate_id"),
        "oracle_source": _normalize_source_name(candidates[best_idx].get("source")),
        "positive_count": len(positive_indices),
        "positive_sources": positive_sources,
        "candidate_sources": [_normalize_source_name(c.get("source")) for c in candidates],
        "candidate_rewards": rewards,
    }
    return record


def _load_reranker_rows(eval_root: Path, split: str) -> dict[str, dict[str, Any]]:
    eval_path = eval_root / f"evaluation_{split}_per_sample.jsonl"
    if not eval_path.exists():
        return {}
    rows = _load_jsonl(eval_path)
    return {row["sample_id"]: row for row in rows}


def _summarize_split(
    split: str,
    candidate_rows: list[dict[str, Any]],
    reranker_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    recovery_records = []
    for row in candidate_rows:
        record = _sample_recovery_record(row)
        if record is not None:
            recovery_records.append(record)

    oracle_source_counts = Counter(record["oracle_source"] for record in recovery_records)
    positive_source_combo_counts = Counter(tuple(sorted(set(record["positive_sources"]))) for record in recovery_records)
    positive_count_hist = Counter(record["positive_count"] for record in recovery_records)
    oracle_gaps = [record["oracle_gap"] for record in recovery_records]

    reranker_hit_count = 0
    reranker_miss_count = 0
    miss_examples = []
    for record in recovery_records:
        rer_row = reranker_rows.get(record["sample_id"])
        if rer_row is None:
            continue
        baseline_reward = float(rer_row["baseline_reward"])
        reranked_reward = float(rer_row["reranked_reward"])
        oracle_reward = float(rer_row["oracle_reward"])
        improved = reranked_reward > baseline_reward + 1e-9
        if improved:
            reranker_hit_count += 1
        else:
            reranker_miss_count += 1
            miss_examples.append(
                {
                    "sample_id": record["sample_id"],
                    "baseline_reward": baseline_reward,
                    "reranked_reward": reranked_reward,
                    "oracle_reward": oracle_reward,
                    "oracle_gap": record["oracle_gap"],
                    "oracle_source": record["oracle_source"],
                    "positive_sources": record["positive_sources"],
                }
            )

    miss_examples = sorted(miss_examples, key=lambda row: row["oracle_gap"], reverse=True)
    return {
        "split": split,
        "num_pools": len(candidate_rows),
        "headroom_pool_count": len(recovery_records),
        "headroom_rate": (len(recovery_records) / len(candidate_rows)) if candidate_rows else 0.0,
        "oracle_gap_mean": _safe_mean(oracle_gaps),
        "oracle_gap_max": max(oracle_gaps) if oracle_gaps else 0.0,
        "oracle_source_counts": dict(oracle_source_counts.most_common()),
        "positive_source_combo_counts": {
            " + ".join(combo): count for combo, count in positive_source_combo_counts.most_common()
        },
        "positive_count_histogram": dict(sorted(positive_count_hist.items())),
        "reranker_headroom_hit_count": reranker_hit_count,
        "reranker_headroom_miss_count": reranker_miss_count,
        "reranker_headroom_recovery_rate": (
            reranker_hit_count / max(reranker_hit_count + reranker_miss_count, 1)
        ),
        "miss_examples": miss_examples[:10],
    }


def _format_summary_md(summary: dict[str, Any]) -> str:
    lines = ["# Mind2Web Stage-B Genuine Recovery Cases", ""]
    for split in summary["splits"]:
        record = summary["by_split"][split]
        lines.extend(
            [
                f"## {split}",
                "",
                f"- num_pools: {record['num_pools']}",
                f"- headroom_pool_count: {record['headroom_pool_count']}",
                f"- headroom_rate: {record['headroom_rate']:.4f}",
                f"- oracle_gap_mean: {record['oracle_gap_mean']:.4f}",
                f"- oracle_gap_max: {record['oracle_gap_max']:.4f}",
                f"- reranker_headroom_recovery_rate: {record['reranker_headroom_recovery_rate']:.4f}",
                "",
                "### Oracle Best Source Counts",
                "",
            ]
        )
        for source, count in record["oracle_source_counts"].items():
            lines.append(f"- {source}: {count}")
        lines.extend(["", "### Positive Source Combos", ""])
        for combo, count in record["positive_source_combo_counts"].items():
            lines.append(f"- {combo}: {count}")
        lines.extend(["", "### Positive Count Histogram", ""])
        for size, count in record["positive_count_histogram"].items():
            lines.append(f"- {size}: {count}")
        lines.extend(["", "### Recovery Miss Examples", ""])
        if record["miss_examples"]:
            for example in record["miss_examples"]:
                lines.append(
                    "- "
                    + (
                        f"{example['sample_id']} | base={example['baseline_reward']:.4f} "
                        f"rerank={example['reranked_reward']:.4f} oracle={example['oracle_reward']:.4f} "
                        f"gap={example['oracle_gap']:.4f} oracle_source={example['oracle_source']} "
                        f"positive_sources={example['positive_sources']}"
                    )
                )
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root)
    reranker_root = Path(args.reranker_eval_root) if args.reranker_eval_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, Any] = {}
    for split in args.splits:
        candidate_path = candidate_root / split / f"candidates_{split}.jsonl"
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing candidate file: {candidate_path}")
        reranker_rows = _load_reranker_rows(reranker_root, split) if reranker_root else {}
        by_split[split] = _summarize_split(split, _load_jsonl(candidate_path), reranker_rows)

    summary = {"splits": args.splits, "by_split": by_split}
    (output_dir / "recovery_cases_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "recovery_cases_summary.md").write_text(_format_summary_md(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
