#!/usr/bin/env python3
"""Compare official-split Mind2Web Stage-B reranker runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Mind2Web Stage-B runs.")
    parser.add_argument("--baseline-reranker", type=str, required=True)
    parser.add_argument("--candidate-dir", type=str, required=True, help="Candidate dir for the compared run")
    parser.add_argument("--compared-reranker", type=str, required=True)
    parser.add_argument("--label", type=str, required=True, help="Compared run label")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--splits", nargs="*", default=["test_task", "test_website", "test_domain"])
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _load_official_eval(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    baseline = _load_official_eval(Path(args.baseline_reranker) / "official_split_evaluations.json")
    compared = _load_official_eval(Path(args.compared_reranker) / "official_split_evaluations.json")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in args.splits:
        base_metrics = baseline[split]["metrics"]
        cur_metrics = compared[split]["metrics"]
        rows.append(
            {
                "split": split,
                "sft_first_choice_reward": cur_metrics["full_pool_baseline_mean_reward"],
                "baseline_reranked_reward": base_metrics["full_pool_reranked_mean_reward"],
                f"{args.label}_reranked_reward": cur_metrics["full_pool_reranked_mean_reward"],
                "baseline_gain": base_metrics["full_pool_reward_gain"],
                f"{args.label}_gain": cur_metrics["full_pool_reward_gain"],
                "oracle_reward": cur_metrics["full_pool_oracle_mean_reward"],
                "sft_first_choice_point_accuracy": cur_metrics["full_pool_baseline_point_accuracy"],
                f"{args.label}_point_accuracy": cur_metrics["full_pool_reranked_point_accuracy"],
                "parseable_rate": cur_metrics["full_pool_reranked_parseable_rate"],
                f"{args.label}_recovery_toward_oracle": _safe_div(
                    cur_metrics["full_pool_reward_gain"],
                    cur_metrics["full_pool_oracle_gain_upper_bound"],
                ),
                "baseline_recovery_toward_oracle": _safe_div(
                    base_metrics["full_pool_reward_gain"],
                    base_metrics["full_pool_oracle_gain_upper_bound"],
                ),
            }
        )

    json_path = output_dir / "official_split_comparison.json"
    md_path = output_dir / "official_split_comparison.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    lines = [
        "| Split | SFT reward | Baseline rerank | Compared rerank | Oracle | Baseline gain | Compared gain | SFT point acc | Compared point acc | Parseable | Baseline recovery | Compared recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['sft_first_choice_reward']:.4f} | {row['baseline_reranked_reward']:.4f} | "
            f"{row[f'{args.label}_reranked_reward']:.4f} | {row['oracle_reward']:.4f} | "
            f"{row['baseline_gain']:+.4f} | {row[f'{args.label}_gain']:+.4f} | "
            f"{row['sft_first_choice_point_accuracy']:.4f} | {row[f'{args.label}_point_accuracy']:.4f} | "
            f"{row['parseable_rate']:.4f} | {row['baseline_recovery_toward_oracle']:.4f} | "
            f"{row[f'{args.label}_recovery_toward_oracle']:.4f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
