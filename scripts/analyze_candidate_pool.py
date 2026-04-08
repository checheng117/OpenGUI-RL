#!/usr/bin/env python3
"""Analyze rerankability and diversity diagnostics for candidate pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.io import load_jsonl
from gui_grounding.utils.logger import get_logger

logger = get_logger("analyze_candidate_pool")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate pool rerankability")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/candidate_generation_clip_grid_expanded/candidates_train_expanded.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/candidate_generation_clip_grid_expanded/diagnostics",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k for diagnostics clipping")
    return parser.parse_args()


def _draw_headroom_bar(metrics: dict, out_path: Path) -> None:
    w, h = 800, 420
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw.text((24, 20), "Candidate Pool Headroom Diagnostics", fill="black", font=font)
    categories = [
        ("first-choice already best", metrics["frac_first_choice_best"]),
        ("has positive headroom", metrics["frac_positive_headroom"]),
        ("first-choice NOT best", metrics["frac_first_choice_not_best"]),
    ]
    bar_x = 80
    bar_y = 90
    bar_h = 44
    bar_gap = 74
    max_w = 620
    for i, (name, val) in enumerate(categories):
        y = bar_y + i * bar_gap
        draw.rectangle([bar_x, y, bar_x + max_w, y + bar_h], outline="black", width=1)
        fill_w = int(max_w * max(0.0, min(1.0, val)))
        draw.rectangle([bar_x, y, bar_x + fill_w, y + bar_h], fill="#4e79a7")
        draw.text((bar_x + 8, y + 12), f"{name}: {val:.3f}", fill="white", font=font)

    draw.text(
        (24, 330),
        f"Oracle achievable mean reward gain: {metrics['avg_oracle_gain']:+.4f}",
        fill="black",
        font=font,
    )
    img.save(out_path)


def _draw_oracle_gain_hist(gains: list[float], out_path: Path) -> None:
    w, h = 800, 420
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((24, 20), "Oracle Gain Histogram (best reward - first reward)", fill="black", font=font)

    bins = [0.0, 1e-6, 0.01, 0.05, 0.1, 0.2, 1.0]
    counts = [0] * (len(bins) - 1)
    for g in gains:
        for i in range(len(bins) - 1):
            if bins[i] <= g < bins[i + 1]:
                counts[i] += 1
                break
        else:
            if g >= bins[-1]:
                counts[-1] += 1
    max_count = max(counts) if counts else 1
    x0 = 70
    y0 = 340
    bw = 90
    for i, c in enumerate(counts):
        bh = int((c / max_count) * 220) if max_count > 0 else 0
        x = x0 + i * (bw + 12)
        draw.rectangle([x, y0 - bh, x + bw, y0], fill="#f28e2b", outline="black")
        label = f"[{bins[i]:.3f},{bins[i+1]:.3f})"
        draw.text((x, y0 + 8), label, fill="black", font=font)
        draw.text((x + 28, y0 - bh - 20), str(c), fill="black", font=font)
    img.save(out_path)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = load_jsonl(in_path)
    if not pools:
        raise RuntimeError(f"No candidate pools found in {in_path}")

    reward_spreads = []
    reward_means = []
    first_choice_best_flags = []
    positive_headroom_flags = []
    oracle_gains = []
    action_div_counts = []
    grid_div_counts = []
    duplicate_ratios = []
    candidates_per_pool = []

    subset_oracle_gains = []
    subset_spreads = []
    subset_count = 0

    for pool in pools:
        candidates = list(pool.get("candidates", []))
        if args.top_k is not None:
            candidates = candidates[: args.top_k]
        if not candidates:
            continue
        rewards = [float(c["reward"]["total_reward"]) for c in candidates]
        first_reward = rewards[0]
        best_reward = max(rewards)
        min_reward = min(rewards)
        spread = best_reward - min_reward
        oracle_gain = best_reward - first_reward

        first_is_best = first_reward >= best_reward - 1e-9
        has_headroom = oracle_gain > 1e-9

        reward_spreads.append(spread)
        reward_means.append(sum(rewards) / len(rewards))
        first_choice_best_flags.append(1.0 if first_is_best else 0.0)
        positive_headroom_flags.append(1.0 if has_headroom else 0.0)
        oracle_gains.append(oracle_gain)
        candidates_per_pool.append(len(candidates))
        action_div_counts.append(len({c.get("action_type") for c in candidates}))
        grid_div_counts.append(len({int(c.get("grid_id", -1)) for c in candidates}))

        uniq = {
            (c.get("action_type"), int(c.get("grid_id", -1)), tuple(c.get("bbox", [])))
            for c in candidates
        }
        duplicate_ratio = 1.0 - (len(uniq) / len(candidates))
        duplicate_ratios.append(duplicate_ratio)

        if has_headroom:
            subset_count += 1
            subset_oracle_gains.append(oracle_gain)
            subset_spreads.append(spread)

    n = max(len(candidates_per_pool), 1)
    metrics = {
        "input_path": str(in_path),
        "num_pools": len(candidates_per_pool),
        "avg_candidates_per_pool": sum(candidates_per_pool) / n,
        "avg_reward_mean_per_pool": sum(reward_means) / n,
        "avg_reward_spread_per_pool": sum(reward_spreads) / n,
        "frac_first_choice_best": sum(first_choice_best_flags) / n,
        "frac_first_choice_not_best": 1.0 - (sum(first_choice_best_flags) / n),
        "frac_positive_headroom": sum(positive_headroom_flags) / n,
        "avg_oracle_gain": sum(oracle_gains) / n,
        "avg_action_diversity_per_pool": sum(action_div_counts) / n,
        "avg_grid_diversity_per_pool": sum(grid_div_counts) / n,
        "avg_duplicate_ratio_per_pool": sum(duplicate_ratios) / n,
        "subset_not_best_count": subset_count,
        "subset_not_best_frac": subset_count / n,
        "subset_not_best_avg_oracle_gain": (sum(subset_oracle_gains) / max(subset_count, 1)),
        "subset_not_best_avg_reward_spread": (sum(subset_spreads) / max(subset_count, 1)),
    }

    summary_json = out_dir / "candidate_pool_diagnostics_summary.json"
    table_md = out_dir / "candidate_pool_diagnostics_table.md"
    headroom_fig = out_dir / "candidate_pool_headroom.png"
    gain_hist_fig = out_dir / "candidate_pool_oracle_gain_hist.png"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| # pools | {metrics['num_pools']} |",
        f"| avg candidates/pool | {metrics['avg_candidates_per_pool']:.3f} |",
        f"| avg reward spread/pool | {metrics['avg_reward_spread_per_pool']:.6f} |",
        f"| frac first-choice already best | {metrics['frac_first_choice_best']:.6f} |",
        f"| frac positive headroom | {metrics['frac_positive_headroom']:.6f} |",
        f"| avg oracle gain | {metrics['avg_oracle_gain']:.6f} |",
        f"| avg action diversity/pool | {metrics['avg_action_diversity_per_pool']:.3f} |",
        f"| avg grid diversity/pool | {metrics['avg_grid_diversity_per_pool']:.3f} |",
        f"| avg duplicate ratio/pool | {metrics['avg_duplicate_ratio_per_pool']:.6f} |",
        f"| pools where first-choice NOT best | {metrics['subset_not_best_count']} ({metrics['subset_not_best_frac']:.6f}) |",
        f"| subset avg oracle gain | {metrics['subset_not_best_avg_oracle_gain']:.6f} |",
    ]
    table_md.write_text("\n".join(lines), encoding="utf-8")

    _draw_headroom_bar(metrics, headroom_fig)
    _draw_oracle_gain_hist(oracle_gains, gain_hist_fig)

    logger.info("Saved diagnostics summary: %s", summary_json)
    logger.info("Saved diagnostics table: %s", table_md)
    logger.info("Saved headroom figure: %s", headroom_fig)
    logger.info("Saved oracle gain histogram: %s", gain_hist_fig)


if __name__ == "__main__":
    main()
