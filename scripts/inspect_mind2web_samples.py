#!/usr/bin/env python3
"""Inspect Multimodal-Mind2Web samples: print stats, visualize bbox/click.

Usage:
    python scripts/inspect_mind2web_samples.py --split train --max-samples 20
    python scripts/inspect_mind2web_samples.py --split test_task --max-samples 10 --save-dir outputs/inspection
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.logger import get_logger

logger = get_logger("inspect_mind2web")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect Mind2Web samples")
    p.add_argument("--split", type=str, default="train", help="Dataset split")
    p.add_argument("--max-samples", type=int, default=20, help="Samples to load")
    p.add_argument("--save-dir", type=str, default="outputs/inspection", help="Dir for visualizations")
    p.add_argument("--num-vis", type=int, default=5, help="Number of samples to visualize")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Mind2Web split=%s, max_samples=%d ...", args.split, args.max_samples)

    from gui_grounding.data.mind2web_dataset import Mind2WebDataset

    ds = Mind2WebDataset(
        split=args.split,
        max_samples=args.max_samples,
        cache_screenshots=True,
    )

    logger.info("Loaded %d samples.", len(ds))
    if len(ds) == 0:
        logger.warning("No samples loaded. Exiting.")
        return

    # ------------------------------------------------------------------
    # Field statistics
    # ------------------------------------------------------------------
    action_types = Counter()
    has_bbox = 0
    has_click = 0
    has_element_id = 0
    has_dom_candidates = 0
    websites = Counter()
    domains = Counter()
    img_sizes: list[tuple[int, int]] = []

    for s in ds:
        if s.action_type:
            action_types[s.action_type] += 1
        if s.target_bbox is not None:
            has_bbox += 1
        if s.click_point is not None:
            has_click += 1
        if s.target_element_id is not None:
            has_element_id += 1
        if s.dom_candidates:
            has_dom_candidates += 1
        if s.website:
            websites[s.website] += 1
        if s.domain:
            domains[s.domain] += 1

        img = ds.get_screenshot(s)
        if img is not None:
            img_sizes.append(img.size)

    n = len(ds)
    stats = {
        "split": args.split,
        "total_samples": n,
        "action_type_distribution": dict(action_types),
        "has_bbox": f"{has_bbox}/{n} ({has_bbox/n:.1%})",
        "has_click_point": f"{has_click}/{n} ({has_click/n:.1%})",
        "has_element_id": f"{has_element_id}/{n} ({has_element_id/n:.1%})",
        "has_dom_candidates": f"{has_dom_candidates}/{n} ({has_dom_candidates/n:.1%})",
        "unique_websites": len(websites),
        "unique_domains": len(domains),
        "top_websites": dict(websites.most_common(5)),
        "top_domains": dict(domains.most_common(5)),
    }
    if img_sizes:
        widths = [s[0] for s in img_sizes]
        heights = [s[1] for s in img_sizes]
        stats["image_width_range"] = f"{min(widths)}-{max(widths)}"
        stats["image_height_range"] = f"{min(heights)}-{max(heights)}"

    logger.info("=" * 60)
    logger.info("FIELD STATISTICS")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info("  %-30s %s", k, v)

    # Save stats as JSON
    stats_path = save_dir / f"mind2web_{args.split}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Stats saved to %s", stats_path)

    # ------------------------------------------------------------------
    # Print sample details
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("SAMPLE DETAILS (first %d)", min(3, n))
    logger.info("=" * 60)
    for s in list(ds)[:3]:
        logger.info("  sample_id: %s", s.sample_id)
        logger.info("  instruction: %s", s.instruction[:100])
        logger.info("  action_type: %s", s.action_type)
        logger.info("  target_bbox: %s", s.target_bbox.as_tuple() if s.target_bbox else None)
        logger.info("  click_point: %s", s.click_point)
        logger.info("  target_element_id: %s", s.target_element_id)
        logger.info("  website: %s, domain: %s", s.website, s.domain)
        logger.info("  dom_candidates: %d", len(s.dom_candidates) if s.dom_candidates else 0)
        meta = s.metadata
        logger.info("  target_action_reprs: %s", meta.get("target_action_reprs", ""))
        logger.info("  ---")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    from gui_grounding.utils.visualization import draw_prediction, save_visualization

    vis_count = 0
    vis_dir = save_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for s in ds:
        if vis_count >= args.num_vis:
            break
        img = ds.get_screenshot(s)
        if img is None:
            continue

        gt_bbox = s.target_bbox.as_tuple() if s.target_bbox else None
        gt_point = s.click_point

        annotated = draw_prediction(
            img,
            gt_bbox=gt_bbox,
            gt_point=gt_point,
            action_type=s.action_type,
        )

        uid = s.metadata.get("action_uid", s.sample_id)
        out_path = vis_dir / f"{uid}.png"
        save_visualization(annotated, out_path)
        logger.info("Saved visualization: %s", out_path)
        vis_count += 1

    logger.info("Saved %d visualizations to %s", vis_count, vis_dir)

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------
    summary_path = save_dir / f"mind2web_{args.split}_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Mind2Web Inspection — {args.split}\n\n")
        f.write(f"- **Samples loaded**: {n}\n")
        f.write(f"- **Has bbox**: {has_bbox}/{n}\n")
        f.write(f"- **Has click point**: {has_click}/{n}\n")
        f.write(f"- **Has element ID**: {has_element_id}/{n}\n")
        f.write(f"- **Action types**: {dict(action_types)}\n")
        f.write(f"- **Unique websites**: {len(websites)}\n")
        f.write(f"- **Unique domains**: {len(domains)}\n\n")
        f.write(f"See `{stats_path.name}` for full stats and ")
        f.write(f"`visualizations/` for annotated screenshots.\n")
    logger.info("Summary saved to %s", summary_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
