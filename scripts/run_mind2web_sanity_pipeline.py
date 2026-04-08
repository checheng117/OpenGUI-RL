#!/usr/bin/env python3
"""Minimal end-to-end sanity pipeline on real Mind2Web data.

Loads a small batch of real samples, runs dummy candidate generation,
scores with verifiable reward, and computes example metrics.

**No model weights required.** Candidates are dummy/heuristic-based.

Usage:
    python scripts/run_mind2web_sanity_pipeline.py --max-samples 10
    python scripts/run_mind2web_sanity_pipeline.py --split test_task --max-samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed

logger = get_logger("mind2web_sanity_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mind2Web sanity pipeline")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--num-candidates", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/sanity_pipeline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load real data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Load real Mind2Web data")
    logger.info("=" * 60)

    from gui_grounding.data.mind2web_dataset import Mind2WebDataset

    ds = Mind2WebDataset(
        split=args.split,
        max_samples=args.max_samples,
        cache_screenshots=True,
    )
    logger.info("Loaded %d real samples from Mind2Web (split=%s)", len(ds), args.split)

    if len(ds) == 0:
        logger.error("No samples loaded. Cannot continue.")
        return

    # ------------------------------------------------------------------
    # 2. Candidate generation (dummy mode — NOT model-based)
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Generate dummy candidates per sample")
    logger.info("  (These are random perturbations, NOT model predictions)")
    logger.info("=" * 60)

    from gui_grounding.reward.candidate_generator import CandidateGenerator

    gen = CandidateGenerator(mode="dummy", num_candidates=args.num_candidates, seed=args.seed)

    all_candidates = {}
    for sample in ds:
        candidates = gen.generate(sample)
        all_candidates[sample.sample_id] = candidates

    logger.info("Generated %d candidates for each of %d samples.", args.num_candidates, len(ds))

    # ------------------------------------------------------------------
    # 3. Verifiable reward scoring
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Score candidates with verifiable reward")
    logger.info("=" * 60)

    from gui_grounding.reward.verifiable_reward import VerifiableRewardCalculator

    calc = VerifiableRewardCalculator()
    reward_details = []

    for sample in ds:
        gt_bbox = sample.target_bbox.as_tuple() if sample.target_bbox else None
        for cand in all_candidates[sample.sample_id]:
            result = calc.compute(
                sample_id=cand.candidate_id,
                pred_bbox=cand.bbox.as_tuple() if cand.bbox else None,
                gt_bbox=gt_bbox,
                pred_click=cand.click_point,
                pred_action=cand.action_type,
                gt_action=sample.action_type,
                pred_element_id=cand.element_id,
                gt_element_id=sample.target_element_id,
            )
            reward_details.append({
                "sample_id": sample.sample_id,
                "candidate_id": cand.candidate_id,
                "total_reward": result.total_reward,
                "elem": result.components.element_correct,
                "iou": result.components.iou,
                "click": result.components.click_inside_target,
                "act": result.components.action_type_correct,
                "penalty": result.components.invalid_format_penalty,
            })

    # Print a few reward results
    logger.info("Reward sample (first 3 samples, best candidate each):")
    seen = set()
    for d in sorted(reward_details, key=lambda x: -x["total_reward"]):
        sid = d["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        logger.info(
            "  %s: best_reward=%.4f (elem=%.0f iou=%.3f click=%.0f act=%.0f)",
            sid[:50], d["total_reward"],
            d["elem"], d["iou"], d["click"], d["act"],
        )
        if len(seen) >= 3:
            break

    # ------------------------------------------------------------------
    # 4. Metrics on dummy predictions (first candidate as "prediction")
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Example metrics (first candidate = dummy prediction)")
    logger.info("  NOTE: These are NOT real model predictions.")
    logger.info("=" * 60)

    from gui_grounding.evaluation.metrics import compute_all_metrics

    pred_element_ids = []
    gt_element_ids = []
    pred_bboxes = []
    gt_bboxes = []
    pred_points = []
    pred_actions = []
    gt_actions = []

    for sample in ds:
        candidates = all_candidates[sample.sample_id]
        first_cand = candidates[0] if candidates else None

        gt_element_ids.append(sample.target_element_id)
        gt_bboxes.append(sample.target_bbox.as_tuple() if sample.target_bbox else None)
        gt_actions.append(sample.action_type)

        if first_cand:
            pred_element_ids.append(first_cand.element_id)
            pred_bboxes.append(first_cand.bbox.as_tuple() if first_cand.bbox else None)
            pred_points.append(first_cand.click_point)
            pred_actions.append(first_cand.action_type)
        else:
            pred_element_ids.append(None)
            pred_bboxes.append(None)
            pred_points.append(None)
            pred_actions.append(None)

    metrics = compute_all_metrics(
        pred_element_ids=pred_element_ids,
        gt_element_ids=gt_element_ids,
        pred_bboxes=pred_bboxes,
        gt_bboxes=gt_bboxes,
        pred_points=pred_points,
        pred_actions=pred_actions,
        gt_actions=gt_actions,
    )

    logger.info("Dummy-prediction metrics (NOT real model results):")
    for k, v in metrics.items():
        logger.info("  %-25s %.4f", k, v)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results = {
        "pipeline": "mind2web_sanity",
        "split": args.split,
        "num_samples": len(ds),
        "num_candidates_per_sample": args.num_candidates,
        "candidate_mode": "dummy (random perturbation — NOT model-based)",
        "metrics_disclaimer": "Metrics below are from dummy candidates, NOT real model predictions.",
        "metrics": metrics,
    }
    results_path = out_dir / f"sanity_{args.split}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", results_path)
    logger.info("Done.")


def _exit_cleanly(exit_code: int = 0) -> "NoReturn":
    """Legacy workaround for Python 3.13 finalization crashes."""
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
    if os.getenv("GUI_GROUNDING_LEGACY_HARD_EXIT", "0") == "1":
        _exit_cleanly(0)
