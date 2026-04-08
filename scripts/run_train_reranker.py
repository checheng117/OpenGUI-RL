#!/usr/bin/env python3
"""Stage B: Train the reward-based reranker.

Usage:
    python scripts/run_train_reranker.py --config configs/train/rerank_reward.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import NoReturn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.utils.config import load_config
from gui_grounding.utils.io import save_json
from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed

logger = get_logger("run_train_reranker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reward-based reranker")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass


def main() -> None:
    _bootstrap_env()
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    logger.info("Loaded config: %s", args.config)

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    if args.dry_run:
        logger.info("Dry run — config loaded successfully. Exiting.")
        return

    logger.info("=" * 60)
    logger.info("Reranker Training Pipeline")
    logger.info("=" * 60)

    from gui_grounding.models.candidate_scorer import CandidateScorer
    from gui_grounding.training.trainer_reranker import RerankerTrainer

    model_cfg = cfg.get("reranker_model", {})
    scorer = CandidateScorer(
        scoring_mode="learned",
        input_dim=int(model_cfg.get("input_dim", 17)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    trainer = RerankerTrainer(
        scorer=scorer,
        train_candidates_path=data_cfg.get(
            "candidates_path",
            "outputs/candidate_generation_clip_grid/candidates_train.jsonl",
        ),
        output_dir=training_cfg.get("output_dir", "outputs/reranker"),
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        num_epochs=training_cfg.get("num_epochs", 5),
        batch_size=training_cfg.get("batch_size", 64),
        margin=training_cfg.get("margin", 0.0),
        val_ratio=training_cfg.get("val_ratio", 0.2),
        min_reward_diff=training_cfg.get("min_reward_diff", 1e-6),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        num_workers=training_cfg.get("num_workers", 0),
        device=training_cfg.get("device", "auto"),
        optimization_mode=training_cfg.get("optimization_mode", "pairwise"),
        dpo_beta=training_cfg.get("dpo_beta", 0.1),
        policy_init_checkpoint=training_cfg.get("policy_init_checkpoint", None),
        reference_checkpoint=training_cfg.get("reference_checkpoint", None),
        step5c_baseline_summary_path=training_cfg.get("step5c_baseline_summary_path", None),
        step6a_baseline_summary_path=training_cfg.get("step6a_baseline_summary_path", None),
        export_preference_pairs=training_cfg.get("export_preference_pairs", True),
        pair_construction_mode=training_cfg.get("pair_construction_mode", "all_pairs"),
        pair_weight_mode=training_cfg.get("pair_weight_mode", "uniform"),
        sample_split_mode=training_cfg.get("sample_split_mode", "random"),
        feature_include_structured_relative_support=training_cfg.get(
            "feature_include_structured_relative_support",
            True,
        ),
        pair_reward_gap_threshold=training_cfg.get("pair_reward_gap_threshold", 0.0),
        pair_weight_alpha=training_cfg.get("pair_weight_alpha", 2.0),
        pair_weight_cap=training_cfg.get("pair_weight_cap", 5.0),
        pair_source_decoy_max_sources=training_cfg.get("pair_source_decoy_max_sources", 3),
        pair_recovery_anchor_weight=training_cfg.get("pair_recovery_anchor_weight", 1.5),
        pair_positive_ranking_weight=training_cfg.get("pair_positive_ranking_weight", 1.25),
        pair_source_decoy_weight=training_cfg.get("pair_source_decoy_weight", 1.1),
        pair_same_source_decoy_weight=training_cfg.get("pair_same_source_decoy_weight", 1.15),
        pair_cross_source_bonus=training_cfg.get("pair_cross_source_bonus", 0.1),
        pair_source_prior_bonus=training_cfg.get("pair_source_prior_bonus", 0.25),
        pair_pool_gap_bonus=training_cfg.get("pair_pool_gap_bonus", 0.0),
        pair_rare_source_bonus=training_cfg.get("pair_rare_source_bonus", 0.0),
        pair_rare_signature_bonus=training_cfg.get("pair_rare_signature_bonus", 0.0),
        pair_negative_strength_bonus=training_cfg.get("pair_negative_strength_bonus", 0.0),
        pair_point_first_bonus=training_cfg.get("pair_point_first_bonus", 0.0),
        pair_point_first_support_anchor_weight=training_cfg.get("pair_point_first_support_anchor_weight", 1.0),
        pair_disagreement_bonus=training_cfg.get("pair_disagreement_bonus", 0.0),
        pair_positive_signal_bonus=training_cfg.get("pair_positive_signal_bonus", 0.0),
        pair_point_first_all_structured_decoys=training_cfg.get("pair_point_first_all_structured_decoys", False),
        pair_conditional_singleton_bonus=training_cfg.get("pair_conditional_singleton_bonus", 0.0),
        pair_point_first_signal_threshold=training_cfg.get("pair_point_first_signal_threshold", 0.0),
        pair_point_first_gap_threshold=training_cfg.get("pair_point_first_gap_threshold", 0.0),
        pair_structured_singleton_signal_threshold=training_cfg.get(
            "pair_structured_singleton_signal_threshold",
            0.0,
        ),
        pair_structured_singleton_gap_threshold=training_cfg.get(
            "pair_structured_singleton_gap_threshold",
            0.0,
        ),
        pair_structured_singleton_decoy_weight=training_cfg.get(
            "pair_structured_singleton_decoy_weight",
            1.0,
        ),
        pair_structured_singleton_support_anchor_weight=training_cfg.get(
            "pair_structured_singleton_support_anchor_weight",
            1.0,
        ),
        checkpoint_selection_mode=training_cfg.get("checkpoint_selection_mode", "full_pool_reward_gain"),
        selection_drop_sources=training_cfg.get("selection_drop_sources", []),
        sample_split_protected_sources=training_cfg.get("sample_split_protected_sources", []),
        seed=seed,
    )

    result = trainer.train()
    evaluation_cfg = cfg.get("evaluation", {})
    candidate_paths = evaluation_cfg.get("candidate_paths", {}) or {}
    if candidate_paths:
        split_results = {}
        for split_name, candidate_path in candidate_paths.items():
            split_results[split_name] = trainer.evaluate_candidate_file(
                candidate_path,
                split_name=split_name,
                save_artifacts=True,
            )
        result["official_split_evaluations"] = split_results
        output_dir = Path(training_cfg.get("output_dir", "outputs/reranker"))
        save_json(split_results, output_dir / "official_split_evaluations.json")
        save_json(result, output_dir / "evaluation_summary.json")
    logger.info("Training result: %s", result)
    logger.info("Done.")


def _exit_cleanly(exit_code: int = 0) -> NoReturn:
    """Legacy workaround for Python 3.13 finalization crashes."""
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
    if os.getenv("GUI_GROUNDING_LEGACY_HARD_EXIT", "0") == "1":
        _exit_cleanly(0)
