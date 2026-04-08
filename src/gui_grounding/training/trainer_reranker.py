"""Learned reranker trainer using reward-derived pairwise supervision."""

from __future__ import annotations

import copy
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from gui_grounding.utils.io import load_json, load_jsonl, save_json, save_jsonl
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class PairwiseCandidateDataset(Dataset):
    """Pairwise preference dataset where pairs come from the same sample."""

    def __init__(self, pairs: list[dict[str, Any]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.pairs[idx]


class RerankerTrainer:
    """Train/evaluate a learned reranker over CLIP-grid candidate pools."""

    def __init__(
        self,
        scorer: Any,
        train_candidates_path: str | Path,
        output_dir: str | Path = "outputs/reranker",
        learning_rate: float = 1e-3,
        num_epochs: int = 20,
        batch_size: int = 64,
        margin: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.2,
        min_reward_diff: float = 1e-6,
        weight_decay: float = 0.0,
        num_workers: int = 0,
        device: str = "cpu",
        optimization_mode: str = "pairwise",
        dpo_beta: float = 0.1,
        policy_init_checkpoint: str | Path | None = None,
        reference_checkpoint: str | Path | None = None,
        step5c_baseline_summary_path: str | Path | None = None,
        step6a_baseline_summary_path: str | Path | None = None,
        export_preference_pairs: bool = True,
        pair_construction_mode: str = "all_pairs",
        pair_weight_mode: str = "uniform",
        sample_split_mode: str = "random",
        feature_include_structured_relative_support: bool = True,
        pair_reward_gap_threshold: float = 0.0,
        pair_weight_alpha: float = 2.0,
        pair_weight_cap: float = 5.0,
        pair_source_decoy_max_sources: int = 3,
        pair_recovery_anchor_weight: float = 1.5,
        pair_positive_ranking_weight: float = 1.25,
        pair_source_decoy_weight: float = 1.1,
        pair_same_source_decoy_weight: float = 1.15,
        pair_cross_source_bonus: float = 0.1,
        pair_source_prior_bonus: float = 0.25,
        pair_pool_gap_bonus: float = 0.0,
        pair_rare_source_bonus: float = 0.0,
        pair_rare_signature_bonus: float = 0.0,
        pair_negative_strength_bonus: float = 0.0,
        pair_point_first_bonus: float = 0.0,
        pair_point_first_support_anchor_weight: float = 1.0,
        pair_disagreement_bonus: float = 0.0,
        pair_positive_signal_bonus: float = 0.0,
        pair_point_first_all_structured_decoys: bool = False,
        pair_conditional_singleton_bonus: float = 0.0,
        pair_point_first_signal_threshold: float = 0.0,
        pair_point_first_gap_threshold: float = 0.0,
        pair_structured_singleton_signal_threshold: float = 0.0,
        pair_structured_singleton_gap_threshold: float = 0.0,
        pair_structured_singleton_decoy_weight: float = 1.0,
        pair_structured_singleton_support_anchor_weight: float = 1.0,
        checkpoint_selection_mode: str = "full_pool_reward_gain",
        selection_drop_sources: list[str] | None = None,
        sample_split_protected_sources: list[str] | None = None,
    ) -> None:
        assert optimization_mode in ("pairwise", "dpo_style"), f"Unknown optimization mode: {optimization_mode}"
        assert pair_construction_mode in (
            "all_pairs",
            "headroom_hard_negative",
            "recovery_source_aware",
            "rare_recovery_targeted",
            "conditional_singleton_recovery",
        ), (
            f"Unknown pair_construction_mode: {pair_construction_mode}"
        )
        assert pair_weight_mode in ("uniform", "reward_gap", "source_aware_recovery", "rare_recovery_targeted"), (
            f"Unknown pair_weight_mode: {pair_weight_mode}"
        )
        assert sample_split_mode in ("random", "headroom_source_stratified"), (
            f"Unknown sample_split_mode: {sample_split_mode}"
        )
        assert checkpoint_selection_mode in (
            "full_pool_reward_gain",
            "headroom_subset_reward_gain",
            "headroom_then_full",
        ), f"Unknown checkpoint_selection_mode: {checkpoint_selection_mode}"
        self.scorer = scorer
        self.train_candidates_path = Path(train_candidates_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.margin = margin
        self.seed = seed
        self.val_ratio = val_ratio
        self.min_reward_diff = min_reward_diff
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.optimization_mode = optimization_mode
        self.dpo_beta = float(dpo_beta)
        self.policy_init_checkpoint = Path(policy_init_checkpoint) if policy_init_checkpoint else None
        self.reference_checkpoint = Path(reference_checkpoint) if reference_checkpoint else None
        self.step5c_baseline_summary_path = (
            Path(step5c_baseline_summary_path) if step5c_baseline_summary_path else None
        )
        self.step6a_baseline_summary_path = (
            Path(step6a_baseline_summary_path) if step6a_baseline_summary_path else None
        )
        self.export_preference_pairs = export_preference_pairs
        self.pair_construction_mode = pair_construction_mode
        self.pair_weight_mode = pair_weight_mode
        self.sample_split_mode = sample_split_mode
        self.feature_include_structured_relative_support = bool(feature_include_structured_relative_support)
        self.pair_reward_gap_threshold = float(pair_reward_gap_threshold)
        self.pair_weight_alpha = float(pair_weight_alpha)
        self.pair_weight_cap = float(pair_weight_cap)
        self.pair_source_decoy_max_sources = max(int(pair_source_decoy_max_sources), 0)
        self.pair_recovery_anchor_weight = float(pair_recovery_anchor_weight)
        self.pair_positive_ranking_weight = float(pair_positive_ranking_weight)
        self.pair_source_decoy_weight = float(pair_source_decoy_weight)
        self.pair_same_source_decoy_weight = float(pair_same_source_decoy_weight)
        self.pair_cross_source_bonus = float(pair_cross_source_bonus)
        self.pair_source_prior_bonus = float(pair_source_prior_bonus)
        self.pair_pool_gap_bonus = float(pair_pool_gap_bonus)
        self.pair_rare_source_bonus = float(pair_rare_source_bonus)
        self.pair_rare_signature_bonus = float(pair_rare_signature_bonus)
        self.pair_negative_strength_bonus = float(pair_negative_strength_bonus)
        self.pair_point_first_bonus = float(pair_point_first_bonus)
        self.pair_point_first_support_anchor_weight = float(pair_point_first_support_anchor_weight)
        self.pair_disagreement_bonus = float(pair_disagreement_bonus)
        self.pair_positive_signal_bonus = float(pair_positive_signal_bonus)
        self.pair_point_first_all_structured_decoys = bool(pair_point_first_all_structured_decoys)
        self.pair_conditional_singleton_bonus = float(pair_conditional_singleton_bonus)
        self.pair_point_first_signal_threshold = float(pair_point_first_signal_threshold)
        self.pair_point_first_gap_threshold = float(pair_point_first_gap_threshold)
        self.pair_structured_singleton_signal_threshold = float(pair_structured_singleton_signal_threshold)
        self.pair_structured_singleton_gap_threshold = float(pair_structured_singleton_gap_threshold)
        self.pair_structured_singleton_decoy_weight = float(pair_structured_singleton_decoy_weight)
        self.pair_structured_singleton_support_anchor_weight = float(
            pair_structured_singleton_support_anchor_weight
        )
        self.checkpoint_selection_mode = checkpoint_selection_mode
        self.selection_drop_sources = {self._normalize_source_name(source) for source in (selection_drop_sources or [])}
        self.sample_split_protected_sources = {
            self._normalize_source_name(source) for source in (sample_split_protected_sources or [])
        }
        self.source_recovery_priors: dict[str, float] = {}
        self.recovery_source_counts: dict[str, int] = {}
        self.recovery_signature_counts: dict[str, int] = {}
        self.reference_scorer: Any | None = None
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scorer = self.scorer.to(self.device)
        if self.policy_init_checkpoint is not None:
            self._load_checkpoint(self.scorer, self.policy_init_checkpoint)
        self.optimizer = AdamW(self.scorer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.rng = random.Random(seed)
        logger.info(
            "RerankerTrainer initialized: data=%s output=%s device=%s mode=%s pair_mode=%s weight_mode=%s",
            self.train_candidates_path,
            self.output_dir,
            self.device,
            self.optimization_mode,
            self.pair_construction_mode,
            self.pair_weight_mode,
        )

    @staticmethod
    def _candidate_priority(candidate: dict[str, Any], fallback_idx: int) -> tuple[float, float, float, float]:
        def _safe_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        return (
            _safe_float(candidate.get("score", 0.0)),
            _safe_float(candidate.get("confidence", 0.0)),
            _safe_float(candidate.get("joint_log_prob", 0.0)),
            -_safe_float(candidate.get("rank", fallback_idx + 1)),
        )

    @staticmethod
    def _load_checkpoint(model: Any, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        load_result = model.load_state_dict(state, strict=False)
        logger.info(
            "Loaded checkpoint into model: %s (missing=%d unexpected=%d)",
            path,
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
        )

    def _init_reference_scorer_if_needed(self) -> None:
        if self.optimization_mode != "dpo_style":
            return
        self.reference_scorer = copy.deepcopy(self.scorer).to(self.device)
        ref_path = self.reference_checkpoint or self.policy_init_checkpoint
        if ref_path is not None:
            self._load_checkpoint(self.reference_scorer, ref_path)
        self.reference_scorer.eval()
        for p in self.reference_scorer.parameters():
            p.requires_grad = False

    @staticmethod
    def _stats(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 1.0
        mean = sum(values) / len(values)
        var = sum((v - mean) * (v - mean) for v in values) / max(len(values), 1)
        std = math.sqrt(max(var, 1e-12))
        return max(values), mean, std

    @staticmethod
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

    @staticmethod
    def _rank_bucket(rank: int) -> str:
        rank = int(rank)
        if rank <= 2:
            return "rank_2"
        if rank <= 4:
            return "rank_3_4"
        if rank <= 6:
            return "rank_5_6"
        return "rank_7_8"

    @staticmethod
    def _positive_count_bucket(count: int) -> str:
        count = int(count)
        if count <= 1:
            return "singleton"
        if count == 2:
            return "double"
        return "multi3+"

    def _recovery_signature(
        self,
        candidate: dict[str, Any],
        candidate_rank: int,
        positive_count: int,
    ) -> str:
        provenance = candidate.get("provenance") or {}
        extra_provenance = provenance.get("extra_provenance") or {}
        parser_metadata = candidate.get("parser_metadata") or {}
        return "|".join(
            [
                self._normalize_source_name(candidate.get("source")),
                self._rank_bucket(candidate_rank),
                self._positive_count_bucket(positive_count),
                "point_first" if provenance.get("point_first_prompt") else "non_point_first",
                "decoupled" if provenance.get("decoupled_point_native_decode") else "coupled",
                str(parser_metadata.get("resolved_click_mode") or "none"),
                str(parser_metadata.get("resolved_bbox_mode") or "none"),
                "bbox_reconciled"
                if bool((extra_provenance.get("bbox_reconciliation") or {}).get("applied"))
                else "bbox_native",
            ]
        )

    def _headroom_recovery_info(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        candidates = sample.get("candidates", [])
        if len(candidates) < 2:
            return None
        rewards = [float(c.get("reward", {}).get("total_reward", 0.0)) for c in candidates]
        baseline_idx = 0
        baseline_reward = rewards[baseline_idx]
        positive_indices = [
            idx for idx, reward in enumerate(rewards) if idx != baseline_idx and reward > baseline_reward + self.min_reward_diff
        ]
        if not positive_indices:
            return None
        best_idx = max(
            positive_indices,
            key=lambda idx: (rewards[idx], *self._candidate_priority(candidates[idx], idx)),
        )
        best_candidate = candidates[best_idx]
        return {
            "baseline_idx": baseline_idx,
            "baseline_reward": baseline_reward,
            "positive_indices": positive_indices,
            "best_idx": best_idx,
            "best_source": self._normalize_source_name(best_candidate.get("source")),
            "best_signature": self._recovery_signature(best_candidate, best_idx + 1, len(positive_indices)),
            "pool_oracle_gap": float(rewards[best_idx] - baseline_reward),
        }

    def _negative_decoy_priority(
        self,
        candidate: dict[str, Any],
        reward: float,
        fallback_idx: int,
    ) -> tuple[float, float, float, float, float, float, float]:
        components = candidate.get("reward", {}).get("components", {}) or {}
        dom_match = candidate.get("dom_match") or {}
        parseable = 1.0 if (candidate.get("structured_output_diagnostics") or {}).get("json_parse_success") is not False else 0.0
        base_priority = self._candidate_priority(candidate, fallback_idx)
        return (
            float(reward),
            float(components.get("click_inside_target", 0.0)),
            float(components.get("iou", 0.0)),
            float(dom_match.get("best_iou", 0.0)),
            parseable,
            base_priority[2],
            base_priority[3],
        )

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _candidate_bbox(cls, candidate: dict[str, Any]) -> list[float]:
        bbox = candidate.get("bbox_proposal") or candidate.get("bbox") or [0.0, 0.0, 1.0, 1.0]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return [0.0, 0.0, 1.0, 1.0]
        return [cls._safe_float(v, 0.0) for v in bbox]

    @classmethod
    def _candidate_click(cls, candidate: dict[str, Any]) -> list[float]:
        click = candidate.get("click_point") or [0.0, 0.0]
        if not isinstance(click, list) or len(click) != 2:
            return [0.0, 0.0]
        return [cls._safe_float(v, 0.0) for v in click]

    @staticmethod
    def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
        if len(box_a) != 4 or len(box_b) != 4:
            return 0.0
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0.0 else 0.0

    @staticmethod
    def _click_distance(click_a: list[float], click_b: list[float]) -> float:
        if len(click_a) != 2 or len(click_b) != 2:
            return 0.0
        dx = click_a[0] - click_b[0]
        dy = click_a[1] - click_b[1]
        return math.sqrt(dx * dx + dy * dy)

    def _sample_image_size(self, sample: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[float, float, float]:
        width = self._safe_float(sample.get("image_width"), 0.0)
        height = self._safe_float(sample.get("image_height"), 0.0)
        if width <= 0.0:
            width = max((self._candidate_bbox(candidate)[2] for candidate in candidates), default=1.0)
        if height <= 0.0:
            height = max((self._candidate_bbox(candidate)[3] for candidate in candidates), default=1.0)
        width = max(width, 1.0)
        height = max(height, 1.0)
        image_diag = max(math.sqrt(width * width + height * height), 1.0)
        return width, height, image_diag

    def _candidate_positive_signal(self, candidate: dict[str, Any]) -> float:
        components = candidate.get("reward", {}).get("components", {}) or {}
        return self._safe_float(components.get("click_inside_target"), 0.0) + self._safe_float(
            components.get("iou"),
            0.0,
        )

    def _point_first_disagreement_support_info(
        self,
        sample: dict[str, Any],
        recovery_info: dict[str, Any] | None,
        normalized_sources: list[str],
    ) -> dict[str, Any] | None:
        if recovery_info is None:
            return None
        best_idx = int(recovery_info["best_idx"])
        positive_indices = set(int(idx) for idx in recovery_info["positive_indices"])
        if recovery_info["best_source"] != "point_first_sampled_t0p7":
            return None
        if len(positive_indices) != 1:
            return None
        if best_idx + 1 < 7:
            return None

        candidates = sample.get("candidates", [])
        _, _, image_diag = self._sample_image_size(sample, candidates)
        best_bbox = self._candidate_bbox(candidates[best_idx])
        best_click = self._candidate_click(candidates[best_idx])

        structured_disagreement_indices: list[int] = []
        support_anchor_by_source: dict[str, tuple[int, tuple[float, float, float, float, float, float, float]]] = {}
        max_iou_to_any = 0.0
        min_click_distance_to_structured = image_diag
        tight_support_count = 0

        for idx, candidate in enumerate(candidates):
            if idx == best_idx:
                continue
            bbox_iou = self._bbox_iou(best_bbox, self._candidate_bbox(candidate))
            click_distance = self._click_distance(best_click, self._candidate_click(candidate))
            max_iou_to_any = max(max_iou_to_any, bbox_iou)
            if bbox_iou >= 0.5 or click_distance <= 48.0:
                tight_support_count += 1
            source = normalized_sources[idx]
            if idx in positive_indices or idx == int(recovery_info["baseline_idx"]):
                continue
            if source == "structured_sampled_t0p6":
                min_click_distance_to_structured = min(min_click_distance_to_structured, click_distance)
                structured_disagreement_indices.append(idx)
                continue
            if source in {"hybrid_point_structured", "point_native_primary", "point_first_structured"}:
                priority = self._negative_decoy_priority(
                    candidate,
                    float(candidate.get("reward", {}).get("total_reward", 0.0)),
                    idx,
                )
                prev = support_anchor_by_source.get(source)
                if prev is None or priority > prev[1]:
                    support_anchor_by_source[source] = (idx, priority)

        best_positive_signal = self._candidate_positive_signal(candidates[best_idx])
        is_targeted = (
            len(structured_disagreement_indices) >= 3
            and max_iou_to_any <= 1e-9
            and min_click_distance_to_structured >= 80.0
            and best_positive_signal >= self.pair_point_first_signal_threshold
            and float(recovery_info["pool_oracle_gap"]) >= self.pair_point_first_gap_threshold
        )
        return {
            "is_targeted": is_targeted,
            "structured_disagreement_indices": structured_disagreement_indices,
            "support_anchor_indices": [
                idx
                for _, (idx, _) in sorted(
                    support_anchor_by_source.items(),
                    key=lambda item: item[1][1],
                    reverse=True,
                )
            ],
            "tight_support_count": tight_support_count,
            "max_iou_to_any": float(max_iou_to_any),
            "min_click_distance_to_structured_norm": float(min_click_distance_to_structured / image_diag),
            "best_positive_signal": float(best_positive_signal),
        }

    def _structured_singleton_disambiguation_info(
        self,
        sample: dict[str, Any],
        recovery_info: dict[str, Any] | None,
        normalized_sources: list[str],
    ) -> dict[str, Any] | None:
        if recovery_info is None:
            return None
        best_idx = int(recovery_info["best_idx"])
        positive_indices = set(int(idx) for idx in recovery_info["positive_indices"])
        if recovery_info["best_source"] != "structured_sampled_t0p6":
            return None
        if len(positive_indices) != 1:
            return None
        if best_idx + 1 < 3 or best_idx + 1 > 4:
            return None

        candidates = sample.get("candidates", [])
        _, _, image_diag = self._sample_image_size(sample, candidates)
        best_bbox = self._candidate_bbox(candidates[best_idx])
        best_click = self._candidate_click(candidates[best_idx])

        same_source_decoys: list[tuple[int, tuple[float, float, float, float, float, float, float]]] = []
        support_anchor_by_source: dict[str, tuple[int, tuple[float, float, float, float, float, float, float]]] = {}
        max_same_source_iou = 0.0
        min_same_source_click_distance = image_diag
        tight_support_count = 0

        for idx, candidate in enumerate(candidates):
            if idx == best_idx:
                continue
            bbox_iou = self._bbox_iou(best_bbox, self._candidate_bbox(candidate))
            click_distance = self._click_distance(best_click, self._candidate_click(candidate))
            if bbox_iou >= 0.5 or click_distance <= 48.0:
                tight_support_count += 1
            source = normalized_sources[idx]
            if idx in positive_indices or idx == int(recovery_info["baseline_idx"]):
                continue
            if source == "structured_sampled_t0p6":
                same_source_decoys.append(
                    (
                        idx,
                        self._negative_decoy_priority(
                            candidate,
                            float(candidate.get("reward", {}).get("total_reward", 0.0)),
                            idx,
                        ),
                    )
                )
                max_same_source_iou = max(max_same_source_iou, bbox_iou)
                min_same_source_click_distance = min(min_same_source_click_distance, click_distance)
                continue
            if source in {"hybrid_point_structured", "point_native_primary", "point_first_structured"}:
                priority = self._negative_decoy_priority(
                    candidate,
                    float(candidate.get("reward", {}).get("total_reward", 0.0)),
                    idx,
                )
                prev = support_anchor_by_source.get(source)
                if prev is None or priority > prev[1]:
                    support_anchor_by_source[source] = (idx, priority)

        best_positive_signal = self._candidate_positive_signal(candidates[best_idx])
        is_targeted = (
            len(same_source_decoys) >= 2
            and best_positive_signal >= self.pair_structured_singleton_signal_threshold
            and float(recovery_info["pool_oracle_gap"]) >= self.pair_structured_singleton_gap_threshold
            and max_same_source_iou <= 0.05
            and min_same_source_click_distance >= 80.0
        )
        return {
            "is_targeted": is_targeted,
            "same_source_decoy_indices": [
                idx
                for idx, _ in sorted(
                    same_source_decoys,
                    key=lambda item: item[1],
                    reverse=True,
                )
            ],
            "support_anchor_indices": [
                idx
                for _, (idx, _) in sorted(
                    support_anchor_by_source.items(),
                    key=lambda item: item[1][1],
                    reverse=True,
                )
            ],
            "best_positive_signal": float(best_positive_signal),
            "tight_support_count": tight_support_count,
            "max_same_source_iou": float(max_same_source_iou),
            "min_same_source_click_distance_norm": float(min_same_source_click_distance / image_diag),
        }

    def _isolated_point_first_false_alarm_indices(
        self,
        sample: dict[str, Any],
        normalized_sources: list[str],
        rewards: list[float],
        baseline_reward: float,
    ) -> list[int]:
        candidates = sample.get("candidates", [])
        _, _, image_diag = self._sample_image_size(sample, candidates)
        indices: list[int] = []
        structured_indices = [
            idx for idx, source in enumerate(normalized_sources) if source == "structured_sampled_t0p6"
        ]
        if len(structured_indices) < 3:
            return indices
        for idx, candidate in enumerate(candidates):
            if normalized_sources[idx] != "point_first_sampled_t0p7":
                continue
            if rewards[idx] > baseline_reward + self.min_reward_diff:
                continue
            if idx + 1 < 7:
                continue
            point_bbox = self._candidate_bbox(candidate)
            point_click = self._candidate_click(candidate)
            max_iou_to_any = 0.0
            min_click_distance_to_structured = image_diag
            for other_idx, other_candidate in enumerate(candidates):
                if other_idx == idx:
                    continue
                bbox_iou = self._bbox_iou(point_bbox, self._candidate_bbox(other_candidate))
                click_distance = self._click_distance(point_click, self._candidate_click(other_candidate))
                max_iou_to_any = max(max_iou_to_any, bbox_iou)
                if other_idx in structured_indices:
                    min_click_distance_to_structured = min(min_click_distance_to_structured, click_distance)
            if max_iou_to_any <= 1e-9 and min_click_distance_to_structured >= 80.0:
                indices.append(idx)
        return indices

    def _build_feature_rows(self, sample: dict[str, Any]) -> list[list[float]]:
        candidates = sample.get("candidates", [])
        n = len(candidates)
        if n == 0:
            return []

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _candidate_bbox(candidate: dict[str, Any]) -> list[float]:
            bbox = candidate.get("bbox_proposal") or candidate.get("bbox") or [0.0, 0.0, 1.0, 1.0]
            if not isinstance(bbox, list) or len(bbox) != 4:
                return [0.0, 0.0, 1.0, 1.0]
            return [_safe_float(v, 0.0) for v in bbox]

        def _candidate_click(candidate: dict[str, Any]) -> list[float]:
            click = candidate.get("click_point") or [0.0, 0.0]
            if not isinstance(click, list) or len(click) != 2:
                return [0.0, 0.0]
            return [_safe_float(v, 0.0) for v in click]

        def _normalize_source(source: str) -> str:
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

        def _source_bucket(source: str) -> str:
            normalized_source = _normalize_source(source)
            if normalized_source == "stagea_first_choice":
                return "first_choice"
            if normalized_source in {"point_first_structured", "point_first_sampled_t0p7"}:
                return "point_first"
            if normalized_source == "point_native_primary":
                return "point_native"
            if normalized_source == "hybrid_point_structured":
                return "hybrid"
            if normalized_source == "structured_sampled_t0p6":
                return "sampled"
            if normalized_source == "legacy_clip_grid":
                return "legacy_clip"
            return "other"

        def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
            if len(box_a) != 4 or len(box_b) != 4:
                return 0.0
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
            area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
            denom = area_a + area_b - inter
            return inter / denom if denom > 0.0 else 0.0

        def _click_distance(click_a: list[float], click_b: list[float]) -> float:
            if len(click_a) != 2 or len(click_b) != 2:
                return 0.0
            dx = click_a[0] - click_b[0]
            dy = click_a[1] - click_b[1]
            return math.sqrt(dx * dx + dy * dy)

        scores = [_safe_float(c.get("score", 0.0), 0.0) for c in candidates]
        confs = [_safe_float(c.get("confidence", 0.0), 0.0) for c in candidates]
        jlps = [_safe_float(c.get("joint_log_prob", 0.0), 0.0) for c in candidates]
        alps = [_safe_float(c.get("action_log_prob", 0.0), 0.0) for c in candidates]
        glps = [_safe_float(c.get("grid_log_prob", 0.0), 0.0) for c in candidates]

        score_max, score_mean, score_std = self._stats(scores)
        conf_max, conf_mean, conf_std = self._stats(confs)
        jlp_max, jlp_mean, jlp_std = self._stats(jlps)
        alp_max, _, _ = self._stats(alps)
        glp_max, _, _ = self._stats(glps)

        score_order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        score_rank = {idx: rank for rank, idx in enumerate(score_order)}

        action_groups: dict[str, list[int]] = {}
        for idx, c in enumerate(candidates):
            action = str(c.get("action_type", "click"))
            action_groups.setdefault(action, []).append(idx)
        action_local_rank: dict[int, int] = {}
        for indices in action_groups.values():
            indices_sorted = sorted(indices, key=lambda i: jlps[i], reverse=True)
            for local_rank, idx in enumerate(indices_sorted):
                action_local_rank[idx] = local_rank

        normalized_sources = [_normalize_source(str(c.get("source", ""))) for c in candidates]
        source_groups: dict[str, list[int]] = {}
        for idx, source in enumerate(normalized_sources):
            source_groups.setdefault(source, []).append(idx)
        source_local_rank: dict[int, int] = {}
        for indices in source_groups.values():
            indices_sorted = sorted(indices, key=lambda i: scores[i], reverse=True)
            for local_rank, idx in enumerate(indices_sorted):
                source_local_rank[idx] = local_rank
        structured_indices = source_groups.get("structured_sampled_t0p6", [])
        structured_raw_areas = []
        structured_raw_widths = []
        for structured_idx in structured_indices:
            bx1, by1, bx2, by2 = [float(v) for v in _candidate_bbox(candidates[structured_idx])]
            structured_raw_areas.append(max(0.0, bx2 - bx1) * max(0.0, by2 - by1))
            structured_raw_widths.append(max(0.0, bx2 - bx1))
        min_structured_raw_area = min(structured_raw_areas, default=0.0)
        min_structured_raw_width = min(structured_raw_widths, default=0.0)

        width = _safe_float(sample.get("image_width"), 0.0)
        height = _safe_float(sample.get("image_height"), 0.0)
        if width <= 0.0:
            width = max(_candidate_bbox(c)[2] for c in candidates)
        if height <= 0.0:
            height = max(_candidate_bbox(c)[3] for c in candidates)
        width = max(width, 1.0)
        height = max(height, 1.0)
        image_diag = math.sqrt(width * width + height * height)
        image_diag = max(image_diag, 1.0)
        max_grid_id = max(int(c.get("grid_id", 0)) for c in candidates)
        max_grid_id = max(max_grid_id, 1)
        first_candidate = candidates[0]
        first_bbox = _candidate_bbox(first_candidate)
        first_click = _candidate_click(first_candidate)
        first_action = str(first_candidate.get("action_type", "click"))
        first_dom = first_candidate.get("dom_match") or {}
        first_dom_best_iou = _safe_float(first_dom.get("best_iou"), 0.0)
        distinct_source_count = max(len(source_groups), 1)
        exact_source_keys = (
            "stagea_first_choice",
            "structured_sampled_t0p6",
            "point_first_structured",
            "point_native_primary",
            "point_first_sampled_t0p7",
            "hybrid_point_structured",
            "legacy_clip_grid",
            "other",
        )

        features: list[list[float]] = []
        for idx, candidate in enumerate(candidates):
            bbox = _candidate_bbox(candidate)
            click = _candidate_click(candidate)
            grid_id = int(candidate.get("grid_id", 0))
            rank = float(candidate.get("rank", idx + 1))
            score = scores[idx]
            conf = confs[idx]
            jlp = jlps[idx]
            alp = alps[idx]
            glp = glps[idx]
            diag = candidate.get("structured_output_diagnostics") or {}
            parser_meta = candidate.get("parser_metadata") or {}
            dom_match = candidate.get("dom_match") or {}
            provenance = candidate.get("provenance") or {}

            x1, y1, x2, y2 = [float(v) for v in bbox]
            raw_width = max(0.0, x2 - x1)
            raw_height = max(0.0, y2 - y1)
            raw_area = raw_width * raw_height
            x1n = x1 / width
            y1n = y1 / height
            x2n = x2 / width
            y2n = y2 / height
            bw = max(0.0, x2n - x1n)
            bh = max(0.0, y2n - y1n)
            area = bw * bh
            aspect = bw / max(bh, 1e-6)
            cx = float(click[0]) / width
            cy = float(click[1]) / height
            center_dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

            action = str(candidate.get("action_type", "click"))
            one_hot = [1.0 if action == k else 0.0 for k in ("click", "type", "select", "hover")]
            normalized_source = normalized_sources[idx]
            source_bucket = _source_bucket(normalized_source)
            source_one_hot = [
                1.0 if source_bucket == key else 0.0
                for key in ("first_choice", "point_first", "point_native", "hybrid", "sampled", "legacy_clip", "other")
            ]
            exact_source_one_hot = [1.0 if normalized_source == key else 0.0 for key in exact_source_keys]

            grid_cols = 6.0
            row = math.floor(grid_id / grid_cols) / 4.0
            col = (grid_id % int(grid_cols)) / grid_cols

            rank_norm = rank / max(n, 1)
            score_percentile = 1.0 - (score_rank[idx] / max(n - 1, 1))
            local_group_size = max(len(action_groups.get(action, [])), 1)
            action_local_rank_norm = action_local_rank.get(idx, 0) / max(local_group_size - 1, 1)
            source_group_size = max(len(source_groups.get(normalized_source, [])), 1)
            source_local_rank_norm = source_local_rank.get(idx, 0) / max(source_group_size - 1, 1)

            failure_tags = diag.get("failure_tags") or []
            bbox_valid = 1.0 if x2 >= x1 and y2 >= y1 else 0.0
            click_valid = 1.0 if len(click) == 2 else 0.0
            json_parse_success = 1.0 if diag.get("json_parse_success") is True else 0.0
            action_valid = 1.0 if diag.get("action_type_valid") is True else 0.0
            point_pass_conf = _safe_float(parser_meta.get("point_pass_confidence"), 0.0)
            structure_pass_conf = _safe_float(parser_meta.get("structure_pass_confidence"), 0.0)
            generation_temperature = _safe_float(provenance.get("generation_temperature"), 0.0)
            point_first_prompt = 1.0 if provenance.get("point_first_prompt") else 0.0
            decoupled_decode = 1.0 if provenance.get("decoupled_point_native_decode") else 0.0
            dom_available = 1.0 if dom_match.get("available") else 0.0
            dom_best_iou = _safe_float(dom_match.get("best_iou"), 0.0)
            dom_click_inside = 1.0 if dom_match.get("click_inside_best_match") else 0.0
            dom_text_overlap = _safe_float(dom_match.get("instruction_text_overlap"), 0.0)
            exact_element_match = 1.0 if dom_match.get("exact_element_id_match") else 0.0
            element_hint_present = 1.0 if candidate.get("element_hint_id") else 0.0
            gating_meta = candidate.get("gating_metadata") or {}
            source_priority_score = _safe_float(gating_meta.get("source_priority"), 0.0) / 100.0
            click_distance_to_first_norm = _click_distance(click, first_click) / image_diag
            bbox_iou_to_first = _bbox_iou(bbox, first_bbox)
            action_match_first = 1.0 if action == first_action else 0.0
            dom_iou_delta_vs_first = dom_best_iou - first_dom_best_iou
            same_action_neighbors = []
            for other_idx, other_candidate in enumerate(candidates):
                if other_idx == idx:
                    continue
                other_action = str(other_candidate.get("action_type", "click"))
                if other_action != action:
                    continue
                other_bbox = _candidate_bbox(other_candidate)
                other_click = _candidate_click(other_candidate)
                click_dist = _click_distance(click, other_click)
                bbox_overlap = _bbox_iou(bbox, other_bbox)
                same_action_neighbors.append(
                    {
                        "source": normalized_sources[other_idx],
                        "click_dist": click_dist,
                        "bbox_iou": bbox_overlap,
                    }
                )
            support_neighbors = [
                row
                for row in same_action_neighbors
                if row["click_dist"] <= 96.0 or row["bbox_iou"] >= 0.5
            ]
            tight_support_neighbors = [
                row
                for row in same_action_neighbors
                if row["click_dist"] <= 48.0 or row["bbox_iou"] >= 0.5
            ]
            same_source_support_neighbors = [
                row
                for row in support_neighbors
                if row["source"] == normalized_source
            ]
            same_source_tight_support_neighbors = [
                row
                for row in tight_support_neighbors
                if row["source"] == normalized_source
            ]
            structured_neighbors = [
                row
                for row in same_action_neighbors
                if row["source"] == "structured_sampled_t0p6"
            ]
            structured_support_neighbors = [
                row
                for row in structured_neighbors
                if row["click_dist"] <= 96.0 or row["bbox_iou"] >= 0.5
            ]
            support_count_norm = len(support_neighbors) / max(n - 1, 1)
            tight_support_count_norm = len(tight_support_neighbors) / max(n - 1, 1)
            same_source_support_count_norm = len(same_source_support_neighbors) / max(source_group_size - 1, 1)
            same_source_tight_support_count_norm = len(same_source_tight_support_neighbors) / max(source_group_size - 1, 1)
            distinct_support_sources_norm = len({row["source"] for row in support_neighbors}) / max(distinct_source_count - 1, 1)
            max_support_bbox_iou = max((row["bbox_iou"] for row in same_action_neighbors), default=0.0)
            min_support_click_dist_norm = min((row["click_dist"] for row in same_action_neighbors), default=image_diag) / image_diag
            structured_neighbor_count_norm = len(structured_neighbors) / max(n - 1, 1)
            structured_support_count_norm = len(structured_support_neighbors) / max(len(structured_neighbors), 1)
            max_structured_bbox_iou = max((row["bbox_iou"] for row in structured_neighbors), default=0.0)
            min_structured_click_dist_norm = min((row["click_dist"] for row in structured_neighbors), default=image_diag) / image_diag
            area_vs_smallest_structured = 0.0
            width_vs_smallest_structured = 0.0
            if min_structured_raw_area > 0.0:
                area_vs_smallest_structured = (min_structured_raw_area - raw_area) / min_structured_raw_area
            if min_structured_raw_width > 0.0:
                width_vs_smallest_structured = (min_structured_raw_width - raw_width) / min_structured_raw_width

            feature_row = [
                score,
                conf,
                jlp,
                alp,
                glp,
                rank_norm,
                bw,
                bh,
                area,
                cx,
                cy,
                row,
                col,
                *one_hot,
                *source_one_hot,
                *exact_source_one_hot,
                x1n,
                y1n,
                x2n,
                y2n,
                aspect,
                center_dist,
                score - score_max,
                score - score_mean,
                (score - score_mean) / max(score_std, 1e-6),
                score_percentile,
                jlp - jlp_max,
                jlp - jlp_mean,
                (jlp - jlp_mean) / max(jlp_std, 1e-6),
                alp - alp_max,
                glp - glp_max,
                conf - conf_max,
                conf / max(conf_max, 1e-6),
                (conf - conf_mean) / max(conf_std, 1e-6),
                action_local_rank_norm,
                source_group_size / max(n, 1),
                source_local_rank_norm,
                grid_id / max_grid_id,
                rank_norm - score_percentile,
                bbox_valid,
                click_valid,
                json_parse_success,
                action_valid,
                1.0 if diag.get("bbox_from_click_fallback") else 0.0,
                1.0 if diag.get("click_from_bbox_fallback") else 0.0,
                1.0 if diag.get("bbox_clipped") else 0.0,
                1.0 if diag.get("click_point_clipped") else 0.0,
                float(len(failure_tags)),
                point_pass_conf,
                structure_pass_conf,
                generation_temperature,
                point_first_prompt,
                decoupled_decode,
                element_hint_present,
                dom_available,
                dom_best_iou,
                dom_click_inside,
                dom_text_overlap,
                exact_element_match,
                source_priority_score,
                click_distance_to_first_norm,
                bbox_iou_to_first,
                action_match_first,
                dom_iou_delta_vs_first,
                support_count_norm,
                distinct_support_sources_norm,
                max_support_bbox_iou,
                min_support_click_dist_norm,
            ]
            if self.feature_include_structured_relative_support:
                feature_row.extend(
                    [
                        tight_support_count_norm,
                        same_source_support_count_norm,
                        same_source_tight_support_count_norm,
                        structured_neighbor_count_norm,
                        structured_support_count_norm,
                        max_structured_bbox_iou,
                        min_structured_click_dist_norm,
                        area_vs_smallest_structured,
                        width_vs_smallest_structured,
                    ]
                )
            features.append(feature_row)
        return features

    def _build_pairs(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for sample in samples:
            candidates = sample.get("candidates", [])
            if len(candidates) < 2:
                continue
            feature_rows = self._build_feature_rows(sample)
            rewards = [float(c.get("reward", {}).get("total_reward", 0.0)) for c in candidates]
            normalized_sources = [self._normalize_source_name(c.get("source")) for c in candidates]
            recovery_info = self._headroom_recovery_info(sample)
            baseline_idx = 0
            baseline_reward = rewards[baseline_idx]
            best_reward = max(rewards)
            pool_oracle_gap = best_reward - baseline_reward
            _, _, image_diag = self._sample_image_size(sample, candidates)
            point_first_info = self._point_first_disagreement_support_info(sample, recovery_info, normalized_sources)
            structured_singleton_info: dict[str, Any] | None = None

            pair_specs: list[tuple[int, int, str]] = []
            if self.pair_construction_mode == "all_pairs":
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        pair_specs.append((i, j, "all_pairs"))
            elif self.pair_construction_mode == "headroom_hard_negative":
                if recovery_info is None:
                    # No headroom => skip this pool for supervision
                    continue
                for idx in recovery_info["positive_indices"]:
                    pair_specs.append((idx, baseline_idx, "recovery_vs_first_choice"))
            elif self.pair_construction_mode in (
                "recovery_source_aware",
                "rare_recovery_targeted",
                "conditional_singleton_recovery",
            ):
                if recovery_info is None:
                    continue
                positive_indices = sorted(
                    recovery_info["positive_indices"],
                    key=lambda idx: (rewards[idx], *self._candidate_priority(candidates[idx], idx)),
                    reverse=True,
                )
                best_idx = recovery_info["best_idx"]
                positive_set = set(positive_indices)

                for pos_idx in positive_indices:
                    pair_specs.append((pos_idx, baseline_idx, "recovery_vs_first_choice"))

                for pos_idx in positive_indices[1:]:
                    if rewards[best_idx] > rewards[pos_idx] + self.min_reward_diff:
                        pair_specs.append((best_idx, pos_idx, "best_vs_other_positive"))

                structured_singleton_info = self._structured_singleton_disambiguation_info(
                    sample,
                    recovery_info,
                    normalized_sources,
                )
                if self.pair_construction_mode == "recovery_source_aware":
                    decoy_by_source: dict[str, tuple[int, tuple[float, float, float, float]]] = {}
                    for idx, candidate in enumerate(candidates):
                        if idx == baseline_idx or idx in positive_set:
                            continue
                        source = normalized_sources[idx]
                        priority = self._candidate_priority(candidate, idx)
                        prev = decoy_by_source.get(source)
                        if prev is None or priority > prev[1]:
                            decoy_by_source[source] = (idx, priority)
                    source_decoys = [
                        idx
                        for _, (idx, _) in sorted(
                            decoy_by_source.items(),
                            key=lambda item: item[1][1],
                            reverse=True,
                        )
                    ]
                    for neg_idx in source_decoys[: self.pair_source_decoy_max_sources]:
                        pair_specs.append((best_idx, neg_idx, "best_vs_source_decoy"))
                else:
                    same_source_decoys = [
                        idx
                        for idx in range(len(candidates))
                        if idx != baseline_idx
                        and idx not in positive_set
                        and normalized_sources[idx] == normalized_sources[best_idx]
                    ]

                    def _append_generic_wrong_source_pairs() -> None:
                        wrong_source_sorted = sorted(
                            [
                                idx
                                for idx in range(len(candidates))
                                if idx != baseline_idx
                                and idx not in positive_set
                                and normalized_sources[idx] != normalized_sources[best_idx]
                            ],
                            key=lambda idx: self._negative_decoy_priority(candidates[idx], rewards[idx], idx),
                            reverse=True,
                        )
                        seen_sources: set[str] = set()
                        for neg_idx in wrong_source_sorted:
                            neg_source = normalized_sources[neg_idx]
                            if neg_source in seen_sources:
                                continue
                            pair_specs.append((best_idx, neg_idx, "best_vs_best_wrong_source"))
                            seen_sources.add(neg_source)
                            if len(seen_sources) >= self.pair_source_decoy_max_sources:
                                break

                    if self.pair_construction_mode == "rare_recovery_targeted":
                        if same_source_decoys:
                            neg_idx = max(
                                same_source_decoys,
                                key=lambda idx: self._negative_decoy_priority(candidates[idx], rewards[idx], idx),
                            )
                            pair_specs.append((best_idx, neg_idx, "best_vs_same_source_decoy"))

                        if point_first_info is not None and point_first_info["is_targeted"]:
                            if self.pair_point_first_all_structured_decoys:
                                for neg_idx in point_first_info["structured_disagreement_indices"]:
                                    pair_specs.append((best_idx, neg_idx, "point_first_best_vs_structured_disagreement"))
                            else:
                                structured_idx = max(
                                    point_first_info["structured_disagreement_indices"],
                                    key=lambda idx: self._negative_decoy_priority(candidates[idx], rewards[idx], idx),
                                    default=None,
                                )
                                if structured_idx is not None:
                                    pair_specs.append((best_idx, structured_idx, "point_first_best_vs_structured_disagreement"))
                            for neg_idx in point_first_info["support_anchor_indices"][
                                : self.pair_source_decoy_max_sources
                            ]:
                                pair_specs.append((best_idx, neg_idx, "point_first_support_anchor"))
                        else:
                            _append_generic_wrong_source_pairs()
                    else:
                        if structured_singleton_info is not None and structured_singleton_info["is_targeted"]:
                            for neg_idx in structured_singleton_info["same_source_decoy_indices"]:
                                pair_specs.append((best_idx, neg_idx, "structured_singleton_best_vs_structured_decoy"))
                            for neg_idx in structured_singleton_info["support_anchor_indices"][
                                : self.pair_source_decoy_max_sources
                            ]:
                                pair_specs.append((best_idx, neg_idx, "structured_singleton_support_anchor"))
                        elif point_first_info is not None and point_first_info["is_targeted"]:
                            structured_idx = max(
                                point_first_info["structured_disagreement_indices"],
                                key=lambda idx: self._negative_decoy_priority(candidates[idx], rewards[idx], idx),
                                default=None,
                            )
                            if structured_idx is not None:
                                pair_specs.append((best_idx, structured_idx, "point_first_best_vs_structured_disagreement"))
                            anchor_limit = max(self.pair_source_decoy_max_sources - 1, 1)
                            for neg_idx in point_first_info["support_anchor_indices"][:anchor_limit]:
                                pair_specs.append((best_idx, neg_idx, "point_first_support_anchor"))
                        else:
                            if same_source_decoys:
                                neg_idx = max(
                                    same_source_decoys,
                                    key=lambda idx: self._negative_decoy_priority(candidates[idx], rewards[idx], idx),
                                )
                                pair_specs.append((best_idx, neg_idx, "best_vs_same_source_decoy"))
                            _append_generic_wrong_source_pairs()

                    for neg_idx in self._isolated_point_first_false_alarm_indices(
                        sample,
                        normalized_sources,
                        rewards,
                        baseline_reward,
                    ):
                        if neg_idx != best_idx:
                            pair_specs.append((best_idx, neg_idx, "best_vs_point_first_false_alarm"))
            else:
                raise ValueError(f"Unsupported pair_construction_mode: {self.pair_construction_mode}")

            deduped_specs: list[tuple[int, int, str]] = []
            seen_pair_specs: set[tuple[int, int, str]] = set()
            for spec in pair_specs:
                if spec in seen_pair_specs:
                    continue
                deduped_specs.append(spec)
                seen_pair_specs.add(spec)

            for i, j, pair_type in deduped_specs:
                c1 = candidates[i]
                c2 = candidates[j]
                r1 = rewards[i]
                r2 = rewards[j]
                if abs(r1 - r2) <= self.min_reward_diff:
                    continue
                if r1 > r2:
                    pos_idx, neg_idx = i, j
                    pos_reward, neg_reward = r1, r2
                else:
                    pos_idx, neg_idx = j, i
                    pos_reward, neg_reward = r2, r1
                reward_diff = float(pos_reward - neg_reward)
                if reward_diff < self.pair_reward_gap_threshold:
                    continue
                pool_best_source = recovery_info["best_source"] if recovery_info is not None else normalized_sources[pos_idx]
                pool_best_signature = (
                    recovery_info["best_signature"]
                    if recovery_info is not None
                    else self._recovery_signature(candidates[pos_idx], pos_idx + 1, 1)
                )
                best_source_count = max(int(self.recovery_source_counts.get(pool_best_source, 0)), 1)
                max_source_count = max(self.recovery_source_counts.values(), default=1)
                best_signature_count = max(int(self.recovery_signature_counts.get(pool_best_signature, 0)), 1)
                max_signature_count = max(self.recovery_signature_counts.values(), default=1)
                source_rarity = 1.0 - (best_source_count / max(max_source_count, 1))
                signature_rarity = 1.0 - (best_signature_count / max(max_signature_count, 1))
                negative_strength = max(0.0, neg_reward - baseline_reward)
                pos_bbox = self._candidate_bbox(candidates[pos_idx])
                neg_bbox = self._candidate_bbox(candidates[neg_idx])
                pos_click = self._candidate_click(candidates[pos_idx])
                neg_click = self._candidate_click(candidates[neg_idx])
                pair_bbox_iou = self._bbox_iou(pos_bbox, neg_bbox)
                pair_click_distance_norm = self._click_distance(pos_click, neg_click) / image_diag
                disagreement_strength = 0.5 * (1.0 - min(max(pair_bbox_iou, 0.0), 1.0)) + 0.5 * min(
                    pair_click_distance_norm / 0.25,
                    1.0,
                )
                pos_components = candidates[pos_idx].get("reward", {}).get("components", {}) or {}
                neg_components = candidates[neg_idx].get("reward", {}).get("components", {}) or {}
                pair_positive_signal_advantage = max(
                    self._safe_float(pos_components.get("click_inside_target"), 0.0)
                    - self._safe_float(neg_components.get("click_inside_target"), 0.0),
                    0.0,
                ) + max(
                    self._safe_float(pos_components.get("iou"), 0.0)
                    - self._safe_float(neg_components.get("iou"), 0.0),
                    0.0,
                )
                point_first_targeted_pool = bool(point_first_info and point_first_info["is_targeted"])
                structured_singleton_targeted_pool = bool(
                    structured_singleton_info and structured_singleton_info["is_targeted"]
                )
                if self.pair_weight_mode == "uniform":
                    pair_weight = 1.0
                elif self.pair_weight_mode == "reward_gap":
                    pair_weight = min(1.0 + self.pair_weight_alpha * reward_diff, self.pair_weight_cap)
                else:
                    pos_source = normalized_sources[pos_idx]
                    neg_source = normalized_sources[neg_idx]
                    pair_weight = 1.0 + self.pair_weight_alpha * reward_diff
                    pair_weight *= 1.0 + self.pair_pool_gap_bonus * min(pool_oracle_gap, 1.0)
                    if pair_type == "recovery_vs_first_choice":
                        pair_weight *= self.pair_recovery_anchor_weight
                    elif pair_type == "best_vs_other_positive":
                        pair_weight *= self.pair_positive_ranking_weight
                    elif pair_type == "best_vs_source_decoy":
                        pair_weight *= self.pair_source_decoy_weight
                    elif pair_type == "best_vs_same_source_decoy":
                        pair_weight *= self.pair_same_source_decoy_weight
                    elif pair_type == "best_vs_best_wrong_source":
                        pair_weight *= self.pair_source_decoy_weight
                    elif pair_type == "best_vs_point_first_false_alarm":
                        pair_weight *= self.pair_source_decoy_weight
                    elif pair_type == "point_first_best_vs_structured_disagreement":
                        pair_weight *= self.pair_source_decoy_weight
                    elif pair_type == "point_first_support_anchor":
                        pair_weight *= self.pair_point_first_support_anchor_weight
                    elif pair_type == "structured_singleton_best_vs_structured_decoy":
                        pair_weight *= self.pair_structured_singleton_decoy_weight
                    elif pair_type == "structured_singleton_support_anchor":
                        pair_weight *= self.pair_structured_singleton_support_anchor_weight
                    if pos_source != neg_source:
                        pair_weight *= 1.0 + self.pair_cross_source_bonus
                    pair_weight *= 1.0 + self.pair_source_prior_bonus * self.source_recovery_priors.get(pos_source, 0.0)
                    pair_weight *= 1.0 + self.pair_rare_source_bonus * source_rarity
                    pair_weight *= 1.0 + self.pair_rare_signature_bonus * signature_rarity
                    pair_weight *= 1.0 + self.pair_negative_strength_bonus * min(negative_strength, 1.0)
                    if point_first_targeted_pool and pos_source == "point_first_sampled_t0p7":
                        pair_weight *= 1.0 + self.pair_point_first_bonus
                        pair_weight *= 1.0 + self.pair_disagreement_bonus * disagreement_strength
                        pair_weight *= 1.0 + self.pair_positive_signal_bonus * min(pair_positive_signal_advantage, 1.0)
                        pair_weight *= 1.0 + self.pair_conditional_singleton_bonus * min(pool_oracle_gap, 1.0)
                    elif structured_singleton_targeted_pool and pos_source == "structured_sampled_t0p6":
                        pair_weight *= 1.0 + self.pair_conditional_singleton_bonus * min(
                            pair_positive_signal_advantage,
                            1.0,
                        )
                    elif pair_type == "best_vs_point_first_false_alarm":
                        pair_weight *= 1.0 + self.pair_disagreement_bonus * disagreement_strength
                    pair_weight = min(pair_weight, self.pair_weight_cap)
                pairs.append(
                    {
                        "sample_id": sample["sample_id"],
                        "pool_id": sample["sample_id"],
                        "preferred_candidate_id": str(candidates[pos_idx].get("candidate_id", pos_idx)),
                        "rejected_candidate_id": str(candidates[neg_idx].get("candidate_id", neg_idx)),
                        "preferred_rank": float(candidates[pos_idx].get("rank", pos_idx + 1)),
                        "rejected_rank": float(candidates[neg_idx].get("rank", neg_idx + 1)),
                        "reward_diff": reward_diff,
                        "pair_weight": float(pair_weight),
                        "pair_construction_mode": self.pair_construction_mode,
                        "pair_weight_mode": self.pair_weight_mode,
                        "pair_type": pair_type,
                        "pos_source": normalized_sources[pos_idx],
                        "neg_source": normalized_sources[neg_idx],
                        "pool_best_source": pool_best_source,
                        "pool_best_signature": pool_best_signature,
                        "pool_best_source_count": best_source_count,
                        "pool_best_signature_count": best_signature_count,
                        "pool_source_rarity": float(source_rarity),
                        "pool_signature_rarity": float(signature_rarity),
                        "negative_strength": float(negative_strength),
                        "pool_oracle_gap": float(pool_oracle_gap),
                        "pos_vs_first_gap": float(pos_reward - baseline_reward),
                        "neg_vs_first_gap": float(neg_reward - baseline_reward),
                        "point_first_targeted_pool": point_first_targeted_pool,
                        "structured_singleton_targeted_pool": structured_singleton_targeted_pool,
                        "pair_bbox_iou": float(pair_bbox_iou),
                        "pair_click_distance_norm": float(pair_click_distance_norm),
                        "pair_disagreement_strength": float(disagreement_strength),
                        "pair_positive_signal_advantage": float(pair_positive_signal_advantage),
                        "pos_features": feature_rows[pos_idx],
                        "neg_features": feature_rows[neg_idx],
                        "pos_reward": pos_reward,
                        "neg_reward": neg_reward,
                    }
                )
        return pairs

    def _split_samples(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.sample_split_mode == "headroom_source_stratified":
            val_target = max(1, int(len(samples) * self.val_ratio))
            headroom_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            non_headroom_samples: list[dict[str, Any]] = []
            for sample in samples:
                recovery_info = self._headroom_recovery_info(sample)
                if recovery_info is None:
                    non_headroom_samples.append(sample)
                    continue
                headroom_groups[recovery_info["best_source"]].append((sample, recovery_info))

            train_samples: list[dict[str, Any]] = []
            eval_samples: list[dict[str, Any]] = []
            for best_source, items in headroom_groups.items():
                self.rng.shuffle(items)
                signature_counts = Counter(info["best_signature"] for _, info in items)
                items = sorted(
                    items,
                    key=lambda item: (
                        -signature_counts[item[1]["best_signature"]],
                        item[1]["pool_oracle_gap"],
                    ),
                )
                group_size = len(items)
                if group_size <= 1:
                    train_samples.extend(sample for sample, _ in items)
                    continue
                if best_source in self.sample_split_protected_sources:
                    train_samples.extend(sample for sample, _ in items)
                    continue
                val_count = int(round(group_size * self.val_ratio))
                if group_size >= 3:
                    val_count = max(1, val_count)
                val_count = min(max(val_count, 0), group_size - 1)
                eval_samples.extend(sample for sample, _ in items[:val_count])
                train_samples.extend(sample for sample, _ in items[val_count:])

            self.rng.shuffle(non_headroom_samples)
            remaining_val = max(val_target - len(eval_samples), 0)
            eval_samples.extend(non_headroom_samples[:remaining_val])
            train_samples.extend(non_headroom_samples[remaining_val:])
            self.rng.shuffle(train_samples)
            self.rng.shuffle(eval_samples)
            return train_samples, eval_samples

        idx = list(range(len(samples)))
        self.rng.shuffle(idx)
        val_count = max(1, int(len(samples) * self.val_ratio))
        val_idx = set(idx[:val_count])
        train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
        val_samples = [s for i, s in enumerate(samples) if i in val_idx]
        return train_samples, val_samples

    def _pair_collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pos = torch.tensor([item["pos_features"] for item in batch], dtype=torch.float32)
        neg = torch.tensor([item["neg_features"] for item in batch], dtype=torch.float32)
        weights = torch.tensor([float(item.get("pair_weight", 1.0)) for item in batch], dtype=torch.float32)
        return {"pos": pos, "neg": neg, "weights": weights}

    def _save_preference_pairs(self, pairs: list[dict[str, Any]], split_name: str) -> str:
        rows = []
        for p in pairs:
            rows.append(
                {
                    "sample_id": p["sample_id"],
                    "pool_id": p["pool_id"],
                    "preferred_candidate_id": p["preferred_candidate_id"],
                    "rejected_candidate_id": p["rejected_candidate_id"],
                    "preferred_rank": p["preferred_rank"],
                    "rejected_rank": p["rejected_rank"],
                    "preferred_reward": p["pos_reward"],
                    "rejected_reward": p["neg_reward"],
                    "reward_diff": p["reward_diff"],
                    "pair_weight": float(p.get("pair_weight", 1.0)),
                    "pair_construction_mode": p.get("pair_construction_mode", self.pair_construction_mode),
                    "pair_weight_mode": p.get("pair_weight_mode", self.pair_weight_mode),
                    "pair_type": p.get("pair_type", "unknown"),
                    "preferred_source": p.get("pos_source", "other"),
                    "rejected_source": p.get("neg_source", "other"),
                    "pool_best_source": p.get("pool_best_source", "other"),
                    "pool_best_signature": p.get("pool_best_signature", ""),
                    "pool_best_source_count": p.get("pool_best_source_count", 0),
                    "pool_best_signature_count": p.get("pool_best_signature_count", 0),
                    "pool_source_rarity": p.get("pool_source_rarity", 0.0),
                    "pool_signature_rarity": p.get("pool_signature_rarity", 0.0),
                    "negative_strength": p.get("negative_strength", 0.0),
                    "pool_oracle_gap": p.get("pool_oracle_gap", 0.0),
                    "preferred_vs_first_gap": p.get("pos_vs_first_gap", 0.0),
                    "rejected_vs_first_gap": p.get("neg_vs_first_gap", 0.0),
                    "point_first_targeted_pool": p.get("point_first_targeted_pool", False),
                    "structured_singleton_targeted_pool": p.get("structured_singleton_targeted_pool", False),
                    "pair_bbox_iou": p.get("pair_bbox_iou", 0.0),
                    "pair_click_distance_norm": p.get("pair_click_distance_norm", 0.0),
                    "pair_disagreement_strength": p.get("pair_disagreement_strength", 0.0),
                    "pair_positive_signal_advantage": p.get("pair_positive_signal_advantage", 0.0),
                }
            )
        out_path = self.output_dir / f"preference_pairs_{split_name}.jsonl"
        save_jsonl(rows, out_path)
        return str(out_path)

    def _estimate_source_recovery_priors(self, samples: list[dict[str, Any]]) -> dict[str, float]:
        best_source_counts: Counter[str] = Counter()
        recovery_pool_count = 0
        for sample in samples:
            recovery_info = self._headroom_recovery_info(sample)
            if recovery_info is None:
                continue
            recovery_pool_count += 1
            best_source_counts[recovery_info["best_source"]] += 1
        if recovery_pool_count <= 0:
            return {}
        return {
            source: count / recovery_pool_count
            for source, count in best_source_counts.items()
        }

    def _estimate_recovery_counts(self, samples: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
        source_counts: Counter[str] = Counter()
        signature_counts: Counter[str] = Counter()
        for sample in samples:
            recovery_info = self._headroom_recovery_info(sample)
            if recovery_info is None:
                continue
            source_counts[recovery_info["best_source"]] += 1
            signature_counts[recovery_info["best_signature"]] += 1
        return dict(source_counts), dict(signature_counts)

    def _save_supervision_summary(
        self,
        train_samples: list[dict[str, Any]],
        eval_samples: list[dict[str, Any]],
        train_pairs: list[dict[str, Any]],
        eval_pairs: list[dict[str, Any]],
    ) -> dict[str, str]:
        def _count_headroom(samples: list[dict[str, Any]]) -> int:
            count = 0
            for sample in samples:
                candidates = sample.get("candidates", [])
                if len(candidates) < 2:
                    continue
                rewards = [float(c.get("reward", {}).get("total_reward", 0.0)) for c in candidates]
                if max(rewards) > rewards[0] + self.min_reward_diff:
                    count += 1
            return count

        def _pair_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
            if not pairs:
                return {
                    "count": 0,
                    "pair_type_counts": {},
                    "preferred_source_counts": {},
                    "rejected_source_counts": {},
                    "pool_best_source_counts": {},
                    "pool_best_signature_counts": {},
                    "reward_diff_mean": 0.0,
                    "reward_diff_max": 0.0,
                    "pair_weight_mean": 0.0,
                    "point_first_targeted_count": 0,
                    "structured_singleton_targeted_count": 0,
                }
            reward_diffs = [float(pair["reward_diff"]) for pair in pairs]
            pair_weights = [float(pair.get("pair_weight", 1.0)) for pair in pairs]
            return {
                "count": len(pairs),
                "pair_type_counts": dict(Counter(pair.get("pair_type", "unknown") for pair in pairs)),
                "preferred_source_counts": dict(Counter(pair.get("pos_source", "other") for pair in pairs)),
                "rejected_source_counts": dict(Counter(pair.get("neg_source", "other") for pair in pairs)),
                "pool_best_source_counts": dict(Counter(pair.get("pool_best_source", "other") for pair in pairs)),
                "pool_best_signature_counts": dict(Counter(pair.get("pool_best_signature", "") for pair in pairs)),
                "reward_diff_mean": sum(reward_diffs) / len(reward_diffs),
                "reward_diff_max": max(reward_diffs),
                "pair_weight_mean": sum(pair_weights) / len(pair_weights),
                "point_first_targeted_count": sum(1 for pair in pairs if pair.get("point_first_targeted_pool")),
                "structured_singleton_targeted_count": sum(
                    1 for pair in pairs if pair.get("structured_singleton_targeted_pool")
                ),
            }

        summary = {
            "pair_construction_mode": self.pair_construction_mode,
            "pair_weight_mode": self.pair_weight_mode,
            "sample_split_mode": self.sample_split_mode,
            "feature_include_structured_relative_support": self.feature_include_structured_relative_support,
            "checkpoint_selection_mode": self.checkpoint_selection_mode,
            "pair_reward_gap_threshold": self.pair_reward_gap_threshold,
            "pair_weight_alpha": self.pair_weight_alpha,
            "pair_weight_cap": self.pair_weight_cap,
            "pair_source_decoy_max_sources": self.pair_source_decoy_max_sources,
            "pair_recovery_anchor_weight": self.pair_recovery_anchor_weight,
            "pair_positive_ranking_weight": self.pair_positive_ranking_weight,
            "pair_source_decoy_weight": self.pair_source_decoy_weight,
            "pair_same_source_decoy_weight": self.pair_same_source_decoy_weight,
            "pair_cross_source_bonus": self.pair_cross_source_bonus,
            "pair_source_prior_bonus": self.pair_source_prior_bonus,
            "pair_pool_gap_bonus": self.pair_pool_gap_bonus,
            "pair_rare_source_bonus": self.pair_rare_source_bonus,
            "pair_rare_signature_bonus": self.pair_rare_signature_bonus,
            "pair_negative_strength_bonus": self.pair_negative_strength_bonus,
            "pair_point_first_bonus": self.pair_point_first_bonus,
            "pair_point_first_support_anchor_weight": self.pair_point_first_support_anchor_weight,
            "pair_disagreement_bonus": self.pair_disagreement_bonus,
            "pair_positive_signal_bonus": self.pair_positive_signal_bonus,
            "pair_point_first_all_structured_decoys": self.pair_point_first_all_structured_decoys,
            "pair_conditional_singleton_bonus": self.pair_conditional_singleton_bonus,
            "pair_point_first_signal_threshold": self.pair_point_first_signal_threshold,
            "pair_point_first_gap_threshold": self.pair_point_first_gap_threshold,
            "pair_structured_singleton_signal_threshold": self.pair_structured_singleton_signal_threshold,
            "pair_structured_singleton_gap_threshold": self.pair_structured_singleton_gap_threshold,
            "pair_structured_singleton_decoy_weight": self.pair_structured_singleton_decoy_weight,
            "pair_structured_singleton_support_anchor_weight": self.pair_structured_singleton_support_anchor_weight,
            "sample_split_protected_sources": sorted(self.sample_split_protected_sources),
            "source_recovery_priors": self.source_recovery_priors,
            "recovery_source_counts": self.recovery_source_counts,
            "recovery_signature_counts": self.recovery_signature_counts,
            "num_train_headroom_pools": _count_headroom(train_samples),
            "num_eval_headroom_pools": _count_headroom(eval_samples),
            "train_pairs": _pair_stats(train_pairs),
            "eval_pairs": _pair_stats(eval_pairs),
        }

        json_path = self.output_dir / "supervision_summary.json"
        md_path = self.output_dir / "supervision_summary.md"
        save_json(summary, json_path)

        lines = [
            "# Mind2Web Stage-B Reranker Supervision Summary",
            "",
            f"- pair_construction_mode: {self.pair_construction_mode}",
            f"- pair_weight_mode: {self.pair_weight_mode}",
            f"- sample_split_mode: {self.sample_split_mode}",
            f"- feature_include_structured_relative_support: {self.feature_include_structured_relative_support}",
            f"- checkpoint_selection_mode: {self.checkpoint_selection_mode}",
            f"- num_train_headroom_pools: {summary['num_train_headroom_pools']}",
            f"- num_eval_headroom_pools: {summary['num_eval_headroom_pools']}",
            "",
            "## Source Recovery Priors",
            "",
        ]
        for source, value in sorted(self.source_recovery_priors.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {source}: {value:.4f}")
        lines.extend(["", "## Recovery Source Counts", ""])
        for source, value in sorted(self.recovery_source_counts.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {source}: {value}")
        lines.extend(["", "## Recovery Signature Counts", ""])
        for signature, value in sorted(self.recovery_signature_counts.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {signature}: {value}")
        for split_name in ("train_pairs", "eval_pairs"):
            section = summary[split_name]
            lines.extend(
                [
                    "",
                    f"## {split_name}",
                    "",
                    f"- count: {section['count']}",
                    f"- reward_diff_mean: {section['reward_diff_mean']:.4f}",
                    f"- reward_diff_max: {section['reward_diff_max']:.4f}",
                    f"- pair_weight_mean: {section['pair_weight_mean']:.4f}",
                    f"- point_first_targeted_count: {section['point_first_targeted_count']}",
                    f"- structured_singleton_targeted_count: {section['structured_singleton_targeted_count']}",
                    "",
                    "### Pair Types",
                    "",
                ]
            )
            for key, value in section["pair_type_counts"].items():
                lines.append(f"- {key}: {value}")
            lines.extend(["", "### Preferred Sources", ""])
            for key, value in section["preferred_source_counts"].items():
                lines.append(f"- {key}: {value}")
            lines.extend(["", "### Rejected Sources", ""])
            for key, value in section["rejected_source_counts"].items():
                lines.append(f"- {key}: {value}")
            lines.extend(["", "### Pool Best Sources", ""])
            for key, value in section["pool_best_source_counts"].items():
                lines.append(f"- {key}: {value}")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "supervision_summary_path": str(json_path),
            "supervision_summary_md_path": str(md_path),
        }

    def _checkpoint_sort_key(self, metrics: dict[str, float]) -> tuple[float, ...]:
        if self.checkpoint_selection_mode == "headroom_subset_reward_gain":
            return (
                float(metrics.get("headroom_subset_reward_gain", 0.0)),
                float(metrics.get("full_pool_reward_gain", 0.0)),
                float(metrics.get("headroom_subset_reranked_best_recovery_rate", 0.0)),
            )
        if self.checkpoint_selection_mode == "headroom_then_full":
            return (
                float(metrics.get("headroom_subset_reward_gain", 0.0)),
                float(metrics.get("headroom_subset_reranked_best_recovery_rate", 0.0)),
                float(metrics.get("full_pool_reward_gain", 0.0)),
            )
        return (
            float(metrics.get("full_pool_reward_gain", 0.0)),
            float(metrics.get("headroom_subset_reward_gain", 0.0)),
            float(metrics.get("headroom_subset_reranked_best_recovery_rate", 0.0)),
        )

    def _run_epoch(self, loader: DataLoader) -> float:
        self.scorer.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            pos = batch["pos"].to(self.device)
            neg = batch["neg"].to(self.device)
            weights = batch["weights"].to(self.device)
            s_pos = self.scorer(pos)
            s_neg = self.scorer(neg)
            if self.optimization_mode == "pairwise":
                loss_vec = -F.logsigmoid(s_pos - s_neg - self.margin)
            elif self.optimization_mode == "dpo_style":
                if self.reference_scorer is None:
                    raise RuntimeError("reference_scorer is not initialized for dpo_style training.")
                with torch.inference_mode():
                    r_pos = self.reference_scorer(pos)
                    r_neg = self.reference_scorer(neg)
                # Lightweight DPO-style objective on scorer logits.
                # Treat scorer outputs as policy/reference log-prob proxies per candidate.
                pi_delta = s_pos - s_neg
                ref_delta = r_pos - r_neg
                loss_vec = -F.logsigmoid(self.dpo_beta * (pi_delta - ref_delta))
            else:
                raise ValueError(f"Unsupported optimization mode: {self.optimization_mode}")
            loss = (loss_vec * weights).sum() / weights.sum().clamp_min(1e-6)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            n += 1
        return total_loss / max(n, 1)

    def _select_best_index(self, sample: dict[str, Any]) -> int:
        candidates = sample.get("candidates", [])
        if not candidates:
            return 0
        feats = self._build_feature_rows(sample)
        x = torch.tensor(feats, dtype=torch.float32, device=self.device)
        self.scorer.eval()
        with torch.inference_mode():
            scores = self.scorer(x)
            if self.selection_drop_sources:
                for idx, candidate in enumerate(candidates):
                    normalized_source = self._normalize_source_name(candidate.get("source"))
                    if normalized_source in self.selection_drop_sources:
                        scores[idx] = float("-inf")
                if torch.isinf(scores).all():
                    return 0
            return int(torch.argmax(scores).item())

    def _aggregate_eval_subset(self, rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
        n = max(len(rows), 1)
        baseline_reward = sum(r["baseline_reward"] for r in rows) / n
        reranked_reward = sum(r["reranked_reward"] for r in rows) / n
        oracle_reward = sum(r["oracle_reward"] for r in rows) / n
        return {
            f"{prefix}_count": len(rows),
            f"{prefix}_baseline_mean_reward": baseline_reward,
            f"{prefix}_reranked_mean_reward": reranked_reward,
            f"{prefix}_oracle_mean_reward": oracle_reward,
            f"{prefix}_reward_gain": reranked_reward - baseline_reward,
            f"{prefix}_oracle_gain_upper_bound": oracle_reward - baseline_reward,
            f"{prefix}_rerank_win_rate": (sum(1.0 for r in rows if r["reranked_reward"] > r["baseline_reward"]) / n),
            f"{prefix}_baseline_best_recovery_rate": (
                sum(1.0 for r in rows if abs(r["baseline_reward"] - r["oracle_reward"]) <= 1e-9) / n
            ),
            f"{prefix}_reranked_best_recovery_rate": (
                sum(1.0 for r in rows if abs(r["reranked_reward"] - r["oracle_reward"]) <= 1e-9) / n
            ),
            f"{prefix}_baseline_mean_iou": (sum(r["baseline_iou"] for r in rows) / n),
            f"{prefix}_reranked_mean_iou": (sum(r["reranked_iou"] for r in rows) / n),
            f"{prefix}_oracle_mean_iou": (sum(r["oracle_iou"] for r in rows) / n),
            f"{prefix}_baseline_iou_at_0_5": (sum(r["baseline_iou_at_0_5"] for r in rows) / n),
            f"{prefix}_reranked_iou_at_0_5": (sum(r["reranked_iou_at_0_5"] for r in rows) / n),
            f"{prefix}_oracle_iou_at_0_5": (sum(r["oracle_iou_at_0_5"] for r in rows) / n),
            f"{prefix}_baseline_action_type_correct": (sum(r["baseline_action"] for r in rows) / n),
            f"{prefix}_reranked_action_type_correct": (sum(r["reranked_action"] for r in rows) / n),
            f"{prefix}_oracle_action_type_correct": (sum(r["oracle_action"] for r in rows) / n),
            f"{prefix}_baseline_click_inside_target": (sum(r["baseline_click"] for r in rows) / n),
            f"{prefix}_reranked_click_inside_target": (sum(r["reranked_click"] for r in rows) / n),
            f"{prefix}_oracle_click_inside_target": (sum(r["oracle_click"] for r in rows) / n),
            f"{prefix}_baseline_point_accuracy": (sum(r["baseline_click"] for r in rows) / n),
            f"{prefix}_reranked_point_accuracy": (sum(r["reranked_click"] for r in rows) / n),
            f"{prefix}_oracle_point_accuracy": (sum(r["oracle_click"] for r in rows) / n),
            f"{prefix}_baseline_parseable_rate": (sum(r["baseline_parseable"] for r in rows) / n),
            f"{prefix}_reranked_parseable_rate": (sum(r["reranked_parseable"] for r in rows) / n),
            f"{prefix}_oracle_parseable_rate": (sum(r["oracle_parseable"] for r in rows) / n),
            f"{prefix}_baseline_to_oracle_gap": oracle_reward - baseline_reward,
            f"{prefix}_reranked_to_oracle_gap": oracle_reward - reranked_reward,
        }

    def _evaluate_samples_internal(self, eval_samples: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        per_sample = []
        for sample in eval_samples:
            candidates = sample.get("candidates", [])
            if not candidates:
                continue
            rewards = [float(c["reward"]["total_reward"]) for c in candidates]
            oracle_idx = int(max(range(len(candidates)), key=lambda i: rewards[i]))
            first = candidates[0]
            rerank_idx = self._select_best_index(sample)
            reranked = candidates[rerank_idx]
            oracle = candidates[oracle_idx]

            bcomp = first["reward"]["components"]
            rcomp = reranked["reward"]["components"]
            ocomp = oracle["reward"]["components"]
            bdiag = first.get("structured_output_diagnostics") or {}
            rdiag = reranked.get("structured_output_diagnostics") or {}
            odiag = oracle.get("structured_output_diagnostics") or {}
            per_sample.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "baseline_index": 0,
                    "reranked_index": int(rerank_idx),
                    "oracle_index": int(oracle_idx),
                    "baseline_candidate_id": first.get("candidate_id"),
                    "reranked_candidate_id": reranked.get("candidate_id"),
                    "oracle_candidate_id": oracle.get("candidate_id"),
                    "baseline_source": self._normalize_source_name(first.get("source")),
                    "reranked_source": self._normalize_source_name(reranked.get("source")),
                    "oracle_source": self._normalize_source_name(oracle.get("source")),
                    "baseline_reward": float(first["reward"]["total_reward"]),
                    "reranked_reward": float(reranked["reward"]["total_reward"]),
                    "oracle_reward": float(oracle["reward"]["total_reward"]),
                    "baseline_iou": float(bcomp.get("iou", 0.0)),
                    "reranked_iou": float(rcomp.get("iou", 0.0)),
                    "oracle_iou": float(ocomp.get("iou", 0.0)),
                    "baseline_iou_at_0_5": 1.0 if float(bcomp.get("iou", 0.0)) >= 0.5 else 0.0,
                    "reranked_iou_at_0_5": 1.0 if float(rcomp.get("iou", 0.0)) >= 0.5 else 0.0,
                    "oracle_iou_at_0_5": 1.0 if float(ocomp.get("iou", 0.0)) >= 0.5 else 0.0,
                    "baseline_action": float(bcomp.get("action_type_correct", 0.0)),
                    "reranked_action": float(rcomp.get("action_type_correct", 0.0)),
                    "oracle_action": float(ocomp.get("action_type_correct", 0.0)),
                    "baseline_click": float(bcomp.get("click_inside_target", 0.0)),
                    "reranked_click": float(rcomp.get("click_inside_target", 0.0)),
                    "oracle_click": float(ocomp.get("click_inside_target", 0.0)),
                    "baseline_parseable": 1.0 if bdiag.get("json_parse_success") is not False else 0.0,
                    "reranked_parseable": 1.0 if rdiag.get("json_parse_success") is not False else 0.0,
                    "oracle_parseable": 1.0 if odiag.get("json_parse_success") is not False else 0.0,
                }
            )

        full_metrics = self._aggregate_eval_subset(per_sample, prefix="full_pool")
        headroom_rows = [r for r in per_sample if r["oracle_reward"] > r["baseline_reward"] + 1e-9]
        headroom_metrics = self._aggregate_eval_subset(headroom_rows, prefix="headroom_subset")
        result = {
            "num_eval_samples": len(per_sample),
            "num_headroom_subset_samples": len(headroom_rows),
            **full_metrics,
            **headroom_metrics,
        }
        # Backward-compatible aliases used by earlier Step-5 artifacts/scripts.
        result.update(
            {
                "baseline_mean_reward": result["full_pool_baseline_mean_reward"],
                "reranked_mean_reward": result["full_pool_reranked_mean_reward"],
                "reward_gain": result["full_pool_reward_gain"],
                "baseline_mean_iou": result["full_pool_baseline_mean_iou"],
                "reranked_mean_iou": result["full_pool_reranked_mean_iou"],
                "baseline_point_accuracy": result["full_pool_baseline_point_accuracy"],
                "reranked_point_accuracy": result["full_pool_reranked_point_accuracy"],
                "baseline_iou_at_0_5": result["full_pool_baseline_iou_at_0_5"],
                "reranked_iou_at_0_5": result["full_pool_reranked_iou_at_0_5"],
                "baseline_action_type_correct": result["full_pool_baseline_action_type_correct"],
                "reranked_action_type_correct": result["full_pool_reranked_action_type_correct"],
                "baseline_click_inside_target": result["full_pool_baseline_click_inside_target"],
                "reranked_click_inside_target": result["full_pool_reranked_click_inside_target"],
                "baseline_parseable_rate": result["full_pool_baseline_parseable_rate"],
                "reranked_parseable_rate": result["full_pool_reranked_parseable_rate"],
                "rerank_win_rate": result["full_pool_rerank_win_rate"],
            }
        )
        return result, per_sample

    def evaluate(self, eval_samples: list[dict[str, Any]]) -> dict[str, float]:
        result, _ = self._evaluate_samples_internal(eval_samples)
        return result

    def evaluate_candidate_file(
        self,
        candidate_path: str | Path,
        split_name: str | None = None,
        save_artifacts: bool = True,
    ) -> dict[str, Any]:
        candidate_path = Path(candidate_path)
        samples = load_jsonl(candidate_path)
        metrics, per_sample = self._evaluate_samples_internal(samples)
        result = {
            "candidate_path": str(candidate_path),
            "split_name": split_name or candidate_path.stem,
            "num_samples": len(samples),
            "metrics": metrics,
        }
        if save_artifacts:
            stem = split_name or candidate_path.stem
            save_json(result, self.output_dir / f"evaluation_{stem}.json")
            save_jsonl(per_sample, self.output_dir / f"evaluation_{stem}_per_sample.jsonl")
        return result

    def _save_comparison_artifacts(self, metrics: dict[str, float]) -> dict[str, str]:
        table_path = self.output_dir / "comparison_table.md"
        lines = [
            "| Scope | Metric | First-choice | Reranked | Oracle | Delta |",
            "|---|---|---:|---:|---:|---:|",
            f"| Full pool | Mean reward | {metrics['full_pool_baseline_mean_reward']:.4f} | {metrics['full_pool_reranked_mean_reward']:.4f} | {metrics['full_pool_oracle_mean_reward']:.4f} | {metrics['full_pool_reward_gain']:+.4f} |",
            f"| Full pool | Point accuracy | {metrics['full_pool_baseline_point_accuracy']:.4f} | {metrics['full_pool_reranked_point_accuracy']:.4f} | {metrics['full_pool_oracle_point_accuracy']:.4f} | {metrics['full_pool_reranked_point_accuracy']-metrics['full_pool_baseline_point_accuracy']:+.4f} |",
            f"| Full pool | Mean IoU | {metrics['full_pool_baseline_mean_iou']:.4f} | {metrics['full_pool_reranked_mean_iou']:.4f} | - | {metrics['full_pool_reranked_mean_iou']-metrics['full_pool_baseline_mean_iou']:+.4f} |",
            f"| Full pool | IoU@0.5 | {metrics['full_pool_baseline_iou_at_0_5']:.4f} | {metrics['full_pool_reranked_iou_at_0_5']:.4f} | {metrics['full_pool_oracle_iou_at_0_5']:.4f} | {metrics['full_pool_reranked_iou_at_0_5']-metrics['full_pool_baseline_iou_at_0_5']:+.4f} |",
            f"| Full pool | Action correct | {metrics['full_pool_baseline_action_type_correct']:.4f} | {metrics['full_pool_reranked_action_type_correct']:.4f} | - | {metrics['full_pool_reranked_action_type_correct']-metrics['full_pool_baseline_action_type_correct']:+.4f} |",
            f"| Full pool | Parseable rate | {metrics['full_pool_baseline_parseable_rate']:.4f} | {metrics['full_pool_reranked_parseable_rate']:.4f} | {metrics['full_pool_oracle_parseable_rate']:.4f} | {metrics['full_pool_reranked_parseable_rate']-metrics['full_pool_baseline_parseable_rate']:+.4f} |",
            f"| Full pool | Win rate | - | {metrics['full_pool_rerank_win_rate']:.4f} | - | - |",
            f"| Full pool | Best recovery rate | {metrics['full_pool_baseline_best_recovery_rate']:.4f} | {metrics['full_pool_reranked_best_recovery_rate']:.4f} | 1.0000 | {metrics['full_pool_reranked_best_recovery_rate']-metrics['full_pool_baseline_best_recovery_rate']:+.4f} |",
            f"| Headroom subset | Mean reward | {metrics['headroom_subset_baseline_mean_reward']:.4f} | {metrics['headroom_subset_reranked_mean_reward']:.4f} | {metrics['headroom_subset_oracle_mean_reward']:.4f} | {metrics['headroom_subset_reward_gain']:+.4f} |",
            f"| Headroom subset | Win rate | - | {metrics['headroom_subset_rerank_win_rate']:.4f} | - | - |",
            f"| Headroom subset | Best recovery rate | {metrics['headroom_subset_baseline_best_recovery_rate']:.4f} | {metrics['headroom_subset_reranked_best_recovery_rate']:.4f} | 1.0000 | {metrics['headroom_subset_reranked_best_recovery_rate']-metrics['headroom_subset_baseline_best_recovery_rate']:+.4f} |",
        ]
        table_path.write_text("\n".join(lines), encoding="utf-8")

        fig_path = self.output_dir / "comparison_reward_full_vs_headroom.png"
        width, height = 900, 420
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((20, 20), "Baseline vs Reranked vs Oracle (Full + Headroom)", fill="black", font=font)
        series = [
            ("Full-baseline", metrics["full_pool_baseline_mean_reward"], "#4e79a7"),
            ("Full-rerank", metrics["full_pool_reranked_mean_reward"], "#f28e2b"),
            ("Full-oracle", metrics["full_pool_oracle_mean_reward"], "#59a14f"),
            ("Head-baseline", metrics["headroom_subset_baseline_mean_reward"], "#9c755f"),
            ("Head-rerank", metrics["headroom_subset_reranked_mean_reward"], "#e15759"),
            ("Head-oracle", metrics["headroom_subset_oracle_mean_reward"], "#76b7b2"),
        ]
        vmax = max(v for _, v, _ in series) if series else 1e-6
        vmax = max(vmax, 1e-6)
        bar_w = 110
        max_h = 220
        base_y = 320
        start_x = 60
        gap = 24
        for i, (name, value, color) in enumerate(series):
            x1 = start_x + i * (bar_w + gap)
            bh = int((value / vmax) * max_h)
            y_top = base_y - bh
            draw.rectangle([x1, min(y_top, base_y), x1 + bar_w, max(y_top, base_y)], fill=color, outline="black")
            draw.text((x1, base_y + 8), name, fill="black", font=font)
            draw.text((x1, min(y_top, base_y) - 20), f"{value:.4f}", fill="black", font=font)
        draw.text(
            (20, 380),
            f"Full gain: {metrics['full_pool_reward_gain']:+.4f} | Headroom gain: {metrics['headroom_subset_reward_gain']:+.4f}",
            fill="black",
            font=font,
        )
        img.save(fig_path)
        return {"table_path": str(table_path), "figure_path": str(fig_path)}

    def _save_step5c_comparison_artifacts(self, metrics: dict[str, float]) -> dict[str, str]:
        if self.step5c_baseline_summary_path is None or not self.step5c_baseline_summary_path.exists():
            return {}

        baseline_summary = load_json(self.step5c_baseline_summary_path)
        b = baseline_summary.get("best_metrics", baseline_summary)

        full_step5c = float(b["full_pool_reranked_mean_reward"])
        head_step5c = float(b["headroom_subset_reranked_mean_reward"])
        full_step6a = float(metrics["full_pool_reranked_mean_reward"])
        head_step6a = float(metrics["headroom_subset_reranked_mean_reward"])
        full_oracle = float(metrics["full_pool_oracle_mean_reward"])
        head_oracle = float(metrics["headroom_subset_oracle_mean_reward"])

        table_path = self.output_dir / "comparison_step5c_vs_step6a.md"
        lines = [
            "| Scope | Step5c | Step6A(dpo_style) | Oracle | Delta vs Step5c |",
            "|---|---:|---:|---:|---:|",
            f"| Full pool mean reward | {full_step5c:.4f} | {full_step6a:.4f} | {full_oracle:.4f} | {full_step6a-full_step5c:+.4f} |",
            f"| Headroom subset mean reward | {head_step5c:.4f} | {head_step6a:.4f} | {head_oracle:.4f} | {head_step6a-head_step5c:+.4f} |",
            f"| Full pool best recovery | {float(b['full_pool_reranked_best_recovery_rate']):.4f} | {metrics['full_pool_reranked_best_recovery_rate']:.4f} | 1.0000 | {metrics['full_pool_reranked_best_recovery_rate']-float(b['full_pool_reranked_best_recovery_rate']):+.4f} |",
            f"| Headroom best recovery | {float(b['headroom_subset_reranked_best_recovery_rate']):.4f} | {metrics['headroom_subset_reranked_best_recovery_rate']:.4f} | 1.0000 | {metrics['headroom_subset_reranked_best_recovery_rate']-float(b['headroom_subset_reranked_best_recovery_rate']):+.4f} |",
        ]
        table_path.write_text("\n".join(lines), encoding="utf-8")

        fig_path = self.output_dir / "comparison_step5c_vs_step6a_oracle.png"
        width, height = 820, 420
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((20, 20), "Step5c vs Step6A(dpo_style) vs Oracle", fill="black", font=font)
        series = [
            ("Full-step5c", full_step5c, "#4e79a7"),
            ("Full-step6a", full_step6a, "#f28e2b"),
            ("Full-oracle", full_oracle, "#59a14f"),
            ("Head-step5c", head_step5c, "#9c755f"),
            ("Head-step6a", head_step6a, "#e15759"),
            ("Head-oracle", head_oracle, "#76b7b2"),
        ]
        vmax = max(v for _, v, _ in series)
        vmax = max(vmax, 1e-6)
        bar_w = 95
        gap = 24
        start_x = 40
        base_y = 320
        max_h = 230
        for i, (name, value, color) in enumerate(series):
            x1 = start_x + i * (bar_w + gap)
            bh = int((value / vmax) * max_h)
            y_top = base_y - bh
            draw.rectangle([x1, min(y_top, base_y), x1 + bar_w, max(y_top, base_y)], fill=color, outline="black")
            draw.text((x1, base_y + 8), name, fill="black", font=font)
            draw.text((x1, min(y_top, base_y) - 18), f"{value:.4f}", fill="black", font=font)
        draw.text(
            (20, 380),
            f"Delta full: {full_step6a-full_step5c:+.4f} | Delta headroom: {head_step6a-head_step5c:+.4f}",
            fill="black",
            font=font,
        )
        img.save(fig_path)
        return {"step5c_table_path": str(table_path), "step5c_figure_path": str(fig_path)}

    def _save_step5c_step6a_step6a5_comparison(self, metrics: dict[str, float]) -> dict[str, str]:
        if self.step5c_baseline_summary_path is None or not self.step5c_baseline_summary_path.exists():
            return {}
        if self.step6a_baseline_summary_path is None or not self.step6a_baseline_summary_path.exists():
            return {}

        s5 = load_json(self.step5c_baseline_summary_path).get("best_metrics", {})
        s6a = load_json(self.step6a_baseline_summary_path).get("best_metrics", {})
        s65 = metrics

        table_path = self.output_dir / "comparison_step5c_step6a_step6a5.md"
        lines = [
            "| Scope | Step5c | Step6A | Step6A.5 | Oracle |",
            "|---|---:|---:|---:|---:|",
            f"| Full pool mean reward | {float(s5['full_pool_reranked_mean_reward']):.4f} | {float(s6a['full_pool_reranked_mean_reward']):.4f} | {float(s65['full_pool_reranked_mean_reward']):.4f} | {float(s65['full_pool_oracle_mean_reward']):.4f} |",
            f"| Headroom subset mean reward | {float(s5['headroom_subset_reranked_mean_reward']):.4f} | {float(s6a['headroom_subset_reranked_mean_reward']):.4f} | {float(s65['headroom_subset_reranked_mean_reward']):.4f} | {float(s65['headroom_subset_oracle_mean_reward']):.4f} |",
            f"| Full pool best recovery | {float(s5['full_pool_reranked_best_recovery_rate']):.4f} | {float(s6a['full_pool_reranked_best_recovery_rate']):.4f} | {float(s65['full_pool_reranked_best_recovery_rate']):.4f} | 1.0000 |",
            f"| Headroom best recovery | {float(s5['headroom_subset_reranked_best_recovery_rate']):.4f} | {float(s6a['headroom_subset_reranked_best_recovery_rate']):.4f} | {float(s65['headroom_subset_reranked_best_recovery_rate']):.4f} | 1.0000 |",
        ]
        table_path.write_text("\n".join(lines), encoding="utf-8")

        fig_path = self.output_dir / "comparison_step5c_step6a_step6a5_oracle.png"
        width, height = 980, 430
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((20, 20), "Step5c vs Step6A vs Step6A.5 vs Oracle", fill="black", font=font)
        series = [
            ("Full-s5", float(s5["full_pool_reranked_mean_reward"]), "#4e79a7"),
            ("Full-s6a", float(s6a["full_pool_reranked_mean_reward"]), "#f28e2b"),
            ("Full-s6a5", float(s65["full_pool_reranked_mean_reward"]), "#e15759"),
            ("Full-oracle", float(s65["full_pool_oracle_mean_reward"]), "#59a14f"),
            ("Head-s5", float(s5["headroom_subset_reranked_mean_reward"]), "#9c755f"),
            ("Head-s6a", float(s6a["headroom_subset_reranked_mean_reward"]), "#76b7b2"),
            ("Head-s6a5", float(s65["headroom_subset_reranked_mean_reward"]), "#edc948"),
            ("Head-oracle", float(s65["headroom_subset_oracle_mean_reward"]), "#af7aa1"),
        ]
        vmax = max(v for _, v, _ in series)
        vmax = max(vmax, 1e-6)
        bar_w = 90
        gap = 18
        start_x = 24
        base_y = 330
        max_h = 240
        for i, (name, value, color) in enumerate(series):
            x1 = start_x + i * (bar_w + gap)
            bh = int((value / vmax) * max_h)
            y_top = base_y - bh
            draw.rectangle([x1, min(y_top, base_y), x1 + bar_w, max(y_top, base_y)], fill=color, outline="black")
            draw.text((x1, base_y + 8), name, fill="black", font=font)
            draw.text((x1, min(y_top, base_y) - 18), f"{value:.4f}", fill="black", font=font)
        draw.text(
            (20, 390),
            f"Delta(head) s6a5-s5: {float(s65['headroom_subset_reranked_mean_reward'])-float(s5['headroom_subset_reranked_mean_reward']):+.4f}",
            fill="black",
            font=font,
        )
        img.save(fig_path)
        return {"step5c_step6a_step6a5_table_path": str(table_path), "step5c_step6a_step6a5_figure_path": str(fig_path)}

    def _save_checkpoint(self) -> str:
        ckpt_dir = self.output_dir / "checkpoint-best"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state_dict(), ckpt_dir / "model.pt")
        meta = {
            "input_dim": int(getattr(self.scorer, "input_dim", 15)),
            "hidden_dim": int(getattr(self.scorer, "hidden_dim", 64)),
        }
        save_json(meta, ckpt_dir / "meta.json")
        return str(ckpt_dir)

    def train(self) -> dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Learned Reranker Training (CLIP-grid candidates)")
        logger.info("=" * 60)
        samples = load_jsonl(self.train_candidates_path)
        if not samples:
            raise RuntimeError(f"No samples loaded from {self.train_candidates_path}")
        train_samples, eval_samples = self._split_samples(samples)
        self.recovery_source_counts, self.recovery_signature_counts = self._estimate_recovery_counts(train_samples)
        self.source_recovery_priors = self._estimate_source_recovery_priors(train_samples)
        pair_samples = self._build_pairs(train_samples)
        eval_pair_samples = self._build_pairs(eval_samples)
        if not pair_samples:
            raise RuntimeError("No pairwise samples constructed; check reward variability in candidate pool.")
        self._init_reference_scorer_if_needed()
        logger.info(
            "Loaded samples: total=%d train=%d eval=%d train_pairs=%d eval_pairs=%d",
            len(samples),
            len(train_samples),
            len(eval_samples),
            len(pair_samples),
            len(eval_pair_samples),
        )

        preference_artifacts: dict[str, str] = {}
        if self.export_preference_pairs:
            preference_artifacts["train_preference_pairs_path"] = self._save_preference_pairs(pair_samples, "train")
            preference_artifacts["eval_preference_pairs_path"] = self._save_preference_pairs(eval_pair_samples, "eval")
        supervision_artifacts = self._save_supervision_summary(train_samples, eval_samples, pair_samples, eval_pair_samples)

        loader = DataLoader(
            PairwiseCandidateDataset(pair_samples),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._pair_collate,
        )

        history = []
        best_key: tuple[float, ...] | None = None
        best_metrics: dict[str, float] | None = None
        best_epoch = 0
        for epoch in range(1, self.num_epochs + 1):
            loss = self._run_epoch(loader)
            metrics = self.evaluate(eval_samples)
            row = {"epoch": epoch, "train_pairwise_loss": loss, **metrics}
            history.append(row)
            logger.info(
                "epoch=%d loss=%.4f full_gain=%+.4f head_gain=%+.4f full_base=%.4f full_rerank=%.4f",
                epoch,
                loss,
                metrics["full_pool_reward_gain"],
                metrics["headroom_subset_reward_gain"],
                metrics["full_pool_baseline_mean_reward"],
                metrics["full_pool_reranked_mean_reward"],
            )
            current_key = self._checkpoint_sort_key(metrics)
            if best_key is None or current_key > best_key:
                best_key = current_key
                best_metrics = metrics
                best_epoch = epoch
                ckpt = self._save_checkpoint()
                logger.info("Saved best checkpoint: %s", ckpt)

        if best_metrics is None:
            raise RuntimeError("No evaluation metrics produced.")
        best_checkpoint_model = self.output_dir / "checkpoint-best" / "model.pt"
        if best_checkpoint_model.exists():
            self._load_checkpoint(self.scorer, best_checkpoint_model)
        artifact_paths = self._save_comparison_artifacts(best_metrics)
        step5c_compare_paths = self._save_step5c_comparison_artifacts(best_metrics)
        multi_compare_paths = self._save_step5c_step6a_step6a5_comparison(best_metrics)
        summary = {
            "status": "ok",
            "data_path": str(self.train_candidates_path),
            "optimization_mode": self.optimization_mode,
            "dpo_beta": self.dpo_beta,
            "policy_init_checkpoint": str(self.policy_init_checkpoint) if self.policy_init_checkpoint else None,
            "reference_checkpoint": str(self.reference_checkpoint) if self.reference_checkpoint else None,
            "pair_construction_mode": self.pair_construction_mode,
            "pair_weight_mode": self.pair_weight_mode,
            "sample_split_mode": self.sample_split_mode,
            "feature_include_structured_relative_support": self.feature_include_structured_relative_support,
            "pair_reward_gap_threshold": self.pair_reward_gap_threshold,
            "pair_weight_alpha": self.pair_weight_alpha,
            "pair_weight_cap": self.pair_weight_cap,
            "pair_source_decoy_max_sources": self.pair_source_decoy_max_sources,
            "pair_recovery_anchor_weight": self.pair_recovery_anchor_weight,
            "pair_positive_ranking_weight": self.pair_positive_ranking_weight,
            "pair_source_decoy_weight": self.pair_source_decoy_weight,
            "pair_same_source_decoy_weight": self.pair_same_source_decoy_weight,
            "pair_cross_source_bonus": self.pair_cross_source_bonus,
            "pair_source_prior_bonus": self.pair_source_prior_bonus,
            "pair_pool_gap_bonus": self.pair_pool_gap_bonus,
            "pair_rare_source_bonus": self.pair_rare_source_bonus,
            "pair_rare_signature_bonus": self.pair_rare_signature_bonus,
            "pair_negative_strength_bonus": self.pair_negative_strength_bonus,
            "pair_point_first_bonus": self.pair_point_first_bonus,
            "pair_point_first_support_anchor_weight": self.pair_point_first_support_anchor_weight,
            "pair_disagreement_bonus": self.pair_disagreement_bonus,
            "pair_positive_signal_bonus": self.pair_positive_signal_bonus,
            "pair_point_first_all_structured_decoys": self.pair_point_first_all_structured_decoys,
            "pair_conditional_singleton_bonus": self.pair_conditional_singleton_bonus,
            "pair_point_first_signal_threshold": self.pair_point_first_signal_threshold,
            "pair_point_first_gap_threshold": self.pair_point_first_gap_threshold,
            "pair_structured_singleton_signal_threshold": self.pair_structured_singleton_signal_threshold,
            "pair_structured_singleton_gap_threshold": self.pair_structured_singleton_gap_threshold,
            "pair_structured_singleton_decoy_weight": self.pair_structured_singleton_decoy_weight,
            "pair_structured_singleton_support_anchor_weight": self.pair_structured_singleton_support_anchor_weight,
            "checkpoint_selection_mode": self.checkpoint_selection_mode,
            "sample_split_protected_sources": sorted(self.sample_split_protected_sources),
            "num_total_samples": len(samples),
            "num_train_samples": len(train_samples),
            "num_eval_samples": len(eval_samples),
            "num_pairwise_train_samples": len(pair_samples),
            "num_pairwise_eval_samples": len(eval_pair_samples),
            "feature_dim": int(getattr(self.scorer, "input_dim", 0)),
            "source_recovery_priors": self.source_recovery_priors,
            "recovery_source_counts": self.recovery_source_counts,
            "recovery_signature_counts": self.recovery_signature_counts,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            **artifact_paths,
            **step5c_compare_paths,
            **multi_compare_paths,
            **preference_artifacts,
            **supervision_artifacts,
            "checkpoint_path": str(self.output_dir / "checkpoint-best"),
        }
        save_json(history, self.output_dir / "training_history.json")
        save_json(summary, self.output_dir / "evaluation_summary.json")
        logger.info("Saved training history: %s", self.output_dir / "training_history.json")
        logger.info("Saved evaluation summary: %s", self.output_dir / "evaluation_summary.json")
        return summary
