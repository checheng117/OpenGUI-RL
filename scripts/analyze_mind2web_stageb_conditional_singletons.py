#!/usr/bin/env python3
"""Compact singleton-focused analysis for conditional Stage-B recovery supervision."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze conditional singleton recovery patterns.")
    parser.add_argument("--candidate-root", type=str, required=True)
    parser.add_argument("--current-reranker-root", type=str, required=True)
    parser.add_argument("--failed-reranker-root", type=str, default=None)
    parser.add_argument("--current-pairs-path", type=str, default=None)
    parser.add_argument("--failed-pairs-path", type=str, default=None)
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox(candidate: dict[str, Any]) -> list[float]:
    bbox = candidate.get("bbox_proposal") or candidate.get("bbox") or [0.0, 0.0, 1.0, 1.0]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    return [_safe_float(v, 0.0) for v in bbox]


def _click(candidate: dict[str, Any]) -> list[float]:
    click = candidate.get("click_point") or [0.0, 0.0]
    if not isinstance(click, list) or len(click) != 2:
        return [0.0, 0.0]
    return [_safe_float(v, 0.0) for v in click]


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
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
    dx = click_a[0] - click_b[0]
    dy = click_a[1] - click_b[1]
    return math.sqrt(dx * dx + dy * dy)


def _rank_bucket(rank: int) -> str:
    if rank <= 2:
        return "rank_2"
    if rank <= 4:
        return "rank_3_4"
    if rank <= 6:
        return "rank_5_6"
    return "rank_7_8"


def _positive_count_bucket(count: int) -> str:
    if count <= 1:
        return "singleton"
    if count == 2:
        return "double"
    return "multi3+"


def _recovery_signature(candidate: dict[str, Any], rank: int, positive_count: int) -> str:
    provenance = candidate.get("provenance") or {}
    extra_provenance = provenance.get("extra_provenance") or {}
    parser_metadata = candidate.get("parser_metadata") or {}
    return "|".join(
        [
            _normalize_source_name(candidate.get("source")),
            _rank_bucket(rank),
            _positive_count_bucket(positive_count),
            "point_first" if provenance.get("point_first_prompt") else "non_point_first",
            "decoupled" if provenance.get("decoupled_point_native_decode") else "coupled",
            str(parser_metadata.get("resolved_click_mode") or "none"),
            str(parser_metadata.get("resolved_bbox_mode") or "none"),
            "bbox_reconciled"
            if bool((extra_provenance.get("bbox_reconciliation") or {}).get("applied"))
            else "bbox_native",
        ]
    )


def _load_eval_rows(eval_root: Path | None, split: str) -> dict[str, dict[str, Any]]:
    if eval_root is None:
        return {}
    path = eval_root / f"evaluation_{split}_per_sample.jsonl"
    if not path.exists():
        return {}
    return {row["sample_id"]: row for row in _load_jsonl(path)}


def _positive_signal(candidate: dict[str, Any]) -> float:
    components = candidate.get("reward", {}).get("components", {}) or {}
    return _safe_float(components.get("click_inside_target"), 0.0) + _safe_float(components.get("iou"), 0.0)


def _singleton_case_record(
    split: str,
    sample: dict[str, Any],
    current_row: dict[str, Any] | None,
    failed_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    candidates = sample.get("candidates", [])
    if len(candidates) < 2:
        return None
    rewards = [_safe_float(c.get("reward", {}).get("total_reward", 0.0), 0.0) for c in candidates]
    baseline_reward = rewards[0]
    positive_indices = [idx for idx, reward in enumerate(rewards) if idx != 0 and reward > baseline_reward + 1e-9]
    if len(positive_indices) != 1:
        return None
    best_idx = positive_indices[0]
    best_candidate = candidates[best_idx]
    best_source = _normalize_source_name(best_candidate.get("source"))
    best_bbox = _bbox(best_candidate)
    best_click = _click(best_candidate)
    same_source_iou = 0.0
    same_source_click = float("inf")
    structured_count = 0
    max_iou_to_any = 0.0
    min_click_to_structured = float("inf")
    for idx, candidate in enumerate(candidates):
        if idx == best_idx:
            continue
        source = _normalize_source_name(candidate.get("source"))
        bbox_iou = _bbox_iou(best_bbox, _bbox(candidate))
        click_distance = _click_distance(best_click, _click(candidate))
        max_iou_to_any = max(max_iou_to_any, bbox_iou)
        if source == "structured_sampled_t0p6":
            structured_count += 1
            min_click_to_structured = min(min_click_to_structured, click_distance)
        if source == best_source:
            same_source_iou = max(same_source_iou, bbox_iou)
            same_source_click = min(same_source_click, click_distance)
    dom_match = best_candidate.get("dom_match") or {}
    record = {
        "split": split,
        "sample_id": sample.get("sample_id"),
        "best_source": best_source,
        "best_rank": best_idx + 1,
        "oracle_gap": rewards[best_idx] - baseline_reward,
        "positive_signal": _positive_signal(best_candidate),
        "click_inside_target": _safe_float((best_candidate.get("reward", {}).get("components", {}) or {}).get("click_inside_target"), 0.0),
        "iou_reward": _safe_float((best_candidate.get("reward", {}).get("components", {}) or {}).get("iou"), 0.0),
        "dom_best_iou": _safe_float(dom_match.get("best_iou"), 0.0),
        "signature": _recovery_signature(best_candidate, best_idx + 1, 1),
        "structured_count": structured_count,
        "max_iou_to_any": max_iou_to_any,
        "min_click_to_structured": 0.0 if min_click_to_structured == float("inf") else min_click_to_structured,
        "max_same_source_iou": same_source_iou,
        "min_same_source_click": 0.0 if same_source_click == float("inf") else same_source_click,
        "current_reranked_source": current_row.get("reranked_source") if current_row else None,
        "current_reranked_reward": _safe_float(current_row.get("reranked_reward"), 0.0) if current_row else None,
        "failed_reranked_source": failed_row.get("reranked_source") if failed_row else None,
        "failed_reranked_reward": _safe_float(failed_row.get("reranked_reward"), 0.0) if failed_row else None,
    }
    record["point_first_target"] = bool(
        best_source == "point_first_sampled_t0p7"
        and best_idx + 1 >= 7
        and structured_count >= 3
        and max_iou_to_any <= 1e-9
        and record["min_click_to_structured"] >= 80.0
    )
    record["structured_singleton_target"] = bool(
        best_source == "structured_sampled_t0p6"
        and 3 <= best_idx + 1 <= 4
        and structured_count >= 2
        and record["max_same_source_iou"] <= 0.05
        and record["min_same_source_click"] >= 80.0
    )
    return record


def _load_pair_rows(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(path):
        grouped[row["sample_id"]].append(row)
    return grouped


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root)
    current_root = Path(args.current_reranker_root)
    failed_root = Path(args.failed_reranker_root) if args.failed_reranker_root else None
    current_pairs = _load_pair_rows(Path(args.current_pairs_path)) if args.current_pairs_path else {}
    failed_pairs = _load_pair_rows(Path(args.failed_pairs_path)) if args.failed_pairs_path else {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict[str, Any]]] = {}
    train_point_first_positive_signals: list[float] = []
    train_point_first_negative_signals: list[float] = []
    train_target_pair_summary: dict[str, Any] = {}

    for split in args.splits:
        candidate_path = candidate_root / split / f"candidates_{split}.jsonl"
        current_rows = _load_eval_rows(current_root, split)
        failed_rows = _load_eval_rows(failed_root, split) if failed_root else {}
        records: list[dict[str, Any]] = []
        for sample in _load_jsonl(candidate_path):
            record = _singleton_case_record(
                split,
                sample,
                current_rows.get(str(sample.get("sample_id"))),
                failed_rows.get(str(sample.get("sample_id"))),
            )
            if record is None:
                continue
            records.append(record)

            if split == "train" and record["point_first_target"]:
                if record["oracle_gap"] > 0.0:
                    train_point_first_positive_signals.append(record["positive_signal"])
            if split == "train":
                candidates = sample.get("candidates", [])
                rewards = [_safe_float(c.get("reward", {}).get("total_reward", 0.0), 0.0) for c in candidates]
                baseline_reward = rewards[0] if rewards else 0.0
                structured_indices = [
                    idx for idx, candidate in enumerate(candidates) if _normalize_source_name(candidate.get("source")) == "structured_sampled_t0p6"
                ]
                for idx, candidate in enumerate(candidates):
                    if _normalize_source_name(candidate.get("source")) != "point_first_sampled_t0p7":
                        continue
                    if idx + 1 < 7 or len(structured_indices) < 3:
                        continue
                    point_bbox = _bbox(candidate)
                    point_click = _click(candidate)
                    max_iou_to_any = 0.0
                    min_click_to_structured = float("inf")
                    for other_idx, other_candidate in enumerate(candidates):
                        if other_idx == idx:
                            continue
                        max_iou_to_any = max(max_iou_to_any, _bbox_iou(point_bbox, _bbox(other_candidate)))
                        if other_idx in structured_indices:
                            min_click_to_structured = min(
                                min_click_to_structured,
                                _click_distance(point_click, _click(other_candidate)),
                            )
                    if max_iou_to_any > 1e-9 or min_click_to_structured < 80.0:
                        continue
                    if rewards[idx] <= baseline_reward + 1e-9:
                        train_point_first_negative_signals.append(_positive_signal(candidate))

        split_records[split] = records

    train_targets = [
        record
        for record in split_records.get("train", [])
        if record["point_first_target"] or record["structured_singleton_target"]
    ]
    for record in train_targets:
        train_target_pair_summary[record["sample_id"]] = {
            "signature": record["signature"],
            "current_pair_types": dict(Counter(row.get("pair_type", "unknown") for row in current_pairs.get(record["sample_id"], []))),
            "failed_pair_types": dict(Counter(row.get("pair_type", "unknown") for row in failed_pairs.get(record["sample_id"], []))),
        }

    official_targets = [
        record
        for split in ("test_task", "test_website", "test_domain")
        for record in split_records.get(split, [])
        if record["point_first_target"] or record["structured_singleton_target"]
    ]

    summary = {
        "point_first_positive_signal_threshold_recommendation": 0.05,
        "point_first_gap_threshold_recommendation": 0.02,
        "structured_singleton_signal_threshold_recommendation": 1.20,
        "structured_singleton_gap_threshold_recommendation": 0.35,
        "train_point_first_positive_signal_count": len(train_point_first_positive_signals),
        "train_point_first_negative_signal_count": len(train_point_first_negative_signals),
        "train_point_first_positive_signals": sorted(train_point_first_positive_signals, reverse=True),
        "train_point_first_negative_signal_max": max(train_point_first_negative_signals, default=0.0),
        "official_target_cases": official_targets,
        "train_target_cases": train_targets,
        "train_target_pair_summary": train_target_pair_summary,
    }
    (output_dir / "conditional_singleton_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Mind2Web Stage-B Conditional Singleton Analysis",
        "",
        "## High-Signal Singleton Conditions",
        "",
        "- point-first singleton condition: `point_first_sampled_t0p7`, rank `7+`, at least `3` structured decoys, zero bbox-overlap to the pool, min structured click distance `>= 80`, positive signal `>= 0.05`, oracle gap `>= 0.02`",
        "- structured singleton condition: `structured_sampled_t0p6`, rank `3/4`, singleton positive, at least `2` same-source structured decoys, max same-source IoU `<= 0.05`, min same-source click distance `>= 80`, positive signal `>= 1.20`, oracle gap `>= 0.35`",
        "",
        "## Point-First False-Alarm Prior",
        "",
        f"- train isolated point-first positives above threshold: {len([v for v in train_point_first_positive_signals if v >= 0.05])}",
        f"- train isolated point-first negatives above threshold: {sum(1 for v in train_point_first_negative_signals if v >= 0.05)}",
        f"- max train isolated point-first negative signal: {max(train_point_first_negative_signals, default=0.0):.4f}",
        "",
        "## Official Singleton Cases That Still Matter",
        "",
    ]

    for record in sorted(official_targets, key=lambda row: (row["split"], -row["oracle_gap"])):
        lines.append(
            "- "
            + (
                f"{record['split']} {record['sample_id']} | source={record['best_source']} rank={record['best_rank']} "
                f"gap={record['oracle_gap']:.4f} signal={record['positive_signal']:.4f} "
                f"current={record['current_reranked_source']}:{(record['current_reranked_reward'] or 0.0):.4f} "
                f"failed={record['failed_reranked_source']}:{(record['failed_reranked_reward'] or 0.0):.4f}"
            )
        )

    lines.extend(["", "## Train Coverage For Target Signatures", ""])
    for record in sorted(train_targets, key=lambda row: (-row["oracle_gap"], row["sample_id"])):
        pair_summary = train_target_pair_summary.get(record["sample_id"], {})
        lines.append(
            "- "
            + (
                f"{record['sample_id']} | signature={record['signature']} gap={record['oracle_gap']:.4f} "
                f"current_pairs={pair_summary.get('current_pair_types', {})} "
                f"failed_pairs={pair_summary.get('failed_pair_types', {})}"
            )
        )

    (output_dir / "conditional_singleton_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
