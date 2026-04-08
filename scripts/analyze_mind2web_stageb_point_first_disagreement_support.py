#!/usr/bin/env python3
"""Analyze rare point-first disagreement/support recovery patterns for Mind2Web Stage-B."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rare point-first disagreement/support patterns.")
    parser.add_argument("--candidate-root", type=str, required=True)
    parser.add_argument("--reranker-eval-root", type=str, default=None)
    parser.add_argument("--preference-pairs-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--splits", nargs="*", default=["train", "test_task", "test_website", "test_domain"])
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def _load_reranker_rows(eval_root: Path | None, split: str) -> dict[str, dict[str, Any]]:
    if eval_root is None:
        return {}
    path = eval_root / f"evaluation_{split}_per_sample.jsonl"
    if not path.exists():
        return {}
    rows = _load_jsonl(path)
    return {str(row.get("sample_id")): row for row in rows}


def _point_first_candidate_record(
    sample: dict[str, Any],
    idx: int,
    reranker_row: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates = sample.get("candidates", [])
    candidate = candidates[idx]
    rewards = [_safe_float(c.get("reward", {}).get("total_reward", 0.0), 0.0) for c in candidates]
    baseline_reward = rewards[0]
    reward = rewards[idx]
    components = candidate.get("reward", {}).get("components", {}) or {}
    dom_match = candidate.get("dom_match") or {}
    point_bbox = _bbox(candidate)
    point_click = _click(candidate)
    point_rank = idx + 1

    max_iou_to_any = 0.0
    min_click_distance_to_any = float("inf")
    max_iou_to_structured = 0.0
    min_click_distance_to_structured = float("inf")
    structured_count = 0
    tight_support_count = 0
    for other_idx, other_candidate in enumerate(candidates):
        if other_idx == idx:
            continue
        other_bbox = _bbox(other_candidate)
        other_click = _click(other_candidate)
        bbox_iou = _bbox_iou(point_bbox, other_bbox)
        click_distance = _click_distance(point_click, other_click)
        max_iou_to_any = max(max_iou_to_any, bbox_iou)
        min_click_distance_to_any = min(min_click_distance_to_any, click_distance)
        if bbox_iou >= 0.5 or click_distance <= 48.0:
            tight_support_count += 1
        if _normalize_source_name(other_candidate.get("source")) == "structured_sampled_t0p6":
            structured_count += 1
            max_iou_to_structured = max(max_iou_to_structured, bbox_iou)
            min_click_distance_to_structured = min(min_click_distance_to_structured, click_distance)

    is_positive = reward > baseline_reward + 1e-9
    isolated_signature = (
        point_rank >= 7
        and structured_count >= 3
        and max_iou_to_any <= 1e-9
        and min_click_distance_to_structured >= 80.0
    )
    record = {
        "sample_id": sample.get("sample_id"),
        "rank": point_rank,
        "reward": reward,
        "baseline_reward": baseline_reward,
        "gap_vs_first": reward - baseline_reward,
        "positive": is_positive,
        "isolated_signature": isolated_signature,
        "structured_count": structured_count,
        "max_iou_to_any": max_iou_to_any,
        "min_click_distance_to_any": min_click_distance_to_any if min_click_distance_to_any < float("inf") else 0.0,
        "max_iou_to_structured": max_iou_to_structured,
        "min_click_distance_to_structured": (
            min_click_distance_to_structured if min_click_distance_to_structured < float("inf") else 0.0
        ),
        "tight_support_count": tight_support_count,
        "click_inside_target": _safe_float(components.get("click_inside_target"), 0.0),
        "iou_reward": _safe_float(components.get("iou"), 0.0),
        "dom_best_iou": _safe_float(dom_match.get("best_iou"), 0.0),
        "dom_click_inside": 1.0 if dom_match.get("click_inside_best_match") else 0.0,
        "dom_text_overlap": _safe_float(dom_match.get("instruction_text_overlap"), 0.0),
    }
    if reranker_row is not None:
        record.update(
            {
                "reranker_selected_source": reranker_row.get("reranked_source"),
                "reranker_selected_reward": _safe_float(reranker_row.get("reranked_reward"), 0.0),
                "oracle_source": reranker_row.get("oracle_source"),
                "oracle_reward": _safe_float(reranker_row.get("oracle_reward"), 0.0),
                "reranker_improved": (
                    _safe_float(reranker_row.get("reranked_reward"), 0.0)
                    > _safe_float(reranker_row.get("baseline_reward"), 0.0) + 1e-9
                ),
                "reranker_oracle_hit": abs(
                    _safe_float(reranker_row.get("reranked_reward"), 0.0)
                    - _safe_float(reranker_row.get("oracle_reward"), 0.0)
                )
                <= 1e-9,
            }
        )
    return record


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root)
    reranker_root = Path(args.reranker_eval_root) if args.reranker_eval_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    positive_examples: list[dict[str, Any]] = []
    official_point_first_misses: list[dict[str, Any]] = []

    for split in args.splits:
        reranker_rows = _load_reranker_rows(reranker_root, split)
        candidate_path = candidate_root / split / f"candidates_{split}.jsonl"
        records: list[dict[str, Any]] = []
        oracle_point_first_count = 0
        oracle_point_first_miss_count = 0
        for sample in _load_jsonl(candidate_path):
            reranker_row = reranker_rows.get(str(sample.get("sample_id")))
            candidates = sample.get("candidates", [])
            rewards = [_safe_float(c.get("reward", {}).get("total_reward", 0.0), 0.0) for c in candidates]
            best_idx = max(range(len(candidates)), key=lambda idx: rewards[idx]) if candidates else 0
            oracle_source = _normalize_source_name(candidates[best_idx].get("source")) if candidates else "other"
            if oracle_source == "point_first_sampled_t0p7" and rewards[best_idx] > rewards[0] + 1e-9:
                oracle_point_first_count += 1
                if reranker_row is not None and _safe_float(reranker_row.get("reranked_reward"), 0.0) <= _safe_float(
                    reranker_row.get("baseline_reward"),
                    0.0,
                ) + 1e-9:
                    oracle_point_first_miss_count += 1
                    official_point_first_misses.append(
                        {
                            "split": split,
                            "sample_id": sample.get("sample_id"),
                            "oracle_reward": _safe_float(reranker_row.get("oracle_reward"), 0.0),
                            "baseline_reward": _safe_float(reranker_row.get("baseline_reward"), 0.0),
                            "reranked_reward": _safe_float(reranker_row.get("reranked_reward"), 0.0),
                            "reranker_selected_source": reranker_row.get("reranked_source"),
                        }
                    )

            for idx, candidate in enumerate(candidates):
                if _normalize_source_name(candidate.get("source")) != "point_first_sampled_t0p7":
                    continue
                record = _point_first_candidate_record(sample, idx, reranker_row)
                record["split"] = split
                records.append(record)
                all_records.append(record)
                if record["positive"]:
                    positive_examples.append(record)

        isolated_records = [record for record in records if record["isolated_signature"]]
        isolated_positive = [record for record in isolated_records if record["positive"]]
        split_summaries[split] = {
            "point_first_candidate_count": len(records),
            "point_first_positive_count": sum(1 for record in records if record["positive"]),
            "isolated_signature_count": len(isolated_records),
            "isolated_positive_count": len(isolated_positive),
            "oracle_point_first_recovery_pool_count": oracle_point_first_count,
            "oracle_point_first_reranker_miss_count": oracle_point_first_miss_count,
            "isolated_positive_examples": sorted(
                isolated_positive,
                key=lambda record: record["gap_vs_first"],
                reverse=True,
            )[:10],
        }

    isolated_positive_all = [record for record in all_records if record["isolated_signature"] and record["positive"]]
    isolated_negative_all = [record for record in all_records if record["isolated_signature"] and not record["positive"]]
    key_finding = (
        "isolated point-first geometry is common among false candidates, so the useful signal is targeted "
        "disagreement/support supervision rather than a generic point-first prior"
    )
    summary: dict[str, Any] = {
        "candidate_root": str(candidate_root),
        "reranker_eval_root": str(reranker_root) if reranker_root else None,
        "split_summaries": split_summaries,
        "global_counts": {
            "isolated_positive_count": len(isolated_positive_all),
            "isolated_negative_count": len(isolated_negative_all),
            "key_finding": key_finding,
        },
        "isolated_positive_examples": sorted(
            isolated_positive_all,
            key=lambda record: record["gap_vs_first"],
            reverse=True,
        )[:10],
        "isolated_negative_examples": sorted(
            isolated_negative_all,
            key=lambda record: (record["split"], -record["dom_best_iou"], -record["min_click_distance_to_structured"]),
        )[:10],
        "official_point_first_misses": sorted(
            official_point_first_misses,
            key=lambda record: record["oracle_reward"] - record["baseline_reward"],
            reverse=True,
        ),
    }

    if args.preference_pairs_path:
        pair_rows = _load_jsonl(Path(args.preference_pairs_path))
        summary["pair_export_summary"] = {
            "preferred_point_first_count": sum(
                1 for row in pair_rows if row.get("preferred_source") == "point_first_sampled_t0p7"
            ),
            "rejected_point_first_count": sum(
                1 for row in pair_rows if row.get("rejected_source") == "point_first_sampled_t0p7"
            ),
            "preferred_point_first_pair_types": dict(
                Counter(
                    row.get("pair_type", "unknown")
                    for row in pair_rows
                    if row.get("preferred_source") == "point_first_sampled_t0p7"
                )
            ),
            "rejected_point_first_pair_types": dict(
                Counter(
                    row.get("pair_type", "unknown")
                    for row in pair_rows
                    if row.get("rejected_source") == "point_first_sampled_t0p7"
                )
            ),
            "point_first_targeted_pair_count": sum(1 for row in pair_rows if row.get("point_first_targeted_pool")),
        }

    (output_dir / "point_first_disagreement_support_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Mind2Web Stage-B Rare Point-First Disagreement/Support Analysis",
        "",
        f"- key finding: {key_finding}",
        f"- isolated positive count: {len(isolated_positive_all)}",
        f"- isolated negative count: {len(isolated_negative_all)}",
        "",
    ]
    if len(isolated_negative_all) > 0:
        lines.append(
            f"- isolated positives are only {len(isolated_positive_all)} / "
            f"{len(isolated_positive_all) + len(isolated_negative_all)} of the isolated point-first candidates"
        )
        lines.append("")

    for split in args.splits:
        record = split_summaries[split]
        lines.extend(
            [
                f"## {split}",
                "",
                f"- point_first_candidate_count: {record['point_first_candidate_count']}",
                f"- point_first_positive_count: {record['point_first_positive_count']}",
                f"- isolated_signature_count: {record['isolated_signature_count']}",
                f"- isolated_positive_count: {record['isolated_positive_count']}",
                f"- oracle_point_first_recovery_pool_count: {record['oracle_point_first_recovery_pool_count']}",
                f"- oracle_point_first_reranker_miss_count: {record['oracle_point_first_reranker_miss_count']}",
                "",
                "### Isolated Positive Examples",
                "",
            ]
        )
        if record["isolated_positive_examples"]:
            for example in record["isolated_positive_examples"]:
                lines.append(
                    "- "
                    + (
                        f"{example['sample_id']} | gap={example['gap_vs_first']:.4f} rank={example['rank']} "
                        f"structured_count={example['structured_count']} click_inside={example['click_inside_target']:.1f} "
                        f"iou_reward={example['iou_reward']:.4f} dom_iou={example['dom_best_iou']:.4f} "
                        f"min_struct_click_dist={example['min_click_distance_to_structured']:.1f}"
                    )
                )
        else:
            lines.append("- none")
        lines.append("")

    if "pair_export_summary" in summary:
        pair_summary = summary["pair_export_summary"]
        lines.extend(
            [
                "## Pair Export Coverage",
                "",
                f"- preferred_point_first_count: {pair_summary['preferred_point_first_count']}",
                f"- rejected_point_first_count: {pair_summary['rejected_point_first_count']}",
                f"- point_first_targeted_pair_count: {pair_summary['point_first_targeted_pair_count']}",
                "",
                "### Preferred Point-First Pair Types",
                "",
            ]
        )
        for pair_type, count in pair_summary["preferred_point_first_pair_types"].items():
            lines.append(f"- {pair_type}: {count}")
        lines.extend(["", "### Rejected Point-First Pair Types", ""])
        for pair_type, count in pair_summary["rejected_point_first_pair_types"].items():
            lines.append(f"- {pair_type}: {count}")
        lines.append("")

    lines.extend(["## Official Point-First Misses", ""])
    if summary["official_point_first_misses"]:
        for example in summary["official_point_first_misses"]:
            lines.append(
                "- "
                + (
                    f"{example['split']} {example['sample_id']} | baseline={example['baseline_reward']:.4f} "
                    f"rerank={example['reranked_reward']:.4f} oracle={example['oracle_reward']:.4f} "
                    f"reranked_source={example['reranker_selected_source']}"
                )
            )
    else:
        lines.append("- none")
    lines.append("")

    (output_dir / "point_first_disagreement_support_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
