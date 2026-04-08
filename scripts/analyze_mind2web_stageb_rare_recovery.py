#!/usr/bin/env python3
"""Compact rare-recovery analysis for Mind2Web Stage-B candidate pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rare Stage-B recovery patterns.")
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
    parser_metadata = candidate.get("parser_metadata") or {}
    extra_provenance = provenance.get("extra_provenance") or {}
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


def _load_reranker_rows(eval_root: Path | None, split: str) -> dict[str, dict[str, Any]]:
    if eval_root is None:
        return {}
    path = eval_root / f"evaluation_{split}_per_sample.jsonl"
    if not path.exists():
        return {}
    rows = _load_jsonl(path)
    return {str(row.get("sample_id")): row for row in rows}


def _pool_recovery_record(pool: dict[str, Any], reranker_row: dict[str, Any] | None) -> dict[str, Any] | None:
    candidates = pool.get("candidates", [])
    if len(candidates) < 2:
        return None
    rewards = [float(c.get("reward", {}).get("total_reward", 0.0)) for c in candidates]
    baseline_reward = rewards[0]
    positive_indices = [idx for idx, reward in enumerate(rewards) if idx != 0 and reward > baseline_reward + 1e-9]
    if not positive_indices:
        return None
    best_idx = max(positive_indices, key=lambda idx: (rewards[idx], idx))
    best_candidate = candidates[best_idx]
    oracle_gap = float(rewards[best_idx] - baseline_reward)
    improved = None
    oracle_hit = None
    reranked_source = None
    reranked_reward = None
    if reranker_row is not None:
        reranked_reward = float(reranker_row.get("reranked_reward", 0.0))
        improved = reranked_reward > baseline_reward + 1e-9
        oracle_hit = abs(reranked_reward - float(reranker_row.get("oracle_reward", 0.0))) <= 1e-9
        reranked_source = reranker_row.get("reranked_source")
    return {
        "sample_id": pool.get("sample_id"),
        "oracle_source": _normalize_source_name(best_candidate.get("source")),
        "oracle_signature": _recovery_signature(best_candidate, best_idx + 1, len(positive_indices)),
        "oracle_gap": oracle_gap,
        "oracle_rank": int(best_idx + 1),
        "positive_count": int(len(positive_indices)),
        "improved": improved,
        "oracle_hit": oracle_hit,
        "reranked_source": reranked_source,
        "reranked_reward": reranked_reward,
    }


def _format_signature(signature: str) -> str:
    return signature.replace("|", " | ")


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root)
    reranker_root = Path(args.reranker_eval_root) if args.reranker_eval_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits:
        candidate_path = candidate_root / split / f"candidates_{split}.jsonl"
        reranker_rows = _load_reranker_rows(reranker_root, split)
        records: list[dict[str, Any]] = []
        for pool in _load_jsonl(candidate_path):
            record = _pool_recovery_record(pool, reranker_rows.get(str(pool.get("sample_id"))))
            if record is not None:
                records.append(record)
        records_by_split[split] = records

    train_records = records_by_split.get("train", [])
    train_source_counts = dict(Counter(record["oracle_source"] for record in train_records))
    train_signature_counts = dict(Counter(record["oracle_signature"] for record in train_records))

    important_patterns: dict[str, dict[str, Any]] = {}
    split_summaries: dict[str, Any] = {}
    for split, records in records_by_split.items():
        source_counts = Counter(record["oracle_source"] for record in records)
        signature_counts = Counter(record["oracle_signature"] for record in records)
        improved_rate = 0.0
        oracle_hit_rate = 0.0
        evaluated = [record for record in records if record["improved"] is not None]
        if evaluated:
            improved_rate = sum(1.0 for record in evaluated if record["improved"]) / len(evaluated)
            oracle_hit_rate = sum(1.0 for record in evaluated if record["oracle_hit"]) / len(evaluated)
        split_summaries[split] = {
            "headroom_pool_count": len(records),
            "oracle_source_counts": dict(source_counts),
            "oracle_signature_counts": dict(signature_counts),
            "oracle_gap_sum": sum(record["oracle_gap"] for record in records),
            "oracle_gap_max": max((record["oracle_gap"] for record in records), default=0.0),
            "improved_rate": improved_rate,
            "oracle_hit_rate": oracle_hit_rate,
        }

        if split == "train":
            continue
        for record in records:
            signature = record["oracle_signature"]
            pattern = important_patterns.setdefault(
                signature,
                {
                    "signature": signature,
                    "source": record["oracle_source"],
                    "train_count": int(train_signature_counts.get(signature, 0)),
                    "official_count": 0,
                    "official_gap_sum": 0.0,
                    "splits": Counter(),
                    "no_improve_count": 0,
                    "no_oracle_hit_count": 0,
                    "examples": [],
                },
            )
            pattern["official_count"] += 1
            pattern["official_gap_sum"] += float(record["oracle_gap"])
            pattern["splits"][split] += 1
            if record["improved"] is False:
                pattern["no_improve_count"] += 1
            if record["oracle_hit"] is False:
                pattern["no_oracle_hit_count"] += 1
            pattern["examples"].append(
                {
                    "split": split,
                    "sample_id": record["sample_id"],
                    "oracle_gap": record["oracle_gap"],
                    "improved": record["improved"],
                    "oracle_hit": record["oracle_hit"],
                    "reranked_source": record["reranked_source"],
                }
            )

    important_rows = sorted(
        important_patterns.values(),
        key=lambda row: (
            -float(row["official_gap_sum"]),
            int(row["train_count"]),
            -int(row["no_improve_count"]),
            -int(row["no_oracle_hit_count"]),
        ),
    )
    for row in important_rows:
        row["splits"] = dict(row["splits"])
        row["examples"] = sorted(row["examples"], key=lambda item: item["oracle_gap"], reverse=True)[:6]

    summary = {
        "candidate_root": str(candidate_root),
        "reranker_eval_root": str(reranker_root) if reranker_root else None,
        "train_source_counts": train_source_counts,
        "train_signature_counts": train_signature_counts,
        "split_summaries": split_summaries,
        "important_underlearned_patterns": important_rows,
    }
    (output_dir / "rare_recovery_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Mind2Web Stage-B Rare Recovery Analysis",
        "",
        "## Train Headroom Sources",
        "",
    ]
    for source, count in sorted(train_source_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Train Headroom Signatures", ""])
    for signature, count in sorted(train_signature_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {_format_signature(signature)}: {count}")

    for split in args.splits:
        record = split_summaries[split]
        lines.extend(
            [
                "",
                f"## {split}",
                "",
                f"- headroom_pool_count: {record['headroom_pool_count']}",
                f"- oracle_gap_sum: {record['oracle_gap_sum']:.4f}",
                f"- oracle_gap_max: {record['oracle_gap_max']:.4f}",
                f"- improved_rate: {record['improved_rate']:.4f}",
                f"- oracle_hit_rate: {record['oracle_hit_rate']:.4f}",
                "",
                "### Oracle Sources",
                "",
            ]
        )
        for source, count in record["oracle_source_counts"].items():
            lines.append(f"- {source}: {count}")
        lines.extend(["", "### Oracle Signatures", ""])
        for signature, count in record["oracle_signature_counts"].items():
            lines.append(f"- {_format_signature(signature)}: {count}")

    lines.extend(["", "## Important Underlearned Patterns", ""])
    for row in important_rows:
        split_text = ", ".join(f"{split}:{count}" for split, count in sorted(row["splits"].items()))
        lines.append(
            "- "
            + (
                f"{_format_signature(row['signature'])} | train_count={row['train_count']} "
                f"official_count={row['official_count']} official_gap_sum={row['official_gap_sum']:.4f} "
                f"no_improve={row['no_improve_count']} no_oracle_hit={row['no_oracle_hit_count']} "
                f"splits={split_text}"
            )
        )
        for example in row["examples"][:3]:
            lines.append(
                "- "
                + (
                    f"{example['split']} {example['sample_id']} gap={example['oracle_gap']:.4f} "
                    f"improved={example['improved']} oracle_hit={example['oracle_hit']} "
                    f"reranked_source={example['reranked_source']}"
                )
            )

    (output_dir / "rare_recovery_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
