#!/usr/bin/env python3
"""Apply interpretable post-hoc source gating to existing Stage-B candidate pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply post-hoc source gating to candidate JSONL files.")
    parser.add_argument("--input", type=str, required=True, help="Input sample-level candidate JSONL")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument(
        "--drop-sources",
        nargs="*",
        default=["point_first_structured", "point_native_primary"],
        help="Normalized source prefixes to remove from export",
    )
    return parser.parse_args()


def _normalize_source_name(source: str | None) -> str:
    source = str(source or "")
    for prefix in (
        "structured_sampled_t0p6",
        "point_first_sampled_t0p7",
        "point_first_structured",
        "point_native_primary",
        "hybrid_point_structured",
        "stagea_first_choice",
    ):
        if source.startswith(prefix):
            return prefix
    return source or "unknown"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    drop_sources = {_normalize_source_name(source) for source in args.drop_sources}
    sample_rows = _load_jsonl(input_path)
    flat_rows: list[dict] = []
    dropped_hist: Counter[str] = Counter()
    kept_hist: Counter[str] = Counter()

    gated_rows: list[dict] = []
    for row in sample_rows:
        gated_candidates = []
        for candidate in row.get("candidates", []):
            normalized_source = _normalize_source_name(candidate.get("source"))
            if normalized_source in drop_sources:
                dropped_hist[normalized_source] += 1
                continue
            updated = dict(candidate)
            gating_meta = dict(updated.get("gating_metadata") or {})
            gating_meta.update(
                {
                    "posthoc_source_gate_kept": True,
                    "posthoc_source_gate_rule": "drop_harmful_sources_v1",
                    "normalized_source": normalized_source,
                }
            )
            updated["gating_metadata"] = gating_meta
            gated_candidates.append(updated)
            kept_hist[normalized_source] += 1

        for rank, candidate in enumerate(gated_candidates, start=1):
            candidate["rank"] = rank

        updated_row = dict(row)
        updated_row["top_k"] = len(gated_candidates)
        updated_row["candidates"] = gated_candidates
        gated_rows.append(updated_row)

        for candidate in gated_candidates:
            flat_rows.append(
                {
                    "sample_id": updated_row.get("sample_id"),
                    "split": updated_row.get("split"),
                    "instruction": updated_row.get("instruction"),
                    "image_path": updated_row.get("image_path"),
                    "image_width": updated_row.get("image_width"),
                    "image_height": updated_row.get("image_height"),
                    **candidate,
                }
            )

    candidates_path = output_dir / f"candidates_{args.split}.jsonl"
    flat_path = output_dir / f"candidates_{args.split}_flat.jsonl"
    summary_path = output_dir / f"summary_{args.split}.json"
    _save_jsonl(gated_rows, candidates_path)
    _save_jsonl(flat_rows, flat_path)

    summary = {
        "split": args.split,
        "input_path": str(input_path),
        "output_path": str(candidates_path),
        "num_samples": len(gated_rows),
        "avg_candidates_per_sample": sum(len(row.get("candidates", [])) for row in gated_rows) / max(len(gated_rows), 1),
        "drop_sources": sorted(drop_sources),
        "kept_source_histogram": dict(kept_hist),
        "dropped_source_histogram": dict(dropped_hist),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
