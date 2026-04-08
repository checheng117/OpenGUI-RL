#!/usr/bin/env python3
"""Compare two ScreenSpot-v2 evaluation outputs and save before/after artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OVERALL_METRICS = [
    "evaluated_samples",
    "point_accuracy",
    "iou@0.5",
    "mean_iou",
    "action_type_valid_rate",
    "parseable_output_rate",
    "valid_bbox_rate",
    "valid_click_point_rate",
]
GROUP_METRICS = [
    "count",
    "point_accuracy",
    "iou@0.5",
    "mean_iou",
    "action_type_valid_rate",
    "parseable_output_rate",
    "valid_bbox_rate",
    "valid_click_point_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ScreenSpot-v2 evaluation runs")
    parser.add_argument("--before-summary", required=True, type=str)
    parser.add_argument("--before-subgroups", required=True, type=str)
    parser.add_argument("--after-summary", required=True, type=str)
    parser.add_argument("--after-subgroups", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--output-md", required=True, type=str)
    parser.add_argument("--before-label", type=str, default="before")
    parser.add_argument("--after-label", type=str, default="after")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compute_metric_triplet(before: dict, after: dict, keys: list[str]) -> dict:
    triplet: dict[str, dict | int | float | None] = {}
    for key in keys:
        before_value = before.get(key)
        after_value = after.get(key)
        delta = None
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            delta = after_value - before_value
        triplet[key] = {
            "before": before_value,
            "after": after_value,
            "delta": delta,
        }
    return triplet


def _compare_group_section(before: dict, after: dict) -> dict:
    compared: dict[str, dict] = {}
    for group_name in sorted(set(before) | set(after)):
        compared[group_name] = _compute_metric_triplet(
            before.get(group_name, {}),
            after.get(group_name, {}),
            GROUP_METRICS,
        )
    return compared


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_overall_table(comparison: dict, before_label: str, after_label: str) -> list[str]:
    lines = [
        "## Overall",
        "",
        f"| Metric | {before_label} | {after_label} | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in OVERALL_METRICS:
        triplet = comparison["overall"][metric]
        lines.append(
            f"| {metric} | {_fmt(triplet['before'])} | {_fmt(triplet['after'])} | {_fmt(triplet['delta'])} |"
        )
    lines.append("")
    return lines


def _render_group_table(section_name: str, rows: dict, before_label: str, after_label: str) -> list[str]:
    lines = [
        f"## {section_name.replace('_', ' ').title()}",
        "",
        f"| Group | Metric | {before_label} | {after_label} | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for group_name, metrics in rows.items():
        for metric in GROUP_METRICS:
            triplet = metrics[metric]
            lines.append(
                f"| {group_name} | {metric} | {_fmt(triplet['before'])} | {_fmt(triplet['after'])} | {_fmt(triplet['delta'])} |"
            )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()

    before_summary = _load_json(args.before_summary)
    before_subgroups = _load_json(args.before_subgroups)
    after_summary = _load_json(args.after_summary)
    after_subgroups = _load_json(args.after_subgroups)

    comparison = {
        "labels": {
            "before": args.before_label,
            "after": args.after_label,
        },
        "artifacts": {
            "before_summary": args.before_summary,
            "before_subgroups": args.before_subgroups,
            "after_summary": args.after_summary,
            "after_subgroups": args.after_subgroups,
        },
        "overall": _compute_metric_triplet(before_summary, after_summary, OVERALL_METRICS),
        "platform": _compare_group_section(
            before_subgroups.get("platform", {}),
            after_subgroups.get("platform", {}),
        ),
        "element_type": _compare_group_section(
            before_subgroups.get("element_type", {}),
            after_subgroups.get("element_type", {}),
        ),
        "data_source": _compare_group_section(
            before_subgroups.get("data_source", {}),
            after_subgroups.get("data_source", {}),
        ),
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# ScreenSpot-v2 Before/After Comparison",
        "",
        f"- Before: `{args.before_label}`",
        f"- After: `{args.after_label}`",
        "",
    ]
    md_lines.extend(_render_overall_table(comparison, args.before_label, args.after_label))
    md_lines.extend(_render_group_table("platform", comparison["platform"], args.before_label, args.after_label))
    md_lines.extend(
        _render_group_table("element_type", comparison["element_type"], args.before_label, args.after_label)
    )
    md_lines.extend(_render_group_table("data_source", comparison["data_source"], args.before_label, args.after_label))
    output_md.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
