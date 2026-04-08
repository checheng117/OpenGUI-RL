#!/usr/bin/env python3
"""Package final benchmark figures and tables for the course deliverables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_JSON = REPO_ROOT / "outputs/screenspot_v2_eval_qwen2_5_vl_3b_model_resized/comparison_vs_baseline.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/final_packaging"

BEFORE_COLOR = "#4C78A8"
AFTER_COLOR = "#F58518"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def metric_triplet(name: str, payload: dict) -> dict:
    return {
        "metric": name,
        "before": payload["before"],
        "after": payload["after"],
        "delta": payload["delta"],
    }


def write_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_grouped_metric_chart(
    categories: list[str],
    before_values: list[float],
    after_values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(categories))
    width = 0.36
    bars_before = ax.bar(x - width / 2, before_values, width, label="Before", color=BEFORE_COLOR)
    bars_after = ax.bar(x + width / 2, after_values, width, label="After", color=AFTER_COLOR)
    annotate_bars(ax, bars_before)
    annotate_bars(ax, bars_after)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    rotation = 15 if max(len(cat) for cat in categories) > 10 else 0
    ax.set_xticklabels(categories, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comparison = load_json(DEFAULT_COMPARISON_JSON)
    figures_dir = DEFAULT_OUTPUT_DIR / "figures"
    tables_dir = DEFAULT_OUTPUT_DIR / "tables"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    overall_metrics = [
        metric_triplet("Point accuracy", comparison["overall"]["point_accuracy"]),
        metric_triplet("IoU@0.5", comparison["overall"]["iou@0.5"]),
        metric_triplet("Mean IoU", comparison["overall"]["mean_iou"]),
        metric_triplet("Action-type validity", comparison["overall"]["action_type_valid_rate"]),
        metric_triplet("Parseable output rate", comparison["overall"]["parseable_output_rate"]),
        metric_triplet("Valid bbox rate", comparison["overall"]["valid_bbox_rate"]),
        metric_triplet("Valid click-point rate", comparison["overall"]["valid_click_point_rate"]),
    ]
    overall_chart_labels = [
        "Point accuracy",
        "IoU@0.5",
        "Mean IoU",
        "Action valid",
        "Parseable",
        "Valid bbox",
        "Valid click",
    ]

    overall_rows = [
        [
            metric["metric"],
            format_float(metric["before"]),
            format_float(metric["after"]),
            f"{metric['delta']:+.4f}",
        ]
        for metric in overall_metrics
    ]
    write_markdown_table(
        tables_dir / "screenspot_v2_core_metrics_summary.md",
        ["Metric", "Before", "After", "Delta"],
        overall_rows,
    )
    write_csv(
        tables_dir / "screenspot_v2_core_metrics_summary.csv",
        ["metric", "before", "after", "delta"],
        [[row[0], row[1], row[2], row[3]] for row in overall_rows],
    )

    plot_grouped_metric_chart(
        overall_chart_labels,
        [metric["before"] for metric in overall_metrics],
        [metric["after"] for metric in overall_metrics],
        "ScreenSpot-v2 Full Benchmark: Overall Before/After",
        "Rate / score",
        figures_dir / "screenspot_v2_overall_before_after.png",
    )

    platform_order = ["desktop", "web", "mobile"]
    plot_grouped_metric_chart(
        platform_order,
        [comparison["platform"][k]["point_accuracy"]["before"] for k in platform_order],
        [comparison["platform"][k]["point_accuracy"]["after"] for k in platform_order],
        "ScreenSpot-v2 Point Accuracy by Platform",
        "Point accuracy",
        figures_dir / "screenspot_v2_platform_point_accuracy_before_after.png",
    )

    element_order = ["text", "icon"]
    plot_grouped_metric_chart(
        element_order,
        [comparison["element_type"][k]["point_accuracy"]["before"] for k in element_order],
        [comparison["element_type"][k]["point_accuracy"]["after"] for k in element_order],
        "ScreenSpot-v2 Point Accuracy by Element Type",
        "Point accuracy",
        figures_dir / "screenspot_v2_element_type_point_accuracy_before_after.png",
    )

    source_order = ["windows", "macos", "ios", "android"]
    plot_grouped_metric_chart(
        source_order,
        [comparison["data_source"][k]["point_accuracy"]["before"] for k in source_order],
        [comparison["data_source"][k]["point_accuracy"]["after"] for k in source_order],
        "ScreenSpot-v2 Point Accuracy by Key Source Split",
        "Point accuracy",
        figures_dir / "screenspot_v2_source_split_point_accuracy_before_after.png",
    )

    breakdown_headers = ["group_type", "group", "count", "before_point_accuracy", "after_point_accuracy", "delta_point_accuracy"]
    breakdown_rows: list[list[object]] = []
    for group_type, keys in [
        ("platform", platform_order),
        ("element_type", element_order),
        ("data_source", source_order),
    ]:
        for key in keys:
            item = comparison[group_type][key]
            breakdown_rows.append(
                [
                    group_type,
                    key,
                    item["count"]["after"],
                    format_float(item["point_accuracy"]["before"]),
                    format_float(item["point_accuracy"]["after"]),
                    f"{item['point_accuracy']['delta']:+.4f}",
                ]
            )

    write_csv(
        tables_dir / "screenspot_v2_point_accuracy_breakdowns.csv",
        breakdown_headers,
        breakdown_rows,
    )
    write_markdown_table(
        tables_dir / "screenspot_v2_point_accuracy_breakdowns.md",
        [header.replace("_", " ").title() for header in breakdown_headers],
        [[str(cell) for cell in row] for row in breakdown_rows],
    )

    manifest = {
        "source_comparison_json": str(DEFAULT_COMPARISON_JSON.relative_to(REPO_ROOT)),
        "figures": {
            "overall_before_after": str((figures_dir / "screenspot_v2_overall_before_after.png").relative_to(REPO_ROOT)),
            "platform_point_accuracy": str((figures_dir / "screenspot_v2_platform_point_accuracy_before_after.png").relative_to(REPO_ROOT)),
            "element_type_point_accuracy": str((figures_dir / "screenspot_v2_element_type_point_accuracy_before_after.png").relative_to(REPO_ROOT)),
            "source_split_point_accuracy": str((figures_dir / "screenspot_v2_source_split_point_accuracy_before_after.png").relative_to(REPO_ROOT)),
        },
        "tables": {
            "core_metrics_md": str((tables_dir / "screenspot_v2_core_metrics_summary.md").relative_to(REPO_ROOT)),
            "core_metrics_csv": str((tables_dir / "screenspot_v2_core_metrics_summary.csv").relative_to(REPO_ROOT)),
            "point_accuracy_breakdowns_md": str((tables_dir / "screenspot_v2_point_accuracy_breakdowns.md").relative_to(REPO_ROOT)),
            "point_accuracy_breakdowns_csv": str((tables_dir / "screenspot_v2_point_accuracy_breakdowns.csv").relative_to(REPO_ROOT)),
        },
    }
    (DEFAULT_OUTPUT_DIR / "packaging_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
