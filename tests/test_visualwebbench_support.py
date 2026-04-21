"""Tests for VisualWebBench supplementary benchmark support."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_grounding.data.visualwebbench_dataset import _normalized_bbox_to_absolute
from gui_grounding.evaluation.visualwebbench_metrics import (
    aggregate_visualwebbench_records,
    map_prediction_to_choice,
    score_visualwebbench_grounding,
)


def test_visualwebbench_bbox_conversion_from_normalized_coordinates():
    bbox = _normalized_bbox_to_absolute(
        [0.25, 0.10, 0.50, 0.40],
        image_width=1280,
        image_height=640,
    )
    assert bbox is not None
    assert bbox.as_tuple() == (320.0, 64.0, 640.0, 256.0)


def test_visualwebbench_choice_mapping_prefers_point_inside_candidate():
    predicted_choice_index, details = map_prediction_to_choice(
        predicted_bbox=[300, 100, 420, 240],
        predicted_click_point=[350, 150],
        candidate_boxes=[
            [0, 0, 100, 100],
            [300, 120, 420, 260],
            [500, 100, 650, 260],
        ],
    )
    assert predicted_choice_index == 1
    assert details["mode"] == "point_inside_single"


def test_visualwebbench_scoring_uses_bbox_iou_fallback_when_point_misses():
    score = score_visualwebbench_grounding(
        predicted_bbox=[300, 100, 420, 240],
        predicted_click_point=[1000, 1000],
        predicted_action_type="click",
        candidate_boxes=[
            [0, 0, 100, 100],
            [305, 105, 415, 235],
            [500, 100, 650, 260],
        ],
        target_choice_index=1,
        image_size=[1280, 640],
        task_type="element_ground",
        website="example.com",
    )
    assert score["predicted_choice_index"] == 1
    assert score["official_choice_correct"] is True
    assert score["point_in_box"] is False
    assert score["iou"] > 0.7
    assert score["choice_mapping_mode"] == "bbox_max_iou"


def test_visualwebbench_aggregate_metrics_counts_choice_accuracy():
    overall, subgroup_metrics = aggregate_visualwebbench_records(
        [
            {
                "status": "ok",
                "bbox_proposal": [0, 0, 10, 10],
                "click_point": [5, 5],
                "action_type_valid": True,
                "json_parse_success": True,
                "predicted_choice_valid": True,
                "official_choice_correct": True,
                "point_in_box": True,
                "iou_at_0_5": True,
                "iou": 1.0,
                "candidate_slot_predicted": False,
                "candidate_slot_grounded": False,
                "task_type": "element_ground",
                "target_area_bucket": "small",
                "distractor_overlap_bucket": "low",
            },
            {
                "status": "ok",
                "bbox_proposal": None,
                "click_point": None,
                "action_type_valid": False,
                "json_parse_success": False,
                "predicted_choice_valid": False,
                "official_choice_correct": False,
                "point_in_box": False,
                "iou_at_0_5": False,
                "iou": 0.0,
                "candidate_slot_predicted": True,
                "candidate_slot_grounded": True,
                "task_type": "action_ground",
                "target_area_bucket": "tiny",
                "distractor_overlap_bucket": "high",
            },
        ],
        group_fields=["task_type"],
    )
    assert overall["count"] == 2
    assert overall["official_choice_accuracy"] == 0.5
    assert overall["point_accuracy"] == 0.5
    assert overall["candidate_slot_prediction_rate"] == 0.5
    assert subgroup_metrics["task_type"]["element_ground"]["official_choice_accuracy"] == 1.0
