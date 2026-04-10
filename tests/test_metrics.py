"""Tests for evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from gui_grounding.evaluation.metrics import (
    action_type_accuracy,
    best_of_k_improvement,
    compute_all_metrics,
    element_accuracy,
    invalid_format_rate,
    iou_at_threshold,
    mean_normalized_click_l1,
    mean_iou,
    point_accuracy,
    reranked_gain,
)


class TestElementAccuracy:
    def test_all_correct(self):
        assert element_accuracy(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_none_correct(self):
        assert element_accuracy(["x", "y"], ["a", "b"]) == pytest.approx(0.0)

    def test_partial(self):
        assert element_accuracy(["a", "x", "c"], ["a", "b", "c"]) == pytest.approx(2 / 3)

    def test_empty(self):
        assert element_accuracy([], []) == 0.0


class TestPointAccuracy:
    def test_inside(self):
        acc = point_accuracy([(50, 50)], [(10, 10, 100, 100)])
        assert acc == pytest.approx(1.0)

    def test_outside(self):
        acc = point_accuracy([(5, 5)], [(10, 10, 100, 100)])
        assert acc == pytest.approx(0.0)

    def test_mixed(self):
        acc = point_accuracy(
            [(50, 50), (5, 5)],
            [(10, 10, 100, 100), (10, 10, 100, 100)],
        )
        assert acc == pytest.approx(0.5)

    def test_none_values(self):
        acc = point_accuracy([None, (50, 50)], [(10, 10, 100, 100), (10, 10, 100, 100)])
        assert acc == pytest.approx(1.0)


class TestIoUAtThreshold:
    def test_perfect(self):
        boxes = [(0, 0, 10, 10)]
        assert iou_at_threshold(boxes, boxes, threshold=0.5) == pytest.approx(1.0)

    def test_below_threshold(self):
        pred = [(0, 0, 10, 10)]
        gt = [(100, 100, 200, 200)]
        assert iou_at_threshold(pred, gt, threshold=0.5) == pytest.approx(0.0)


class TestMeanIoU:
    def test_perfect(self):
        boxes = [(10, 20, 100, 80)]
        assert mean_iou(boxes, boxes) == pytest.approx(1.0)

    def test_none_handling(self):
        pred = [None, (10, 20, 100, 80)]
        gt = [(10, 20, 100, 80), (10, 20, 100, 80)]
        assert mean_iou(pred, gt) == pytest.approx(1.0)


class TestActionTypeAccuracy:
    def test_all_correct(self):
        assert action_type_accuracy(["click", "type"], ["click", "type"]) == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert action_type_accuracy(["Click"], ["click"]) == pytest.approx(1.0)


class TestMeanNormalizedClickL1:
    def test_zero_when_exact(self):
        err = mean_normalized_click_l1(
            pred_points=[(50, 25)],
            gt_points=[(50, 25)],
            image_sizes=[(100, 50)],
        )
        assert err == pytest.approx(0.0)

    def test_normalizes_by_image_size(self):
        err = mean_normalized_click_l1(
            pred_points=[(75, 50)],
            gt_points=[(50, 25)],
            image_sizes=[(100, 50)],
        )
        assert err == pytest.approx(0.375)


class TestInvalidFormatRate:
    def test_zero_when_all_predictions_are_valid(self):
        rate = invalid_format_rate(
            pred_bboxes=[(0, 0, 10, 10)],
            pred_points=[(5, 5)],
            image_sizes=[(20, 20)],
        )
        assert rate == pytest.approx(0.0)

    def test_counts_missing_predictions_as_invalid(self):
        rate = invalid_format_rate(
            pred_bboxes=[None, (0, 0, 10, 10)],
            pred_points=[None, (5, 5)],
            image_sizes=[(20, 20), (20, 20)],
        )
        assert rate == pytest.approx(0.5)

    def test_flags_out_of_bounds_clicks(self):
        rate = invalid_format_rate(
            pred_points=[(25, 5)],
            image_sizes=[(20, 20)],
        )
        assert rate == pytest.approx(1.0)


class TestRerankingMetrics:
    def test_best_of_k_improvement(self):
        fc = [True, False, True, False]
        bok = [True, True, True, False]
        gain = best_of_k_improvement(fc, bok)
        assert gain == pytest.approx(0.25)

    def test_reranked_gain(self):
        fc_rewards = [0.5, 0.3, 0.8]
        rr_rewards = [0.7, 0.6, 0.9]
        gain = reranked_gain(fc_rewards, rr_rewards)
        assert gain > 0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        metrics = compute_all_metrics(
            pred_element_ids=["a"],
            gt_element_ids=["a"],
            pred_bboxes=[(0, 0, 10, 10)],
            gt_bboxes=[(0, 0, 10, 10)],
            pred_points=[(5, 5)],
            pred_actions=["click"],
            gt_actions=["click"],
        )
        expected_keys = {
            "element_accuracy", "point_accuracy", "mean_iou",
            "iou@0.5", "iou@0.75", "action_type_accuracy",
        }
        assert expected_keys == set(metrics.keys())
