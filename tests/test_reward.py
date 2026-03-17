"""Tests for verifiable reward computation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from gui_grounding.reward.verifiable_reward import (
    VerifiableRewardCalculator,
    action_type_correct,
    bbox_iou,
    click_inside_target,
    element_correct,
    invalid_format_penalty,
)


class TestElementCorrect:
    def test_match(self):
        assert element_correct("btn_1", "btn_1") == 1.0

    def test_mismatch(self):
        assert element_correct("btn_1", "btn_2") == 0.0

    def test_none_pred(self):
        assert element_correct(None, "btn_1") == 0.0

    def test_none_gt(self):
        assert element_correct("btn_1", None) == 0.0


class TestBBoxIoU:
    def test_perfect_overlap(self):
        box = (10, 20, 100, 80)
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0 < iou < 1
        expected = 25 / (100 + 100 - 25)
        assert iou == pytest.approx(expected)

    def test_contained(self):
        iou = bbox_iou((2, 2, 8, 8), (0, 0, 10, 10))
        assert iou == pytest.approx(36 / 100)

    def test_zero_area(self):
        assert bbox_iou((5, 5, 5, 5), (0, 0, 10, 10)) == pytest.approx(0.0)


class TestClickInsideTarget:
    def test_inside(self):
        assert click_inside_target((50, 50), (10, 10, 100, 100)) == 1.0

    def test_outside(self):
        assert click_inside_target((5, 5), (10, 10, 100, 100)) == 0.0

    def test_on_edge(self):
        assert click_inside_target((10, 50), (10, 10, 100, 100)) == 1.0

    def test_corner(self):
        assert click_inside_target((100, 100), (10, 10, 100, 100)) == 1.0


class TestActionTypeCorrect:
    def test_match(self):
        assert action_type_correct("click", "click") == 1.0

    def test_case_insensitive(self):
        assert action_type_correct("Click", "click") == 1.0

    def test_mismatch(self):
        assert action_type_correct("click", "type") == 0.0

    def test_none(self):
        assert action_type_correct(None, "click") == 0.0


class TestInvalidFormatPenalty:
    def test_valid_bbox(self):
        assert invalid_format_penalty(pred_bbox=(10, 20, 100, 80)) == 0.0

    def test_inverted_bbox(self):
        assert invalid_format_penalty(pred_bbox=(100, 80, 10, 20)) == 1.0

    def test_negative_coords(self):
        assert invalid_format_penalty(pred_bbox=(-5, 10, 100, 80)) == 1.0

    def test_no_prediction(self):
        assert invalid_format_penalty() == 1.0

    def test_valid_click(self):
        assert invalid_format_penalty(pred_click=(50, 50)) == 0.0

    def test_out_of_bounds(self):
        assert invalid_format_penalty(pred_bbox=(0, 0, 2000, 100), image_width=1000) == 1.0


class TestVerifiableRewardCalculator:
    def setup_method(self):
        self.calc = VerifiableRewardCalculator()

    def test_perfect_prediction(self):
        result = self.calc.compute(
            sample_id="test_001",
            pred_element_id="btn_1",
            gt_element_id="btn_1",
            pred_bbox=(10, 20, 100, 80),
            gt_bbox=(10, 20, 100, 80),
            pred_click=(55, 50),
            pred_action="click",
            gt_action="click",
        )
        assert result.total_reward > 0
        assert result.components.element_correct == 1.0
        assert result.components.iou == pytest.approx(1.0)
        assert result.components.click_inside_target == 1.0
        assert result.components.action_type_correct == 1.0
        assert result.is_valid_format

    def test_completely_wrong(self):
        result = self.calc.compute(
            sample_id="test_002",
            pred_element_id="btn_999",
            gt_element_id="btn_1",
            pred_bbox=(500, 500, 600, 600),
            gt_bbox=(10, 20, 100, 80),
            pred_click=(550, 550),
            pred_action="type",
            gt_action="click",
        )
        assert result.total_reward < 1.0
        assert result.components.element_correct == 0.0

    def test_custom_weights(self):
        calc = VerifiableRewardCalculator(weights={"element_correct": 2.0, "iou": 0.0})
        result = calc.compute(
            sample_id="test_003",
            pred_element_id="btn_1",
            gt_element_id="btn_1",
        )
        assert result.weights_used["element_correct"] == 2.0

    def test_batch_compute(self):
        predictions = [
            {"sample_id": "s1", "bbox": (10, 20, 100, 80), "action_type": "click"},
            {"sample_id": "s2", "bbox": (50, 50, 150, 150), "action_type": "type"},
        ]
        ground_truths = [
            {"sample_id": "s1", "bbox": (10, 20, 100, 80), "action_type": "click"},
            {"sample_id": "s2", "bbox": (50, 50, 150, 150), "action_type": "type"},
        ]
        results = self.calc.compute_batch(predictions, ground_truths)
        assert len(results) == 2
        assert all(r.total_reward > 0 for r in results)
