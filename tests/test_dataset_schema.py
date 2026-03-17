"""Tests for data schemas."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from gui_grounding.data.schemas import (
    ActionType,
    BBox,
    CandidateAction,
    CandidateElement,
    GroundingSample,
    PredictionResult,
)


class TestBBox:
    def test_valid_bbox(self):
        bbox = BBox(x1=10, y1=20, x2=100, y2=80)
        assert bbox.as_tuple() == (10, 20, 100, 80)

    def test_center(self):
        bbox = BBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.center == (50, 50)

    def test_area(self):
        bbox = BBox(x1=0, y1=0, x2=10, y2=10)
        assert bbox.area == 100

    def test_invalid_x_raises(self):
        with pytest.raises(Exception):
            BBox(x1=100, y1=0, x2=10, y2=100)

    def test_invalid_y_raises(self):
        with pytest.raises(Exception):
            BBox(x1=0, y1=100, x2=100, y2=10)

    def test_zero_area(self):
        bbox = BBox(x1=5, y1=5, x2=5, y2=5)
        assert bbox.area == 0


class TestGroundingSample:
    def test_minimal_sample(self):
        sample = GroundingSample(
            sample_id="test_001",
            dataset_name="mind2web",
            split="train",
            image_path="/path/to/img.png",
            instruction="Click the button",
        )
        assert sample.sample_id == "test_001"
        assert sample.target_bbox is None

    def test_full_sample(self):
        sample = GroundingSample(
            sample_id="test_002",
            dataset_name="mind2web",
            split="train",
            image_path="/path/to/img.png",
            instruction="Click the login button",
            action_type=ActionType.CLICK,
            target_bbox=BBox(x1=10, y1=20, x2=100, y2=80),
            click_point=(55, 50),
            website="example.com",
            domain="ecommerce",
        )
        assert sample.action_type == "click"
        assert sample.target_bbox.area > 0

    def test_action_type_enum(self):
        assert ActionType.CLICK.value == "click"
        assert ActionType.TYPE.value == "type"
        assert ActionType.SELECT.value == "select"
        assert ActionType.HOVER.value == "hover"

    def test_candidate_element(self):
        elem = CandidateElement(
            element_id="elem_1",
            bbox=BBox(x1=0, y1=0, x2=50, y2=30),
            text="Login",
            tag="button",
        )
        assert elem.element_id == "elem_1"
        assert elem.bbox.area == 1500

    def test_sample_with_candidates(self):
        sample = GroundingSample(
            sample_id="test_003",
            dataset_name="mind2web",
            split="train",
            image_path="img.png",
            instruction="Click login",
            dom_candidates=[
                CandidateElement(element_id="e1", text="Login"),
                CandidateElement(element_id="e2", text="Register"),
            ],
        )
        assert len(sample.dom_candidates) == 2


class TestPredictionResult:
    def test_basic(self):
        pred = PredictionResult(
            sample_id="test_001",
            predicted_action_type="click",
            predicted_bbox=BBox(x1=10, y1=20, x2=100, y2=80),
            confidence=0.95,
        )
        assert pred.confidence == 0.95


class TestCandidateAction:
    def test_basic(self):
        cand = CandidateAction(
            candidate_id="c1",
            action_type="click",
            bbox=BBox(x1=10, y1=20, x2=100, y2=80),
            click_point=(55, 50),
            source="model",
        )
        assert cand.source == "model"
