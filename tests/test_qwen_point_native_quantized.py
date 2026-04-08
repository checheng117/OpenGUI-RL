"""Tests for quantized point-native Qwen parsing/prompting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_grounding.data.schemas import ActionType, BBox, GroundingSample
from gui_grounding.models.qwen2_vl_grounding import QwenVLGroundingModel


def _make_stub_model() -> QwenVLGroundingModel:
    model = object.__new__(QwenVLGroundingModel)
    model.coordinate_frame = "original"
    model.coordinate_format = "normalized"
    model.point_first_prompt = True
    model.web_mobile_hotspot_prompt = False
    model.decoupled_point_native_decode = True
    model.coordinate_quantization_bins = 1000
    model.point_native_secondary_bbox_only = True
    model.edge_click_interior_threshold = 0.0
    model.edge_click_interior_position = 0.45
    return model


def test_quantized_combined_prediction_parses_point_and_bbox():
    model = _make_stub_model()
    pred, parsed = model._parse_prediction(  # type: ignore[attr-defined]
        response_text='{"point_bin":[500,250],"action_type":"click","bbox_bin":[450,200,550,300]}',
        sample_id="sample-1",
        image_size=(1000, 400),
    )
    assert pred.predicted_action_type == "click"
    assert pred.predicted_click_point is not None
    assert pred.predicted_bbox is not None
    assert pred.predicted_click_point[0] == pytest.approx(500.5005, rel=1e-4)
    assert pred.predicted_click_point[1] == pytest.approx(100.1001, rel=1e-4)
    assert pred.predicted_bbox.x1 < pred.predicted_click_point[0] < pred.predicted_bbox.x2
    assert parsed["_resolved_click_mode"] == "quantized_bin"
    assert parsed["_resolved_bbox_mode"] == "quantized_bin"


def test_quantized_point_native_prompts_use_point_then_bbox_support():
    model = _make_stub_model()
    sample = GroundingSample(
        sample_id="sample-1",
        dataset_name="mind2web",
        split="train",
        image_path="unused.jpg",
        instruction="Click the search button",
        action_type=ActionType.CLICK,
        target_bbox=BBox(x1=100, y1=40, x2=200, y2=100),
        click_point=(150, 70),
    )
    point_prompt = model._build_point_native_prompt(sample, (1000, 400))  # type: ignore[attr-defined]
    support_prompt = model._build_secondary_structure_prompt(  # type: ignore[attr-defined]
        sample,
        (1000, 400),
        primary_click=(150, 70),
    )
    assert '"point_bin": [x_bin, y_bin]' in point_prompt
    assert '"bbox_bin": [x1_bin, y1_bin, x2_bin, y2_bin]' in support_prompt
    assert '"action_type"' not in support_prompt
