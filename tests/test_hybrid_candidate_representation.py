"""Tests for the Stage-A hybrid candidate representation path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_grounding.data.candidate_representation import build_candidate_prompt_context
from gui_grounding.data.schemas import ActionType, BBox, CandidateElement, GroundingSample
from gui_grounding.models.qwen2_vl_grounding import QwenVLGroundingModel


def _make_sample() -> GroundingSample:
    return GroundingSample(
        sample_id="sample-1",
        dataset_name="mind2web",
        split="train",
        image_path="unused.jpg",
        instruction="Click the save button",
        action_type=ActionType.CLICK,
        target_element_id="42",
        target_bbox=BBox(x1=300, y1=200, x2=420, y2=260),
        click_point=(360, 230),
        dom_candidates=[
            CandidateElement(
                element_id="99",
                bbox=BBox(x1=0, y1=0, x2=1000, y2=600),
                tag="div",
                text="Welcome to account settings",
                attributes={"id": "app"},
            ),
            CandidateElement(
                element_id="42",
                bbox=BBox(x1=300, y1=200, x2=420, y2=260),
                tag="button",
                text="Save changes",
                attributes={"role": "button", "id": "save-btn"},
            ),
        ],
    )


def _make_stub_model() -> QwenVLGroundingModel:
    model = object.__new__(QwenVLGroundingModel)
    model.coordinate_frame = "original"
    model.coordinate_format = "normalized"
    model.point_first_prompt = True
    model.target_field_order = "point_bbox_action"
    model.point_primary_bbox_anchored_prompt = False
    model.use_candidate_anchors = True
    model.max_prompt_candidates = 32
    model.candidate_grounding_from_slot = True
    model.web_mobile_hotspot_prompt = False
    model.decoupled_point_native_decode = False
    model.coordinate_quantization_bins = None
    model.point_native_secondary_bbox_only = False
    model.edge_click_interior_threshold = 0.0
    model.edge_click_interior_position = 0.45
    return model


def test_candidate_prompt_context_finds_target_slot_after_reordering():
    context = build_candidate_prompt_context(_make_sample(), (1000, 600), max_candidates=32)
    assert context["candidate_count"] == 2
    assert context["target_slot"] == 1
    assert "Candidate anchors" in context["candidate_prompt_block"]
    assert 'text="Save changes"' in context["candidate_prompt_block"]


def test_parse_prediction_uses_candidate_slot_anchor_for_grounding():
    model = _make_stub_model()
    pred, parsed = model._parse_prediction(  # type: ignore[attr-defined]
        response_text='{"predicted_click_point":[0.05,0.05],"predicted_bbox":[0.0,0.0,0.1,0.1],"action_type":"click","candidate_slot":2}',
        sample_id="sample-1",
        image_size=(1000, 600),
        candidate_entries=[
            {
                "slot": 1,
                "element_id": "99",
                "bbox": BBox(x1=0, y1=0, x2=1000, y2=600),
                "text": "Welcome to account settings",
            },
            {
                "slot": 2,
                "element_id": "42",
                "bbox": BBox(x1=300, y1=200, x2=420, y2=260),
                "text": "Save changes",
            },
        ],
    )
    assert pred.predicted_candidate_slot == 2
    assert pred.predicted_element_id == "42"
    assert pred.predicted_bbox is not None
    assert pred.predicted_bbox.as_tuple() == pytest.approx((300, 200, 420, 260))
    assert pred.predicted_click_point == pytest.approx((360, 230))
    assert parsed["_candidate_slot_used_for_grounding"] is True
