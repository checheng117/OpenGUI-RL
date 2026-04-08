from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.data.schemas import BBox
from gui_grounding.data.schemas import GroundingSample
from gui_grounding.models.qwen2_vl_grounding import (
    _build_web_mobile_hotspot_instruction,
    _ensure_bbox_contains_click,
    _refine_edge_click_in_bbox,
    _resolve_point,
)


def test_refine_edge_click_in_bbox_moves_top_left_click_inward():
    bbox = BBox(x1=100.0, y1=200.0, x2=200.0, y2=300.0)
    refined_click, metadata = _refine_edge_click_in_bbox(
        pred_click=(100.0, 200.0),
        pred_bbox=bbox,
        threshold=0.2,
        interior_position=0.45,
    )

    assert refined_click == (145.0, 245.0)
    assert metadata is not None
    assert metadata["applied"] is True
    assert metadata["reason"] == "top_left_edge_click_moved_inward"


def test_refine_edge_click_in_bbox_keeps_interior_click():
    bbox = BBox(x1=100.0, y1=200.0, x2=200.0, y2=300.0)
    refined_click, metadata = _refine_edge_click_in_bbox(
        pred_click=(160.0, 255.0),
        pred_bbox=bbox,
        threshold=0.2,
        interior_position=0.45,
    )

    assert refined_click == (160.0, 255.0)
    assert metadata is not None
    assert metadata["applied"] is False
    assert metadata["reason"] == "click_not_in_top_left_band"


def test_build_web_mobile_hotspot_instruction_for_mobile_icon():
    sample = GroundingSample(
        sample_id="s1",
        dataset_name="screenspot_v2",
        split="test",
        image_path="/tmp/example.png",
        instruction="tap the weather icon",
        platform="mobile",
        metadata={"element_type": "icon"},
    )

    instruction = _build_web_mobile_hotspot_instruction(sample)

    assert "clickable or tappable hotspot" in instruction
    assert "left or top edge" in instruction
    assert "icon button or glyph" in instruction


def test_build_web_mobile_hotspot_instruction_omits_desktop():
    sample = GroundingSample(
        sample_id="s2",
        dataset_name="screenspot_v2",
        split="test",
        image_path="/tmp/example.png",
        instruction="open settings",
        platform="desktop",
        metadata={"element_type": "text"},
    )

    assert _build_web_mobile_hotspot_instruction(sample) == ""


def test_ensure_bbox_contains_click_expands_without_moving_click():
    bbox = BBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
    reconciled_bbox, metadata = _ensure_bbox_contains_click(
        pred_bbox=bbox,
        pred_click=(42.0, 25.0),
    )

    assert reconciled_bbox is not None
    assert reconciled_bbox.as_tuple() == (10.0, 20.0, 42.0, 40.0)
    assert metadata is not None
    assert metadata["applied"] is True
    assert metadata["reason"] == "expanded_bbox_to_include_primary_click"


def test_resolve_point_accepts_normalized_coordinates():
    point, mode = _resolve_point([0.25, 0.5], coord_size=(200, 100))

    assert point == (50.0, 50.0)
    assert mode == "normalized"
