"""Tests for Mind2Web dataset adapter — field mapping and robustness.

These tests use synthetic rows that match the real HF dataset schema
so they run offline without downloading the actual dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_grounding.data.mind2web_dataset import (
    _parse_bbox_xywh,
    _parse_candidate,
    _parse_operation,
)
from gui_grounding.data.schemas import ActionType, BBox


# -----------------------------------------------------------------------
# _parse_bbox_xywh
# -----------------------------------------------------------------------

class TestParseBboxXywh:
    def test_normal(self):
        bbox = _parse_bbox_xywh("100,200,50,30")
        assert bbox is not None
        assert bbox.x1 == pytest.approx(100)
        assert bbox.y1 == pytest.approx(200)
        assert bbox.x2 == pytest.approx(150)
        assert bbox.y2 == pytest.approx(230)

    def test_float_coords(self):
        bbox = _parse_bbox_xywh("283.1875,220.390625,93.59375,33")
        assert bbox is not None
        assert bbox.x2 == pytest.approx(283.1875 + 93.59375)

    def test_zero_size(self):
        bbox = _parse_bbox_xywh("10,20,0,0")
        assert bbox is not None
        assert bbox.area == 0

    def test_negative_width(self):
        assert _parse_bbox_xywh("10,20,-5,30") is None

    def test_empty_string(self):
        assert _parse_bbox_xywh("") is None

    def test_malformed(self):
        assert _parse_bbox_xywh("not,a,bbox") is None

    def test_too_few_parts(self):
        assert _parse_bbox_xywh("10,20") is None


# -----------------------------------------------------------------------
# _parse_operation
# -----------------------------------------------------------------------

class TestParseOperation:
    def test_click(self):
        at, val = _parse_operation('{"op": "CLICK", "value": ""}')
        assert at == ActionType.CLICK
        assert val == ""

    def test_type(self):
        at, val = _parse_operation('{"op": "TYPE", "value": "hello world"}')
        assert at == ActionType.TYPE
        assert val == "hello world"

    def test_select(self):
        at, val = _parse_operation('{"op": "SELECT", "value": "option1"}')
        assert at == ActionType.SELECT

    def test_unknown_op(self):
        at, val = _parse_operation('{"op": "DRAG", "value": ""}')
        assert at is None

    def test_invalid_json(self):
        at, val = _parse_operation("not json")
        assert at is None
        assert val == ""


# -----------------------------------------------------------------------
# _parse_candidate
# -----------------------------------------------------------------------

class TestParseCandidate:
    def _make_candidate_json(self, **kwargs) -> str:
        attrs = {
            "backend_node_id": "123",
            "bounding_box_rect": "100,200,50,30",
            **(kwargs.get("extra_attrs", {})),
        }
        obj = {
            "tag": kwargs.get("tag", "button"),
            "attributes": json.dumps(attrs),
            "backend_node_id": "123",
            **{k: v for k, v in kwargs.items() if k not in ("tag", "extra_attrs")},
        }
        return json.dumps(obj)

    def test_basic(self):
        cand = _parse_candidate(self._make_candidate_json())
        assert cand is not None
        assert cand.element_id == "123"
        assert cand.tag == "button"
        assert cand.bbox is not None
        assert cand.bbox.x1 == pytest.approx(100)
        assert cand.bbox.x2 == pytest.approx(150)

    def test_with_aria_label(self):
        cand = _parse_candidate(
            self._make_candidate_json(extra_attrs={"aria_label": "Submit form"})
        )
        assert "Submit form" in cand.text

    def test_with_cleaned_html_text(self):
        cand = _parse_candidate(
            self._make_candidate_json(extra_attrs={}),
            cleaned_html='<button backend_node_id="123"><text backend_node_id="124">Search flights</text></button>',
        )
        assert "Search flights" in cand.text

    def test_missing_bbox(self):
        attrs = {"backend_node_id": "999"}
        obj = {"tag": "span", "attributes": json.dumps(attrs), "backend_node_id": "999"}
        cand = _parse_candidate(json.dumps(obj))
        assert cand is not None
        assert cand.bbox is None

    def test_invalid_json(self):
        assert _parse_candidate("not valid json") is None


# -----------------------------------------------------------------------
# GroundingSample mapping (offline, with a fake row)
# -----------------------------------------------------------------------

class TestRowToSample:
    """Test _row_to_sample logic using a synthetic HF-like row."""

    def _make_row(self, **overrides) -> dict:
        pos_attrs = json.dumps({
            "backend_node_id": "42",
            "bounding_box_rect": "100,200,80,40",
            "aria_label": "Login button",
        })
        pos_cand = json.dumps({
            "tag": "button",
            "attributes": pos_attrs,
            "is_original_target": True,
            "is_top_level_target": True,
            "backend_node_id": "42",
        })

        row = {
            "action_uid": "abc-123",
            "annotation_id": "ann-456",
            "confirmed_task": "Click the login button",
            "operation": '{"op": "CLICK", "value": ""}',
            "pos_candidates": [pos_cand],
            "neg_candidates": [],
            "target_action_reprs": "[button] Login -> CLICK",
            "target_action_index": "0",
            "screenshot": Image.new("RGB", (1280, 800), "white"),
            "website": "example",
            "domain": "General",
            "subdomain": "Auth",
            "action_reprs": ["[button] Login -> CLICK"],
        }
        row.update(overrides)
        return row

    def test_basic_mapping(self):
        from gui_grounding.data.mind2web_dataset import Mind2WebDataset

        # We'll call _row_to_sample directly, bypassing HF loading
        adapter = Mind2WebDataset.__new__(Mind2WebDataset)
        adapter.split = "train"
        adapter.max_candidates = 32
        adapter.cache_screenshots = False
        adapter.screenshot_dir = Path("/tmp/test_screenshots")
        adapter._screenshots = {}

        sample, screenshot = adapter._row_to_sample(self._make_row())

        assert sample.sample_id == "mind2web_train_abc-123"
        assert sample.instruction == "Click the login button"
        assert sample.action_type == "click"
        assert sample.target_element_id == "42"
        assert sample.target_bbox is not None
        assert sample.target_bbox.x1 == pytest.approx(100)
        assert sample.target_bbox.y1 == pytest.approx(200)
        assert sample.target_bbox.x2 == pytest.approx(180)  # 100 + 80
        assert sample.target_bbox.y2 == pytest.approx(240)  # 200 + 40
        assert sample.click_point is not None
        assert sample.click_point[0] == pytest.approx(140)  # center x
        assert sample.website == "example"
        assert sample.domain == "General"
        assert sample.platform == "web"
        assert screenshot is not None

    def test_missing_pos_candidates(self):
        from gui_grounding.data.mind2web_dataset import Mind2WebDataset

        adapter = Mind2WebDataset.__new__(Mind2WebDataset)
        adapter.split = "train"
        adapter.max_candidates = 32
        adapter.cache_screenshots = False
        adapter.screenshot_dir = Path("/tmp/test_screenshots")
        adapter._screenshots = {}

        row = self._make_row(pos_candidates=[])
        sample, _ = adapter._row_to_sample(row)

        assert sample.target_bbox is None
        assert sample.click_point is None
        assert sample.target_element_id is None

    def test_type_action_with_value(self):
        from gui_grounding.data.mind2web_dataset import Mind2WebDataset

        adapter = Mind2WebDataset.__new__(Mind2WebDataset)
        adapter.split = "train"
        adapter.max_candidates = 32
        adapter.cache_screenshots = False
        adapter.screenshot_dir = Path("/tmp/test_screenshots")
        adapter._screenshots = {}

        row = self._make_row(
            operation='{"op": "TYPE", "value": "New York"}',
            confirmed_task="Type New York into search",
        )
        sample, _ = adapter._row_to_sample(row)

        assert sample.action_type == "type"
        assert sample.metadata["typed_value"] == "New York"

    def test_no_screenshot(self):
        from gui_grounding.data.mind2web_dataset import Mind2WebDataset

        adapter = Mind2WebDataset.__new__(Mind2WebDataset)
        adapter.split = "train"
        adapter.max_candidates = 32
        adapter.cache_screenshots = False
        adapter.screenshot_dir = Path("/tmp/test_screenshots")
        adapter._screenshots = {}

        row = self._make_row(screenshot=None)
        sample, screenshot = adapter._row_to_sample(row)

        assert sample.image_path == ""
        assert screenshot is None
