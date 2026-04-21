"""Unified data schemas for GUI grounding samples.

Every dataset adapter should convert raw annotations into these structures
so that downstream training, reward computation, and evaluation use a
single canonical representation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ActionType(str, Enum):
    """Supported GUI action types."""

    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    HOVER = "hover"


class BBox(BaseModel):
    """Axis-aligned bounding box in absolute pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def x2_ge_x1(cls, v: float, info) -> float:
        if "x1" in info.data and v < info.data["x1"]:
            raise ValueError("x2 must be >= x1")
        return v

    @field_validator("y2")
    @classmethod
    def y2_ge_y1(cls, v: float, info) -> float:
        if "y1" in info.data and v < info.data["y1"]:
            raise ValueError("y2 must be >= y1")
        return v

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class CandidateElement(BaseModel):
    """A candidate UI element that might be the grounding target."""

    element_id: str
    bbox: Optional[BBox] = None
    text: str = ""
    tag: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    score: Optional[float] = None


class GroundingSample(BaseModel):
    """Canonical representation of a single GUI grounding sample.

    All dataset adapters (Mind2Web, ScreenSpot-v2, etc.) should convert
    their raw rows into this schema before feeding to models or metrics.
    """

    sample_id: str
    dataset_name: str
    split: str

    image_path: str
    instruction: str

    action_type: Optional[ActionType] = None
    target_element_id: Optional[str] = None
    target_bbox: Optional[BBox] = None
    click_point: Optional[tuple[float, float]] = None

    ocr_text: Optional[str] = None
    dom_candidates: Optional[list[CandidateElement]] = None

    website: Optional[str] = None
    domain: Optional[str] = None
    platform: Optional[str] = None

    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)


class PredictionResult(BaseModel):
    """Model output for a single sample."""

    sample_id: str
    predicted_action_type: Optional[str] = None
    predicted_bbox: Optional[BBox] = None
    predicted_click_point: Optional[tuple[float, float]] = None
    predicted_element_id: Optional[str] = None
    predicted_candidate_slot: Optional[int] = None
    confidence: Optional[float] = None
    candidate_scores: Optional[list[float]] = None


class CandidateAction(BaseModel):
    """A single candidate action produced during candidate generation."""

    candidate_id: str
    action_type: Optional[str] = None
    bbox: Optional[BBox] = None
    click_point: Optional[tuple[float, float]] = None
    element_id: Optional[str] = None
    source: str = "model"
    score: Optional[float] = None
