"""Compact DOM/OCR-style candidate representation helpers for Mind2Web."""

from __future__ import annotations

import html
import math
import re
from typing import Any

from gui_grounding.data.schemas import BBox, CandidateElement, GroundingSample

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "for",
    "from",
    "at",
    "by",
    "and",
    "or",
    "is",
    "be",
    "are",
    "with",
    "that",
    "this",
    "click",
    "select",
    "type",
    "hover",
    "into",
    "your",
    "you",
    "new",
    "find",
    "buy",
    "get",
    "open",
    "go",
    "choose",
    "pick",
    "up",
    "down",
    "as",
    "it",
    "its",
    "than",
    "then",
    "all",
    "time",
    "most",
    "show",
    "me",
}
_INTERACTIVE_TAGS = {
    "a",
    "button",
    "input",
    "select",
    "textarea",
    "option",
    "label",
    "li",
    "td",
    "img",
    "span",
}
_INTERACTIVE_ROLES = {
    "button",
    "link",
    "tab",
    "checkbox",
    "radio",
    "combobox",
    "textbox",
    "option",
    "menuitem",
    "switch",
}
_ATTR_KEYS = (
    "role",
    "type",
    "placeholder",
    "title",
    "aria_label",
    "name",
    "id",
    "value",
    "input_value",
)


def _clean_text(text: str | None, *, max_chars: int | None = None) -> str:
    normalized = html.unescape(str(text or ""))
    normalized = " ".join(normalized.split())
    if max_chars is not None and len(normalized) > max_chars:
        return normalized[: max_chars - 1].rstrip() + "…"
    return normalized


def _tokenize(text: str | None) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall((text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def candidate_attribute_summary(candidate: CandidateElement, *, max_chars: int = 80) -> str:
    values: list[str] = []
    attrs = candidate.attributes or {}
    for key in _ATTR_KEYS:
        value = _clean_text(attrs.get(key))
        if not value:
            continue
        if value.lower() in {existing.lower() for existing in values}:
            continue
        if key in {"role", "type", "name", "id"}:
            values.append(f"{key}={value}")
        else:
            values.append(value)
    return _clean_text(" | ".join(values), max_chars=max_chars)


def candidate_primary_text(candidate: CandidateElement, *, max_chars: int = 120) -> str:
    primary = _clean_text(candidate.text, max_chars=max_chars)
    if primary:
        return primary
    return candidate_attribute_summary(candidate, max_chars=max_chars)


def normalize_candidate_bbox(
    bbox: BBox | None,
    image_size: tuple[int, int],
) -> list[float]:
    if bbox is None:
        return [0.0, 0.0, 0.0, 0.0]
    width, height = image_size
    return [
        round(float(bbox.x1) / max(float(width), 1.0), 4),
        round(float(bbox.y1) / max(float(height), 1.0), 4),
        round(float(bbox.x2) / max(float(width), 1.0), 4),
        round(float(bbox.y2) / max(float(height), 1.0), 4),
    ]


def _candidate_area_ratio(candidate: CandidateElement, image_size: tuple[int, int]) -> float:
    if candidate.bbox is None:
        return 1.0
    width, height = image_size
    return float(candidate.bbox.area) / max(float(width * height), 1.0)


def _candidate_rank_features(
    sample: GroundingSample,
    candidate: CandidateElement,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    instruction_tokens = _tokenize(sample.instruction)
    primary_text = candidate_primary_text(candidate)
    attribute_summary = candidate_attribute_summary(candidate)
    combined_tokens = _tokenize(" ".join(part for part in (primary_text, attribute_summary, candidate.tag) if part))
    overlap = len(instruction_tokens & combined_tokens)
    role = _clean_text(candidate.attributes.get("role") if candidate.attributes else "")
    interactive = 1 if candidate.tag.lower() in _INTERACTIVE_TAGS or role.lower() in _INTERACTIVE_ROLES else 0
    clickable = 1 if _clean_text(candidate.attributes.get("is_clickable") if candidate.attributes else "") else 0
    area_ratio = _candidate_area_ratio(candidate, image_size)
    area_penalty = 0.0
    if area_ratio > 0.5:
        area_penalty = 3.0
    elif area_ratio > 0.2:
        area_penalty = 1.5
    elif area_ratio > 0.05:
        area_penalty = 0.5
    text_present = 1 if primary_text else 0
    score = (
        overlap * 3.0
        + interactive * 0.8
        + clickable * 0.4
        + text_present * 0.15
        - area_penalty
    )
    return {
        "score": round(score, 4),
        "overlap": overlap,
        "interactive": interactive,
        "clickable": clickable,
        "area_ratio": area_ratio,
        "primary_text": primary_text,
        "attribute_summary": attribute_summary,
    }


def _candidate_sort_key(
    sample: GroundingSample,
    candidate: CandidateElement,
    image_size: tuple[int, int],
) -> tuple[float, int, int, float, float, float, str]:
    features = _candidate_rank_features(sample, candidate, image_size)
    bbox = candidate.bbox
    x1 = float(bbox.x1) if bbox is not None else math.inf
    y1 = float(bbox.y1) if bbox is not None else math.inf
    return (
        float(features["score"]),
        int(features["overlap"]),
        int(features["interactive"]) + int(features["clickable"]),
        -float(features["area_ratio"]),
        -y1,
        -x1,
        str(candidate.element_id),
    )


def build_candidate_prompt_context(
    sample: GroundingSample,
    image_size: tuple[int, int],
    *,
    max_candidates: int = 32,
) -> dict[str, Any]:
    raw_candidates = list(sample.dom_candidates or [])
    if max_candidates > 0:
        raw_candidates = raw_candidates[:max_candidates]
    if not raw_candidates:
        return {
            "entries": [],
            "candidate_prompt_block": "",
            "target_slot": None,
            "candidate_count": 0,
        }

    sorted_candidates = sorted(
        raw_candidates,
        key=lambda candidate: _candidate_sort_key(sample, candidate, image_size),
        reverse=True,
    )

    entries: list[dict[str, Any]] = []
    target_slot: int | None = None
    for slot, candidate in enumerate(sorted_candidates, start=1):
        features = _candidate_rank_features(sample, candidate, image_size)
        bbox_norm = normalize_candidate_bbox(candidate.bbox, image_size)
        primary_text = str(features["primary_text"])
        attribute_summary = str(features["attribute_summary"])
        line = (
            f"[{slot}] tag={candidate.tag or 'unknown'} "
            f"node={candidate.element_id or 'unknown'} "
            f"box={bbox_norm}"
        )
        if primary_text:
            line += f' text="{primary_text}"'
        if attribute_summary and attribute_summary.lower() not in primary_text.lower():
            line += f' attrs="{attribute_summary}"'
        entries.append(
            {
                "slot": slot,
                "element_id": candidate.element_id,
                "tag": candidate.tag,
                "bbox": candidate.bbox,
                "bbox_normalized": bbox_norm,
                "text": primary_text,
                "attribute_summary": attribute_summary,
                "rank_score": features["score"],
                "overlap": features["overlap"],
                "prompt_line": line,
            }
        )
        if target_slot is None and sample.target_element_id and candidate.element_id == str(sample.target_element_id):
            target_slot = slot

    if target_slot is None and sample.target_bbox is not None:
        for entry in entries:
            bbox = entry["bbox"]
            if bbox is None:
                continue
            if bbox.as_tuple() == sample.target_bbox.as_tuple():
                target_slot = int(entry["slot"])
                break

    prompt_block = (
        "Candidate anchors (sorted by DOM/text relevance and geometry, not by dataset label order):\n"
        + "\n".join(entry["prompt_line"] for entry in entries)
        + "\n"
    )
    return {
        "entries": entries,
        "candidate_prompt_block": prompt_block,
        "target_slot": target_slot,
        "candidate_count": len(entries),
    }


def resolve_candidate_slot_entry(
    candidate_entries: list[dict[str, Any]],
    predicted_slot: int | None,
) -> dict[str, Any] | None:
    if predicted_slot is None or predicted_slot < 1:
        return None
    index = int(predicted_slot) - 1
    if index < 0 or index >= len(candidate_entries):
        return None
    return candidate_entries[index]
