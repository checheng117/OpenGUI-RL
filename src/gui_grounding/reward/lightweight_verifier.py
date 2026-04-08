"""Dual-path candidate generation and lightweight verifier utilities."""

from __future__ import annotations

import math
from typing import Any

from gui_grounding.constants import ACTION_TYPES
from gui_grounding.reward.verifiable_reward import invalid_format_penalty


DUAL_PATH_CANDIDATE_SCHEMA = {
    "version": "bbox_click_action_v3_dual_path_candidates",
    "primary_fields": ["bbox_proposal", "click_point", "action_type"],
    "candidate_sources": [
        "point_native_primary",
        "structured_single_pass",
        "hybrid_point_structured",
    ],
    "secondary_fields": [
        "candidate_id",
        "source_path",
        "element_hint_id",
        "confidence",
        "click_provenance",
        "bbox_provenance",
        "action_provenance",
        "parser_metadata",
        "source_artifacts",
    ],
}

LIGHTWEIGHT_VERIFIER_SCHEMA = {
    "version": "lightweight_dual_path_verifier_v1",
    "selection_mode": "hybrid_default_with_structured_override",
}

DEFAULT_LIGHTWEIGHT_VERIFIER_CONFIG = {
    "structured_override_confidence_min": 0.95,
    "structured_override_min_click_disagreement_px": 20.0,
    "source_prior_point_native": 0.25,
    "source_prior_structured": 0.10,
    "source_prior_hybrid": 0.85,
    "format_bonus": 0.60,
    "action_valid_bonus": 0.25,
    "click_inside_own_bbox_bonus": 0.40,
    "other_bbox_support_bonus": 0.35,
    "confidence_bonus_weight": 0.20,
    "hybrid_supported_by_structured_bonus": 0.20,
    "structured_override_bonus": 1.50,
    "hybrid_override_penalty": 1.20,
    "point_candidate_penalty": 0.15,
}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_point(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    x = _safe_float(value[0])
    y = _safe_float(value[1])
    if x is None or y is None:
        return None
    return [x, y]


def _as_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    coords = [_safe_float(v) for v in value]
    if any(v is None for v in coords):
        return None
    x1, y1, x2, y2 = coords  # type: ignore[misc]
    x_low, x_high = sorted((x1, x2))
    y_low, y_high = sorted((y1, y2))
    if x_high <= x_low or y_high <= y_low:
        return None
    return [x_low, y_low, x_high, y_high]


def _normalize_action_type(value: Any) -> str | None:
    if value is None:
        return None
    action = str(value).strip().lower()
    if action in ACTION_TYPES:
        return action
    return None


def _click_inside_bbox(click_point: list[float] | None, bbox: list[float] | None) -> bool:
    if click_point is None or bbox is None:
        return False
    x, y = click_point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _click_distance(click_a: list[float] | None, click_b: list[float] | None) -> float:
    if click_a is None or click_b is None:
        return float("inf")
    return math.dist(click_a, click_b)


def _expand_bbox_to_include_click(
    bbox: list[float] | None,
    click_point: list[float] | None,
    delta: float = 12.0,
) -> tuple[list[float] | None, dict[str, Any]]:
    if click_point is None and bbox is None:
        return None, {"applied": False, "reason": "missing_bbox_and_click"}
    if bbox is None:
        click_x, click_y = click_point  # type: ignore[misc]
        return (
            [click_x - delta, click_y - delta, click_x + delta, click_y + delta],
            {
                "applied": True,
                "reason": "derived_bbox_from_click",
                "delta": delta,
            },
        )
    if click_point is None:
        return bbox, {"applied": False, "reason": "missing_click"}
    if _click_inside_bbox(click_point, bbox):
        return bbox, {"applied": False, "reason": "bbox_already_contains_click"}
    click_x, click_y = click_point
    x1, y1, x2, y2 = bbox
    expanded = [min(x1, click_x), min(y1, click_y), max(x2, click_x), max(y2, click_y)]
    return (
        expanded,
        {
            "applied": True,
            "reason": "expanded_bbox_to_include_click",
            "original_bbox": bbox,
            "click_point": click_point,
            "expanded_bbox": expanded,
        },
    )


def _extract_provenance(record: dict[str, Any], key: str, default: str) -> str:
    parsed_payload = record.get("parsed_model_payload") or {}
    value = parsed_payload.get(key)
    return str(value) if value is not None else default


def _build_base_candidate(
    *,
    candidate_id: str,
    source_path: str,
    click_point: list[float] | None,
    bbox_proposal: list[float] | None,
    action_type: str | None,
    element_hint_id: str | None,
    confidence: float | None,
    click_provenance: str,
    bbox_provenance: str,
    action_provenance: str,
    source_artifacts: dict[str, str],
    parser_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_path": source_path,
        "click_point": click_point,
        "bbox_proposal": bbox_proposal,
        "action_type": action_type,
        "action_type_valid": action_type is not None,
        "element_hint_id": element_hint_id,
        "confidence": confidence,
        "click_provenance": click_provenance,
        "bbox_provenance": bbox_provenance,
        "action_provenance": action_provenance,
        "parser_metadata": parser_metadata,
        "source_artifacts": source_artifacts,
    }


def build_dual_path_candidates(
    point_record: dict[str, Any],
    structured_record: dict[str, Any],
    *,
    point_artifact_label: str,
    structured_artifact_label: str,
) -> list[dict[str, Any]]:
    """Build explicit dual-path candidates from saved same-protocol predictions."""
    point_click = _as_point(point_record.get("click_point"))
    point_bbox = _as_bbox(point_record.get("bbox_proposal"))
    point_action = _normalize_action_type(point_record.get("action_type"))
    point_conf = _safe_float(point_record.get("confidence"))

    structured_click = _as_point(structured_record.get("click_point"))
    structured_bbox = _as_bbox(structured_record.get("bbox_proposal"))
    structured_action = _normalize_action_type(structured_record.get("action_type"))
    structured_conf = _safe_float(structured_record.get("confidence"))

    point_candidate = _build_base_candidate(
        candidate_id=f"{point_record['sample_id']}_point_native_primary",
        source_path="point_native_primary",
        click_point=point_click,
        bbox_proposal=point_bbox,
        action_type=point_action,
        element_hint_id=point_record.get("element_hint_id"),
        confidence=point_conf,
        click_provenance=_extract_provenance(
            point_record,
            "_resolved_click_provenance",
            "point_native_primary_pass",
        ),
        bbox_provenance=_extract_provenance(
            point_record,
            "_resolved_bbox_provenance",
            "point_native_support_bbox",
        ),
        action_provenance=_extract_provenance(
            point_record,
            "_resolved_action_provenance",
            "point_native_support_action",
        ),
        source_artifacts={
            "point_native_prediction": point_artifact_label,
        },
        parser_metadata={
            "json_parse_success": bool(point_record.get("json_parse_success")),
            "raw_response_nonempty": bool(point_record.get("raw_response_nonempty")),
            "parsed_model_payload": point_record.get("parsed_model_payload") or {},
        },
    )

    structured_candidate = _build_base_candidate(
        candidate_id=f"{structured_record['sample_id']}_structured_single_pass",
        source_path="structured_single_pass",
        click_point=structured_click,
        bbox_proposal=structured_bbox,
        action_type=structured_action,
        element_hint_id=structured_record.get("element_hint_id"),
        confidence=structured_conf,
        click_provenance="structured_single_pass",
        bbox_provenance="structured_single_pass",
        action_provenance="structured_single_pass",
        source_artifacts={
            "structured_prediction": structured_artifact_label,
        },
        parser_metadata={
            "json_parse_success": bool(structured_record.get("json_parse_success")),
            "raw_response_nonempty": bool(structured_record.get("raw_response_nonempty")),
            "parsed_model_payload": structured_record.get("parsed_model_payload") or {},
        },
    )

    hybrid_bbox, hybrid_bbox_metadata = _expand_bbox_to_include_click(
        structured_bbox,
        point_click,
    )
    hybrid_confidences = [v for v in (point_conf, structured_conf) if v is not None]
    hybrid_conf = min(hybrid_confidences) if hybrid_confidences else None
    hybrid_candidate = _build_base_candidate(
        candidate_id=f"{point_record['sample_id']}_hybrid_point_structured",
        source_path="hybrid_point_structured",
        click_point=point_click,
        bbox_proposal=hybrid_bbox,
        action_type=structured_action or point_action,
        element_hint_id=structured_record.get("element_hint_id") or point_record.get("element_hint_id"),
        confidence=hybrid_conf,
        click_provenance=point_candidate["click_provenance"],
        bbox_provenance=(
            "structured_single_pass_bbox_expanded_for_point"
            if hybrid_bbox_metadata.get("applied")
            else "structured_single_pass_bbox"
        ),
        action_provenance=structured_candidate["action_provenance"] if structured_action is not None else point_candidate["action_provenance"],
        source_artifacts={
            "point_native_prediction": point_artifact_label,
            "structured_prediction": structured_artifact_label,
        },
        parser_metadata={
            "json_parse_success": bool(point_record.get("json_parse_success")) and bool(structured_record.get("json_parse_success")),
            "raw_response_nonempty": bool(point_record.get("raw_response_nonempty")) or bool(structured_record.get("raw_response_nonempty")),
            "bbox_reconciliation": hybrid_bbox_metadata,
            "point_native_parsed_model_payload": point_record.get("parsed_model_payload") or {},
            "structured_parsed_model_payload": structured_record.get("parsed_model_payload") or {},
        },
    )

    return [point_candidate, structured_candidate, hybrid_candidate]


def _candidate_feature_bundle(
    candidate: dict[str, Any],
    *,
    point_candidate: dict[str, Any],
    structured_candidate: dict[str, Any],
) -> dict[str, Any]:
    click_point = candidate.get("click_point")
    bbox_proposal = candidate.get("bbox_proposal")
    source_path = str(candidate.get("source_path"))
    confidence = _safe_float(candidate.get("confidence")) or 0.0
    parser_metadata = candidate.get("parser_metadata") or {}

    other_bbox_support_count = 0
    if source_path != "point_native_primary" and _click_inside_bbox(click_point, point_candidate.get("bbox_proposal")):
        other_bbox_support_count += 1
    if source_path != "structured_single_pass" and _click_inside_bbox(click_point, structured_candidate.get("bbox_proposal")):
        other_bbox_support_count += 1

    features = {
        "parseable_output": bool(parser_metadata.get("json_parse_success")),
        "format_valid": invalid_format_penalty(
            pred_bbox=tuple(bbox_proposal) if bbox_proposal is not None else None,
            pred_click=tuple(click_point) if click_point is not None else None,
        )
        == 0.0,
        "action_type_valid": bool(candidate.get("action_type_valid")),
        "click_inside_own_bbox": _click_inside_bbox(click_point, bbox_proposal),
        "other_bbox_support_count": other_bbox_support_count,
        "confidence": confidence,
    }
    return features


def score_dual_path_candidates(
    candidates: list[dict[str, Any]],
    verifier_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score dual-path candidates and pick the selected candidate."""
    cfg = {**DEFAULT_LIGHTWEIGHT_VERIFIER_CONFIG, **(verifier_config or {})}

    point_candidate = next(c for c in candidates if c["source_path"] == "point_native_primary")
    structured_candidate = next(c for c in candidates if c["source_path"] == "structured_single_pass")
    hybrid_candidate = next(c for c in candidates if c["source_path"] == "hybrid_point_structured")

    click_distance = _click_distance(point_candidate.get("click_point"), structured_candidate.get("click_point"))
    structured_click_in_point_bbox = _click_inside_bbox(
        structured_candidate.get("click_point"),
        point_candidate.get("bbox_proposal"),
    )
    point_click_in_structured_bbox = _click_inside_bbox(
        point_candidate.get("click_point"),
        structured_candidate.get("bbox_proposal"),
    )
    structured_confidence = _safe_float(structured_candidate.get("confidence")) or 0.0
    structured_override_trigger = (
        click_distance >= float(cfg["structured_override_min_click_disagreement_px"])
        and structured_click_in_point_bbox
        and not point_click_in_structured_bbox
        and structured_confidence >= float(cfg["structured_override_confidence_min"])
    )

    cross_path_context = {
        "click_distance_px": click_distance,
        "point_click_in_structured_bbox": point_click_in_structured_bbox,
        "structured_click_in_point_bbox": structured_click_in_point_bbox,
        "structured_confidence": structured_confidence,
        "structured_override_trigger": structured_override_trigger,
    }

    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        features = _candidate_feature_bundle(
            candidate,
            point_candidate=point_candidate,
            structured_candidate=structured_candidate,
        )
        source_path = str(candidate["source_path"])
        score_components = {
            "source_prior": float(
                cfg[
                    {
                        "point_native_primary": "source_prior_point_native",
                        "structured_single_pass": "source_prior_structured",
                        "hybrid_point_structured": "source_prior_hybrid",
                    }[source_path]
                ]
            ),
            "format_bonus": float(cfg["format_bonus"]) if features["format_valid"] else -float(cfg["format_bonus"]),
            "action_valid_bonus": (
                float(cfg["action_valid_bonus"])
                if features["action_type_valid"]
                else -float(cfg["action_valid_bonus"])
            ),
            "click_inside_own_bbox_bonus": (
                float(cfg["click_inside_own_bbox_bonus"])
                if features["click_inside_own_bbox"]
                else -float(cfg["click_inside_own_bbox_bonus"])
            ),
            "other_bbox_support_bonus": float(cfg["other_bbox_support_bonus"]) * float(features["other_bbox_support_count"]),
            "confidence_bonus": float(cfg["confidence_bonus_weight"]) * float(features["confidence"]),
            "hybrid_supported_by_structured_bonus": 0.0,
            "structured_override_bonus": 0.0,
            "hybrid_override_penalty": 0.0,
            "point_candidate_penalty": 0.0,
        }

        if source_path == "hybrid_point_structured" and point_click_in_structured_bbox:
            score_components["hybrid_supported_by_structured_bonus"] = float(cfg["hybrid_supported_by_structured_bonus"])
        if source_path == "structured_single_pass" and structured_override_trigger:
            score_components["structured_override_bonus"] = float(cfg["structured_override_bonus"])
        if source_path == "hybrid_point_structured" and structured_override_trigger:
            score_components["hybrid_override_penalty"] = -float(cfg["hybrid_override_penalty"])
        if source_path == "point_native_primary":
            score_components["point_candidate_penalty"] = -float(cfg["point_candidate_penalty"])

        total_score = sum(score_components.values())
        scored_candidate = {
            **candidate,
            "verifier_features": features,
            "verifier_score_components": score_components,
            "verifier_total_score": total_score,
        }
        scored_candidates.append(scored_candidate)

    scored_candidates.sort(key=lambda row: row["verifier_total_score"], reverse=True)
    selected_candidate = scored_candidates[0]

    return {
        "verifier_schema": LIGHTWEIGHT_VERIFIER_SCHEMA,
        "verifier_config_used": cfg,
        "cross_path_context": cross_path_context,
        "candidates": scored_candidates,
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_source_path": selected_candidate["source_path"],
        "selected_candidate": selected_candidate,
    }
