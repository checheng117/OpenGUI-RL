from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.reward.lightweight_verifier import build_dual_path_candidates, score_dual_path_candidates


def _make_record(
    *,
    sample_id: str,
    click_point,
    bbox_proposal,
    action_type: str = "click",
    confidence: float = 0.99,
    parsed_payload: dict | None = None,
):
    return {
        "sample_id": sample_id,
        "instruction": "test instruction",
        "image_path": "/tmp/example.png",
        "platform": "desktop",
        "element_type": "text",
        "data_source": "test",
        "click_point": click_point,
        "bbox_proposal": bbox_proposal,
        "action_type": action_type,
        "element_hint_id": "target",
        "confidence": confidence,
        "json_parse_success": True,
        "raw_response_nonempty": True,
        "parsed_model_payload": parsed_payload or {},
        "target_bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
        "target_click_point": [50.0, 50.0],
    }


def test_build_dual_path_candidates_adds_hybrid_bbox_expansion():
    point_record = _make_record(
        sample_id="s1",
        click_point=[90.0, 90.0],
        bbox_proposal=[70.0, 70.0, 95.0, 95.0],
        parsed_payload={"_resolved_click_provenance": "point_native_primary_pass"},
    )
    structured_record = _make_record(
        sample_id="s1",
        click_point=[55.0, 55.0],
        bbox_proposal=[40.0, 40.0, 70.0, 70.0],
        confidence=0.95,
    )

    candidates = build_dual_path_candidates(
        point_record,
        structured_record,
        point_artifact_label="point.jsonl",
        structured_artifact_label="structured.jsonl",
    )

    hybrid_candidate = next(c for c in candidates if c["source_path"] == "hybrid_point_structured")
    assert hybrid_candidate["click_point"] == [90.0, 90.0]
    assert hybrid_candidate["bbox_proposal"] == [40.0, 40.0, 90.0, 90.0]
    assert hybrid_candidate["bbox_provenance"] == "structured_single_pass_bbox_expanded_for_point"


def test_verifier_selects_structured_when_override_trigger_fires():
    point_record = _make_record(
        sample_id="s2",
        click_point=[90.0, 90.0],
        bbox_proposal=[50.0, 50.0, 120.0, 120.0],
        confidence=0.99,
        parsed_payload={"_resolved_click_provenance": "point_native_primary_pass"},
    )
    structured_record = _make_record(
        sample_id="s2",
        click_point=[60.0, 60.0],
        bbox_proposal=[55.0, 55.0, 70.0, 70.0],
        confidence=0.95,
    )

    candidates = build_dual_path_candidates(
        point_record,
        structured_record,
        point_artifact_label="point.jsonl",
        structured_artifact_label="structured.jsonl",
    )
    verifier_result = score_dual_path_candidates(candidates)

    assert verifier_result["cross_path_context"]["structured_override_trigger"] is True
    assert verifier_result["selected_source_path"] == "structured_single_pass"


def test_verifier_defaults_to_hybrid_when_paths_are_consistent():
    point_record = _make_record(
        sample_id="s3",
        click_point=[60.0, 60.0],
        bbox_proposal=[50.0, 50.0, 90.0, 90.0],
        confidence=0.99,
        parsed_payload={"_resolved_click_provenance": "point_native_primary_pass"},
    )
    structured_record = _make_record(
        sample_id="s3",
        click_point=[62.0, 62.0],
        bbox_proposal=[55.0, 55.0, 95.0, 95.0],
        confidence=0.95,
    )

    candidates = build_dual_path_candidates(
        point_record,
        structured_record,
        point_artifact_label="point.jsonl",
        structured_artifact_label="structured.jsonl",
    )
    verifier_result = score_dual_path_candidates(candidates)

    assert verifier_result["cross_path_context"]["structured_override_trigger"] is False
    assert verifier_result["selected_source_path"] == "hybrid_point_structured"
