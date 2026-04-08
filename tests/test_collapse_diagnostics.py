from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_grounding.evaluation.collapse_diagnostics import compute_prediction_collapse_diagnostics


def test_compute_prediction_collapse_diagnostics_flags_repeated_templates(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), color="white").save(image_path)

    records = [
        {
            "image_path": str(image_path),
            "predicted_bbox": [100.0, 100.0, 120.0, 120.0],
            "predicted_click_point": [110.0, 110.0],
            "target_bbox": [20.0, 20.0, 80.0, 60.0],
        },
        {
            "image_path": str(image_path),
            "predicted_bbox": [100.0, 100.0, 120.0, 120.0],
            "predicted_click_point": [110.0, 110.0],
            "target_bbox": [25.0, 25.0, 85.0, 65.0],
        },
        {
            "image_path": str(image_path),
            "predicted_bbox": [90.0, 90.0, 130.0, 130.0],
            "predicted_click_point": [110.0, 110.0],
            "target_bbox": [40.0, 10.0, 120.0, 50.0],
        },
    ]

    diagnostics = compute_prediction_collapse_diagnostics(records)

    assert diagnostics["num_records"] == 3
    assert diagnostics["dominant_bbox"]["value"] == [100.0, 100.0, 120.0, 120.0]
    assert diagnostics["dominant_bbox"]["count"] == 2
    assert diagnostics["dominant_bbox_fraction"] == 2 / 3
    assert diagnostics["dominant_click_point"]["value"] == [110.0, 110.0]
    assert diagnostics["dominant_click_point"]["count"] == 3
    assert diagnostics["collapse_score"] == 1.0
