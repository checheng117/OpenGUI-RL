"""CLIP-based grid grounding fallback for real single-sample inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from gui_grounding.data.schemas import BBox, GroundingSample, PredictionResult
from gui_grounding.models.base_model import BaseGroundingModel
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


def _infer_action_type(instruction: str) -> str:
    lower = instruction.lower()
    if any(k in lower for k in ("type", "enter", "input")):
        return "type"
    if "select" in lower:
        return "select"
    if "hover" in lower:
        return "hover"
    return "click"


class CLIPGridGroundingModel(BaseGroundingModel):
    """Real (non-dummy) grounding via CLIP patch-text matching."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        grid_cols: int = 6,
        grid_rows: int = 4,
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self._torch = torch
        self._model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name
        logger.info(
            "Loaded CLIP grid grounding model '%s' on device=%s (grid=%dx%d)",
            model_name,
            device,
            grid_cols,
            grid_rows,
        )

    def _build_grid_patches(
        self,
        image: Image.Image,
    ) -> tuple[list[Image.Image], list[tuple[float, float, float, float]]]:
        width, height = image.size
        patches: list[Image.Image] = []
        boxes: list[tuple[float, float, float, float]] = []
        cell_w = width / self.grid_cols
        cell_h = height / self.grid_rows

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x1 = int(round(c * cell_w))
                y1 = int(round(r * cell_h))
                x2 = int(round((c + 1) * cell_w))
                y2 = int(round((r + 1) * cell_h))
                box = (float(x1), float(y1), float(x2), float(y2))
                patches.append(image.crop((x1, y1, x2, y2)))
                boxes.append(box)
        return patches, boxes

    def predict_with_details(
        self,
        sample: GroundingSample,
    ) -> tuple[PredictionResult, str, dict[str, Any]]:
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        patches, boxes = self._build_grid_patches(image)

        with self._torch.inference_mode():
            inputs = self._processor(
                text=[sample.instruction],
                images=patches,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self._model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self._model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            if not hasattr(image_features, "norm"):
                image_features = image_features.pooler_output
            if not hasattr(text_features, "norm"):
                text_features = text_features.pooler_output

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            probs = logits.softmax(dim=0).squeeze(-1)
            best_idx = int(probs.argmax().item())
            confidence = float(probs[best_idx].item())

        x1, y1, x2, y2 = boxes[best_idx]
        pred_bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
        pred = PredictionResult(
            sample_id=sample.sample_id,
            predicted_action_type=_infer_action_type(sample.instruction),
            predicted_bbox=pred_bbox,
            predicted_click_point=pred_bbox.center,
            predicted_element_id=None,
            confidence=confidence,
        )
        parsed = {
            "backend": "clip_grid",
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "best_patch_index": best_idx,
            "best_patch_bbox": [x1, y1, x2, y2],
            "patch_confidence": confidence,
        }
        raw_text = f"clip_grid best_patch={best_idx} conf={confidence:.4f}"
        return pred, raw_text, parsed

    def predict(
        self,
        sample: GroundingSample,
        **kwargs,
    ) -> PredictionResult:
        pred, _, _ = self.predict_with_details(sample)
        return pred

    def predict_batch(
        self,
        batch: dict[str, Any],
        **kwargs,
    ) -> list[PredictionResult]:
        samples = batch.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("batch['samples'] must be a list[GroundingSample]")
        outputs: list[PredictionResult] = []
        for sample in samples:
            outputs.append(self.predict(sample))
        return outputs

    def generate_candidates(
        self,
        sample: GroundingSample,
        num_candidates: int = 5,
        **kwargs,
    ) -> list[PredictionResult]:
        # Deterministic backend: return the same prediction repeated.
        pred = self.predict(sample)
        return [pred] * num_candidates
