"""Grounding head: predicts bounding boxes or click points.

Takes hidden representations from :class:`VLMBackbone` and outputs
spatial predictions for the target UI element.
"""

from __future__ import annotations

from typing import Optional

from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class GroundingHead:
    """Predict target bounding box or click point from VLM features.

    Two modes are supported:

    1. **Regression** — directly predict (x1, y1, x2, y2) or (cx, cy).
    2. **Selection** — score candidate bounding boxes and select the best.

    TODO(stage-2): Implement as a proper ``nn.Module`` with a small MLP
    or cross-attention layer on top of VLM hidden states.
    """

    def __init__(self, mode: str = "regression", hidden_dim: int = 768) -> None:
        assert mode in ("regression", "selection"), f"Unknown mode: {mode}"
        self.mode = mode
        self.hidden_dim = hidden_dim
        logger.info("[Scaffold] GroundingHead created (mode=%s, hidden_dim=%d)", mode, hidden_dim)

    def forward(self, hidden_states, candidates=None):
        """Compute grounding predictions.

        Parameters
        ----------
        hidden_states : Tensor
            Batch of hidden representations from the VLM.
        candidates : list[list[BBox]], optional
            Pre-extracted candidate boxes (required for selection mode).

        Returns
        -------
        dict with keys ``bbox_pred`` and/or ``click_pred``.

        TODO(stage-2): Replace stub with real nn.Module forward.
        """
        logger.debug("[Scaffold] GroundingHead.forward — returning dummy predictions.")
        batch_size = 1
        return {
            "bbox_pred": [[0.0, 0.0, 100.0, 100.0]] * batch_size,
            "click_pred": [[50.0, 50.0]] * batch_size,
        }
