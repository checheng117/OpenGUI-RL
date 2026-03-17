"""Action-type prediction head.

Classifies the next action as one of: click, type, select, hover.
"""

from __future__ import annotations

from gui_grounding.constants import ACTION_TYPES, NUM_ACTION_TYPES
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class ActionHead:
    """Classify the GUI action type from VLM hidden states.

    TODO(stage-2): Implement as an ``nn.Module`` with a linear
    classification layer on top of the [CLS] / pooled output.
    """

    def __init__(self, hidden_dim: int = 768) -> None:
        self.hidden_dim = hidden_dim
        self.num_classes = NUM_ACTION_TYPES
        logger.info("[Scaffold] ActionHead created (num_classes=%d)", self.num_classes)

    def forward(self, hidden_states):
        """Predict action-type logits.

        Returns
        -------
        dict with key ``action_logits`` of shape (batch, num_classes).

        TODO(stage-2): Replace stub with real nn.Module forward.
        """
        logger.debug("[Scaffold] ActionHead.forward — returning dummy logits.")
        batch_size = 1
        dummy_logits = [[1.0, 0.0, 0.0, 0.0]] * batch_size
        return {"action_logits": dummy_logits}
