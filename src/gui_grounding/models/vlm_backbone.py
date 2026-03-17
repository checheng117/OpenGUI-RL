"""Vision-Language Model backbone wrapper.

Encapsulates the image + text encoding so that grounding heads and
action heads receive a unified hidden representation regardless of
the specific VLM used.

TODO(stage-2): Integrate Qwen2-VL processor and model loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VLMOutput:
    """Container for VLM encoder outputs."""

    hidden_states: Any = None
    pooled_output: Any = None
    image_features: Any = None
    text_features: Any = None


class VLMBackbone:
    """Wrapper around a pretrained Vision-Language Model.

    In scaffold mode this returns dummy outputs.  Once a real model is
    loaded, :meth:`encode` produces actual hidden representations.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cpu",
        load_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

        if load_model:
            self._load_model()
        else:
            logger.info(
                "[Scaffold] VLMBackbone initialized without loading weights. "
                "Set load_model=True or call load() to load '%s'.",
                model_name,
            )

    def _load_model(self) -> None:
        """Load the VLM and processor from HuggingFace.

        TODO(stage-2): Implement actual model loading with:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, device_map=self.device, torch_dtype="auto"
            )
        """
        logger.warning(
            "[Scaffold] Model loading not yet implemented. "
            "Will return dummy outputs. Model: %s",
            self.model_name,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(
        self,
        images: list,
        texts: list[str],
    ) -> VLMOutput:
        """Encode a batch of image-text pairs.

        Returns :class:`VLMOutput` with hidden states suitable for
        downstream heads.

        TODO(stage-2): Implement real forward pass through the VLM.
        """
        if not self.is_loaded:
            logger.debug("[Scaffold] Returning dummy VLMOutput.")
            return VLMOutput()

        raise NotImplementedError("TODO(stage-2): real VLM forward pass")

    def generate(
        self,
        images: list,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate text outputs (for models that produce grounding as text).

        TODO(stage-2): Implement generation with the VLM.
        """
        if not self.is_loaded:
            logger.debug("[Scaffold] Returning dummy generation output.")
            return [f"[DUMMY] bbox=[0,0,100,100] action=click"] * len(prompts)

        raise NotImplementedError("TODO(stage-2): real VLM generation")
