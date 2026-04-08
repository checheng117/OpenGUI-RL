"""Vision-Language Model backbone wrapper for real Qwen-VL inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """Wrapper around a pretrained Vision-Language Model."""

    BACKBONE_REGISTRY = {
        "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen3_vl_2b": "Qwen/Qwen3-VL-2B-Instruct",
        # Legacy alias kept for backward compatibility only.
        "qwen2_vl_2b_legacy": "Qwen/Qwen2-VL-2B-Instruct",
    }

    def __init__(
        self,
        model_name: str = "qwen2_5_vl_3b",
        device: str = "auto",
        torch_dtype: str = "auto",
        load_model: bool = False,
        attn_implementation: str = "sdpa",
        gpu_memory_utilization: float = 0.9,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        log_generate_diagnostics: bool = False,
        adapter_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.hf_model_id = self._resolve_model_name(model_name)
        self.device = device
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.gpu_memory_utilization = gpu_memory_utilization
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.log_generate_diagnostics = log_generate_diagnostics
        self.adapter_path = adapter_path
        self.resolved_torch_dtype = None
        self.last_generate_stats: dict[str, Any] = {}
        self._model = None
        self._processor = None

        if load_model:
            self._load_model()
        else:
            logger.info(
                "[Scaffold] VLMBackbone initialized without loading weights. "
                "Set load_model=True to load '%s'.",
                self.hf_model_id,
            )

    def _resolve_model_name(self, model_name: str) -> str:
        return self.BACKBONE_REGISTRY.get(model_name, model_name)

    def _resolve_local_model_snapshot(self, token: str | None) -> str | None:
        """Return a cached local snapshot path when the backbone is already present."""
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            return None

        try:
            local_snapshot = snapshot_download(
                repo_id=self.hf_model_id,
                local_files_only=True,
                token=token,
            )
        except Exception:
            return None

        if local_snapshot and Path(local_snapshot).exists():
            return local_snapshot
        return None

    def _load_model(self) -> None:
        """Load the VLM and processor from HuggingFace."""
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Loading VLMBackbone requires transformers and torch."
            ) from exc

        target_device = self.device
        if target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = self._resolve_torch_dtype(target_device)
        self.resolved_torch_dtype = dtype
        # Some environments hang on Xet-backed large file fetches; keep HTTP fallback.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

        model_kwargs: dict[str, Any] = {"dtype": dtype}
        if target_device == "cuda":
            model_kwargs["device_map"] = "auto"
            max_mem = int(self.gpu_memory_utilization * 24)
            model_kwargs["max_memory"] = {0: f"{max_mem}GiB", "cpu": "48GiB"}
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        logger.info(
            "Loading VLM backbone '%s' on device=%s dtype=%s attn_impl=%s min_pixels=%s max_pixels=%s",
            self.hf_model_id,
            target_device,
            str(dtype),
            self.attn_implementation,
            self.min_pixels,
            self.max_pixels,
        )
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            try:
                from huggingface_hub import get_token

                token = get_token()
                if token:
                    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
            except Exception:
                token = None

        pretrained_source = self._resolve_local_model_snapshot(token=token) or self.hf_model_id
        if pretrained_source != self.hf_model_id:
            logger.info("Using cached local backbone snapshot: %s", pretrained_source)

        def _load_once() -> None:
            self._processor = AutoProcessor.from_pretrained(
                pretrained_source,
                trust_remote_code=True,
                token=token,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained_source,
                trust_remote_code=True,
                token=token,
                **model_kwargs,
            )

        try:
            _load_once()
        except Exception as exc:
            fallback_endpoint = os.getenv("GUI_GROUNDING_HF_FALLBACK_ENDPOINT", "https://hf-mirror.com")
            direct_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
            err = str(exc)
            should_retry = (
                direct_endpoint != fallback_endpoint
                and any(
                    marker in err
                    for marker in (
                        "ProxyError",
                        "Connection reset by peer",
                        "ConnectError",
                        "Cannot send a request, as the client has been closed",
                    )
                )
            )
            if not should_retry:
                raise
            logger.warning(
                "Primary HF endpoint failed (%s). Retrying with fallback endpoint: %s",
                type(exc).__name__,
                fallback_endpoint,
            )
            os.environ["HF_ENDPOINT"] = fallback_endpoint
            _load_once()

        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("Loading adapter_path requires peft to be installed.") from exc
            logger.info("Loading PEFT adapter from %s", self.adapter_path)
            self._model = PeftModel.from_pretrained(
                self._model,
                self.adapter_path,
                is_trainable=False,
            )

        if target_device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        self.device = target_device
        logger.info("VLM backbone loaded successfully.")

    def _resolve_torch_dtype(self, target_device: str):
        import torch

        if self.torch_dtype == "auto":
            if target_device == "cuda":
                return torch.bfloat16
            return torch.float32
        if self.torch_dtype == "bfloat16":
            return torch.bfloat16
        if self.torch_dtype == "float16":
            return torch.float16
        if self.torch_dtype == "float32":
            return torch.float32
        if self.torch_dtype == "fp16":
            return torch.float16
        if self.torch_dtype == "bf16":
            return torch.bfloat16
        raise ValueError(f"Unsupported torch_dtype='{self.torch_dtype}'.")

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

    def _build_messages(self, images: list, prompts: list[str]) -> list[list[dict[str, Any]]]:
        messages = []
        for image, prompt in zip(images, prompts):
            image_ref = image
            if isinstance(image, Path):
                image_ref = str(image)
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_ref},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )
        return messages

    def generate(self, images: list, prompts: list[str], max_new_tokens: int = 128, temperature: float = 0.0, num_return_sequences: int = 1) -> list[str]:
        """Generate text outputs from image+instruction prompts."""
        if not self.is_loaded:
            self._load_model()

        import torch
        from qwen_vl_utils import process_vision_info

        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length.")

        messages = self._build_messages(images, prompts)
        texts = [
            self._processor.apply_chat_template(  # type: ignore[union-attr]
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = int(self.max_pixels)

        model_inputs = self._processor(**processor_kwargs)  # type: ignore[operator]

        if self.device == "cuda":
            model_inputs = model_inputs.to("cuda")

        image_grid_thw = getattr(model_inputs, "image_grid_thw", None)
        image_grid_list = image_grid_thw.tolist() if image_grid_thw is not None else None
        pixel_values = getattr(model_inputs, "pixel_values", None)
        pixel_values_shape = list(pixel_values.shape) if pixel_values is not None else None
        self.last_generate_stats = {
            "batch_size": len(images),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "num_return_sequences": int(num_return_sequences),
            "pixel_values_shape": pixel_values_shape,
            "image_grid_thw": image_grid_list,
            "attn_implementation": self.attn_implementation,
            "dtype": str(self.resolved_torch_dtype),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        if self.log_generate_diagnostics:
            logger.info("Qwen generate diagnostics: %s", self.last_generate_stats)

        with torch.inference_mode():
            generated_ids = self._model.generate(  # type: ignore[union-attr]
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                do_sample=temperature > 0,
                num_return_sequences=num_return_sequences,
            )

        if num_return_sequences == 1:
            input_ids = model_inputs.input_ids
            trimmed_ids = [
                output_ids[len(input_id):]
                for input_id, output_ids in zip(input_ids, generated_ids)
            ]
        else:
            input_len = model_inputs.input_ids.shape[1]
            trimmed_ids = [output_ids[input_len:] for output_ids in generated_ids]

        decoded = self._processor.batch_decode(  # type: ignore[union-attr]
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded
