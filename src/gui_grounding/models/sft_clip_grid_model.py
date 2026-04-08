"""CLIP-grid SFT model for Stage A baseline training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class SFTCLIPGridModel(nn.Module):
    """Trainable CLIP-based model for action + grid-cell prediction."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_actions: int = 4,
        grid_rows: int = 4,
        grid_cols: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_grids = grid_rows * grid_cols
        self.num_actions = num_actions

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        projection_dim = int(getattr(self.clip_model.config, "projection_dim", 512))
        fusion_dim = projection_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.grid_head = nn.Linear(hidden_dim, self.num_grids)

    def _extract_embeds(self, model_outputs):
        image_embeds = getattr(model_outputs, "image_embeds", None)
        text_embeds = getattr(model_outputs, "text_embeds", None)

        if image_embeds is None:
            image_embeds = model_outputs.vision_model_output.pooler_output
        if text_embeds is None:
            text_embeds = model_outputs.text_model_output.pooler_output
        return image_embeds, text_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
        image_embeds, text_embeds = self._extract_embeds(outputs)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        fused = torch.cat(
            [
                image_embeds,
                text_embeds,
                torch.abs(image_embeds - text_embeds),
                image_embeds * text_embeds,
            ],
            dim=-1,
        )
        hidden = self.fusion(fused)
        action_logits = self.action_head(hidden)
        grid_logits = self.grid_head(hidden)
        return {"action_logits": action_logits, "grid_logits": grid_logits}
