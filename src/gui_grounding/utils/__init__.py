"""Utility helpers for GUI grounding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gui_grounding.utils.logger import get_logger
from gui_grounding.utils.seed import set_seed

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = ["load_config", "get_logger", "set_seed"]


def load_config(*args, **kwargs) -> "DictConfig":
    """Lazily import the config loader to avoid eager OmegaConf imports."""
    from gui_grounding.utils.config import load_config as _load_config

    return _load_config(*args, **kwargs)
