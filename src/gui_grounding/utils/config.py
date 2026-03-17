"""YAML configuration loading with OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from gui_grounding.constants import CONFIGS_DIR, PROJECT_ROOT


def load_config(
    config_path: str | Path,
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Load a YAML config and optionally merge CLI overrides.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML file.  Resolved in order:
        1. As-is (absolute or relative to cwd)
        2. Relative to project root  (e.g. ``configs/train/sft.yaml``)
        3. Relative to ``configs/``  (e.g. ``train/sft.yaml``)
    overrides : list[str], optional
        Dot-list overrides, e.g. ``["train.lr=1e-4", "train.epochs=5"]``.

    Returns
    -------
    DictConfig
        Merged, read-only configuration object.
    """
    config_path = Path(config_path)

    if not config_path.is_absolute():
        candidates = [
            config_path,
            PROJECT_ROOT / config_path,
            CONFIGS_DIR / config_path,
        ]
        resolved = next((p for p in candidates if p.exists()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"Config file not found. Searched:\n"
                + "\n".join(f"  - {p}" for p in candidates)
            )
        config_path = resolved
    elif not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist a DictConfig as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert a DictConfig to a plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)
