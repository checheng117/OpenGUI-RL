"""Unified logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "gui_grounding",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Return a consistently configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically module ``__name__``).
    level : int
        Logging level.
    log_file : Path, optional
        If given, also write logs to this file.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
