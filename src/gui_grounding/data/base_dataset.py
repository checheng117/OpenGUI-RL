"""Abstract base dataset for GUI grounding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from gui_grounding.data.schemas import GroundingSample
from gui_grounding.utils.logger import get_logger

logger = get_logger(__name__)


class BaseGroundingDataset(ABC):
    """Base class that all dataset adapters should extend.

    Subclasses must implement :meth:`_load_raw` and :meth:`_to_sample` so that
    the rest of the pipeline works with :class:`GroundingSample` objects.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self._samples: list[GroundingSample] = []

        if self.data_dir.exists():
            self._load()
        else:
            logger.warning(
                "Data directory %s does not exist. "
                "Dataset will be empty until data is downloaded.",
                self.data_dir,
            )

    def _load(self) -> None:
        raw_items = self._load_raw()
        if self.max_samples is not None:
            raw_items = raw_items[: self.max_samples]
        self._samples = [self._to_sample(item) for item in raw_items]
        logger.info(
            "Loaded %d samples from %s (split=%s)", len(self._samples), self.name, self.split
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset identifier string."""

    @abstractmethod
    def _load_raw(self) -> list:
        """Load raw annotation items from disk."""

    @abstractmethod
    def _to_sample(self, raw_item) -> GroundingSample:
        """Convert a single raw item to the canonical schema."""

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GroundingSample:
        return self._samples[idx]

    def __iter__(self):
        return iter(self._samples)
