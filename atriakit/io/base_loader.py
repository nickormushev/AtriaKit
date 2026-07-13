from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from atriakit.models.ecg_data import ECGData


@runtime_checkable
class BaseLoader(Protocol):
    """Protocol for ECG file loaders: implement ``load(path) -> ECGData``."""

    def load(self, path: str | Path) -> ECGData:
        """Load an ECG recording from *path* and return an :class:`ECGData` instance."""
        ...
