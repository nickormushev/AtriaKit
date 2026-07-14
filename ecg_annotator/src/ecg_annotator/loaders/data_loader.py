"""Unified data loader — discovers ECG files of any supported format in one directory sweep."""

import logging
from pathlib import Path

from ecg_annotator.loaders import dcm_loader
from ecg_annotator.loaders.dcm_loader import dcm_sort_key, load_dcm_ecg
from ecg_annotator.models.recording import ECGRecording

log = logging.getLogger(__name__)

# Maps each supported extension to its sort_key_fn for discovery.
# Each loader module owns its sort key logic so DataLoader stays format-agnostic.
_SORT_KEYS: dict[str, object] = {
    **{ext: dcm_sort_key for ext in dcm_loader.SUPPORTED_EXTENSIONS},
}


class DataLoader:
    """Discovers and loads ECG files from a directory, routing by file extension."""

    def __init__(self, data_dir: str | None):
        self.patient_file_paths: list[str] = []
        if data_dir:
            self._discover(data_dir)

    def _discover(self, data_dir: str) -> None:
        """Single directory sweep: collect supported files, get their sort keys, sort."""
        entries: list[tuple[str, str]] = []

        for path in Path(data_dir).rglob("*"):
            if not path.is_file():
                continue
            sort_key_fn = _SORT_KEYS.get(path.suffix)
            if sort_key_fn is None:
                continue

            key = sort_key_fn(str(path))
            if key is not None:
                entries.append((key, str(path)))

        entries.sort(key=lambda x: x[0])
        self.patient_file_paths = [p for _, p in entries]
        log.info("Discovered %d ECG files in %s", len(self.patient_file_paths), data_dir)

    def get_file_count(self) -> int:
        return len(self.patient_file_paths)

    def get_path(self, idx: int) -> str | None:
        if idx < 0 or idx >= len(self.patient_file_paths):
            return None
        return self.patient_file_paths[idx]

    def load_patient_by_idx(self, idx: int) -> ECGRecording | None:
        if idx < 0 or idx >= len(self.patient_file_paths):
            return None
        path = self.patient_file_paths[idx]
        ext = Path(path).suffix

        # Access load functions by name so patching in tests works correctly.
        if ext in dcm_loader.SUPPORTED_EXTENSIONS:
            return load_dcm_ecg(path)
        return None
