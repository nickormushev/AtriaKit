from __future__ import annotations

from pathlib import Path

from atriakit.models.ecg_data import ECGData
from atriakit.io.base_loader import BaseLoader
from atriakit.io.dicom_loader import DicomLoader


class ECGLoader:
    """Loads ECG files by routing to the appropriate loader by file extension.

    Handles ``.dcm`` and ``.ima`` files via :class:`DicomLoader` by default.
    Register additional loaders for other formats with :meth:`register`.

    Example::

        loader = ECGLoader()
        loader.register(".edf", MyEdfLoader())
        pipeline = Pipeline(..., loader=loader)
    """

    def __init__(self, loaders: dict[str, BaseLoader] | None = None):
        self._loaders: dict[str, BaseLoader] = loaders if loaders is not None else {
            ".dcm": DicomLoader(),
            ".ima": DicomLoader(),
        }

    def register(self, extension: str, loader: BaseLoader) -> None:
        """Register *loader* for files with the given *extension* (e.g. ``".edf"``)."""
        self._loaders[extension.lower()] = loader

    def load(self, path: str | Path) -> ECGData:
        """Load an ECG file and return an :class:`ECGData` instance.

        Raises:
            ValueError: If no loader is registered for the file's extension.
        """
        ext = Path(path).suffix.lower()
        loader = self._loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"No loader registered for extension {ext!r}. "
                f"Registered: {list(self._loaders)}. "
                "Use ECGLoader.register() to add support for this format."
            )
        return loader.load(path)
