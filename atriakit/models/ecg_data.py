from __future__ import annotations

import numpy as np

from atriakit.preprocessing.signals import SignalPreprocessor
from atriakit.utils import convert_ecg_segment_to_vcg


class ECGData:
    """Container for a single ECG recording with cached preprocessing.

    Attributes:
        ecg: Raw signal array with shape ``(n_leads, n_samples)``.
        sampling_frequency: Sampling frequency in Hz.
        content_date: Recording date from the source file.
        content_start_time: Recording time from the source file.
    """

    def __init__(
        self,
        ecg: np.ndarray,
        fs: int | float,
        lead_to_index: dict[str, int],
        *,
        content_date: str | None = None,
        content_time: str | None = None,
    ):
        self.ecg = ecg
        self.sampling_frequency = fs
        self._lead_to_index = lead_to_index
        self.content_date = content_date
        self.content_start_time = content_time
        self._ecg_cache: dict = {}
        self._vcg_cache: dict = {}

    def set_ecg(self, ecg: np.ndarray) -> None:
        self.ecg = ecg
        self._ecg_cache = {}
        self._vcg_cache = {}

    def get_sampling_frequency(self) -> int | float:
        return self.sampling_frequency

    def get_content_start_time(self) -> str | None:
        return self.content_start_time

    def get_content_date(self) -> str | None:
        return self.content_date

    def get_lead_to_index(self) -> dict[str, int]:
        return self._lead_to_index

    def get_leads(self) -> list[str]:
        return list(self._lead_to_index.keys())

    def get_ecg(
        self,
        preprocessor: SignalPreprocessor | None = None,
    ) -> np.ndarray:
        if preprocessor is None:
            preprocessor = SignalPreprocessor()

        cache_key = preprocessor.cache_key()

        if cache_key in self._ecg_cache:
            return self._ecg_cache[cache_key]

        ecg = preprocessor.preprocess(
            self.ecg,
            self.sampling_frequency,
            lead_to_index=self._lead_to_index,
        )

        ecg.flags.writeable = False
        self._ecg_cache[cache_key] = ecg
        return ecg

    def get_lead_signal(
        self,
        lead: str,
        preprocessor: SignalPreprocessor | None = None,
    ) -> np.ndarray:
        ecg = self.get_ecg(preprocessor=preprocessor)
        try:
            lead_idx = self._lead_to_index[lead]
        except KeyError as err:
            raise ValueError(
                f"Unknown lead: {lead!r}. Valid leads: {list(self._lead_to_index)}"
            ) from err
        return ecg[lead_idx]

    def get_vcg(
        self,
        preprocessor: SignalPreprocessor | None = None,
    ) -> np.ndarray:
        if preprocessor is None:
            preprocessor = SignalPreprocessor()

        cache_key = preprocessor.cache_key()
        if cache_key in self._vcg_cache:
            return self._vcg_cache[cache_key]

        ecg = self.get_ecg(preprocessor=preprocessor)
        vcg = convert_ecg_segment_to_vcg(ecg, self._lead_to_index)
        vcg.flags.writeable = False
        self._vcg_cache[cache_key] = vcg
        return vcg
