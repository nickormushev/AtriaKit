from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from atriakit.models.annotations import Annotations
from atriakit.configs.signal_preprocessor_config import SignalPreprocessorConfig
from atriakit.models.annotation_schema import AnnotationSchema
from atriakit.utils import build_p_onset_spline_baseline, normalize


def bandpass_signal(
    ecg: np.ndarray,
    sampling_frequency: float,
    *,
    lowcut: float,
    highcut: float,
    order: int = 2,
) -> np.ndarray:
    b, a = butter(
        order,
        [
            lowcut / (sampling_frequency / 2),
            highcut / (sampling_frequency / 2),
        ],
        btype="bandpass",
    )
    return filtfilt(b, a, ecg)


def apply_notch_filter(
    ecg: np.ndarray,
    sampling_frequency: float,
    *,
    notch_freq: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    nyquist = 0.5 * sampling_frequency
    norm_notch_freq = notch_freq / nyquist
    if not 0 < norm_notch_freq < 1:
        raise ValueError(
            f"Invalid notch frequency {notch_freq} Hz for sampling frequency "
            f"{sampling_frequency} Hz."
        )
    b, a = iirnotch(norm_notch_freq, quality_factor)
    return filtfilt(b, a, ecg, axis=1)


class SignalPreprocessor:
    """Configurable signal preprocessing strategy.

    Attributes:
        config: Active preprocessing settings (filter cutoffs, normalization
            type and stats, baseline correction mode).
    """

    def __init__(self, config: SignalPreprocessorConfig | None = None):
        self.config = config or SignalPreprocessorConfig()
        self._baseline_onset_indices: dict[str, np.ndarray] | None = None

    def with_overrides(self, **config_overrides) -> "SignalPreprocessor":
        """Return a copy with the given config fields replaced."""
        return SignalPreprocessor(replace(self.config, **config_overrides))

    def cache_key(self) -> tuple:
        """Return a hashable key that uniquely identifies this preprocessor's configuration.

        Used by ``ECGData.get_ecg`` to cache preprocessed signals.
        """
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.config.lowcut,
            self.config.highcut,
            self.config.order,
            (
                tuple(self.config.notch_freq)
                if isinstance(self.config.notch_freq, list)
                else self.config.notch_freq
            ),
            self.config.quality_factor,
            self.config.normalization_type,
            self.config.global_baseline_mode,
            self.config.global_baseline_smoothing_factor,
            None if self.config.mean is None else tuple(np.asarray(self.config.mean).ravel()),
            None if self.config.std is None else tuple(np.asarray(self.config.std).ravel()),
            self._onset_indices_cache_key(),
        )

    def _onset_indices_cache_key(self) -> tuple | None:
        if self._baseline_onset_indices is None:
            return None
        return tuple(
            (lead, tuple(indices.tolist()))
            for lead, indices in sorted(self._baseline_onset_indices.items())
        )

    def uses_normalization(self) -> bool:
        if self.config.normalization_type == "none":
            return False
        if self.config.normalization_type == "zscore":
            return True
        raise ValueError(
            f"Unsupported normalization_type: {self.config.normalization_type!r}."
        )

    def preprocess(
        self,
        ecg: np.ndarray,
        sampling_frequency: float,
        lead_to_index: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Apply the configured preprocessing steps to an ECG array.

        Applies notch filtering, bandpass/lowpass/highpass filtering, spline
        baseline correction, and z-score normalization in that order, depending
        on the active config.

        Args:
            ecg: Raw signal array of shape ``(n_leads, n_samples)``.
            sampling_frequency: Sampling frequency in Hz.
            lead_to_index: Mapping from lead name to row index in ``ecg``.
                Required for spline baseline correction; ignored otherwise.

        Returns:
            Preprocessed signal array with the same shape as ``ecg``.

        Raises:
            ValueError: If configured for zscore normalization without fitted
                ``mean``/``std``, or if ``normalization_type`` is unsupported.
        """
        processed = ecg
        if self.config.notch_freq is not None:
            freqs = (
                self.config.notch_freq
                if isinstance(self.config.notch_freq, list)
                else [self.config.notch_freq]
            )
            for freq in freqs:
                processed = apply_notch_filter(
                    processed,
                    sampling_frequency,
                    notch_freq=freq,
                    quality_factor=self.config.quality_factor,
                )

        if self.config.lowcut is not None and self.config.highcut is not None:
            processed = bandpass_signal(
                processed,
                sampling_frequency,
                lowcut=self.config.lowcut,
                highcut=self.config.highcut,
                order=self.config.order,
            )
        elif self.config.highcut is not None:
            b, a = butter(
                self.config.order,
                self.config.highcut / (sampling_frequency / 2),
                btype="low",
            )
            processed = filtfilt(b, a, processed)
        elif self.config.lowcut is not None:
            b, a = butter(
                self.config.order,
                self.config.lowcut / (sampling_frequency / 2),
                btype="high",
            )
            processed = filtfilt(b, a, processed)

        if (
            self.config.global_baseline_mode == "p_onset_spline"
            and self._baseline_onset_indices
            and lead_to_index is not None
        ):
            processed = self._apply_global_baseline_correction(
                processed,
                self._baseline_onset_indices,
                lead_to_index,
            )

        if self.uses_normalization():
            if self.config.mean is None or self.config.std is None:
                raise ValueError(
                    "SignalPreprocessor is configured for zscore normalization but "
                    "has not been fit. Use with_overrides(mean=..., std=...) to set "
                    "stats, or with_overrides(normalization_type='none') to compute "
                    "statistics from unnormalized data."
                )
            processed = normalize(processed, self.config.mean, self.config.std)

        return processed

    def has_baseline_onsets(self) -> bool:
        return self._baseline_onset_indices is not None

    def clear_baseline_onsets(self) -> None:
        self._baseline_onset_indices = None

    def set_baseline_onsets(
        self, annotations: Annotations
    ) -> dict[str, np.ndarray] | None:
        """Compute per-lead onset indices from annotations and store them for baseline correction.

        Returns:
            Dict mapping lead name to onset index array, or ``None`` when
            baseline correction is disabled or ``annotations`` is empty.
        """
        if self.config.global_baseline_mode == "none" or annotations.empty:
            self._baseline_onset_indices = None
            return None

        result: dict[str, np.ndarray] = {}
        for lead, group in annotations.groupby(AnnotationSchema.LEAD):
            if lead == "VCG":
                continue

            onsets = group[AnnotationSchema.ONSET].to_numpy(dtype=int)
            onsets = onsets[onsets >= 0]

            if onsets.size > 0:
                result[lead] = np.unique(onsets)

        self._baseline_onset_indices = result or None
        return self._baseline_onset_indices

    def _apply_global_baseline_correction(
        self,
        ecg: np.ndarray,
        onset_indices_by_lead: dict[str, np.ndarray],
        lead_to_index: dict[str, int],
    ) -> np.ndarray:
        corrected = ecg.copy()
        for lead, onset_indices in onset_indices_by_lead.items():
            if lead not in lead_to_index:
                continue

            lead_idx = lead_to_index[lead]
            baseline = build_p_onset_spline_baseline(
                corrected[lead_idx],
                onset_indices,
                smoothing_factor=self.config.global_baseline_smoothing_factor,
            )
            corrected[lead_idx] = corrected[lead_idx] - baseline

        return corrected
