from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class SignalPreprocessorConfig:
    """Default parameters for ECG signal preprocessing."""

    lowcut: float | None = (
        None  # Low cutoff for the bandpass filter; `None` disables bandpass filtering.
    )
    highcut: float | None = (
        None  # High cutoff for the bandpass filter; `None` disables bandpass filtering.
    )
    notch_freq: float | list[float] | None = (
        50.0  # Line-noise notch frequency/frequencies in Hz; set to `None` to skip the notch filter.
    )
    order: int = 2  # Butterworth bandpass filter order.
    quality_factor: float = (
        20.0  # Q factor for the notch filter; higher values make the notch narrower.
    )
    global_baseline_mode: Literal["none", "p_onset_spline"] = (
        "p_onset_spline"  # Full-recording baseline correction mode applied before segment extraction.
    )
    global_baseline_smoothing_factor: float = (
        0  # Dimensionless smoothing strength for the SciPy spline fit; 0 keeps exact interpolation, larger values follow only slower baseline drift.
    )
    normalization_type: Literal["none", "zscore"] = (
        "none"  # Dataset-level normalization to apply after filtering; use `"none"` to disable it.
    )
    mean: np.ndarray | None = (
        None  # Per-lead mean used for zscore normalization; Either provide or compute with ECGDataset. Required if `normalization_type` is `"zscore"`.
    )
    std: np.ndarray | None = (
        None  # Per-lead standard deviation used for zscore normalization;  Either provide or compute with ECGDataset. Required if `normalization_type` is `"zscore"`.
    )


def default_signal_preprocessor_config() -> SignalPreprocessorConfig:
    """Low-pass at 120 Hz with 50/100 Hz notch. Used for amplitude and duration features."""
    return SignalPreprocessorConfig(lowcut=None, highcut=120, notch_freq=[50.0, 100.0])


def default_morphology_preprocessor_config() -> SignalPreprocessorConfig:
    """Low-pass at 30 Hz with 50/100 Hz notch. Used for P-wave shape classification and inflection detection."""
    return SignalPreprocessorConfig(lowcut=None, highcut=30, notch_freq=[50.0, 100.0])
