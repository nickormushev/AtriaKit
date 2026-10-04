"""Golden synthetic recording with analytically known P-wave feature values.

How to read this file (top to bottom):

1. Recording layout   - sampling rate, beat spacing, window lengths.
2. Lead recipes       - which waveform each lead carries.
3. Signal building    - recipes -> samples, annotations, in-memory loader.
4. Expected values    - what every feature *should* be, derived on paper from
                        the recipes (never by calling atriakit's feature code).
5. Corruptions        - known artifacts used to test preprocessing.

The recording has all 12 standard leads and eight identical beats at 75 bpm. Four waveform shapes cover
the descriptor families:

    half_sine   A sin(pi j / W)      area, amplitudes, extrema, fragments
    biphasic    A sin(2 pi j / W)    terminal force (PTF), morphology
    ramp        A j / W              offset amplitude/angle, uniform histogram
    notched     piecewise-linear M   several extrema and fragments

Leads I, II, III, aVL and V1-V6 share one window so the Kors VCG sees a clean
loop. aVF is shorter (so per-beat dispersion is non-zero) and aVR is an inverted
V1 wave (the code has aVR-specific logic). III and aVL obey Einthoven/Goldberger
relations to I and II; aVR and aVF deliberately do not. Waveforms are rounded to 1e-9 mV so the
"natural" zeros (sin(pi) and friends) are exact zeros.

Used by ``test_validation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from atriakit import AnnotationsLoader
from atriakit.configs.annotations_loader_config import AnnotationsLoaderConfig
from atriakit.configs.pipeline_config import PipelineConfig
from atriakit.configs.signal_preprocessor_config import SignalPreprocessorConfig
from atriakit.io import ECGLoader
from atriakit.models.annotation_schema import AnnotationSchema
from atriakit.models.ecg_data import ECGData

# ---------------------------------------------------------------------------
# 1. Recording layout
# ---------------------------------------------------------------------------

FS = 500  # Hz
N_BEATS = 8
RR = 400  # samples between beats -> 75 bpm
FIRST_ONSET = 100  # sample of the first P onset
RECORDING_LENGTH = FIRST_ONSET + N_BEATS * RR

WINDOW = 64  # onset -> offset in samples; the offset is inclusive, so 65 samples
WINDOW_AVF = 54  # shorter aVF window, so dispersion = (64 - 54) / FS

QRS_PEAK_DELAY = 40  # samples between P offset and the QRS peak
QRS_HALF_WIDTH = 8
QRS_AMPLITUDE = 1.0  # mV, triangular; read by the heart-rate estimator

BEAT_RATE_BPM = 60 * FS / RR
FRONTAL_AXIS_DEG = 40.0  # direction of the synthetic frontal dipole (leads I, aVF)

# ---------------------------------------------------------------------------
# 2. Lead recipes
# ---------------------------------------------------------------------------

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # standard 12-lead order
KORS_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
QRS_LEADS = ["II", "V5", "V6"]  # the leads the default heart-rate estimator reads

# Kors et al. (1990) regression matrix (rows X, Y, Z; columns I, II, V1-V6), as tabulated in
# Table 1 of https://pmc.ncbi.nlm.nih.gov/articles/PMC9106114 . That is a secondary source:
# check it against the original paper before citing it.
KORS = np.array(
    [
        [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
        [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
        [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31],
    ]
)

# (sample, mV) corners of the notched wave: peak, dip, peak, back to baseline.
NOTCH_KNOTS = [(0, 0.0), (14, 0.10), (28, 0.04), (44, 0.08), (WINDOW, 0.0)]


@dataclass(frozen=True)
class LeadSpec:
    """Waveform recipe for one lead's P wave."""

    shape: str  # "half_sine" | "biphasic" | "ramp" | "notched"
    amplitude: float  # mV; negative inverts the wave
    window: int = WINDOW


def half_sine_sum(window: int) -> float:
    """Closed form of ``sum_{j=0..W} sin(pi j / W)``, which is ``cot(pi / 2W)``."""
    return 1 / np.tan(np.pi / (2 * window))


def _avf_amplitude(amp_lead_i: float) -> float:
    """aVF amplitude that makes the frontal axis come out at ``FRONTAL_AXIS_DEG``.

    With Einthoven geometry I = Vx and aVF = (sqrt(3)/2) Vy, and the code's axis
    is ``atan2(2 sum(aVF), sqrt(3) sum(I))``. Solving for sum(aVF) gives
    ``(sqrt(3)/2) tan(axis) sum(I)``, and each half-sine sum is ``A cot(pi / 2W)``.
    """
    sum_lead_i = amp_lead_i * half_sine_sum(WINDOW)
    wanted_sum_avf = (np.sqrt(3) / 2) * np.tan(np.radians(FRONTAL_AXIS_DEG)) * sum_lead_i
    return wanted_sum_avf / half_sine_sum(WINDOW_AVF)


LEAD_SPECS: dict[str, LeadSpec] = {
    "I": LeadSpec("half_sine", 0.10),
    "II": LeadSpec("half_sine", 0.15),
    "III": LeadSpec("half_sine", 0.05),  # II - I, as Einthoven's law requires
    "aVR": LeadSpec("biphasic", -0.08),  # inverted V1 wave, to exercise the aVR-specific logic
    "aVL": LeadSpec("half_sine", 0.025),  # (I - III) / 2, as the Goldberger relation requires
    "aVF": LeadSpec("half_sine", _avf_amplitude(0.10), window=WINDOW_AVF),
    "V1": LeadSpec("biphasic", 0.08),
    "V2": LeadSpec("ramp", 0.06),
    "V3": LeadSpec("notched", 0.10),
    "V4": LeadSpec("half_sine", 0.12),
    "V5": LeadSpec("half_sine", 0.14),
    "V6": LeadSpec("half_sine", 0.10),
}

# ---------------------------------------------------------------------------
# 3. Signal building
# ---------------------------------------------------------------------------


def p_wave(spec: LeadSpec) -> np.ndarray:
    """One P wave: ``spec.window + 1`` samples from onset to offset inclusive."""
    j = np.arange(spec.window + 1)
    if spec.shape == "half_sine":
        wave = spec.amplitude * np.sin(np.pi * j / spec.window)
    elif spec.shape == "biphasic":
        wave = spec.amplitude * np.sin(2 * np.pi * j / spec.window)
    elif spec.shape == "ramp":
        wave = spec.amplitude * j / spec.window
    elif spec.shape == "notched":
        corner_samples, corner_values = zip(*NOTCH_KNOTS)
        wave = np.interp(j, corner_samples, corner_values)
    else:
        raise ValueError(f"unknown shape {spec.shape!r}")
    return np.round(wave, 9)


def qrs_complex() -> tuple[np.ndarray, int]:
    """Triangular QRS spike and the index of its peak inside the returned array."""
    rising = np.linspace(0, QRS_AMPLITUDE, QRS_HALF_WIDTH + 1)
    falling = rising[-2::-1]
    return np.concatenate([rising, falling]), QRS_HALF_WIDTH


def p_onsets() -> np.ndarray:
    """Sample index of every beat's P onset."""
    return FIRST_ONSET + RR * np.arange(N_BEATS)


def build_signal() -> np.ndarray:
    """Clean golden ECG with shape ``(len(LEADS), RECORDING_LENGTH)`` in mV. Eight identical beats."""
    ecg = np.zeros((len(LEADS), RECORDING_LENGTH))
    qrs, qrs_peak = qrs_complex()

    for lead_index, lead in enumerate(LEADS):
        wave = p_wave(LEAD_SPECS[lead])
        for onset in p_onsets():
            ecg[lead_index, onset : onset + len(wave)] = wave
            if lead in QRS_LEADS:
                qrs_start = onset + WINDOW + QRS_PEAK_DELAY - qrs_peak
                ecg[lead_index, qrs_start : qrs_start + len(qrs)] += qrs
    return ecg


def build_annotations(
    onset_shift: dict[str, int] | None = None,
    offset_shift: dict[str, int] | None = None,
    boundary_mode: str = "per_lead",
):
    """P-wave annotations, one row per lead per beat.

    Args:
        onset_shift: Optional per-lead number of samples added to every onset.
        offset_shift: Optional per-lead number of samples added to every offset.
        boundary_mode: ``"per_lead"`` keeps each lead's own onset/offset;
            ``"cross_lead"`` (the library default) gives every lead in a beat
            the earliest onset and latest offset.
    """
    onset_shift = onset_shift or {}
    offset_shift = offset_shift or {}

    rows = []
    for beat_id, onset in enumerate(p_onsets(), start=1):
        for lead in LEADS:
            true_offset = onset + LEAD_SPECS[lead].window
            rows.append(
                {
                    AnnotationSchema.FILE_PATH: "golden.mem",
                    AnnotationSchema.LEAD: lead,
                    AnnotationSchema.ONSET: int(onset) + onset_shift.get(lead, 0),
                    AnnotationSchema.OFFSET: int(true_offset) + offset_shift.get(lead, 0),
                    AnnotationSchema.P_WAVE_ID: beat_id,
                }
            )

    config = AnnotationsLoaderConfig(boundary_mode=boundary_mode)
    return AnnotationsLoader(config=config).from_dataframe(pd.DataFrame(rows))


class InMemoryLoader:
    """ECG loader serving an array from memory, so no DICOM int16 rounding creeps in."""

    def __init__(self, ecg: np.ndarray, leads: list[str] = LEADS, fs: int = FS):
        self._ecg = ecg
        self._leads = leads
        self._fs = fs

    def load(self, path: str | Path) -> ECGData:
        """Return a fresh ``ECGData`` so cached preprocessing never leaks between runs."""
        lead_to_index = {lead: i for i, lead in enumerate(self._leads)}
        return ECGData(self._ecg.copy(), self._fs, lead_to_index)


def make_ecg_loader(ecg: np.ndarray, leads: list[str] = LEADS) -> ECGLoader:
    """``ECGLoader`` resolving the annotation file name ``golden.mem`` to ``ecg``."""
    return ECGLoader(loaders={".mem": InMemoryLoader(ecg, leads)})


def no_preprocessing_config() -> PipelineConfig:
    """Pipeline config with every filter and baseline correction switched off."""
    off = dict(highcut=None, lowcut=None, notch_freq=None, global_baseline_mode="none")
    return PipelineConfig(
        signal_preprocessor_config=SignalPreprocessorConfig(**off),
        morphology_preprocessor_config=SignalPreprocessorConfig(**off),
    )


def target_preprocessing_config() -> PipelineConfig:
    """The preprocessing actually used for the real-data analysis: 1-150 Hz bandpass,
    no P-onset spline baseline, and no segment-level baseline correction either
    (SegmentConfig's own default, "none"). Morphology preprocessing (shape
    classification, inflection detection) is left at its own library default --
    untouched here, as in the real-data rerun.
    """
    return PipelineConfig(
        signal_preprocessor_config=SignalPreprocessorConfig(
            lowcut=1,
            highcut=150,
            notch_freq=[50.0, 100.0],  # both mains harmonics; 100 Hz is inside a 150 Hz passband
            global_baseline_mode="none",
        ),
    )


# ---------------------------------------------------------------------------
# 4. Expected values (derived from the recipes, not from atriakit)
# ---------------------------------------------------------------------------


def _wave_sum(spec: LeadSpec) -> float:
    """Closed form of ``sum_j x_j`` over one P wave."""
    amp, window = spec.amplitude, spec.window
    if spec.shape == "half_sine":
        return amp * half_sine_sum(window)
    if spec.shape == "biphasic":
        return 0.0  # exactly one full period
    if spec.shape == "ramp":
        return amp * (window + 1) / 2
    # notched: trapezoid under the corners (both end samples are zero)
    return sum(
        (v0 + v1) / 2 * (j1 - j0)
        for (j0, v0), (j1, v1) in zip(NOTCH_KNOTS[:-1], NOTCH_KNOTS[1:])
    )


def _wave_abs_sum(spec: LeadSpec) -> float:
    """Closed form of ``sum_j |x_j|``."""
    if spec.shape == "biphasic":
        # two half-sine lobes, each spanning W/2 samples
        return 2 * abs(spec.amplitude) * half_sine_sum(spec.window // 2)
    return _wave_sum(spec)  # the other shapes never go negative


def _shannon_bits(counts: list[int]) -> float:
    probabilities = np.asarray(counts, dtype=float) / sum(counts)
    return float(-(probabilities * np.log2(probabilities)).sum())


def lead_expected(lead: str) -> dict[str, float | str]:
    """Expected feature values for one lead under the default feature config.

    Implementation conventions that these values encode (the paper should call
    these "adapted" definitions):

    * ``duration`` is ``(offset - onset) / fs``, although the segment has one
      more sample than that.
    * ``area`` is the rectangle-rule sum ``sum|x| / fs`` over every sample.
    * A fragment is a monotonic run between consecutive boundaries (segment ends
      and local extrema); its width is the time between the two boundary samples
      and its height the difference of their values. A monotonic wave is one
      fragment.
    * ``onset_offset_slope`` is ``offset_amp / (n_samples * ms_per_sample)`` in mV/ms.
    """
    spec = LEAD_SPECS[lead]
    area = _wave_abs_sum(spec) / FS
    duration = spec.window / FS

    expected = {
        "duration": duration,
        "area": area,
        "area_to_duration_ratio": area / duration,
        "dispersion": (WINDOW - WINDOW_AVF) / FS,
        "atrial_rate": BEAT_RATE_BPM,
        "heart_rate": BEAT_RATE_BPM,
    }
    shape_expected = {
        "half_sine": _half_sine_expected,
        "biphasic": _biphasic_expected,
        "ramp": _ramp_expected,
        "notched": _notched_expected,
    }[spec.shape]
    expected.update(shape_expected(spec))
    return expected


def _half_sine_expected(spec: LeadSpec) -> dict[str, float | str]:
    amp, window = spec.amplitude, spec.window
    peak = window // 2
    # two fragments: onset -> peak and peak -> offset
    rise_samples, fall_samples = peak, window - peak
    rise_height = amp  # baseline up to the peak value
    fall_height = amp  # peak value back down to baseline
    return {
        "max_amplitude": amp,
        "max_absolute_amplitude": amp,
        "min_amplitude": 0.0,
        "ptp_amplitude": amp,
        "offset_amplitude": 0.0,
        "onset_offset_slope": 0.0,
        "p_wave_morphology": "Monophasic Positive",
        "ptf_auto": 0.0,  # nothing negative to measure
        "complexity": 1,
        "fragment_count": 2,
        "fragment_width": (rise_samples + fall_samples) / 2 / FS,
        "fragment_height": (rise_height + fall_height) / 2,
    }


def _biphasic_expected(spec: LeadSpec) -> dict[str, float | str]:
    window = spec.window
    amp = abs(spec.amplitude)
    quarter, midpoint = window // 4, window // 2
    wave = lambda j: spec.amplitude * np.sin(2 * np.pi * j / window)  # noqa: E731

    # Terminal force starts at the steepest zero crossing (j = W/2) and runs to
    # the offset; aVR (negative amplitude) is inverted back by the definition.
    terminal_samples = window - midpoint + 1
    conventional_samples = window - midpoint - 1  # only the negative run

    # three fragments: onset -> first extremum, extremum -> extremum, -> offset
    fragment_samples = [quarter, 2 * quarter, window - 3 * quarter]
    fragment_heights = [
        abs(wave(quarter)),  # baseline -> first extremum
        abs(wave(3 * quarter) - wave(quarter)),  # first -> second extremum
        amp,  # second extremum -> baseline at the offset
    ]
    is_positive_first = spec.amplitude > 0
    return {
        "max_amplitude": amp,
        "max_absolute_amplitude": amp,
        "min_amplitude": -amp,
        "ptp_amplitude": 2 * amp,
        "offset_amplitude": 0.0,
        "onset_offset_slope": 0.0,
        "p_wave_morphology": (
            "Biphasic Positive-Negative" if is_positive_first else "Biphasic Negative-Positive"
        ),
        "ptf": terminal_samples / FS * amp,
        "ptf_auto": terminal_samples / FS * amp,
        "ptf_zero_crossing": conventional_samples / FS * amp,
        "complexity": 2,
        "fragment_count": 3,
        "fragment_width": sum(fragment_samples) / 3 / FS,
        "fragment_height": sum(fragment_heights) / 3,
    }


def _ramp_expected(spec: LeadSpec) -> dict[str, float | str]:
    amp, window = spec.amplitude, spec.window
    n_samples = window + 1
    ms_per_sample = 1000 / FS
    return {
        "max_amplitude": amp,
        "max_absolute_amplitude": amp,
        "min_amplitude": 0.0,
        "ptp_amplitude": amp,
        "offset_amplitude": amp,
        "onset_offset_slope": amp / (n_samples * ms_per_sample),
        "p_wave_morphology": "Monophasic Positive",
        "ptf_auto": 0.0,
        "complexity": 0,  # monotonic: no extrema
        # ... which makes the whole wave a single monotonic fragment
        "fragment_count": 1,
        "fragment_width": window / FS,
        "fragment_height": amp,
        # 65 evenly spaced samples in 32 equal bins: 2 per bin, 3 in the last
        "shannon_entropy": _shannon_bits([2] * 31 + [3]),
        # a straight line is perfectly self-similar at length m and m + 1
        "sample_entropy": 0.0,
    }


def _notched_expected(spec: LeadSpec) -> dict[str, float | str]:
    # four fragments: baseline -> peak, peak -> dip, dip -> peak, peak -> baseline
    fragment_samples = [14, 14, 16, spec.window - 44]
    fragment_heights = [0.10, 0.06, 0.04, 0.08]
    return {
        "max_amplitude": 0.10,
        "max_absolute_amplitude": 0.10,
        "min_amplitude": 0.0,
        "ptp_amplitude": 0.10,
        "offset_amplitude": 0.0,
        "onset_offset_slope": 0.0,
        "p_wave_morphology": "Complex",
        "ptf_auto": 0.0,
        "complexity": 3,  # peak, dip, peak
        "fragment_count": 4,
        "fragment_width": sum(fragment_samples) / 4 / FS,
        "fragment_height": sum(fragment_heights) / 4,
    }


def vcg_net_vector() -> np.ndarray:
    """Closed-form sum over the window of the Kors X, Y, Z components."""
    lead_sums = np.array([_wave_sum(LEAD_SPECS[lead]) for lead in KORS_LEADS])
    return KORS @ lead_sums


def vcg_loop() -> np.ndarray:
    """Kors ``(3, WINDOW + 1)`` loop of one beat, from the known lead waveforms."""
    leads = np.array([p_wave(LEAD_SPECS[lead]) for lead in KORS_LEADS])
    return KORS @ leads


def vcg_features(loop: np.ndarray, net: np.ndarray | None = None) -> dict[str, float]:
    """VCG descriptors of a ``(3, n)`` loop, from independent numpy computations.

    Args:
        loop: Kors X, Y, Z samples of one P wave.
        net: Sum of each component over the window. Defaults to ``loop.sum(axis=1)``;
            the regular recording passes its closed form instead.
    """
    net_x, net_y, net_z = loop.sum(axis=1) if net is None else net
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(loop)))[::-1]
    largest, middle, smallest = eigenvalues

    return {
        "vcg_axis_azimuth": float(np.degrees(np.arctan2(net_y, net_x))),
        "vcg_axis_elevation": float(np.degrees(np.arctan2(net_z, np.hypot(net_x, net_y)))),
        "vcg_area": float(np.linalg.norm(loop, axis=0).sum() / FS),
        "vcg_eigenvalues_1": float(largest),
        "vcg_eigenvalues_2": float(middle),
        "vcg_eigenvalues_3": float(smallest),
        "vcg_roundness": float(middle / largest),
        "vcg_flatness": float(smallest / (largest + middle)),
    }


def vcg_expected() -> dict[str, float]:
    """VCG values of the regular recording: closed-form axis, numpy everything else."""
    return vcg_features(vcg_loop(), net=vcg_net_vector())


def reference_shannon(wave: np.ndarray, n_bins: int = 32) -> float:
    """Shannon entropy in bits of the amplitude histogram, via ``scipy.stats.entropy``."""
    counts, _ = np.histogram(wave, bins=n_bins)
    return float(scipy_entropy(counts, base=2))


def reference_sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.25) -> float:
    """Brute-force SampEn (Richman & Moorman 2000).

    Chebyshev distance, matches at ``distance <= r``, ``N - m`` templates for
    both lengths, self-matches excluded.
    """
    x = np.asarray(x, dtype=float)
    tolerance = r_factor * np.std(x)

    def count_matches(length: int) -> int:
        templates = np.array([x[i : i + length] for i in range(len(x) - m)])
        distances = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        pairs_including_self = (distances <= tolerance).sum()
        return int((pairs_including_self - len(templates)) // 2)

    longer_matches, shorter_matches = count_matches(m + 1), count_matches(m)
    if longer_matches == 0 or shorter_matches == 0:
        return float("nan")  # SampEn is undefined without matches
    return float(-np.log(longer_matches / shorter_matches))


def entropy_expected(lead: str) -> dict[str, float]:
    """Independent entropy references computed from the lead's known waveform."""
    wave = p_wave(LEAD_SPECS[lead])
    return {
        "shannon_entropy": reference_shannon(wave),
        "sample_entropy": reference_sample_entropy(wave),
    }


# ---------------------------------------------------------------------------
# 5. Corruptions (additive, deterministic)
# ---------------------------------------------------------------------------


def corruptions(seed: int = 0) -> dict[str, np.ndarray]:
    """Known artifacts, each shape ``(RECORDING_LENGTH,)`` in mV, added to every lead.

    * ``drift``    linear baseline ramp of 0.05 mV/s (a P-onset spline removes it exactly)
    * ``wander``   0.25 Hz sinusoidal baseline wander
    * ``mains``    50 Hz and 100 Hz line interference
    * ``hf``       tones between 150 and 245 Hz, above the 120 Hz low-pass corner
    * ``combined`` drift + mains + hf
    """
    t = np.arange(RECORDING_LENGTH) / FS
    rng = np.random.default_rng(seed)

    hf_tones = [
        0.004 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        for freq in (150, 170, 190, 210, 230, 245)
    ]
    artifacts = {
        "drift": 0.05 * t,
        "wander": 0.04 * np.sin(2 * np.pi * 0.25 * t + 0.5),
        "mains": 0.05 * np.sin(2 * np.pi * 50 * t + 0.3)
        + 0.02 * np.sin(2 * np.pi * 100 * t + 1.0),
        "hf": np.sum(hf_tones, axis=0),
    }
    artifacts["combined"] = artifacts["drift"] + artifacts["mains"] + artifacts["hf"]
    return artifacts
