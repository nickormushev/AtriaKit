import numpy as np
from scipy.interpolate import make_splrep

from atriakit.constants import VCG_LEADS as _VCG_LEADS


def apply_baseline_correction(segment: np.ndarray, update_type: str) -> np.ndarray:
    """Apply segment-level baseline correction.

    Args:
        segment: 1-D or 2-D ``(channels, samples)`` signal segment.
        update_type: ``"linear"`` removes a linear trend; ``"onset"`` subtracts
            the first sample of each channel; ``"none"`` returns the segment
            unchanged.

    Returns:
        Baseline-corrected segment (same shape as input).

    Raises:
        ValueError: If ``update_type`` is not one of ``"linear"``, ``"onset"``,
            or ``"none"``.
    """
    if update_type == "linear":
        if segment.ndim > 1:
            # VCG case, apply remove_linear_baseline to each row
            segment = np.array([remove_linear_baseline(ch) for ch in segment])
        else:
            segment = remove_linear_baseline(segment)
    elif update_type == "onset":
        if segment.ndim > 1:
            # Subtract first sample from each channel
            segment = segment - segment[:, 0:1]
        else:
            segment = segment - segment[0]
    elif update_type == "none":
        return segment
    else:
        raise ValueError(f"Unknown update_type: {update_type}")

    return segment


def remove_linear_baseline(segment, onset=0, offset=-1):
    """Remove the linear trend between the first and last sample of the segment."""
    x = np.arange(len(segment))
    if offset == -1:
        offset = len(segment) - 1
    if offset == len(segment):
        offset -= 1

    slope = (segment[int(offset)] - segment[int(onset)]) / (offset - onset)
    intercept = segment[int(onset)] - slope * int(onset)

    baseline = slope * x + intercept
    return segment - baseline


def build_p_onset_spline_baseline(
    signal: np.ndarray,
    onset_indices: np.ndarray,
    smoothing_factor: float = 0.5,
) -> np.ndarray:
    """Estimate and return a baseline wander array by fitting a spline through P-wave onset amplitudes."""
    onset_indices = np.asarray(onset_indices, dtype=int)
    onset_indices = onset_indices[
        (onset_indices >= 0) & (onset_indices < signal.shape[-1])
    ]

    if onset_indices.size == 0:
        return np.zeros_like(signal, dtype=float)

    anchor_x, anchor_y = _build_p_onset_anchor_values(signal, onset_indices)
    x_full = np.arange(signal.shape[-1], dtype=float)

    if signal.ndim == 1:
        return _interpolate_baseline(anchor_x, anchor_y, x_full, smoothing_factor)

    return np.vstack(
        [
            _interpolate_baseline(
                anchor_x,
                channel_anchor_y,
                x_full,
                smoothing_factor,
            )
            for channel_anchor_y in anchor_y
        ]
    )


def _build_p_onset_anchor_values(
    signal: np.ndarray,
    onset_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_onsets, inverse = np.unique(onset_indices, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    anchor_x = unique_onsets.astype(float)

    if signal.ndim == 1:
        anchor_sums = np.zeros(unique_onsets.shape[0], dtype=float)
        np.add.at(anchor_sums, inverse, signal[onset_indices].astype(float))
        return anchor_x, anchor_sums / counts

    anchor_sums = np.zeros((signal.shape[0], unique_onsets.shape[0]), dtype=float)
    for channel_idx, channel in enumerate(signal):
        np.add.at(
            anchor_sums[channel_idx], inverse, channel[onset_indices].astype(float)
        )
    return anchor_x, anchor_sums / counts


def _interpolate_baseline(
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
    x_full: np.ndarray,
    smoothing_factor: float,
) -> np.ndarray:
    if anchor_x.size == 1:
        return np.full(x_full.shape, anchor_y[0], dtype=float)

    if anchor_x.size == 2:
        slope = (anchor_y[1] - anchor_y[0]) / (anchor_x[1] - anchor_x[0])
        intercept = anchor_y[0] - slope * anchor_x[0]
        return slope * x_full + intercept

    smoothing_value = _compute_spline_smoothing_value(anchor_y, smoothing_factor)
    spline_degree = min(3, anchor_x.size - 1)
    spline = make_splrep(anchor_x, anchor_y, s=smoothing_value, k=spline_degree)
    return spline(x_full)


def _compute_spline_smoothing_value(
    anchor_y: np.ndarray,
    smoothing_factor: float,
) -> float:
    if smoothing_factor <= 0:
        return 0.0

    # Scale the smoothing term by both the number of anchors and their variance.
    # This keeps the user-facing factor easier to reason about across recordings.
    variance = float(np.var(anchor_y))
    if variance == 0.0:
        return 0.0
    return float(smoothing_factor) * anchor_y.size * variance


def normalize(arr, mean=None, std=None):
    """Normalize array using precomputed mean and std per lead."""
    if mean is None or std is None:
        raise ValueError("Mean and std must be provided for normalization.")

    norm = (arr - mean[:, None]) / (std[:, None] + 1e-8)
    return norm


def convert_ecg_segment_to_vcg(
    ecg_segment: np.ndarray, lead_to_index: dict[str, int]
) -> np.ndarray:
    """Convert a 12-lead ECG segment to XYZ coordinates using the Kors transformation matrix.

    References:
        Kors et al. (1990). Reconstruction of the Frank vectorcardiogram from
        standard electrocardiographic leads: diagnostic comparison of different
        methods. European Heart Journal, 11(12), 1083-1092.
    """
    missing = [l for l in _VCG_LEADS if l not in lead_to_index]
    if missing:
        raise ValueError(f"VCG requires {_VCG_LEADS}; missing: {missing}")
    ecg_8 = ecg_segment[[lead_to_index[l] for l in _VCG_LEADS], :]  # (8, n_timesteps)
    kors_matrix = np.array(
        [
            [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
            [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
            [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31],
        ]
    )

    ecg_xyz = kors_matrix @ ecg_8  # (3, n_timesteps)

    return ecg_xyz
