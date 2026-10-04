import numpy as np


def fragment_count(per_annotation_fragments: list) -> list:
    """Return the number of fragments for each annotation."""
    counts = [len(fragments) for fragments in per_annotation_fragments]
    return counts


def fragment_width(per_annotation_fragments: list, fs: int) -> list:
    """Return the mean fragment width in seconds for each annotation.

    Returns:
        List of mean widths (s) per annotation. Returns ``0`` when an
        annotation has no fragments.
    """
    avg_widths = []
    for sublist in per_annotation_fragments:
        if len(sublist) == 0:
            avg_widths.append(0)
            continue

        widths = [(frag[1] - frag[0]) / fs for frag in sublist]
        widths = np.array(widths, dtype=float)
        valid = ~np.isnan(widths)

        if valid.any():
            mean_width = widths[valid].mean()
        else:
            mean_width = np.nan

        avg_widths.append(0 if np.isnan(mean_width) else mean_width)

    return avg_widths


def fragment_amplitude(per_annotation_fragments: list) -> list:
    """Return the mean fragment amplitude in mV for each annotation.

    Amplitude is defined as the absolute difference between a fragment's
    start and end sample values.

    Returns:
        List of mean amplitudes (mV) per annotation. Returns ``0`` when an
        annotation has no fragments.
    """
    avg_amplitudes = []
    for sublist in per_annotation_fragments:
        if len(sublist) == 0:
            avg_amplitudes.append(0)
            continue

        amplitudes = [np.abs(frag[3] - frag[2]) for frag in sublist]
        amplitudes = np.array(amplitudes, dtype=float)
        valid = ~np.isnan(amplitudes)
        if valid.any():
            mean_amp = amplitudes[valid].mean()
        else:
            mean_amp = np.nan

        avg_amplitudes.append(0 if np.isnan(mean_amp) else mean_amp)

    return avg_amplitudes


def fragments_finder(
    segment, lead_noise: float, fs, min_fragment_length_ms=10, noise_multiplier=1.0
) -> list:
    """Detect fragments in a P-wave segment.

    A fragment is a monotonic run between two consecutive boundaries, where the
    boundaries are the segment's first sample, its local extrema, and its last
    sample. A fragment is kept if its amplitude change (end - start) reaches
    ``noise_multiplier × lead_noise`` and its duration exceeds
    ``min_fragment_length_ms``. A segment without extrema is a single fragment.

    Args:
        segment: 1-D signal array of the P-wave segment.
        lead_noise: Noise level estimate for the lead in the same units as
            ``segment``. Used to threshold fragment amplitude.
        fs: Sampling frequency in Hz.
        min_fragment_length_ms: Minimum fragment duration in milliseconds.
            Shorter fragments are discarded. Defaults to ``10``.
        noise_multiplier: Amplitude threshold as a multiple of ``lead_noise``.
            Defaults to ``1.0``.

    Returns:
        List of fragments, each represented as a tuple
        ``(start_idx, end_idx, start_amp, end_amp)``.
    """
    if len(segment) == 0:
        return []

    derivative = np.diff(segment)
    extrema = np.where(np.diff(np.sign(derivative)) != 0)[0] + 1
    boundaries = [0, *extrema, len(segment) - 1]

    min_samples = int(min_fragment_length_ms * fs / 1000)

    fragments = []
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        start_amp, end_amp = segment[start_idx], segment[end_idx]
        long_enough = (end_idx - start_idx) > min_samples
        large_enough = abs(end_amp - start_amp) >= noise_multiplier * lead_noise
        if long_enough and large_enough:
            fragments.append((start_idx, end_idx, start_amp, end_amp))

    return fragments
