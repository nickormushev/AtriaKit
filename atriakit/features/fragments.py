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


def _turning_points(segment: np.ndarray, threshold: float) -> list[int]:
    """Indices that split ``segment`` into monotonic runs: its first sample, its turning points, its last sample.

    With ``threshold > 0`` a peak only counts once the signal has fallen at least ``threshold`` below it (and a dip
    once it has risen at least ``threshold`` above it), so noise ripple smaller than the threshold does not split a
    run. With ``threshold <= 0`` every change of slope sign is a turning point.
    """
    last = len(segment) - 1
    if threshold <= 0:
        extrema = np.where(np.diff(np.sign(np.diff(segment))) != 0)[0] + 1
        return [0, *extrema, last]

    points = [0]
    direction = 0  # +1 rising, -1 falling, 0 not clearly moving yet
    candidate = 0  # the highest point so far when rising, the lowest when falling
    for i in range(1, len(segment)):
        move = segment[i] - segment[candidate]
        if direction == 0:
            if abs(move) >= threshold:
                direction, candidate = (1 if move > 0 else -1), i
        elif move * direction > 0:  # still going the same way: a new candidate
            candidate = i
        elif -move * direction >= threshold:  # turned back far enough: the candidate was a real turning point
            points.append(candidate)
            direction, candidate = -direction, i
    return [*points, last]


def fragments_finder(
    segment,
    lead_noise: float,
    fs,
    min_fragment_length_ms=10,
    noise_multiplier=1.0,
    hysteresis: bool = True,
) -> list:
    """Detect fragments in a P-wave segment.

    A fragment is a monotonic run between two consecutive boundaries, where the
    boundaries are the segment's first sample, its turning points, and its last
    sample. A fragment is kept if its amplitude change (end - start) reaches
    ``noise_multiplier × lead_noise`` and its duration exceeds
    ``min_fragment_length_ms``. A segment without turning points is a single fragment.

    With ``hysteresis`` (default), a turning point only counts once the signal
    has reversed from it by that same amplitude threshold: the criterion of
    Murthy et al. (adjacent extrema more than 3 noise SDs apart), applied to
    single beats. Otherwise any change of slope sign is a turning point, and on a
    noisy single beat ripple can cut every deflection into runs shorter than
    ``min_fragment_length_ms``, leaving no fragment at all.

    Args:
        segment: 1-D signal array of the P-wave segment.
        lead_noise: Noise level estimate for the lead in the same units as
            ``segment``. Used to threshold fragment amplitude.
        fs: Sampling frequency in Hz.
        min_fragment_length_ms: Minimum fragment duration in milliseconds.
            Shorter fragments are discarded. Defaults to ``10``.
        noise_multiplier: Amplitude threshold as a multiple of ``lead_noise``.
            Defaults to ``1.0``.
        hysteresis: Count a turning point only after a reversal of at least the
            amplitude threshold. Defaults to ``True``.

    Returns:
        List of fragments, each represented as a tuple
        ``(start_idx, end_idx, start_amp, end_amp)``.
    """
    if len(segment) == 0:
        return []

    threshold = noise_multiplier * lead_noise
    boundaries = _turning_points(np.asarray(segment, dtype=float), threshold if hysteresis else 0.0)

    min_samples = int(min_fragment_length_ms * fs / 1000)

    fragments = []
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        start_amp, end_amp = segment[start_idx], segment[end_idx]
        long_enough = (end_idx - start_idx) > min_samples
        large_enough = abs(end_amp - start_amp) >= noise_multiplier * lead_noise
        if long_enough and large_enough:
            fragments.append((start_idx, end_idx, start_amp, end_amp))

    return fragments
