import numpy as np


def extrema_counter(
    segment,
    lead,
    max_abs_amplitude_per_lead,
    threshold_multiplier: float = 0.1,
) -> int:
    """Count valid extrema in a P-wave segment.

    An extremum is valid if its amplitude difference with both its left and
    right neighbor (the adjacent extremum, or the segment boundary for the
    first and last) exceeds ``threshold_multiplier × max_abs_amplitude``
    for that lead. This filters out noise-level oscillations.

    Args:
        segment: 1-D signal array of the P-wave segment.
        lead: Lead name used to look up the per-lead amplitude threshold.
        max_abs_amplitude_per_lead: Dict mapping lead name to its max absolute
            amplitude, used to scale the threshold.
        threshold_multiplier: Fraction of max amplitude an extremum must
            exceed to be counted. Defaults to ``0.1``.

    Returns:
        Number of valid extrema in the segment.
    """
    if len(segment) < 3:
        return 0

    dt = np.diff(segment)
    extrema = np.where(np.diff(np.sign(dt)) != 0)[0] + 1

    if len(extrema) == 0:
        return 0

    valid_extrema_count = 0
    threshold = threshold_multiplier * max_abs_amplitude_per_lead[lead]

    for idx, curr_idx in enumerate(extrema):
        curr = segment[curr_idx]

        # Left neighbor: previous extremum, or segment start for first
        left = segment[extrema[idx - 1]] if idx > 0 else segment[0]
        # Right neighbor: next extremum, or segment end for last
        right = segment[extrema[idx + 1]] if idx < len(extrema) - 1 else segment[-1]

        if abs(curr - left) >= threshold and abs(curr - right) >= threshold:
            valid_extrema_count += 1

    return valid_extrema_count
