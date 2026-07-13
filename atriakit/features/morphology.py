import numpy as np

def calculate_inflexion(ecg_derivative, row_onset, phase_type):
    """Return the inflexion point index for a biphasic P-wave.

    Finds the zero-crossing of the second derivative with the steepest
    first-derivative slope: steepest descent for ``"Biphasic Positive-Negative"``,
    steepest ascent for ``"Biphasic Negative-Positive"``.

    Args:
        ecg_derivative: First derivative of the trimmed P-wave segment.
        row_onset: Onset index of the P-wave in the original signal, used to
            convert the local index back to signal coordinates.
        phase_type: Morphology label from ``morphology_classifier``; only
            biphasic patterns produce a valid inflexion point.

    Returns:
        Inflexion point index in signal coordinates, or ``-1`` if the phase
        type is not biphasic or no zero-crossings are found.
    """
    ecg_second_derivative = np.diff(ecg_derivative)
    signs = ecg_second_derivative >= 0
    zero_crossings = np.where(np.diff(signs) != 0)[0] + 1

    if len(zero_crossings) == 0:
        return -1

    if phase_type == "Biphasic Positive-Negative":
        selected_idx = zero_crossings[np.argmin(ecg_derivative[zero_crossings])]
        return selected_idx + row_onset + 1
    
    if phase_type == "Biphasic Negative-Positive":
        selected_idx = zero_crossings[np.argmax(ecg_derivative[zero_crossings])]
        return selected_idx + row_onset + 1

    return -1

def validate_group(count, duration_threshold, segment, start, voltage_threshold):
    """Return ``True`` if a sign group meets the duration and amplitude thresholds."""
    end = min(start + count, len(segment))
    ptp_amplitude = np.ptp(segment[start:end])
    return (
        count >= duration_threshold * len(segment)
        and abs(ptp_amplitude) >= voltage_threshold
    )

def extract_sign_groups(signs, duration_threshold, segment, voltage_threshold=0):
    """Extract validated sign groups from the derivative sign array.

    Runs through consecutive same-sign runs in ``signs`` and keeps only those
    that meet the duration and amplitude thresholds, deduplicating adjacent
    identical signs afterwards.

    Args:
        signs: Boolean array where ``True`` = positive derivative sample.
        duration_threshold: Minimum fraction of the segment length a run must
            span to be kept.
        segment: Original P-wave signal, used to measure peak-to-peak amplitude
            for each candidate group.
        voltage_threshold: Minimum peak-to-peak amplitude a group must have to
            be kept. Defaults to ``0``.

    Returns:
        List of booleans representing the validated sign sequence
        (``True`` = positive phase, ``False`` = negative phase).
    """
    groups = []
    count = 1
    if len(signs) == 0:
        print("No signs to process")
        return groups
    last_sign = signs[0]
    start = 0

    for idx, sign in enumerate(signs[1:], start=1):
        if sign == last_sign:
            count += 1
        else:
            if validate_group(
                count, duration_threshold, segment, start, voltage_threshold
            ):
                groups.append(last_sign)

            last_sign = sign
            start = idx
            count = 1

    if validate_group(count, duration_threshold, segment, start, voltage_threshold):
        groups.append(last_sign)

    final_groups = []
    for g in groups:
        if final_groups and g == final_groups[-1]:
            continue
        final_groups.append(g)

    return final_groups


def morphology_classifier(
    ecg_segment: np.ndarray,
    row_onset: int,
    lead_noise: float,
    morphology_min_phase_fraction: float,
    morphology_noise_multiplier: float,
):
    """Classify the morphology of a single P-wave segment.

    Analyses the sign pattern of the first derivative to determine whether the
    P-wave is monophasic, biphasic, complex, or flat. Sign groups are filtered
    by duration and amplitude to suppress noise.

    Args:
        ecg_segment: ECG signal segment corresponding to the annotated P-wave.
        row_onset: Onset index of the P-wave in the original signal.
        lead_noise: Estimated noise level for this lead.
        morphology_min_phase_fraction: Minimum fraction of segment length a
            sign group must span to count as a distinct phase. Also used as the
            amplitude threshold for distinguishing biphasic from complex shapes.
        morphology_noise_multiplier: Multiplier applied to ``lead_noise`` when
            filtering derivative sign groups.

    Returns:
        Tuple of ``(morphology_label, inflexion_point)``, where
        ``morphology_label`` is one of ``"Monophasic Positive"``,
        ``"Monophasic Negative"``, ``"Biphasic Positive-Negative"``,
        ``"Biphasic Negative-Positive"``, ``"Complex"``, ``"Flat"``, or
        ``"N/A"`` for empty segments, and ``inflexion_point`` is the index in
        the original signal (``-1`` if not applicable).
    """
    if len(ecg_segment) == 0:
        return "N/A", -1

    # Trim 1 sample from each end to reduce onset/offset boundary edge effects.
    # Requires at least 5 samples so the trimmed segment has a usable derivative.
    if len(ecg_segment) > 4:
        segment = ecg_segment[1:-1]
        trim = 1
    else:
        segment = ecg_segment
        trim = 0

    ecg_derivative = np.diff(segment)
    signs = ecg_derivative >= 0

    group_signs = extract_sign_groups(
        signs,
        morphology_min_phase_fraction,
        segment,
        lead_noise * morphology_noise_multiplier,
    )

    if len(group_signs) == 0:
        return "Flat", -1

    phase_type = "Complex"
    pattern = tuple(group_signs)
    patterns_map = {
        (True,): "Monophasic Positive",
        (False,): "Monophasic Negative",
        (True, False): "Monophasic Positive",
        (False, True): "Monophasic Negative",
        (True, False, True): "Biphasic Positive-Negative",
        (False, True, False): "Biphasic Negative-Positive",
    }

    if pattern in patterns_map:
        phase_type = patterns_map[pattern]

    # The 3-group patterns (up-down-up / down-up-down) are only genuinely biphasic
    # when the opposing-phase excursion is a meaningful fraction of the dominant peak.
    # Without this check, M-shaped waves and monophasic waves with small artifacts
    # are misclassified as biphasic.
    if pattern == (True, False, True):
        seg_max = float(segment.max())
        seg_min = float(segment.min())
        if seg_max <= 0 or abs(seg_min) < morphology_min_phase_fraction * seg_max:
            phase_type = "Complex"
    elif pattern == (False, True, False):
        seg_min = float(segment.min())
        seg_max = float(segment.max())
        if seg_min >= 0 or abs(seg_max) < morphology_min_phase_fraction * abs(seg_min):
            phase_type = "Complex"

    inflexion_point = calculate_inflexion(ecg_derivative, row_onset + trim, phase_type)

    return (phase_type, int(inflexion_point))
