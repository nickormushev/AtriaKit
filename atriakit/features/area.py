import numpy as np


def ptf_segment_selector(segment, row, onset) -> np.ndarray:
    """Relies on inflection point existing. This method is meant to be used only by ptf in FeatureCalculators."""
    inflection = int(row.inflection_point)

    morph = getattr(row, "p_wave_morphology", "")
    monophasic = isinstance(morph, str) and (
        "Monophasic" in morph or morph == "Complex"
    )

    if inflection == -1 or monophasic:
        return np.array([])

    if (inflection != -1 and inflection < onset) or inflection >= len(segment) + onset:
        raise ValueError(
            f"Invalid inflection point for p_wave_id={row.p_wave_id}, lead={row.lead}: inflection={inflection}, onset={onset}, segment_length={len(segment)}"
        )

    return segment[int(inflection - onset) :]


def ptf(segment, fs):
    """Compute P-wave terminal force as duration × depth of the negative trough.

    Args:
        segment: 1-D signal array of the terminal P-wave segment in mV.
        fs: Sampling frequency in Hz.

    Returns:
        PTF value in mV·s, ``nan`` if ``segment`` is empty, or 0 if the
        segment has no negative part.
    """
    if len(segment) == 0:
        return np.nan

    if not np.any(segment < 0):
        return 0.0

    return (len(segment) / fs) * -np.min(segment)


def _find_inflection(segment: np.ndarray, lead: str = "") -> int:
    """Return the sample index of the inflection point on a biphasic P-wave.

    Searches zero-crossings of the second derivative and picks the one with the
    steepest first-derivative slope — steepest descent for standard leads,
    steepest ascent for aVR (where the wave is inverted). Falls back to the
    global argmin/argmax of the first derivative when no zero-crossings exist.

    Args:
        segment: 1-D signed signal array of the P-wave segment.
        lead: Lead name; ``"aVR"`` inverts the slope criterion.

    Returns:
        Index into ``segment`` of the detected inflection point.
    """
    d1 = np.gradient(segment)
    d2 = np.gradient(d1)
    avr = lead == "aVR"

    sign_changes = np.where(np.diff(np.sign(d2)))[0]

    if len(sign_changes) == 0:
        return int(np.argmax(d1) if avr else np.argmin(d1))

    return int(
        sign_changes[np.argmax(d1[sign_changes])]
        if avr
        else sign_changes[np.argmin(d1[sign_changes])]
    )


def _find_zero_crossing(
    segment: np.ndarray, lead: str = ""
) -> tuple[int, int] | None:
    """Return the ``(start, end)`` slice of the terminal negative run, or ``None``.

    The run starts at the last positive-to-negative crossing at or before the
    deepest negative sample and ends at the first non-negative sample after it
    (or the segment end), so it contains the trough and no positive tail. This
    is the terminal negative deflection of the conventional PTF (Morris index).
    The signal is inverted for aVR.

    Args:
        segment: 1-D signed signal array of the P-wave segment.
        lead: Lead name; ``"aVR"`` inverts the signal before searching.

    Returns:
        Half-open ``(start, end)`` indices into ``segment``, or ``None`` if the
        wave never goes from non-negative to negative.
    """
    signal = -segment if lead == "aVR" else segment
    negative = signal < 0
    crossings = np.where(~negative[:-1] & negative[1:])[0] + 1
    crossings = crossings[crossings <= np.argmin(signal)]
    if len(crossings) == 0:
        return None
    start = int(crossings[-1])
    non_negative = np.flatnonzero(~negative[start:])
    return start, start + int(non_negative[0]) if len(non_negative) else len(signal)


def ptf_auto(
    segment,
    fs,
    lead: str = "",
    seg_morph: np.ndarray | None = None,
    zero_crossing: bool = False,
):
    """Compute PTF without morphology classification by auto-detecting the terminal segment.

    Assumes a biphasic P-wave with a positive initial deflection followed by a
    negative terminal deflection (inverted in aVR). By default the terminal
    segment runs from the inflection point (the zero-crossing of the second
    derivative with the steepest first-derivative slope) to the segment end.
    With ``zero_crossing=True`` it is restricted to the negative run that
    starts at the positive-to-negative zero-crossing and contains the trough,
    which gives the conventional PTF (Morris et al., 1964). Both modes then
    delegate to :func:`ptf`.

    Args:
        segment: 1-D signal array of the full P-wave segment in mV.
        fs: Sampling frequency in Hz.
        lead: Lead name used to handle the aVR inversion. Defaults to ``""``.
        seg_morph: Optional smoothed version of ``segment`` used only for
            inflection/zero-crossing detection; amplitude is always measured
            on ``segment``.
        zero_crossing: If True, use only the negative run that starts at the
            positive-to-negative zero-crossing (conventional PTF) instead of
            the steepest second-derivative inflection to the segment end.
            Returns 0 if the wave has no such crossing.

    Returns:
        PTF value in mV·s, or 0.0 if there is no negative terminal part.

    Raises:
        ValueError: If ``seg_morph`` is given with a different length than
            ``segment``.

    References:
        Morris, J. J., et al. (1964). P wave analysis in valvular heart
        disease. Circulation, 29(2), 242-252.
    """
    if seg_morph is not None and len(seg_morph) != len(segment):
        raise ValueError(
            f"seg_morph length ({len(seg_morph)}) must match segment length ({len(segment)})"
        )
    detect = seg_morph if seg_morph is not None else segment
    if zero_crossing:
        bounds = _find_zero_crossing(detect, lead)
        if bounds is None:
            return 0.0
        terminal = segment[slice(*bounds)]
    else:
        terminal = segment[_find_inflection(detect, lead) :]
    # aVR is inverted, so flip it to put the terminal deflection on the negative side.
    return ptf(-terminal if lead == "aVR" else terminal, fs)


def area(segment, fs):
    """Compute the P-wave area as the integral of the absolute signal.

    Args:
        segment: 1-D signal array of the P-wave segment in mV.
        fs: Sampling frequency in Hz.

    Returns:
        Area in mV·s.
    """
    return np.sum(np.abs(segment)) / fs
