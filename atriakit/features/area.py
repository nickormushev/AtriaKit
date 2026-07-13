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
    """Compute P-wave terminal force as duration × peak absolute amplitude.

    Args:
        segment: 1-D signal array of the terminal P-wave segment in mV.
        fs: Sampling frequency in Hz.

    Returns:
        PTF value in mV·s, or ``nan`` if ``segment`` is empty.
    """
    if len(segment) == 0:
        return np.nan
    return (len(segment) / fs) * np.max(np.abs(segment))


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


def ptf_auto(segment, fs, lead: str = "", seg_morph: np.ndarray | None = None):
    """Compute PTF without morphology classification by auto-detecting the inflection point.

    Assumes a biphasic P-wave with a positive initial deflection followed by a
    negative terminal deflection (inverted in aVR). The inflection point is
    found as the zero-crossing of the second derivative with the steepest
    first-derivative slope, then delegates to :func:`ptf`.

    Args:
        segment: 1-D signal array of the full P-wave segment in mV.
        fs: Sampling frequency in Hz.
        lead: Lead name used to handle the aVR inversion. Defaults to ``""``.
        seg_morph: Optional smoothed version of ``segment`` used only for
            inflection detection; amplitude is always measured on ``segment``.

    Returns:
        PTF value in mV·s, or ``nan`` if the terminal segment is empty.
    """
    inflection_idx = _find_inflection(
        seg_morph if seg_morph is not None else segment, lead
    )
    return ptf(segment[inflection_idx:], fs)


def area(segment, fs):
    """Compute the P-wave area as the integral of the absolute signal.

    Args:
        segment: 1-D signal array of the P-wave segment in mV.
        fs: Sampling frequency in Hz.

    Returns:
        Area in mV·s.
    """
    return np.sum(np.abs(segment)) / fs
