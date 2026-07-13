import numpy as np
import nolds

def shannon_entropy(segment: np.ndarray, n_bins: int = 256, bin_range: tuple = None) -> float:
    """Compute Shannon entropy of a 1-D signal using histogram-based estimation.

    Args:
        segment: 1-D signal array.
        n_bins: Number of histogram bins. Defaults to ``256``.
        bin_range: Fixed ``(min, max)`` amplitude range for the histogram.
            ``None`` uses the signal's own min/max. Set a fixed range for
            cross-recording comparability.

    Returns:
        Shannon entropy in bits, or ``0.0`` if the histogram is empty.
    """
    hist, _ = np.histogram(segment, bins=n_bins, range=bin_range, density=False)

    if np.sum(hist) == 0:
        return 0.0

    prob = hist / np.sum(hist)
    prob = prob[prob > 0]

    return -np.sum(prob * np.log2(prob))

def sample_entropy_nolds(segment, m=2, r_factor=0.2):
    """Compute sample entropy of a 1-D signal using the ``nolds`` library.

    Args:
        segment: 1-D signal array.
        m: Embedding dimension (template length). Defaults to ``2``.
        r_factor: Tolerance as a fraction of the signal's standard deviation.
            Defaults to ``0.2``.

    Returns:
        Sample entropy value, ``0.0`` for a flat signal, or ``nan`` if the
        segment is too short (length ≤ ``m + 1``).
    """
    segment = np.asarray(segment)

    if len(segment) <= m + 1:
        return np.nan

    std = np.std(segment)
    if std == 0:
        return 0.0

    r = r_factor * std

    return nolds.sampen(segment, emb_dim=m, tolerance=r)