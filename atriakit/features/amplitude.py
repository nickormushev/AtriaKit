import numpy as np

def max_amplitude(segment, _) -> np.ndarray:
    return np.max(segment)

def min_amplitude(
    segment, _
) -> np.ndarray:
    return np.min(segment)

def max_absolute_amplitude(
    segment, _
) -> np.ndarray:
    return np.max(np.abs(segment))

def peak_to_peak_amplitude(
    segment, _
) -> np.ndarray:
    return np.ptp(segment)