from datetime import datetime, timedelta

import numpy as np


def time_to_sample(time_sec, fs, sample_count):
    """Convert seconds -> nearest sample index (clipped)."""
    idx = int(round(float(time_sec) * fs))
    idx = max(0, min(idx, sample_count - 1))
    return idx


def sec_to_hms(t_sec, start_time, ms=False):
    """
    Convert seconds offset (from content_start_time) to HH:MM:SS string.
    t_sec can be scalar or array-like.
    """
    if isinstance(t_sec, (list, np.ndarray)):
        return [sec_to_hms(t, start_time) for t in t_sec]

    start_dt = datetime.strptime(str(start_time), "%H%M%S")
    dt = start_dt + timedelta(seconds=float(t_sec))
    if not ms:
        return dt.strftime("%H:%M:%S")

    return dt.strftime("%H:%M:%S") + f":{int(dt.microsecond / 1000):03d}"
