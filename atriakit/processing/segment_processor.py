from collections.abc import Callable

import numpy as np

from atriakit.annotations import Annotations
from atriakit.configs.segment_config import SegmentConfig
from atriakit.models.ecg_data import ECGData
from atriakit.utils import apply_baseline_correction


class SegmentProcessor:
    """Extracts annotation-bounded, baseline-corrected segments and computes metrics over them.

    Args:
        cfg: Segment-boundary and baseline-correction settings.

    Attributes:
        baseline_correction_type: Baseline correction applied to extracted segments.
        skip_first_ms: Samples dropped from the start of each annotated segment.
    """

    def __init__(self, cfg: SegmentConfig):
        self.baseline_correction_type = cfg.baseline_correction_type
        self.skip_first_ms = cfg.skip_first_ms

    @staticmethod
    def _identity_segment(segment: np.ndarray, _row) -> np.ndarray:
        return segment

    def _compute_onset_offset(self, row, signal: np.ndarray, fs: int) -> tuple[int, int]:
        skip_samples = int(self.skip_first_ms / 1000 * fs)
        onset = int(skip_samples + row.onset)
        offset = int(row.offset + 1)
        offset = min(offset, signal.shape[-1])

        if onset < 0 or onset >= offset:
            raise ValueError(
                f"Invalid annotation bounds: onset={onset}, offset={offset}, "
                f"signal_length={len(signal)}, lead={row.lead}, p_wave_id={row.p_wave_id}"
            )

        return onset, offset

    def extract_segment(
        self,
        signal: np.ndarray,
        segment_selector: Callable[[np.ndarray, object], np.ndarray],
        fs: int,
        row,
    ) -> np.ndarray:
        """Slice, baseline-correct, and select the annotated sub-segment of ``signal``.

        Args:
            signal: Source signal, shape ``(n_samples,)`` or ``(n_leads, n_samples)``.
            segment_selector: Callable(segment, row) -> sub-segment to apply after
                baseline correction.
            fs: Sampling frequency in Hz.
            row: Annotation row with ``onset``/``offset`` (and ``lead``, ``p_wave_id``
                used for error messages).

        Returns:
            The selected, baseline-corrected sub-segment.

        Raises:
            ValueError: If the annotation's onset/offset bounds are invalid for
                ``signal``.
        """
        onset, offset = self._compute_onset_offset(row, signal, fs)

        segment = signal[:, onset:offset] if signal.ndim == 2 else signal[onset:offset]

        segment = apply_baseline_correction(segment, self.baseline_correction_type)
        segment = segment_selector(segment, row)

        return segment

    def compute_segment_metric(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        metric_func: Callable[[np.ndarray, object], object],
        get_signal: Callable | None = None,
        segment_selector: Callable[[np.ndarray, object], np.ndarray] | None = None,
        nan_value=np.nan,
    ) -> list:
        """Iterate annotations and apply a metric function to each extracted signal segment.

        Args:
            annotations: Beat annotations with onset/offset columns.
            ecg_data: ECG signal source.
            metric_func: Callable(segment, row) -> scalar applied to each segment.
            get_signal: Optional callable(row) -> signal array; defaults to the
                raw lead signal from ``ecg_data``.
            segment_selector: Optional callable(segment, row) -> sub-segment;
                defaults to the full segment.
            nan_value: Value to use when a segment cannot be extracted.

        Returns:
            List of per-annotation metric values, in annotation order.
        """
        if annotations.empty:
            return np.array([])

        if get_signal is None:
            get_signal = lambda row: ecg_data.get_lead_signal(row.lead)

        if segment_selector is None:
            segment_selector = self._identity_segment

        metric = []
        fs = ecg_data.get_sampling_frequency()
        for row in annotations.itertuples(index=False):
            signal = get_signal(row)
            segment = self.extract_segment(signal, segment_selector, fs, row)

            if len(segment) == 0:
                metric.append(nan_value)
                continue

            metric.append(metric_func(segment, row))

        return metric
