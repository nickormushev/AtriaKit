from collections import namedtuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from atriakit.configs.segment_config import SegmentConfig
from atriakit.models.annotations import Annotations
from atriakit.processing.segment_processor import SegmentProcessor

Row = namedtuple("Row", ["onset", "offset", "lead", "p_wave_id"])


def test_extract_segment_applies_skip_baseline_and_selector():
    """extract_segment should shift onset by skip_first_ms, baseline-correct, then run the selector."""
    processor = SegmentProcessor(
        SegmentConfig(baseline_correction_type="onset", skip_first_ms=2)
    )
    signal = np.array([0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 0.0])
    row = Row(onset=0, offset=4, lead="I", p_wave_id=1)

    # fs=1000 -> skip_first_ms=2 -> skip 2 samples -> slice is signal[2:5] = [10, 11, 12]
    # "onset" baseline correction then subtracts the first sample of the slice.
    segment = processor.extract_segment(
        signal, segment_selector=lambda seg, _row: seg, fs=1000, row=row
    )

    assert np.allclose(segment, [0.0, 1.0, 2.0])


def test_extract_segment_invalid_bounds_raises():
    processor = SegmentProcessor(SegmentConfig())
    signal = np.array([0.0, 1.0, 2.0])
    row = Row(onset=5, offset=1, lead="I", p_wave_id=2)

    with pytest.raises(ValueError, match="Invalid annotation bounds"):
        processor.extract_segment(
            signal, segment_selector=lambda seg, _row: seg, fs=500, row=row
        )


def test_compute_segment_metric_uses_nan_value_for_empty_selection():
    """When segment_selector empties out a segment, its metric should fall back to nan_value."""
    processor = SegmentProcessor(SegmentConfig())
    annotations = Annotations(
        pd.DataFrame(
            {
                "lead": ["I", "I"],
                "onset": [0, 0],
                "offset": [3, 3],
                "p_wave_id": [1, 2],
                "file_path": ["rec1", "rec1"],
            }
        )
    )
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    ecg_data = MagicMock()
    ecg_data.get_sampling_frequency.return_value = 500

    def segment_selector(segment, row):
        return segment if row.p_wave_id == 1 else np.array([])

    metric = processor.compute_segment_metric(
        annotations,
        ecg_data,
        metric_func=lambda segment, row: float(np.sum(segment)),
        get_signal=lambda row: signal,
        segment_selector=segment_selector,
        nan_value=-1,
    )

    assert metric == pytest.approx([10.0, -1])


def test_compute_segment_metric_happy_path_uses_default_signal_and_selector():
    """With no overrides, signals come from ecg_data.get_lead_signal and the full segment reaches metric_func."""
    processor = SegmentProcessor(SegmentConfig())
    annotations = Annotations(
        pd.DataFrame(
            {
                "lead": ["I", "II"],
                "onset": [0, 1],
                "offset": [2, 3],
                "p_wave_id": [1, 2],
                "file_path": ["rec1", "rec1"],
            }
        )
    )
    lead_signals = {
        "I": np.array([1.0, 2.0, 3.0, 4.0]),
        "II": np.array([10.0, 20.0, 30.0, 40.0]),
    }
    ecg_data = MagicMock()
    ecg_data.get_sampling_frequency.return_value = 500
    ecg_data.get_lead_signal.side_effect = lambda lead, **kwargs: lead_signals[lead].copy()

    metric = processor.compute_segment_metric(
        annotations, ecg_data, metric_func=lambda segment, row: float(np.sum(segment))
    )

    assert metric == pytest.approx([6.0, 90.0])


def test_extract_segment_2d_signal_slices_all_leads():
    """A 2-D (n_leads, n_samples) signal should be sliced across every lead, not just one row."""
    processor = SegmentProcessor(SegmentConfig())
    signal = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [0.0, 100.0, 200.0, 300.0, 400.0],
        ]
    )
    row = Row(onset=1, offset=3, lead="VCG", p_wave_id=1)

    segment = processor.extract_segment(
        signal, segment_selector=lambda seg, _row: seg, fs=500, row=row
    )

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
        ]
    )
    assert np.allclose(segment, expected)
