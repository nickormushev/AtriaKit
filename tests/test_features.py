# pylint: disable=missing-function-docstring, unused-argument

from types import MethodType
from unittest.mock import MagicMock

import neurokit2 as nk
import numpy as np
import pandas as pd
import pytest

from atriakit.configs.segment_config import SegmentConfig
from atriakit.configs.signal_preprocessor_config import SignalPreprocessorConfig
from atriakit.feature_calculator import FeatureCalculators
from atriakit.features.area import _find_inflection
from atriakit.features.area import ptf_auto as ptf_auto_fn
from atriakit.features.vcg import vcg_axis_angles
from atriakit.io import AnnotationsLoader
from atriakit.models.annotation_schema import AnnotationSchema
from atriakit.models.annotations import Annotations
from atriakit.preprocessing.signals import SignalPreprocessor
from atriakit.processing.segment_processor import SegmentProcessor


def make_feature_calculator(**overrides):
    feature_calculator = FeatureCalculators.__new__(FeatureCalculators)

    defaults = {
        "baseline_correction_type": "none",
        "skip_first_ms": 0,
        "signal_preprocessor": SignalPreprocessor(
            SignalPreprocessorConfig(lowcut=None, highcut=None, notch_freq=50)
        ),
        "morphology_preprocessor": SignalPreprocessor(
            SignalPreprocessorConfig(lowcut=None, highcut=None, notch_freq=50)
        ),
    }
    defaults.update(overrides)

    feature_calculator.segment_processor = SegmentProcessor(
        SegmentConfig(
            baseline_correction_type=defaults.pop("baseline_correction_type"),
            skip_first_ms=defaults.pop("skip_first_ms"),
        )
    )

    # Only set attributes that still exist on the object (guard against stale keys)
    valid_attrs = set(defaults.keys())
    for attr in valid_attrs:
        setattr(feature_calculator, attr, defaults[attr])

    return feature_calculator


def make_mock_ecg_data(lead_signals, sampling_frequency=500):
    ecg_data = MagicMock()

    _lead_to_index = {lead: i for i, lead in enumerate(lead_signals.keys())}
    ecg_data.get_lead_to_index.return_value = _lead_to_index

    def get_lead_signal(lead, preprocessor=None, **kwargs):
        signal = lead_signals[lead]
        return signal.copy()

    ecg_data.get_lead_signal.side_effect = get_lead_signal
    ecg_data.get_sampling_frequency.return_value = sampling_frequency
    return ecg_data


@pytest.fixture
def sample_annotations():
    return Annotations(
        pd.DataFrame(
            {
                AnnotationSchema.LEAD: ["I", "I", "II"],
                AnnotationSchema.ONSET: [0, 2, 0],
                AnnotationSchema.OFFSET: [2, 4, 3],
                AnnotationSchema.P_WAVE_ID: [1, 2, 3],
                AnnotationSchema.FILE_PATH: ["rec1", "rec1", "rec1"],
                AnnotationSchema.TYPE: ["Before", "Before", "Before"],
            }
        )
    )


@pytest.fixture
def sample_annotations_with_inflection():
    return Annotations(
        pd.DataFrame(
            {
                AnnotationSchema.LEAD: ["I", "II", "I"],
                AnnotationSchema.ONSET: [1, 0, 0],
                AnnotationSchema.OFFSET: [8, 5, 5],
                AnnotationSchema.P_WAVE_ID: [1, 2, 3],
                AnnotationSchema.FILE_PATH: ["rec1", "rec1", "rec1"],
                "inflection_point": [4, 3, -1],
                "p_wave_morphology": [
                    "Biphasic",
                    "Biphasic",
                    "Monophasic Positive",
                ],
            }
        )
    )


@pytest.fixture
def feature_calculator_no_baseline():
    feature_calculator = make_feature_calculator()
    ecg_data = make_mock_ecg_data(
        {
            "I": np.array([0, 1, 2, 3, 4]),
            "II": np.array([-1, 0, 1, -2]),
        }
    )
    return feature_calculator, ecg_data


@pytest.fixture
def feature_calculator_inflection():
    feature_calculator = make_feature_calculator()
    ecg_data = make_mock_ecg_data(
        {
            "I": np.array([0, 1, 2, 1, 0, -1, -2, -1, 0]),
            "II": np.array([-1, 0, 1, 0, -1, 0]),
        }
    )
    return feature_calculator, ecg_data


@pytest.fixture
def sample_annotations_axis():
    return Annotations(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "I",
                    AnnotationSchema.ONSET: 50,
                    AnnotationSchema.OFFSET: 60,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "aVF",
                    AnnotationSchema.ONSET: 52,
                    AnnotationSchema.OFFSET: 62,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
            ]
        )
    )


@pytest.fixture
def feature_calculator_axis():
    feature_calculator = make_feature_calculator()
    ecg_data = make_mock_ecg_data(
        {
            "I": np.pad(np.ones(10), (50, 140)),
            "aVF": np.pad(np.full(10, 2.0), (52, 138)),
        }
    )
    return feature_calculator, ecg_data


@pytest.fixture
def ecg_processor_with_noise():
    sampling_rate = 1000
    ecg_signal = nk.ecg_simulate(
        duration=5,
        sampling_rate=sampling_rate,
        heart_rate=70,
        noise=0.01,
    )

    processor = MagicMock()
    processor.signal_preprocessor = SignalPreprocessor(
        SignalPreprocessorConfig(lowcut=None, highcut=None, notch_freq=50)
    )
    processor.estimate_noise = MethodType(FeatureCalculators.estimate_noise, processor)
    processor._compute_lead_noise = MethodType(
        FeatureCalculators._compute_lead_noise,
        processor,
    )
    processor._get_signal_for_lead = MethodType(
        FeatureCalculators._get_signal_for_lead,
        processor,
    )
    processor._get_lead_signal = MethodType(
        FeatureCalculators._get_signal_for_lead,
        processor,
    )

    ecg_data = MagicMock()
    ecg_data.get_sampling_frequency.return_value = sampling_rate
    ecg_data.get_leads.return_value = ["I"]
    ecg_data.get_lead_signal.side_effect = (
        lambda lead, preprocessor=None, **kwargs: ecg_signal.copy()
    )

    return processor, ecg_data


@pytest.fixture
def sample_annotations_fragments():
    return Annotations(
        pd.DataFrame(
            {
                AnnotationSchema.LEAD: ["I"],
                AnnotationSchema.ONSET: [0],
                AnnotationSchema.OFFSET: [7],
                AnnotationSchema.P_WAVE_ID: [1],
                AnnotationSchema.FILE_PATH: ["rec1"],
            }
        )
    )


@pytest.fixture
def feature_calculator_fragments():
    feature_calculator = make_feature_calculator()
    feature_calculator.estimate_noise = lambda annotations, ecg_data, lead_signal=None, window_in_ms=None, sd_threshold=3: [
        0.01
    ]
    feature_calculator.noise = [0.01]

    signal = np.array([0.01, 0.02, -0.03, 0.05, -0.01, 0.0, 0.02, -0.02])
    ecg_data = make_mock_ecg_data({"I": signal}, sampling_frequency=1)
    return feature_calculator, ecg_data


@pytest.fixture
def feature_calculator_morphology():
    feature_calculator = make_feature_calculator()
    ecg = np.zeros((2, 10))
    ecg[0, 0:5] = np.array([0, 1, 2, 1, 0])
    ecg[1, 0:10] = np.array([0, 1, 2, 1, 0, -1, -2, -1, 0, -5])

    ecg_data = make_mock_ecg_data({"I": ecg[0], "II": ecg[1]})
    feature_calculator.estimate_noise = (
        lambda annotations, ecg_data, lead_signal=None, window_in_ms=None, sd_threshold=3: [
            0.0
        ]
        * 12
    )
    return feature_calculator, ecg_data


@pytest.fixture
def morphology_annotations():
    return Annotations(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "I",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 4,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "II",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 9,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
            ]
        )
    )


@pytest.fixture
def feature_calculator_heart_rate():
    feature_calculator = make_feature_calculator()
    peaks = [50, 150, 250]

    def peak_signal(height):
        signal = np.zeros(300)
        signal[peaks] = height
        return signal

    ecg_data = make_mock_ecg_data(
        {
            "II": peak_signal(1.0),
            "V5": peak_signal(0.8),
            "V6": peak_signal(0.9),
        }
    )
    return feature_calculator, ecg_data


@pytest.fixture
def hr_annotations():
    return AnnotationsLoader().from_dataframe(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "II",
                    AnnotationSchema.ONSET: 40,
                    AnnotationSchema.OFFSET: 45,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "II",
                    AnnotationSchema.ONSET: 140,
                    AnnotationSchema.OFFSET: 145,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "II",
                    AnnotationSchema.ONSET: 240,
                    AnnotationSchema.OFFSET: 245,
                    AnnotationSchema.P_WAVE_ID: 3,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V5",
                    AnnotationSchema.ONSET: 40,
                    AnnotationSchema.OFFSET: 45,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V5",
                    AnnotationSchema.ONSET: 140,
                    AnnotationSchema.OFFSET: 145,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V5",
                    AnnotationSchema.ONSET: 240,
                    AnnotationSchema.OFFSET: 245,
                    AnnotationSchema.P_WAVE_ID: 3,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V6",
                    AnnotationSchema.ONSET: 40,
                    AnnotationSchema.OFFSET: 45,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V6",
                    AnnotationSchema.ONSET: 140,
                    AnnotationSchema.OFFSET: 145,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V6",
                    AnnotationSchema.ONSET: 240,
                    AnnotationSchema.OFFSET: 245,
                    AnnotationSchema.P_WAVE_ID: 3,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
            ]
        )
    )


@pytest.fixture
def feature_calculator_vcg_area():
    feature_calculator = make_feature_calculator()
    feature_calculator.noise = [0.01]

    signal_1d = np.array([0.01, 0.02, -0.03, 0.05, -0.01, 0.0, 0.02, -0.02] * 12)
    signal = np.tile(signal_1d, (12, 1))

    ecg_data = MagicMock()
    ecg_data.get_sampling_frequency.return_value = 500
    ecg_data.get_ecg.side_effect = lambda preprocessor=None, **kwargs: signal.copy()

    return feature_calculator, ecg_data


@pytest.fixture
def feature_calculator_complexity():
    feature_calculator = make_feature_calculator()
    ecg_data = make_mock_ecg_data(
        {
            "I": np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
            "II": np.array([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0]),
            "III": np.zeros(10),
        }
    )
    feature_calculator.max_absolute_amplitude = MagicMock(
        return_value=np.array([2.0, 2.0, 0.0])
    )
    return feature_calculator, ecg_data


@pytest.fixture
def feature_calculator_rpeaks():
    feature_calculator = make_feature_calculator()
    ecg_data = make_mock_ecg_data(
        {
            "II": np.array([0.0, 1.0, 3.0, 2.0, 0.0, 0.0, 1.0, 4.0, 2.0, 0.0]),
            "V5": np.array([0.0, 0.5, 2.5, 2.0, 0.0, 0.0, 0.5, 3.5, 1.5, 0.0]),
            "V6": np.array([0.0, 1.0, 2.8, 2.0, 0.0, 0.0, 0.8, 4.0, 2.0, 0.0]),
        }
    )
    return feature_calculator, ecg_data


def test_ptf(feature_calculator_inflection, sample_annotations_with_inflection):
    feature_calculator, ecg_data = feature_calculator_inflection

    ptf = feature_calculator.ptf(sample_annotations_with_inflection, ecg_data)

    # Correct manual computation:
    # P-wave 1: onset=0, inflection=4 → segment = signal[4:] = [0,-1,-2,-1,0], len=5, min=-2 → 10 / 500 = 0.02
    # P-wave 2: onset=0, inflection=3 → segment = signal[3:] = [0,-1,0], len=3, min=-1 → 3 / 500 = 0.006
    expected = [0.02, 0.006, np.nan]
    np.testing.assert_almost_equal(ptf, expected)


def test_ptf_invalid_inflection(feature_calculator_inflection, sample_annotations):
    feature_calculator, ecg_data = feature_calculator_inflection
    invalid_annotations = sample_annotations.copy()
    invalid_annotations["p_wave_morphology"] = ["Biphasic", "Biphasic", "Biphasic"]
    invalid_annotations["inflection_point"] = [0, 1, 1]

    with pytest.raises(ValueError, match="Invalid inflection point"):
        feature_calculator.ptf(invalid_annotations, ecg_data)


def test_ptf_no_inflection_falls_back_to_auto(
    feature_calculator_inflection, sample_annotations
):
    # Without inflection_point / p_wave_morphology columns, ptf falls back to unsupervised mode.
    feature_calculator, ecg_data = feature_calculator_inflection
    result = feature_calculator.ptf(sample_annotations, ecg_data)
    assert len(result) == len(sample_annotations)


def test_max_amplitude(feature_calculator_no_baseline, sample_annotations):
    feature_calculator, ecg_data = feature_calculator_no_baseline
    max_amplitude = feature_calculator.max_amplitude(sample_annotations, ecg_data)
    np.testing.assert_almost_equal(max_amplitude, [2.0, 4.0, 1.0])


def test_max_amplitude_accepts_annotations_wrapper(
    feature_calculator_no_baseline,
    sample_annotations,
):
    feature_calculator, ecg_data = feature_calculator_no_baseline
    max_amplitude = feature_calculator.max_amplitude(
        sample_annotations.copy(),
        ecg_data,
    )
    np.testing.assert_almost_equal(max_amplitude, [2.0, 4.0, 1.0])


def test_min_amplitude(feature_calculator_no_baseline, sample_annotations):
    feature_calculator, ecg_data = feature_calculator_no_baseline
    min_amplitude = feature_calculator.min_amplitude(sample_annotations, ecg_data)
    np.testing.assert_almost_equal(min_amplitude, [0.0, 2.0, -2.0])


def test_max_absolute_amplitude(feature_calculator_no_baseline, sample_annotations):
    feature_calculator, ecg_data = feature_calculator_no_baseline
    max_absolute_amplitude = feature_calculator.max_absolute_amplitude(
        sample_annotations,
        ecg_data,
    )
    np.testing.assert_almost_equal(max_absolute_amplitude, [2.0, 4.0, 2.0])


def test_area(feature_calculator_no_baseline, sample_annotations):
    feature_calculator, ecg_data = feature_calculator_no_baseline
    feature_calculator.segment_processor.baseline_correction_type = "linear"

    areas = feature_calculator.area(sample_annotations, ecg_data)

    np.testing.assert_almost_equal(areas, [0.0, 0.0, 4.0 / 500.0])


def test_area_to_duration_ratio_zero_duration_raises():
    fc = make_feature_calculator()
    ecg_data = MagicMock()
    fc.area = MagicMock(return_value=np.array([1.0, 2.0]))
    fc.duration = MagicMock(return_value=np.array([0.1, 0.0]))
    annotations = Annotations(
        pd.DataFrame(
            {
                AnnotationSchema.LEAD: ["I", "I"],
                AnnotationSchema.ONSET: [0, 10],
                AnnotationSchema.OFFSET: [5, 15],
                AnnotationSchema.P_WAVE_ID: [0, 1],
                AnnotationSchema.FILE_PATH: ["rec1", "rec1"],
            }
        )
    )

    with pytest.raises(ValueError, match="zero"):
        fc.area_to_duration_ratio(annotations, ecg_data)


def test_axis_method(feature_calculator_axis, sample_annotations_axis):
    feature_calculator, ecg_data = feature_calculator_axis

    axes = feature_calculator.axis(sample_annotations_axis, ecg_data)

    assert isinstance(axes, list)
    assert len(axes) == 1

    net_i = np.sum(ecg_data.get_lead_signal("I")[50:60])
    net_avf = np.sum(ecg_data.get_lead_signal("aVF")[52:62])
    expected_angle = np.degrees(np.arctan2(2 * net_avf, np.sqrt(3) * net_i)) % 360

    np.testing.assert_almost_equal(axes[0], expected_angle)


def test_estimate_noise_basic(ecg_processor_with_noise):
    processor, ecg_data = ecg_processor_with_noise

    raw_signal = ecg_data.get_lead_signal(
        "I",
        preprocessor=processor.signal_preprocessor,
    )
    signals, _ = nk.ecg_process(
        raw_signal,
        sampling_rate=ecg_data.get_sampling_frequency(),
    )

    p_onsets = np.where(signals["ECG_P_Onsets"].to_numpy())[0]
    p_offsets = np.where(signals["ECG_P_Offsets"].to_numpy())[0]
    min_len = min(len(p_onsets), len(p_offsets))

    annotations = Annotations(
        pd.DataFrame(
            {
                AnnotationSchema.ONSET: p_onsets[:min_len].astype(int),
                AnnotationSchema.OFFSET: p_offsets[:min_len].astype(int),
                AnnotationSchema.LEAD: ["I"] * min_len,
                AnnotationSchema.P_WAVE_ID: list(range(min_len)),
                AnnotationSchema.FILE_PATH: ["rec1"] * min_len,
            }
        )
    )
    processor.max_absolute_amplitude.return_value = [np.max(raw_signal)] * min_len

    noise_estimates = processor.estimate_noise(annotations, ecg_data)

    assert isinstance(noise_estimates, (list, np.ndarray))
    assert len(noise_estimates) == 1
    assert all(0 < value < 0.1 for value in noise_estimates)


def test_estimate_noise_empty_annotations(ecg_processor_with_noise):
    processor, ecg_data = ecg_processor_with_noise

    result = processor.estimate_noise(Annotations(pd.DataFrame()), ecg_data)

    assert isinstance(result, list)
    assert len(result) == 0


def test_estimate_noise_vcg_annotations_uses_vcg_signal():
    feature_calculator = make_feature_calculator()
    annotations = Annotations(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "VCG",
                    AnnotationSchema.ONSET: 5,
                    AnnotationSchema.OFFSET: 7,
                    AnnotationSchema.P_WAVE_ID: 0,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "VCG",
                    AnnotationSchema.ONSET: 12,
                    AnnotationSchema.OFFSET: 14,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
            ]
        )
    )
    ecg_data = MagicMock()
    ecg_data.get_sampling_frequency.return_value = 1000
    ecg_data.get_leads.return_value = ["I", "II"]
    ecg_data.get_lead_signal.side_effect = AssertionError(
        "ECG lead signals should not be used for VCG-only noise estimation"
    )

    vcg_signal = np.array(
        [
            0.0,
            0.05,
            -0.05,
            0.05,
            -0.05,
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            0.03,
            -0.03,
            0.0,
            0.0,
            0.0,
        ]
    )

    noise_estimates = feature_calculator.estimate_noise(
        annotations,
        ecg_data,
        vcg_signal,
    )

    assert len(noise_estimates) == 1
    assert noise_estimates[0] > 0


def test_fragments_method(feature_calculator_fragments, sample_annotations_fragments):
    feature_calculator, ecg_data = feature_calculator_fragments

    noise_sd_multiplier = 2.0
    count, widths, amplitudes = feature_calculator.fragment_metrics(
        sample_annotations_fragments,
        ecg_data,
        noise_sd_multiplier=noise_sd_multiplier,
    )
    per_annotation_fragments = feature_calculator.fragments(
        sample_annotations_fragments,
        ecg_data,
        noise_sd_multiplier=noise_sd_multiplier,
    )

    fragments_list = per_annotation_fragments[0]
    np.testing.assert_almost_equal(count[0], len(fragments_list))

    for fragment in fragments_list:
        _, _, first_val, last_val = fragment
        assert (
            np.max(np.abs([first_val, last_val]))
            >= feature_calculator.noise[0] * noise_sd_multiplier
        )

    for width in widths:
        assert width > 0

    expected_width = np.mean(
        [
            (fragment[1] - fragment[0]) / ecg_data.get_sampling_frequency()
            for fragment in fragments_list
        ]
    )
    expected_amplitude = np.mean(
        [np.abs(fragment[3] - fragment[2]) for fragment in fragments_list]
    )

    np.testing.assert_almost_equal(widths[0], expected_width)
    np.testing.assert_almost_equal(amplitudes[0], expected_amplitude)


def test_fragments_method_normalized(
    feature_calculator_fragments, sample_annotations_fragments
):
    feature_calculator, ecg_data = feature_calculator_fragments

    count, widths, amplitudes = feature_calculator.fragment_metrics(
        sample_annotations_fragments,
        ecg_data,
        normalize_by_duration=True,
    )
    fragments_list = feature_calculator.fragments(
        sample_annotations_fragments, ecg_data
    )[0]

    duration = (
        sample_annotations_fragments.iloc[0][AnnotationSchema.OFFSET]
        - sample_annotations_fragments.iloc[0][AnnotationSchema.ONSET]
    ) / ecg_data.get_sampling_frequency()
    normalization_factor = 0.1 / duration

    np.testing.assert_almost_equal(count[0], len(fragments_list) * normalization_factor)

    expected_width = (
        np.mean(
            [
                (fragment[1] - fragment[0]) / ecg_data.get_sampling_frequency()
                for fragment in fragments_list
            ]
        )
        * normalization_factor
    )
    expected_amplitude = (
        np.mean([np.abs(fragment[3] - fragment[2]) for fragment in fragments_list])
        * normalization_factor
    )

    np.testing.assert_almost_equal(widths[0], expected_width)
    np.testing.assert_almost_equal(amplitudes[0], expected_amplitude)


def test_classify_p_wave_morphology(
    feature_calculator_morphology,
    morphology_annotations,
):
    feature_calculator, ecg_data = feature_calculator_morphology

    phase_types, inflection_points = feature_calculator.classify_p_wave_morphology(
        morphology_annotations,
        ecg_data,
    )

    assert phase_types[0] == "Monophasic Positive"
    assert inflection_points[0] == -1
    assert phase_types[1] == "Biphasic Positive-Negative"
    # inflexion_point should correspond to minimum in derivative
    # ecg[1, 0:9] = np.array([0, 1, 2, 1, 0, -1, -2, -1, 0])
    # derivative: [ 1,  1, -1, -1, -1, -1,  1,  1] -> argmin = 2, onset=0 -> 2+0+1=3
    # second derivative check: [ 0, -1,  0,  0,  0,  2,  0] -> positive at index 2
    assert inflection_points[1] == 3


def test_atrial_rate():
    feature_calculator = FeatureCalculators.__new__(FeatureCalculators)
    ecg_data = MagicMock()
    ecg_data.get_sampling_frequency.return_value = 500

    annotations = Annotations(
        pd.DataFrame(
            {
                AnnotationSchema.LEAD: ["I", "II", "I", "II"],
                AnnotationSchema.ONSET: [0, 2, 100, 102],
                AnnotationSchema.OFFSET: [5, 7, 105, 107],
                AnnotationSchema.P_WAVE_ID: [1, 1, 2, 2],
                AnnotationSchema.FILE_PATH: ["rec1", "rec1", "rec1", "rec1"],
            }
        )
    )

    heart_rate = feature_calculator.atrial_rate(annotations, ecg_data)

    assert isinstance(heart_rate, int)
    assert heart_rate == 300


def test_heart_rate(feature_calculator_heart_rate, hr_annotations):
    feature_calculator, ecg_data = feature_calculator_heart_rate

    heart_rate = feature_calculator.heart_rate(hr_annotations, ecg_data)

    assert isinstance(heart_rate, int)
    assert heart_rate == 300


def test_vcg_fragment_metrics_normalize_by_duration(monkeypatch):
    raw_counts = [2.0, 3.0]
    raw_widths = [0.05, 0.04]
    raw_amplitudes = [0.1, 0.2]
    durations = np.array([0.2, 0.4])
    expected_norm_factors = 0.1 / durations

    fc = make_feature_calculator()
    ecg_data = MagicMock()
    ecg_data.get_sampling_frequency.return_value = 500

    fc._get_vcg_signal = MagicMock(return_value=np.ones((3, 50)))
    fc._vcg_fragments_from_signal = MagicMock(return_value=[[], []])
    fc.duration = MagicMock(return_value=durations)
    monkeypatch.setattr(
        "atriakit.feature_calculator.fragment_count", lambda frags: raw_counts
    )
    monkeypatch.setattr(
        "atriakit.feature_calculator.fragment_width", lambda frags, fs: raw_widths
    )
    monkeypatch.setattr(
        "atriakit.feature_calculator.fragment_amplitude", lambda frags: raw_amplitudes
    )

    annotations = Annotations(
        pd.DataFrame(
            {
                AnnotationSchema.LEAD: ["VCG", "VCG"],
                AnnotationSchema.ONSET: [0, 10],
                AnnotationSchema.OFFSET: [5, 15],
                AnnotationSchema.P_WAVE_ID: [0, 1],
                AnnotationSchema.FILE_PATH: ["rec1", "rec2"],
            }
        )
    )
    counts, widths, amplitudes = fc.vcg_fragments(
        annotations, ecg_data, normalize_by_duration=True
    )

    np.testing.assert_allclose(counts, np.array(raw_counts) * expected_norm_factors)
    np.testing.assert_allclose(widths, np.array(raw_widths) * expected_norm_factors)
    np.testing.assert_allclose(
        amplitudes, np.array(raw_amplitudes) * expected_norm_factors
    )


def test_vcg_area_calls_ecgdata(feature_calculator_vcg_area):
    feature_calculator, ecg_data = feature_calculator_vcg_area

    annotations = AnnotationsLoader().from_dataframe(
        pd.DataFrame(
            {
                AnnotationSchema.FILE_PATH: ["rec1", "rec1", "rec1"],
                AnnotationSchema.LEAD: ["I", "II", "III"],
                AnnotationSchema.ONSET: [1, 2, 0],
                AnnotationSchema.OFFSET: [3, 4, 2],
                AnnotationSchema.P_WAVE_ID: [1, 1, 1],
                AnnotationSchema.TYPE: ["Before", "Before", "Before"],
            }
        )
    )

    fake_vcg_segment = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    feature_calculator._get_vcg_signal = MagicMock(return_value=fake_vcg_segment)

    area = feature_calculator.vcg_area(annotations, ecg_data)

    # multilead onset=min(1,2,0)=0, offset=max(3,4,2)=4 → 5 samples [0..4]
    expected_area = np.array([0.0, 1.0, 2.0, 3.0, 4.0]).sum() / 500
    assert isinstance(area, list)
    assert len(area) == 1
    np.testing.assert_almost_equal(area, expected_area)


def test_vcg_eigenfeatures_with_grouped_onsets_offsets():
    feature_calculator = make_feature_calculator(
        onset_feature="onset_g",
        offset_feature="offset_g",
    )
    ecg_data = MagicMock()
    ecg_data.get_sampling_frequency.return_value = 500

    vcg_signal = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    feature_calculator._get_vcg_signal = MagicMock(return_value=vcg_signal)

    annotations = AnnotationsLoader().from_dataframe(
        pd.DataFrame(
            {
                AnnotationSchema.FILE_PATH: ["rec1", "rec1", "rec1"],
                AnnotationSchema.LEAD: ["I", "II", "III"],
                AnnotationSchema.ONSET: [1, 2, 0],
                AnnotationSchema.OFFSET: [3, 4, 2],
                AnnotationSchema.QRS_ONSET: [4, 5, 3],
                "onset_g": [0, 0, 0],
                "offset_g": [4, 4, 4],
                AnnotationSchema.P_WAVE_ID: [1, 1, 1],
                AnnotationSchema.TYPE: ["Before", "Before", "Before"],
            }
        )
    )

    segment = vcg_signal[
        :, annotations.iloc[0]["onset_g"] : annotations.iloc[0]["offset_g"] + 1
    ]
    cov_matrix = np.cov(segment)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    expected_roundness = sorted_eigenvalues[1] / (sorted_eigenvalues[0] + 1e-10)
    expected_flatness = sorted_eigenvalues[2] / (
        sorted_eigenvalues[0] + sorted_eigenvalues[1] + 1e-10
    )

    eigenvalues_result, roundness, flatness, p_wave_ids = (
        feature_calculator.vcg_eigenfeatures(annotations, ecg_data)
    )

    eigenvalues_result = np.asarray(eigenvalues_result)
    roundness = np.asarray(roundness)
    flatness = np.asarray(flatness)
    p_wave_ids = np.asarray(p_wave_ids)

    assert eigenvalues_result.shape == (1, 3)
    assert roundness.shape == (1,)
    assert flatness.shape == (1,)
    assert p_wave_ids.shape == (1,)

    np.testing.assert_almost_equal(eigenvalues_result[0], sorted_eigenvalues)
    np.testing.assert_almost_equal(roundness[0], expected_roundness)
    np.testing.assert_almost_equal(flatness[0], expected_flatness)


# ── vcg_axis_angles ───────────────────────────────────────────────────────────


def _seg(x, y, z):
    """Make a (3, 1) segment with a single time step."""
    return np.array([[x], [y], [z]], dtype=float)


def test_vcg_axis_angles_pure_x():
    # Net dipole along +X: elevation=0, azimuth=0
    elev, azim = vcg_axis_angles(_seg(1, 0, 0))
    assert elev == pytest.approx(0.0)
    assert azim == pytest.approx(0.0)


def test_vcg_axis_angles_pure_y():
    # Net dipole along +Y (inferior): elevation=0, azimuth=90
    elev, azim = vcg_axis_angles(_seg(0, 1, 0))
    assert elev == pytest.approx(0.0)
    assert azim == pytest.approx(90.0)


def test_vcg_axis_angles_pure_z():
    # Net dipole along +Z (anterior): elevation=90, azimuth arbitrary
    elev, _ = vcg_axis_angles(_seg(0, 0, 1))
    assert elev == pytest.approx(90.0)


def test_vcg_axis_angles_45_elevation():
    # Equal X and Z components → elevation = 45°
    elev, azim = vcg_axis_angles(_seg(1, 0, 1))
    assert elev == pytest.approx(45.0)
    assert azim == pytest.approx(0.0)


def test_vcg_axis_angles_45_azimuth():
    # Equal X and Y components → azimuth = 45°, elevation = 0°
    elev, azim = vcg_axis_angles(_seg(1, 1, 0))
    assert elev == pytest.approx(0.0)
    assert azim == pytest.approx(45.0)


def test_vcg_axis_angles_negative_x():
    # Pointing left (−X) → azimuth = ±180°
    elev, azim = vcg_axis_angles(_seg(-1, 0, 0))
    assert elev == pytest.approx(0.0)
    assert abs(azim) == pytest.approx(180.0)


def test_vcg_axis_angles_multi_timestep_sums_correctly():
    # 4 identical time steps of [1, 0, 0] — net = [4, 0, 0] → same angles as [1, 0, 0]
    seg = np.tile(_seg(1, 0, 0), (1, 4))
    elev, azim = vcg_axis_angles(seg)
    assert elev == pytest.approx(0.0)
    assert azim == pytest.approx(0.0)


def test_get_onset_offset_angle_onset_baseline_and_value():
    feature_calculator = make_feature_calculator(baseline_correction_type="onset")
    ecg_data = make_mock_ecg_data({"I": np.array([5.0, 7.0])}, sampling_frequency=1000)

    annotations = Annotations(
        pd.DataFrame(
            {
                AnnotationSchema.LEAD: ["I"],
                AnnotationSchema.ONSET: [0],
                AnnotationSchema.OFFSET: [1],
                AnnotationSchema.P_WAVE_ID: [1],
                AnnotationSchema.FILE_PATH: ["rec1"],
                AnnotationSchema.TYPE: ["Before"],
            }
        )
    )

    angles = feature_calculator.get_onset_offset_angle(annotations, ecg_data)

    assert len(angles) == 1
    np.testing.assert_allclose(
        angles[0],
        np.pi / 4,
        atol=1e-6,
        err_msg="Onset-offset angle should be 45 degrees (pi/4) after onset correction",
    )


def test_complexity(feature_calculator_complexity):
    feature_calculator, ecg_data = feature_calculator_complexity
    annotations = Annotations(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "I",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 4,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "II",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 8,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "III",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 9,
                    AnnotationSchema.P_WAVE_ID: 3,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
            ]
        )
    )

    complexity = feature_calculator.complexity(annotations, ecg_data)

    np.testing.assert_array_equal(complexity, [1, 2, 0])


def test_r_peaks_for_lead(feature_calculator_rpeaks):
    feature_calculator, ecg_data = feature_calculator_rpeaks
    lead_signal = ecg_data.get_lead_signal("II")
    lead_annotations = pd.DataFrame(
        [
            {
                AnnotationSchema.LEAD: "II",
                AnnotationSchema.ONSET: 0,
                AnnotationSchema.OFFSET: 2,
                AnnotationSchema.ONSET_ORIGINAL: 0,
                AnnotationSchema.OFFSET_ORIGINAL: 2,
                AnnotationSchema.P_WAVE_ID: 1,
                AnnotationSchema.FILE_PATH: "rec1",
            },
            {
                AnnotationSchema.LEAD: "II",
                AnnotationSchema.ONSET: 5,
                AnnotationSchema.OFFSET: 6,
                AnnotationSchema.ONSET_ORIGINAL: 5,
                AnnotationSchema.OFFSET_ORIGINAL: 6,
                AnnotationSchema.P_WAVE_ID: 2,
                AnnotationSchema.FILE_PATH: "rec1",
            },
        ]
    )

    r_peaks = feature_calculator._r_peaks_for_lead(
        lead_signal,
        lead_annotations,
        window_samples=5,
    )

    assert r_peaks == [4, 7]


def test_find_r_peaks_median(feature_calculator_rpeaks):
    feature_calculator, ecg_data = feature_calculator_rpeaks
    annotations = AnnotationsLoader().from_dataframe(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "II",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 2,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "II",
                    AnnotationSchema.ONSET: 5,
                    AnnotationSchema.OFFSET: 6,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V5",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 2,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V5",
                    AnnotationSchema.ONSET: 5,
                    AnnotationSchema.OFFSET: 6,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V6",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 2,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "V6",
                    AnnotationSchema.ONSET: 5,
                    AnnotationSchema.OFFSET: 6,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
            ]
        )
    )

    r_peaks_median = feature_calculator.find_r_peaks_median(
        annotations,
        ecg_data,
        leads=["II", "V5", "V6"],
        max_r_peak_window_ms=10,
    )

    np.testing.assert_array_equal(r_peaks_median, np.array([4, 7]))


@pytest.mark.parametrize(
    ("lead_signal", "n_bins", "expected_entropy"),
    [
        (np.ones(10), 8, 0.0),
        (np.array([0.0, 1.0, 2.0, 3.0]), 4, 2.0),
    ],
)
def test_get_shannon_entropy(lead_signal, n_bins, expected_entropy):
    feature_calculator = make_feature_calculator()
    ecg_data = make_mock_ecg_data({"I": lead_signal})

    annotations = Annotations(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "I",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: len(lead_signal),
                    AnnotationSchema.P_WAVE_ID: 0,
                    AnnotationSchema.FILE_PATH: "rec1",
                }
            ]
        )
    )

    entropy = feature_calculator.get_shannon_entropy(
        annotations,
        ecg_data,
        n_bins=n_bins,
    )

    assert len(entropy) == 1
    np.testing.assert_allclose(
        entropy[0],
        expected_entropy,
        atol=1e-6,
        err_msg="Shannon entropy value is incorrect",
    )


def test_get_sample_entropy_basic():
    feature_calculator = make_feature_calculator()
    annotations = Annotations(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "I",
                    AnnotationSchema.P_WAVE_ID: 0,
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 5,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "I",
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 5,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
            ]
        )
    )
    segments = [
        np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        np.ones(5),
    ]

    ecg_data = MagicMock()
    ecg_data.get_lead_signal.side_effect = (
        lambda lead, preprocessor=None, **kwargs: segments.pop(-1)
    )
    ecg_data.get_sampling_frequency.return_value = 500

    entropies = feature_calculator.get_sample_entropy(
        annotations,
        ecg_data,
        m=2,
        r_factor=0.95,
    )

    np.testing.assert_allclose(entropies[0], 0.0, atol=1e-8)
    assert np.isfinite(entropies[1])


def test_global_baseline_computed_once_for_nested_calls():
    # complexity calls max_absolute_amplitude internally; both are wrapped by
    # @_auto_wrap_baseline.  The outer entry (complexity) must set the baseline
    # onsets and the inner call must detect they are already set (reentrant guard).
    feature_calculator = make_feature_calculator()

    baseline_values_seen = []

    original_build = feature_calculator.signal_preprocessor.set_baseline_onsets

    def tracking_build(annotations):
        result = original_build(annotations)
        baseline_values_seen.append(result)
        return result

    feature_calculator.signal_preprocessor.set_baseline_onsets = tracking_build

    annotations = Annotations(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: "I",
                    AnnotationSchema.ONSET: 0,
                    AnnotationSchema.OFFSET: 5,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
                {
                    AnnotationSchema.LEAD: "I",
                    AnnotationSchema.ONSET: 10,
                    AnnotationSchema.OFFSET: 15,
                    AnnotationSchema.P_WAVE_ID: 2,
                    AnnotationSchema.FILE_PATH: "rec1",
                },
            ]
        )
    )
    signal = np.array(
        [0.0, 0.1, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.2, 0.0, 0.0]
    )
    ecg_data = MagicMock()
    ecg_data.get_lead_signal.return_value = signal.copy()
    ecg_data.get_sampling_frequency.return_value = 500
    ecg_data.get_leads.return_value = ["I"]

    feature_calculator.complexity(annotations, ecg_data)

    assert len(baseline_values_seen) == 1, (
        f"_build_global_baseline_onsets called {len(baseline_values_seen)} times; "
        "expected 1 (reentrancy should suppress the nested call from max_absolute_amplitude)"
    )


# ── ptf_auto ─────────────────────────────────────────────────────────
#
# Reference signal (7 samples, fs=7 → duration = 1 s):
#   signal = [0, 1, 0, -1, -2, -1, 0]
#   d1     = [1, 0, -1, -1, 0, 1, 1]   (np.gradient)
#   d2     = [-1, -1, -0.5, 0.5, 1, 0.5, 0]
#   d2 sign-changes at indices 2 and 5
#   d1 at those indices: -1 and 1 → steepest descent at 2
#   inflection = 2, terminal = signal[2:] = [0,-1,-2,-1,0] (5 samples), min = -2
#   ptf_deriv = |-2| * 5/7 = 10/7
_REF_SIGNAL = np.array([0.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0])
_FS = 7  # len == fs → duration factor == 1 for easy hand-checking


def _ann(lead="I", onset=0, offset=None):
    offset = offset if offset is not None else len(_REF_SIGNAL)
    return Annotations(
        pd.DataFrame(
            [
                {
                    AnnotationSchema.LEAD: lead,
                    AnnotationSchema.ONSET: onset,
                    AnnotationSchema.OFFSET: offset,
                    AnnotationSchema.P_WAVE_ID: 1,
                    AnnotationSchema.FILE_PATH: "rec1",
                }
            ]
        )
    )


def _make_dual_mock(signal, morph_signal, lead="I", fs=_FS):
    """ECGData mock that returns different arrays for signal vs morph preprocessor."""
    fc = make_feature_calculator()
    ecg_data = MagicMock()

    def get_lead_signal(ld, preprocessor=None, **kwargs):
        if preprocessor is fc.morphology_preprocessor:
            return morph_signal.copy()
        return signal.copy()

    ecg_data.get_lead_signal.side_effect = get_lead_signal
    ecg_data.get_sampling_frequency.return_value = fs
    return fc, ecg_data


class TestFindInflection:
    def test_biphasic_picks_steepest_descent(self):
        # _REF_SIGNAL: inflection should land at index 2 (see header comment)
        assert _find_inflection(_REF_SIGNAL) == 2

    def test_avr_picks_steepest_ascent(self):
        # Mirror of _REF_SIGNAL: inflection still at 2, but via argmax(d1)
        avr_signal = -_REF_SIGNAL
        assert _find_inflection(avr_signal, lead="aVR") == 2

    def test_no_d2_zero_crossing_falls_back_to_argmin_d1(self):
        # Constant d1 → d2 = 0 everywhere → fallback argmin(d1) = 0
        flat_ramp = np.array([-1.0, -2.0, -3.0, -4.0])
        assert _find_inflection(flat_ramp) == 0


class TestPtfUnsupervisedFn:
    def test_empty_segment_raises(self):
        with pytest.raises((ValueError, IndexError)):
            ptf_auto_fn(np.array([]), fs=_FS)

    def test_basic_formula(self):
        # inflection at 2, terminal = 5 samples, min = -2 → 2 * 5/7 = 10/7
        result = ptf_auto_fn(_REF_SIGNAL, fs=_FS)
        np.testing.assert_allclose(result, 10 / 7)

    def test_avr_uses_max_of_terminal(self):
        # inverted signal: terminal positive, max = 2, terminal = 5 samples → 2 * 5/7 = 10/7
        avr_signal = -_REF_SIGNAL
        result = ptf_auto_fn(avr_signal, fs=_FS, lead="aVR")
        np.testing.assert_allclose(result, 10 / 7)

    def test_seg_morph_controls_inflection(self):
        # without morph: inflection at 2, terminal = 5 samples, min = -2 → 10/7
        # with morph:    inflection at 5, terminal = 2 samples, min = -1 → 2/7
        morph = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0])
        without_morph = ptf_auto_fn(_REF_SIGNAL, fs=_FS)
        with_morph = ptf_auto_fn(_REF_SIGNAL, fs=_FS, seg_morph=morph)
        np.testing.assert_allclose(without_morph, 10 / 7)
        np.testing.assert_allclose(with_morph, 2 / 7)

    def test_amplitude_always_from_original_segment(self):
        # morph shifts inflection to 5; original[5:]= [-5, 0] → min=-5 → 5*2/7 = 10/7
        # if amplitude came from morph: morph[5:]=[0,-1] → min=-1 → 1*2/7 = 2/7
        morph = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0])
        original = _REF_SIGNAL.copy()
        original[5] = -5.0
        result = ptf_auto_fn(original, fs=_FS, seg_morph=morph)
        np.testing.assert_allclose(result, 10 / 7)

    def test_sampling_frequency_scales_result(self):
        # ptf = |peak| * len/fs; doubling fs halves the result
        r1 = ptf_auto_fn(_REF_SIGNAL, fs=7)
        r2 = ptf_auto_fn(_REF_SIGNAL, fs=14)
        np.testing.assert_allclose(r1, 2 * r2)

    def test_monophasic_wave_ptf(self):
        # signal = [0, 0.5, 1, 0.5, 0] — symmetric positive monophasic
        # d1 = [0.5, 0.5, 0, -0.5, -0.5], d2 = [0, -0.25, -0.5, -0.25, 0]
        # no sign changes in d2 → fallback argmin(d1) = 3
        # terminal = signal[3:] = [0.5, 0] → max(abs) = 0.5 → PTF = 2/5 * 0.5 = 0.2
        mono = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        result = ptf_auto_fn(mono, fs=5)
        assert result == pytest.approx(0.2)


class TestPtfUnsupervisedFeatureCalculator:
    def test_uses_morph_preprocessor_for_inflection(self):
        # morph forces inflection at 5 → terminal = 2 samples, min = -1 → 2/7
        # raw signal would give inflection at 2 → 10/7
        morph = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0])
        fc, ecg_data = _make_dual_mock(_REF_SIGNAL, morph)
        result = fc.ptf(_ann(), ecg_data)
        np.testing.assert_allclose(result, [2 / 7])

    def test_empty_annotations_returns_empty(self):
        fc, ecg_data = _make_dual_mock(_REF_SIGNAL, _REF_SIGNAL)
        result = fc.ptf(Annotations(pd.DataFrame()), ecg_data)
        assert len(result) == 0

    def test_multiple_annotations(self):
        # Two beats on lead I concatenated. _compute_onset_offset adds +1 to offset,
        # so annotation offset must be (end - 1) to extract an exact segment.
        # Beat 1: signal[0:7]  → onset=0, offset=6  (6+1=7)
        # Beat 2: signal[7:14] → onset=7, offset=13 (13+1=14)
        signal = np.concatenate([_REF_SIGNAL, _REF_SIGNAL])
        morph = np.concatenate([_REF_SIGNAL, _REF_SIGNAL])
        fc, ecg_data = _make_dual_mock(signal, morph, fs=_FS)
        ann = Annotations(
            pd.DataFrame(
                [
                    {
                        AnnotationSchema.LEAD: "I",
                        AnnotationSchema.ONSET: 0,
                        AnnotationSchema.OFFSET: 6,
                        AnnotationSchema.P_WAVE_ID: 1,
                        AnnotationSchema.FILE_PATH: "rec1",
                    },
                    {
                        AnnotationSchema.LEAD: "I",
                        AnnotationSchema.ONSET: 7,
                        AnnotationSchema.OFFSET: 13,
                        AnnotationSchema.P_WAVE_ID: 2,
                        AnnotationSchema.FILE_PATH: "rec1",
                    },
                ]
            )
        )
        result = fc.ptf(ann, ecg_data)
        assert len(result) == 2
        np.testing.assert_allclose(result[0], result[1])
