import numpy as np
import pandas as pd
import pytest

from atriakit.annotations import Annotations
from atriakit.configs.signal_preprocessor_config import SignalPreprocessorConfig
from atriakit.io import AnnotationsLoader
from atriakit.preprocessing.signals import SignalPreprocessor


@pytest.fixture
def raw_ecg(monkeypatch):
    ecg = np.ones((12, 100))
    monkeypatch.setattr(
        "atriakit.preprocessing.signals.apply_notch_filter",
        lambda ecg, sampling_frequency, notch_freq=50.0, quality_factor=30.0: ecg + 1,
    )
    monkeypatch.setattr(
        "atriakit.preprocessing.signals.bandpass_signal",
        lambda ecg, sampling_frequency, lowcut, highcut, order=2: ecg * 2,
    )
    return ecg


def test_preprocess_no_filter(raw_ecg):
    """
    If no bandpass or normalization is provided,
    only the notch filter should run.
    """
    preprocessor = SignalPreprocessor()

    result = preprocessor.preprocess(raw_ecg, 500)

    expected = np.ones((12, 100)) + 1
    assert np.allclose(result, expected)


def test_preprocess_bandpass(raw_ecg):
    """
    Bandpass should run after notch filter.
    """
    preprocessor = SignalPreprocessor(
        SignalPreprocessorConfig(lowcut=0.5, highcut=40)
    )

    result = preprocessor.preprocess(raw_ecg, 500)

    expected = (np.ones((12, 100)) + 1) * 2
    assert np.allclose(result, expected)


def test_preprocess_normalization(monkeypatch, raw_ecg):
    """
    Normalization should be applied when mean/std are also provided.
    """

    def fake_normalize(x, mean, std):
        return (x - mean) / std

    monkeypatch.setattr("atriakit.preprocessing.signals.normalize", fake_normalize)
    preprocessor = SignalPreprocessor(
        SignalPreprocessorConfig(
            lowcut=0.5, highcut=40, normalization_type="zscore", mean=2, std=2
        ),
    )

    result = preprocessor.preprocess(raw_ecg, 500)

    expected = ((np.ones((12, 100)) + 1) * 2 - 2) / 2
    assert np.allclose(result, expected)


def test_preprocess_class_level_normalization(monkeypatch, raw_ecg):
    """
    Class-level mean/std should be used if parameters are not passed.
    """

    def fake_normalize(x, mean, std):
        return (x - mean) / std

    monkeypatch.setattr("atriakit.preprocessing.signals.normalize", fake_normalize)
    preprocessor = SignalPreprocessor(
        SignalPreprocessorConfig(normalization_type="zscore", mean=2, std=2),
    )

    result = preprocessor.preprocess(raw_ecg, 500)

    expected = ((np.ones((12, 100)) + 1) - 2) / 2
    assert np.allclose(result, expected)


def test_preprocess_skips_normalization_when_disabled(monkeypatch, raw_ecg):
    monkeypatch.setattr(
        "atriakit.preprocessing.signals.normalize",
        lambda *args, **kwargs: pytest.fail(
            "Normalization should not be called when disabled."
        ),
    )
    preprocessor = SignalPreprocessor(
        SignalPreprocessorConfig(
            lowcut=0.5,
            highcut=40,
            normalization_type="none",
            mean=2,
            std=2,
        ),
    )

    result = preprocessor.preprocess(raw_ecg, 500)

    expected = (np.ones((12, 100)) + 1) * 2
    assert np.allclose(result, expected)


def test_preprocess_uses_custom_notch_frequency(raw_ecg):
    calls = []

    def fake_notch(x, sampling_frequency, notch_freq=50.0, quality_factor=30.0):
        calls.append((notch_freq, quality_factor))
        return x + 1

    preprocessor = SignalPreprocessor(SignalPreprocessorConfig(notch_freq=[60.0]))
    preprocessor.config.quality_factor = 30.0  # Set quality factor for testing
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("atriakit.preprocessing.signals.apply_notch_filter", fake_notch)

    preprocessor.preprocess(raw_ecg, 500)
    monkeypatch.undo()

    assert calls == [(60.0, 30.0)]


def test_preprocess_skips_notch_filter_when_disabled(monkeypatch, raw_ecg):
    preprocessor = SignalPreprocessor(SignalPreprocessorConfig(notch_freq=None))
    monkeypatch.setattr(
        "atriakit.preprocessing.signals.apply_notch_filter",
        lambda *args, **kwargs: pytest.fail(
            "Notch filter should not be called when disabled."
        ),
    )

    result = preprocessor.preprocess(raw_ecg, 500)

    assert np.allclose(result, np.ones((12, 100)))


@pytest.mark.parametrize("lowcut,highcut,expected_btype", [
    (None, 40, "low"),
    (0.5, None, "high"),
])
def test_preprocess_single_cutoff_uses_correct_filter_type(monkeypatch, raw_ecg, lowcut, highcut, expected_btype):
    sampling_frequency = 500
    cutoff = (highcut or lowcut) / (sampling_frequency / 2)

    calls = []
    monkeypatch.setattr(
        "atriakit.preprocessing.signals.butter",
        lambda order, c, btype="bandpass": calls.append((btype, c)) or (None, None),
    )
    monkeypatch.setattr("atriakit.preprocessing.signals.filtfilt", lambda b, a, x: x)
    preprocessor = SignalPreprocessor(SignalPreprocessorConfig(lowcut=lowcut, highcut=highcut, notch_freq=None))
    preprocessor.preprocess(raw_ecg, sampling_frequency)
    assert calls == [(expected_btype, cutoff)]


def _spline_preprocessor():
    return SignalPreprocessor(SignalPreprocessorConfig(
        lowcut=None, highcut=None, notch_freq=None,
        global_baseline_mode="p_onset_spline",
        global_baseline_smoothing_factor=0.1,
    ))


def _ann(leads, onsets, offsets):
    return AnnotationsLoader().from_dataframe(pd.DataFrame({
        "file_path": ["f"] * len(leads),
        "lead": leads,
        "onset": onsets,
        "offset": offsets,
        "p_wave_id": list(range(len(leads))),
    }))


def test_extract_onsets_returns_none_when_baseline_disabled():
    preprocessor = SignalPreprocessor(SignalPreprocessorConfig(global_baseline_mode="none"))
    ann = _ann(["I"], [10], [20])
    assert preprocessor.set_baseline_onsets(ann) is None


def test_extract_onsets_returns_none_for_empty_annotations():
    ann = Annotations(pd.DataFrame(columns=["file_path", "lead", "onset", "offset", "p_wave_id"]))
    assert _spline_preprocessor().set_baseline_onsets(ann) is None


def test_extract_onsets_groups_by_lead():
    ann = _ann(["I", "II"], [10, 20], [15, 25])
    result = _spline_preprocessor().set_baseline_onsets(ann)
    assert result is not None
    assert set(result.keys()) == {"I", "II"}
    np.testing.assert_array_equal(result["I"], [10])
    np.testing.assert_array_equal(result["II"], [20])


def test_extract_onsets_excludes_vcg_lead():
    ann = _ann(["I", "VCG"], [10, 5], [15, 10])
    result = _spline_preprocessor().set_baseline_onsets(ann)
    assert result is not None
    assert "VCG" not in result
    assert "I" in result


def test_extract_onsets_excludes_negative_onsets():
    ann = _ann(["I", "I"], [-1, 10], [5, 15])
    result = _spline_preprocessor().set_baseline_onsets(ann)
    assert result is not None
    np.testing.assert_array_equal(result["I"], [10])


def test_extract_onsets_deduplicates():
    ann = _ann(["I", "I"], [10, 10], [15, 15])
    result = _spline_preprocessor().set_baseline_onsets(ann)
    assert result is not None
    np.testing.assert_array_equal(result["I"], [10])


def test_preprocess_global_p_onset_spline_baseline_correction():
    signal = np.linspace(0.0, 1.0, 100)
    signal[11] += 1.0
    signal[41] += 1.0
    signal[71] += 1.0

    ecg = np.zeros((12, 100))
    ecg[0] = signal

    preprocessor = SignalPreprocessor(
        SignalPreprocessorConfig(
            lowcut=None,
            highcut=None,
            notch_freq=None,
            global_baseline_mode="p_onset_spline",
            global_baseline_smoothing_factor=0.1,
        )
    )

    onsets = np.array([10, 40, 70])
    lead_to_index = {"I": 0}
    ann = _ann(["I", "I", "I"], list(onsets), list(onsets + 5))
    preprocessor.set_baseline_onsets(ann)
    corrected = preprocessor.preprocess(ecg, 500, lead_to_index=lead_to_index)

    np.testing.assert_allclose(corrected[0, onsets], 0.0, atol=1e-6)
    np.testing.assert_allclose(corrected[0, [11, 41, 71]], 1.0, atol=5e-2)
