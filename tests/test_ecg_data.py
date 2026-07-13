from unittest.mock import MagicMock

import numpy as np
import pytest

from atriakit.annotations import Annotations
from atriakit.configs.signal_preprocessor_config import SignalPreprocessorConfig
from atriakit.models.ecg_data import ECGData
from atriakit.preprocessing.signals import SignalPreprocessor
from atriakit.io import AnnotationsLoader

_STD_12 = {"I":0,"II":1,"III":2,"aVR":3,"aVL":4,"aVF":5,"V1":6,"V2":7,"V3":8,"V4":9,"V5":10,"V6":11}


@pytest.fixture
def ecg_instance(monkeypatch):
    ecg = ECGData(ecg=np.ones((12, 100)), fs=500, lead_to_index=_STD_12)
    monkeypatch.setattr(
        "atriakit.preprocessing.signals.apply_notch_filter",
        lambda ecg, sampling_frequency, notch_freq=50.0, quality_factor=30.0: ecg + 1,
    )
    return ecg


def test_get_ecg_cache(ecg_instance):
    """Calling get_ecg twice with the same parameters should reuse the cached result."""
    result1 = ecg_instance.get_ecg()
    result2 = ecg_instance.get_ecg()

    assert len(ecg_instance._ecg_cache) == 1
    assert np.allclose(result1, result2)


def test_get_ecg_returns_copy(ecg_instance):
    """Returned array should not modify the cached array."""
    result = ecg_instance.get_ecg()
    result[0, 0] = 999

    cached = list(ecg_instance._ecg_cache.values())[0]
    assert cached[0, 0] != 999


def test_cache_different_norm_params(monkeypatch, ecg_instance):
    """Different normalization parameters should produce different cached results."""

    def fake_normalize(x, mean, std):
        return (x - mean) / std

    monkeypatch.setattr("atriakit.preprocessing.signals.normalize", fake_normalize)
    preprocessor = SignalPreprocessor(
        SignalPreprocessorConfig(normalization_type="zscore", mean=1, std=1),
    )
    r0 = ecg_instance.get_ecg(preprocessor=preprocessor)
    r1 = ecg_instance.get_ecg(preprocessor=preprocessor.with_overrides(mean=1, std=1))
    r2 = ecg_instance.get_ecg(preprocessor=preprocessor.with_overrides(mean=5, std=5))

    assert not np.allclose(r1, r2)
    assert np.allclose(r0, r1)


def test_cache_different_global_baseline_onsets_create_distinct_entries():
    """Different onset states on the preprocessor produce different cache entries."""
    ecg = ECGData(ecg=np.ones((12, 100)), fs=500, lead_to_index=_STD_12)

    def _make_ann(onset):
        return AnnotationsLoader().from_dataframe(
            __import__("pandas").DataFrame([{
                "lead": "I", "onset": onset, "offset": onset + 10,
                "p_wave_id": 1, "file_path": "rec1",
            }])
        )

    preprocessor = SignalPreprocessor(
        SignalPreprocessorConfig(
            lowcut=None, highcut=None, notch_freq=None,
            global_baseline_mode="p_onset_spline",
        )
    )

    preprocessor.set_baseline_onsets(_make_ann(10))
    r1 = ecg.get_ecg(preprocessor=preprocessor)

    preprocessor.set_baseline_onsets(_make_ann(30))
    r2 = ecg.get_ecg(preprocessor=preprocessor)

    assert len(ecg._ecg_cache) == 2


def test_get_lead_signal_invalid_lead_raises(ecg_instance):
    with pytest.raises(ValueError, match="Unknown lead"):
        ecg_instance.get_lead_signal("INVALID")


def test_get_ecg_delegates_to_custom_preprocessor(ecg_instance):
    class CustomPreprocessor:
        def cache_key(self):
            return ("custom",)

        def preprocess(self, ecg, sampling_frequency, lead_to_index=None):
            assert sampling_frequency == 500
            return np.full_like(ecg, 7.0)

    result = ecg_instance.get_ecg(preprocessor=CustomPreprocessor())

    assert np.allclose(result, np.full((12, 100), 7.0))
