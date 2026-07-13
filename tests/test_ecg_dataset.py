import numpy as np
import pandas as pd

from atriakit.annotations import Annotations
from atriakit.configs.signal_preprocessor_config import default_signal_preprocessor_config
from atriakit.io import AnnotationsLoader
from atriakit.processing.dataset import ECGDataset
from atriakit.models.ecg_data import ECGData
from atriakit.preprocessing.signals import SignalPreprocessor

_STD_12 = {"I":0,"II":1,"III":2,"aVR":3,"aVL":4,"aVF":5,"V1":6,"V2":7,"V3":8,"V4":9,"V5":10,"V6":11}


class _MockECGData(ECGData):
    """ECGData that returns deterministic arrays without real preprocessing."""

    def __init__(self):
        # Bypass the real constructor — supply raw fields directly.
        self.ecg = np.ones((12, 1000)) * 2.0
        self.sampling_frequency = 500
        self.content_date = None
        self.content_start_time = None
        self._ecg_cache: dict = {}
        self._filtered = np.ones((12, 1000)) * 3.0
        self._lead_to_index = _STD_12

    def get_ecg(self, preprocessor=None):
        return self._filtered if preprocessor is not None else self.ecg


class _MockLoader:
    """Loader that returns _MockECGData without touching the filesystem."""

    def load(self, path) -> ECGData:
        return _MockECGData()


def _ann(df: pd.DataFrame) -> Annotations:
    return AnnotationsLoader().from_dataframe(df)


# ── calculate_mean_std ────────────────────────────────────────────────────────

def test_calculate_mean_std_uses_configured_preprocessor():
    ann = _ann(pd.DataFrame({
        "file_path": ["a.dcm", "a.dcm"],
        "lead": ["I", "I"],
        "p_wave_id": [1, 2],
        "onset": [200, 400],
        "offset": [250, 450],
        "qrs_onset": [210, 410],
        "type": ["After", "Before"],
    }))

    processor = ECGDataset(dataset_dir="dummy", loader=_MockLoader())

    mean, std = processor.calculate_mean_std(
        ann, preprocessor=SignalPreprocessor(default_signal_preprocessor_config())
    )

    assert np.allclose(mean, np.full(12, 3.0))
    assert np.allclose(std, np.zeros(12))


def test_calculate_mean_std_with_bandpass():
    ann = _ann(pd.DataFrame({
        "file_path": ["b.dcm"],
        "lead": ["I"],
        "p_wave_id": [1],
        "onset": [300],
        "offset": [350],
        "qrs_onset": [310],
        "type": ["After"],
    }))

    processor = ECGDataset(dataset_dir="dummy", loader=_MockLoader())

    mean, std = processor.calculate_mean_std(
        ann,
        preprocessor=SignalPreprocessor(default_signal_preprocessor_config()).with_overrides(
            lowcut=0.5, highcut=40
        ),
    )

    assert np.allclose(mean, np.full(12, 3.0))
    assert np.allclose(std, np.zeros(12))


# ── calculate_mean_std_p_waves ────────────────────────────────────────────────

def test_calculate_mean_std_p_waves():
    ann = _ann(pd.DataFrame({
        "file_path": ["a.dcm", "a.dcm"],
        "lead": ["I", "I"],
        "p_wave_id": [1, 2],
        "onset": [200, 400],
        "offset": [250, 450],
        "qrs_onset": [210, 410],
        "type": ["After", "Before"],
    }))

    processor = ECGDataset(dataset_dir="dummy", loader=_MockLoader())

    mean, std = processor.calculate_mean_std_p_waves(
        ann,
        preprocessor=SignalPreprocessor(default_signal_preprocessor_config()).with_overrides(
            lowcut=1, highcut=2
        ),
    )

    assert np.allclose(mean, np.full(12, 3.0)), f"Expected mean 3, got {mean}"
    assert np.allclose(std, np.zeros(12)), f"Expected std 0, got {std}"
