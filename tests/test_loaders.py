from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pydicom
import pytest

from atriakit.models.ecg_data import ECGData
from atriakit.io import BaseLoader, DicomLoader, ECGLoader

_STD_12 = {"I":0,"II":1,"III":2,"aVR":3,"aVL":4,"aVF":5,"V1":6,"V2":7,"V3":8,"V4":9,"V5":10,"V6":11}
_STD_CHANNELS = [SimpleNamespace(ChannelLabel=name) for name in _STD_12]


def _make_fake_dicom(monkeypatch, waveform=None, ecg_signal=None):
    if ecg_signal is None:
        ecg_signal = np.ones((100, 12))
    if waveform is None:
        waveform = SimpleNamespace(
            MultiplexGroupLabel="ECG",
            SamplingFrequency=500,
            ChannelDefinitionSequence=_STD_CHANNELS,
        )

    ds = MagicMock()
    ds.WaveformSequence = [waveform]
    ds.ContentTime = "120000"
    ds.ContentDate = "20230101"
    ds.waveform_array.return_value = ecg_signal
    ds.copy.return_value = ds

    monkeypatch.setattr("atriakit.io.dicom_loader.pydicom.dcmread", lambda _: ds)
    return ds


# ── DicomLoader ───────────────────────────────────────────────────────────────

def test_dicom_loader_happy_path(monkeypatch):
    received_paths = []
    ds = _make_fake_dicom(monkeypatch)
    monkeypatch.setattr(
        "atriakit.io.dicom_loader.pydicom.dcmread",
        lambda path: received_paths.append(path) or ds,
    )

    ecg_data = DicomLoader().load("fake.dcm")

    assert received_paths == ["fake.dcm"]
    assert isinstance(ecg_data, ECGData)
    assert ecg_data.sampling_frequency == 500
    assert ecg_data.ecg.shape == (12, 100)
    assert ecg_data.content_start_time == "120000"
    assert ecg_data.content_date == "20230101"


def test_dicom_loader_invalid_dicom_raises(monkeypatch):
    monkeypatch.setattr(
        "atriakit.io.dicom_loader.pydicom.dcmread",
        lambda _: (_ for _ in ()).throw(pydicom.errors.InvalidDicomError),
    )
    with pytest.raises(ValueError, match="Not a valid DICOM file"):
        DicomLoader().load("fake.dcm")


def test_dicom_loader_missing_waveform_sequence_raises(monkeypatch):
    ds = MagicMock()
    ds.WaveformSequence = None
    ds.copy.return_value = ds
    monkeypatch.setattr("atriakit.io.dicom_loader.pydicom.dcmread", lambda _: ds)
    with pytest.raises(ValueError, match="WaveformSequence"):
        DicomLoader().load("fake.dcm")


def test_dicom_loader_no_ecg_label_raises(monkeypatch):
    _make_fake_dicom(
        monkeypatch,
        waveform=SimpleNamespace(MultiplexGroupLabel="RHYTHM", SamplingFrequency=500),
    )
    with pytest.raises(ValueError, match="No ECG waveform"):
        DicomLoader().load("fake.dcm")


def test_dicom_loader_missing_sampling_frequency_raises(monkeypatch):
    _make_fake_dicom(
        monkeypatch,
        waveform=SimpleNamespace(MultiplexGroupLabel="ECG"),
    )
    with pytest.raises(ValueError, match="SamplingFrequency"):
        DicomLoader().load("fake.dcm")


# ── ECGLoader ─────────────────────────────────────────────────────────────────

def test_ecg_loader_dispatches_to_dicom(monkeypatch):
    _make_fake_dicom(monkeypatch)
    ecg_data = ECGLoader().load("recording.dcm")
    assert isinstance(ecg_data, ECGData)


def test_ecg_loader_unknown_extension_raises():
    with pytest.raises(ValueError, match="No loader registered for extension"):
        ECGLoader().load("recording.edf")


def test_ecg_loader_register_custom_loader():
    class FakeLoader:
        def load(self, path) -> ECGData:
            return ECGData(ecg=np.zeros((12, 100)), fs=250, lead_to_index=_STD_12)

    loader = ECGLoader()
    loader.register(".EDF", FakeLoader())

    ecg_data = loader.load("recording.edf")

    assert isinstance(ecg_data, ECGData)
    assert ecg_data.sampling_frequency == 250

def test_base_loader_protocol():
    class MyLoader:
        def load(self, path) -> ECGData:
            return ECGData(ecg=np.zeros((12, 10)), fs=500, lead_to_index=_STD_12)

    assert isinstance(MyLoader(), BaseLoader)
