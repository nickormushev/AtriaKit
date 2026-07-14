"""Tests for load_dcm_ecg — signal extraction, PFA handling, and error cases."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ecg_annotator.loaders.dcm_loader import load_dcm_ecg


def _mock_dcmread(ecg_signals, ecg_ch_names, pfa_waveforms=None, pfa_ch_names=None):
    """Return a side_effect function that produces a mocked pydicom dataset."""

    def _read(path, **_):
        ds = MagicMock()
        ds.PatientID = "P001"
        ds.ContentTime = "120000"

        ecg_ws = MagicMock()
        ecg_ws.MultiplexGroupLabel = "ECG"
        ecg_ws.SamplingFrequency = 500.0
        ch_defs = [
            MagicMock(ChannelSourceSequence=[MagicMock(CodeMeaning=n)])
            for n in ecg_ch_names
        ]
        ecg_ws.ChannelDefinitionSequence = ch_defs

        waveform_sequences = [ecg_ws]

        def _waveform_array(i):
            return ecg_signals.T if i == 0 else pfa_waveforms.T

        ds.waveform_array = _waveform_array

        if pfa_waveforms is not None:
            iceg_ws = MagicMock()
            iceg_ws.MultiplexGroupLabel = "ICEG"
            iceg_ws.ChannelDefinitionSequence = [
                MagicMock(ChannelSourceSequence=[MagicMock(CodeMeaning=n)])
                for n in pfa_ch_names
            ]
            waveform_sequences.append(iceg_ws)

        ds.WaveformSequence = waveform_sequences
        return ds

    return _read


def test_no_pfa_when_no_iceg():
    ecg = np.ones((12, 100))
    with patch(
        "ecg_annotator.loaders.dcm_loader.pydicom.dcmread",
        side_effect=_mock_dcmread(ecg, [f"L{i}" for i in range(12)]),
    ):
        result = load_dcm_ecg("mock.dcm")

    assert result is not None
    assert result.ecg_mV.shape == (12, 100)
    assert result.pfa_signals is None
    assert result.pfa_lead_names == []


def test_pfa_separated_and_sorted():
    ecg = np.ones((12, 100))
    pfa = np.array([np.full(100, i) for i in range(3)])
    with patch(
        "ecg_annotator.loaders.dcm_loader.pydicom.dcmread",
        side_effect=_mock_dcmread(
            ecg, [f"L{i}" for i in range(12)], pfa, ["PFA_C", "PFA_A", "PFA_B"]
        ),
    ):
        result = load_dcm_ecg("mock.dcm")

    assert result is not None
    assert result.ecg_mV.shape == (12, 100)
    assert result.pfa_signals.shape == (3, 100)
    assert result.pfa_lead_names == ["PFA_A", "PFA_B", "PFA_C"]
    assert list(result.lead_names) == [f"L{i}" for i in range(12)]


def test_pads_lead_names_when_channel_definitions_are_short():
    """A DICOM whose ChannelDefinitionSequence lists fewer names than actual
    waveform channels should still load, with placeholder names padded on —
    not fail, and not leave lead_names shorter than ecg_mV's channel count."""
    ecg = np.ones((12, 100))
    with patch(
        "ecg_annotator.loaders.dcm_loader.pydicom.dcmread",
        side_effect=_mock_dcmread(ecg, [f"L{i}" for i in range(10)]),
    ):
        result = load_dcm_ecg("mock.dcm")

    assert result is not None
    assert result.ecg_mV.shape == (12, 100)
    assert len(result.lead_names) == 12
    assert list(result.lead_names[:10]) == [f"L{i}" for i in range(10)]
    assert list(result.lead_names[10:]) == ["Lead 10", "Lead 11"]


def test_corrupted_file_returns_none(tmp_path):
    bad = tmp_path / "bad.dcm"
    bad.write_text("not a dicom")
    assert load_dcm_ecg(str(bad)) is None


def test_no_ecg_waveform_returns_none():
    with patch("ecg_annotator.loaders.dcm_loader.pydicom.dcmread") as mock_read:
        ws = MagicMock()
        ws.MultiplexGroupLabel = "OTHER"
        ds = MagicMock()
        ds.WaveformSequence = [ws]
        mock_read.return_value = ds

        assert load_dcm_ecg("dummy.dcm") is None
