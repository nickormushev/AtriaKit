"""Tests for DataLoader — discovery, sorting, and load-by-index across all supported formats."""

import numpy as np
import pydicom
from pathlib import Path
from unittest.mock import patch
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from ecg_annotator.loaders.data_loader import DataLoader


def _write_dicom(
    path: Path,
    patient_id: str,
    content_time: str,
    waveform: bool = False,
    num_channels: int = 12,
    num_samples: int = 4000,
    fs: float = 2000.0,
):
    """Write a minimal valid DICOM. Pass waveform=True to include an ECG waveform sequence."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.ContentTime = content_time

    if waveform:
        data = np.zeros((num_samples, num_channels), dtype=np.int16)
        channels = []
        for _ in range(num_channels):
            ch = Dataset()
            src = Dataset()
            src.CodeMeaning = "Lead"
            ch.ChannelSourceSequence = Sequence([src])
            channels.append(ch)

        wf = Dataset()
        wf.MultiplexGroupLabel = "ECG"
        wf.SamplingFrequency = fs
        wf.NumberOfWaveformChannels = num_channels
        wf.NumberOfWaveformSamples = num_samples
        wf.WaveformBitsAllocated = 16
        wf.WaveformSampleInterpretation = "SS"
        wf.WaveformData = data.tobytes()
        wf.ChannelDefinitionSequence = Sequence(channels)
        ds.WaveformSequence = Sequence([wf])

    pydicom.dcmwrite(str(path), ds)


def _write_dicom_without_required_tags(path: Path):
    """Write a DICOM that is missing PatientID and ContentTime."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    pydicom.dcmwrite(str(path), ds)


# --- Discovery tests ---

def test_discovery_finds_ima_files_recursively(tmp_path):
    subdir = tmp_path / "patient1" / "study"
    subdir.mkdir(parents=True)
    _write_dicom(tmp_path / "top.IMA", "P001", "120000")
    _write_dicom(subdir / "nested.IMA", "P001", "120001")

    loader = DataLoader(str(tmp_path))
    assert len(loader.patient_file_paths) == 2


def test_discovery_accepts_dcm_and_ima(tmp_path):
    _write_dicom(tmp_path / "file.dcm", "P001", "120000")
    _write_dicom(tmp_path / "file.IMA", "P002", "120001")
    (tmp_path / "notes.txt").write_text("ignored")

    loader = DataLoader(str(tmp_path))
    assert len(loader.patient_file_paths) == 2


def test_discovery_skips_dicoms_without_required_tags(tmp_path):
    _write_dicom_without_required_tags(tmp_path / "no_tags.IMA")
    _write_dicom(tmp_path / "valid.IMA", "P001", "120000")

    loader = DataLoader(str(tmp_path))
    assert len(loader.patient_file_paths) == 1


def test_discovery_sorts_by_patient_id_then_time(tmp_path):
    _write_dicom(tmp_path / "b.IMA", "P003", "120000")
    _write_dicom(tmp_path / "a.IMA", "P001", "120000")
    _write_dicom(tmp_path / "c.IMA", "P002", "130000")

    loader = DataLoader(str(tmp_path))
    names = [Path(p).name for p in loader.patient_file_paths]
    assert names == ["a.IMA", "c.IMA", "b.IMA"]


def test_discovery_empty_directory(tmp_path):
    loader = DataLoader(str(tmp_path))
    assert loader.get_file_count() == 0


def test_no_data_dir_gives_empty_loader():
    loader = DataLoader(None)
    assert loader.get_file_count() == 0


def test_get_path_returns_none_for_out_of_bounds_index(tmp_path):
    loader = DataLoader(str(tmp_path))
    assert loader.get_path(-1) is None
    assert loader.get_path(0) is None


# --- Load by index tests ---

def test_load_patient_returns_none_for_negative_index(tmp_path):
    loader = DataLoader(str(tmp_path))
    assert loader.load_patient_by_idx(-1) is None


def test_load_patient_returns_none_for_out_of_bounds_index(tmp_path):
    loader = DataLoader(str(tmp_path))
    assert loader.load_patient_by_idx(0) is None


def test_load_patient_extracts_ecg_signals(tmp_path):
    _write_dicom(
        tmp_path / "ecg.IMA", "P001", "120000",
        waveform=True, num_channels=12, num_samples=4000, fs=2000.0,
    )

    loader = DataLoader(str(tmp_path))
    ecg = loader.load_patient_by_idx(0)

    assert ecg is not None
    assert ecg.ecg_mV.shape == (12, 4000)
    assert ecg.fs_Hz == 2000.0
    assert ecg.patient.id == "P001"


# --- Routing tests ---

def test_dcm_file_uses_dcm_loader(tmp_path):
    """load_patient_by_idx must call load_dcm_ecg for .dcm/.IMA files."""
    _write_dicom(tmp_path / "ecg.IMA", "P001", "120000")
    loader = DataLoader(str(tmp_path))

    with patch("ecg_annotator.loaders.data_loader.load_dcm_ecg") as mock_dcm:
        loader.load_patient_by_idx(0)

    mock_dcm.assert_called_once()
