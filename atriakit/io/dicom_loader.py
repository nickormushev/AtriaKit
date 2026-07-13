from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom

from atriakit.io.base_loader import BaseLoader
from atriakit.models.ecg_data import ECGData


class _WaveformAttributes:
    SEQUENCE = "WaveformSequence"
    SAMPLING_FREQUENCY = "SamplingFrequency"


class DicomLoader(BaseLoader):
    """Loads ECG data from a DICOM file."""

    def load(self, path: str | Path) -> ECGData:
        """Load an ECG recording from a DICOM file.

        Raises:
            ValueError: If the file is not a valid DICOM, contains no ECG
                waveform sequence, or is missing sampling frequency metadata.
        """
        path = str(path)
        try:
            dicom_data = pydicom.dcmread(path).copy()
        except pydicom.errors.InvalidDicomError as e:
            raise ValueError(f"Not a valid DICOM file: {path!r}") from e

        waveform_seq = getattr(dicom_data, _WaveformAttributes.SEQUENCE, None)
        if not waveform_seq:
            raise ValueError(
                f"DICOM file does not contain a valid {_WaveformAttributes.SEQUENCE} sequence."
            )

        ecg_array = None
        waveform = None
        for idx, ws in enumerate(waveform_seq):
            if ws.MultiplexGroupLabel == "ECG":
                ecg_array = dicom_data.waveform_array(idx).T
                waveform = waveform_seq[idx]

        if ecg_array is None:
            raise ValueError("No ECG waveform data found in the DICOM file.")

        fs = getattr(waveform, _WaveformAttributes.SAMPLING_FREQUENCY, None)
        if fs is None:
            raise ValueError(
                f"{_WaveformAttributes.SAMPLING_FREQUENCY} is missing in the waveform metadata."
            )

        lead_to_index = {
            getattr(ch, "ChannelLabel", str(i)): i
            for i, ch in enumerate(waveform.ChannelDefinitionSequence)
        }

        return ECGData(
            ecg=ecg_array,
            fs=fs,
            lead_to_index=lead_to_index,
            content_date=dicom_data.ContentDate,
            content_time=dicom_data.ContentTime,
        )
