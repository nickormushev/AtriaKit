import logging
from datetime import datetime

import numpy as np
import pydicom

from ecg_annotator.models.recording import ECGRecording, PatientInfo

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".dcm", ".IMA"})

def dcm_sort_key(path: str) -> str | None:
    """Return PatientID+ContentTime sort key for a DICOM file, or None to skip it."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["PatientID", "ContentTime"])

        if not hasattr(ds, "PatientID") or not hasattr(ds, "ContentTime"):
            return None

        return str(ds.PatientID) + str(ds.ContentTime)
    except Exception:
        log.warning("Failed to extract sort key from DICOM file: %s", path)
        return None


def load_dcm_ecg(path: str) -> ECGRecording | None:
    """Load a DICOM ECG file and return an ECGRecording, or None on failure."""
    try:
        ds = pydicom.dcmread(path)
        signals = None
        iceg_idx = -1
        lead_names = []
        for i, ws in enumerate(ds.WaveformSequence):
            if ws.MultiplexGroupLabel == "ICEG":
                iceg_idx = i
            if ws.MultiplexGroupLabel == "ECG":
                signals = ds.waveform_array(i).T
                lead_names = [
                    ch.ChannelSourceSequence[0].CodeMeaning
                    for ch in getattr(ws, "ChannelDefinitionSequence", [])
                ]

        if signals is None:
            raise ValueError("No ECG waveform found")

        pfa_signals = None
        pfa_lead_names = []

        if iceg_idx != -1:
            wf = ds.WaveformSequence[iceg_idx]
            arr = ds.waveform_array(iceg_idx).T
            raw_pfa_names = []
            waveforms = []
            for i, ch_seq in enumerate(wf.ChannelDefinitionSequence):
                ch_name = ch_seq.ChannelSourceSequence[0].CodeMeaning
                if "FP" in ch_name or "PFA" in ch_name:
                    raw_pfa_names.append(ch_name)
                    waveforms.append(arr[i])
                    if len(raw_pfa_names) == 5:
                        break
            if waveforms:
                sorted_pairs = sorted(zip(raw_pfa_names, waveforms), key=lambda x: x[0])
                pfa_lead_names, waveforms = zip(*sorted_pairs)
                pfa_lead_names = list(pfa_lead_names)
                pfa_signals = np.vstack(waveforms)

        fs = ds.WaveformSequence[0].SamplingFrequency
        log.info(
            "Loaded patient %s | file=%s | ecg_leads=%d | pfa_leads=%d | fs=%g Hz",
            ds.PatientID, path, signals.shape[0], len(pfa_lead_names), fs,
        )
        return ECGRecording(
            patient=PatientInfo(id=ds.PatientID),
            file_name=path,
            start=datetime.strptime(str(ds.ContentTime).split(".")[0], "%H%M%S") if ds.ContentTime else None,
            lead_names=np.array(lead_names),
            ecg_mV=signals,
            fs_Hz=float(fs),
            pfa_signals=pfa_signals,
            pfa_lead_names=pfa_lead_names,
        )
    except Exception as e:
        log.error("Failed to read %s: %s", path, e)
        return None
