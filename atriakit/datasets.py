"""Synthetic demo ECG data for AtriaKit.

Requires neurokit2, which is not installed by default::

    pip install neurokit2
    from atriakit import create_demo_data
    create_demo_data()                       # writes to ./demo_data/
    create_demo_data(output_dir="./my_data")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
DURATION = 10  # seconds

_LEAD_SCALES = {
    "I":   0.70, "II":  1.00, "III": 0.40,
    "aVR": -0.80, "aVL": 0.30, "aVF": 0.70,
    "V1":  -0.30, "V2":  0.20, "V3":  0.80,
    "V4":  1.20,  "V5":  1.10, "V6":  0.90,
}
_MV_TO_RAW = 1000  # 1 mV → 1000 int16 units; sensitivity = 0.001 mV/unit

DEMO_RECORDINGS = [
    {"heart_rate": 60, "patient_id": "DEMO001", "seed": 0, "date": "20240101", "time": "090000"},
    {"heart_rate": 75, "patient_id": "DEMO002", "seed": 1, "date": "20240615", "time": "143000"},
    {"heart_rate": 90, "patient_id": "DEMO003", "seed": 2, "date": "20241201", "time": "111500"},
]


def _require_neurokit():
    try:
        import neurokit2 as nk
        return nk
    except ImportError:
        raise SystemExit(
            "Demo data generation requires neurokit2.\n"
            "Install it with:  pip install neurokit2"
        )


def simulate_12lead(heart_rate: int, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """Simulate a 12-lead ECG and return a ``(12, n_samples)`` array in mV."""
    nk = _require_neurokit()
    rng = np.random.default_rng(seed)
    lead_ii = np.array(
        nk.ecg_simulate(duration=DURATION, sampling_rate=FS, heart_rate=heart_rate, noise=noise)
    )
    ecg = np.zeros((12, len(lead_ii)))
    for i, lead in enumerate(LEADS):
        per_lead_noise = rng.normal(0, noise * 0.05, len(lead_ii))
        ecg[i] = _LEAD_SCALES[lead] * lead_ii + per_lead_noise
    return ecg


def to_dicom(
    ecg: np.ndarray,
    patient_id: str,
    date: str = "20240101",
    time: str = "120000",
    leads: list[str] | None = None,
) -> FileDataset:
    """Pack an ECG array into a DICOM FileDataset.

    Args:
        ecg: Signal array with shape ``(n_leads, n_samples)`` in mV.
        patient_id: Patient identifier written into DICOM metadata.
        date: Study date in ``YYYYMMDD`` format.
        time: Study time in ``HHMMSS`` format.
        leads: Channel labels matching the row order of ``ecg``.
            Defaults to the standard 12-lead ``LEADS`` list.

    Returns:
        A ``pydicom.FileDataset`` ready to save with ``ds.save_as(path)``.
    """
    if leads is None:
        leads = LEADS
    sop_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = sop_uid
    ds.PatientID = patient_id
    ds.PatientName = f"Demo^{patient_id}"
    ds.ContentDate = date
    ds.ContentTime = time
    ds.StudyDate = date
    ds.StudyTime = time

    wf = Dataset()
    wf.MultiplexGroupLabel = "ECG"
    wf.SamplingFrequency = float(FS)

    n_ch, n_samples = ecg.shape
    wf.add_new(pydicom.tag.Tag(0x003A, 0x0005), "US", n_ch)       # NumberOfChannels
    wf.add_new(pydicom.tag.Tag(0x003A, 0x0010), "UL", n_samples)  # NumberOfWaveformSamples
    wf.WaveformBitsAllocated = 16
    wf.WaveformSampleInterpretation = "SS"

    channels = []
    for lead in leads:
        ch = Dataset()
        ch.ChannelLabel = lead
        units = Dataset()
        units.CodeValue = "mV"
        units.CodingSchemeDesignator = "UCUM"
        units.CodeMeaning = "millivolt"
        ch.ChannelSensitivityUnitsSequence = Sequence([units])
        ch.ChannelSensitivity = str(1.0 / _MV_TO_RAW)
        ch.ChannelSensitivityCorrectionFactor = "1.0"
        ch.WaveformBitsStored = 16
        channels.append(ch)
    wf.ChannelDefinitionSequence = Sequence(channels)

    raw = np.clip((ecg * _MV_TO_RAW).T, -32768, 32767).astype(np.int16)
    wf.WaveformData = raw.tobytes()

    ds.WaveformSequence = Sequence([wf])
    return ds


def detect_p_waves(ecg_12lead: np.ndarray) -> list[dict]:
    """Detect P-wave onsets and offsets on Lead II using NeuroKit2.

    Returns a list of dicts with keys ``onset``, ``offset``, and ``qrs_onset``
    (sample indices at the pipeline's sampling rate).
    """
    nk = _require_neurokit()
    lead_ii = ecg_12lead[1]
    signals, _ = nk.ecg_process(lead_ii, sampling_rate=FS)

    p_onset_mask  = signals.get("ECG_P_Onsets",  pd.Series(0, index=signals.index)).fillna(0).astype(int)
    p_offset_mask = signals.get("ECG_P_Offsets", pd.Series(0, index=signals.index)).fillna(0).astype(int)
    r_peak_mask   = signals.get("ECG_R_Peaks",   pd.Series(0, index=signals.index)).fillna(0).astype(int)

    onsets  = np.where(p_onset_mask  == 1)[0]
    offsets = np.where(p_offset_mask == 1)[0]
    r_peaks = np.where(r_peak_mask   == 1)[0]

    records: list[dict] = []
    used_offsets: set[int] = set()
    for onset in onsets:
        future_offsets = offsets[offsets > onset]
        if not len(future_offsets):
            continue
        offset = int(future_offsets[0])
        if offset in used_offsets:
            continue
        used_offsets.add(offset)
        following_r = r_peaks[r_peaks > offset]
        qrs_onset = int(following_r[0]) - 20 if len(following_r) else -1
        records.append({"onset": int(onset), "offset": int(offset), "qrs_onset": qrs_onset})

    return records


def build_annotations(
    file_path: str,
    patient_id: str,
    p_waves: list[dict],
    seed: int = 42,
) -> pd.DataFrame:
    """Expand Lead II P-wave detections into a per-lead annotations DataFrame.

    Adds small random jitter to onset/offset per lead to simulate realistic
    inter-lead variation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p_wave_id, pw in enumerate(p_waves):
        for lead in LEADS:
            rows.append({
                "patient_id": patient_id,
                "lead":       lead,
                "p_wave_id":  p_wave_id,
                "onset":      pw["onset"]  + int(rng.integers(-2, 3)),
                "offset":     pw["offset"] + int(rng.integers(-2, 3)),
                "qrs_onset":  pw["qrs_onset"],
                "ignore":     False,
                "file_path":  file_path,
            })
    return pd.DataFrame(rows)


def generate(output_dir: str | Path) -> Path:
    """Generate demo DICOM files and an annotations CSV into ``output_dir``.

    Creates one DICOM per entry in ``DEMO_RECORDINGS`` and a combined
    ``demo_annotations.csv``. Returns the output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_annotations: list[pd.DataFrame] = []
    for cfg in DEMO_RECORDINGS:
        patient_id = cfg["patient_id"]
        print(f"[{patient_id}] Simulating 12-lead ECG at {cfg['heart_rate']} bpm …")

        ecg = simulate_12lead(cfg["heart_rate"], seed=cfg["seed"])

        dcm = to_dicom(ecg, patient_id=patient_id, date=cfg["date"], time=cfg["time"])
        dcm_path = out / f"{patient_id}.dcm"
        dcm.save_as(str(dcm_path))
        print(f"  Saved {dcm_path}")

        p_waves = detect_p_waves(ecg)
        print(f"  Detected {len(p_waves)} P-waves")

        anns = build_annotations(f"{patient_id}.dcm", patient_id, p_waves, seed=cfg["seed"])
        all_annotations.append(anns)

    combined = pd.concat(all_annotations, ignore_index=True)
    ann_path = out / "demo_annotations.csv"
    combined.to_csv(ann_path, index=False)
    print(f"\nSaved {ann_path} ({len(combined)} rows across {combined['patient_id'].nunique()} patients)")
    return out