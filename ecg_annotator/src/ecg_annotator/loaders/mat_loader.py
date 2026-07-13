"""Loader for MATLAB v7.3 HDF5 ECG recordings (.mat files saved by sema_h5_2mat)."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matio
import matio.v7.matreader7 as _matio_r7
from matio.matio import read_mat_header as _read_mat_header
import numpy as np

from ecg_annotator.models.recording import ECGRecording, FilterInfo, PatientInfo

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".mat"})

def mat_sort_key(path: str | Path) -> str | None:
    """Return patient_id+start sort key for a MAT file using minimal field reads."""
    try:
        _, _, byte_order = _read_mat_header(str(path))

        with h5py.File(str(path), "r") as f:

            subsystem = _matio_r7.read_subsystem(f, byte_order, False, False)
            reader = _matio_r7.MatRead7(f, subsystem)

            rec = f["rec"]
            pid_raw = reader.read_h5_data(rec["patient"]["id"])
            patient_id = str(np.asarray(pid_raw).flat[0])
            start_raw = reader.read_h5_data(rec["start"])
            start = np.asarray(start_raw, dtype="datetime64[ms]").flat[0]
            start_str = start.astype(datetime).strftime("%Y%m%d%H%M%S") if not np.isnat(start) else "00000000000000"

        return patient_id + start_str
    except Exception:
        log.warning("Failed to extract sort key from MAT file: %s", path)
        return None


def _parse_datetime(val) -> datetime | None:
    ts = np.asarray(val, dtype="datetime64[ms]").flat[0]
    if np.isnat(ts):
        return None
    return ts.astype(datetime).replace(tzinfo=timezone.utc)


def _required(struct, field: str) -> Any:
    """Return struct[field], raising a clear error if absent."""
    try:
        return struct[field]
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"Required field '{field}' missing from MAT file: {e}") from e


def _optional(struct, field: str, default=None) -> Any:
    """Return struct[field], or default with a warning if absent."""
    try:
        return struct[field]
    except (KeyError, IndexError, ValueError):
        log.warning("Optional field '%s' missing, using default: %s", field, default)
        return default


def load_mat_ecg(path: str | Path) -> ECGRecording:
    """Load a MATLAB v7.3 HDF5 ECG recording produced by sema_h5_2mat."""
    path = Path(path)
    log.info("Loading MATLAB ECG: %s", path)

    try:
        data = matio.load_from_mat(str(path))
    except Exception as e:
        raise ValueError(f"Failed to open MAT file '{path}': {e}") from e

    try:
        rec = data["rec"].flat[0]
    except (KeyError, IndexError) as e:
        raise ValueError(f"'{path}' does not contain a 'rec' struct: {e}") from e

    # --- required fields -------------------------------------------------
    ecg_mV = _required(rec, "ecg_mV")                      # (n_samples, n_leads)
    fs_Hz = _required(rec, "fs_Hz").flat[0]
    lead_names = _required(rec, "lead_names").flatten()

    # --- optional metadata -----------------------------------------------
    dimensionality = _optional(rec, "dimensionality", np.str_("")).flat[0]
    file_name = _optional(rec, "file_name", np.str_(path.name)).flat[0]
    acquisition_condition = _optional(rec, "acquision_condition", np.str_("")).flat[0]
    start = _parse_datetime(_optional(rec, "start", np.datetime64("NaT")))

    # --- patient demographics (all optional) -----------------------------
    p_raw = _optional(rec, "patient")
    if p_raw is not None:
        p = p_raw.flat[0]
        patient = PatientInfo(
            id=_optional(p, "id", np.str_("anonymous")).flat[0],
            gender=_optional(p, "gender", np.str_("")).flat[0],
            age_year=_optional(p, "age_year", np.float64("nan")).flat[0],
            height_cm=_optional(p, "height_cm", np.float64("nan")).flat[0],
            weight_kg=_optional(p, "weight_kg", np.float64("nan")).flat[0],
            current_condition=_optional(p, "current_condition", np.array([])).flatten(),
            medical_history=_optional(p, "medical_history", np.array([])).flatten(),
            pacemaker=_optional(p, "pacemaker", np.array([])).flatten(),
        )
    else:
        patient = PatientInfo()

    # --- filter (optional) -----------------------------------------------
    filt_raw = _optional(rec, "filter")
    if filt_raw is not None:
        filt = filt_raw.flat[0]
        bp = _optional(filt, "band_pass_Hz", np.array([float("nan"), float("nan")])).flatten()
        filter_info = FilterInfo(
            band_pass_Hz=(bp[0], bp[1]),
            notch_Hz=_optional(filt, "notch_Hz", np.array([])).flatten(),
        )
    else:
        filter_info = FilterInfo()

    ecg_mV = ecg_mV.T  # MAT stores (n_samples, n_leads); ECGRecording expects (n_leads, n_samples)
    log.info(
        "Loaded %s | leads=%d | fs=%g Hz | dimensionality=%s",
        file_name, ecg_mV.shape[0], fs_Hz, dimensionality,
    )

    return ECGRecording(
        patient=patient,
        file_name=file_name,
        start=start,
        acquisition_condition=acquisition_condition,
        lead_names=lead_names,
        dimensionality=dimensionality,
        ecg_mV=ecg_mV,
        fs_Hz=fs_Hz,
        filter=filter_info,
    )
