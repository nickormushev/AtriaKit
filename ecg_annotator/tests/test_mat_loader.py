"""Tests for mat_loader — sort key extraction and signal loading."""

import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import matio
import matio.v7.matreader7 as _r7

from ecg_annotator.loaders.mat_loader import load_mat_ecg, mat_sort_key


def _write_mat(path: Path, rec: dict) -> Path:
    matio.save_to_mat(str(path), {"rec": rec})
    return path


def _minimal_rec(**overrides) -> dict:
    rec = {
        "patient": {
            "id": "TEST001",
            "gender": "M",
            "age_year": 72.0,
            "height_cm": 175.0,
            "weight_kg": 80.0,
            "pacemaker": [],
            "current_condition": np.array(["indication: FU7"]),
            "medical_history": [],
        },
        "file_name": "test.h5",
        "start": np.datetime64("2021-04-22T07:28:25", "ms"),
        "acquision_condition": "resting supine",
        "lead_names": np.array(["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]),
        "dimensionality": "3D",
        "ecg_mV": np.zeros((5000, 8)),
        "fs_Hz": 1000.0,
        "filter": {"band_pass_Hz": np.array([0.05, 300.0]), "notch_Hz": []},
    }
    rec.update(overrides)
    return rec


def test_load_happy_path(tmp_path):
    path = _write_mat(tmp_path / "rec.mat", _minimal_rec())
    rec = load_mat_ecg(path)

    assert rec.ecg_mV.shape == (8, 5000)  # load_mat_ecg transposes to (n_leads, n_samples)
    assert rec.fs_Hz == 1000.0
    assert list(rec.lead_names) == ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert rec.start == datetime(2021, 4, 22, 7, 28, 25, tzinfo=timezone.utc)
    assert str(rec.dimensionality) == "3D"
    assert str(rec.acquisition_condition) == "resting supine"
    assert str(rec.filter.band_pass_Hz[0]) == pytest.approx(str(0.05))

    p = rec.patient
    assert str(p.id) == "TEST001"
    assert str(p.gender) == "M"
    assert p.age_year == 72.0
    assert [str(c) for c in p.current_condition] == ["indication: FU7"]


@pytest.mark.parametrize("field", ["ecg_mV", "fs_Hz", "lead_names"])
def test_missing_required_field_raises(tmp_path, field):
    rec_dict = _minimal_rec()
    del rec_dict[field]
    path = _write_mat(tmp_path / "rec.mat", rec_dict)
    with pytest.raises(ValueError, match=field):
        load_mat_ecg(path)


def test_missing_optional_fields_use_defaults(tmp_path):
    rec_dict = _minimal_rec()
    del rec_dict["patient"]
    del rec_dict["filter"]
    del rec_dict["start"]
    path = _write_mat(tmp_path / "rec.mat", rec_dict)
    rec = load_mat_ecg(path)

    assert rec.patient.id == "anonymous"
    assert rec.start is None
    assert np.isnan(rec.filter.band_pass_Hz[0])


def test_missing_rec_struct_raises(tmp_path):
    path = tmp_path / "bad.mat"
    matio.save_to_mat(str(path), {"not_rec": np.zeros((1,))})
    with pytest.raises(ValueError, match="rec"):
        load_mat_ecg(path)


def test_invalid_file_raises(tmp_path):
    path = tmp_path / "bad.mat"
    path.write_bytes(b"not a mat file")
    with pytest.raises(ValueError):
        load_mat_ecg(path)


# --- mat_sort_key tests ---

def test_sort_key_reads_only_metadata(tmp_path):
    """mat_sort_key must return patient id and datetime without accessing ecg_mV."""
    path = _write_mat(tmp_path / "rec.mat", _minimal_rec())

    original = _r7.MatRead7.read_h5_data
    accessed = []

    def tracking(self, obj):
        accessed.append(obj.name if hasattr(obj, "name") else str(obj))
        return original(self, obj)

    with patch.object(_r7.MatRead7, "read_h5_data", tracking):
        key = mat_sort_key(path)

    assert key is not None
    assert "TEST001" in key
    assert not any("ecg_mV" in f for f in accessed), (
        f"mat_sort_key accessed ecg_mV — fields touched: {accessed}"
    )
