import pytest
from pathlib import Path

from ecg_annotator.loaders.annotation_loader import AnnotationsLoader, ANNOTATION_COLUMNS


def _row(file_path="file.IMA", patient_id="P001", lead="I", onset=100, offset=200, **kwargs):
    return {
        "patient_id": patient_id,
        "lead": lead,
        "p_wave_id": 0,
        "onset": onset,
        "offset": offset,
        "qrs_onset": -1,
        "ignore": False,
        "comment": "",
        "confidence": "High",
        "type": "Before",
        "file_path": file_path,
        **kwargs,
    }


# --- Initialisation ---

def test_new_loader_starts_empty_when_no_file(tmp_path):
    loader = AnnotationsLoader(tmp_path / "out.csv")
    assert loader.df.empty
    assert list(loader.df.columns) == ANNOTATION_COLUMNS


def test_loader_reads_existing_csv(tmp_path):
    csv = tmp_path / "out.csv"
    loader = AnnotationsLoader(csv)
    loader.replace_file_annotations("file.IMA", [_row(onset=42)])
    loader.save()

    reloaded = AnnotationsLoader(csv)
    assert len(reloaded.df) == 1
    assert reloaded.df.iloc[0]["onset"] == 42


# --- Round-trip save/reload ---

def test_round_trip_preserves_all_columns(tmp_path):
    csv = tmp_path / "out.csv"
    loader = AnnotationsLoader(csv)
    row = _row(patient_id="P999", lead="V1", onset=500, offset=600, comment="test comment", confidence="Low")
    loader.replace_file_annotations("file.IMA", [row])
    loader.save()

    reloaded = AnnotationsLoader(csv)
    r = reloaded.df.iloc[0]
    assert r["patient_id"] == "P999"
    assert r["lead"] == "V1"
    assert r["onset"] == 500
    assert r["offset"] == 600
    assert r["comment"] == "test comment"
    assert r["confidence"] == "Low"


# --- replace_file_annotations ---

def test_replace_removes_previous_rows_for_same_file(tmp_path):
    loader = AnnotationsLoader(tmp_path / "out.csv")
    loader.replace_file_annotations("file.IMA", [_row(onset=100)])
    loader.replace_file_annotations("file.IMA", [_row(onset=999)])

    assert len(loader.df) == 1
    assert loader.df.iloc[0]["onset"] == 999


def test_replace_keeps_other_files_untouched(tmp_path):
    loader = AnnotationsLoader(tmp_path / "out.csv")
    loader.replace_file_annotations("file_a.IMA", [_row(file_path="file_a.IMA", onset=100)])
    loader.replace_file_annotations("file_b.IMA", [_row(file_path="file_b.IMA", onset=200)])
    loader.replace_file_annotations("file_a.IMA", [_row(file_path="file_a.IMA", onset=999)])

    b_rows = loader.df[loader.df["file_path"] == "file_b.IMA"]
    assert len(b_rows) == 1
    assert b_rows.iloc[0]["onset"] == 200


def test_replace_with_empty_list_removes_all_rows(tmp_path):
    loader = AnnotationsLoader(tmp_path / "out.csv")
    loader.replace_file_annotations("file.IMA", [_row(), _row(lead="II")])
    loader.replace_file_annotations("file.IMA", [])

    assert loader.df.empty


# --- get_annotations_for_ecg (fuzzy path matching) ---

def test_get_annotations_matches_same_file(tmp_path):
    path = "ECG.0001.ABC123.IMA"
    loader = AnnotationsLoader(tmp_path / "out.csv")
    loader.replace_file_annotations(path, [_row(file_path=path)])

    result = loader.get_annotations_for_ecg(path)
    assert len(result) == 1


def test_get_annotations_matches_different_path_same_identity(tmp_path):
    """Files regenerated under a new path should still match via first+second-last filename parts."""
    old = "OLD_DIR/ECG.0001.2024.11.ABC123.IMA"
    new = "NEW_DIR/ECG.0001.2025.06.ABC123.IMA"

    loader = AnnotationsLoader(tmp_path / "out.csv")
    loader.replace_file_annotations(old, [_row(file_path=old)])

    result = loader.get_annotations_for_ecg(new)
    assert len(result) == 1


def test_get_annotations_no_match_for_different_patient(tmp_path):
    loader = AnnotationsLoader(tmp_path / "out.csv")
    loader.replace_file_annotations("ECG.0001.ABC123.IMA", [_row()])

    result = loader.get_annotations_for_ecg("JJ.ECG.0001.ABC123.IMA")
    assert result.empty


def test_get_annotations_empty_loader_returns_empty(tmp_path):
    loader = AnnotationsLoader(tmp_path / "out.csv")
    result = loader.get_annotations_for_ecg("ECG.0001.ABC123.IMA")
    assert result.empty
