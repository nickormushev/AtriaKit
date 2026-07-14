"""Tests for ECGAnnotator business logic that does not depend on rendering."""

import math
import pandas as pd
from unittest.mock import MagicMock

import numpy as np
import pytest

from ecg_annotator.annotator import ECGAnnotator
from ecg_annotator.models.state import AnnotationSession
from ecg_annotator.const import MAIN_ECG_LEAD_NAMES
from ecg_annotator.models.recording import ECGRecording, PatientInfo


def _make_ecg(signals=None, fs=1000.0) -> ECGRecording:
    sigs = signals if signals is not None else np.zeros((12, 10_000))
    return ECGRecording(patient=PatientInfo(id="P_TEST"), ecg_mV=sigs, fs_Hz=fs)


def _stub(signals=None, fs=1000.0, window_duration=30.0) -> ECGAnnotator:
    """Minimal ECGAnnotator without going through __init__."""
    ann = ECGAnnotator.__new__(ECGAnnotator)
    ann.ecg = _make_ecg(signals, fs)
    ann.annotation_session = AnnotationSession.for_leads(MAIN_ECG_LEAD_NAMES, "High")
    ann.window_duration = window_duration
    ann.plotter = MagicMock()
    return ann


def test_update_view_start(qtbot):
    # 300 s signal, 30 s window
    ann = _stub(signals=np.zeros((12, 300_000)), fs=1000.0, window_duration=30.0)

    ann._update_view_start(math.inf)
    assert ann.view_start == 0.0

    ann._update_view_start(5000)
    assert ann.view_start == pytest.approx(4.0)  # 5.0 s - 1.0 s offset

    # Near end: clipped so window fits within signal
    ann._update_view_start(299_000)
    assert ann.view_start == pytest.approx(300_000 / 1000.0 - 30.0)

    # Signal shorter than window: clamps to 0
    short = _stub(signals=np.zeros((12, 100)), fs=1000.0, window_duration=30.0)
    short._update_view_start(50)
    assert short.view_start == 0.0


def test_from_annotations_adds_points():
    ecg = _make_ecg(signals=np.ones((12, 1000)))
    lead_name_to_index = {name: i for i, name in enumerate(MAIN_ECG_LEAD_NAMES)}

    # Valid annotation: all three point types added, earliest updated
    anns = pd.DataFrame([{
        "lead": MAIN_ECG_LEAD_NAMES[0], "onset": 100, "offset": 200, "qrs_onset": 50,
        "comment": None, "confidence": "High", "type": "Before", "ignore": False,
    }])
    session, earliest = AnnotationSession.from_annotations(
        anns, list(MAIN_ECG_LEAD_NAMES), lead_name_to_index, ecg
    )
    lead = session.lead_states[0]
    assert len(lead.red_points) == 1
    assert len(lead.green_points) == 1
    assert len(lead.blue_points) == 1
    assert earliest == 100

    # Out-of-bounds onset (e.g. corrupt CSV): point silently skipped, earliest unchanged
    anns_oob = pd.DataFrame([{
        "lead": MAIN_ECG_LEAD_NAMES[0], "onset": 9999, "offset": 200, "qrs_onset": -1,
        "comment": None, "confidence": "High", "type": "Before", "ignore": False,
    }])
    session2, earliest2 = AnnotationSession.from_annotations(
        anns_oob, list(MAIN_ECG_LEAD_NAMES), lead_name_to_index, ecg
    )
    assert session2.lead_states[0].red_points == []
    assert earliest2 == math.inf


def test_find_near_point(qtbot):
    ann = _stub(fs=1000.0)
    ann.plotter.get_view.return_value = (0.0, 30.0)

    assert ann.find_near_point(1.0, []) is None
    assert ann.find_near_point(1.0, [(1.0, 0.0)]) == 0       # within tolerance
    assert ann.find_near_point(1.1, [(1.0, 0.0)]) is None    # 100 samples away

    # Point outside the current view window is skipped
    ann.plotter.get_view.return_value = (0.0, 5.0)
    assert ann.find_near_point(10.0, [(10.0, 0.0)]) is None


def test_get_next_unannotated_patient_file_idx(tmp_path, qtbot):
    ann = _stub()
    paths = [tmp_path / "a.dcm", tmp_path / "b.dcm", tmp_path / "c.dcm"]
    ann.ecg_data_loader = MagicMock()
    ann.ecg_data_loader.get_file_count.return_value = len(paths)
    ann.ecg_data_loader.get_path.side_effect = lambda i: str(paths[i])
    ann.annotations_loader = MagicMock()

    # a and b already annotated, c is not
    ann.annotations_loader.get_file_annotations.side_effect = [
        pd.DataFrame([{"x": 1}]),
        pd.DataFrame([{"x": 1}]),
        pd.DataFrame(),
    ]
    assert ann.get_next_unnanotated_patient_file_idx(-1) == 2

    # All annotated → -1
    ann.annotations_loader.get_file_annotations.side_effect = None
    ann.annotations_loader.get_file_annotations.return_value = pd.DataFrame([{"x": 1}])
    assert ann.get_next_unnanotated_patient_file_idx(-1) == -1


# --- _add_point / on_click helpers ------------------------------------------

def _click_stub():
    """Stub with everything on_click needs (plotter fully mocked)."""
    ann = _stub(signals=np.zeros((12, 1000)), fs=1000.0)
    ann.patient_file_idx = 0
    ann.num_ecg_leads = 12
    ann.pixel_tolerance = 100
    ann.cached_scaled_leads = [np.zeros(1000) for _ in range(12)]
    ann.plotter.get_view.return_value = (0.0, 30.0)
    ann.plotter.get_ignored_leads.return_value = [False] * 12
    ann.plotter.snap_to_line.return_value = (0.1, 100)   # (x_snap, y_idx)
    ann.plotter.distance_to_line.return_value = 5         # well within tolerance
    return ann


def test_add_point(qtbot):
    ann = _stub()

    ann._add_point("left", 0, 1.0, 0.5)
    assert ann.annotation_session.lead_states[0].red_points == [(1.0, 0.5)]

    ann._add_point("right", 0, 2.0, 0.5)
    assert ann.annotation_session.lead_states[0].green_points == [(2.0, 0.5)]

    ann._add_point("shift_left", 0, 3.0, 0.5)
    assert ann.annotation_session.lead_states[0].blue_points == [(3.0, 0.5)]


def test_on_click_adds_then_removes_point(qtbot):
    ann = _click_stub()

    ann.on_click(0.1, 0.0, "left")
    assert len(ann.annotation_session.lead_states[0].red_points) == 1

    # Second click at the same position removes the point
    ann.on_click(0.1, 0.0, "left")
    assert len(ann.annotation_session.lead_states[0].red_points) == 0


def test_on_click_guards(qtbot):
    # No patient loaded → early return before any plotter call
    ann = _click_stub()
    ann.patient_file_idx = -1
    ann.on_click(0.1, 0.0, "left")
    assert all(len(ls.red_points) == 0 for ls in ann.annotation_session.lead_states)

    # Click outside current view window → no point added
    ann = _click_stub()
    ann.on_click(50.0, 0.0, "left")
    assert all(len(ls.red_points) == 0 for ls in ann.annotation_session.lead_states)

    # Click on an ignored lead → no point added
    ann = _click_stub()
    ann.plotter.get_ignored_leads.return_value = [True] * 12
    ann.on_click(0.1, 0.0, "left")
    assert all(len(ls.red_points) == 0 for ls in ann.annotation_session.lead_states)

    # Click too far from the ECG line → no point added
    ann = _click_stub()
    ann.plotter.distance_to_line.return_value = 200  # > pixel_tolerance (100)
    ann.on_click(0.1, 0.0, "left")
    assert all(len(ls.red_points) == 0 for ls in ann.annotation_session.lead_states)


def test_on_key_pans_and_updates_segments(qtbot):
    ann = _stub(signals=np.zeros((12, 30_000)), fs=1000.0)
    ann.cached_scaled_leads = [np.zeros(30_000) for _ in range(12)]
    ann._update_segments = MagicMock()

    ann.on_key("right")
    ann.plotter.pan.assert_called_once_with("right", ann.ecg.start_hms, ann.cached_scaled_leads)
    ann._update_segments.assert_called_once()

    # No ECG loaded → no-op
    ann2 = _stub()
    ann2.ecg = None
    ann2._update_segments = MagicMock()
    ann2.on_key("left")
    ann2.plotter.pan.assert_not_called()
    ann2._update_segments.assert_not_called()


def test_validate_and_save(qtbot):
    # No patient loaded → delegates to _clear_and_load, never saves
    ann = _stub()
    ann.patient_file_idx = -1
    ann._clear_and_load_patient_by_idx = MagicMock()
    ann.save = MagicMock()
    ann.validate_and_save(0, lambda x: x + 1)
    ann._clear_and_load_patient_by_idx.assert_called_once()
    ann.save.assert_not_called()

    # Popup already open → no-op
    ann = _stub()
    ann.patient_file_idx = 0
    ann.plotter.is_popup_visible.return_value = True
    ann.save = MagicMock()
    ann.validate_and_save(1, lambda x: x + 1)
    ann.save.assert_not_called()

    # Validation fails → show_popup called, save skipped
    ann = _stub()
    ann.patient_file_idx = 0
    ann.plotter.is_popup_visible.return_value = False
    ann.validate_points = MagicMock(side_effect=ValueError("bad points"))
    ann.save = MagicMock()
    ann.validate_and_save(1, lambda x: x + 1)
    ann.plotter.show_popup.assert_called_once()
    ann.save.assert_not_called()

    # Happy path → save called
    ann = _stub()
    ann.patient_file_idx = 0
    ann.plotter.is_popup_visible.return_value = False
    ann.validate_points = MagicMock()
    ann.save = MagicMock()
    ann.validate_and_save(1, lambda x: x + 1)
    ann.save.assert_called_once()


def test_load_patient_signals(qtbot):
    # No files → immediately marks as done
    ann = _stub()
    ann.ecg_data_loader = MagicMock()
    ann.ecg_data_loader.patient_file_paths = []
    ann.load_patient_signals(0, lambda x: x + 1)
    assert ann.patient_file_idx == -1
    assert ann.ecg is None

    # First file fails to load, second succeeds → lands on second file
    ann = _stub()
    ann.ecg_data_loader = MagicMock()
    ann.ecg_data_loader.patient_file_paths = ["a.dcm", "b.dcm"]
    ann.ecg_data_loader.get_file_count.return_value = 2

    def _load(idx):
        if idx == 1:
            ann.ecg = _make_ecg(np.zeros((12, 1000)))

    ann._switch_to_patient_by_idx = MagicMock(side_effect=_load)
    ann.load_patient_signals(0, lambda x: x + 1)
    assert ann.patient_file_idx == 1
    assert ann.ecg is not None


# --- _load_patient_file_by_idx helpers ---------------------------------------

def _lpf_stub():
    """Stub wired for _load_patient_file_by_idx: loaders mocked, ECG vars set."""
    ann = _stub(signals=np.ones((12, 10_000)), fs=1000.0)
    ann.main_lead_names = MAIN_ECG_LEAD_NAMES
    ann.default_confidence = "High"
    ann.lead_name_to_index = {name: i for i, name in enumerate(MAIN_ECG_LEAD_NAMES)}
    ann.annotations_loader = MagicMock()

    ecg_mock = MagicMock()
    ecg_mock.file_name = "test.dcm"
    ecg_mock.lead_names = list(MAIN_ECG_LEAD_NAMES)
    ecg_mock.ecg_mV = np.ones((12, 300_000))
    ecg_mock.fs_Hz = 1000.0

    def _load_ecg(idx):
        ann.ecg = ecg_mock
        ann.window_duration = 30.0
        return ecg_mock

    ann._load_ecg_data = MagicMock(side_effect=_load_ecg)
    return ann


def _ann_row(**kwargs):
    row = {
        "lead": MAIN_ECG_LEAD_NAMES[0], "confidence": "High", "type": "Before",
        "ignore": False, "onset": 100, "offset": 200, "qrs_onset": -1,
        "comment": None, "patient_id": "P001", "p_wave_id": 0, "file_path": "test.dcm",
    }
    row.update(kwargs)
    return row


def test_load_patient_file_by_idx_guards(qtbot):
    # idx=-1 → resets state and returns without loading anything
    ann = _lpf_stub()
    ann._switch_to_patient_by_idx(-1)
    assert ann.ecg is None
    assert ann.annotation_session.comment == ""

    # ecg_data is None (loader failed) → ecg stays None
    ann = _lpf_stub()
    ann._load_ecg_data = MagicMock(return_value=None)
    ann._switch_to_patient_by_idx(0)
    assert ann.ecg is None

    # No existing annotations → ECG loaded but session is fresh
    ann = _lpf_stub()
    ann.annotations_loader.get_annotations_for_ecg.return_value = pd.DataFrame()
    ann._switch_to_patient_by_idx(0)
    assert ann.ecg is not None
    assert all(len(ls.red_points) == 0 for ls in ann.annotation_session.lead_states)


def test_load_patient_file_by_idx_restores_annotations(qtbot):
    # Comment is restored from first annotation row
    ann = _lpf_stub()
    ann.annotations_loader.get_annotations_for_ecg.return_value = pd.DataFrame(
        [_ann_row(comment="study note")]
    )
    ann._switch_to_patient_by_idx(0)
    assert ann.annotation_session.comment == "study note"

    # Ignored lead is marked ignored, no points added
    ann = _lpf_stub()
    ann.annotations_loader.get_annotations_for_ecg.return_value = pd.DataFrame(
        [_ann_row(ignore=True, comment=None)]
    )
    ann._switch_to_patient_by_idx(0)
    assert ann.annotation_session.lead_states[0].ignored is True
    assert ann.annotation_session.lead_states[0].red_points == []

    # Unknown lead name is silently skipped
    ann = _lpf_stub()
    ann.annotations_loader.get_annotations_for_ecg.return_value = pd.DataFrame(
        [_ann_row(lead="NONEXISTENT", comment=None)]
    )
    ann._switch_to_patient_by_idx(0)
    assert all(len(ls.red_points) == 0 for ls in ann.annotation_session.lead_states)

    # Valid annotation → onset/offset points added and view_start updated
    ann = _lpf_stub()
    ann.annotations_loader.get_annotations_for_ecg.return_value = pd.DataFrame(
        [_ann_row(onset=5000, offset=5100, qrs_onset=-1, comment=None)]
    )
    ann._switch_to_patient_by_idx(0)
    assert len(ann.annotation_session.lead_states[0].red_points) == 1
    assert len(ann.annotation_session.lead_states[0].green_points) == 1
    assert ann.view_start == pytest.approx(4.0)  # 5000/1000 - 1.0


# --- variable lead count -----------------------------------------------------

def _variable_lead_stub(lead_names: list[str]):
    """Stub where the loaded patient has a specific set of leads."""
    ann = _lpf_stub()
    ecg_mock = MagicMock()
    ecg_mock.file_name = "test.dcm"
    ecg_mock.lead_names = lead_names
    ecg_mock.ecg_mV = np.ones((len(lead_names), 300_000))
    ecg_mock.fs_Hz = 1000.0

    def _load_ecg(idx):
        ann.ecg = ecg_mock
        ann.window_duration = 30.0
        return ecg_mock

    ann._load_ecg_data = MagicMock(side_effect=_load_ecg)
    ann.annotations_loader.get_annotations_for_ecg.return_value = pd.DataFrame()
    return ann


def test_configure_leads_updates_plotter():
    """_configure_leads must call plotter.update_lead_controls with the new names."""
    names = [f"L{i}" for i in range(5)]
    ann = _variable_lead_stub(names)

    ann._switch_to_patient_by_idx(0)

    ann.plotter.update_lead_controls.assert_called_once_with(names)


def test_configure_leads_variable_count_updates_session():
    """Loading a patient with N leads produces a session with N lead states."""
    for n in (1, 5, 17):
        names = [f"Lead{i}" for i in range(n)]
        ann = _variable_lead_stub(names)
        ann._switch_to_patient_by_idx(0)
        assert len(ann.annotation_session.lead_states) == n
        assert ann.main_lead_names == names
