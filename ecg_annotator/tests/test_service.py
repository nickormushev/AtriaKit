import pytest

from ecg_annotator.annotations.state import AnnotationSession
from ecg_annotator.annotations.validation import validate_annotation_session
from ecg_annotator.annotations.serialization import build_annotation_rows


def make_session():
    session = AnnotationSession.for_leads(["I", "II"])
    session.annotation_type = "Before"
    return session


def test_validate_annotation_session_normalizes_points():
    session = make_session()
    lead_one = session.lead_states[0]
    lead_two = session.lead_states[1]

    lead_one.red_points = [(0.1, 0.0), (0.1, 0.0)]
    lead_one.green_points = [(0.2, 0.0)]
    lead_two.red_points = [(0.4, 0.0)]
    lead_two.green_points = [(0.5, 0.0)]

    validate_annotation_session(session, "120000", distance_threshold=0.4)

    assert lead_one.red_points == [(0.1, 0.0)]


def test_validate_annotation_session_requires_annotation_type_for_active_leads():
    session = AnnotationSession.for_leads(["I"])
    session.lead_states[0].red_points = [(0.1, 0.0)]
    session.lead_states[0].green_points = [(0.2, 0.0)]

    with pytest.raises(ValueError, match="Please select at least one"):
        validate_annotation_session(session, "120000", distance_threshold=0.4)


def test_build_annotation_rows_serializes_session_state():
    session = make_session()
    session.comment = "looks good"

    lead_one = session.lead_states[0]
    lead_one.confidence = "Med"
    lead_one.red_points = [(0.1, 0.0)]
    lead_one.green_points = [(0.2, 0.0)]
    lead_one.blue_points = [(0.25, 0.0)]

    lead_two = session.lead_states[1]
    lead_two.ignored = True

    rows = build_annotation_rows(
        session,
        patient_id="p1",
        patient_file="file.IMA",
        time_to_sample=lambda time_sec: int(round(time_sec * 100)),
    )

    assert rows == [
        {
            "patient_id": "p1",
            "lead": "I",
            "p_wave_id": 0,
            "onset": 10,
            "offset": 20,
            "qrs_onset": 25,
            "ignore": False,
            "comment": "looks good",
            "type": "Before",
            "confidence": "Med",
            "file_path": "file.IMA",
        },
        {
            "patient_id": "p1",
            "lead": "II",
            "p_wave_id": -1,
            "onset": -1,
            "offset": -1,
            "qrs_onset": -1,
            "ignore": True,
            "type": "Ignored",
            "comment": "looks good",
            "confidence": "High",
            "file_path": "file.IMA",
        },
    ]
