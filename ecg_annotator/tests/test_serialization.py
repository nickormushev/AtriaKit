from ecg_annotator.annotations.state import AnnotationSession, LeadAnnotationState
from ecg_annotator.annotations.serialization import build_annotation_rows

_sample = lambda t: int(t * 1000)  # 1000 Hz identity helper


def _session(leads):
    return AnnotationSession(lead_states=leads, annotation_type="Before", comment="test")


def test_ignored_lead_produces_sentinel_row():
    session = _session([LeadAnnotationState("I", ignored=True)])
    rows = build_annotation_rows(session, "P001", "f.dcm", _sample)
    assert len(rows) == 1
    row = rows[0]
    assert row["ignore"] is True
    assert row["onset"] == row["offset"] == row["qrs_onset"] == -1
    assert row["type"] == "Ignored"
    assert row["lead"] == "I"


def test_normal_annotation_with_qrs_onset():
    lead = LeadAnnotationState(
        "I",
        red_points=[(0.1, 0)],
        green_points=[(0.2, 0)],
        blue_points=[(0.3, 0)],
    )
    session = _session([lead])
    rows = build_annotation_rows(session, "P001", "f.dcm", _sample)
    assert len(rows) == 1
    row = rows[0]
    assert row["onset"] == 100
    assert row["offset"] == 200
    assert row["qrs_onset"] == 300
    assert row["ignore"] is False
    assert row["type"] == "Before"
    assert row["comment"] == "test"


def test_normal_annotation_without_qrs_onset():
    lead = LeadAnnotationState("I", red_points=[(0.1, 0)], green_points=[(0.2, 0)])
    rows = build_annotation_rows(_session([lead]), "P001", "f.dcm", _sample)
    assert rows[0]["qrs_onset"] == -1


def test_multiple_waves_produce_multiple_rows():
    lead = LeadAnnotationState(
        "I",
        red_points=[(0.1, 0), (0.5, 0)],
        green_points=[(0.2, 0), (0.6, 0)],
    )
    rows = build_annotation_rows(_session([lead]), "P001", "f.dcm", _sample)
    assert len(rows) == 2
    assert rows[0]["p_wave_id"] == 0
    assert rows[1]["p_wave_id"] == 1
