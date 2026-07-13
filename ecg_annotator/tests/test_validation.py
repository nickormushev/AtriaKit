import pytest
from ecg_annotator.annotations.state import AnnotationSession, LeadAnnotationState
from ecg_annotator.annotations.validation import validate_annotation_session

CST = "000000"
THRESH = 0.4


def _session(red=None, green=None, blue=None, ignored=False, annotation_type="Before"):
    return AnnotationSession(
        lead_states=[LeadAnnotationState(
            lead_name="I",
            red_points=red or [],
            green_points=green or [],
            blue_points=blue or [],
            ignored=ignored,
        )],
        annotation_type=annotation_type,
    )


def test_valid_session_passes():
    validate_annotation_session(
        _session(red=[(0.1, 0)], green=[(0.2, 0)]), CST, THRESH
    )


def test_no_annotation_type_raises():
    with pytest.raises(ValueError, match="annotation type"):
        validate_annotation_session(
            _session(red=[(0.1, 0)], green=[(0.2, 0)], annotation_type=None), CST, THRESH
        )


def test_all_leads_ignored_bypasses_type_and_points_check():
    # Neither annotation type nor points required when every lead is ignored
    validate_annotation_session(_session(ignored=True, annotation_type=None), CST, THRESH)


def test_empty_points_raises():
    with pytest.raises(ValueError, match="requires at least one"):
        validate_annotation_session(_session(red=[], green=[(0.2, 0)]), CST, THRESH)


def test_unmapped_red_raises():
    # Two reds, one green: gap rule leaves the first red unmapped
    with pytest.raises(ValueError, match="red"):
        validate_annotation_session(
            _session(red=[(0.1, 0), (0.2, 0)], green=[(0.3, 0)]), CST, THRESH
        )


def test_distance_too_large_raises():
    with pytest.raises(ValueError, match="too large"):
        validate_annotation_session(
            _session(red=[(0.0, 0)], green=[(0.5, 0)]), CST, THRESH  # 0.5 >= 0.4
        )


def test_inconsistent_pair_count_raises():
    session = AnnotationSession(
        lead_states=[
            LeadAnnotationState("I",  red_points=[(0.1, 0), (0.3, 0)], green_points=[(0.2, 0), (0.35, 0)]),
            LeadAnnotationState("II", red_points=[(0.1, 0)],            green_points=[(0.2, 0)]),
        ],
        annotation_type="Before",
    )
    with pytest.raises(ValueError, match="inconsistent"):
        validate_annotation_session(session, CST, THRESH)
