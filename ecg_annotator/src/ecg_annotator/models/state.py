import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from ecg_annotator.models.recording import ECGRecording


log = logging.getLogger(__name__)

Point = tuple[float, float]


@dataclass
class LeadAnnotationState:
    lead_name: str
    red_points: list[Point] = field(default_factory=list)
    green_points: list[Point] = field(default_factory=list)
    blue_points: list[Point] = field(default_factory=list)
    ignored: bool = False
    confidence: str = "High"


def _add_points_from_ann(
    lead_state: LeadAnnotationState,
    ann,
    ecg: ECGRecording,
    lead_idx: int,
    earliest: float,
) -> float:
    def add_if_valid(points_list: list[Point], idx: int, color_name: str) -> bool:
        if 0 <= idx < ecg.ecg_mV.shape[1]:
            y = ecg.ecg_mV[lead_idx, idx]
            points_list.append((idx / ecg.fs_Hz, y))
            return True
        elif idx != -1:
            log.warning(
                "ann.%s %d out of bounds for signals with shape %s",
                color_name, idx, ecg.ecg_mV.shape,
            )
        return False

    if add_if_valid(lead_state.red_points, ann.onset, "onset"):
        earliest = min(earliest, ann.onset)
    add_if_valid(lead_state.green_points, ann.offset, "offset")
    if ann.qrs_onset != -1:
        add_if_valid(lead_state.blue_points, ann.qrs_onset, "qrs_onset")

    return earliest


@dataclass
class AnnotationSession:
    lead_states: list[LeadAnnotationState]
    comment: str = ""
    annotation_type: str | None = None

    @classmethod
    def for_leads(cls, lead_names: list[str], default_confidence: str = "High") -> "AnnotationSession":
        return cls(
            lead_states=[
                LeadAnnotationState(lead_name=name, confidence=default_confidence)
                for name in lead_names
            ]
        )

    @classmethod
    def from_annotations(
        cls,
        anns: pd.DataFrame,
        lead_names: list[str],
        lead_name_to_index: dict[str, int],
        ecg: ECGRecording,
        default_confidence: str = "High",
    ) -> tuple["AnnotationSession", float]:
        """Build a session hydrated from saved CSV annotation rows.

        Returns (session, earliest_onset_sample) where earliest is math.inf if no points.
        """
        session = cls.for_leads(lead_names, default_confidence=default_confidence)
        if anns.empty:
            return session, math.inf

        comment = anns.iloc[0].comment
        if comment is not None and not pd.isna(comment):
            session.comment = comment

        earliest = math.inf
        for ann in anns.itertuples():
            lead_idx = lead_name_to_index.get(ann.lead)
            if lead_idx is None:
                continue

            lead_state = session.lead_states[lead_idx]
            if ann.confidence is not None and not pd.isna(ann.confidence):
                lead_state.confidence = ann.confidence
            if ann.type is not None and not pd.isna(ann.type) and ann.type != "Ignored":
                session.annotation_type = ann.type

            if ann.ignore:
                lead_state.ignored = True
                continue

            earliest = _add_points_from_ann(lead_state, ann, ecg, lead_idx, earliest)

        return session, earliest
