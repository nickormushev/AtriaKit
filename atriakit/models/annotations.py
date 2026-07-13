from __future__ import annotations

from typing import Protocol


class AnnotationSchema:
    """Column name constants for the annotations DataFrame.

    Required: ``onset``, ``offset``, ``lead``, ``file_path``, ``p_wave_id``.
    """

    ONSET                = "onset"
    OFFSET               = "offset"
    ONSET_ORIGINAL       = "onset_original"
    OFFSET_ORIGINAL      = "offset_original"
    ONSET_MULTILEAD      = "onset_multilead"
    OFFSET_MULTILEAD     = "offset_multilead"
    QRS_ONSET            = "qrs_onset"
    QRS_ONSET_ORIGINAL   = "qrs_onset_original"
    LEAD                 = "lead"
    FILE_PATH            = "file_path"
    P_WAVE_ID            = "p_wave_id"
    PATIENT_ID           = "patient_id"
    TYPE                 = "type"
    IGNORE               = "ignore"

    REQUIRED = frozenset({ONSET, OFFSET, LEAD, FILE_PATH, P_WAVE_ID})


class AnnotationRow(Protocol):
    """Typed row returned by ``Annotations.iter_rows()``."""

    onset:                float
    offset:               float
    onset_original:       float
    offset_original:      float
    qrs_onset:            float
    qrs_onset_original:   float
    lead:                 str
    file_path:            str
    p_wave_id:            int
    patient_id:           str