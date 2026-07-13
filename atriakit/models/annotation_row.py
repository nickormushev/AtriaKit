from __future__ import annotations

from typing import Protocol


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
