from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class SegmentConfig:
    """Segment-boundary and baseline-correction settings for ``SegmentProcessor``."""

    baseline_correction_type: Literal["none", "linear", "onset"] = "none"  # Baseline correction applied to extracted segments before feature calculation.
    skip_first_ms: int = 0  # Drops the first samples of each annotated segment before feature extraction.
