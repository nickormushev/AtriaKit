from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class AnnotationsLoaderConfig:
    """Configuration for the AnnotationsLoader.

    Parameters
    ----------
    boundary_mode : {"per_lead", "cross_lead"}
        Controls which onset/offset values are treated as active after loading.

        ``"cross_lead"`` (default)
            ``onset`` and ``offset`` hold the multilead consensus values
            (cross-lead minimum onset, cross-lead maximum offset).
            No substitution is performed.

        ``"per_lead"``
            ``onset`` and ``offset`` are replaced by ``onset_original`` and
            ``offset_original`` — the raw per-lead annotation values.
    """

    boundary_mode: Literal["per_lead", "cross_lead"] = "cross_lead"
