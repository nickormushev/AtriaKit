from __future__ import annotations

import numpy as np
import pandas as pd

from atriakit.models.annotations import Annotations
from atriakit.models.annotation_schema import AnnotationSchema

_DEDUP_COLUMNS = [
    AnnotationSchema.PATIENT_ID,
    AnnotationSchema.FILE_PATH,
    AnnotationSchema.LEAD,
    AnnotationSchema.ONSET,
    AnnotationSchema.OFFSET,
    AnnotationSchema.TYPE,
]


def prepare_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize, deduplicate, and filter a raw annotation DataFrame.

    Normalizes ``file_path`` path separators to forward slashes, drops
    duplicate rows, and removes rows flagged with ``ignore``.

    Args:
        df: Raw annotation DataFrame, typically loaded from a CSV or directory.

    Returns:
        Cleaned copy of ``df`` ready for multilead enrichment.
    """
    prepared = df.copy()
    if AnnotationSchema.FILE_PATH in prepared.columns:
        prepared[AnnotationSchema.FILE_PATH] = (
            prepared[AnnotationSchema.FILE_PATH]
            .astype(str)
            .str.replace("\\", "/", regex=False)
        )

    duplicate_subset = [col for col in _DEDUP_COLUMNS if col in prepared.columns]
    if duplicate_subset:
        prepared = prepared.drop_duplicates(duplicate_subset)

    if AnnotationSchema.IGNORE in prepared.columns:
        prepared = prepared[~prepared[AnnotationSchema.IGNORE]].drop(
            columns=[AnnotationSchema.IGNORE]
        )
    return prepared


def group_p_waves(
    annotations: Annotations,
    sampling_rate: int,
    tolerance_in_ms: int = 200,
) -> Annotations:
    """Assign a ``p_wave_id`` to annotations that belong to the same beat.

    Annotations whose onsets fall within ``tolerance_in_ms`` of each other are
    considered the same beat and receive the same ``p_wave_id``.

    Args:
        annotations: Per-lead annotation table sorted by file.
        sampling_rate: Sampling frequency in Hz, used to convert the tolerance
            from milliseconds to samples.
        tolerance_in_ms: Maximum onset difference (ms) for two annotations to
            be grouped as the same beat. Defaults to 200 ms.

    Returns:
        A copy of ``annotations`` with the ``p_wave_id`` column populated.
    """
    if annotations.empty:
        return annotations

    grouped = annotations.sort_values(by=AnnotationSchema.ONSET)
    tolerance = (tolerance_in_ms * sampling_rate) / 1000
    p_wave_id = 0
    grouped[AnnotationSchema.P_WAVE_ID] = -1

    for row in grouped.itertuples():
        onset = row.onset
        mask = (grouped[AnnotationSchema.ONSET] - onset).abs() < tolerance

        if grouped.loc[mask, AnnotationSchema.P_WAVE_ID].eq(-1).all():
            grouped.loc[mask, AnnotationSchema.P_WAVE_ID] = p_wave_id
            p_wave_id += 1

    return grouped
