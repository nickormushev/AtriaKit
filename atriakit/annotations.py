from __future__ import annotations

from typing import Iterator, cast

import numpy as np
import pandas as pd

from atriakit.models.annotations import AnnotationRow, AnnotationSchema


def _maybe_wrap(result: object) -> object:
    """Return an Annotations if *result* is a full-column DataFrame, else return as-is."""
    if isinstance(result, pd.DataFrame) and not (
        AnnotationSchema.REQUIRED - set(result.columns)
    ):
        return Annotations._wrap(result)
    return result


class _AnnotationsLocIndexer:
    """Proxy for ``df.loc`` that wraps row-slice results as Annotations."""

    def __init__(self, loc) -> None:
        self._loc = loc

    def __getitem__(self, key):
        return _maybe_wrap(self._loc[key])

    def __setitem__(self, key, value) -> None:
        self._loc[key] = value


class _AnnotationsGroupBy:
    """Wraps a pandas DataFrameGroupBy so iteration yields (key, Annotations) pairs."""

    def __init__(self, groupby) -> None:
        self._groupby = groupby

    def __iter__(self) -> Iterator[tuple]:
        for key, group in self._groupby:
            yield key, Annotations._wrap(group)

    def __getattr__(self, attr):
        return getattr(self._groupby, attr)

    def __getitem__(self, key):
        return self._groupby[key]


class Annotations:
    """P-wave annotation set backed by a per-lead DataFrame.

    Requires columns ``onset``, ``offset``, ``lead``, ``file_path``, and
    ``p_wave_id``. Row-preserving operations (``loc``, ``filter_by_lead``,
    ``sort_values``) return a new ``Annotations``; column-reducing operations
    return the raw pandas type.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Annotations expects a pandas DataFrame.")

        missing = AnnotationSchema.REQUIRED - set(df.columns)
        if missing and not df.empty:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}. "
                f"Required: {sorted(AnnotationSchema.REQUIRED)}"
            )

        self._df: pd.DataFrame = df.copy()

        if (
            AnnotationSchema.ONSET in self._df.columns
            and AnnotationSchema.ONSET_ORIGINAL not in self._df.columns
        ):
            self._df[AnnotationSchema.ONSET_ORIGINAL] = self._df[AnnotationSchema.ONSET]
            self._df[AnnotationSchema.OFFSET_ORIGINAL] = self._df[AnnotationSchema.OFFSET]
            if AnnotationSchema.QRS_ONSET in self._df.columns:
                self._df[AnnotationSchema.QRS_ONSET_ORIGINAL] = self._df[AnnotationSchema.QRS_ONSET]

    @classmethod
    def _wrap(cls, df: pd.DataFrame) -> Annotations:
        """Internal trusted constructor: skips copy and validation.

        Caller guarantees *df* is already a fresh DataFrame with all required
        columns present (produced by a domain method from a validated
        ``Annotations._df``).
        """
        obj = object.__new__(cls)
        obj._df = df
        return obj

    @classmethod
    def _from_any(cls, annotations: pd.DataFrame | Annotations) -> Annotations:
        """Internal helper: wrap a raw DataFrame or pass through an existing Annotations."""
        if isinstance(annotations, cls):
            return annotations
        return cls(annotations)

    # ── domain methods ────────────────────────────────────────────────────────

    def iter_rows(self) -> Iterator[AnnotationRow]:
        """Iterate over rows with typed attribute access (``row.onset``, ``row.lead``, etc.)."""
        return cast(Iterator[AnnotationRow], self._df.itertuples(index=False))

    def copy(self) -> Annotations:
        return Annotations._wrap(self._df.copy())

    def sort_values(self, by, **kwargs) -> Annotations:
        return Annotations._wrap(self._df.sort_values(by, **kwargs))

    def filter_by_lead(self, lead: str) -> Annotations:
        """Return a copy of annotations for *lead* only, sorted by onset."""
        df = self._df
        return Annotations._wrap(
            df[df[AnnotationSchema.LEAD] == lead].sort_values(AnnotationSchema.ONSET)
        )

    def vcg_annotations(self) -> Annotations:
        """Return a single-row-per-beat view with the lead relabelled as ``"VCG"``.

        Uses the cross-lead boundary span (``onset_multilead`` / ``offset_multilead``)
        as the onset/offset so the VCG segment always covers all leads.
        """
        if self._df.empty:
            return Annotations._wrap(pd.DataFrame(columns=list(self._df.columns)))
        lead = self._df[AnnotationSchema.LEAD].unique()[0]
        vcg_df = self._df[self._df[AnnotationSchema.LEAD] == lead].copy()
        vcg_df[AnnotationSchema.LEAD] = "VCG"
        if AnnotationSchema.ONSET_MULTILEAD in vcg_df.columns:
            vcg_df[AnnotationSchema.ONSET] = vcg_df[AnnotationSchema.ONSET_MULTILEAD]
            vcg_df[AnnotationSchema.OFFSET] = vcg_df[AnnotationSchema.OFFSET_MULTILEAD]
        if AnnotationSchema.ONSET_ORIGINAL in vcg_df.columns:
            vcg_df[AnnotationSchema.ONSET_ORIGINAL] = np.nan
            vcg_df[AnnotationSchema.OFFSET_ORIGINAL] = np.nan
        if AnnotationSchema.QRS_ONSET_ORIGINAL in vcg_df.columns:
            vcg_df[AnnotationSchema.QRS_ONSET_ORIGINAL] = np.nan
        return Annotations._wrap(vcg_df)

    def groupby(self, by, **kwargs) -> _AnnotationsGroupBy:
        """Return an AnnotationsGroupBy that yields Annotations subgroups on iteration."""
        return _AnnotationsGroupBy(self._df.groupby(by, **kwargs))

    # ── DataFrame proxy ───────────────────────────────────────────────────────

    @property
    def empty(self) -> bool:
        return self._df.empty

    @property
    def columns(self):
        return self._df.columns

    @property
    def index(self):
        return self._df.index

    @property
    def loc(self):
        return _AnnotationsLocIndexer(self._df.loc)

    @property
    def iloc(self):
        return self._df.iloc

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, key):
        return _maybe_wrap(self._df[key])

    def __setitem__(self, key, value) -> None:
        self._df[key] = value

    def itertuples(self, **kwargs):
        return self._df.itertuples(**kwargs)

    def drop(self, *args, inplace: bool = False, **kwargs):
        if inplace:
            self._df.drop(*args, inplace=True, **kwargs)
            return None
        return _maybe_wrap(self._df.drop(*args, **kwargs))

    def to_csv(self, *args, **kwargs):
        return self._df.to_csv(*args, **kwargs)

    def get_hash(self) -> bytes:
        return pd.util.hash_pandas_object(self._df, index=True).values.tobytes()
    
    def get_df(self, copy: bool = False) -> pd.DataFrame:
        return self._df.copy() if copy else self._df

