from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from atriakit.annotations import Annotations
from atriakit.configs.annotations_loader_config import AnnotationsLoaderConfig
from atriakit.models.annotations import AnnotationSchema
from atriakit.preprocessing.annotations import prepare_annotations


def _compute_multilead(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a per-lead annotation DataFrame with cross-lead boundary columns.

    Saves the original per-lead values into ``onset_original``/``offset_original``,
    then computes the cross-lead min onset and max offset per beat into
    ``onset_multilead``/``offset_multilead``. The active ``onset``/``offset``
    columns are left as per-lead values; ``_apply_boundary_mode`` overwrites them
    if cross-lead mode is requested.
    """
    if df.empty or AnnotationSchema.ONSET_MULTILEAD in df.columns:
        return df

    enriched = df
    enriched[AnnotationSchema.ONSET_ORIGINAL] = enriched[AnnotationSchema.ONSET]
    enriched[AnnotationSchema.OFFSET_ORIGINAL] = enriched[AnnotationSchema.OFFSET]
    if AnnotationSchema.QRS_ONSET in enriched.columns:
        enriched[AnnotationSchema.QRS_ONSET_ORIGINAL] = enriched[
            AnnotationSchema.QRS_ONSET
        ]
        enriched = enriched.drop(columns=[AnnotationSchema.QRS_ONSET])

    group_cols = [AnnotationSchema.FILE_PATH, AnnotationSchema.P_WAVE_ID]
    if AnnotationSchema.TYPE in enriched.columns:
        group_cols.append(AnnotationSchema.TYPE)

    agg: dict[str, str] = {
        AnnotationSchema.ONSET_ORIGINAL: "min",
        AnnotationSchema.OFFSET_ORIGINAL: "max",
    }
    if AnnotationSchema.QRS_ONSET_ORIGINAL in enriched.columns:
        agg[AnnotationSchema.QRS_ONSET_ORIGINAL] = "min"

    multilead = (
        enriched.groupby(group_cols, as_index=False)
        .agg(agg)
        .rename(
            columns={
                AnnotationSchema.ONSET_ORIGINAL: AnnotationSchema.ONSET_MULTILEAD,
                AnnotationSchema.OFFSET_ORIGINAL: AnnotationSchema.OFFSET_MULTILEAD,
                AnnotationSchema.QRS_ONSET_ORIGINAL: AnnotationSchema.QRS_ONSET,
            }
        )
    )

    cols_to_join = [AnnotationSchema.ONSET_MULTILEAD, AnnotationSchema.OFFSET_MULTILEAD]
    if AnnotationSchema.QRS_ONSET in multilead.columns:
        cols_to_join.append(AnnotationSchema.QRS_ONSET)

    return enriched.join(multilead.set_index(group_cols)[cols_to_join], on=group_cols)


def _apply_boundary_mode(
    df: pd.DataFrame, config: AnnotationsLoaderConfig
) -> pd.DataFrame:
    """Copy multilead values into onset/offset when cross_lead mode is active.

    Per-lead is the default — onset/offset already hold per-lead values after
    :func:`_compute_multilead`, so no action is needed in that case.
    """
    if (
        config.boundary_mode == "cross_lead"
        and AnnotationSchema.ONSET_MULTILEAD in df.columns
    ):
        df[AnnotationSchema.ONSET] = df[AnnotationSchema.ONSET_MULTILEAD]
        df[AnnotationSchema.OFFSET] = df[AnnotationSchema.OFFSET_MULTILEAD]
    return df


@runtime_checkable
class BaseAnnotationsLoader(Protocol):
    """Protocol for annotation file loaders: implement ``load(path) -> pd.DataFrame``."""

    def load(self, path: str | Path) -> pd.DataFrame:
        """Read the file at *path* and return a raw annotation DataFrame."""
        ...


class CsvAnnotationsLoader:
    """Loads a single annotation CSV file into a raw DataFrame."""

    def load(self, path: str | Path) -> pd.DataFrame:
        """Read the CSV file at *path* and return a raw annotation DataFrame."""
        return pd.read_csv(path)


class AnnotationsLoader:
    """Loads annotation files and returns an :class:`Annotations` object.

    Routes loading to the appropriate loader by file extension. Handles ``.csv``
    by default; register additional loaders for other formats with :meth:`register`.

    Attributes:
        config: Controls annotation preprocessing (e.g. boundary mode).

    Example::

        ann = AnnotationsLoader().load("annotations.csv")
        ann = AnnotationsLoader().load("annotations_dir/")
        ann = AnnotationsLoader(
            config=AnnotationsLoaderConfig(boundary_mode="per_lead")
        ).load("annotations.csv")
    """

    def __init__(
        self,
        loaders: dict[str, BaseAnnotationsLoader] | None = None,
        config: AnnotationsLoaderConfig | None = None,
    ):
        self._loaders: dict[str, BaseAnnotationsLoader] = loaders or {
            ".csv": CsvAnnotationsLoader(),
        }
        self.config = config or AnnotationsLoaderConfig()

    def register(self, extension: str, loader: BaseAnnotationsLoader) -> None:
        """Register *loader* for files with *extension* (e.g. ``".parquet"``)."""
        self._loaders[extension.lower()] = loader

    def from_dataframe(self, df: pd.DataFrame) -> Annotations:
        """Create an :class:`Annotations` object from an in-memory DataFrame.

        Normalizes file paths, drops duplicates and ignored rows, then applies
        multilead onset/offset enrichment.
        """
        df = prepare_annotations(df)
        df = _compute_multilead(df)
        df = _apply_boundary_mode(df, self.config)
        return Annotations(df)

    def load(self, path: str | Path) -> Annotations:
        """Load annotations from a file or directory and return an :class:`Annotations` object.

        If *path* is a directory, all files with registered extensions are loaded
        and concatenated. Normalizes file paths, drops duplicates and ignored rows,
        then applies multilead onset/offset enrichment.
        """
        path = Path(path)
        if path.is_dir():
            files = sorted(
                f
                for f in path.rglob("*")
                if f.is_file() and f.suffix.lower() in self._loaders
            )
            if not files:
                raise ValueError(
                    f"No annotation files with registered extensions "
                    f"({list(self._loaders)}) found in directory {path}."
                )
            df = pd.concat(
                [self._load_file(f) for f in files],
                ignore_index=True,
            )
        else:
            df = self._load_file(path)

        return self.from_dataframe(df)

    def _load_file(self, path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        loader = self._loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"No loader registered for extension {ext!r}. "
                f"Registered: {list(self._loaders)}. "
                "Use AnnotationsLoader.register() to add support for this format."
            )
        return loader.load(path)
