import hashlib
import logging
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from atriakit.configs.annotations_loader_config import AnnotationsLoaderConfig
from atriakit.configs.feature_computation_config import FeatureComputationConfig
from atriakit.configs.pipeline_config import PipelineConfig
from atriakit.configs.segment_config import SegmentConfig
from atriakit.configs.signal_preprocessor_config import SignalPreprocessorConfig
from atriakit.feature_calculator import FeatureCalculators
from atriakit.io import AnnotationsLoader, ECGLoader
from atriakit.models.annotation_schema import AnnotationSchema
from atriakit.models.annotations import Annotations
from atriakit.models.ecg_data import ECGData
from atriakit.preprocessing import group_p_waves
from atriakit.preprocessing.signals import SignalPreprocessor
from atriakit.processing.dataset import ECGDataset

LOGGER = logging.getLogger(__name__)


class Pipeline:
    """Batch-processing pipeline for computing P-wave features across ECG files.

    Args:
        ecg_base_path: Base directory containing the ECG files referenced by
            the annotation table. Defaults to the current directory.
        pipeline_config: Batch pipeline parameters, entropy/complexity/noise/
            morphology/fragmentation settings, segment-boundary/baseline
            correction settings, preprocessing filters, and beat-grouping
            tolerance. Defaults to ``PipelineConfig()``.
        loader: ECG file loader. Defaults to ``ECGLoader()``.
        debug: Emit additional debug logging. Defaults to ``False``.
        logger: Logger instance. Defaults to the module logger.

    Attributes:
        ecg_base_path: Resolved base directory for ECG files.
        pipeline_config: Active pipeline config.
    """

    def __init__(
        self,
        *,
        ecg_base_path: str | Path = ".",
        pipeline_config: PipelineConfig | None = None,
        loader: ECGLoader | None = None,
        debug: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.ecg_base_path = Path(ecg_base_path)
        self.loader = loader or ECGLoader()
        self.debug = debug
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.logger = logger or LOGGER

    @staticmethod
    def _load_yaml_mapping(config_path: str | Path) -> tuple[dict, Path]:
        resolved_config_path = Path(config_path).expanduser().resolve()
        with resolved_config_path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)

        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"YAML config at {resolved_config_path} must contain a mapping."
            )

        return loaded, resolved_config_path.parent

    @staticmethod
    def _validate_mapping_keys(
        mapping: dict,
        *,
        section_name: str,
        allowed_keys: set[str],
    ) -> None:
        unknown_keys = set(mapping) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown keys in '{section_name}' config: {unknown}.")

    @classmethod
    def _resolve_nested_config(
        cls,
        section: dict,
        key: str,
        config_cls: type,
        section_name: str,
    ) -> None:
        """Replace ``section[key]`` in place with a ``config_cls`` instance, if present."""
        if key not in section:
            return
        value = section[key]
        if not isinstance(value, dict):
            raise ValueError(f"'{section_name}' config must be a mapping.")
        cls._validate_mapping_keys(
            value,
            section_name=section_name,
            allowed_keys={field.name for field in fields(config_cls)},
        )
        section[key] = config_cls(**value)

    @staticmethod
    def _resolve_optional_path(
        value: str | Path | None,
        *,
        config_dir: Path,
    ) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        return path

    @classmethod
    def _pipeline_kwargs_from_yaml(
        cls,
        config_path: str | Path,
    ) -> tuple[dict, Annotations, dict]:
        config, config_dir = cls._load_yaml_mapping(config_path)
        cls._validate_mapping_keys(
            config,
            section_name="top-level",
            allowed_keys={
                "annotations",
                "annotations_csv",
                "annotations_dir",
                "cache_dir",
                "debug",
                "ecg_base_path",
                "pipeline",
                "run",
                "save_path",
            },
        )

        annotations_section = config.get("annotations") or {}
        pipeline_section = dict(config.get("pipeline") or {})
        run_section = config.get("run") or {}

        for section_name, section_value in (
            ("annotations", annotations_section),
            ("pipeline", pipeline_section),
            ("run", run_section),
        ):
            if not isinstance(section_value, dict):
                raise ValueError(f"'{section_name}' config must be a mapping.")

        cls._validate_mapping_keys(
            annotations_section,
            section_name="annotations",
            allowed_keys={field.name for field in fields(AnnotationsLoaderConfig)},
        )
        cls._validate_mapping_keys(
            pipeline_section,
            section_name="pipeline",
            allowed_keys={field.name for field in fields(PipelineConfig)},
        )
        cls._resolve_nested_config(
            pipeline_section,
            "feature_computation",
            FeatureComputationConfig,
            "pipeline.feature_computation",
        )
        cls._resolve_nested_config(
            pipeline_section,
            "segment_config",
            SegmentConfig,
            "pipeline.segment_config",
        )
        cls._resolve_nested_config(
            pipeline_section,
            "signal_preprocessor_config",
            SignalPreprocessorConfig,
            "pipeline.signal_preprocessor_config",
        )
        cls._resolve_nested_config(
            pipeline_section,
            "morphology_preprocessor_config",
            SignalPreprocessorConfig,
            "pipeline.morphology_preprocessor_config",
        )
        cls._validate_mapping_keys(
            run_section,
            section_name="run",
            allowed_keys={"cache_dir", "save_path"},
        )

        ecg_base_path = cls._resolve_optional_path(
            config.get("ecg_base_path"),
            config_dir=config_dir,
        )
        if ecg_base_path is None:
            raise ValueError("'ecg_base_path' is required in the YAML config.")

        annotations_dir = cls._resolve_optional_path(
            config.get("annotations_dir"),
            config_dir=config_dir,
        )
        annotations_csv = cls._resolve_optional_path(
            config.get("annotations_csv"),
            config_dir=config_dir,
        )
        if (annotations_dir is None) == (annotations_csv is None):
            raise ValueError(
                "Provide exactly one of 'annotations_dir' or 'annotations_csv' in the YAML config."
            )

        annotations_loader_config = AnnotationsLoaderConfig(**annotations_section)
        annotations_path = (
            annotations_dir if annotations_dir is not None else annotations_csv
        )
        annotations = AnnotationsLoader(config=annotations_loader_config).load(
            annotations_path
        )

        pipeline_kwargs = {
            "ecg_base_path": ecg_base_path,
            "pipeline_config": PipelineConfig(**pipeline_section),
            "debug": config.get("debug", False),
        }

        run_kwargs = {
            "save_path": cls._resolve_optional_path(
                run_section.get("save_path", config.get("save_path")),
                config_dir=config_dir,
            ),
            "cache_dir": cls._resolve_optional_path(
                run_section.get("cache_dir", config.get("cache_dir")),
                config_dir=config_dir,
            ),
        }

        return pipeline_kwargs, annotations, run_kwargs

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Pipeline":
        """Construct a Pipeline and load annotations from a YAML config file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Tuple of ``(pipeline, annotations)``.

        Raises:
            ValueError: If the YAML is malformed, contains unknown config
                keys, is missing ``ecg_base_path``, or does not specify
                exactly one of ``annotations_dir``/``annotations_csv``.
        """
        pipeline_kwargs, annotations, _ = cls._pipeline_kwargs_from_yaml(config_path)
        return cls(**pipeline_kwargs), annotations

    @classmethod
    def run_from_yaml(
        cls,
        config_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """Construct a Pipeline from a YAML config file and run it immediately.

        Convenience wrapper around ``from_yaml`` + ``run``.

        Args:
            config_path: Path to the YAML configuration file.
            cache_dir: Directory for caching normalization statistics. Overrides
                the value in the YAML file when provided.

        Returns:
            Annotations with feature columns added.

        Raises:
            ValueError: If the YAML is malformed, contains unknown config
                keys, is missing ``ecg_base_path``, or does not specify
                exactly one of ``annotations_dir``/``annotations_csv``.
        """
        pipeline_kwargs, annotations, run_kwargs = cls._pipeline_kwargs_from_yaml(
            config_path
        )
        pipeline = cls(**pipeline_kwargs)
        return pipeline.run(
            annotations,
            cache_dir=cache_dir or run_kwargs.get("cache_dir", ".ecg_cache"),
        )

    def _generate_mean_and_std(
        self,
        preprocessor: SignalPreprocessor,
        annotations: Annotations,
        save_path=None,
    ):
        if save_path is not None and Path(save_path).exists():
            data = np.load(save_path)
            self.logger.info(
                "Loaded cached normalization statistics from %s.", save_path
            )
            return data["mean"], data["std"]

        self.logger.info(
            "Computing normalization statistics with lowcut=%s and highcut=%s.",
            preprocessor.config.lowcut,
            preprocessor.config.highcut,
        )
        ecg_dataset = ECGDataset(
            dataset_dir=str(self.ecg_base_path),
            loader=self.loader,
        )
        mean, std = ecg_dataset.calculate_mean_std_p_waves(
            annotations, preprocessor=preprocessor
        )

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, mean=mean, std=std)
            self.logger.info("Saved normalization statistics to %s.", save_path)

        self.logger.info(
            "Finished normalization statistics with lowcut=%s and highcut=%s.",
            preprocessor.config.lowcut,
            preprocessor.config.highcut,
        )
        return mean, std

    def _prepare_preprocessor(
        self,
        preprocessor: SignalPreprocessor,
        annotations: Annotations,
        *,
        save_path: str | Path | None = None,
    ) -> SignalPreprocessor:
        if preprocessor.config.mean is not None and preprocessor.config.std is not None:
            return preprocessor
        mean, std = self._generate_mean_and_std(
            preprocessor,
            annotations,
            save_path=save_path,
        )
        return preprocessor.with_overrides(mean=mean, std=std)

    def _normalization_cache_path(
        self,
        preprocessor: SignalPreprocessor,
        preprocessor_name: str,
        annotations: Annotations,
        *,
        cache_dir: str | Path,
    ) -> Path:
        annotations_hash = annotations.get_hash()
        digest = hashlib.sha256()
        digest.update(str(self.ecg_base_path.resolve()).encode("utf-8"))
        digest.update(preprocessor_name.encode("utf-8"))
        digest.update(repr(preprocessor.config).encode("utf-8"))
        digest.update(annotations_hash)
        return Path(cache_dir) / f"{preprocessor_name}_{digest.hexdigest()[:16]}.npz"

    def _load_ecg(self, file_path: str) -> ECGData:
        return self.loader.load(self.ecg_base_path / file_path)

    def _iter_feature_groups(self, annotations: Annotations):
        """Yield (file_path, [type_subgroup, ...]) for each ECG file."""
        for file_path, file_annotations in annotations.groupby(
            AnnotationSchema.FILE_PATH, sort=False
        ):
            if AnnotationSchema.TYPE not in file_annotations.columns:
                yield file_path, [file_annotations]
                continue

            type_groups = [
                subgroup
                for _, subgroup in file_annotations.groupby(
                    AnnotationSchema.TYPE, sort=False
                )
            ]
            yield file_path, type_groups

    def run(
        self,
        annotations: Annotations,
        *,
        cache_dir: str | Path | None = ".ecg_cache",
    ) -> Annotations:
        """Compute all features for each annotation and return an enriched table.

        Iterates over ECG files referenced in ``annotations``, loads each file,
        groups annotations into beats, and calls ``compute_all`` for each group.
        Features are written into new columns on the returned ``Annotations``.

        Args:
            annotations: Input annotation table (one row per lead per beat).
            cache_dir: Directory for caching preprocessor normalization statistics
                across runs. Pass ``None`` to disable caching.

        Returns:
            A copy of ``annotations`` with feature columns added.
        """
        self.logger.info(
            "Starting pipeline run for %d annotations across %d files.",
            len(annotations),
            annotations[AnnotationSchema.FILE_PATH].nunique(),
        )

        morphology_preprocessor = SignalPreprocessor(
            self.pipeline_config.morphology_preprocessor_config
        )
        if morphology_preprocessor.uses_normalization():
            self.logger.info("Preparing morphology normalization statistics.")
            morphology_preprocessor = self._prepare_preprocessor(
                morphology_preprocessor,
                annotations,
                save_path=(
                    None
                    if cache_dir is None
                    else self._normalization_cache_path(
                        morphology_preprocessor,
                        "morphology_preprocessor",
                        annotations,
                        cache_dir=cache_dir,
                    )
                ),
            )
            self.logger.info("Morphology normalization statistics are ready.")
        else:
            self.logger.info(
                "Skipping morphology normalization because normalization_type is %s.",
                morphology_preprocessor.config.normalization_type,
            )

        signal_preprocessor = SignalPreprocessor(self.pipeline_config.signal_preprocessor_config)
        if signal_preprocessor.uses_normalization():
            self.logger.info("Preparing feature normalization statistics.")
            signal_preprocessor = self._prepare_preprocessor(
                signal_preprocessor,
                annotations,
                save_path=(
                    None
                    if cache_dir is None
                    else self._normalization_cache_path(
                        signal_preprocessor,
                        "signal_preprocessor",
                        annotations,
                        cache_dir=cache_dir,
                    )
                ),
            )
            self.logger.info("Feature normalization statistics are ready.")
        else:
            self.logger.info(
                "Skipping feature normalization because normalization_type is %s.",
                signal_preprocessor.config.normalization_type,
            )

        feature_calculators = FeatureCalculators(
            self.pipeline_config.segment_config,
            signal_preprocessor=signal_preprocessor,
            morphology_preprocessor=morphology_preprocessor,
        )

        working = annotations.copy()
        skipped_indices = []
        iterator = self._iter_feature_groups(annotations)
        self.logger.info(
            "Starting per-file feature extraction over %d file(s).",
            annotations[AnnotationSchema.FILE_PATH].nunique(),
        )
        iterator = tqdm(
            iterator, total=annotations[AnnotationSchema.FILE_PATH].nunique()
        )

        for file_path, subgroups in iterator:
            self.logger.info(
                "Processing file %s with %d subgroup(s).",
                file_path,
                len(subgroups),
            )
            if self.debug:
                subgroup_count = sum(len(subgroup) for subgroup in subgroups)
                self.logger.debug(
                    "Processing %s with %d subgroup(s) and %d annotations.",
                    file_path,
                    len(subgroups),
                    subgroup_count,
                )
            try:
                ecg = self._load_ecg(file_path)
            except FileNotFoundError:
                self.logger.warning(
                    "File not found: %s. Skipping associated annotations.",
                    self.ecg_base_path / file_path,
                )
                skipped_indices.extend(
                    annotations[
                        annotations[AnnotationSchema.FILE_PATH] == file_path
                    ].index
                )
                continue

            sampling_rate = ecg.get_sampling_frequency()
            for subgroup_idx, subgroup in enumerate(subgroups, start=1):
                self.logger.info(
                    "Computing subgroup %d/%d for %s (%d annotation(s)).",
                    subgroup_idx,
                    len(subgroups),
                    file_path,
                    len(subgroup),
                )
                prepared_subgroup = group_p_waves(
                    subgroup,
                    sampling_rate,
                    tolerance_in_ms=self.pipeline_config.group_tolerance_ms,
                )
                working.loc[prepared_subgroup.index, prepared_subgroup.columns] = (
                    prepared_subgroup.get_df()
                )
                features = feature_calculators.compute_all(
                    prepared_subgroup,
                    ecg,
                    self.pipeline_config.feature_computation,
                )
                working.loc[prepared_subgroup.index, features.columns] = features

        if skipped_indices:
            working.drop(skipped_indices, inplace=True)
            self.logger.warning(
                "Dropped %d annotations because their ECG files could not be loaded.",
                len(skipped_indices),
            )

        self.logger.info(
            "Finished pipeline run with %d annotations in the result.",
            len(working),
        )

        return working
