import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atriakit import FeatureComputationConfig, Pipeline, PipelineConfig
from atriakit.models.annotations import Annotations
from atriakit.configs.signal_preprocessor_config import SignalPreprocessorConfig
from atriakit.feature_calculator import FeatureCalculators
from atriakit.models.annotation_schema import AnnotationSchema
from atriakit.models.ecg_data import ECGData
from atriakit.io import AnnotationsLoader, BaseLoader

_STD_12 = {"I":0,"II":1,"III":2,"aVR":3,"aVL":4,"aVF":5,"V1":6,"V2":7,"V3":8,"V4":9,"V5":10,"V6":11}
from atriakit.preprocessing.signals import SignalPreprocessor
import atriakit.processing.pipeline as pipeline_module


def _ann(df: pd.DataFrame) -> Annotations:
    """Wrap a test DataFrame in Annotations via the loader."""
    return AnnotationsLoader().from_dataframe(df)


class _FakeLoader:
    """Loader that returns a minimal ECGData without touching the filesystem."""

    def __init__(self, raise_not_found=False):
        self.loaded_paths = []
        self._raise = raise_not_found

    def load(self, path) -> ECGData:
        self.loaded_paths.append(path)
        if self._raise:
            raise FileNotFoundError(path)
        return ECGData(ecg=np.ones((12, 100)), fs=500, lead_to_index=_STD_12)


def test_pipeline_run_batches_by_file_and_type(monkeypatch, tmp_path):
    annotations = _ann(
        pd.DataFrame(
            {
                AnnotationSchema.FILE_PATH: ["record1.IMA", "record1.IMA", "record1.IMA"],
                AnnotationSchema.LEAD: ["I", "II", "I"],
                AnnotationSchema.ONSET: [0, 5, 50],
                AnnotationSchema.OFFSET: [5, 10, 55],
                AnnotationSchema.TYPE: ["Before", "Before", "After"],
                AnnotationSchema.P_WAVE_ID: [1, 1, 2],
            }
        )
    )

    fake_loader = _FakeLoader()

    pipeline = Pipeline(
        ecg_base_path=tmp_path,
        loader=fake_loader,
        pipeline_config=PipelineConfig(
            feature_computation=FeatureComputationConfig(
                shannon_entropy_n_bins=128,
                shannon_entropy_bin_range=(-2.0, 2.0),
                sample_entropy_m=3,
                sample_entropy_r_factor=0.5,
                extrema_threshold_multiplier=0.2,
            ),
            signal_preprocessor_config=SignalPreprocessorConfig(
                lowcut=1,
                highcut=40,
                normalization_type="zscore",
            ),
        ),
    )

    pipeline._generate_mean_and_std = lambda preprocessor, annotations, save_path=None: (
        np.array([preprocessor.config.lowcut]),
        np.array([preprocessor.config.highcut]),
    )

    compute_calls = []

    def fake_compute_all(self, annotation_group, ecg_data, feature_computation_config):
        assert isinstance(annotation_group, Annotations)
        compute_calls.append(
            {
                "extrema_threshold_multiplier": feature_computation_config.extrema_threshold_multiplier,
                "indices": list(annotation_group.index),
                "n_bins": feature_computation_config.shannon_entropy_n_bins,
                "bin_range": feature_computation_config.shannon_entropy_bin_range,
                "sample_entropy_m": feature_computation_config.sample_entropy_m,
                "sample_entropy_r_factor": feature_computation_config.sample_entropy_r_factor,
                "signal_preprocessor_mean": self.signal_preprocessor.config.mean,
                "signal_preprocessor_std": self.signal_preprocessor.config.std,
            }
        )
        return pd.DataFrame(
            {"dummy_feature": np.arange(len(annotation_group.index))},
            index=annotation_group.index,
        )

    monkeypatch.setattr(FeatureCalculators, "compute_all", fake_compute_all)

    result = pipeline.run(annotations)

    assert len(fake_loader.loaded_paths) == 1
    assert fake_loader.loaded_paths[0].name == "record1.IMA"
    assert np.allclose(compute_calls[0]["signal_preprocessor_mean"], np.array([1]))
    assert np.allclose(compute_calls[0]["signal_preprocessor_std"], np.array([40]))
    assert len(compute_calls) == 2
    assert compute_calls[0]["extrema_threshold_multiplier"] == 0.2
    assert compute_calls[0]["n_bins"] == 128
    assert compute_calls[0]["bin_range"] == (-2.0, 2.0)
    assert compute_calls[0]["sample_entropy_m"] == 3
    assert compute_calls[0]["sample_entropy_r_factor"] == 0.5
    assert "dummy_feature" in result.columns
    assert result.loc[0, AnnotationSchema.P_WAVE_ID] == 0
    assert result.loc[1, AnnotationSchema.P_WAVE_ID] == 0
    assert result.loc[2, AnnotationSchema.P_WAVE_ID] == 0


def test_ecg_public_import_exposes_pipeline():
    from atriakit import Pipeline as PublicPipeline

    assert PublicPipeline is Pipeline


def test_pipeline_logs_missing_files(tmp_path, caplog):
    annotations = _ann(
        pd.DataFrame(
            {
                AnnotationSchema.FILE_PATH: ["missing.IMA"],
                AnnotationSchema.LEAD: ["I"],
                AnnotationSchema.ONSET: [0],
                AnnotationSchema.OFFSET: [5],
                AnnotationSchema.P_WAVE_ID: [1],
            }
        )
    )

    pipeline = Pipeline(
        ecg_base_path=tmp_path,
        loader=_FakeLoader(raise_not_found=True),
    )
    pipeline._generate_mean_and_std = lambda preprocessor, annotations, save_path=None: (
        np.array([preprocessor.config.lowcut]),
        np.array([preprocessor.config.highcut]),
    )

    with caplog.at_level(logging.WARNING, logger="atriakit.processing.pipeline"):
        result = pipeline.run(annotations)

    assert result.empty
    assert "File not found" in caplog.text
    assert "Dropped 1 annotations" in caplog.text


def test_pipeline_run_reuses_cached_normalization_stats(monkeypatch, tmp_path):
    annotations = _ann(
        pd.DataFrame(
            {
                AnnotationSchema.FILE_PATH: ["record1.IMA", "record1.IMA"],
                AnnotationSchema.LEAD: ["I", "II"],
                AnnotationSchema.ONSET: [0, 5],
                AnnotationSchema.OFFSET: [5, 10],
                AnnotationSchema.TYPE: ["Before", "Before"],
                AnnotationSchema.P_WAVE_ID: [1, 1],
            }
        )
    )

    normalization_calls = []

    def fake_calculate_mean_std_p_waves(self, annotations, preprocessor=None):
        normalization_calls.append(
            (preprocessor.config.lowcut, preprocessor.config.highcut)
        )
        return (
            np.full(12, preprocessor.config.lowcut, dtype=float),
            np.full(12, preprocessor.config.highcut, dtype=float),
        )

    monkeypatch.setattr(
        pipeline_module.ECGDataset,
        "calculate_mean_std_p_waves",
        fake_calculate_mean_std_p_waves,
    )

    def fake_compute_all(self, annotation_group, ecg_data, feature_computation_config):
        return pd.DataFrame(
            {"dummy_feature": np.arange(len(annotation_group.index))},
            index=annotation_group.index,
        )

    monkeypatch.setattr(FeatureCalculators, "compute_all", fake_compute_all)

    cache_dir = tmp_path / "cache"
    pipeline_config = PipelineConfig(
        signal_preprocessor_config=SignalPreprocessorConfig(
            lowcut=1, highcut=40, normalization_type="zscore"
        ),
        morphology_preprocessor_config=SignalPreprocessorConfig(
            lowcut=1, highcut=30, normalization_type="zscore"
        ),
    )

    first_pipeline = Pipeline(
        ecg_base_path=tmp_path,
        loader=_FakeLoader(),
        pipeline_config=pipeline_config,
    )
    first_pipeline.run(annotations, cache_dir=cache_dir)

    assert len(normalization_calls) == 2
    assert len(list(cache_dir.glob("*.npz"))) == 2

    second_pipeline = Pipeline(
        ecg_base_path=tmp_path,
        loader=_FakeLoader(),
        pipeline_config=pipeline_config,
    )
    second_pipeline.run(annotations, cache_dir=cache_dir)

    assert len(normalization_calls) == 2


def test_pipeline_run_skips_normalization_when_disabled_in_preprocessor_config(
    monkeypatch, tmp_path,
):
    annotations = _ann(
        pd.DataFrame(
            {
                AnnotationSchema.FILE_PATH: ["record1.IMA"],
                AnnotationSchema.LEAD: ["I"],
                AnnotationSchema.ONSET: [0],
                AnnotationSchema.OFFSET: [5],
                AnnotationSchema.P_WAVE_ID: [1],
            }
        )
    )

    pipeline = Pipeline(
        ecg_base_path=tmp_path,
        loader=_FakeLoader(),
        pipeline_config=PipelineConfig(
            signal_preprocessor_config=SignalPreprocessorConfig(
                lowcut=1, highcut=40, normalization_type="none"
            ),
            morphology_preprocessor_config=SignalPreprocessorConfig(
                lowcut=1, highcut=30, normalization_type="none"
            ),
        ),
    )

    pipeline._generate_mean_and_std = lambda *args, **kwargs: pytest.fail(
        "Normalization stats should not be computed when normalization is disabled."
    )

    captured = []

    def fake_compute_all(self, *args, **kwargs):
        captured.append((self.signal_preprocessor.config.mean, self.signal_preprocessor.config.std))
        return pd.DataFrame({"dummy_feature": [0]}, index=[0])

    monkeypatch.setattr(FeatureCalculators, "compute_all", fake_compute_all)

    result = pipeline.run(annotations, cache_dir=tmp_path / "cache")

    assert "dummy_feature" in result.columns
    assert captured[0][0] is None
    assert captured[0][1] is None


def test_pipeline_run_does_not_mutate_caller_annotations(monkeypatch, tmp_path):
    annotations = _ann(
        pd.DataFrame(
            {
                AnnotationSchema.FILE_PATH: ["record1.IMA"],
                AnnotationSchema.LEAD: ["I"],
                AnnotationSchema.ONSET: [0],
                AnnotationSchema.OFFSET: [5],
                AnnotationSchema.P_WAVE_ID: [1],
            }
        )
    )
    columns_before = list(annotations.columns)
    onset_before = list(annotations._df[AnnotationSchema.ONSET])

    pipeline = Pipeline(
        ecg_base_path=tmp_path,
        loader=_FakeLoader(),
        pipeline_config=PipelineConfig(
            signal_preprocessor_config=SignalPreprocessorConfig(
                lowcut=1, highcut=40, normalization_type="none"
            ),
            morphology_preprocessor_config=SignalPreprocessorConfig(
                lowcut=1, highcut=30, normalization_type="none"
            ),
        ),
    )
    monkeypatch.setattr(
        FeatureCalculators,
        "compute_all",
        lambda self, *args, **kwargs: pd.DataFrame({"dummy_feature": [0]}, index=[0]),
    )

    pipeline.run(annotations)

    assert list(annotations.columns) == columns_before
    assert list(annotations._df[AnnotationSchema.ONSET]) == onset_before


def test_pipeline_from_yaml_loads_relative_paths_and_configs(tmp_path):
    annotations_csv = tmp_path / "annotations.csv"
    annotations = pd.DataFrame(
        {
            AnnotationSchema.FILE_PATH: ["record1.IMA"],
            AnnotationSchema.LEAD: ["I"],
            AnnotationSchema.ONSET: [0],
            AnnotationSchema.OFFSET: [5],
            AnnotationSchema.P_WAVE_ID: [1],
        }
    )
    annotations.to_csv(annotations_csv, index=False)
    (tmp_path / "dicoms").mkdir()

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ecg_base_path: dicoms",
                "annotations_csv: annotations.csv",
                "debug: true",
                "pipeline:",
                "  group_tolerance_ms: 123",
                "  signal_preprocessor_config:",
                "    lowcut: 2",
                "    highcut: 30",
                "    normalization_type: none",
                "  feature_computation:",
                "    shannon_entropy_n_bins: 99",
            ]
        ),
        encoding="utf-8",
    )

    pipeline, _ = Pipeline.from_yaml(config_path)

    assert pipeline.ecg_base_path == (tmp_path / "dicoms").resolve()
    assert pipeline.debug is True
    assert pipeline.pipeline_config.group_tolerance_ms == 123
    assert pipeline.pipeline_config.signal_preprocessor_config.lowcut == 2
    assert pipeline.pipeline_config.signal_preprocessor_config.highcut == 30
    assert pipeline.pipeline_config.signal_preprocessor_config.normalization_type == "none"
    assert pipeline.pipeline_config.feature_computation.shannon_entropy_n_bins == 99


def test_pipeline_run_from_yaml(tmp_path):
    annotations_csv = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            AnnotationSchema.FILE_PATH: ["record1.IMA"],
            AnnotationSchema.LEAD: ["I"],
            AnnotationSchema.ONSET: [0],
            AnnotationSchema.OFFSET: [5],
            AnnotationSchema.P_WAVE_ID: [1],
        }
    ).to_csv(annotations_csv, index=False)
    (tmp_path / "dicoms").mkdir()

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ecg_base_path: dicoms",
                "annotations_csv: annotations.csv",
                "pipeline:",
                "  signal_preprocessor_config:",
                "    lowcut: 1",
                "    highcut: 40",
                "    normalization_type: none",
                "  morphology_preprocessor_config:",
                "    lowcut: 1",
                "    highcut: 30",
                "    normalization_type: none",
            ]
        ),
        encoding="utf-8",
    )

    pipeline_kwargs, annotations, _ = Pipeline._pipeline_kwargs_from_yaml(config_path)
    assert len(annotations) == 1
    pipeline = Pipeline(**pipeline_kwargs)
    assert pipeline.ecg_base_path == (tmp_path / "dicoms").resolve()
