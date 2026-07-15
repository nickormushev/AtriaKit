"""Public package interface for the ECG feature library."""

from importlib import import_module

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("atriakit")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "create_demo_data",
    "AnnotationRow",
    "Annotations",
    "AnnotationsLoaderConfig",
    "AnnotationsLoader",
    "BaseLoader",
    "DicomLoader",
    "ECGData",
    "ECGLoader",
    "FeatureCalculatorConfig",
    "FeatureCalculators",
    "Pipeline",
    "PipelineConfig",
    "SegmentConfig",
    "SignalPreprocessor",
    "SignalPreprocessorConfig",
]

_PUBLIC_IMPORTS = {
    "create_demo_data": ("atriakit.datasets", "generate"),
    "AnnotationRow": ("atriakit.models.annotation_row", "AnnotationRow"),
    "Annotations": ("atriakit.models.annotations", "Annotations"),
    "AnnotationsLoaderConfig": ("atriakit.configs.annotations_loader_config", "AnnotationsLoaderConfig"),
    "AnnotationsLoader": ("atriakit.io", "AnnotationsLoader"),
    "BaseLoader": ("atriakit.io", "BaseLoader"),
    "DicomLoader": ("atriakit.io", "DicomLoader"),
    "ECGData": ("atriakit.models.ecg_data", "ECGData"),
    "ECGLoader": ("atriakit.io", "ECGLoader"),
    "FeatureCalculatorConfig": (
        "atriakit.configs.feature_calculator_config",
        "FeatureCalculatorConfig",
    ),
    "FeatureCalculators": ("atriakit.feature_calculator", "FeatureCalculators"),
    "Pipeline": ("atriakit.processing.pipeline", "Pipeline"),
    "PipelineConfig": ("atriakit.configs.pipeline_config", "PipelineConfig"),
    "SegmentConfig": ("atriakit.configs.segment_config", "SegmentConfig"),
    "SignalPreprocessor": ("atriakit.preprocessing", "SignalPreprocessor"),
    "SignalPreprocessorConfig": (
        "atriakit.configs.signal_preprocessor_config",
        "SignalPreprocessorConfig",
    ),
}


def __getattr__(name):
    if name not in _PUBLIC_IMPORTS:
        raise AttributeError(f"module 'atriakit' has no attribute {name!r}")

    module_name, attr_name = _PUBLIC_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
