from dataclasses import dataclass, field

from atriakit.configs.feature_computation_config import FeatureComputationConfig
from atriakit.configs.segment_config import SegmentConfig
from atriakit.configs.signal_preprocessor_config import (
    SignalPreprocessorConfig,
    default_morphology_preprocessor_config,
    default_signal_preprocessor_config,
)


@dataclass(slots=True)
class PipelineConfig:
    """Batch pipeline parameters."""

    feature_computation: FeatureComputationConfig = field(
        default_factory=FeatureComputationConfig  # settings passed to FeatureCalculators.compute_all(); see FeatureComputationConfig
    )
    segment_config: SegmentConfig = field(
        default_factory=SegmentConfig  # segment-boundary and baseline-correction settings; see SegmentConfig
    )
    signal_preprocessor_config: SignalPreprocessorConfig = field(
        default_factory=default_signal_preprocessor_config  # preprocessing for amplitude and duration features
    )
    morphology_preprocessor_config: SignalPreprocessorConfig = field(
        default_factory=default_morphology_preprocessor_config  # preprocessing for shape-based features
    )

    # Grouping
    group_tolerance_ms: int = (
        200  # max time gap (ms) between P-wave peaks across leads to be grouped as the same beat
    )
