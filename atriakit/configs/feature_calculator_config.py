from dataclasses import dataclass, field

from atriakit.configs.segment_config import SegmentConfig


@dataclass(slots=True)
class FeatureCalculatorConfig:
    """Configuration for feature extraction.

    Feature-computation parameters (noise thresholds, morphology thresholds,
    entropy settings, fragment settings) belong in PipelineConfig.
    """

    noise_estimation_window_ms: int = 50  # Pre-onset window used to estimate per-lead baseline noise.
    noise_estimate: object | None = None  # Optional externally supplied noise estimate; normally inferred from the ECG instead.
    segment_config: SegmentConfig = field(default_factory=SegmentConfig)  # Segment-boundary and baseline-correction settings.