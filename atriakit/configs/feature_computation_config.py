from dataclasses import dataclass


@dataclass(slots=True)
class FeatureComputationConfig:
    """Feature-computation parameters passed to FeatureCalculators.compute_all()."""

    # Entropy
    shannon_entropy_n_bins: int = 32  # number of histogram bins for Shannon entropy
    shannon_entropy_bin_range: tuple[float, float] | None = (
        None  # fixed amplitude range for the histogram; None uses per-signal min/max. For comparable results should be set to the same value across the datasets.
    )
    sample_entropy_m: int = (
        2  # embedding dimension (template length) for sample entropy
    )
    sample_entropy_r_factor: float = (
        0.25  # tolerance as fraction of segment SD; controls how strictly templates must match
    )

    # Complexity / extrema
    extrema_threshold_multiplier: float = (
        0.1  # peak must exceed this fraction of max amplitude to be counted (complexity feature)
    )

    # Noise & morphology
    noise_sd_multiplier: float = (
        3.0  # a signal deflection must exceed this many noise SDs to be counted as a fragment or extremum
    )
    morphology_min_phase_fraction: float = (
        0.1  # minimum fraction of segment length a sign group must span to count as a distinct morphology phase (0.1 = at least 10% of P-wave)
    )
    morphology_noise_sd_multiplier: float = (
        3.0  # same as noise_sd_multiplier but applied only inside the morphology classifier
    )

    # Fragmentation
    fragment_noise_multiplier: float = (
        3.0  # a fragment's amplitude change must exceed this many noise SDs to count as a valid fragment boundary
    )
    min_fragment_length_ms: float = (
        0.0  # fragments shorter than this (ms) are discarded
    )
    normalize_by_duration: bool = (
        False  # normalise fragment metrics per 100ms of P-wave duration; True matches the original paper
    )
