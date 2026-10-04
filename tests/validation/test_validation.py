"""Quantitative validation of the full pipeline against analytically known values.

A synthetic recording (see ``reference_cases``) is run through the real ``Pipeline``
and each descriptor is compared with a value derived on paper from the waveform
recipe, or with an independent numpy/scipy computation where no closed form exists.

1. Preprocessing off: features vs analytic truth; any mismatch is an implementation error.
2. Artifact suppression: a known artifact must perturb features far less with the
   analysis preprocessing (``reference_cases.target_preprocessing_config``) than without.

The residual left by each artifact is reported by
``paper/preprocessing_artifact_analysis.py``, not asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

import reference_cases as rc
from atriakit import Pipeline
from atriakit.configs.feature_computation_config import FeatureComputationConfig

# Exact-comparison tolerances; waveforms are rounded to 1e-9 mV.
RTOL = 1e-6
ATOL = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_pipeline(ecg: np.ndarray, pipeline_config=None):
    """Run the real Pipeline on ``ecg`` and return the feature DataFrame."""
    pipeline = Pipeline(
        ecg_base_path=".",
        pipeline_config=pipeline_config,
        loader=rc.make_ecg_loader(ecg),
    )
    result = pipeline.run(rc.build_annotations(), cache_dir=None)
    return result.get_df() if hasattr(result, "get_df") else result


def run_without_preprocessing(feature_config: FeatureComputationConfig | None = None):
    config = rc.no_preprocessing_config()
    if feature_config is not None:
        config.feature_computation = feature_config
    return run_pipeline(rc.build_signal(), config)


def values_for(result, lead: str, feature: str) -> np.ndarray:
    """One value per beat for ``feature`` on ``lead``."""
    return result.loc[result["lead"] == lead, feature].to_numpy()


def assert_close(actual, expected, label: str, rtol=RTOL, atol=ATOL) -> None:
    """Compare a column of per-beat values with one expected value (number or text)."""
    if isinstance(expected, str):
        assert set(actual) == {expected}, f"{label}: got {set(actual)}, expected {expected!r}"
        return
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), expected, rtol=rtol, atol=atol, err_msg=label
    )


@pytest.fixture(scope="module")
def clean():
    """Golden recording, preprocessing off, default feature config."""
    return run_without_preprocessing()


@pytest.fixture(scope="module")
def clean_zero_crossing():
    """Same, with the conventional zero-crossing PTF enabled."""
    return run_without_preprocessing(FeatureComputationConfig(ptf_zero_crossing=True))


# ---------------------------------------------------------------------------
# 1. Preprocessing OFF: exact agreement
# ---------------------------------------------------------------------------

LEAD_FEATURES = [
    # duration and area family
    "duration", "area", "area_to_duration_ratio", "dispersion",
    # amplitude family
    "max_amplitude", "min_amplitude", "max_absolute_amplitude", "ptp_amplitude",
    "offset_amplitude", "onset_offset_slope",
    # shape family
    "p_wave_morphology", "ptf_auto", "complexity",
    "fragment_count", "fragment_width", "fragment_height",
]


# One lead per waveform shape, plus aVF (shorter window) and aVR (inverted).
SHAPE_LEADS = ["I", "aVF", "aVR", "V1", "V2", "V3"]


@pytest.mark.parametrize("lead", SHAPE_LEADS)
@pytest.mark.parametrize("feature", LEAD_FEATURES)
def test_lead_feature_matches_analytic_value(clean, lead, feature):
    expected = rc.lead_expected(lead)[feature]
    assert_close(values_for(clean, lead, feature), expected, f"{lead}/{feature}")


# --- terminal force (PTF) --------------------------------------------------

# aVR is the inverted V1 wave, so every PTF variant should match V1's value. The
# classifier places aVR's inflection one sample late (exact-zero second difference
# counts as positive); pinned as an xfail until the classifier rework.
AVR_INFLECTION_ONE_SAMPLE_LATE = pytest.mark.xfail(
    strict=True,
    reason="calculate_inflexion treats an exact-zero second difference as positive, so the "
    "Negative-Positive (aVR) inflection lands one sample after V1's",
)
ONE_SAMPLE_OF_TERMINAL_SEGMENT = 1 / (rc.WINDOW - rc.WINDOW // 2 + 1)  # ~3% of the terminal part


@pytest.mark.parametrize("lead, rtol", [("V1", RTOL), ("aVR", ONE_SAMPLE_OF_TERMINAL_SEGMENT + RTOL)])
def test_supervised_ptf_on_biphasic_wave(clean, lead, rtol):
    expected = rc.lead_expected(lead)["ptf"]
    assert_close(values_for(clean, lead, "ptf"), expected, f"{lead}/ptf", rtol=rtol)


@pytest.mark.parametrize("lead", ["V1", "aVR"])
def test_zero_crossing_ptf_matches_conventional_definition(clean_zero_crossing, lead):
    expected = rc.lead_expected(lead)["ptf_zero_crossing"]
    actual = values_for(clean_zero_crossing, lead, "ptf_auto")
    assert_close(actual, expected, f"{lead}/ptf zero-crossing")


@pytest.mark.parametrize(
    "lead", ["V1", pytest.param("aVR", marks=AVR_INFLECTION_ONE_SAMPLE_LATE)]
)
def test_inflection_point_is_the_steepest_zero_crossing(clean, lead):
    expected = rc.p_onsets() + rc.WINDOW // 2  # absolute sample index, one per beat
    assert_close(values_for(clean, lead, "inflection_point"), expected, f"{lead}/inflection")


# --- entropy ---------------------------------------------------------------


@pytest.mark.parametrize("lead", SHAPE_LEADS)
def test_entropies_match_independent_implementations(clean, lead):
    for feature, expected in rc.entropy_expected(lead).items():
        assert_close(values_for(clean, lead, feature), expected, f"{lead}/{feature}")


def test_ramp_entropies_match_closed_form(clean):
    """65 evenly spaced samples in 32 bins, and a perfectly self-similar line (SampEn 0)."""
    closed_form = rc.lead_expected("V2")
    for feature in ("shannon_entropy", "sample_entropy"):
        assert_close(values_for(clean, "V2", feature), closed_form[feature], f"V2/{feature}")


# --- axis and vectorcardiogram --------------------------------------------


def test_frontal_axis_recovers_the_synthetic_dipole_angle(clean):
    assert_close(clean["axis"], rc.FRONTAL_AXIS_DEG, "axis")


@pytest.mark.parametrize("feature", list(rc.vcg_expected()))
def test_vcg_feature_matches_reference(clean, feature):
    assert_close(clean[feature], rc.vcg_expected()[feature], feature)


def test_kors_matrix_matches_the_published_table():
    """Feeding unit leads through the converter reads the matrix columns back out.

    ``rc.KORS`` is the coefficient table of Kors et al. (1990), copied from a published
    reproduction of it (see ``reference_cases``), not from ``atriakit.utils``.
    """
    from atriakit.utils import convert_ecg_segment_to_vcg

    one_hot_leads = np.eye(len(rc.KORS_LEADS))
    lead_to_index = {lead: i for i, lead in enumerate(rc.KORS_LEADS)}
    np.testing.assert_allclose(convert_ecg_segment_to_vcg(one_hot_leads, lead_to_index), rc.KORS)


# --- known cross_lead limitation (strict xfail) ---------------------------
# cross_lead pads aVF's shorter window with flat baseline; the classifier's ``>= 0``
# check reads it as a rising phase and can relabel the wave.
# ---------------------------------------------------------------------------


def run_with_boundary_mode(boundary_mode: str):
    """Clean recording, preprocessing off, so only the boundary mode varies."""
    pipeline = Pipeline(
        ecg_base_path=".",
        pipeline_config=rc.no_preprocessing_config(),
        loader=rc.make_ecg_loader(rc.build_signal()),
    )
    annotations = rc.build_annotations(boundary_mode=boundary_mode)
    result = pipeline.run(annotations, cache_dir=None)
    return result.get_df() if hasattr(result, "get_df") else result


@pytest.mark.xfail(
    strict=True,
    reason="classify_p_wave_morphology counts aVF's flat cross_lead padding as a rising "
    "phase, so its label changes under cross_lead",
)
def test_morphology_label_is_the_same_under_both_boundary_modes():
    own_window = values_for(run_with_boundary_mode("per_lead"), "aVF", "p_wave_morphology")
    shared_window = values_for(run_with_boundary_mode("cross_lead"), "aVF", "p_wave_morphology")
    assert set(shared_window) == set(own_window)


# ---------------------------------------------------------------------------
# 2. Preprocessing suppresses the artifacts it is designed to suppress
#
# Smoke check that preprocessing is wired up and effective, not a quality measure: a
# loose factor set below the weakest observed pair (mains, ~9x). Only pairs the filter
# targets: drift/wander (high-pass) and mains (notch).
# ---------------------------------------------------------------------------

MIN_SUPPRESSION_FACTOR = 5.0

PERTURBED_FEATURES = {
    "drift": ["area", "max_amplitude", "min_amplitude", "offset_amplitude", "axis", "vcg_area"],
    "wander": ["area", "max_amplitude", "min_amplitude", "offset_amplitude", "axis", "vcg_area"],
    "mains": ["area", "max_amplitude", "min_amplitude", "offset_amplitude", "ptp_amplitude"],
}


def worst_feature_shift(reference, actual, feature: str) -> float:
    """Largest absolute change in ``feature`` across every lead and beat."""
    worst = 0.0
    for lead in rc.LEADS:
        expected = values_for(reference, lead, feature).astype(float)
        got = values_for(actual, lead, feature).astype(float)
        worst = max(worst, float(np.abs(got - expected).max()))
    return worst


@pytest.fixture(scope="module")
def artifact_suppression():
    """{artifact: {feature: (error_without_preprocessing, error_with_preprocessing)}}."""
    on, off = rc.target_preprocessing_config(), rc.no_preprocessing_config()
    clean_on = run_pipeline(rc.build_signal(), on)
    clean_off = run_pipeline(rc.build_signal(), off)

    measured = {}
    for artifact_name, artifact in rc.corruptions().items():
        if artifact_name not in PERTURBED_FEATURES:
            continue
        corrupted = rc.build_signal() + artifact[None, :]
        bad_on = run_pipeline(corrupted, on)
        bad_off = run_pipeline(corrupted, off)
        measured[artifact_name] = {
            feature: (
                worst_feature_shift(clean_off, bad_off, feature),
                worst_feature_shift(clean_on, bad_on, feature),
            )
            for feature in PERTURBED_FEATURES[artifact_name]
        }
    return measured


@pytest.mark.parametrize("artifact_name", list(PERTURBED_FEATURES))
def test_preprocessing_suppresses_known_artifacts(artifact_suppression, artifact_name):
    too_weak = {}
    for feature, (error_off, error_on) in artifact_suppression[artifact_name].items():
        assert error_off > 0, f"{artifact_name} does not perturb {feature} at all; wrong pair"
        factor = error_off / max(error_on, 1e-15)
        if factor < MIN_SUPPRESSION_FACTOR:
            too_weak[feature] = round(factor, 1)

    assert not too_weak, (
        f"{artifact_name}: suppressed by less than {MIN_SUPPRESSION_FACTOR}x "
        f"(factor per feature): {too_weak}"
    )


@pytest.fixture(scope="module")
def clean_default():
    """Golden recording through the preprocessing used for the real-data analysis."""
    return run_pipeline(rc.build_signal(), rc.target_preprocessing_config())
