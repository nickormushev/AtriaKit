"""Quantitative validation of the full pipeline against analytically known values.

A synthetic recording (see ``reference_cases``) is run through the real ``Pipeline``
and each descriptor is compared with a value derived on paper from the waveform
recipe, or with an independent numpy/scipy computation where no closed form exists.

1. Preprocessing off: features vs analytic truth; any mismatch is an implementation error.
2. Artifact suppression: a known artifact must perturb features far less with the
   manuscript's preprocessing (``reference_cases.target_preprocessing_config``, read from
   ``examples/atriakit_manuscript_config.yaml``) than without.
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

# aVR is the inverted V1 wave, so every PTF variant should match V1's value.


@pytest.mark.parametrize("lead", ["V1", "aVR"])
def test_supervised_ptf_on_biphasic_wave(clean, lead):
    expected = rc.lead_expected(lead)["ptf"]
    assert_close(values_for(clean, lead, "ptf"), expected, f"{lead}/ptf")


@pytest.mark.parametrize("lead", ["V1", "aVR"])
def test_zero_crossing_ptf_matches_conventional_definition(clean_zero_crossing, lead):
    expected = rc.lead_expected(lead)["ptf_zero_crossing"]
    actual = values_for(clean_zero_crossing, lead, "ptf_auto")
    assert_close(actual, expected, f"{lead}/ptf zero-crossing")


@pytest.mark.parametrize("lead", ["V1", "aVR"])
def test_inflection_point_is_the_steepest_zero_crossing(clean, lead):
    expected = rc.p_onsets() + rc.WINDOW // 2  # absolute sample index, one per beat
    assert_close(values_for(clean, lead, "inflection_point"), expected, f"{lead}/inflection")


@pytest.mark.parametrize("lead", ["V1", "aVR"])
def test_supervised_and_default_ptf_agree_on_biphasic_wave(clean, lead):
    assert_close(values_for(clean, lead, "ptf"), values_for(clean, lead, "ptf_auto"), f"{lead}/ptf vs ptf_auto")


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


# --- Shannon entropy with the manuscript's binning ---------------------------

MANUSCRIPT_SHANNON_BINS = 128
MANUSCRIPT_SHANNON_RANGE_MV = (-2.0, 2.0)


@pytest.fixture(scope="module")
def clean_manuscript_shannon():
    """Golden recording, preprocessing off, Shannon entropy binned as in the manuscript."""
    return run_without_preprocessing(
        FeatureComputationConfig(
            shannon_entropy_n_bins=MANUSCRIPT_SHANNON_BINS,
            shannon_entropy_bin_range=MANUSCRIPT_SHANNON_RANGE_MV,
        )
    )


@pytest.mark.parametrize("lead", SHAPE_LEADS)
def test_shannon_entropy_with_the_manuscript_binning(clean_manuscript_shannon, lead):
    expected = rc.reference_shannon(
        rc.p_wave(rc.LEAD_SPECS[lead]), MANUSCRIPT_SHANNON_BINS, MANUSCRIPT_SHANNON_RANGE_MV
    )
    actual = values_for(clean_manuscript_shannon, lead, "shannon_entropy")
    assert_close(actual, expected, f"{lead}/shannon_entropy (manuscript binning)")


# --- rates -------------------------------------------------------------------


def test_atrial_and_heart_rate_match_the_beat_spacing(clean):
    """Eight beats 400 samples apart at 500 Hz: 75 bpm for both rates."""
    for feature in ("atrial_rate", "heart_rate"):
        assert_close(clean[feature], rc.BEAT_RATE_BPM, feature)


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


# ---------------------------------------------------------------------------
# 2. Preprocessing suppresses the artifacts it is designed to suppress
#
# Smoke check that preprocessing is wired up and effective, not a quality measure: a
# loose factor set below the weakest observed pair (mains / min amplitude, ~14x with the
# manuscript configuration). Only pairs the filter targets: drift/wander (high-pass) and
# mains (notch).
# ---------------------------------------------------------------------------

MIN_SUPPRESSION_FACTOR = 5.0

PERTURBED_FEATURES = {
    "drift": ["area", "max_amplitude", "min_amplitude", "offset_amplitude", "axis", "vcg_area"],
    "wander": ["area", "max_amplitude", "min_amplitude", "offset_amplitude", "axis", "vcg_area"],
    "mains": ["area", "max_amplitude", "min_amplitude", "offset_amplitude", "ptp_amplitude"],
}

# offset_amplitude reads a single sample, so on the first and last beat it picks up the
# edge transients of the zero-phase filters on this short recording; those beats are skipped.
INTERIOR_BEATS_ONLY = {"offset_amplitude"}


def worst_feature_shift(reference, actual, feature: str) -> float:
    """Largest absolute change in ``feature`` across every lead and beat."""
    beats = slice(1, -1) if feature in INTERIOR_BEATS_ONLY else slice(None)
    worst = 0.0
    for lead in rc.LEADS:
        expected = values_for(reference, lead, feature).astype(float)[beats]
        got = values_for(actual, lead, feature).astype(float)[beats]
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


