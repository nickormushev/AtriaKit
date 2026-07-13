"""Integration tests: end-to-end pipeline and lead-ordering invariance.

All tests share a single module-scoped dataset directory generated once and
deleted after the last test in this module finishes.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atriakit import Pipeline, AnnotationsLoader
from atriakit.constants import VCG_FEATURE_COLUMNS
from atriakit.datasets import simulate_12lead, generate, to_dicom, LEADS
from atriakit.feature_calculator import FeatureCalculators
from atriakit.io import ECGLoader
from atriakit.models.ecg_data import ECGData


# ---------------------------------------------------------------------------
# Shared dataset (generated once per module, cleaned up in finally)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_dir():
    tmp = tempfile.mkdtemp(prefix="atriakit_integration_")
    try:
        generate(tmp)
        yield Path(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def demo_annotations(demo_dir):
    return AnnotationsLoader().load(str(demo_dir / "demo_annotations.csv"))


# ---------------------------------------------------------------------------
# End-to-end pipeline: all features present
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end_generates_all_feature_columns(demo_dir, demo_annotations):
    """Pipeline.run() on generated demo data returns a complete feature DataFrame."""
    pipeline = Pipeline(ecg_base_path=demo_dir)
    result = pipeline.run(demo_annotations)

    assert not result.empty, "Pipeline returned empty DataFrame"
    expected_cols = {
        "p_wave_morphology", "inflection_point",
        "duration", "area", "area_to_duration_ratio",
        "max_absolute_amplitude", "max_amplitude", "min_amplitude", "ptp_amplitude",
        "ptf", "dispersion", "atrial_rate", "heart_rate",
        "offset_amplitude", "onset_offset_angle",
        "complexity", "shannon_entropy", "sample_entropy",
        "fragment_count", "fragment_width", "fragment_height",
        "axis",
        "vcg_area", "vcg_axis_azimuth", "vcg_axis_elevation",
        "vcg_eigenvalues_1", "vcg_eigenvalues_2", "vcg_eigenvalues_3",
        "vcg_flatness", "vcg_roundness",
        "vcg_sum_fragment_count", "vcg_sum_fragment_height", "vcg_sum_fragment_width",
        "vcg_x_fragment_count", "vcg_x_fragment_height", "vcg_x_fragment_width",
        "vcg_y_fragment_count", "vcg_y_fragment_height", "vcg_y_fragment_width",
        "vcg_z_fragment_count", "vcg_z_fragment_height", "vcg_z_fragment_width",
    }
    missing = expected_cols - set(result.columns)
    assert not missing, f"Missing feature columns: {missing}"
    assert len(result) == len(demo_annotations)


# ---------------------------------------------------------------------------
# Lead-order invariance with a non-standard lead count (9 leads, one is "V9")
#
# We use the 8 Kors leads (required for VCG) plus a fictional "V9" channel.
# Two DICOMs are created from the same signals but with channels in different
# order. The pipeline must produce identical features for both.
# ---------------------------------------------------------------------------

_LEADS_9 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "V9"]
_LEADS_9_REV = list(reversed(_LEADS_9))

_ANN_ROWS = [
    {"lead": "I",  "onset": 80,  "offset": 110, "p_wave_id": 1},
    {"lead": "II", "onset": 82,  "offset": 112, "p_wave_id": 2},
]


def _make_annotations(file_name: str) -> object:
    rows = [{**row, "file_path": file_name} for row in _ANN_ROWS]
    return AnnotationsLoader().from_dataframe(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def lead_order_dir(demo_dir):
    """Write two 9-lead DICOMs: standard and reversed channel order."""
    import numpy as np
    ecg_12 = simulate_12lead(heart_rate=60, seed=99)
    rng = np.random.default_rng(0)

    # V9 is a fictional extra lead — just synthetic noise
    v9 = rng.normal(0, 0.05, ecg_12.shape[1])

    # Assemble 9-lead arrays in standard and reversed order
    kors_rows = np.array([LEADS.index(name) for name in _LEADS_9[:-1]])  # drop V9 index
    ecg_9_std = np.vstack([ecg_12[kors_rows], v9])                       # (9, n_samples)
    ecg_9_rev = ecg_9_std[[_LEADS_9.index(name) for name in _LEADS_9_REV]]

    path_std = demo_dir / "LEADS9_STD.dcm"
    path_rev = demo_dir / "LEADS9_REV.dcm"
    to_dicom(ecg_9_std, patient_id="LEADS9_STD", leads=_LEADS_9).save_as(str(path_std))
    to_dicom(ecg_9_rev, patient_id="LEADS9_REV", leads=_LEADS_9_REV).save_as(str(path_rev))

    return path_std, path_rev


def test_dicom_loader_maps_9_lead_channels_correctly(lead_order_dir):
    """DicomLoader correctly maps 9 channels (including non-standard V9) by label."""
    path_std, path_rev = lead_order_dir

    ecg_std = ECGLoader().load(str(path_std))
    ecg_rev = ECGLoader().load(str(path_rev))

    assert ecg_std.get_leads() == _LEADS_9
    assert ecg_rev.get_leads() == _LEADS_9_REV
    assert ecg_std.get_lead_to_index()["V9"] == 8
    assert ecg_rev.get_lead_to_index()["V9"] == 0


def test_pipeline_features_invariant_to_lead_order(lead_order_dir, demo_dir):
    """Pipeline.run() produces identical numeric features for both channel orderings."""
    path_std, path_rev = lead_order_dir

    pipeline = Pipeline(ecg_base_path=demo_dir)
    result_std = pipeline.run(_make_annotations(path_std.name))
    result_rev = pipeline.run(_make_annotations(path_rev.name))

    df_std = result_std.get_df()
    df_rev = result_rev.get_df()

    assert list(df_std.columns) == list(df_rev.columns)

    numeric_cols = df_std.select_dtypes(include="number").columns
    for col in numeric_cols:
        np.testing.assert_allclose(
            df_std[col].values,
            df_rev[col].values,
            rtol=1e-8,
            equal_nan=True,
            err_msg=f"Column '{col}' differs between standard and reversed lead order",
        )


# ---------------------------------------------------------------------------
# compute_all with too few leads: VCG and axis skip gracefully with NaN columns
# ---------------------------------------------------------------------------

def test_compute_all_skips_vcg_and_axis_when_leads_missing():
    """compute_all returns NaN for VCG and axis columns when the required leads
    are absent, instead of crashing. All other features still compute normally."""
    ecg_12 = simulate_12lead(heart_rate=60, seed=7)

    # Keep only II and aVR — no VCG leads (need I, V1–V6), no axis leads (need I, aVF).
    subset = ["II", "aVR"]
    ecg_2 = ecg_12[[LEADS.index(name) for name in subset], :]
    ecg_data = ECGData(ecg=ecg_2, fs=500, lead_to_index={name: i for i, name in enumerate(subset)})

    ann = AnnotationsLoader().from_dataframe(pd.DataFrame([
        {"lead": "II", "onset": 80, "offset": 110, "p_wave_id": 1, "file_path": "f"},
    ]))

    result = FeatureCalculators().compute_all(ann, ecg_data)

    # VCG and axis columns must all be NaN
    for col in VCG_FEATURE_COLUMNS + ["axis"]:
        assert col in result.columns, f"Expected column '{col}' to be present"
        assert result[col].isna().all(), f"Expected column '{col}' to be all-NaN"

    # Non-VCG features should have computed values
    assert not result["area"].isna().all()
    assert not result["max_amplitude"].isna().all()
