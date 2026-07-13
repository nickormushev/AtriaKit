import os
from tokenize import group
import numpy as np
import glob
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from atriakit.paper.annotation_processor import AnnotationProcessor  # <-- adjust import
from atriakit.io.annotations_loader import _prepare_annotations
from atriakit.models.annotation_schema import AnnotationSchema


def test_prepare_annotations_deduplicates_after_path_normalization():
    annotations = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "file_path": [
                r"BaselSensis\PatientA\StudyX\Name.1234.IMA",
                "BaselSensis/PatientA/StudyX/Name.1234.IMA",
            ],
            "lead": ["I", "I"],
            "onset": [10, 10],
            "offset": [20, 20],
            "type": ["Before", "Before"],
            AnnotationSchema.IGNORE: [False, False],
        }
    )

    prepared = _prepare_annotations(annotations)

    assert len(prepared) == 1
    assert prepared.iloc[0]["file_path"] == "BaselSensis/PatientA/StudyX/Name.1234.IMA"


def test_fix_file_paths(tmp_path, monkeypatch):
    """
    Test that fix_file_paths correctly rebuilds file paths
    based on directory structure and file ID matching.
    """

    # --------------------------------------------------------
    # Setup: Create fake annotation with incorrect path
    # --------------------------------------------------------
    df = pd.DataFrame({"file_path": ["BaselSensis/PatientA/StudyX/Name.1234.IMA"]})

    processor = AnnotationProcessor.__new__(AnnotationProcessor)
    processor.annotations = df.copy()

    # --------------------------------------------------------
    # Create fake DICOM directory with expected structure
    # --------------------------------------------------------
    dicom_root = tmp_path / "dicom"
    patient_dir = dicom_root / "PatientA" / "StudyX"
    patient_dir.mkdir(parents=True)

    # File expected to match *.1234.IMA
    real_file = patient_dir / "SomePrefix.1234.IMA"
    real_file.write_text("dummy content")

    # --------------------------------------------------------
    # Monkeypatch glob.glob so we use the real filesystem
    # --------------------------------------------------------
    monkeypatch.setattr(glob, "glob", lambda pattern: [str(real_file)])

    # --------------------------------------------------------
    # Run the method
    # --------------------------------------------------------
    processor.fix_file_paths(str(dicom_root))

    # --------------------------------------------------------
    # Assert: file_path was corrected
    # --------------------------------------------------------
    expected_relative = os.path.join("PatientA", "StudyX", "SomePrefix.1234.IMA")
    assert processor.annotations.loc[0, "file_path"] == expected_relative


def test_group_p_waves(tmp_path):
    # Fake sampling rate so ms → samples is deterministic
    sampling_rate = 500  # 500 Hz means 1 sample = 2 ms

    # Tolerance: 100 ms → 50 samples
    tolerance_ms = 100

    # Fake annotations for one file, two leads
    # Onsets differ by 40 ms → 20 samples → within tolerance
    df = pd.DataFrame(
        {
            "file_path": ["a", "a", "a"],
            "lead": ["I", "I", "I"],
            "onset": [1000, 1010, 2000],  # samples
            "offset": [1050, 1060, 2050],
            "p_wave_id": [1, 2, 3],  # inconsistent
            "type": ["Before", "Before", "Before"],
        }
    )

    processor = AnnotationProcessor.__new__(AnnotationProcessor)
    processor.annotations = df.copy()

    # Mock ECGData(path).get_sampling_frequency()
    processor.group_p_waves(
        df, sampling_rate=sampling_rate, tolerance_in_ms=tolerance_ms
    )

    # After grouping:
    # Rows 0 and 1 should share the same p_wave_id (1 or 2 — either is fine)
    # Row 2 should remain distinct
    p1, p2, p3 = processor.annotations["p_wave_id"].tolist()

    # The first two must now match
    assert p1 == p2, "P-wave IDs were not grouped correctly"

    # They must not match the last one
    assert p1 != p3, "Unrelated P-wave was incorrectly merged"


def test_separate_annotations_type_both():
    # Mock annotation DataFrame
    annotations = pd.DataFrame(
        {
            "file_path": ["file2"] * 4,
            "onset": [6040, 5, 10, 6000],  # file2 has 'Both' far apart
            "type": ["Both"] * 4,
            "patient_id": ["p1"] * 4,
        }
    )

    processor = AnnotationProcessor.__new__(AnnotationProcessor)
    processor.annotations = annotations.copy()

    processor.separate_annotations_type_both(annotations, sampling_rate=1)

    # -----------------------------
    # Check that 'Both' annotations were split
    file2 = processor.annotations[processor.annotations["file_path"] == "file2"]
    assert set(file2["type"].unique()) == {"Before", "After"}

    # Ensure indices are disjoint
    before_idx = processor.annotations[processor.annotations["type"] == "Before"].index
    after_idx = processor.annotations[processor.annotations["type"] == "After"].index
    assert set(before_idx).isdisjoint(set(after_idx))

    # Total number of annotations is unchanged
    assert len(processor.annotations) == 4

    assert list(before_idx) == [1, 2]
    assert len(after_idx) == 2  # Two 'After' annotations


def test_add_global_onset_and_offset_uses_max_95_percentile_across_types():
    # ----------------------------------
    # Create fake annotations (1 patient)
    # ----------------------------------
    df = pd.DataFrame(
        {
            "patient_id": ["p1"] * 6,
            "type": ["Before"] * 3 + ["After"] * 3,
            "onset": [10, 20, 30, 40, 50, 60],
            "offset": [
                20,
                30,
                50,
                90,
                110,
                160,
            ],  # durations: [10,10,20] and [50,60,100]
        }
    )

    # Durations:
    # Before: [10, 10, 20] → 90p ≈ 19
    # After:  [50, 60, 100] → 90p ≈ 96
    # max_90p = ~96

    processor = AnnotationProcessor.__new__(AnnotationProcessor)
    processor.annotations = df.copy()

    # ----------------------------------
    # Run method on single-patient slice
    # ----------------------------------
    out = processor.add_global_onset_and_offset(processor.annotations)

    # ----------------------------------
    # Expected max 90th percentile
    # ----------------------------------
    expected_90_before = np.percentile([10, 10, 20], 90)
    expected_90_after = np.percentile([50, 60, 100], 90)
    expected_max_90 = max(expected_90_before, expected_90_after)

    # ----------------------------------
    # Assertions
    # ----------------------------------
    # onset_g should equal onset
    assert np.allclose(out["onset_g"], out["onset"])

    # offset_g should equal onset + max_90
    assert np.allclose(out["offset_g"], out["onset"] + expected_max_90)

    # Columns must exist
    assert "onset_g" in processor.annotations.columns
    assert "offset_g" in processor.annotations.columns


def test_calculate_qrs_onsets_r_peaks_applies_median_to_all_leads():
    # Mock annotations: 3 P-waves, 3 leads each
    df = pd.DataFrame(
        [
            {
                "file_path": "a.dcm",
                "p_wave_id": 0,
                "lead": "II",
                "type": "Before",
                "onset": 0,
                "offset": 50,
            },
            {
                "file_path": "a.dcm",
                "p_wave_id": 0,
                "lead": "V5",
                "type": "Before",
                "onset": 0,
                "offset": 50,
            },
            {
                "file_path": "a.dcm",
                "p_wave_id": 0,
                "lead": "V6",
                "type": "Before",
                "onset": 0,
                "offset": 50,
            },
            {
                "file_path": "a.dcm",
                "p_wave_id": 1,
                "lead": "II",
                "type": "Before",
                "onset": 100,
                "offset": 150,
            },
            {
                "file_path": "a.dcm",
                "p_wave_id": 1,
                "lead": "V5",
                "type": "Before",
                "onset": 100,
                "offset": 150,
            },
            {
                "file_path": "a.dcm",
                "p_wave_id": 1,
                "lead": "V6",
                "type": "Before",
                "onset": 100,
                "offset": 150,
            },
            {
                "file_path": "a.dcm",
                "p_wave_id": 2,
                "lead": "II",
                "type": "Before",
                "onset": 200,
                "offset": 250,
            },
            {
                "file_path": "a.dcm",
                "p_wave_id": 2,
                "lead": "V5",
                "type": "Before",
                "onset": 200,
                "offset": 250,
            },
            {
                "file_path": "a.dcm",
                "p_wave_id": 2,
                "lead": "V6",
                "type": "Before",
                "onset": 200,
                "offset": 250,
            },
        ]
    )

    processor = AnnotationProcessor.__new__(AnnotationProcessor)
    processor.annotations = df.copy()
    processor.qrs_onset_margin_ms = 50  # 50 ms margin
    processor.feature_calculators = MagicMock()

    # Sampling frequency 500 Hz
    fs = 500

    r_peaks = [60, 160, 260]

    def find_r_peaks_median(annotations, ecg_data):
        return [r_peaks.pop(0)]

    # Mock find_r_peaks_median to return predetermined R-peaks
    processor.feature_calculators.find_r_peaks_median = find_r_peaks_median
    ecg_data = MagicMock()

    # Run method
    out = processor.calculate_qrs_onsets_r_peaks(df, ecg_data, fs=fs)

    # Convert 50 ms → samples
    margin_samples = int(50 / 1000 * fs)  # 25 samples

    expected_onsets = [60 - margin_samples, 160 - margin_samples, 260 - margin_samples]

    # Check each P-wave's onset is applied to all leads
    for p_wave_id, expected in enumerate(expected_onsets):
        mask = out["p_wave_id"] == p_wave_id
        assert all(
            out.loc[mask, "qrs_onset_ms"] == expected
        ), f"P-wave {p_wave_id} QRS onset mismatch"

    # All rows should now have the new qrs_onset_ms column
    assert "qrs_onset_ms" in out.columns
    assert len(out) == 9  # all leads retained
