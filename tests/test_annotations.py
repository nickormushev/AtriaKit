from atriakit.models.annotations import Annotations
from atriakit.configs.annotations_loader_config import AnnotationsLoaderConfig
from atriakit.io import AnnotationsLoader
from atriakit.models.annotation_schema import AnnotationSchema
import pandas as pd
import pytest


@pytest.fixture
def sample_annotations_multiple_leads():
    # Two P-waves (1 and 2), each annotated across two leads
    return pd.DataFrame(
        {
            "p_wave_id": [1, 1, 2, 2],
            "file_path": ["file1", "file1", "file1", "file1"],
            "type": ["A", "A", "B", "B"],
            "lead": ["I", "II", "I", "II"],
            "onset": [2, 1, 0, 1],  # per-lead onsets
            "offset": [4, 5, 2, 3],  # per-lead offsets
            "qrs_onset": [5, 6, 3, 4],
        }
    )


def test_annotations_class_compute_multilead_rows_cross_lead(sample_annotations_multiple_leads):
    annotations = AnnotationsLoader(config=AnnotationsLoaderConfig(boundary_mode="cross_lead")).from_dataframe(
        sample_annotations_multiple_leads.copy(),
    )

    # Cross-lead values are exposed on every per-lead row via the enriched dataframe
    df = annotations._df

    # P-wave 1: earliest onset = min(2,1)=1, latest offset=max(4,5)=5
    pw1 = df[df["p_wave_id"] == 1]
    assert (pw1["onset"] == 1).all()
    assert (pw1["offset"] == 5).all()
    assert (pw1["qrs_onset"] == 5).all()

    # P-wave 2: earliest onset = min(0,1)=0, latest offset=max(3,2)=3
    pw2 = df[df["p_wave_id"] == 2]
    assert (pw2["onset"] == 0).all()
    assert (pw2["offset"] == 3).all()
    assert (pw2["qrs_onset"] == 3).all()


def test_annotations_class_compute_multilead_rows_per_lead(sample_annotations_multiple_leads):
    annotations = AnnotationsLoader(config=AnnotationsLoaderConfig(boundary_mode="per_lead")).from_dataframe(
        sample_annotations_multiple_leads.copy(),
    )

    df = annotations._df

    # onset/offset reflect each lead's own annotation values
    pw1 = df[df["p_wave_id"] == 1].set_index("lead")
    assert pw1.loc["I", "onset"] == 2
    assert pw1.loc["II", "onset"] == 1
    assert pw1.loc["I", "offset"] == 4
    assert pw1.loc["II", "offset"] == 5

    # onset_original matches onset in per_lead mode
    assert (df["onset"] == df["onset_original"]).all()
    assert (df["offset"] == df["offset_original"]).all()


def test_annotations_class_compute_multilead(sample_annotations_multiple_leads):
    annotations = AnnotationsLoader(config=AnnotationsLoaderConfig(boundary_mode="cross_lead")).from_dataframe(
        sample_annotations_multiple_leads.copy(),
    )
    df = annotations._df

    # Enriched dataframe retains all per-lead rows; onset/offset are multilead
    assert len(df) == len(sample_annotations_multiple_leads)
    assert AnnotationSchema.ONSET_ORIGINAL in df.columns
    assert AnnotationSchema.OFFSET_ORIGINAL in df.columns


def test_annotations_class_get_vcg_annotations_uses_multilead_values():
    raw = pd.DataFrame(
        {
            "p_wave_id": [1, 1],
            "file_path": ["file1", "file1"],
            "type": ["Before", "Before"],
            "lead": ["I", "II"],
            "onset": [12, 10],
            "offset": [18, 20],
            "qrs_onset": [25, 25],
        }
    )

    annotations = AnnotationsLoader(config=AnnotationsLoaderConfig(boundary_mode="cross_lead")).from_dataframe(raw)
    vcg = annotations.vcg_annotations()

    assert isinstance(vcg, Annotations)
    assert list(vcg._df["lead"].unique()) == ["VCG"]
    # multilead onset = min(12, 10) = 10; offset = max(18, 20) = 20
    assert list(vcg._df["onset"]) == [10]
    assert list(vcg._df["offset"]) == [20]
    # per-lead values are NaN for VCG
    assert vcg._df["onset_original"].isna().all()
    assert vcg._df["offset_original"].isna().all()


def test_annotations_preserves_dataframe_index():
    # Non-zero-based index simulates a subset of a larger DataFrame (real usage in
    # annotation_processor: type_group.index = [14, 15, ...]).  The merge inside
    # _build_df_cache must not reset the index, otherwise features written back via
    # self.annotations.loc[type_group.index, ...] land on wrong rows.
    df = pd.DataFrame(
        {
            "p_wave_id": [0, 0, 1, 1],
            "file_path": ["f", "f", "f", "f"],
            "lead": ["I", "II", "I", "II"],
            "onset": [10, 9, 20, 21],
            "offset": [15, 16, 25, 26],
        },
        index=[100, 200, 300, 400],
    )

    ann = AnnotationsLoader(config=AnnotationsLoaderConfig(boundary_mode="cross_lead")).from_dataframe(df)
    assert list(ann._df.index) == [100, 200, 300, 400]


def test_annotations_loader_load_directory_combines_csvs(tmp_path):
    a = pd.DataFrame({"file_path": ["f"], "lead": ["I"], "onset": [0], "offset": [5], "p_wave_id": [1]})
    b = pd.DataFrame({"file_path": ["f"], "lead": ["II"], "onset": [1], "offset": [6], "p_wave_id": [1]})
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)

    ann = AnnotationsLoader().load(tmp_path)

    assert isinstance(ann, Annotations)
    assert set(ann._df["lead"].unique()) == {"I", "II"}


def test_annotations_loader_load_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No annotation files with registered extensions"):
        AnnotationsLoader().load(tmp_path)


def test_annotations_loader_load_directory_skips_unregistered_extensions(tmp_path):
    a = pd.DataFrame({"file_path": ["f"], "lead": ["I"], "onset": [0], "offset": [5], "p_wave_id": [1]})
    a.to_csv(tmp_path / "a.csv", index=False)
    (tmp_path / "notes.txt").write_text("ignore me")

    ann = AnnotationsLoader().load(tmp_path)

    assert len(ann) == 1


def test_annotations_loader_load_directory_crawls_subdirectories(tmp_path):
    sub = tmp_path / "patient_01"
    sub.mkdir()
    a = pd.DataFrame({"file_path": ["f"], "lead": ["I"], "onset": [0], "offset": [5], "p_wave_id": [1]})
    b = pd.DataFrame({"file_path": ["f"], "lead": ["II"], "onset": [1], "offset": [6], "p_wave_id": [1]})
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(sub / "b.csv", index=False)

    ann = AnnotationsLoader().load(tmp_path)

    assert set(ann._df["lead"].unique()) == {"I", "II"}


def test_annotations_loader_load_directory_uses_registered_loader_per_extension(tmp_path):
    csv_df = pd.DataFrame({"file_path": ["f"], "lead": ["I"], "onset": [0], "offset": [5], "p_wave_id": [1]})
    tsv_df = pd.DataFrame({"file_path": ["f"], "lead": ["II"], "onset": [1], "offset": [6], "p_wave_id": [1]})
    csv_df.to_csv(tmp_path / "a.csv", index=False)
    tsv_df.to_csv(tmp_path / "b.tsv", index=False, sep="\t")

    class TsvLoader:
        def load(self, path):
            return pd.read_csv(path, sep="\t")

    loader = AnnotationsLoader()
    loader.register(".tsv", TsvLoader())
    ann = loader.load(tmp_path)

    assert set(ann._df["lead"].unique()) == {"I", "II"}


def _make_df(**extra):
    base = {
        "p_wave_id": [0, 0, 1, 1],
        "file_path": ["f", "f", "f", "f"],
        "lead": ["I", "II", "I", "II"],
        "onset": [10, 11, 20, 21],
        "offset": [15, 16, 25, 26],
        "onset_original": [10, 11, 20, 21],
        "offset_original": [15, 16, 25, 26],
    }
    base.update(extra)
    return pd.DataFrame(base)


def test_constructor_isolates_from_source_df():
    """Mutations to the source DataFrame must not affect Annotations internals."""
    df = _make_df()
    ann = Annotations(df)
    df["onset"] = 999
    assert (ann._df["onset"] != 999).all()


def test_sort_values_does_not_mutate_source():
    """sort_values must return a new Annotations without affecting the original."""
    ann = Annotations(_make_df())
    original_order = list(ann._df["onset"])
    sorted_ann = ann.sort_values(by="onset", ascending=False)
    assert list(ann._df["onset"]) == original_order
    assert list(sorted_ann._df["onset"]) == sorted(original_order, reverse=True)


def test_filter_by_lead_does_not_mutate_source():
    """filter_by_lead must return a new Annotations without affecting the original."""
    ann = Annotations(_make_df())
    filtered = ann.filter_by_lead("I")
    filtered["onset"] = 999
    assert (ann._df["onset"] != 999).all()
    assert list(filtered._df["lead"].unique()) == ["I"]


def test_copy_is_independent():
    """copy() must return an independent Annotations."""
    ann = Annotations(_make_df())
    c = ann.copy()
    c["onset"] = 999
    assert (ann._df["onset"] != 999).all()


def test_groupby_yields_annotations():
    """groupby must yield (key, Annotations) pairs."""
    ann = Annotations(_make_df())
    for lead, group in ann.groupby("lead"):
        assert isinstance(group, Annotations)
        assert (group._df["lead"] == lead).all()


def test_loc_mask_returns_annotations():
    """loc with a boolean mask must return an Annotations."""
    ann = Annotations(_make_df())
    result = ann.loc[ann._df["lead"] == "I"]
    assert isinstance(result, Annotations)
    assert list(result._df["lead"].unique()) == ["I"]


def test_getitem_column_returns_series():
    """Column access returns a raw Series, not Annotations."""
    ann = Annotations(_make_df())
    col = ann["onset"]
    assert isinstance(col, pd.Series)


def test_drop_inplace_removes_rows():
    """drop(inplace=True) removes rows from _df in place."""
    ann = Annotations(_make_df())
    idx_to_drop = ann._df.index[0]
    ann.drop([idx_to_drop], inplace=True)
    assert idx_to_drop not in ann._df.index
    assert len(ann) == 3


def test_vcg_annotations_does_not_mutate_source():
    """vcg_annotations must not modify the original Annotations."""
    ann = Annotations(_make_df())
    leads_before = list(ann._df["lead"])
    ann.vcg_annotations()
    assert list(ann._df["lead"]) == leads_before


def test_setitem_mutates_in_place():
    """__setitem__ must mutate _df directly (used by pipeline write-back)."""
    ann = Annotations(_make_df())
    ann["onset"] = 42
    assert (ann._df["onset"] == 42).all()


