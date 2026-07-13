import numpy as np
import pytest
import pandas as pd
from atriakit.utils import match_lead_annotations, convert_ecg_segment_to_vcg



# --- Fixtures for sample lead annotations ---
@pytest.fixture
def lead_1_df():
    return pd.DataFrame({
        "offset": [1, 2, 3],
        "onset": [10, 30, 50],
        "p_wave_id": [1, 2, 3]
    })

@pytest.fixture
def lead_2_df():
    return pd.DataFrame({
        "offset": [10, 20, 30],
        "onset": [12, 32, 60],
        "p_wave_id": [101, 102, 103]
    })

# --- Test normal pairing ---
def test_match_lead_annotations_basic(lead_1_df, lead_2_df):
    paired = match_lead_annotations(lead_1_df, lead_2_df, max_len=5)
    
    # Only first two should pair: diff = 2, 2, 10 → last exceeds max_len
    assert len(paired) == 2
    # Check the paired onsets
    paired_onsets = [(p[0]['onset'], p[1]['onset']) for p in paired]
    assert paired_onsets == [(10, 12), (30, 32)]

# --- Test no matches ---
def test_match_lead_annotations_no_matches(lead_1_df):
    lead_2 = pd.DataFrame({
        "offset": [10, 20],
        "onset": [100, 200],
        "p_wave_id": [101, 102]
    })
    paired = match_lead_annotations(lead_1_df, lead_2, max_len=5)
    assert len(paired) == 0

# --- Test unequal lengths ---
def test_match_lead_annotations_unequal_lengths(lead_1_df):
    lead_2 = pd.DataFrame({
        "offset": [10, 20],
        "onset": [12, 32],
        "p_wave_id": [101, 102]
    })
    paired = match_lead_annotations(lead_1_df, lead_2, max_len=5)
    assert len(paired) == 2
    paired_onsets = [(p[0]['onset'], p[1]['onset']) for p in paired]
    assert paired_onsets == [(10, 12), (30, 32)]

# --- Test empty inputs ---
def test_match_lead_annotations_empty():
    paired = match_lead_annotations(pd.DataFrame(columns=["offset","onset"]), 
                                    pd.DataFrame(columns=["offset","onset"]))
    assert len(paired) == 0

    paired = match_lead_annotations(pd.DataFrame(columns=["offset","onset"]),
                                    pd.DataFrame({"offset":[1], "onset":[10]}))
    assert len(paired) == 0


# ---------------------------------------------------------------------------
# convert_ecg_segment_to_vcg
# ---------------------------------------------------------------------------

_KORS_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
_KORS_LTI = {name: i for i, name in enumerate(_KORS_LEADS)}


def test_vcg_raises_when_kors_leads_missing():
    ecg = np.ones((3, 100))
    lead_to_index = {"I": 0, "aVF": 1, "II": 2}  # missing V1–V6
    with pytest.raises(ValueError, match="V1"):
        convert_ecg_segment_to_vcg(ecg, lead_to_index)


def test_vcg_succeeds_with_all_kors_leads():
    ecg = np.random.default_rng(0).standard_normal((8, 200))
    result = convert_ecg_segment_to_vcg(ecg, _KORS_LTI)
    assert result.shape == (3, 200)


def test_vcg_succeeds_when_extra_leads_present():
    # 9 leads: 8 Kors + V9; V9 should be ignored
    lead_to_index = {**_KORS_LTI, "V9": 8}
    ecg = np.random.default_rng(1).standard_normal((9, 200))
    result = convert_ecg_segment_to_vcg(ecg, lead_to_index)
    assert result.shape == (3, 200)
