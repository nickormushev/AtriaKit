import numpy as np
import pandas as pd
import pytest

from atriakit.utils import convert_ecg_segment_to_vcg

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
