import pytest
from ecg_annotator.utils.time import time_to_sample, sec_to_hms


def test_time_to_sample():
    assert time_to_sample(1.5, 1000, 5000) == 1500
    assert time_to_sample(-1.0, 1000, 5000) == 0       # clamp low
    assert time_to_sample(10.0, 1000, 5000) == 4999    # clamp high


def test_sec_to_hms_scalar():
    # start 01:00:00 + 3661s = 02:01:01
    assert sec_to_hms(3661.0, "010000") == "02:01:01"


def test_sec_to_hms_array():
    result = sec_to_hms([0.0, 60.0], "000000")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "00:00:00"
    assert result[1] == "00:01:00"
