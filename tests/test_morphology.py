import numpy as np
from atriakit.features.morphology import extract_sign_groups

segment = np.zeros(20)  # Dummy segment, not used in this case
segment[0] = 100

def test_empty_input():
    assert extract_sign_groups([], duration_threshold=0.1, segment=segment) == []

def test_single_sign_all_true():
    signs = [True] * 20
    assert extract_sign_groups(signs, duration_threshold=0.1, segment=segment) == [True]


def test_single_sign_all_false():
    signs = [False] * 30
    segment = np.zeros(30)  # Dummy segment, not used in this case
    segment[0] = 100
    assert extract_sign_groups(signs, duration_threshold=0.1, segment=segment) == [False]


def test_noise_block_below_threshold():
    # 2 True samples (noise), 18 False samples (real), len = 20
    # threshold = 0.1 → minimum block = 2 samples
    # 2 True meets threshold (equal) → decide whether it should count
    # If you want 2-sample blocks to be discarded, set threshold > 0.1 in real usage.
    signs = [True] * 2 + [False] * 18
    output = extract_sign_groups(signs, duration_threshold=0.15, segment=segment)  # min block = 3
    assert output == [False]


def test_biphasic_valid_two_blocks():
    # True(10) then False(10) with threshold 0.1 → both valid
    signs = [True] * 10 + [False] * 10
    result = extract_sign_groups(signs, duration_threshold=0.1, segment=segment)
    assert result == [True, False]


def test_three_phase_pattern_with_noise_block_removed():
    # True(2) -> False(15) -> True(13); threshold=0.1 means min block >=3
    signs = [True]*2 + [False]*15 + [True]*13
    segment = np.zeros(len(signs))
    segment[0] = 100
    result = extract_sign_groups(signs, duration_threshold=0.1, segment=segment)
    assert result == [False, True]  # initial 2-True block removed


def test_deduplication_no_double_entries():
    # Blocks: True(12), True(5), False(10)
    # After thresholding, the two True blocks should merge to one.
    signs = [True]*12 + [True]*5 + [False]*10
    result = extract_sign_groups(signs, duration_threshold=0.05, segment=segment)
    assert result == [True, False]


def test_end_block_threshold():
    # Last block is below threshold → should NOT be included
    signs = [True]*20 + [False]*2
    segment = np.zeros(len(signs))
    segment[0] = 100
    segment[-1] = 100
    result = extract_sign_groups(signs, duration_threshold=0.1, segment=segment)  # threshold block ≥2.1 → require ≥3
    assert result == [True]


def test_multiple_noise_blocks():
    # True(2), False(2), True(20)
    signs = [True]*2 + [False]*2 + [True]*20
    segment = np.zeros(len(signs))
    segment[0] = 100
    result = extract_sign_groups(signs, duration_threshold=0.1, segment=segment)  # threshold block ≥3
    assert result == [True]  # only the last block survives