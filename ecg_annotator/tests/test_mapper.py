from ecg_annotator.annotations.mapper import map_red_to_green_with_gaps


def _pts(*xs):
    """Build a point list from x-coordinates (y=0)."""
    return [(x, 0) for x in xs]


def test_simple_pair():
    mapped, unred, ungreen = map_red_to_green_with_gaps(_pts(0.1), _pts(0.2))
    assert mapped == {0.1: 0.2}
    assert not unred and not ungreen


def test_green_before_red_is_unmapped():
    # green at 0.0 comes before the only red at 0.1 → green unmapped
    mapped, unred, ungreen = map_red_to_green_with_gaps(_pts(0.1), _pts(0.0, 0.2))
    assert mapped == {0.1: 0.2}
    assert 0.0 in ungreen and not unred


def test_gap_rule_unmaps_earlier_red():
    # two reds at 0.1 and 0.2, one green at 0.3
    # 0.1 has a red (0.2) between it and the green → 0.1 is unmapped
    mapped, unred, ungreen = map_red_to_green_with_gaps(_pts(0.1, 0.2), _pts(0.3))
    assert mapped == {0.2: 0.3}
    assert 0.1 in unred and not ungreen


def test_multiple_non_overlapping_pairs():
    mapped, unred, ungreen = map_red_to_green_with_gaps(
        _pts(0.1, 0.5), _pts(0.2, 0.6)
    )
    assert mapped == {0.1: 0.2, 0.5: 0.6}
    assert not unred and not ungreen


def test_trailing_unpaired_red():
    # second red has no following green → unmapped
    mapped, unred, ungreen = map_red_to_green_with_gaps(_pts(0.1, 0.5), _pts(0.2))
    assert mapped == {0.1: 0.2}
    assert 0.5 in unred and not ungreen


def test_trailing_unpaired_green():
    # second green has no preceding red → unmapped
    mapped, unred, ungreen = map_red_to_green_with_gaps(_pts(0.1), _pts(0.2, 0.9))
    assert mapped == {0.1: 0.2}
    assert 0.9 in ungreen and not unred
