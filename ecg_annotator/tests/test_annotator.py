import pytest

from ecg_annotator.annotations.mapper import map_red_to_green_with_gaps


def test_basic_mapping():
    """
    red:   1, 2
    green: 3, 10
    Only red(2) → green(3)
    """
    red = [(1, 0), (2, 0)]
    green = [(3, 0), (10, 0)]

    mapped, unmapped_red, unmapped_green = map_red_to_green_with_gaps(red, green)

    assert mapped == {2: 3}
    assert unmapped_red == {1}
    assert unmapped_green == {10}


def test_two_valid_matches():
    """
    red:   1, 5
    green: 3, 10
    Both mappings valid
    """
    red = [(1, 0), (5, 0)]
    green = [(3, 0), (10, 0)]

    mapped, unmapped_red, unmapped_green = map_red_to_green_with_gaps(red, green)

    assert mapped == {1: 3, 5: 10}
    assert unmapped_red == set()
    assert unmapped_green == set()


def test_closest_valid_match():
    """
    red: 1
    green: 3, 10
    Should pick green(3) as closest
    """
    red = [(1, 0)]
    green = [(3, 0), (10, 0)]

    mapped, unmapped_red, unmapped_green = map_red_to_green_with_gaps(red, green)

    assert mapped == {1: 3}
    assert unmapped_red == set()
    assert unmapped_green == {10}


def test_red_blocked_by_next_red():
    """
    red:   1, 3
    green: 5
    red(1) blocked by red(3)
    Only red(3) → green(5)
    """
    red = [(1, 0), (3, 0)]
    green = [(5, 0)]

    mapped, unmapped_red, unmapped_green = map_red_to_green_with_gaps(red, green)

    assert mapped == {3: 5}
    assert unmapped_red == {1}
    assert unmapped_green == set()


def test_no_valid_matches():
    """
    All greens are to the left of all reds
    """
    red = [(5, 0), (10, 0)]
    green = [(1, 0), (2, 0)]

    mapped, unmapped_red, unmapped_green = map_red_to_green_with_gaps(red, green)

    assert mapped == {}
    assert unmapped_red == {5, 10}
    assert unmapped_green == {1, 2}
