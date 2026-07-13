"""Tests for pure computation functions in ECGPlotter.

plotter.py is excluded from coverage (omit in pyproject.toml) because the
bulk of it is rendering/widget code that cannot be unit-tested without a real
display pipeline. These tests verify correctness of the handful of pure
functions so regressions are caught.
"""

import math

import numpy as np
import pytest
from ecg_annotator.plotter import ECGPlotter, _build_parallel_line_data, _snap_to_step

TIME_SEC = np.arange(0, 10, 0.001)  # 10 s at 1 kHz


def _plotter(time_sec=TIME_SEC):
    return ECGPlotter(parent_widget=None, time_sec=time_sec, plot=None)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_snap_to_step():
    assert _snap_to_step(0.0, 0.04) == pytest.approx(0.0)
    assert _snap_to_step(0.123, 0.04) == pytest.approx(0.12)  # round(3.075) = 3
    assert _snap_to_step(0.16, 0.04) == pytest.approx(0.16)


def test_build_parallel_line_data():
    # vertical=True: each value → two x-coords equal to value, one NaN
    xs, ys = _build_parallel_line_data(
        np.array([0.5]), start=-1.0, end=1.0, vertical=True
    )
    assert len(xs) == 3
    assert xs[0] == pytest.approx(0.5) and xs[1] == pytest.approx(0.5)
    assert math.isnan(xs[2])
    assert ys[0] == pytest.approx(-1.0) and ys[1] == pytest.approx(1.0)

    # vertical=False: each value → two y-coords equal to value, one NaN
    xs, ys = _build_parallel_line_data(
        np.array([2.0]), start=0.0, end=10.0, vertical=False
    )
    assert len(ys) == 3
    assert ys[0] == pytest.approx(2.0) and ys[1] == pytest.approx(2.0)
    assert math.isnan(ys[2])
    assert xs[0] == pytest.approx(0.0) and xs[1] == pytest.approx(10.0)

    # Two values → 6 elements, NaN at index 2 and 5
    xs, ys = _build_parallel_line_data(
        np.array([1.0, 3.0]), start=0.0, end=5.0, vertical=True
    )
    assert len(xs) == 6
    assert math.isnan(xs[2]) and math.isnan(xs[5])


# ---------------------------------------------------------------------------
# ECGPlotter pure methods
# ---------------------------------------------------------------------------


def test_scale_lead():
    p = _plotter()
    data = np.array([1.0, -1.0])

    # ECG lead: amplitude_scale=10.0, vertical_spacing=-5 per lead
    result = p.scale_lead(data, 0)
    assert result == pytest.approx(data * 10.0 + 0 * (-5))

    result = p.scale_lead(data, 1)
    assert result == pytest.approx(data * 10.0 + 1 * (-5))

    # Higher index: same uniform scaling
    result = p.scale_lead(data, 12)
    assert result == pytest.approx(data * 10.0 + 12 * (-5))


def test_scale_pfa_lead():
    p = _plotter()
    data = np.array([1.0, -1.0])
    n_main = 12

    # PFA lead 0: no amplitude scaling, just vertical offset
    result = p.scale_pfa_lead(data, 0, n_main)
    expected = data + (n_main + p.PFA_EXTRA_GAP_LEADS) * (-5)
    assert result == pytest.approx(expected)

    # PFA lead 1: one extra vertical_spacing step
    result = p.scale_pfa_lead(data, 1, n_main)
    expected = data + (n_main + p.PFA_EXTRA_GAP_LEADS + 1) * (-5)
    assert result == pytest.approx(expected)


def test_effective_lead_count():
    p = _plotter()

    # No PFA leads: effective count is just the main lead count.
    assert p.effective_lead_count(12, 0) == 12
    assert p.effective_lead_count(5, 0) == 5

    # PFA leads present: main + gap + PFA leads.
    assert p.effective_lead_count(12, 5) == pytest.approx(
        12 + p.PFA_EXTRA_GAP_LEADS + 5
    )
    assert p.effective_lead_count(5, 2) == pytest.approx(5 + p.PFA_EXTRA_GAP_LEADS + 2)


def test_clamp_window_start():
    # Empty time axis → always 0.0
    p_empty = _plotter(time_sec=np.array([]))
    assert p_empty._clamp_window_start(5.0) == pytest.approx(0.0)

    # 100 s signal, 30 s window → upper bound = 70.0
    long_time = np.arange(0, 100, 0.001)
    p = _plotter(time_sec=long_time)
    p.window_duration = 30.0
    assert p._clamp_window_start(50.0) == pytest.approx(50.0)  # inside
    upper = float(long_time[-1]) - 30.0
    assert p._clamp_window_start(95.0) == pytest.approx(upper)  # clipped to upper bound
    assert p._clamp_window_start(-1.0) == pytest.approx(0.0)  # clipped to lower bound


def test_window_slice():
    long_time = np.arange(0, 100, 0.001)  # 100 s at 1 kHz
    p = _plotter(time_sec=long_time)
    p.window_duration = 30.0

    ws = p._window_slice(5.0)
    assert ws.start_time == pytest.approx(5.0)
    assert ws.end_time == pytest.approx(35.0)
    # x_values are relative to start_time
    assert ws.x_values[0] == pytest.approx(0.0, abs=1e-6)
    assert ws.x_values[-1] == pytest.approx(30.0, abs=0.002)
    # indices must be contiguous
    assert len(ws.indices) == len(ws.x_values)


def test_snap_to_line():
    p = _plotter()  # time_sec 0…9.999 at 1 kHz, 10_000 samples

    # Exact sample → snaps to that sample
    t, idx = p.snap_to_line(0.001, len(p.time_sec))
    assert t == pytest.approx(0.001, abs=1e-9)
    assert idx == 1

    # x_click beyond signal end → clamps to last sample
    t_last, idx_last = p.snap_to_line(100.0, len(p.time_sec))
    assert idx_last == len(p.time_sec) - 1
    assert t_last == pytest.approx(p.time_sec[-1])


# ---------------------------------------------------------------------------
# Variable lead count — requires Qt widgets (qtbot ensures QApplication exists)
# ---------------------------------------------------------------------------


def test_update_lead_controls_rebuilds_rows(qtbot):
    """update_lead_controls clears old rows and creates exactly N new ones."""
    from PyQt6.QtWidgets import QVBoxLayout

    p = _plotter()
    p._lead_rows_layout = QVBoxLayout()
    # Build initial 12-lead rows
    for i in range(12):
        p._lead_rows_layout.addWidget(p._build_lead_controls_row(i, f"L{i}"))
    assert len(p.ignore_checks) == 12

    p.update_lead_controls([f"X{i}" for i in range(5)])

    assert len(p.ignore_checks) == 5
    assert len(p.radio_buttons) == 5
    assert len(p.get_ignored_leads()) == 5


def test_update_lead_controls_noop_when_not_initialised(qtbot):
    """update_lead_controls does not raise before add_ignore_checkbuttons is called."""
    p = _plotter()
    p.ignore_checks = None
    p.update_lead_controls(["A", "B", "C"])  # _lead_rows_layout is None → no-op
    assert p.ignore_checks is None and p.ignore_checks != []
