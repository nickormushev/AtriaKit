"""Tests for Navigator's window hosting and the File menu open actions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from ecg_annotator.navigation import Navigator

_app = QApplication.instance() or QApplication([])

_created_windows = []


@pytest.fixture(autouse=True)
def _cleanup_windows():
    """QMainWindow has no WA_DeleteOnClose, so top-level windows would
    otherwise accumulate across tests and can crash the offscreen platform."""
    yield
    for window in _created_windows:
        window.deleteLater()
    _created_windows.clear()
    _app.processEvents()


def _navigator():
    annotator = MagicMock()
    annotator.plotter.main_container = QWidget()
    nav = Navigator(annotator)
    _created_windows.append(nav.window)
    return nav, annotator


def test_hosts_the_annotator_container():
    nav, annotator = _navigator()
    assert nav.window.centralWidget() is annotator.plotter.main_container


def test_file_menu_structure():
    nav, _ = _navigator()
    menus = [action.text() for action in nav.window.menuBar().actions()]
    assert menus == ["File"]

    file_menu = nav.window.menuBar().actions()[0].menu()
    assert [action.text() for action in file_menu.actions()] == [
        "Open File...",
        "Open Folder...",
    ]


def test_open_folder_action_calls_open_directory():
    nav, annotator = _navigator()
    file_menu = nav.window.menuBar().actions()[0].menu()
    action = {a.text(): a for a in file_menu.actions()}["Open Folder..."]

    with patch(
        "ecg_annotator.navigation.QFileDialog.getExistingDirectory",
        return_value="/data/patients",
    ):
        action.trigger()

    annotator.open_directory.assert_called_once_with(Path("/data/patients"))
    annotator.open_file.assert_not_called()


def test_open_file_action_calls_open_file():
    nav, annotator = _navigator()
    file_menu = nav.window.menuBar().actions()[0].menu()
    action = {a.text(): a for a in file_menu.actions()}["Open File..."]

    with patch(
        "ecg_annotator.navigation.QFileDialog.getOpenFileName",
        return_value=("/data/patients/p001.IMA", ""),
    ):
        action.trigger()

    annotator.open_file.assert_called_once_with(Path("/data/patients/p001.IMA"))
    annotator.open_directory.assert_not_called()


def test_cancelled_dialogs_are_no_ops():
    nav, annotator = _navigator()
    file_menu = nav.window.menuBar().actions()[0].menu()
    actions = {a.text(): a for a in file_menu.actions()}

    with patch(
        "ecg_annotator.navigation.QFileDialog.getExistingDirectory", return_value=""
    ):
        actions["Open Folder..."].trigger()
    with patch(
        "ecg_annotator.navigation.QFileDialog.getOpenFileName", return_value=("", "")
    ):
        actions["Open File..."].trigger()

    annotator.open_directory.assert_not_called()
    annotator.open_file.assert_not_called()
