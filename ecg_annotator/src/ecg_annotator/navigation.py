"""Main window: hosts the annotator and a File menu for opening files/folders."""

from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMainWindow

from ecg_annotator.annotator import ECGAnnotator


class Navigator:
    """Owns the main window and the File menu.

    The annotator renders itself into ``plotter.main_container``; Navigator
    only hosts that widget in a QMainWindow and drives the open actions
    (Open File / Open Folder) that swap the loaded recordings.
    """

    def __init__(self, annotator: ECGAnnotator):
        self.annotator = annotator

        self.window = QMainWindow()
        self.window.setWindowTitle("ECG Annotator")
        self.window.setCentralWidget(annotator.plotter.main_container)
        self._build_menu_bar()

    def show(self) -> None:
        self.window.show()

    def _build_menu_bar(self) -> None:
        """Build the File menu with Open File / Open Folder actions."""
        file_menu = self.window.menuBar().addMenu("File")

        open_file_action = file_menu.addAction("Open File...")
        open_file_action.triggered.connect(self._open_file)

        open_folder_action = file_menu.addAction("Open Folder...")
        open_folder_action.triggered.connect(self._open_folder)

    def _open_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self.window, "Open Folder")
        if selected:
            self.annotator.open_directory(Path(selected))

    def _open_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self.window, "Open File")
        if selected:
            self.annotator.open_file(Path(selected))
