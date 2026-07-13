import logging
import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QFileDialog

from ecg_annotator.annotator import ECGAnnotator
from ecg_annotator.config import load_config
from ecg_annotator.logging_config import setup_logging
from ecg_annotator.plotter import ECGPlotter

log = logging.getLogger(__name__)


def main():
    config = load_config()

    if len(sys.argv) > 1:
        working_dir = Path(sys.argv[1])
    else:
        _app = QApplication.instance() or QApplication(sys.argv)
        _selected = QFileDialog.getExistingDirectory(None, "Select DICOM directory")
        if not _selected:
            sys.exit(0)
        working_dir = Path(_selected)

    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        output_file = Path(config.output_path).expanduser()

    setup_logging(output_file.parent)

    log.info("Starting ecg-annotator | dir=%s | output=%s", working_dir, output_file)
    log.info(
        "Config: distance_threshold=%s  default_confidence=%s  amplitude_scale=%s",
        config.distance_threshold,
        config.default_confidence,
        config.amplitude_scale,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title="ECG Annotator")
    win.setBackground("w")
    plot = win.addPlot()
    plot.setMenuEnabled(False)
    plotter = ECGPlotter(win, amplitude_scale=config.amplitude_scale, plot=plot)

    annotator = ECGAnnotator(working_dir, output_file, config, plotter)

    plotter.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
