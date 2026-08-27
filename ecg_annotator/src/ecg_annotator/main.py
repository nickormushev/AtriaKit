import logging
import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from ecg_annotator.annotator import ECGAnnotator
from ecg_annotator.config import load_config
from ecg_annotator.logging_config import setup_logging
from ecg_annotator.navigation import Navigator
from ecg_annotator.plotter import ECGPlotter

log = logging.getLogger(__name__)


def main():
    config = load_config()

    # No directory on the command line: start empty and let the user pick one
    # from the File menu (Open File / Open Folder).
    working_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

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
    win = pg.GraphicsLayoutWidget(title="ECG Annotator")
    win.setBackground("w")
    plot = win.addPlot()
    plot.setMenuEnabled(False)
    plotter = ECGPlotter(win, amplitude_scale=config.amplitude_scale, plot=plot)

    annotator = ECGAnnotator(working_dir, output_file, config, plotter)

    navigator = Navigator(annotator)
    navigator.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
