import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(output_dir: Path) -> None:
    log_file = output_dir / "ecg-annotator.log"

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(_file_handler(log_file, formatter))
    root.addHandler(_stream_handler(formatter))

    logging.getLogger(__name__).info("Logging to %s", log_file)


def _file_handler(log_file: Path, formatter: logging.Formatter) -> logging.FileHandler:
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(formatter)
    return handler


def _stream_handler(formatter: logging.Formatter) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    return handler
