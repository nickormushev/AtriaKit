import logging
import pytest

from ecg_annotator.logging_config import setup_logging


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    for handler in root.handlers:
        handler.close()
    root.handlers = original_handlers
    root.setLevel(original_level)


def test_log_file_created_in_output_dir(tmp_path):
    setup_logging(tmp_path)
    assert (tmp_path / "ecg-annotator.log").exists()


def test_log_file_contains_written_message(tmp_path):
    setup_logging(tmp_path)
    logging.getLogger("test").warning("sentinel message")

    content = (tmp_path / "ecg-annotator.log").read_text()
    assert "sentinel message" in content


def test_root_logger_has_file_and_stream_handlers(tmp_path):
    setup_logging(tmp_path)
    handler_types = {type(h) for h in logging.getLogger().handlers}
    assert logging.FileHandler in handler_types
    assert logging.StreamHandler in handler_types
