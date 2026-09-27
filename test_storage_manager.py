import logging
from rich.logging import RichHandler
from clearml.storage.manager import StorageManager

def test_storage_manager_does_not_break_root_logger(tmp_path):
    log_file = tmp_path / "test.log"
    local_file = tmp_path / "dummy.txt"
    local_file.write_text("hello")

    handlers = [
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(log_file, mode='w')
    ]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)

    logging.info("before download")
    StorageManager.get_local_copy(str(local_file))
    logging.info("after download")

    with open(log_file) as f:
        contents = f.read()
    assert "before download" in contents, "'before download' missing from log file"
    assert "after download" in contents, "'after download' missing from log file"