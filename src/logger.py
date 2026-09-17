import logging
import sys
from pathlib import Path
from src.config import LOGS_DIR

LOG_FORMAT = "[%(asctime)s] [PID:%(process)d] [%(name)s] [%(levelname)s] %(message)s"
LOG_FILE = LOGS_DIR / "tracker.log"


def setup_logging(level=logging.INFO) -> None:
    """
    Configures centralized logging for the application.
    Output is sent to both stdout and logs/tracker.log using square-bracket formatting:
    [%(asctime)s] [PID:%(process)d] [%(name)s] [%(levelname)s] %(message)s
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root_logger.addHandler(stream_handler)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

