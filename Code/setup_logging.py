"""Logger options."""

import logging
import sys
import io
from pathlib import Path
from datetime import datetime as dt

LOG_SAVE_PATH = Path(__file__).parents[1]
LOG_LEVEL_FILE = logging.DEBUG
LOG_LEVEL_CONSOLE = logging.DEBUG


class LoggerWriter(io.TextIOBase):
    """Custom class to redirect output to logger"""

    def __init__(self, logger, level) -> None:
        self.logger = logger
        self.level = level

    def write(self, message) -> None:
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())


def setup_logging(logger_name: str) -> logging.Logger:
    """
    Logger that simultaneously writes to a log file
    and the standard console output.
    """

    # Create the logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL_CONSOLE)

    timestamp = dt.now().strftime("%Y%m%d-%H%M%S")

    # Create a file handler that writes to a file
    file_handler = logging.FileHandler(LOG_SAVE_PATH / f"{logger_name}_{timestamp}.log")
    file_handler.setLevel(LOG_LEVEL_FILE)

    # Create a stream handler that prints to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL_CONSOLE)

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the formatter for both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect stdout and stderr to the logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    return logger
