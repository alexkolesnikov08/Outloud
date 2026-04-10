"""Logging with levels, file output, and rotation."""

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "outloud-logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3


class OutLoudFormatter(logging.Formatter):
    """Formatter with colored levels."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        return f"{color}{record.levelname}{self.RESET}: {record.getMessage()}"


def get_logger(name: str) -> logging.Logger:
    """Get a logger with file + console handlers.

    File: all levels (DEBUG+)
    Console: WARNING+ only (to keep output clean)
    """
    logger = logging.getLogger(f"outloud.{name}")

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ─── File handler: all levels, rotating
    log_file = LOG_DIR / f"outloud_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT,
        encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)

    # ─── Error file: errors only
    error_file = LOG_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT,
        encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"))
    logger.addHandler(error_handler)

    # ─── Console handler: WARNING+ only
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(OutLoudFormatter())
    logger.addHandler(console)

    logger.propagate = False
    return logger
