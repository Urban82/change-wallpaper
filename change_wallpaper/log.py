# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Danilo Treffiletti

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

ROOT_LOGGER = "change-wallpaper"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_COUNT = 5


def get_logger(module: str) -> logging.Logger:
    """Logger named change-wallpaper.<name> for a change_wallpaper.<name> module."""
    return logging.getLogger(f"{ROOT_LOGGER}.{module.rsplit('.', 1)[-1]}")


class IsoFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    formatter = IsoFormatter(LOG_FORMAT)
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        # current file + 4 backups = 5 files in total
        rotating = logging.handlers.RotatingFileHandler(
            path, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_COUNT - 1, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        logger.addHandler(rotating)
