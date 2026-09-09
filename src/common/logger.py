"""Logging utility for SemEval-2027 Task 3."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "semeval2027", level: int = logging.INFO) -> logging.Logger:
    """Returns a standardized logger with formatted output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
