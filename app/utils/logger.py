"""
Centralized logging configuration for the Enterprise HR AI application.
"""
import logging
import sys
from app.utils.config import LOG_LEVEL, LOG_FORMAT


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


# App lifecycle logger
app_logger = get_logger("hr_ai.app")
api_logger = get_logger("hr_ai.api")
ml_logger = get_logger("hr_ai.ml")
