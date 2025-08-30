"""
Logging utilities for the ML research pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Try to import colorlog, fall back to standard logging if not available
try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        log_format: Custom log format string
        use_colors: Whether to use colored output for console

    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Default format
    if log_format is None:
        if use_colors:
            log_format = (
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - "
                "%(message)s%(reset)s"
            )
        else:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    if use_colors and HAS_COLORLOG:
        formatter = colorlog.ColoredFormatter(
            log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        # Fall back to standard formatter
        if use_colors and not HAS_COLORLOG:
            # Remove color codes from format if colorlog not available
            log_format = log_format.replace("%(log_color)s", "").replace(
                "%(reset)s", ""
            )

        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Use non-colored format for file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)

    def log_info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def log_debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)

    def log_warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def log_error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def log_critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
