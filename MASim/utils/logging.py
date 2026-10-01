"""Structured logging with rich console output."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from memarena.runtime import configure_live_output

console = Console(stderr=True)


class FlushRichHandler(RichHandler):
    """Rich logging handler that flushes after every emitted record."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.console.file.flush()
        except Exception:
            pass


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging with rich handler."""
    configure_live_output()
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[FlushRichHandler(console=console, rich_tracebacks=True, show_path=False)],
        force=True,
    )
    # Suppress verbose HTTP request logs from networking libraries
    for noisy in ("urllib3", "httpx", "httpcore", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger = logging.getLogger("masim")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def get_logger(name: str = "masim") -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
