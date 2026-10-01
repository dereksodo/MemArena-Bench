"""Runtime helpers shared by MemArena command-line entry points."""

from __future__ import annotations

import os
import sys
from typing import TextIO


def _configure_stream(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


def configure_live_output() -> None:
    """Prefer live CLI/log output over buffered writes."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _configure_stream(sys.stdout)
    _configure_stream(sys.stderr)


def flush_standard_streams() -> None:
    """Best-effort flush for stdout/stderr."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
