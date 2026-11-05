"""Logging of the entry points.

Library modules log through `logging.getLogger(__name__)` and never configure logging.
Each command-line entry point calls `setup_logging` once: messages then go to standard
output, without prefix, so that `run.log` reads as before. A user of the library in a
notebook can call it too, or configure logging as usual.
"""

import logging
import sys


class _StdoutHandler(logging.StreamHandler):
    """Writes to the current sys.stdout (also when it is replaced, as in tests)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def setup_logging(level: int = logging.INFO) -> None:
    """Send INFO messages and above to stdout, as plain lines (idempotent)."""
    root = logging.getLogger()
    if any(getattr(h, "_rfl", False) for h in root.handlers):
        return
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._rfl = True
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("mne", "moabb", "matplotlib", "urllib3", "pooch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
