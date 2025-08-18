"""
SIGALRM-based wall-clock timeout for *Unix-like* systems (macOS, Linux).

Usage
-----
    with time_limit(30):
        long_running_call()
"""
from __future__ import annotations
import contextlib
import signal
from types import FrameType
from typing import Generator, Optional


@contextlib.contextmanager
def time_limit(seconds: float | None) -> Generator[None, None, None]:
    def _raise_timeout(signum: int, frame: Optional[FrameType]) -> None:  # noqa: D401
        raise TimeoutError("time limit reached")

    if seconds is None or seconds <= 0:
        # No watchdog – behave as a no-op context manager
        yield
        return

    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        # Always disable the timer & restore handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original_handler)
