"""
Pluggable Clock abstraction for deterministic testing and distributed time.

Provides a clean abstraction over time.time() with support for:
- System clock (default)
- Deterministic clock (for testing)
- HLC-backed distributed clock (future)
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class Clock(ABC):
    """
    Abstract base class for clock implementations.

    All time-dependent code should use a Clock instance instead of
    calling time.time() directly.
    """

    @abstractmethod
    def now(self) -> float:
        """
        Get current time in seconds since epoch.

        Returns:
            float: Current time in seconds.
        """
        pass

    @abstractmethod
    def now_ns(self) -> int:
        """
        Get current time in nanoseconds since epoch.

        Returns:
            int: Current time in nanoseconds.
        """
        pass

    @abstractmethod
    def sleep(self, duration: float) -> None:
        """
        Sleep for specified duration.

        Args:
            duration: Sleep duration in seconds.
        """
        pass


class SystemClock(Clock):
    """
    System clock implementation using time.time().

    This is the default clock used in production. It delegates to the
    system's real-time clock.
    """

    def now(self) -> float:
        """Get current system time."""
        return time.time()

    def now_ns(self) -> int:
        """Get current system time in nanoseconds."""
        return time.time_ns()

    def sleep(self, duration: float) -> None:
        """Sleep using system sleep."""
        time.sleep(duration)


class DeterministicClock(Clock):
    """
    Deterministic clock for testing.

    Provides full control over time, allowing tests to:
    - Set arbitrary timestamps
    - Advance time programmatically
    - Ensure deterministic behavior

    Example:
        >>> clock = DeterministicClock(start_time=1000.0)
        >>> clock.now()
        1000.0
        >>> clock.advance(10.0)
        >>> clock.now()
        1010.0
    """

    def __init__(self, start_time: float = 0.0):
        """
        Initialize deterministic clock.

        Args:
            start_time: Initial time in seconds since epoch.
        """
        self._current_time = start_time
        self._sleep_count = 0

    def now(self) -> float:
        """Get current virtual time."""
        return self._current_time

    def now_ns(self) -> int:
        """Get current virtual time in nanoseconds."""
        return int(self._current_time * 1_000_000_000)

    def sleep(self, duration: float) -> None:
        """
        Virtual sleep (advances time without blocking).

        Args:
            duration: Duration to advance time.
        """
        self._current_time += duration
        self._sleep_count += 1

    def advance(self, duration: float) -> None:
        """
        Advance time by specified duration.

        Args:
            duration: Duration to advance in seconds.
        """
        self._current_time += duration

    def set_time(self, timestamp: float) -> None:
        """
        Set absolute time.

        Args:
            timestamp: New time in seconds since epoch.
        """
        self._current_time = timestamp

    def get_sleep_count(self) -> int:
        """
        Get number of sleep() calls.

        Useful for testing that code sleeps expected number of times.

        Returns:
            int: Number of sleep calls.
        """
        return self._sleep_count


class MonotonicClock(Clock):
    """
    Monotonic clock implementation.

    Uses time.monotonic() which is guaranteed to never go backwards,
    making it suitable for measuring intervals even if system time
    is adjusted.
    """

    def __init__(self):
        """Initialize monotonic clock with reference point."""
        self._reference = time.monotonic()
        self._epoch_offset = time.time() - self._reference

    def now(self) -> float:
        """Get current monotonic time (converted to epoch-like value)."""
        return time.monotonic() + self._epoch_offset

    def now_ns(self) -> int:
        """Get current monotonic time in nanoseconds."""
        return time.monotonic_ns() + int(self._epoch_offset * 1_000_000_000)

    def sleep(self, duration: float) -> None:
        """Sleep using system sleep."""
        time.sleep(duration)


@dataclass
class ClockContext:
    """
    Context manager for temporarily using a different clock.

    Example:
        >>> with ClockContext(DeterministicClock()):
        ...     # Code here uses deterministic clock
        ...     pass
    """

    clock: Clock
    _previous_clock: Optional[Clock] = None

    def __enter__(self):
        """Set clock for context."""
        global _global_clock
        self._previous_clock = _global_clock
        _global_clock = self.clock
        return self.clock

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous clock."""
        global _global_clock
        _global_clock = self._previous_clock


# Global clock instance (defaults to SystemClock)
_global_clock: Clock = SystemClock()


def get_clock() -> Clock:
    """
    Get the current global clock instance.

    Returns:
        Clock: Current clock instance.
    """
    return _global_clock


def set_clock(clock: Clock) -> None:
    """
    Set the global clock instance.

    Args:
        clock: Clock instance to use globally.
    """
    global _global_clock
    _global_clock = clock


def now() -> float:
    """
    Get current time from global clock.

    Returns:
        float: Current time in seconds since epoch.
    """
    return _global_clock.now()


def now_ns() -> int:
    """
    Get current time from global clock in nanoseconds.

    Returns:
        int: Current time in nanoseconds since epoch.
    """
    return _global_clock.now_ns()


def sleep(duration: float) -> None:
    """
    Sleep for specified duration using global clock.

    Args:
        duration: Sleep duration in seconds.
    """
    _global_clock.sleep(duration)


# Export public API
__all__ = [
    'Clock',
    'SystemClock',
    'DeterministicClock',
    'MonotonicClock',
    'ClockContext',
    'get_clock',
    'set_clock',
    'now',
    'now_ns',
    'sleep'
]
