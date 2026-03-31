"""Shared utility helpers for optimization control modules."""

from collections import deque
from typing import Deque, Iterable, List


def bounded(value: float, lower: float, upper: float) -> float:
    """Clamp value to closed interval [lower, upper]."""
    return max(lower, min(upper, value))


def moving_average(window: Deque[float]) -> float:
    """Compute arithmetic mean over a non-empty deque."""
    if not window:
        return 0.0
    return sum(window) / len(window)


def stddev(values: Iterable[float]) -> float:
    """Compute population standard deviation with empty-input safety."""
    sequence: List[float] = [float(value) for value in values]
    if not sequence:
        return 0.0
    mean_value = sum(sequence) / len(sequence)
    variance = sum((value - mean_value) ** 2 for value in sequence) / len(sequence)
    return variance**0.5


def make_window(maxlen: int) -> Deque[float]:
    """Create bounded deque window with validated size."""
    return deque(maxlen=max(2, int(maxlen)))
