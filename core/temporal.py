"""Bounded histories and fresh-observation counters."""
from math import hypot


def append_sample(history, frame, value, limit):
    history.append((frame, value))
    while len(history) > limit:
        history.popleft()


def motion(history, frame, window):
    """Average pixels/frame; a detection gap starts a new motion segment."""
    samples = list(history)
    if len(samples) < 2 or samples[-1][0] != frame:
        return (0.0, 0.0), False
    tail = [samples[-1]]
    for sample in reversed(samples[:-1]):
        if tail[-1][0] - sample[0] != 1 or len(tail) > window:
            break
        tail.append(sample)
    if len(tail) < 2:
        return (0.0, 0.0), False
    end, start = tail[0], tail[-1]
    dt = end[0] - start[0]
    return ((end[1][0] - start[1][0]) / dt, (end[1][1] - start[1][1]) / dt), True


def magnitude(vector):
    return hypot(*vector)


def count(runtime, name, condition, *, fresh=True, hold=False):
    """Unknown during a grace period pauses; false evidence resets. Never adds synthetic frames."""
    if not fresh:
        if not hold:
            runtime.counters[name] = 0
    else:
        runtime.counters[name] = runtime.counters.get(name, 0) + 1 if condition else 0
    return runtime.counters.get(name, 0)
