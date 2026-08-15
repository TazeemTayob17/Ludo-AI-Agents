# Tests for training/schedules.py.

import pytest

from training.schedules import linear_epsilon


# Checks episode 0 gives exactly the start value.
def test_starts_at_start_value():
    assert linear_epsilon(0, start=1.0, end=0.1, decay_episodes=100) == pytest.approx(1.0)


# Checks reaching decay_episodes gives exactly the end value.
def test_reaches_end_value_at_decay_episodes():
    assert linear_epsilon(100, start=1.0, end=0.1, decay_episodes=100) == pytest.approx(0.1)


# Checks the value holds at the end value beyond decay_episodes, never overshooting.
def test_holds_at_end_value_past_decay_episodes():
    assert linear_epsilon(500, start=1.0, end=0.1, decay_episodes=100) == pytest.approx(0.1)


# Checks the midpoint is exactly halfway between start and end.
def test_midpoint_is_linear():
    assert linear_epsilon(50, start=1.0, end=0.0, decay_episodes=100) == pytest.approx(0.5)


# Checks a decay_episodes of zero jumps straight to the end value.
def test_zero_decay_episodes_jumps_to_end():
    assert linear_epsilon(0, start=1.0, end=0.1, decay_episodes=0) == pytest.approx(0.1)
