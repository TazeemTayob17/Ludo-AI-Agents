# Tests for env/dice.py's die-rolling function.

import numpy as np

from env.dice import roll_die


# Checks every roll lands in the valid 1-6 range.
def test_roll_die_stays_in_range():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        assert 1 <= roll_die(rng) <= 6


# Checks the roll distribution is roughly uniform over many draws.
def test_roll_die_distribution_is_roughly_uniform():
    rng = np.random.default_rng(1)
    counts = {face: 0 for face in range(1, 7)}
    num_rolls = 60000
    for _ in range(num_rolls):
        counts[roll_die(rng)] += 1
    expected = num_rolls / 6
    for count in counts.values():
        assert abs(count - expected) / expected < 0.05


# Checks the same seed produces the same sequence of rolls.
def test_roll_die_is_deterministic_given_a_seed():
    rolls_a = [roll_die(np.random.default_rng(42)) for _ in range(50)]
    rolls_b = [roll_die(np.random.default_rng(42)) for _ in range(50)]
    assert rolls_a == rolls_b
