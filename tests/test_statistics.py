# Tests for training/statistics.py's confidence interval and significance testing helpers.

import pytest

from training.statistics import (
    holm_adjusted_p_values,
    mean_and_std,
    t_confidence_interval_across_seeds,
    two_proportion_z_test,
    wilson_confidence_interval,
)


# Checks the Wilson interval's point estimate matches the raw proportion and brackets it.
def test_wilson_interval_point_estimate_and_bounds():
    ci = wilson_confidence_interval(wins=10, num_episodes=100)
    assert ci.point_estimate == pytest.approx(0.1)
    assert ci.lower < ci.point_estimate < ci.upper
    assert 0.0 <= ci.lower and ci.upper <= 1.0


# Checks a bigger sample at the same proportion gives a tighter (narrower) interval.
def test_wilson_interval_narrows_with_more_episodes():
    narrow = wilson_confidence_interval(wins=100, num_episodes=1000)
    wide = wilson_confidence_interval(wins=10, num_episodes=100)
    assert (narrow.upper - narrow.lower) < (wide.upper - wide.lower)


# Checks two identical proportions give z=0 and p=1 (no evidence of any difference).
def test_z_test_identical_proportions_gives_zero_z_and_p_one():
    result = two_proportion_z_test(wins_a=20, n_a=200, wins_b=20, n_b=200)
    assert result.z == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


# Checks a clearly large, well-powered difference in win rate is flagged significant.
def test_z_test_flags_a_large_well_powered_difference_as_significant():
    result = two_proportion_z_test(wins_a=400, n_a=1000, wins_b=100, n_b=1000)
    assert result.p_value < 0.001


# Checks mean_and_std matches hand-calculated values for a small known sample.
def test_mean_and_std_matches_hand_calculation():
    stats = mean_and_std([0.1, 0.2, 0.3])
    assert stats.mean == pytest.approx(0.2)
    assert stats.std == pytest.approx(0.1)


# Checks a single-value sample has zero std rather than raising or returning NaN.
def test_mean_and_std_single_value_has_zero_std():
    stats = mean_and_std([0.5])
    assert stats.mean == pytest.approx(0.5)
    assert stats.std == 0.0


# Checks the across-seed interval is centred on the seed mean and far wider than the pooled
# interval over the same games - the whole reason it exists.
def test_across_seed_interval_is_wider_than_the_pooled_binomial_interval():
    win_rates = [0.06, 0.08, 0.06, 0.08, 0.12]
    seed_ci = t_confidence_interval_across_seeds(win_rates)
    pooled_ci = wilson_confidence_interval(wins=800, num_episodes=10000)

    assert seed_ci.point_estimate == pytest.approx(0.08)
    assert seed_ci.lower < 0.08 < seed_ci.upper
    assert (seed_ci.upper - seed_ci.lower) > 5 * (pooled_ci.upper - pooled_ci.lower)


# Checks a single seed yields a zero-width interval rather than raising.
def test_across_seed_interval_with_one_seed_is_zero_width():
    ci = t_confidence_interval_across_seeds([0.07])
    assert ci.lower == ci.upper == ci.point_estimate == pytest.approx(0.07)


# Checks Holm scales the smallest p-value by the number of tests and leaves ordering intact.
def test_holm_adjustment_inflates_p_values_and_preserves_order():
    adjusted = holm_adjusted_p_values([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert all(a >= p for a, p in zip(adjusted, [0.01, 0.04, 0.03]))


# Checks Holm never reports a probability above 1.
def test_holm_adjustment_is_capped_at_one():
    assert all(p <= 1.0 for p in holm_adjusted_p_values([0.5, 0.6, 0.9]))
