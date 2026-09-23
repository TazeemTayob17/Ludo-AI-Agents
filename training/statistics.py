# Significance testing and confidence intervals for tournament win rates (Step 11.3).
# Implemented from closed-form formulas (no scipy dependency) since the project's
# requirements.txt deliberately doesn't include it.

from __future__ import annotations

import math
from dataclasses import dataclass

_Z_95 = 1.959963984540054  # two-tailed 95% critical value, standard normal

# Standard normal CDF via the error function, used for two-tailed p-values below.
def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@dataclass
class ConfidenceInterval:
    point_estimate: float
    lower: float
    upper: float

# Wilson score interval for a binomial proportion - more reliable than the naive normal
# approximation when win counts are small (as they are here, e.g. ~3-6 wins out of 50).
def wilson_confidence_interval(wins: int, num_episodes: int, z: float = _Z_95) -> ConfidenceInterval:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    p = wins / num_episodes
    denominator = 1.0 + z**2 / num_episodes
    center = p + z**2 / (2 * num_episodes)
    spread = z * math.sqrt(p * (1 - p) / num_episodes + z**2 / (4 * num_episodes**2))
    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    return ConfidenceInterval(point_estimate=p, lower=max(0.0, lower), upper=min(1.0, upper))

@dataclass
class TwoProportionZTest:
    proportion_a: float
    proportion_b: float
    z: float
    p_value: float

# Two-tailed two-proportion z-test (pooled variance) - tests whether two win rates
# (e.g. a trained agent vs. the Random baseline) differ by more than sampling noise.
def two_proportion_z_test(wins_a: int, n_a: int, wins_b: int, n_b: int) -> TwoProportionZTest:
    p_a = wins_a / n_a
    p_b = wins_b / n_b
    pooled = (wins_a + wins_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0.0:
        z = 0.0
    else:
        z = (p_a - p_b) / se
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return TwoProportionZTest(proportion_a=p_a, proportion_b=p_b, z=z, p_value=p_value)

@dataclass
class MeanStd:
    mean: float
    std: float

# Sample mean and (n-1)-denominator standard deviation across seeds - used to report
# "mean ± std across seeds" per 10.5/11.3 rather than a single run's number.
def mean_and_std(values: list[float]) -> MeanStd:
    if not values:
        raise ValueError("values must be non-empty")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return MeanStd(mean=mean, std=0.0)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return MeanStd(mean=mean, std=math.sqrt(variance))

# Two-tailed 95% Student-t critical values by degrees of freedom, for the small seed counts
# used here (5 seeds -> df=4). Tabulated rather than computed, to stay scipy-free.
_T_95_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

# 95% confidence interval for the mean across seeds, treating each seed as one observation.
# This is the honest interval for "how well does this method do", because the seeds are
# different trained policies - pooling their games into one binomial (see
# pooled_win_rate_95ci) instead assumes one policy and reports a far narrower interval.
def t_confidence_interval_across_seeds(values: list[float]) -> ConfidenceInterval:
    stats = mean_and_std(values)
    # One seed carries no information about across-seed spread; report a zero-width interval
    # rather than raising, matching mean_and_std's std=0.0 convention for a single value.
    if len(values) == 1:
        return ConfidenceInterval(point_estimate=stats.mean, lower=stats.mean, upper=stats.mean)
    degrees_of_freedom = len(values) - 1
    t = _T_95_BY_DF.get(degrees_of_freedom, 1.96)
    margin = t * stats.std / math.sqrt(len(values))
    return ConfidenceInterval(
        point_estimate=stats.mean, lower=max(0.0, stats.mean - margin), upper=min(1.0, stats.mean + margin)
    )

# Holm-Bonferroni adjusted p-values, controlling the family-wise error rate across the several
# agent-vs-baseline comparisons Step 11.3 runs against the same Random baseline.
def holm_adjusted_p_values(p_values: list[float]) -> list[float]:
    num_tests = len(p_values)
    ordered = sorted(range(num_tests), key=lambda i: p_values[i])
    adjusted = [0.0] * num_tests
    running_max = 0.0
    for rank, index in enumerate(ordered):
        candidate = (num_tests - rank) * p_values[index]
        running_max = max(running_max, min(1.0, candidate))
        adjusted[index] = running_max
    return adjusted
