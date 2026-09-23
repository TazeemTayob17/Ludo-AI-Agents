# Tests for training/tournament.py. Deliberately tiny episode counts - proves the
# tournament runner and its metrics are wired correctly, not a real evaluation result.

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from training.tournament import TournamentResult, run_tournament, wrap_choose_action_fn


# Checks a small tournament produces a result with plausible ranges for every metric.
def test_run_tournament_produces_a_well_formed_result():
    policy = wrap_choose_action_fn(RandomAgent(rng=np.random.default_rng(0)))
    result = run_tournament("random_test", policy, HeuristicAgent(), num_episodes=5, seed=0, max_turns=200)

    assert result.num_episodes == 5
    assert 0 <= result.wins <= 5
    assert 0.0 <= result.win_rate <= 1.0
    assert result.avg_episode_length > 0
    assert result.total_captures_made >= 0
    assert result.total_times_captured >= 0


# Checks capture_to_death_ratio is a real ratio when captures happened, dividing made by suffered.
def test_capture_to_death_ratio_computed_correctly_when_captures_happened():
    result = TournamentResult(
        agent_label="x", num_episodes=10, wins=3, total_episode_length=500,
        total_captures_made=6, total_times_captured=3,
    )
    assert result.capture_to_death_ratio == 2.0


# Checks capture_to_death_ratio is None (not a divide-by-zero) when never captured.
def test_capture_to_death_ratio_is_none_when_never_captured():
    result = TournamentResult(
        agent_label="x", num_episodes=10, wins=1, total_episode_length=500,
        total_captures_made=2, total_times_captured=0,
    )
    assert result.capture_to_death_ratio is None


# Checks the tournament's rng is reproducible: same seed gives the same win count.
def test_same_seed_gives_reproducible_results():
    def make_result():
        policy = wrap_choose_action_fn(RandomAgent(rng=np.random.default_rng(1)))
        return run_tournament("random_test", policy, HeuristicAgent(), num_episodes=5, seed=42, max_turns=200)

    first = make_result()
    second = make_result()
    assert first.wins == second.wins
    assert first.total_episode_length == second.total_episode_length
