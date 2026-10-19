# Tests for training/score_baselines.py.

from training.score_baselines import score


# Checks one worker scores the requested number of games for each agent type without error.
def test_score_runs_for_each_agent():
    for agent in ("expert", "heuristic", "random", "rollout"):
        result = score(agent, "heuristic", num_games=2, num_rollouts=1, workers=1)
        assert result.num_episodes == 2
        assert 0 <= result.wins <= 2
