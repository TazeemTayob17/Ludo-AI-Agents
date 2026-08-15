# Tests for training/episode_stats.py's agent_won helper - the fix for a real bug where
# "terminated" (the game ended in a win by anyone) was being conflated with "the agent won".

from training.episode_stats import agent_won


# Checks a truncated (non-terminated) episode is never a win.
def test_not_terminated_is_never_a_win():
    assert agent_won(terminated=False, info={"winner": 0}, agent_player_id=0) is False


# Checks a terminated episode where the agent itself won counts as a win.
def test_terminated_with_agent_as_winner_is_a_win():
    assert agent_won(terminated=True, info={"winner": 2}, agent_player_id=2) is True


# Checks a terminated episode where an OPPONENT won does not count as a win for the agent.
def test_terminated_with_an_opponent_as_winner_is_not_a_win():
    assert agent_won(terminated=True, info={"winner": 1}, agent_player_id=0) is False


# Checks a missing "winner" key is treated defensively as not a win, not an error.
def test_missing_winner_key_is_not_a_win():
    assert agent_won(terminated=True, info={}, agent_player_id=0) is False
