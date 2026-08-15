# Tests for env/toy_env.py's simplified single-agent environment.

import numpy as np

from env.board import BASE_POSITION, FINISH_POSITION
from env.toy_env import ToyLudoEnv


# Checks reset places tokens at the deliberately mixed starting layout, not all-legal.
def test_reset_uses_the_mixed_starting_layout():
    env = ToyLudoEnv()
    obs, info = env.reset()
    assert env.board.get(0, 0) == BASE_POSITION
    assert env.board.get(0, 1) == 20
    assert env.board.get(0, 2) == 50
    assert env.board.get(0, 3) == FINISH_POSITION
    assert not info["action_mask"].all()  # never trivially all-legal


# Checks the finished token (index 3) is never offered as a legal action.
def test_finished_token_is_never_legal():
    env = ToyLudoEnv()
    obs, info = env.reset()
    for _ in range(30):
        assert not info["action_mask"][3]
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


# Checks step rejects an action that isn't currently legal.
def test_step_rejects_an_illegal_action():
    import pytest

    env = ToyLudoEnv()
    obs, info = env.reset()
    illegal = [a for a in range(4) if not info["action_mask"][a]]
    assert illegal
    with pytest.raises(ValueError):
        env.step(illegal[0])


# Checks a full episode reaches Home (terminated) well within the step cap, always picking
# the first legal action - proving the deterministic setup is solvable, not stuck/broken.
def test_full_episode_reaches_home_with_a_fixed_policy():
    env = ToyLudoEnv(max_steps=200)
    obs, info = env.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert steps < 200
    assert terminated
    assert not truncated
