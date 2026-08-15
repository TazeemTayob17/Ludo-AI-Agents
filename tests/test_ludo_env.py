# Tests for env/ludo_env.py's Gymnasium wrapper.

import numpy as np
import pytest

from env.game_state import GameState
from env.ludo_env import LudoEnv
from env.rewards import captured_penalty, truncation_reward
from env.state_encoding import OBSERVATION_SIZE


# A scripted policy that always picks the first legal token offered.
def _first_legal(board, player_id, roll, legal_tokens):
    return legal_tokens[0]


# A fixed sequence of rolls fed in place of real dice, cycling once exhausted.
class _CyclingRng:
    def __init__(self, rolls):
        self._rolls = list(rolls)
        self._index = 0

    def integers(self, low, high):
        value = self._rolls[self._index % len(self._rolls)]
        self._index += 1
        return value


# A scripted rng that always rolls a non-six, so nobody can ever exit Base.
class _NeverSixRng:
    def integers(self, low, high):
        return 3


# Checks reset returns a well-formed observation and a usable action mask.
def test_reset_returns_observation_and_action_mask():
    env = LudoEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert info["action_mask"].dtype == bool
    assert info["action_mask"].any()
    assert info["dice_roll"] in range(1, 7)
    assert info["current_player"] == 0


# Checks step rejects an action that isn't currently legal.
def test_step_rejects_an_illegal_action():
    env = LudoEnv(rng=_CyclingRng([3]))  # a non-six: only an already-exited token can move
    env.game = GameState(max_turns=100)
    env._rng = env._injected_rng
    env.game.board.set(0, 0, 10)  # token 0 legal; tokens 1-3 still in Base, illegal on a 3
    env._run_until_agent_turn_or_done()

    info = env._build_info()
    illegal_actions = [a for a in range(4) if not info["action_mask"][a]]
    assert illegal_actions == [1, 2, 3]
    with pytest.raises(ValueError):
        env.step(illegal_actions[0])


# Checks a full episode runs to completion (terminated or truncated) without crashing.
def test_full_random_episode_completes():
    env = LudoEnv(max_turns=100)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        legal = np.flatnonzero(info["action_mask"])
        action = int(rng.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert steps < 5000  # generous circuit breaker; not expected to trigger
    assert terminated or truncated


# Checks repeated no-legal-move turns for every seat still terminate cleanly via truncation.
def test_repeated_no_legal_move_turns_do_not_stall_and_truncate_cleanly():
    env = LudoEnv(max_turns=20, rng=_NeverSixRng())
    env.reset(seed=0)
    assert env.game.truncated is True
    assert env.game.terminated is False
    assert env.game.turn_count == 20


# Checks a capture that happens during a fast-forwarded opponent turn is folded into
# the reward returned by the agent's NEXT step(), not the step() during which it occurred.
def test_opponent_capture_during_fast_forward_is_folded_into_next_step_reward():
    # roll sequence: [agent's move, player1's capturing move, player1's bonus move,
    # player2's dead roll, player3's dead roll, agent's next move]
    rolls = [2, 4, 3, 3, 3, 2]
    env = LudoEnv(opponent_policy=_first_legal, max_turns=100, rng=_CyclingRng(rolls))

    # Bypass reset() (which always starts from an empty board) so the scenario can be
    # placed on the board before any dice are rolled.
    env.game = GameState(max_turns=100)
    env._rng = env._injected_rng
    env.game.board.set(0, 0, 10)  # agent's token, global square 10, unsafe
    env.game.board.set(0, 1, 20)  # a second agent token so its first move needn't touch token 0
    env.game.board.set(1, 0, 45)  # player 1: relative 45 -> global 6; a roll of 4 lands on global 10
    env._run_until_agent_turn_or_done()

    info = env._build_info()
    assert info["action_mask"][1]

    obs, reward_1, terminated, truncated, info = env.step(1)  # agent moves token 1, not token 0
    assert env.game.board.get(0, 0) == -1  # captured back to Base during the fast-forward
    assert reward_1 == pytest.approx(0.0)  # capture must NOT show up in this step's reward yet

    obs, reward_2, terminated, truncated, info = env.step(1)  # agent's next decision
    assert reward_2 == pytest.approx(captured_penalty())  # folded in on the very next step


# Checks a captured penalty is paid out immediately if the episode ends before the
# agent's next step(), since there is no future call left to fold it into.
def test_captured_penalty_is_paid_immediately_if_episode_ends_before_next_step():
    class _FillingRng:
        def __init__(self, rolls):
            self._rolls = list(rolls)

        def integers(self, low, high):
            return self._rolls.pop(0) if self._rolls else 3

    # roll sequence: [agent's move (ends its turn), player1's capturing move]; max_turns=2
    # means the game truncates on player1's turn, in the same fast-forward as the capture.
    env = LudoEnv(opponent_policy=_first_legal, max_turns=2, rng=_FillingRng([2, 4]))
    env.game = GameState(max_turns=2)
    env._rng = env._injected_rng
    env.game.board.set(0, 0, 10)  # agent's token, global square 10, unsafe
    env.game.board.set(0, 1, 20)  # a second agent token so its first move needn't touch token 0
    env.game.board.set(1, 0, 45)  # player 1: relative 45 -> global 6; a roll of 4 lands on global 10
    env._run_until_agent_turn_or_done()

    obs, reward, terminated, truncated, info = env.step(1)
    assert truncated is True
    assert env.game.board.get(0, 0) == -1  # captured back to Base
    assert reward == pytest.approx(truncation_reward() + captured_penalty())
    assert env._pending_captured_penalty == 0.0


# Checks render produces readable text without raising.
def test_render_produces_text():
    env = LudoEnv()
    env.reset(seed=0)
    text = env.render()
    assert "player 0" in text
