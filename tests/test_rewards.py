# Tests for env/rewards.py's reward computation.

from env.board import FINISH_POSITION, BoardState
from env.moves import MoveOutcome
from env.rewards import (
    REWARD_CONFIG,
    captured_penalty,
    compute_move_reward,
    loss_reward,
    truncation_reward,
    win_reward,
)


# Checks exiting Base is rewarded at the configured value.
def test_reward_for_exiting_base():
    board = BoardState()
    board.set(0, 0, 0)
    outcome = MoveOutcome(token_id=0, exited_base=True, captured_player=None, captured_token=None, reached_home=False)
    assert compute_move_reward(outcome, board, player_id=0) == REWARD_CONFIG["exit_base"]


# Checks capturing an opponent is rewarded at the configured value.
def test_reward_for_capturing_opponent():
    board = BoardState()
    board.set(0, 0, 14)  # not a safe square
    outcome = MoveOutcome(token_id=0, exited_base=False, captured_player=2, captured_token=0, reached_home=False)
    assert compute_move_reward(outcome, board, player_id=0) == REWARD_CONFIG["capture_opponent"]


# Checks reaching Home is rewarded at the configured value.
def test_reward_for_reaching_home():
    board = BoardState()
    board.set(0, 0, FINISH_POSITION)
    outcome = MoveOutcome(token_id=0, exited_base=False, captured_player=None, captured_token=None, reached_home=True)
    assert compute_move_reward(outcome, board, player_id=0) == REWARD_CONFIG["reach_home"]


# Checks landing on a safe square (not via Base or Home) is rewarded at the configured value.
def test_reward_for_landing_on_a_safe_square():
    board = BoardState()
    board.set(0, 0, 8)  # global square 8, a star safe square
    outcome = MoveOutcome(token_id=0, exited_base=False, captured_player=None, captured_token=None, reached_home=False)
    assert compute_move_reward(outcome, board, player_id=0) == REWARD_CONFIG["safe_square_entry"]


# Checks landing on a plain unsafe square with no capture earns no reward.
def test_no_reward_for_a_plain_unsafe_landing():
    board = BoardState()
    board.set(0, 0, 10)  # global square 10, not safe
    outcome = MoveOutcome(token_id=0, exited_base=False, captured_player=None, captured_token=None, reached_home=False)
    assert compute_move_reward(outcome, board, player_id=0) == 0.0


# Checks a capture that also lands on Home stacks both rewards.
def test_capture_and_reach_home_rewards_stack():
    board = BoardState()
    board.set(0, 0, FINISH_POSITION)
    outcome = MoveOutcome(token_id=0, exited_base=False, captured_player=2, captured_token=0, reached_home=True)
    expected = REWARD_CONFIG["capture_opponent"] + REWARD_CONFIG["reach_home"]
    assert compute_move_reward(outcome, board, player_id=0) == expected


# Checks the standalone game-event reward lookups return their configured values.
def test_standalone_event_rewards():
    assert captured_penalty() == REWARD_CONFIG["got_captured"]
    assert win_reward() == REWARD_CONFIG["win"]
    assert loss_reward() == REWARD_CONFIG["loss"]
    assert truncation_reward() == REWARD_CONFIG["truncated"]
