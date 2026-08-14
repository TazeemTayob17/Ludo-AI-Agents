# Reward values and reward computation for the learning agent's own moves and game events.

from __future__ import annotations

from env.board import BoardState, is_on_main_track, is_safe_square, relative_to_global
from env.moves import MoveOutcome

REWARD_CONFIG = {
    "exit_base": 0.2,
    "safe_square_entry": 0.2,
    "capture_opponent": 1.0,
    "reach_home": 1.0,
    "win": 2.0,
    "got_captured": -1.0,
    "loss": -2.0,
    "truncated": -0.5,
}

# Computes the reward for the agent's own move, given what it did and where it landed.
def compute_move_reward(outcome: MoveOutcome, board: BoardState, player_id: int) -> float:
    reward = REWARD_CONFIG["capture_opponent"] if outcome.captured_player is not None else 0.0

    if outcome.reached_home:
        reward += REWARD_CONFIG["reach_home"]
    elif outcome.exited_base:
        reward += REWARD_CONFIG["exit_base"]
    else:
        new_position = board.get(player_id, outcome.token_id)
        if is_on_main_track(new_position) and is_safe_square(relative_to_global(player_id, new_position)):
            reward += REWARD_CONFIG["safe_square_entry"]

    return reward

# Returns the penalty for having one of the agent's own tokens captured by an opponent.
def captured_penalty() -> float:
    return REWARD_CONFIG["got_captured"]

# Returns the reward for winning the game.
def win_reward() -> float:
    return REWARD_CONFIG["win"]

# Returns the reward for losing the game (an opponent finished first).
def loss_reward() -> float:
    return REWARD_CONFIG["loss"]

# Returns the penalty for the episode being truncated without a winner.
def truncation_reward() -> float:
    return REWARD_CONFIG["truncated"]
