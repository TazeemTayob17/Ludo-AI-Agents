# Reward values and reward computation for the learning agent's own moves and game events.

from __future__ import annotations

from env.board import (
    FINISH_POSITION,
    NUM_PLAYERS,
    NUM_TOKENS_PER_PLAYER,
    BoardState,
    is_on_main_track,
    is_safe_square,
    relative_to_global,
)
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

# Step 11.4's ablation target: only the terminal win/loss signal survives, at the same
# magnitudes as REWARD_CONFIG, so the comparison isolates shaping's effect, not scale.
SPARSE_REWARD_CONFIG = {
    "exit_base": 0.0,
    "safe_square_entry": 0.0,
    "capture_opponent": 0.0,
    "reach_home": 0.0,
    "win": REWARD_CONFIG["win"],
    "got_captured": 0.0,
    "loss": REWARD_CONFIG["loss"],
    "truncated": 0.0,
}

# Stage 2's win-dominant mode: only win/loss are paid directly; progress comes from the potential term in LudoEnv.
POTENTIAL_REWARD_CONFIG = {
    "exit_base": 0.0,
    "safe_square_entry": 0.0,
    "capture_opponent": 0.0,
    "reach_home": 0.0,
    "win": 1.0,
    "got_captured": 0.0,
    "loss": -1.0,
    "truncated": 0.0,
}

# Largest possible total progress for one player (every token Home), used to scale the potential into [-1, 1].
_MAX_PLAYER_PROGRESS = NUM_TOKENS_PER_PLAYER * (FINISH_POSITION + 1)

# Race standing for player_id: own total progress minus the opponents' mean progress, scaled to [-1, 1].
def progress_potential(board: BoardState, player_id: int) -> float:
    def total(owner: int) -> int:
        return sum(board.get(owner, token_id) + 1 for token_id in range(NUM_TOKENS_PER_PLAYER))

    opponents = [total(owner) for owner in range(NUM_PLAYERS) if owner != player_id]
    return (total(player_id) - sum(opponents) / len(opponents)) / _MAX_PLAYER_PROGRESS

# Computes the reward for the agent's own move, given what it did and where it landed.
def compute_move_reward(outcome: MoveOutcome, board: BoardState, player_id: int, config: dict | None = None) -> float:
    cfg = config if config is not None else REWARD_CONFIG
    reward = cfg["capture_opponent"] if outcome.captured_player is not None else 0.0

    if outcome.reached_home:
        reward += cfg["reach_home"]
    elif outcome.exited_base:
        reward += cfg["exit_base"]
    else:
        new_position = board.get(player_id, outcome.token_id)
        if is_on_main_track(new_position) and is_safe_square(relative_to_global(player_id, new_position)):
            reward += cfg["safe_square_entry"]

    return reward

# Returns the penalty for having one of the agent's own tokens captured by an opponent.
def captured_penalty(config: dict | None = None) -> float:
    return (config if config is not None else REWARD_CONFIG)["got_captured"]

# Returns the reward for winning the game.
def win_reward(config: dict | None = None) -> float:
    return (config if config is not None else REWARD_CONFIG)["win"]

# Returns the reward for losing the game (an opponent finished first).
def loss_reward(config: dict | None = None) -> float:
    return (config if config is not None else REWARD_CONFIG)["loss"]

# Returns the penalty for the episode being truncated without a winner.
def truncation_reward(config: dict | None = None) -> float:
    return (config if config is not None else REWARD_CONFIG)["truncated"]
