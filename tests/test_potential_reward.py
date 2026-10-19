# Tests for the potential-based reward mode (Stage 2): progress_potential, POTENTIAL_REWARD_CONFIG and LudoEnv's shaping term.

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from env.board import FINISH_POSITION, BoardState
from env.ludo_env import LudoEnv
from env.rewards import POTENTIAL_REWARD_CONFIG, REWARD_CONFIG, progress_potential
from training.config import TrainingConfig
from training.run_training import build_env, build_reward_config


# Checks the potential is zero when nobody has moved and positive/negative when ahead/behind.
def test_progress_potential_sign():
    board = BoardState()
    assert progress_potential(board, 0) == 0.0
    board.set(0, 0, 20)
    assert progress_potential(board, 0) > 0.0
    assert progress_potential(board, 1) < 0.0


# Checks the potential stays within [-1, 1] at the extremes.
def test_progress_potential_bounds():
    board = BoardState()
    for token in range(4):
        board.set(0, token, FINISH_POSITION)
    assert progress_potential(board, 0) == 1.0
    assert progress_potential(board, 1) < 0.0
    assert -1.0 <= progress_potential(board, 1) <= 1.0


# Checks the potential config pays only win and loss.
def test_potential_config_pays_only_win_and_loss():
    nonzero = {key for key, value in POTENTIAL_REWARD_CONFIG.items() if value != 0.0}
    assert nonzero == {"win", "loss"}
    assert POTENTIAL_REWARD_CONFIG["win"] == 1.0
    assert POTENTIAL_REWARD_CONFIG["loss"] == -1.0


# Checks the discounted shaping terms of a whole game sum to minus the starting potential, for several gammas.
def test_shaping_telescopes_over_a_full_game():
    for gamma in (0.9, 0.99, 1.0):
        for seed in range(4):
            env = LudoEnv(
                opponent_policy=HeuristicAgent(),
                reward_config=POTENTIAL_REWARD_CONFIG,
                shaping_gamma=gamma,
                rng=np.random.default_rng(seed),
            )
            rng = np.random.default_rng(seed + 100)
            obs, info = env.reset()
            phi_start = progress_potential(env.game.board, env.agent_player_id)
            total, discount, done = 0.0, 1.0, env.game.terminated or env.game.truncated
            while not done:
                legal = np.flatnonzero(info["action_mask"])
                obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
                done = terminated or truncated
                if terminated:
                    reward -= 1.0 if env.game.winner == env.agent_player_id else -1.0
                total += discount * reward
                discount *= gamma
            assert abs(total - (-phi_start)) < 1e-9


# Checks shaping stays off by default, so the dense reward is unchanged.
def test_shaping_off_by_default():
    assert LudoEnv().shaping_gamma is None
    assert build_env(TrainingConfig(agent_type="dqn", seed=0, num_episodes=1), dice_seed=0).shaping_gamma is None


# Checks reward_mode="potential" selects the potential config and passes the training gamma to the env.
def test_potential_mode_is_wired_into_training():
    config = TrainingConfig(agent_type="dqn", seed=0, num_episodes=1, reward_mode="potential", gamma=0.99)
    assert build_reward_config(config) is POTENTIAL_REWARD_CONFIG
    assert build_env(config, dice_seed=0).shaping_gamma == 0.99
    assert build_reward_config(TrainingConfig(agent_type="dqn", seed=0, num_episodes=1)) is REWARD_CONFIG
