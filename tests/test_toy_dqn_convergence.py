# Step 9.2/9.3's checkpoint: DQN visibly learns on the toy env, with masking exercised.

import numpy as np

from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer
from env.toy_env import ToyLudoEnv
from training.toy_trainer import train_on_toy_env

NUM_TRAIN_EPISODES = 150
NUM_EVAL_EPISODES = 30
MIN_WIN_RATE = 0.9
MAX_LOSS_RATIO = 0.5  # the later training loss must be at most half the early training loss


# Checks training loss trends down and the trained greedy policy reliably reaches Home.
def test_dqn_loss_decreases_and_agent_reliably_reaches_home():
    agent = DQNAgent(gamma=0.95, learning_rate=1e-3, epsilon=0.2, rng=np.random.default_rng(0))
    buffer = ReplayBuffer(capacity=5000, rng=np.random.default_rng(0))
    env = ToyLudoEnv()

    result = train_on_toy_env(agent, env, buffer, num_episodes=NUM_TRAIN_EPISODES)

    assert result.wins == NUM_TRAIN_EPISODES  # this deterministic toy task should always be solvable

    losses = result.losses
    assert len(losses) > 0
    window = max(1, len(losses) // 5)
    early_avg_loss = sum(losses[:window]) / window
    late_avg_loss = sum(losses[-window:]) / window
    assert late_avg_loss < early_avg_loss * MAX_LOSS_RATIO, (
        f"loss did not clearly decrease: early={early_avg_loss:.4f}, late={late_avg_loss:.4f}"
    )

    agent.epsilon = 0.0  # evaluate the learned (greedy) policy, not the exploring one
    wins = 0
    for _ in range(NUM_EVAL_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(obs, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            wins += 1

    win_rate = wins / NUM_EVAL_EPISODES
    assert win_rate >= MIN_WIN_RATE, f"greedy win rate {win_rate:.2%} did not clear {MIN_WIN_RATE:.0%}"
