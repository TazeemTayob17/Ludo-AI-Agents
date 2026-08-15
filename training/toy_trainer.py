# Trains a DQNAgent on ToyLudoEnv - Step 9's cheapest bug-catching checkpoint.

from __future__ import annotations

from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer
from env.toy_env import ToyLudoEnv

# Records what happened during toy training, for the convergence checks in Step 9.2.
class ToyTrainingResult:

    # Stores the per-update loss history and how many episodes reached Home.
    def __init__(self, losses: list[float], episodes: int, wins: int) -> None:
        self.losses = losses
        self.episodes = episodes
        self.wins = wins

# Plays num_episodes on the toy env, training the agent once enough experience is collected.
def train_on_toy_env(
    agent: DQNAgent,
    env: ToyLudoEnv,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    batch_size: int = 32,
    min_buffer_size: int = 200,
    sync_every_steps: int = 50,
) -> ToyTrainingResult:
    losses: list[float] = []
    wins = 0
    total_steps = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(obs, info["action_mask"])
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, terminated, info["action_mask"], next_info["action_mask"])
            obs, info = next_obs, next_info
            total_steps += 1

            if len(replay_buffer) >= min_buffer_size:
                batch = replay_buffer.sample(batch_size)
                losses.append(agent.train_on_batch(batch))
            if total_steps % sync_every_steps == 0:
                agent.sync_target_network()

        if terminated:
            wins += 1

    return ToyTrainingResult(losses=losses, episodes=num_episodes, wins=wins)
