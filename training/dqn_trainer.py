# Runs training episodes for a DQNAgent (or its Double DQN variant) against LudoEnv.

from __future__ import annotations

from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer
from env.ludo_env import LudoEnv
from training.episode_stats import EpisodeStats, agent_won

# Owns a DQNAgent and its replay buffer, and runs one episode at a time.
class DQNTrainer:

    # Sets the agent, buffer, and the batching/sync settings for training updates.
    def __init__(
        self,
        agent: DQNAgent,
        replay_buffer: ReplayBuffer,
        batch_size: int = 32,
        min_buffer_size: int = 1000,
        sync_every_steps: int = 500,
    ) -> None:
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.sync_every_steps = sync_every_steps
        self._total_steps = 0

    # Plays one full episode at the given exploration rate, training on sampled batches as it goes.
    def run_episode(self, env: LudoEnv, epsilon: float) -> EpisodeStats:
        self.agent.epsilon = epsilon
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        length = 0
        losses: list[float] = []

        while not (terminated or truncated):
            action = self.agent.select_action(obs, info["action_mask"])
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            self.replay_buffer.push(
                obs, action, reward, next_obs, terminated, info["action_mask"], next_info["action_mask"]
            )
            obs, info = next_obs, next_info
            total_reward += reward
            length += 1
            self._total_steps += 1

            if len(self.replay_buffer) >= self.min_buffer_size:
                batch = self.replay_buffer.sample(self.batch_size)
                losses.append(self.agent.train_on_batch(batch))
            if self._total_steps % self.sync_every_steps == 0:
                self.agent.sync_target_network()

        avg_loss = sum(losses) / len(losses) if losses else None
        won = agent_won(terminated, info, env.agent_player_id)
        return EpisodeStats(
            total_reward=total_reward, won=won, episode_length=length, epsilon=epsilon, avg_loss=avg_loss
        )

    # Picks the greedy (non-exploring) action for evaluation, restoring epsilon afterward.
    def select_greedy_action(self, env: LudoEnv, obs, info) -> int:
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        action = self.agent.select_action(obs, info["action_mask"])
        self.agent.epsilon = original_epsilon
        return action
