# Runs training episodes for a TabularQAgent against LudoEnv.

from __future__ import annotations

from agents.tabular_q_agent import TabularQAgent, discretize_state
from env.ludo_env import LudoEnv
from training.episode_stats import EpisodeStats

# Owns a TabularQAgent and runs one online-updated episode at a time (no replay buffer).
class TabularQTrainer:

    # Sets the agent this trainer will run and update.
    def __init__(self, agent: TabularQAgent) -> None:
        self.agent = agent

    # Plays one full episode at the given exploration rate, updating the Q-table after each move.
    def run_episode(self, env: LudoEnv, epsilon: float) -> EpisodeStats:
        self.agent.epsilon = epsilon
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        length = 0
        state = discretize_state(env.game.board, env.agent_player_id)

        while not (terminated or truncated):
            legal_tokens = tuple(int(t) for t in info["action_mask"].nonzero()[0])
            action = self.agent(env.game.board, env.agent_player_id, info["dice_roll"], legal_tokens)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(env.game.board, env.agent_player_id)
            next_legal_tokens = tuple(int(t) for t in info["action_mask"].nonzero()[0])
            self.agent.update(state, action, reward, next_state, next_legal_tokens, done=terminated)
            state = next_state
            total_reward += reward
            length += 1

        return EpisodeStats(
            total_reward=total_reward, won=terminated, episode_length=length, epsilon=epsilon, avg_loss=None
        )

    # Picks the greedy (non-exploring) action for evaluation, restoring epsilon afterward.
    def select_greedy_action(self, env: LudoEnv, obs, info) -> int:
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        legal_tokens = tuple(int(t) for t in info["action_mask"].nonzero()[0])
        action = self.agent(env.game.board, env.agent_player_id, info["dice_roll"], legal_tokens)
        self.agent.epsilon = original_epsilon
        return action
