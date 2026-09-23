# Step 11's tournament: many greedy episodes for one policy against a fixed 3-opponent lineup.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from env.ludo_env import LudoEnv, OpponentPolicy
from env.state_encoding import ObservationSpec
from training.episode_stats import agent_won

# A greedy action-selection callable matching DQNTrainer/TabularQTrainer.select_greedy_action.
class GreedyPolicy(Protocol):
    def __call__(self, env: LudoEnv, obs: np.ndarray, info: dict) -> int: ...

# Adapts a ChooseActionFn-style baseline policy into the GreedyPolicy interface the tournament uses.
def wrap_choose_action_fn(policy: Callable) -> GreedyPolicy:
    def select(env: LudoEnv, obs: np.ndarray, info: dict) -> int:
        legal_tokens = tuple(int(t) for t in info["action_mask"].nonzero()[0])
        return policy(env.game.board, env.agent_player_id, info["dice_roll"], legal_tokens)

    return select

# Aggregate results of one tournament: wins, game lengths, and capture counts.
@dataclass
class TournamentResult:
    agent_label: str
    num_episodes: int
    wins: int
    total_episode_length: int
    total_captures_made: int
    total_times_captured: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.num_episodes

    @property
    def avg_episode_length(self) -> float:
        return self.total_episode_length / self.num_episodes

    # Ratio of opponent tokens captured to the agent's own tokens lost, or None if never captured.
    @property
    def capture_to_death_ratio(self) -> float | None:
        if self.total_times_captured == 0:
            return None
        return self.total_captures_made / self.total_times_captured

# Plays num_episodes greedily for one policy against the opponent lineup on a reproducible dice seed.
def run_tournament(
    agent_label: str,
    select_action: GreedyPolicy,
    opponent_policy: OpponentPolicy,
    num_episodes: int,
    seed: int,
    max_turns: int = 1000,
    observation_spec: ObservationSpec | None = None,
) -> TournamentResult:
    env = LudoEnv(
        opponent_policy=opponent_policy,
        max_turns=max_turns,
        rng=np.random.default_rng(seed),
        observation_spec=observation_spec,
    )

    wins = 0
    total_length = 0
    total_captures_made = 0
    total_times_captured = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated, truncated = env.game.terminated, env.game.truncated
        while not (terminated or truncated):
            action = select_action(env, obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
        if agent_won(terminated, info, env.agent_player_id):
            wins += 1
        total_length += env.game.turn_count
        total_captures_made += info["episode_captures_made"]
        total_times_captured += info["episode_times_captured"]

    return TournamentResult(
        agent_label=agent_label,
        num_episodes=num_episodes,
        wins=wins,
        total_episode_length=total_length,
        total_captures_made=total_captures_made,
        total_times_captured=total_times_captured,
    )
