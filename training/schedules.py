# Epsilon-decay schedule shared by DQN and Tabular Q-learning training.

from __future__ import annotations

# Linearly decays epsilon from start to end over decay_episodes, then holds at end.
def linear_epsilon(episode: int, start: float, end: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return end
    fraction = min(1.0, episode / decay_episodes)
    return start + fraction * (end - start)
