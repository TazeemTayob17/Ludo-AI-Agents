# What one training episode produced, shared by every trainer type.

from __future__ import annotations

from dataclasses import dataclass

# Records one episode's outcome: reward, win/loss, length, exploration rate, and training loss.
@dataclass
class EpisodeStats:
    total_reward: float
    won: bool
    episode_length: int
    epsilon: float
    avg_loss: float | None
