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


# Checks whether the agent itself won - terminated alone is also true when an opponent wins.
def agent_won(terminated: bool, info: dict, agent_player_id: int) -> bool:
    return bool(terminated and info.get("winner") == agent_player_id)
