# A fixed-capacity experience replay buffer, storing the action mask for both s and s'.

from __future__ import annotations

from collections import deque

import numpy as np

# Stores transitions and samples uniformly at random, independent of the environment.
class ReplayBuffer:

    # Sets the buffer's fixed capacity and its own seeded random generator.
    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._rng = rng if rng is not None else np.random.default_rng()

    # Returns how many transitions are currently stored.
    def __len__(self) -> int:
        return len(self._buffer)

    # Appends one transition, including the legal-action mask for both s and s'.
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self._buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))

    # Samples a random batch, returned as stacked numpy arrays ready for tensor conversion.
    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = self._rng.integers(0, len(self._buffer), size=batch_size)
        batch = [self._buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(masks, dtype=bool),
            np.array(next_masks, dtype=bool),
        )
