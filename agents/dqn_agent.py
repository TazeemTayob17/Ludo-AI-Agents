# The DQN / Double DQN agent: action selection, the masked Bellman target, and the training step.

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from agents.dqn_network import QNetwork
from env.board import BoardState, NUM_TOKENS_PER_PLAYER
from env.state_encoding import encode_observation

# Holds the online/target networks and implements masked action selection, the (optionally
# Double-DQN) masked Bellman target, and one optimizer step.
class DQNAgent:

    # Builds both networks (target starts synced to online) and the optimizer.
    def __init__(
        self,
        double_dqn: bool = False,
        hidden_size: int = 128,
        gamma: float = 0.95,
        learning_rate: float = 1e-3,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.double_dqn = double_dqn
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = rng if rng is not None else np.random.default_rng()

        self.online_network = QNetwork(hidden_size=hidden_size)
        self.target_network = QNetwork(hidden_size=hidden_size)
        self.sync_target_network()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)

    # Copies the online network's weights into the target network.
    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    # Returns the online network's raw Q-values for a single observation, as a numpy array.
    def q_values(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            return self.online_network(tensor).squeeze(0).numpy()

    # Epsilon-greedy over legal actions only; a masked-out action can never be selected.
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        legal = np.flatnonzero(action_mask)
        if self._rng.random() < self.epsilon:
            return int(legal[self._rng.integers(0, len(legal))])
        values = self.q_values(observation)
        masked_values = np.where(action_mask, values, -np.inf)
        return int(np.argmax(masked_values))

    # Matches the ChooseActionFn signature so this can be used as a policy directly.
    def __call__(self, board: BoardState, player_id: int, roll: int, legal_tokens: tuple[int, ...]) -> int:
        observation = encode_observation(board, player_id)
        mask = np.zeros(NUM_TOKENS_PER_PLAYER, dtype=bool)
        for token_id in legal_tokens:
            mask[token_id] = True
        return self.select_action(observation, mask)

    # Computes the masked Bellman target, using the Double DQN action-selection/evaluation
    # split when enabled, and treating a fully-masked next state (or a terminal transition)
    # as contributing zero future value.
    def compute_targets(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_action_masks: np.ndarray,
        dones: np.ndarray,
    ) -> torch.Tensor:
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32)
        next_masks_t = torch.as_tensor(next_action_masks, dtype=torch.bool)
        dones_t = torch.as_tensor(dones, dtype=torch.float32)

        with torch.no_grad():
            target_q = self.target_network(next_states_t)
            masked_target_q = target_q.masked_fill(~next_masks_t, float("-inf"))

            if self.double_dqn:
                online_q = self.online_network(next_states_t)
                masked_online_q = online_q.masked_fill(~next_masks_t, float("-inf"))
                next_actions = masked_online_q.argmax(dim=1, keepdim=True)
                next_values = masked_target_q.gather(1, next_actions).squeeze(1)
            else:
                next_values = masked_target_q.max(dim=1).values

            no_legal_next_action = ~next_masks_t.any(dim=1)
            next_values = torch.where(no_legal_next_action, torch.zeros_like(next_values), next_values)

            return rewards_t + self.gamma * next_values * (1.0 - dones_t)

    # Computes the loss between the taken actions' predicted Q-values and the Bellman target.
    def compute_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        next_action_masks: np.ndarray,
    ) -> torch.Tensor:
        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64)

        targets = self.compute_targets(rewards, next_states, next_action_masks, dones)
        predicted = self.online_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        return F.mse_loss(predicted, targets)

    # Runs one optimizer step on a sampled batch and returns the scalar loss value.
    def train_on_batch(self, batch: tuple[np.ndarray, ...]) -> float:
        states, actions, rewards, next_states, dones, _masks, next_masks = batch
        loss = self.compute_loss(states, actions, rewards, next_states, dones, next_masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
