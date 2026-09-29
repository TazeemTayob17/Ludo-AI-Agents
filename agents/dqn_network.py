# The Q-network: 3 fully-connected layers with ReLU, one Q-value per token action.

from __future__ import annotations

import torch
from torch import nn

from env.board import NUM_TOKENS_PER_PLAYER
from env.state_encoding import OBSERVATION_SIZE

# Maps an observation batch to a batch of 4 raw Q-values, one per token id.
class QNetwork(nn.Module):

    # Builds the 3 linear layers; hidden_size is the only architectural knob.
    def __init__(
        self,
        input_size: int = OBSERVATION_SIZE,
        hidden_size: int = 128,
        output_size: int = NUM_TOKENS_PER_PLAYER,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    # Runs a batch of observations through the network; no activation on the output layer.
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(observations))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Scores each of the 4 tokens with one shared small network, same (batch, obs) -> (batch, 4) contract as QNetwork.
class SharedTokenQNetwork(nn.Module):

    # Builds the context encoder and shared per-token scorer; feature sizes are per-token widths (0 if a block is off).
    def __init__(
        self,
        context_size: int,
        move_feature_size: int,
        threat_feature_size: int,
        hidden_size: int = 128,
        num_tokens: int = NUM_TOKENS_PER_PLAYER,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.move_feature_size = move_feature_size
        self.threat_feature_size = threat_feature_size
        self.num_tokens = num_tokens

        self.context_encoder = nn.Sequential(nn.Linear(context_size, hidden_size), nn.ReLU())
        self.token_scorer = nn.Sequential(
            nn.Linear(hidden_size + move_feature_size + threat_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    # Slices out one token's own move/threat features from the shared observation layout.
    def _token_slice(self, observations: torch.Tensor, token_id: int) -> torch.Tensor:
        pieces = []
        if self.move_feature_size:
            start = self.context_size + token_id * self.move_feature_size
            pieces.append(observations[:, start:start + self.move_feature_size])
        if self.threat_feature_size:
            move_block_size = self.num_tokens * self.move_feature_size
            start = self.context_size + move_block_size + token_id * self.threat_feature_size
            pieces.append(observations[:, start:start + self.threat_feature_size])
        if not pieces:
            return observations.new_zeros((observations.shape[0], 0))
        return torch.cat(pieces, dim=1)

    # Encodes the shared board context once, then scores each token with the same weights.
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        context_embedding = self.context_encoder(observations[:, :self.context_size])
        token_values = [
            self.token_scorer(torch.cat([context_embedding, self._token_slice(observations, token_id)], dim=1))
            for token_id in range(self.num_tokens)
        ]
        return torch.cat(token_values, dim=1)
