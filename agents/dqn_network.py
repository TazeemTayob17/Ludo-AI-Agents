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
