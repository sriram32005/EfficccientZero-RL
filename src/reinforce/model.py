"""Policy-gradient network (REINFORCE with a learned baseline) on the Nature-DQN conv backbone."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _init(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ReinforceNet(nn.Module):
    def __init__(self, obs_shape, action_space_size: int, hidden: int = 512):
        super().__init__()
        c, h, w = obs_shape
        g = float(np.sqrt(2.0))
        self.features = nn.Sequential(
            _init(nn.Conv2d(c, 32, 8, stride=4), g),
            nn.ReLU(inplace=True),
            _init(nn.Conv2d(32, 64, 4, stride=2), g),
            nn.ReLU(inplace=True),
            _init(nn.Conv2d(64, 64, 3, stride=1), g),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, c, h, w)).shape[1]

        # Shared trunk -> separate policy / baseline heads. The policy output layer gets a tiny
        # init gain so the initial policy is ~uniform; the value layer gets gain 1.
        self.trunk = nn.Sequential(_init(nn.Linear(flat, hidden), g), nn.ReLU(inplace=True))
        self.policy_out = _init(nn.Linear(hidden, action_space_size), 0.01)
        self.value_out = _init(nn.Linear(hidden, 1), 1.0)
        self.action_space_size = action_space_size

    def forward(self, obs):
        x = self.features(obs.float() / 255.0)
        x = self.trunk(x)
        return self.policy_out(x), self.value_out(x)  # (B, A), (B, 1)
