"""The Nature-DQN convolutional Q-network (Mnih et al., 2015), optionally dueling."""
from __future__ import annotations

import torch
import torch.nn as nn


class DQNNet(nn.Module):
    def __init__(self, obs_shape, action_space_size: int, dueling: bool = True, hidden: int = 512):
        super().__init__()
        c, h, w = obs_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, c, h, w)).shape[1]
        self.dueling = dueling
        if dueling:
            self.value = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
            self.advantage = nn.Sequential(
                nn.Linear(flat, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, action_space_size)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(flat, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, action_space_size)
            )
        self.action_space_size = action_space_size

    def forward(self, obs):
        x = self.features(obs.float() / 255.0)
        if self.dueling:
            v = self.value(x)
            a = self.advantage(x)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(x)
