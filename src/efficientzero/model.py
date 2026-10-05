"""EfficientZero network.

Three differences from vanilla MuZero, all implemented here -- these are exactly the
three contributions of the EfficientZero paper (Ye et al., NeurIPS 2021) and they are
what make it ~50x more sample efficient than DQN-style model-free agents:

1. Self-supervised consistency loss (SimSiam) between the dynamics-predicted next
   latent state and the representation of the *real* next observation  -> `project()`.
2. End-to-end *value prefix* prediction with an LSTM instead of per-step reward
   prediction -> `DynamicsNetwork` returns a value-prefix distribution + LSTM state.
3. Model-based off-policy correction on the value targets -> handled in `trainer.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.utils import support_to_scalar


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------
def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv3x3(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class DownSample(nn.Module):
    """96x96 -> 6x6 downsampling tower (identical structure to the EfficientZero paper)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.res1 = ResidualBlock(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.pool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.res3 = ResidualBlock(out_channels)
        self.pool2 = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # 96 -> 48
        x = self.res1(x)
        x = F.relu(self.bn2(self.conv2(x)))   # 48 -> 24
        x = self.res2(x)
        x = self.pool1(x)                     # 24 -> 12
        x = self.res3(x)
        x = self.pool2(x)                     # 12 -> 6
        return x


def renormalize(x: torch.Tensor) -> torch.Tensor:
    """Scale each latent state into [0, 1] (MuZero appendix G) -- keeps the model stable."""
    flat = x.view(x.size(0), -1)
    mn = flat.min(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    mx = flat.max(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    return (x - mn) / (mx - mn + 1e-5)


def mlp(in_dim, hidden, out_dim, use_bn=True, out_bn=False):
    layers = [nn.Linear(in_dim, hidden)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden, out_dim))
    if out_bn:
        layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------------------
# sub networks
# --------------------------------------------------------------------------------------
class RepresentationNetwork(nn.Module):
    """observation stack -> latent state s0."""

    def __init__(self, in_channels, channels, blocks, downsample=True):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.stem = DownSample(in_channels, channels)
        else:
            self.stem = nn.Sequential(conv3x3(in_channels, channels), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return renormalize(x)


class DynamicsNetwork(nn.Module):
    """(latent state, action) -> next latent state + *value prefix* through an LSTM.

    The LSTM is the key trick: instead of asking the model to predict each individual
    step reward (which is nearly always 0 in Atari and gives a very weak signal),
    EfficientZero predicts the cumulative discounted reward since the last reset, so a
    single sparse reward supervises every step of the unroll.
    """

    def __init__(self, channels, blocks, reduced_channels, spatial, lstm_hidden, fc_hidden, support_size):
        super().__init__()
        self.conv = conv3x3(channels + 1, channels)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        self.prefix_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.prefix_bn = nn.BatchNorm2d(reduced_channels)
        self.flat_dim = reduced_channels * spatial * spatial
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(input_size=self.flat_dim, hidden_size=lstm_hidden)
        self.prefix_bn_after = nn.BatchNorm1d(lstm_hidden)
        self.prefix_head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden), nn.ReLU(inplace=True), nn.Linear(fc_hidden, 2 * support_size + 1)
        )

    def forward(self, state_action, reward_hidden):
        prev_state = state_action[:, :-1, :, :]
        x = F.relu(self.bn(self.conv(state_action)))
        x = x + prev_state  # residual over the previous latent state
        x = F.relu(x)
        for b in self.blocks:
            x = b(x)
        next_state = renormalize(x)

        p = F.relu(self.prefix_bn(self.prefix_conv(x)))
        p = p.reshape(p.size(0), -1).unsqueeze(0)            # (1, B, flat) -- seq len 1
        p, reward_hidden = self.lstm(p, reward_hidden)
        p = F.relu(self.prefix_bn_after(p.squeeze(0)))
        value_prefix_logits = self.prefix_head(p)
        return next_state, reward_hidden, value_prefix_logits


class PredictionNetwork(nn.Module):
    """latent state -> (policy logits, categorical value)."""

    def __init__(self, channels, blocks, reduced_channels, spatial, fc_hidden, action_space_size, support_size):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        self.value_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.value_bn = nn.BatchNorm2d(reduced_channels)
        self.policy_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.policy_bn = nn.BatchNorm2d(reduced_channels)
        flat = reduced_channels * spatial * spatial
        self.value_head = mlp(flat, fc_hidden, 2 * support_size + 1)
        self.policy_head = mlp(flat, fc_hidden, action_space_size)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        v = F.relu(self.value_bn(self.value_conv(x))).reshape(x.size(0), -1)
        p = F.relu(self.policy_bn(self.policy_conv(x))).reshape(x.size(0), -1)
        return self.policy_head(p), self.value_head(v)


# --------------------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------------------
class NetOutput:
    __slots__ = ("state", "reward_hidden", "value_prefix_logits", "policy_logits", "value_logits")

    def __init__(self, state, reward_hidden, value_prefix_logits, policy_logits, value_logits):
        self.state = state
        self.reward_hidden = reward_hidden
        self.value_prefix_logits = value_prefix_logits
        self.policy_logits = policy_logits
        self.value_logits = value_logits


class EfficientZeroNet(nn.Module):
    def __init__(self, obs_shape, action_space_size: int, cfg):
        super().__init__()
        c, h, w = obs_shape
        channels = int(cfg.channels)
        blocks = int(cfg.blocks)
        self.action_space_size = int(action_space_size)
        self.support_size = int(cfg.support_size)
        self.lstm_hidden = int(cfg.lstm_hidden)
        self.proj_dim = int(cfg.proj_dim)
        downsample = bool(cfg.downsample)
        spatial = h // 16 if downsample else h

        self.representation_net = RepresentationNetwork(c, channels, blocks, downsample)
        self.dynamics_net = DynamicsNetwork(
            channels, blocks, int(cfg.reduced_channels), spatial, self.lstm_hidden, int(cfg.fc_hidden), self.support_size
        )
        self.prediction_net = PredictionNetwork(
            channels,
            blocks,
            int(cfg.reduced_channels),
            spatial,
            int(cfg.fc_hidden),
            self.action_space_size,
            self.support_size,
        )

        # SimSiam projector / predictor for the consistency loss
        flat = channels * spatial * spatial
        self.projection = nn.Sequential(
            nn.Linear(flat, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim // 2),
            nn.BatchNorm1d(self.proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_dim // 2, self.proj_dim),
        )
        self.spatial = spatial
        self.channels = channels

    # ----- primitives -----------------------------------------------------------------
    def representation(self, obs):
        return self.representation_net(obs.float() / 255.0)

    def prediction(self, state):
        return self.prediction_net(state)

    def dynamics(self, state, reward_hidden, action):
        """`action` is a LongTensor of shape (B,)."""
        b = state.size(0)
        action_plane = torch.ones((b, 1, state.size(2), state.size(3)), device=state.device, dtype=state.dtype)
        action_plane = action_plane * action.view(b, 1, 1, 1).to(state.dtype) / float(self.action_space_size)
        x = torch.cat((state, action_plane), dim=1)
        return self.dynamics_net(x, reward_hidden)

    def project(self, state, with_grad: bool = True):
        flat = state.reshape(state.size(0), -1)
        proj = self.projection(flat)
        if with_grad:
            return self.projection_head(proj)
        return proj.detach()

    # ----- inference API used by MCTS --------------------------------------------------
    def initial_inference(self, obs) -> NetOutput:
        state = self.representation(obs)
        policy_logits, value_logits = self.prediction(state)
        b = obs.size(0)
        zeros = torch.zeros(1, b, self.lstm_hidden, device=obs.device)
        prefix_zero = torch.zeros(b, 2 * self.support_size + 1, device=obs.device)
        return NetOutput(state, (zeros, zeros.clone()), prefix_zero, policy_logits, value_logits)

    def recurrent_inference(self, state, reward_hidden, action) -> NetOutput:
        next_state, reward_hidden, value_prefix_logits = self.dynamics(state, reward_hidden, action)
        policy_logits, value_logits = self.prediction(next_state)
        return NetOutput(next_state, reward_hidden, value_prefix_logits, policy_logits, value_logits)

    # ----- helpers ---------------------------------------------------------------------
    def value_scalar(self, value_logits):
        return support_to_scalar(value_logits, self.support_size)

    def new_reward_hidden(self, batch_size: int, device):
        zeros = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (zeros, zeros.clone())
