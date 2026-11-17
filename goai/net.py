"""Dual-head (policy + value) residual network with auxiliary heads.

Additions over the plain AlphaZero net (from the KataGo paper):
  - global pooling bias in some residual blocks (whole-board context)
  - ownership head: predicts per-point final ownership (+1/-1/0)
  - score head: predicts the final score difference (scaled by /20)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPool(nn.Module):
    """Global mean+max pooling -> per-channel bias (whole-board context)."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels * 2, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        mean = x.mean(dim=(2, 3))
        maxv = x.amax(dim=(2, 3))
        g = torch.cat([mean, maxv], dim=1)
        return self.fc(g).view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_global: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gp = GlobalPool(channels) if use_global else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.gp is not None:
            out = out + self.gp(out)  # inject whole-board context
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 19, in_planes: int = 17,
                 channels: int = 96, blocks: int = 8):
        super().__init__()
        self.board_size = board_size
        self.n = board_size * board_size

        self.conv = nn.Conv2d(in_planes, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, use_global=(i % 4 == 3))
            for i in range(blocks)
        ])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 1, 1)
        self.policy_bn = nn.BatchNorm2d(1)
        self.policy_pass = nn.Linear(channels, 1)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, channels)
        self.value_fc2 = nn.Linear(channels, 1)

        # ownership head: per-point +1 (mine) / -1 (opponent) / 0
        self.own_conv = nn.Conv2d(channels, 1, 1)
        self.own_bn = nn.BatchNorm2d(1)

        # score head: final score difference / 20
        self.score_fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for blk in self.blocks:
            x = blk(x)
        g = x.mean(dim=(2, 3))  # global mean pool

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p_board = p.view(p.size(0), -1)
        policy = torch.cat([p_board, self.policy_pass(g)], dim=1)

        # value
        v = F.relu(self.value_bn(self.value_conv(x))).view(x.size(0), -1)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))

        # ownership + score
        own = torch.tanh(self.own_conv(x))
        score = self.score_fc(g)
        return policy, v, own, score


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
