"""Dual-head residual network (policy + value) in PyTorch.

AlphaZero-style pre-activation ResNet. Auxiliary heads (ownership / score)
from the KataGo paper are added later as a documented improvement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 9, in_planes: int = 17,
                 channels: int = 64, blocks: int = 6):
        super().__init__()
        self.board_size = board_size
        self.n = board_size * board_size

        self.conv = nn.Conv2d(in_planes, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        # policy head: logits over n board points + 1 pass
        self.policy_conv = nn.Conv2d(channels, 1, 1)
        self.policy_bn = nn.BatchNorm2d(1)
        self.policy_pass = nn.Linear(channels, 1)

        # value head: scalar in [-1, 1] (current player's winning probability -> tanh)
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for blk in self.blocks:
            x = blk(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))      # (B,1,S,S)
        p_board = p.view(p.size(0), -1)                     # (B,n)
        g = x.mean(dim=(2, 3))                              # global mean pool -> (B,C)
        p_pass = self.policy_pass(g)                        # (B,1)
        policy = torch.cat([p_board, p_pass], dim=1)        # (B,n+1)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))       # (B,1,S,S)
        v = v.view(v.size(0), -1)
        v = self.value_fc2(F.relu(self.value_fc1(v)))
        v = torch.tanh(v)                                   # (B,1)
        return policy, v


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
