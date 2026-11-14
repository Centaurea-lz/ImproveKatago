"""PUCT Monte-Carlo tree search with GPU-batched leaf evaluation.

Key design points (the engine contribution of the project):
  - PUCT selection with first-play-urgency (cFPU), following KataGo.
  - Leaves are collected into a batch and evaluated in a single GPU forward pass
    ("GPU batch parallel evaluation at leaf nodes").
  - Dirichlet noise at the root for exploration.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .board import GoBoard
from .features import encode


class Node:
    __slots__ = ("board", "move", "parent", "prior", "children",
                 "visit_count", "value_sum", "is_expanded")

    def __init__(self, board, move=None, parent=None, prior=0.0):
        self.board = board
        self.move = move
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def q(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(self, net, device="cuda", c_puct=1.1, c_fpu=0.2,
                 num_simulations=200, batch_size=64,
                 dirichlet_alpha=0.03, dirichlet_eps=0.25):
        self.net = net.eval()
        self.device = device
        self.c_puct = c_puct
        self.c_fpu = c_fpu
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    # ---- neural net ----
    def _evaluate(self, boards):
        xs = np.stack([encode(b) for b in boards])
        x = torch.from_numpy(xs).to(self.device)
        with torch.no_grad():
            logits, values = self.net(x)
        return logits.cpu().numpy(), values.cpu().numpy().reshape(-1)

    # ---- tree ops ----
    def _expand(self, node, logits, dirichlet=False):
        legal = node.board.legal_moves()
        l = logits.copy()
        l[~legal] = -1e30
        l = l - l.max()
        probs = np.exp(l)
        probs /= probs.sum()

        if dirichlet:
            board_legal = legal.copy()
            board_legal[-1] = False  # no noise on the pass move
            n = int(board_legal.sum())
            if n > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * n)
                probs[board_legal] = ((1 - self.dirichlet_eps) * probs[board_legal]
                                      + self.dirichlet_eps * noise)

        node.is_expanded = True
        for move in np.nonzero(legal)[0]:
            move = int(move)
            child_board = node.board.clone()
            child_board.play(move)
            node.children[move] = Node(child_board, move=move, parent=node,
                                       prior=float(probs[move]))

    def _select(self, node):
        total = node.visit_count
        parent_q = node.q()
        explored_prior = sum(c.prior for c in node.children.values()
                             if c.visit_count > 0)
        best_child = None
        best_score = -1e30
        for c in node.children.values():
            if c.visit_count == 0:
                # first-play urgency (KataGo): V(c) = V(n) - cFPU*sqrt(P_explored)
                q = parent_q - self.c_fpu * math.sqrt(max(explored_prior, 1e-8))
            else:
                q = c.q()
            u = self.c_puct * c.prior * math.sqrt(total) / (1 + c.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = c
        return best_child

    def _backup(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _run_batch(self, root):
        leaves = []
        while len(leaves) < self.batch_size:
            node = root
            path = []
            while node.is_expanded and not node.board.done:
                path.append(node)
                node = self._select(node)
            if node.board.done:
                self._backup(path, -node.board.outcome_for(node.board.current))
            else:
                leaves.append((node, path))
        if not leaves:
            return
        logits, values = self._evaluate([n.board for n, _ in leaves])
        for (leaf, path), logit, val in zip(leaves, logits, values):
            self._expand(leaf, logit)
            leaf.visit_count = 1
            leaf.value_sum = float(val)
            self._backup(path, -float(val))

    # ---- public ----
    def search(self, board: GoBoard, dirichlet=True) -> np.ndarray:
        """Run MCTS from `board`, return visit-count policy over n+1 moves."""
        root = Node(board.clone())
        logits, values = self._evaluate([board])
        self._expand(root, logits[0], dirichlet=dirichlet)
        root.visit_count = 1
        root.value_sum = float(values[0])
        while root.visit_count < self.num_simulations:
            self._run_batch(root)
        return self._policy(root)

    def _policy(self, root) -> np.ndarray:
        n = root.board.n + 1
        counts = np.zeros(n, dtype=np.float64)
        for move, c in root.children.items():
            counts[move] = c.visit_count
        s = counts.sum()
        return counts / s if s > 0 else counts


def sample_move(probs: np.ndarray, temperature: float = 1.0, legal=None):
    """Sample a move from a probability distribution, with temperature."""
    if legal is not None:
        probs = probs * legal
        if probs.sum() <= 0:
            return int(np.nonzero(legal)[0][0])
    if temperature <= 0:
        return int(np.argmax(probs))
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
