"""PUCT Monte-Carlo tree search with GPU-batched leaf evaluation.

Key design points:
  - PUCT selection with first-play-urgency (cFPU), following KataGo.
  - Only the top-`max_children` moves (by policy prior) are expanded per node,
    keeping the tree small (Leela Zero / KataGo style).
  - Leaves are collected with *virtual loss* so a batch explores distinct
    leaves, then evaluated in one GPU forward pass.
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
                 num_simulations=100, batch_size=48,
                 dirichlet_alpha=0.03, dirichlet_eps=0.25,
                 max_children=48, pass_penalty=0.0):
        self.net = net.eval()
        self.device = device
        self.c_puct = c_puct
        self.c_fpu = c_fpu
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.max_children = max_children
        self.pass_penalty = pass_penalty
        self.root = None  # current tree root, reused across moves

    # ---- neural net ----
    def _evaluate(self, boards):
        xs = np.stack([encode(b) for b in boards])
        x = torch.from_numpy(xs).to(self.device)
        x = x.to(dtype=self.net.conv.weight.dtype)  # fp16 if the net is halved
        with torch.no_grad():
            logits, v_logits, _own, _score = self.net(x)
        # value logits -> scalar win probability in [-1, 1]
        values = 2.0 * torch.softmax(v_logits, dim=1)[:, 0] - 1.0
        return logits.cpu().numpy(), values.cpu().numpy()

    # ---- tree ops ----
    def _expand(self, node, logits, dirichlet=False):
        legal = node.board.legal_moves()
        l = logits.copy().astype(np.float32)  # avoid fp16 overflow in masking
        l[~legal] = -1e30
        l[node.board.pass_move] -= self.pass_penalty  # discourage premature pass
        l = l - l.max()
        probs = np.exp(l)
        probs /= probs.sum()

        if dirichlet:
            board_legal = legal.copy()
            board_legal[-1] = False
            n = int(board_legal.sum())
            if n > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * n)
                probs[board_legal] = ((1 - self.dirichlet_eps) * probs[board_legal]
                                      + self.dirichlet_eps * noise)
            probs /= probs.sum()

        # expand only the top-max_children moves (+ pass) to keep the tree small
        selected = {node.board.pass_move}
        for move in np.argsort(probs)[::-1]:
            move = int(move)
            if move == node.board.pass_move or not legal[move]:
                continue
            selected.add(move)
            if len(selected) >= self.max_children:
                break

        node.is_expanded = True
        for move in selected:
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
                q = parent_q - self.c_fpu * math.sqrt(max(explored_prior, 1e-8))
            else:
                q = c.q()
            u = self.c_puct * c.prior * math.sqrt(total) / (1 + c.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = c
        return best_child

    def _backup_virtual(self, path, value):
        """Convert virtual loss to real backup. path[0] is the root (real);
        path[1:] received virtual loss (visit+1, value-1)."""
        val = value
        for i in range(len(path) - 1, 0, -1):
            path[i].value_sum += (val + 1.0)
            val = -val
        path[0].visit_count += 1
        path[0].value_sum += val

    def _run_batch(self, root):
        leaves = []
        while len(leaves) < self.batch_size and root.visit_count < self.num_simulations:
            node = root
            path = [root]
            while node.is_expanded and not node.board.done:
                child = self._select(node)
                child.visit_count += 1   # virtual loss
                child.value_sum -= 1.0   # pessimistic (assume a loss)
                path.append(child)
                node = child
            if node.board.done:
                value = node.board.outcome_for(node.board.current)
                self._backup_virtual(path, value)
            else:
                leaves.append((node, path))
        if not leaves:
            return
        logits, values = self._evaluate([n.board for n, _ in leaves])
        for (leaf, path), logit, val in zip(leaves, logits, values):
            self._expand(leaf, logit)
            self._backup_virtual(path, float(val))

    # ---- public ----
    def _search_root(self, board: GoBoard, dirichlet=True):
        root = self.root
        if (root is None
                or not np.array_equal(root.board.board, board.board)
                or root.board.current != board.current
                or root.board.ko_point != board.ko_point):
            root = Node(board.clone())
            logits, values = self._evaluate([board])
            self._expand(root, logits[0], dirichlet=dirichlet)
            root.visit_count = 1
            root.value_sum = float(values[0])
        while root.visit_count < self.num_simulations:
            self._run_batch(root)
        self.root = root
        return root

    def search(self, board: GoBoard, dirichlet=True):
        """Run MCTS from `board`; return (policy over n+1 moves, root value)."""
        root = self._search_root(board, dirichlet=dirichlet)
        return self._policy(root), root.q()

    def best_move(self, board: GoBoard, lcb_factor=2.0):
        """Search and return the move with the highest lower-confidence bound.

        More robust than argmax(visits): favours moves whose value is high AND
        well-established (many visits), penalising uncertain few-visit moves.
        """
        root = self._search_root(board, dirichlet=False)
        total = root.visit_count
        best = board.pass_move
        best_lcb = -1e30
        for move, c in root.children.items():
            if c.visit_count == 0:
                continue
            q = c.q()
            lcb = q - lcb_factor * math.sqrt(math.log(total + 1) / c.visit_count)
            if lcb > best_lcb:
                best_lcb = lcb
                best = move
        return best

    def advance(self, move):
        """Descend to the child after `move` is played (tree reuse)."""
        if self.root is not None and move in self.root.children:
            child = self.root.children[move]
            child.parent = None
            self.root = child
        else:
            self.root = None

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
