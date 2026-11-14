"""Parse SGF game records (KGS) into supervised training samples.

Each sample is (features, move, value) where
  - features: (17, size, size) tensor from the mover's perspective,
  - move: flat index of the played move (n == pass),
  - value: +1/-1 game outcome from the mover's perspective.
"""
from __future__ import annotations

import gzip
import os

import numpy as np
from sgfmill import sgf

from .board import GoBoard
from .features import encode


def sgf_to_samples(data: bytes, board_size: int = 9):
    """Replay one SGF game and return its samples, or [] if unsuitable."""
    try:
        game = sgf.Sgf_game.from_bytes(data)
    except Exception:
        return []
    if game.get_size() != board_size:
        return []
    winner = game.get_winner()  # 'b' | 'w' | None
    if winner == "b":
        black_win = 1.0
    elif winner == "w":
        black_win = -1.0
    else:
        return []  # no decisive result

    board = GoBoard(size=board_size, komi=game.get_komi() or 7.5, superko=False)
    samples = []
    try:
        for node in game.get_main_sequence():
            if any(node.get_setup_stones()):
                return []  # skip handicap / setup games for now
            colour, move = node.get_move()
            if colour is None:
                continue
            if move is None:
                mi = board.pass_move
            else:
                mi = move[0] * board_size + move[1]
            value = black_win if colour == "b" else -black_win
            samples.append((encode(board), mi, value))
            board.play(mi)
    except (ValueError, AssertionError):
        return []  # illegal under our rules -> skip
    return samples


def iter_sgf_files(directory):
    for root, _dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(".sgf") or f.endswith(".sgf.gz"):
                yield os.path.join(root, f)


def _read(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return fh.read()
    with open(path, "rb") as fh:
        return fh.read()


def build_dataset(sgf_dir, out_path, board_size=9, max_samples=1_000_000):
    """Parse a directory of SGF files into a compressed .npz dataset."""
    X = np.zeros((max_samples, 17, board_size, board_size), dtype=np.uint8)  # 0/1 features
    M = np.zeros(max_samples, dtype=np.int64)
    V = np.zeros(max_samples, dtype=np.float32)
    idx = 0
    for path in iter_sgf_files(sgf_dir):
        try:
            samples = sgf_to_samples(_read(path), board_size)
        except Exception:
            continue
        for x, m, v in samples:
            X[idx] = x
            M[idx] = m
            V[idx] = v
            idx += 1
            if idx >= max_samples:
                break
        if idx >= max_samples:
            break
    if idx == 0:
        raise RuntimeError("no samples parsed from " + sgf_dir)
    np.savez_compressed(out_path, X=X[:idx], move=M[:idx], value=V[:idx])
    return idx
