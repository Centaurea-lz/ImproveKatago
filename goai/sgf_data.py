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

from .board import BLACK, GoBoard
from .features import encode


def _winrate_from_comment(comment):
    """Extract the side-to-move winrate from a KataGo move comment.

    KataGo comments look like '0.47 0.53 0.00 -0.3 v=400' where the first
    number is the winrate of the side to move.
    """
    if not comment:
        return None
    try:
        wr = float(str(comment).split()[0])
        if 0.0 <= wr <= 1.0:
            return wr
    except (ValueError, IndexError):
        pass
    return None


def sgf_to_samples(data: bytes, board_size: int = 9):
    """Replay one SGF game; return (feat, move, value, ownership, score) samples.

    ownership is the final per-point ownership (+1/-1/0) from the mover's
    perspective; score is the final score difference / 20 from the mover's
    perspective.
    """
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

    komi = game.get_komi() or 7.5

    # first pass: collect moves and play to the end
    board = GoBoard(size=board_size, komi=komi, superko=False)
    moves = []
    try:
        for node in game.get_main_sequence():
            if any(node.get_setup_stones()):
                return []  # skip handicap / setup games for now
            colour, move = node.get_move()
            if colour is None:
                continue
            mi = board.pass_move if move is None else move[0] * board_size + move[1]
            try:
                comment = node.get("C")
            except KeyError:
                comment = None
            moves.append((colour, mi, comment))
            board.play(mi)
    except (ValueError, AssertionError):
        return []  # illegal under our rules -> skip

    own_black = board.ownership(BLACK)       # (size, size)
    score_black = board.score_for(BLACK) / 20.0

    # second pass: replay and generate samples
    board2 = GoBoard(size=board_size, komi=komi, superko=False)
    samples = []
    try:
        for colour, mi, comment in moves:
            feats = encode(board2)
            value = black_win if colour == "b" else -black_win  # hard game outcome
            if colour == "b":
                own, score = own_black, score_black
            else:
                own, score = -own_black, -score_black
            samples.append((feats, mi, value, own, score))
            board2.play(mi)
    except (ValueError, AssertionError):
        return []
    return samples


def iter_sgf_files(directory):
    for root, _dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(".sgf") or f.endswith(".sgf.gz") or f.endswith(".sgfs"):
                yield os.path.join(root, f)


def _split_sgfs(data):
    """Split a multi-game .sgfs file into individual game byte strings."""
    text = data.decode("utf-8", "replace")
    games = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                games.append(text[start:i + 1].encode("utf-8"))
                start = None
    return games


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
    O = np.zeros((max_samples, board_size, board_size), dtype=np.int8)  # -1/0/1
    S = np.zeros(max_samples, dtype=np.float32)
    idx = 0
    for path in iter_sgf_files(sgf_dir):
        data = _read(path)
        games = _split_sgfs(data) if path.endswith(".sgfs") else [data]
        for game_data in games:
            try:
                samples = sgf_to_samples(game_data, board_size)
            except Exception:
                continue
            for x, m, v, o, s in samples:
                X[idx] = x
                M[idx] = m
                V[idx] = v
                O[idx] = o
                S[idx] = s
                idx += 1
                if idx >= max_samples:
                    break
            if idx >= max_samples:
                break
        if idx >= max_samples:
            break
    if idx == 0:
        raise RuntimeError("no samples parsed from " + sgf_dir)
    np.savez_compressed(out_path, X=X[:idx], move=M[:idx], value=V[:idx],
                        own=O[:idx], score=S[:idx])
    return idx


def moves_to_sgf(moves, size, komi, winner):
    """Convert a game (flat move indices, pass=size*size) to SGF text.

    `winner` is BLACK(1) or WHITE(2). Coordinates use the SGF convention
    (top-left origin); our board uses row-from-bottom, so rows are flipped.
    """
    head = (f"(;GM[1]FF[4]CA[UTF-8]SZ[{size}]KM[{komi:g}]"
            f"PB[Black]PW[White]RE[{'B' if winner == BLACK else 'W'}+R]")
    body = []
    for i, m in enumerate(moves):
        color = "B" if i % 2 == 0 else "W"
        if m == size * size:
            body.append(f";{color}[]")
        else:
            r, c = divmod(int(m), size)
            pt = chr(97 + c) + chr(97 + (size - 1 - r))
            body.append(f";{color}[{pt}]")
    return head + "".join(body) + ")"
