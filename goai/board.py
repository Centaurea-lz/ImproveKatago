"""Go board logic (Tromp-Taylor-style rules, 9x9 primary, any size).

Features
--------
- stone placement, captures, suicide prohibition
- simple ko + optional superko (position keyed by board + side to move)
- pass, two-consecutive-passes game end
- Tromp-Taylor area scoring with komi

Coordinates are flat indices ``0 .. size*size-1``, row-major (``p = r*size + c``).
The pass move is encoded as ``size*size``.
"""
from __future__ import annotations

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2


def opponent(color: int) -> int:
    return 3 - color  # BLACK <-> WHITE


class GoBoard:
    def __init__(self, size: int = 9, komi: float = 7.5, superko: bool = True,
                 history_len: int = 8):
        self.size = size
        self.n = size * size
        self.pass_move = self.n
        self.komi = komi
        self.superko = superko
        self.history_len = history_len

        self.board = np.zeros(self.n, dtype=np.int8)   # 0 empty, 1 black, 2 white
        self.current = BLACK
        self.consecutive_passes = 0
        self.ko_point = -1                              # forbidden point under simple ko
        self.recent: list[np.ndarray] = [self.board.copy()]  # newest last
        self.done = False
        self.winner = EMPTY
        self.score = 0.0                                # final score, black's perspective
        if self.superko:
            self._pos_seen = set()
            self._pos_seen.add(self._pos_key())

    # ---- position key / clone ----
    def _pos_key(self):
        return (self.board.tobytes(), self.current)

    def clone(self) -> "GoBoard":
        b = GoBoard.__new__(GoBoard)
        b.size = self.size
        b.n = self.n
        b.pass_move = self.pass_move
        b.komi = self.komi
        b.superko = self.superko
        b.history_len = self.history_len
        b.board = self.board.copy()
        b.current = self.current
        b.consecutive_passes = self.consecutive_passes
        b.ko_point = self.ko_point
        b.recent = [x.copy() for x in self.recent]
        b.done = self.done
        b.winner = self.winner
        b.score = self.score
        if self.superko:
            b._pos_seen = set(self._pos_seen)
        return b

    # ---- geometry ----
    def _neighbors(self, p: int):
        r, c = divmod(p, self.size)
        if r > 0:
            yield p - self.size
        if r < self.size - 1:
            yield p + self.size
        if c > 0:
            yield p - 1
        if c < self.size - 1:
            yield p + 1

    def _group(self, p: int, board: np.ndarray | None = None):
        """Return (group, liberties) of the stone/group containing p."""
        board = self.board if board is None else board
        color = board[p]
        assert color != EMPTY
        group: list[int] = []
        liberties: set[int] = set()
        seen: set[int] = set()
        stack = [p]
        while stack:
            q = stack.pop()
            if q in seen:
                continue
            seen.add(q)
            group.append(q)
            for nb in self._neighbors(q):
                if board[nb] == EMPTY:
                    liberties.add(nb)
                elif board[nb] == color:
                    stack.append(nb)
        return group, liberties

    def _remove(self, group, board: np.ndarray | None = None):
        board = self.board if board is None else board
        for p in group:
            board[p] = EMPTY

    # ---- legality ----
    def _would_be_legal(self, p: int) -> bool:
        if p == self.pass_move:
            return True
        if self.board[p] != EMPTY or p == self.ko_point:
            return False
        b = self.board.copy()
        opp = opponent(self.current)
        b[p] = self.current
        captured = 0
        for nb in self._neighbors(p):
            if b[nb] == opp:
                group, libs = self._group(nb, b)
                if not libs:
                    captured += len(group)
                    self._remove(group, b)
        group, libs = self._group(p, b)
        if not libs:                       # suicide
            return False
        if self.superko:
            key = (b.tobytes(), opponent(self.current))
            if key in self._pos_seen:
                return False
        return True

    def legal_moves(self) -> np.ndarray:
        """Boolean array of length n+1; index n is the pass move."""
        legal = np.zeros(self.n + 1, dtype=bool)
        if self.done:
            return legal
        legal[self.pass_move] = True
        for p in range(self.n):
            if self.board[p] == EMPTY and self._would_be_legal(p):
                legal[p] = True
        return legal

    # ---- play ----
    def play(self, move: int) -> "GoBoard":
        if self.done:
            raise ValueError("game already over")
        if not self._would_be_legal(move):
            raise ValueError(f"illegal move {move}")

        if move == self.pass_move:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
            opp = opponent(self.current)
            self.board[move] = self.current
            captured: list[int] = []
            for nb in self._neighbors(move):
                if self.board[nb] == opp:
                    group, libs = self._group(nb)
                    if not libs:
                        captured.extend(group)
                        self._remove(group)
            group, libs = self._group(move)
            if len(captured) == 1 and len(group) == 1 and len(libs) == 1:
                self.ko_point = captured[0]
            else:
                self.ko_point = -1

        self.recent.append(self.board.copy())
        if len(self.recent) > self.history_len:
            self.recent.pop(0)

        self.current = opponent(self.current)
        if self.superko:
            self._pos_seen.add(self._pos_key())

        if self.consecutive_passes >= 2:
            self._end_game()
        return self

    # ---- scoring ----
    def _area_scores(self):
        black = int((self.board == BLACK).sum())
        white = int((self.board == WHITE).sum())
        visited = np.zeros(self.n, dtype=bool)
        for p in range(self.n):
            if self.board[p] != EMPTY or visited[p]:
                continue
            region: list[int] = []
            stack = [p]
            visited[p] = True
            border: set[int] = set()
            while stack:
                q = stack.pop()
                region.append(q)
                for nb in self._neighbors(q):
                    if self.board[nb] == EMPTY:
                        if not visited[nb]:
                            visited[nb] = True
                            stack.append(nb)
                    else:
                        border.add(int(self.board[nb]))
            if border == {BLACK}:
                black += len(region)
            elif border == {WHITE}:
                white += len(region)
        return black, white

    def _end_game(self):
        self.done = True
        black, white = self._area_scores()
        self.score = black - white - self.komi  # black's perspective (white gets komi)
        if self.score > 0:
            self.winner = BLACK
        elif self.score < 0:
            self.winner = WHITE
        else:
            self.winner = EMPTY

    def ownership(self, color: int):
        """Per-point ownership from `color`'s perspective: +1 mine, -1 opp, 0 neutral."""
        own = np.zeros(self.n, dtype=np.float32)
        own[self.board == color] = 1.0
        own[self.board == opponent(color)] = -1.0
        visited = np.zeros(self.n, dtype=bool)
        for p in range(self.n):
            if self.board[p] != EMPTY or visited[p]:
                continue
            region = []
            stack = [p]
            visited[p] = True
            border = set()
            while stack:
                q = stack.pop()
                region.append(q)
                for nb in self._neighbors(q):
                    if self.board[nb] == EMPTY:
                        if not visited[nb]:
                            visited[nb] = True
                            stack.append(nb)
                    else:
                        border.add(int(self.board[nb]))
            if border == {color}:
                for q in region:
                    own[q] = 1.0
            elif border == {opponent(color)}:
                for q in region:
                    own[q] = -1.0
        return own.reshape(self.size, self.size)

    def score_for(self, color: int) -> float:
        """Final score difference (points) from `color`'s perspective."""
        black, white = self._area_scores()
        diff = black - white - self.komi  # black's perspective
        return diff if color == BLACK else -diff

    def outcome_for(self, color: int) -> float:
        """+1 if `color` won, -1 if lost, 0 if draw (only meaningful when done)."""
        if self.winner == EMPTY:
            return 0.0
        return 1.0 if self.winner == color else -1.0

    def __repr__(self):
        glyph = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = []
        for r in range(self.size):
            rows.append(" ".join(glyph[int(self.board[r * self.size + c])]
                                 for c in range(self.size)))
        return "\n".join(rows)
