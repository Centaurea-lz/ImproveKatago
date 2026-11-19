"""Numba-accelerated Go board kernels for the hot MCTS path.

The pure-Python ``GoBoard.legal_moves()`` is the dominant MCTS cost: it calls
``_would_be_legal(p)`` for every point, and that method copies the whole board
361 times and flood-fills per point. These kernels run simple-ko legality on a
flat int8 board with preallocated scratch buffers, so there is one board copy
per *call* (not per point) and no interpreter overhead in the inner loops.

Only simple ko (``ko_point``) is handled here; superko positions fall back to
the Python implementation (self-play / MCTS use ``superko=False`` anyway).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # graceful fallback so board.py imports without numba
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@njit(cache=True)
def _group_liberties(board, n, size, p, visited, stack):
    """Flood-fill the group at ``p``; return its distinct liberty count.

    ``visited`` is a scratch int8 array: 0 = unseen, 1 = group stone,
    2 = already-counted liberty. ``stack`` is a scratch int32 array of size n.
    """
    color = board[p]
    sp = 0
    stack[sp] = p
    sp += 1
    visited[p] = 1
    libs = 0
    while sp > 0:
        sp -= 1
        q = stack[sp]
        r = q // size
        c = q % size
        if r > 0:
            nb = q - size
            v = board[nb]
            if v == 0:
                if visited[nb] != 2:
                    visited[nb] = 2
                    libs += 1
            elif v == color and visited[nb] == 0:
                visited[nb] = 1
                stack[sp] = nb
                sp += 1
        if r < size - 1:
            nb = q + size
            v = board[nb]
            if v == 0:
                if visited[nb] != 2:
                    visited[nb] = 2
                    libs += 1
            elif v == color and visited[nb] == 0:
                visited[nb] = 1
                stack[sp] = nb
                sp += 1
        if c > 0:
            nb = q - 1
            v = board[nb]
            if v == 0:
                if visited[nb] != 2:
                    visited[nb] = 2
                    libs += 1
            elif v == color and visited[nb] == 0:
                visited[nb] = 1
                stack[sp] = nb
                sp += 1
        if c < size - 1:
            nb = q + 1
            v = board[nb]
            if v == 0:
                if visited[nb] != 2:
                    visited[nb] = 2
                    libs += 1
            elif v == color and visited[nb] == 0:
                visited[nb] = 1
                stack[sp] = nb
                sp += 1
    return libs


@njit(cache=True)
def legal_moves_fast(board, n, size, current, ko_point):
    """Return a bool array of length ``n + 1`` (index ``n`` is pass)."""
    opp = 3 - current
    work = board.copy()
    legal = np.zeros(n + 1, dtype=np.bool_)
    legal[n] = True
    visited = np.zeros(n, dtype=np.int8)
    stack = np.empty(n, dtype=np.int32)

    for p in range(n):
        if work[p] != 0 or p == ko_point:
            continue
        work[p] = current
        visited[:] = 0
        if _group_liberties(work, n, size, p, visited, stack) > 0:
            legal[p] = True
            work[p] = 0
            continue
        # no liberties -> suicide unless it captures an adjacent opponent group
        captured = False
        r = p // size
        c = p % size
        if r > 0 and work[p - size] == opp:
            visited[:] = 0
            if _group_liberties(work, n, size, p - size, visited, stack) == 0:
                captured = True
        if not captured and r < size - 1 and work[p + size] == opp:
            visited[:] = 0
            if _group_liberties(work, n, size, p + size, visited, stack) == 0:
                captured = True
        if not captured and c > 0 and work[p - 1] == opp:
            visited[:] = 0
            if _group_liberties(work, n, size, p - 1, visited, stack) == 0:
                captured = True
        if not captured and c < size - 1 and work[p + 1] == opp:
            visited[:] = 0
            if _group_liberties(work, n, size, p + 1, visited, stack) == 0:
                captured = True
        legal[p] = captured
        work[p] = 0

    return legal


@njit(cache=True)
def _collect_group(board, n, size, p, visited, stack, members):
    """Flood-fill the group at p; store its points in `members`, return count."""
    color = board[p]
    sp = 0
    stack[sp] = p
    sp += 1
    visited[p] = 1
    cnt = 0
    while sp > 0:
        sp -= 1
        q = stack[sp]
        members[cnt] = q
        cnt += 1
        r = q // size
        c = q % size
        if r > 0:
            nb = q - size
            if board[nb] == color and visited[nb] == 0:
                visited[nb] = 1
                stack[sp] = nb
                sp += 1
        if r < size - 1:
            nb = q + size
            if board[nb] == color and visited[nb] == 0:
                visited[nb] = 1
                stack[sp] = nb
                sp += 1
        if c > 0:
            nb = q - 1
            if board[nb] == color and visited[nb] == 0:
                visited[nb] = 1
                stack[sp] = nb
                sp += 1
        if c < size - 1:
            nb = q + 1
            if board[nb] == color and visited[nb] == 0:
                visited[nb] = 1
                stack[sp] = nb
                sp += 1
    return cnt


@njit(cache=True)
def play_fast(board, n, size, move, current):
    """Place `current` at `move` (assumed legal, non-pass); remove captures and
    compute the simple-ko point. Returns (new_board, new_ko_point)."""
    opp = 3 - current
    work = board.copy()
    work[move] = current
    visited = np.zeros(n, dtype=np.int8)
    stack = np.empty(n, dtype=np.int32)
    members = np.empty(n, dtype=np.int32)

    captured = 0
    captured_point = -1
    r = move // size
    c = move % size

    # remove each captured (zero-liberty) opponent neighbour group
    for nb in (move - size, move + size, move - 1, move + 1):
        if nb < 0 or nb >= n:
            continue
        if nb == move - 1 and c == 0:
            continue
        if nb == move + 1 and c == size - 1:
            continue
        if work[nb] != opp:
            continue
        visited[:] = 0
        if _group_liberties(work, n, size, nb, visited, stack) == 0:
            visited[:] = 0
            cnt = _collect_group(work, n, size, nb, visited, stack, members)
            captured += cnt
            if cnt == 1:
                captured_point = members[0]
            for i in range(cnt):
                work[members[i]] = 0

    # simple ko: exactly one stone captured, capturing group is a single stone
    # with a single liberty
    new_ko = -1
    if captured == 1:
        visited[:] = 0
        placed_size = _collect_group(work, n, size, move, visited, stack, members)
        visited[:] = 0
        placed_libs = _group_liberties(work, n, size, move, visited, stack)
        if placed_size == 1 and placed_libs == 1:
            new_ko = captured_point

    return work, new_ko
