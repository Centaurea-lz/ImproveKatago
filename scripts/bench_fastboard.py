"""Correctness + speed benchmark for the numba legal_moves_fast kernel."""
import time

import numpy as np

from goai.board import EMPTY, GoBoard
from goai.fastboard import legal_moves_fast


def slow_legal(b):
    legal = np.zeros(b.n + 1, dtype=bool)
    legal[b.pass_move] = True
    for p in range(b.n):
        if b.board[p] == EMPTY and b._would_be_legal(p):
            legal[p] = True
    return legal


def main():
    rng = np.random.default_rng(1)
    for trial in range(100):
        b = GoBoard(size=19, superko=False)
        for _ in range(int(rng.integers(0, 90))):
            lm = b.legal_moves()
            moves = np.nonzero(lm[: b.pass_move])[0]
            if len(moves) == 0:
                break
            b.play(int(rng.choice(moves)))
        fast = legal_moves_fast(b.board, b.n, b.size, b.current, b.ko_point)
        slow = slow_legal(b)
        assert np.array_equal(fast, slow), f"mismatch trial {trial}"
    print("correctness: OK (100 random positions)")

    b = GoBoard(size=19, superko=False)
    rng2 = np.random.default_rng(2)
    for _ in range(60):
        lm = b.legal_moves()
        moves = np.nonzero(lm[: b.pass_move])[0]
        if len(moves) == 0:
            break
        b.play(int(rng2.choice(moves)))

    for _ in range(10):  # numba warm-up / compile
        legal_moves_fast(b.board, b.n, b.size, b.current, b.ko_point)

    t = time.time()
    for _ in range(500):
        legal_moves_fast(b.board, b.n, b.size, b.current, b.ko_point)
    t_fast = (time.time() - t) / 500

    t = time.time()
    for _ in range(20):
        slow_legal(b)
    t_slow = (time.time() - t) / 20

    print(f"legal_moves: fast {t_fast * 1000:.3f}ms, slow {t_slow * 1000:.3f}ms, "
          f"speedup {t_slow / t_fast:.1f}x")


if __name__ == "__main__":
    main()
