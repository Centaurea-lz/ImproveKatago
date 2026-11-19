"""Find the first divergence between numba fastboard and the slow reference,
over full games (to catch ko/capture edge cases the single-move test misses)."""
import numpy as np

from goai.board import GoBoard
from goai.fastboard import legal_moves_fast, play_fast

SIZE = 19
N = SIZE * SIZE


def _slow_group(board, p):
    color = board[p]
    seen = set()
    stack = [p]
    seen.add(p)
    libs = set()
    while stack:
        q = stack.pop()
        r, c = q // SIZE, q % SIZE
        for nb in (q - SIZE, q + SIZE, q - 1, q + 1):
            if nb < 0 or nb >= N:
                continue
            if nb == q - 1 and c == 0:
                continue
            if nb == q + 1 and c == SIZE - 1:
                continue
            if board[nb] == 0:
                libs.add(nb)
            elif board[nb] == color and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return list(seen), libs


def slow_play_step(board, ko, current, move):
    work = board.copy()
    work[move] = current
    opp = 3 - current
    captured = []
    r, c = move // SIZE, move % SIZE
    for nb in (move - SIZE, move + SIZE, move - 1, move + 1):
        if nb < 0 or nb >= N:
            continue
        if nb == move - 1 and c == 0:
            continue
        if nb == move + 1 and c == SIZE - 1:
            continue
        if work[nb] == opp:
            group, libs = _slow_group(work, nb)
            if not libs:
                captured.extend(group)
                for p in group:
                    work[p] = 0
    group, libs = _slow_group(work, move)
    new_ko = captured[0] if (len(captured) == 1 and len(group) == 1
                             and len(libs) == 1) else -1
    return work, new_ko


def main():
    rng = np.random.default_rng(42)
    for trial in range(20):
        b = GoBoard(size=SIZE, superko=False)
        ref_board = np.zeros(N, dtype=np.int8)
        ref_ko = -1
        ref_current = 1
        for step in range(400):
            if b.done:
                break
            lm = b.legal_moves()
            moves = np.nonzero(lm[: b.pass_move])[0]
            if len(moves) == 0:
                break
            m = int(rng.choice(moves))
            b.play(m)
            ref_board, ref_ko = slow_play_step(ref_board, ref_ko, ref_current, m)
            ref_current = 3 - ref_current
            if not np.array_equal(b.board, ref_board) or b.ko_point != ref_ko:
                print(f"DIVERGENCE trial {trial} step {step} move {m} "
                      f"current_before={3 - ref_current}")
                diff = np.nonzero(b.board != ref_board)[0]
                print("differing points:", diff)
                print("fast board:")
                print(b.board.reshape(SIZE, SIZE))
                print("ref board:")
                print(ref_board.reshape(SIZE, SIZE))
                print(f"fast ko={b.ko_point} ref ko={ref_ko}")
                return
        print(f"trial {trial}: {step} moves, no divergence")
    print("ALL FULL GAMES MATCH")


if __name__ == "__main__":
    main()
