"""Small-board divergence test: captures/ko happen far more often on small
boards, so this reproduces the fastboard bug much faster than 19x19."""
import numpy as np

from goai.board import GoBoard


def _slow_group(board, size, p):
    n = size * size
    color = board[p]
    seen = {p}
    stack = [p]
    libs = set()
    while stack:
        q = stack.pop()
        r, c = q // size, q % size
        for nb in (q - size, q + size, q - 1, q + 1):
            if nb < 0 or nb >= n:
                continue
            if nb == q - 1 and c == 0:
                continue
            if nb == q + 1 and c == size - 1:
                continue
            if board[nb] == 0:
                libs.add(nb)
            elif board[nb] == color and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return list(seen), libs


def slow_play_step(board, ko, current, move, size):
    n = size * size
    work = board.copy()
    work[move] = current
    opp = 3 - current
    captured = []
    r, c = move // size, move % size
    for nb in (move - size, move + size, move - 1, move + 1):
        if nb < 0 or nb >= n:
            continue
        if nb == move - 1 and c == 0:
            continue
        if nb == move + 1 and c == size - 1:
            continue
        if work[nb] == opp:
            group, libs = _slow_group(work, size, nb)
            if not libs:
                captured.extend(group)
                for p in group:
                    work[p] = 0
    group, libs = _slow_group(work, size, move)
    new_ko = captured[0] if (len(captured) == 1 and len(group) == 1
                             and len(libs) == 1) else -1
    return work, new_ko


def main():
    for size in (5, 7, 9):
        rng = np.random.default_rng(100 + size)
        for trial in range(200):
            b = GoBoard(size=size, superko=False)
            ref_board = np.zeros(size * size, dtype=np.int8)
            ref_ko = -1
            ref_current = 1
            for step in range(size * size * 3):
                if b.done:
                    break
                lm = b.legal_moves()
                moves = np.nonzero(lm[: b.pass_move])[0]
                if len(moves) == 0:
                    break
                m = int(rng.choice(moves))
                b.play(m)
                ref_board, ref_ko = slow_play_step(ref_board, ref_ko,
                                                   ref_current, m, size)
                ref_current = 3 - ref_current
                if not np.array_equal(b.board, ref_board) or b.ko_point != ref_ko:
                    print(f"DIVERGENCE size={size} trial={trial} step={step} "
                          f"move={m}")
                    print("fast board:")
                    print(b.board.reshape(size, size))
                    print("ref board:")
                    print(ref_board.reshape(size, size))
                    print(f"fast ko={b.ko_point} ref ko={ref_ko}")
                    return
        print(f"size {size}: 200 games, no divergence")
    print("ALL SMALL-BOARD GAMES MATCH")


if __name__ == "__main__":
    main()
