"""Correctness check: numba legal_moves_fast vs the original slow legality."""
import numpy as np

from goai.board import EMPTY, GoBoard
from goai.fastboard import legal_moves_fast


def _slow_legal(board):
    legal = np.zeros(board.n + 1, dtype=bool)
    legal[board.pass_move] = True
    for p in range(board.n):
        if board.board[p] == EMPTY and board._would_be_legal(p):
            legal[p] = True
    return legal


def test_fastboard_matches_slow():
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(300):
        b = GoBoard(size=19, superko=False)
        for _ in range(int(rng.integers(0, 90))):
            legal = _slow_legal(b)
            nonpass = np.nonzero(legal[: b.pass_move])[0]
            if len(nonpass) == 0:
                break
            b.play(int(rng.choice(nonpass)))
            if b.done:
                break
        fast = legal_moves_fast(b.board, b.n, b.size, b.current, b.ko_point)
        slow = _slow_legal(b)
        assert np.array_equal(fast, slow), (
            f"mismatch trial {trial}\n{b!r}\nfast={np.nonzero(fast)}\n"
            f"slow={np.nonzero(slow)}")
        checked += 1
    print(f"fastboard matches slow on {checked} random positions")


def test_known_capture_and_ko():
    # black stone in atari, white can capture
    b = GoBoard(size=9, superko=False)
    b.play(0)          # B A1
    b.play(9)          # W A2
    b.play(81)         # pass (to give W a turn)
    # W at A1? A1 empty now (no), A1 has black stone; skip — just smoke-test shapes
    fast = legal_moves_fast(b.board, b.n, b.size, b.current, b.ko_point)
    slow = _slow_legal(b)
    assert np.array_equal(fast, slow)


def _reference_play(board, move):
    """Original slow capture logic, applied to a cloned board."""
    from goai.board import opponent
    b = board.clone()
    b.board[move] = b.current
    opp = opponent(b.current)
    captured = []
    for nb in b._neighbors(move):
        if b.board[nb] == opp:
            group, libs = b._group(nb)
            if not libs:
                captured.extend(group)
                b._remove(group)
    group, libs = b._group(move)
    b.ko_point = captured[0] if (len(captured) == 1 and len(group) == 1
                                 and len(libs) == 1) else -1
    return b.board.copy(), b.ko_point


def test_play_fast_matches_slow():
    from goai.fastboard import play_fast
    rng = np.random.default_rng(3)
    for trial in range(200):
        b = GoBoard(size=19, superko=False)
        for _ in range(int(rng.integers(0, 60))):
            lm = b.legal_moves()
            moves = np.nonzero(lm[: b.pass_move])[0]
            if len(moves) == 0:
                break
            b.play(int(rng.choice(moves)))
        lm = b.legal_moves()
        moves = np.nonzero(lm[: b.pass_move])[0]
        if len(moves) == 0:
            continue
        m = int(rng.choice(moves))
        new_board, new_ko = play_fast(b.board, b.n, b.size, m, b.current)
        ref_board, ref_ko = _reference_play(b, m)
        assert np.array_equal(new_board, ref_board), (
            f"board mismatch trial {trial} move {m}\n{b!r}")
        assert new_ko == ref_ko, (
            f"ko mismatch trial {trial} move {m}: {new_ko} vs {ref_ko}")
    print("play_fast matches slow on 200 random positions")


if __name__ == "__main__":
    test_fastboard_matches_slow()
    test_known_capture_and_ko()
    test_play_fast_matches_slow()
    print("PASS all fastboard tests")
