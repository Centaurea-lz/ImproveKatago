import numpy as np

from goai.board import BLACK, EMPTY, WHITE, GoBoard


def test_capture():
    b = GoBoard(size=5, komi=0, superko=False)
    b.board[:] = EMPTY
    b.board[1 * 5 + 1] = WHITE          # (1,1)
    b.board[0 * 5 + 1] = BLACK          # (0,1)
    b.board[1 * 5 + 0] = BLACK          # (1,0)
    b.board[1 * 5 + 2] = BLACK          # (1,2)
    b.current = BLACK
    assert b._would_be_legal(2 * 5 + 1)  # (2,1)
    b.play(2 * 5 + 1)
    assert b.board[1 * 5 + 1] == EMPTY   # white captured
    assert b.board[2 * 5 + 1] == BLACK


def test_suicide():
    b = GoBoard(size=5, komi=0, superko=False)
    b.board[:] = EMPTY
    b.board[0 * 5 + 1] = WHITE          # (0,1) opponent fills one liberty
    b.board[1 * 5 + 0] = WHITE          # (1,0) opponent fills the other
    b.current = BLACK
    assert not b._would_be_legal(0)      # (0,0) corner has no liberties -> suicide


def test_ko():
    b = GoBoard(size=5, komi=0, superko=False)
    b.board[:] = EMPTY
    b.board[6] = BLACK                   # (1,1) black stone to be captured
    b.board[1] = WHITE                   # (0,1)
    b.board[5] = WHITE                   # (1,0)
    b.board[11] = WHITE                  # (2,1) -> black's only liberty is (1,2)
    b.board[2] = BLACK                   # (0,2)
    b.board[12] = BLACK                  # (2,2)
    b.board[8] = BLACK                   # (1,3) -> capturing stone stays single, one liberty
    b.current = WHITE
    assert b._would_be_legal(7)          # (1,2) captures the single black stone
    b.play(7)
    assert b.board[6] == EMPTY           # black captured
    assert b.ko_point == 6
    assert b.current == BLACK
    assert not b._would_be_legal(6)      # immediate recapture forbidden


def test_two_pass_end_and_komi():
    b = GoBoard(size=9, komi=7.5, superko=False)
    b.play(b.pass_move)
    b.play(b.pass_move)
    assert b.done
    assert b.winner == WHITE             # empty board -> white wins by komi
    assert b.score == -7.5


def test_legal_moves_shape_and_pass():
    b = GoBoard(size=9)
    legal = b.legal_moves()
    assert legal.shape == (82,)
    assert legal[81]                     # pass always legal
    assert legal.sum() == 82             # empty board: every point + pass legal


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all board tests passed")
