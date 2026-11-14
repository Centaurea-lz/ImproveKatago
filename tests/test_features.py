from goai.board import BLACK, GoBoard
from goai.features import encode


def test_encode_shape_and_planes():
    b = GoBoard(size=9)
    b.play(0)   # black
    b.play(1)   # white
    assert b.current == BLACK
    x = encode(b)
    assert x.shape == (17, 9, 9)
    assert (x[16] == 1.0).all()                     # black to move
    flat0 = x[0].reshape(-1)
    flat1 = x[1].reshape(-1)
    assert flat0[0] == 1.0                          # black stone at 0, newest board
    assert flat1[1] == 1.0                          # white stone at 1, newest board


if __name__ == "__main__":
    test_encode_shape_and_planes()
    print("PASS test_encode_shape_and_planes")
    print("all feature tests passed")
