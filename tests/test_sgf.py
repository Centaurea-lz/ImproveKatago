from goai.sgf_data import sgf_to_samples

SGF = ("(;GM[1]FF[4]SZ[9]KM[7.5]RE[B+5.5]PB[black]PW[white]"
       ";B[ee];W[dd];B[cc];W[cd];B[ed])")


def test_sgf_to_samples():
    samples = sgf_to_samples(SGF.encode(), board_size=9)
    assert len(samples) == 5
    # value from mover's perspective: black wins -> B moves +1, W moves -1
    values = [v for _, _, v, _, _ in samples]
    assert values == [1.0, -1.0, 1.0, -1.0, 1.0]
    for x, m, _v, o, _s in samples:
        assert x.shape == (17, 9, 9)
        assert 0 <= m <= 81
        assert o.shape == (9, 9)
    # first move is the centre point (orientation-independent)
    assert samples[0][1] == 4 * 9 + 4


def test_wrong_size_skipped():
    sgf19 = ("(;GM[1]FF[4]SZ[19]KM[7.5]RE[B+1.5]PB[b]PW[w];B[pp])")
    assert sgf_to_samples(sgf19.encode(), board_size=9) == []


def test_moves_to_sgf_roundtrip():
    from goai.board import BLACK
    from goai.sgf_data import moves_to_sgf
    size = 9
    moves = [40, 30, 20, 47, size * size]  # last is a pass
    s = moves_to_sgf(moves, size, 7.5, BLACK)
    samples = sgf_to_samples(s.encode(), board_size=size)
    assert [m for _, m, _v, _o, _s in samples] == moves


if __name__ == "__main__":
    test_sgf_to_samples()
    print("PASS test_sgf_to_samples")
    test_wrong_size_skipped()
    print("PASS test_wrong_size_skipped")
    test_moves_to_sgf_roundtrip()
    print("PASS test_moves_to_sgf_roundtrip")
    print("all sgf tests passed")
