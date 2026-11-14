from goai.sgf_data import sgf_to_samples

SGF = ("(;GM[1]FF[4]SZ[9]KM[7.5]RE[B+5.5]PB[black]PW[white]"
       ";B[ee];W[dd];B[cc];W[cd];B[ed])")


def test_sgf_to_samples():
    samples = sgf_to_samples(SGF.encode(), board_size=9)
    assert len(samples) == 5
    # value from mover's perspective: black wins -> B moves +1, W moves -1
    values = [v for _, _, v in samples]
    assert values == [1.0, -1.0, 1.0, -1.0, 1.0]
    for x, m, _ in samples:
        assert x.shape == (17, 9, 9)
        assert 0 <= m <= 81
    # first move is the centre point (orientation-independent)
    assert samples[0][1] == 4 * 9 + 4


def test_wrong_size_skipped():
    sgf19 = ("(;GM[1]FF[4]SZ[19]KM[7.5]RE[B+1.5]PB[b]PW[w];B[pp])")
    assert sgf_to_samples(sgf19.encode(), board_size=9) == []


if __name__ == "__main__":
    test_sgf_to_samples()
    print("PASS test_sgf_to_samples")
    test_wrong_size_skipped()
    print("PASS test_wrong_size_skipped")
    print("all sgf tests passed")
