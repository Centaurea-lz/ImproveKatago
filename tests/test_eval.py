"""Test GTP coordinate conversion and GNU Go communication (no GPU)."""
from goai.eval import GTPEngine, flat_to_gtp, gtp_to_flat


def test_coord_roundtrip():
    size = 19
    for m in [0, 8, 18, 19, 60, 360, size * size]:
        assert gtp_to_flat(flat_to_gtp(m, size), size) == m
    assert flat_to_gtp(0, 19) == "A1"
    assert flat_to_gtp(8, 19) == "J1"   # skips I
    assert flat_to_gtp(360, 19) == "T19"
    assert flat_to_gtp(60, 19) == "D4"
    assert flat_to_gtp(19 * 19, 19) == "pass"


def test_gnugo_gtp():
    g = GTPEngine(["gnugo", "--mode", "gtp", "--level", "10"])
    g.setup(19, 7.5)
    g.play("B", "D4")
    v = g.genmove("W")
    assert v and v.lower() != "resign"
    print("gnugo responded:", v)
    g.close()


if __name__ == "__main__":
    test_coord_roundtrip()
    print("PASS test_coord_roundtrip")
    test_gnugo_gtp()
    print("PASS test_gnugo_gtp")
    print("all eval tests passed")
