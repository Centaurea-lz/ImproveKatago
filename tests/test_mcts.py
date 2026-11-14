import numpy as np
import torch

from goai.board import GoBoard
from goai.mcts import MCTS, sample_move
from goai.net import PolicyValueNet


def _make_mcts():
    net = PolicyValueNet(board_size=9, channels=16, blocks=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcts = MCTS(net.to(device), device=device, num_simulations=64, batch_size=16)
    return mcts


def test_search_returns_valid_policy():
    mcts = _make_mcts()
    b = GoBoard(size=9)
    probs, _ = mcts.search(b)
    legal = b.legal_moves()
    assert probs.shape == (82,)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert (probs >= 0).all()
    assert (probs[~legal] == 0).all()        # no mass on illegal moves
    assert (probs[legal].sum() > 0.99)       # mass concentrated on legal moves


def test_search_after_some_moves():
    mcts = _make_mcts()
    b = GoBoard(size=9)
    b.play(40)   # center-ish
    b.play(41)
    probs, _ = mcts.search(b)
    legal = b.legal_moves()
    assert abs(probs.sum() - 1.0) < 1e-6
    assert (probs[~legal] == 0).all()


def test_sample_move():
    probs = np.zeros(82)
    probs[0] = 0.9
    probs[1] = 0.1
    legal = np.zeros(82, dtype=bool)
    legal[0] = legal[1] = True
    for t in (0.0, 0.5, 1.0):
        m = sample_move(probs, temperature=t, legal=legal)
        assert m in (0, 1)


if __name__ == "__main__":
    test_search_returns_valid_policy()
    print("PASS test_search_returns_valid_policy")
    test_search_after_some_moves()
    print("PASS test_search_after_some_moves")
    test_sample_move()
    print("PASS test_sample_move")
    print("all mcts tests passed")
