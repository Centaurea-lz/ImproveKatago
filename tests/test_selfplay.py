import torch

from goai.net import PolicyValueNet
from goai.selfplay import play_game


def test_play_game_runs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = PolicyValueNet(board_size=9, channels=16, blocks=2).to(device)
    samples, board, _ = play_game(net, device=device, board_size=9,
                                  num_simulations=40, batch_size=16, max_moves=80)
    assert len(samples) > 0
    x, p, z = samples[0]
    assert x.shape == (17, 9, 9)
    assert p.shape == (82,)
    assert z in (-1.0, 1.0, 0.0)
    print(f"game samples={len(samples)} done={board.done} winner={board.winner}")


if __name__ == "__main__":
    test_play_game_runs()
    print("PASS test_play_game_runs")
