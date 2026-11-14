"""Self-play game generation with MCTS (mini-AlphaZero)."""
from __future__ import annotations

from .board import GoBoard
from .features import encode
from .mcts import MCTS, sample_move


def play_game(net, device="cuda", board_size=9, num_simulations=200,
              batch_size=64, temperature=1.0, temp_moves=20,
              max_moves=600, dirichlet=True):
    """Play one self-play game; return (samples, final_board).

    Each sample is (features, policy_probs, outcome) with features/policy from
    the mover's perspective and outcome +1/-1 from the mover's perspective.
    """
    board = GoBoard(size=board_size)
    mcts = MCTS(net, device=device, num_simulations=num_simulations,
                batch_size=batch_size)
    traj = []  # (features, policy, player_color)
    move_num = 0
    while not board.done and move_num < max_moves:
        probs = mcts.search(board, dirichlet=dirichlet)
        traj.append((encode(board), probs, board.current))
        t = temperature if move_num < temp_moves else 0.0
        move = sample_move(probs, temperature=t, legal=board.legal_moves())
        board.play(move)
        move_num += 1
    samples = [(x, p, float(board.outcome_for(c))) for x, p, c in traj]
    return samples, board
