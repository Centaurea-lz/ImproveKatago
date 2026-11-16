"""Self-play game generation with MCTS (mini-AlphaZero).

Games end by two passes or by resignation (so they do not run for hundreds of
moves), and each game can be saved as an SGF file for review.
"""
from __future__ import annotations

import os

from .board import GoBoard
from .features import encode
from .mcts import MCTS, sample_move
from .sgf_data import moves_to_sgf


def play_game(net, device="cuda", board_size=19, num_simulations=100,
              batch_size=64, temperature=1.0, temp_moves=20,
              max_moves=300, dirichlet=True, no_pass_moves=200,
              resign_threshold=-0.9, resign_consecutive=5,
              sgf_dir=None, game_id=0):
    """Play one self-play game; return (samples, final_board).

    Each sample is (features, policy_probs, outcome) with outcome +1/-1 from
    the mover's perspective.
    """
    board = GoBoard(size=board_size, superko=False)
    mcts = MCTS(net, device=device, num_simulations=num_simulations,
                batch_size=batch_size)
    traj = []   # (features, policy, player_color)
    moves = []
    move_num = 0
    bad_count = 0
    resigned = False

    while not board.done and move_num < max_moves:
        probs, value = mcts.search(board, dirichlet=dirichlet)  # value: mover's perspective
        traj.append((encode(board), probs, board.current))
        t = temperature if move_num < temp_moves else 0.0
        legal = board.legal_moves()
        if move_num < no_pass_moves:
            legal = legal.copy()
            legal[board.pass_move] = False  # never pass in the opening
        move = sample_move(probs, temperature=t, legal=legal)
        moves.append(int(move))
        board.play(move)
        move_num += 1

        # resignation: if the player who just moved was almost certainly losing
        if value < resign_threshold:
            bad_count += 1
        else:
            bad_count = 0
        if bad_count >= resign_consecutive:
            resigned = True
            break

    if board.done:
        winner = board.winner
    elif resigned:
        winner = board.current  # opponent of the resigning side wins
    else:
        board._end_game()       # hit move cap: decide by Tromp-Taylor score
        winner = board.winner

    if sgf_dir is not None:
        os.makedirs(sgf_dir, exist_ok=True)
        sgf_text = moves_to_sgf(moves, board_size, board.komi, winner)
        with open(os.path.join(sgf_dir, f"game_{game_id:06d}.sgf"), "w") as fh:
            fh.write(sgf_text)

    if winner == 0:  # draw (essentially impossible with 7.5 komi)
        samples = [(x, p, 0.0) for x, p, _c in traj]
    else:
        samples = [(x, p, (1.0 if c == winner else -1.0)) for x, p, c in traj]
    return samples, board
