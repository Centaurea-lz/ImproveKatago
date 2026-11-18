"""Self-play game generation with MCTS (mini-AlphaZero).

Optional teacher-in-the-loop (KataGo teacher):
  - teacher-anchored value: value target = (1-beta)*outcome + beta*(2*wr-1)
  - disagreement collection: move sequences where the student's top move
    differs from the teacher's, to be used as starts for future games.
"""
from __future__ import annotations

import os

import numpy as np

from .board import GoBoard
from .features import encode
from .mcts import MCTS, sample_move
from .sgf_data import moves_to_sgf


def play_game(net, device="cuda", board_size=19, num_simulations=100,
              batch_size=48, temperature=1.0, temp_moves=20,
              max_moves=300, dirichlet=True, no_pass_moves=200,
              resign_threshold=-0.9, resign_consecutive=5,
              sgf_dir=None, game_id=0, teacher=None, beta=0.3,
              start_moves=None):
    """Play one self-play game.

    Returns (samples, final_board, disagreement_starts).
      samples: list of (features, policy, value), value in [-1, 1].
      disagreement_starts: list of move sequences (lists of flat moves) at
        positions where the student's top move differs from the teacher's.
    """
    board = GoBoard(size=board_size, superko=False)
    mcts = MCTS(net, device=device, num_simulations=num_simulations,
                batch_size=batch_size, pass_penalty=5.0)

    moves = []
    if teacher is not None:
        teacher.clear()
    if start_moves:
        for m in start_moves:
            board.play(m)
            moves.append(int(m))
            if teacher is not None:
                teacher.play(3 - board.current, m)

    traj = []       # (features, policy, color, teacher_wr, teacher_move, student_move)
    disagreement_starts = []
    move_num = len(moves)
    bad_count = 0
    resigned = False

    while not board.done and move_num < max_moves:
        probs, value = mcts.search(board, dirichlet=dirichlet)
        student_move = int(np.argmax(probs))
        t_wr = None
        t_move = None
        if teacher is not None:
            t_wr, t_move = teacher.evaluate(board.current)
        traj.append((encode(board), probs, board.current, t_wr, t_move, student_move))
        if t_move is not None and t_move != student_move:
            disagreement_starts.append(list(moves))

        t = temperature if move_num < temp_moves else 0.0
        legal = board.legal_moves()
        if move_num < no_pass_moves:
            legal = legal.copy()
            legal[board.pass_move] = False
        move = sample_move(probs, temperature=t, legal=legal)
        moves.append(int(move))
        board.play(move)
        if teacher is not None:
            teacher.play(3 - board.current, move)
        move_num += 1

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
        winner = board.current
    else:
        board._end_game()
        winner = board.winner

    if sgf_dir is not None:
        os.makedirs(sgf_dir, exist_ok=True)
        sgf_text = moves_to_sgf(moves, board_size, board.komi, winner)
        with open(os.path.join(sgf_dir, f"game_{game_id:06d}.sgf"), "w") as fh:
            fh.write(sgf_text)

    samples = []
    for (x, p, color, t_wr, t_move, s_move) in traj:
        outcome = 1.0 if color == winner else -1.0
        if t_wr is not None:
            value = (1.0 - beta) * outcome + beta * (2.0 * t_wr - 1.0)
        else:
            value = outcome
        samples.append((x, p, value))
    return samples, board, disagreement_starts
