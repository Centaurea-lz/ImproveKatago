"""KataGo teacher: evaluates positions during student self-play.

Used for the teacher-in-the-loop RL (teacher-anchored value + disagreement).
The teacher tracks the game via `play` commands and, at each position, we ask
`kata-genmove_analyze` for the side-to-move winrate and the best move, then
`undo` the move to leave the teacher's board unchanged.
"""
from __future__ import annotations

from .eval import GTPEngine, flat_to_gtp, gtp_to_flat


class KataGoTeacher:
    def __init__(self, cmd, board_size=19, komi=7.5):
        self.size = board_size
        self.gtp = GTPEngine(cmd)
        self.gtp.setup(board_size, komi)

    def play(self, color, move):
        """Update the teacher's board after `color` plays `move`."""
        self.gtp.play("B" if color == 1 else "W", flat_to_gtp(move, self.size))

    def clear(self):
        """Reset the teacher's board (start of a new game)."""
        self.gtp.send("clear_board")

    def evaluate(self, color):
        """Analyse the current position; return (winrate, best_move_flat).

        `color` is the side to move (1 black, 2 white). winrate is in [0, 1].
        best_move_flat is the flat move index (size*size == pass), or None on
        failure.
        """
        letter = "B" if color == 1 else "W"
        out = self.gtp.send(f"kata-genmove_analyze {letter}")
        wr = None
        best = None
        for line in out:
            s = line.strip()
            if not s.startswith("info"):
                continue
            toks = s.split()
            for i, t in enumerate(toks):
                if t == "move" and i + 1 < len(toks) and best is None:
                    best = toks[i + 1]  # first "move" is the top move
                elif t == "winrate" and i + 1 < len(toks) and wr is None:
                    try:
                        wr = float(toks[i + 1])
                    except ValueError:
                        wr = None
            break  # only the root info line matters
        self.gtp.send("undo")  # leave the board unchanged

        if best is None or best.lower() in ("", "pass", "resign"):
            best_flat = self.size * self.size
        else:
            best_flat = gtp_to_flat(best, self.size)
        return wr, best_flat

    def close(self):
        self.gtp.close()
