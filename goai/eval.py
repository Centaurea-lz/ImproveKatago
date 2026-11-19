"""Evaluation framework: play our net against GTP engines (e.g. GNU Go).

Uses the Go Text Protocol to talk to a reference engine, runs matches with
our MCTS net, and estimates Elo from win rate.
"""
from __future__ import annotations

import argparse
import math
import subprocess

import numpy as np
import torch

from .board import BLACK, WHITE, GoBoard
from .mcts import MCTS
from .net import PolicyValueNet

# GTP column letters: A-T without I (19 columns)
GTP_COLS = "ABCDEFGHJKLMNOPQRST"


def flat_to_gtp(move: int, size: int) -> str:
    if move == size * size:
        return "pass"
    r, c = divmod(move, size)
    return f"{GTP_COLS[c]}{r + 1}"


def gtp_to_flat(vertex: str, size: int) -> int:
    v = vertex.strip().upper()
    if v in ("PASS", ""):
        return size * size
    col, row = v[0], int(v[1:])
    c = GTP_COLS.index(col)
    r = row - 1  # GTP rows are 1-based from the bottom
    return r * size + c


class GTPEngine:
    def __init__(self, cmd):
        # stdbuf -o0 forces unbuffered stdout so responses arrive promptly.
        # NOTE: do not read an "id" response here — some engines (GNU Go)
        # send nothing until they receive a command.
        full_cmd = ["stdbuf", "-o0", "-e0"] + list(cmd)
        self.proc = subprocess.Popen(full_cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True,
                                     bufsize=1)

    def _read(self):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                break
            line = line.rstrip("\n")
            if line == "":
                break
            lines.append(line)
        return lines

    def send(self, command):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        return self._read()

    def genmove(self, color: str) -> str:
        out = self.send(f"genmove {color}")
        for line in out:
            if line.startswith("="):
                return line[1:].strip()
        return "resign"

    def setup(self, size, komi):
        self.send(f"boardsize {size}")
        self.send(f"komi {komi}")
        self.send("clear_board")

    def play(self, color: str, vertex: str):
        self.send(f"play {color} {vertex}")

    def close(self):
        self.send("quit")
        self.proc.terminate()


class OurBot:
    def __init__(self, net, device="cuda", sims=100):
        self.net = net
        self.device = device
        self.mcts = MCTS(net, device=device, num_simulations=sims, batch_size=48)

    def choose_move(self, board: GoBoard) -> int:
        return self.mcts.best_move(board, lcb_factor=2.0)

    def advance(self, move: int):
        self.mcts.advance(move)


def play_match(our_bot, gtp, our_color, size=19, komi=7.5, max_moves=400):
    """Play one match; return winner color (BLACK/WHITE)."""
    board = GoBoard(size=size, komi=komi, superko=False)
    gtp.setup(size, komi)
    move_num = 0
    while not board.done and move_num < max_moves:
        color = board.current
        color_letter = "B" if color == BLACK else "W"
        if color == our_color:
            move = our_bot.choose_move(board)
            gtp.play(color_letter, flat_to_gtp(move, size))
            board.play(move)
        else:
            vertex = gtp.genmove(color_letter)
            if "resign" in vertex.lower():
                board.winner = our_color
                return our_color
            move = gtp_to_flat(vertex, size)
            # The opponent is a trusted engine; its move is legal under ITS
            # rules (e.g. KataGo uses positional superko). Our board uses
            # simple ko, whose heuristic can rarely flag a legal move as
            # illegal, so apply the move unchecked (the capture logic is the
            # same either way).
            board.play_unchecked(move)
        our_bot.advance(move)
        move_num += 1
    if board.done:
        return board.winner
    return board.current  # move cap / fallback


def elo_from_winrate(winrate):
    winrate = max(0.001, min(0.999, winrate))
    return -400.0 * math.log10(1.0 / winrate - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt_sl6/sl_final.pt")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--gnugo-level", type=int, default=10)
    ap.add_argument("--engine-cmd", default=None,
                    help="opponent GTP command, space-separated (overrides gnugo)")
    ap.add_argument("--label", default=None, help="opponent label for reporting")
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--fp16", action="store_true", help="use FP16 inference")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(args.board_size, channels=args.channels,
                         blocks=args.blocks).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    net = net.eval()
    if args.fp16:
        net = net.half()
    bot = OurBot(net, device=device, sims=args.sims)

    if args.engine_cmd:
        engine = GTPEngine(args.engine_cmd.split())
        label = args.label or "opponent"
    else:
        engine = GTPEngine(["gnugo", "--mode", "gtp",
                            "--level", str(args.gnugo_level)])
        label = args.label or f"GNU Go level {args.gnugo_level}"

    wins = 0
    for g in range(args.games):
        our_color = BLACK if g % 2 == 0 else WHITE  # alternate colors
        winner = play_match(bot, engine, our_color, size=args.board_size,
                            komi=args.komi)
        won = winner == our_color
        wins += won
        print(f"game {g}: our_color={our_color} winner={winner} "
              f"{'WIN' if won else 'LOSS'}", flush=True)

    engine.close()
    wr = wins / args.games
    print(f"result: {wins}/{args.games} win_rate={wr:.3f} "
          f"elo_diff={elo_from_winrate(wr):+.1f} (vs {label})", flush=True)


if __name__ == "__main__":
    main()
