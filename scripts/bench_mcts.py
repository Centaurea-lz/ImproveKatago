"""Benchmark MCTS speed and generate a few demo SGF games."""
import os
import time

import torch

from goai.board import GoBoard
from goai.mcts import MCTS
from goai.net import PolicyValueNet
from goai.selfplay import play_game

net = PolicyValueNet(19, channels=96, blocks=8)
net.load_state_dict(torch.load("ckpt_sl6/sl_final.pt", map_location="cuda"))
net = net.cuda().eval()

m = MCTS(net, device="cuda", num_simulations=100, batch_size=64)
b = GoBoard(size=19, superko=False)
t = time.time()
for _ in range(10):
    m.search(b, dirichlet=True)
print(f"avg search (100 sims) = {(time.time() - t) / 10:.3f}s", flush=True)

os.makedirs("demo_sgf", exist_ok=True)
for gid in range(3):
    t = time.time()
    samples, board = play_game(net, device="cuda", board_size=19,
                               num_simulations=100, batch_size=64,
                               sgf_dir="demo_sgf", game_id=gid)
    print(f"game {gid}: moves={len(samples)} done={board.done} "
          f"winner={board.winner} time={time.time() - t:.1f}s", flush=True)
print("done", flush=True)
