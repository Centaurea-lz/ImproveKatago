"""Generate a few self-play games with the SL net, save SGFs, and time them."""
import os
import time

import torch

from goai.net import PolicyValueNet
from goai.selfplay import play_game

net = PolicyValueNet(19, channels=96, blocks=8)
net.load_state_dict(torch.load("ckpt_sl6/sl_final.pt", map_location="cuda"))
net = net.cuda().eval()

os.makedirs("demo_sgf", exist_ok=True)
for gid in range(4):
    t = time.time()
    samples, board = play_game(net, device="cuda", board_size=19,
                               num_simulations=100, batch_size=64,
                               sgf_dir="demo_sgf", game_id=gid)
    dt = time.time() - t
    print(f"game {gid}: moves={len(samples)} done={board.done} "
          f"winner={board.winner} time={dt:.1f}s", flush=True)
print("demo games written to demo_sgf/", flush=True)
