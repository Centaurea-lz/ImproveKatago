"""Regenerate demo games and verify there is no pass in the opening."""
import os
import re
import time

import torch

from goai.net import PolicyValueNet
from goai.selfplay import play_game

net = PolicyValueNet(19, channels=96, blocks=8)
net.load_state_dict(torch.load("ckpt_sl6/sl_final.pt", map_location="cuda"))
net = net.cuda().eval()

os.makedirs("demo_sgf", exist_ok=True)
all_ok = True
for gid in range(3):
    t = time.time()
    samples, board = play_game(net, device="cuda", board_size=19,
                               num_simulations=100, batch_size=64,
                               sgf_dir="demo_sgf", game_id=gid)
    sgf = open(f"demo_sgf/game_{gid:06d}.sgf").read()
    moves = re.findall(r"[BW]\[([a-s]{2}|)\]", sgf)
    first_pass = next((i for i, m in enumerate(moves) if m == ""), None)
    early = first_pass is not None and first_pass < 30
    all_ok = all_ok and not early
    print(f"game {gid}: total={len(samples)} first_pass_at={first_pass} "
          f"{'EARLY!' if early else 'ok'} time={time.time() - t:.0f}s", flush=True)
print("VERIFY:", "PASS" if all_ok else "FAIL", flush=True)
