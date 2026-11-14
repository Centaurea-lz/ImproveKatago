"""Quick sanity check: what does the SL net prefer on an empty board?"""
import numpy as np
import torch

from goai.board import GoBoard
from goai.mcts import MCTS
from goai.net import PolicyValueNet

net = PolicyValueNet(19, channels=96, blocks=8)
net.load_state_dict(torch.load("ckpt_sl/sl_final.pt", map_location="cuda"))
net = net.cuda().eval()

m = MCTS(net, device="cuda", num_simulations=300, batch_size=64)
b = GoBoard(size=19)
probs = m.search(b, dirichlet=False)
top = np.argsort(probs)[::-1][:8]
print("top moves on empty 19x19 board (row,col : visit_prob):")
for i in top:
    mv = "pass" if int(i) == 361 else f"({int(i) // 19},{int(i) % 19})"
    print(f"  {mv:>12}  {probs[i]:.3f}")
