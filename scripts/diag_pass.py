"""Diagnose the net's pass probability on opening positions."""
import numpy as np
import torch

from goai.board import GoBoard
from goai.features import encode
from goai.net import PolicyValueNet

net = PolicyValueNet(19, channels=96, blocks=8)
net.load_state_dict(torch.load("ckpt_sl6/sl_final.pt", map_location="cuda"))
net = net.cuda().eval()


def raw_policy_and_value(board):
    x = torch.from_numpy(encode(board)[None]).cuda()
    with torch.no_grad():
        logits, v = net(x)
    logits = logits[0].cpu().numpy()
    legal = board.legal_moves()
    l = logits.copy()
    l[~legal] = -1e30
    l = l - l.max()
    probs = np.exp(l)
    probs /= probs.sum()
    return probs, float(v.item())


b0 = GoBoard(size=19, superko=False)
p0, v0 = raw_policy_and_value(b0)
print(f"empty board (Black to move): pass_prob={p0[361]:.4f} value={v0:.3f}")

b1 = GoBoard(size=19, superko=False)
b1.play(318)  # B[oc]
p1, v1 = raw_policy_and_value(b1)
print(f"after B[oc] (White to move): pass_prob={p1[361]:.4f} value={v1:.3f}")
