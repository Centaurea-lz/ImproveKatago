"""Self-play RL fine-tuning loop (single-machine mini-AlphaZero)."""
from __future__ import annotations

import argparse
import collections
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .net import PolicyValueNet
from .selfplay import play_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None, help="initial weights (SL checkpoint)")
    ap.add_argument("--out", default="ckpt_rl")
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=20)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--steps-per-iter", type=int, default=200)
    ap.add_argument("--buffer-size", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(args.board_size, channels=args.channels,
                         blocks=args.blocks).to(device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=device))
        print("loaded init weights from", args.init)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer = collections.deque(maxlen=args.buffer_size)
    os.makedirs(args.out, exist_ok=True)

    for it in range(args.iters):
        samples = []
        for g in range(args.games_per_iter):
            s, _ = play_game(net, device=device, board_size=args.board_size,
                             num_simulations=args.sims)
            samples.extend(s)
        buffer.extend(samples)

        net.train()
        losses = []
        for _ in range(args.steps_per_iter):
            batch = random.sample(buffer, min(args.batch_size, len(buffer)))
            X = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
            P = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
            Z = torch.tensor([b[2] for b in batch], dtype=torch.float32,
                             device=device)
            logits, v = net(X)
            loss_p = -(P * F.log_softmax(logits, dim=1)).sum(1).mean()
            loss_v = F.mse_loss(v.view(-1), Z)
            loss = loss_p + loss_v
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"iter {it} new_samples {len(samples)} "
              f"avg_loss {np.mean(losses):.4f}", flush=True)
        if it % 5 == 0 or it == args.iters - 1:
            torch.save(net.state_dict(), os.path.join(args.out, f"rl_iter{it}.pt"))

    torch.save(net.state_dict(), os.path.join(args.out, "rl_final.pt"))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
