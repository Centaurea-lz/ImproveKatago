"""Supervised pre-training on KGS samples (policy CE + value MSE).

Learning-rate schedule: warm-up then cosine annealing (per project proposal).
Memory note: features are loaded as uint8 and cast to float32 per batch.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .net import PolicyValueNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="ckpt_sl")
    ap.add_argument("--board-size", type=int, default=19)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = np.load(args.data)
    X, M, V, O, S = d["X"], d["move"], d["value"], d["own"], d["score"]
    n = X.shape[0]
    perm = np.random.permutation(n)
    split = int(0.95 * n)
    tr_idx, va_idx = perm[:split], perm[split:]
    print(f"samples={n} train={split} val={n - split}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(args.board_size, channels=args.channels,
                         blocks=args.blocks).to(device)

    Xt = torch.from_numpy(X)  # uint8; cast to float32 per batch
    Mt = torch.from_numpy(M)
    Vt = torch.from_numpy(V)
    Ot = torch.from_numpy(O.astype(np.float32))
    St = torch.from_numpy(S)

    def make_loader(idx):
        ds = TensorDataset(Xt[idx], Mt[idx], Vt[idx], Ot[idx], St[idx])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                          pin_memory=(device.type == "cuda"))

    train_loader = make_loader(tr_idx)
    val_loader = make_loader(va_idx)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.out, exist_ok=True)
    for epoch in range(args.epochs):
        net.train()
        tot_l = tot_p = tot_v = 0.0
        for Xb, Mb, Vb, Ob, Sb in train_loader:
            Xb = Xb.float().to(device)
            Mb, Vb, Ob, Sb = Mb.to(device), Vb.to(device), Ob.to(device), Sb.to(device)
            logits, v_logits, own, score = net(Xb)
            loss_p = F.cross_entropy(logits, Mb)
            # value: cross-entropy on win/loss distribution (soft blended target)
            p_win = Vb  # P(win) in [0, 1]
            p_dist = torch.stack([p_win, 1.0 - p_win], dim=1)
            loss_v = -(p_dist * F.log_softmax(v_logits, dim=1)).sum(1).mean()
            loss_own = F.mse_loss(own, Ob.unsqueeze(1))
            loss_score = F.mse_loss(score.view(-1), Sb)
            loss = loss_p + loss_v + loss_own + 0.1 * loss_score
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot_l += loss.item()
            tot_p += loss_p.item()
            tot_v += loss_v.item()

        net.eval()
        acc = tot = 0
        with torch.no_grad():
            for Xb, Mb, Vb, Ob, Sb in val_loader:
                Xb = Xb.float().to(device)
                Mb = Mb.to(device)
                logits, _v, _own, _score = net(Xb)
                acc += (logits.argmax(1) == Mb).sum().item()
                tot += Mb.size(0)
        nb = len(train_loader)
        print(f"epoch {epoch} loss {tot_l / nb:.4f} policy {tot_p / nb:.4f} "
              f"value {tot_v / nb:.4f} top1_acc {acc / tot:.4f}", flush=True)
        torch.save(net.state_dict(), os.path.join(args.out, f"sl_epoch{epoch}.pt"))

    torch.save(net.state_dict(), os.path.join(args.out, "sl_final.pt"))
    print("saved to", args.out, flush=True)


if __name__ == "__main__":
    main()
