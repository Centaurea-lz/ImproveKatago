"""Benchmark inference speed: FP32 vs FP16 (and TensorRT if available).

Measures both raw forward-pass latency and full MCTS search latency on the
40b256 net — the latter is what actually matters for self-play/eval throughput.
"""
import argparse
import time

import numpy as np
import torch

from goai.board import GoBoard
from goai.features import encode
from goai.mcts import MCTS
from goai.net import PolicyValueNet


def bench_forward(net, x, n=50):
    with torch.no_grad():
        for _ in range(5):
            net(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            net(x)
        torch.cuda.synchronize()
    return (time.time() - t) / n


def bench_mcts(net, board, sims=100, batch=48, n=8):
    m = MCTS(net, device="cuda", num_simulations=sims, batch_size=batch)
    t = time.time()
    for _ in range(n):
        m.search(board, dirichlet=False)
    return (time.time() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt_distill_b40_v2/sl_final.pt")
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--board-size", type=int, default=19)
    args = ap.parse_args()

    net = PolicyValueNet(args.board_size, channels=args.channels,
                         blocks=args.blocks)
    net.load_state_dict(torch.load(args.model, map_location="cuda"), strict=False)
    net = net.cuda().eval()

    board = GoBoard(size=args.board_size, superko=False)
    x = torch.from_numpy(np.stack([encode(board)] * 48)).float().cuda()

    t_fwd32 = bench_forward(net, x)
    t_mcts32 = bench_mcts(net, board, sims=args.sims)

    net16 = net.half()
    x16 = x.half()
    t_fwd16 = bench_forward(net16, x16)
    t_mcts16 = bench_mcts(net16, board, sims=args.sims)

    print(f"FP32: forward {t_fwd32 * 1000:.2f}ms, mcts {t_mcts32:.3f}s/search")
    print(f"FP16: forward {t_fwd16 * 1000:.2f}ms, mcts {t_mcts16:.3f}s/search")
    print(f"speedup: forward {t_fwd32 / t_fwd16:.2f}x, "
          f"mcts {t_mcts32 / t_mcts16:.2f}x")


if __name__ == "__main__":
    main()
