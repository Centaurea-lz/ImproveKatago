"""Replicate eval play_match vs KataGo 5visit; dump board state on illegal move."""
import torch
import numpy as np

from goai.board import BLACK, GoBoard
from goai.eval import GTPEngine, OurBot, flat_to_gtp, gtp_to_flat
from goai.net import PolicyValueNet

SIZE = 19
CKPT = "ckpt_distill_b40_v2/sl_final.pt"
KATAGO = ("/root/katago/katago.sh gtp "
          "-model /root/katago/kata1-tf3-b11c768-s12382M-d6533M.bin.gz "
          "-config /root/katago/gtp_5visit.cfg")


def main():
    net = PolicyValueNet(SIZE, channels=256, blocks=40).cuda().eval()
    net.load_state_dict(torch.load(CKPT, map_location="cuda"), strict=False)
    bot = OurBot(net, device="cuda", sims=400)
    gtp = GTPEngine(KATAGO.split())
    board = GoBoard(size=SIZE, komi=7.5, superko=False)
    gtp.setup(SIZE, 7.5)
    our_color = BLACK
    move_num = 0
    while not board.done and move_num < 400:
        color = board.current
        letter = "B" if color == BLACK else "W"
        if color == our_color:
            move = bot.choose_move(board)
            gtp.play(letter, flat_to_gtp(move, SIZE))
            board.play(move)
        else:
            vertex = gtp.genmove(letter)
            if "resign" in vertex.lower():
                print("opponent resigned")
                return
            move = gtp_to_flat(vertex, SIZE)
            try:
                board.play(move)
            except ValueError as e:
                print(f"ILLEGAL at move_num={move_num} color={letter} move={move} "
                      f"vertex={vertex}")
                print(f"occupied={board.board[move] != 0}  is_ko={move == board.ko_point}")
                print(f"ko_point={board.ko_point}")
                print("board grid (0 empty, 1 B, 2 W):")
                for r in range(SIZE):
                    print("".join(str(int(board.board[r * SIZE + c]))
                                  for c in range(SIZE)))
                print(f"current={board.current} done={board.done}")
                return
        bot.advance(move)
        move_num += 1
    print(f"completed without illegal move (move_num={move_num}, done={board.done})")


if __name__ == "__main__":
    main()
