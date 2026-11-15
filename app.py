"""Flask web UI for human-vs-AI Go.

Endpoints:
  GET  /            -> the board page
  POST /new_game    -> reset the game (params: size, human_color, sims)
  POST /play        -> human plays a move; returns AI's reply
  POST /pass        -> human passes
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from goai.board import BLACK, WHITE, GoBoard
from goai.mcts import MCTS
from goai.net import PolicyValueNet

app = Flask(__name__)

G = {
    "board": None,
    "net": None,
    "mcts": None,
    "human_color": BLACK,
    "ai_color": WHITE,
    "sims": 100,
    "message": "",
}


def _board_list():
    b = G["board"]
    return b.board.reshape(b.size, b.size)[::-1].tolist()  # flip to top-origin


def _state():
    b = G["board"]
    return {
        "board": _board_list(),
        "current": int(b.current),
        "done": bool(b.done),
        "winner": int(b.winner),
        "human_color": G["human_color"],
        "ai_color": G["ai_color"],
        "message": G["message"],
    }


def _ai_move():
    b = G["board"]
    if b.done:
        return None
    probs, value = G["mcts"].search(b, dirichlet=False)
    move = int(np.argmax(probs))
    b.play(move)
    return move


def _end_message():
    b = G["board"]
    if b.done:
        if b.winner == G["human_color"]:
            G["message"] = "You win!"
        elif b.winner == G["ai_color"]:
            G["message"] = "AI wins!"
        else:
            G["message"] = "Draw."
    else:
        G["message"] = ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/new_game", methods=["POST"])
def new_game():
    data = request.get_json(force=True)
    size = int(data.get("size", 19))
    G["human_color"] = int(data.get("human_color", BLACK))
    G["ai_color"] = 3 - G["human_color"]
    G["sims"] = int(data.get("sims", 100))
    G["mcts"].num_simulations = G["sims"]
    G["board"] = GoBoard(size=size, komi=7.5, superko=False)
    G["message"] = ""
    ai_move = None
    if G["ai_color"] == BLACK:  # AI plays first
        ai_move = _ai_move()
        _end_message()
    return jsonify({"state": _state(), "ai_move": ai_move})


@app.route("/play", methods=["POST"])
def play():
    data = request.get_json(force=True)
    size = G["board"].size
    row = int(data["row"])
    move = (size - 1 - row) * size + int(data["col"])  # row is top-origin
    b = G["board"]
    if b.done or b.current != G["human_color"]:
        return jsonify({"error": "not your turn"}), 400
    if not b.legal_moves()[move]:
        return jsonify({"error": "illegal move"}), 400
    b.play(move)
    _end_message()
    ai_move = None
    if not b.done and b.current == G["ai_color"]:
        ai_move = _ai_move()
        _end_message()
    return jsonify({"state": _state(), "ai_move": ai_move})


@app.route("/pass", methods=["POST"])
def pas():
    b = G["board"]
    if b.done or b.current != G["human_color"]:
        return jsonify({"error": "not your turn"}), 400
    b.play(b.pass_move)
    _end_message()
    ai_move = None
    if not b.done and b.current == G["ai_color"]:
        ai_move = _ai_move()
        _end_message()
    return jsonify({"state": _state(), "ai_move": ai_move})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt_sl6/sl_final.pt")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(19, channels=args.channels, blocks=args.blocks)
    net.load_state_dict(torch.load(args.model, map_location=device))
    G["net"] = net.eval().to(device)
    G["mcts"] = MCTS(net, device=device, num_simulations=args.sims,
                     batch_size=48)
    G["board"] = GoBoard(size=19, komi=7.5, superko=False)
    G["sims"] = args.sims
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
