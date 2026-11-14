# Single-machine Go AI Training System (mini-AlphaZero)

BSc project — *Design and Implementation of a Single-machine Go-playing AI
Training System Based on Deep Learning* (SWJTU-LEEDS).

Reference paper: D. J. Wu, *Accelerating Self-Play Learning in Go* (KataGo),
arXiv:1902.10565.

## Scope

9x9-primary Go AI with a dual-head residual network, trained via
supervised pre-training (KGS SGF) + self-play RL fine-tuning, searched with a
PUCT MCTS that batches leaf evaluations on GPU, accelerated with TensorRT,
with a Flask web UI and a standardized evaluation framework.

## Layout

```
goai/
  board.py      # Go rules: capture/suicide/ko/superko, pass, Tromp-Taylor scoring
  features.py   # board -> 17-plane AlphaZero-style input tensor
  net.py        # dual-head residual network (policy + value)
  (next) mcts.py, selfplay.py, train_sl.py, train_rl.py, eval.py
tests/
scripts/setup_server.sh
```

## Run the rule/feature tests locally (numpy only)

```
python -I -m pytest tests/   # or run each test file directly
```
