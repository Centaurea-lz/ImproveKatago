#!/bin/bash
# Run the distilled 40b256-v2 net against the KataGo low-visit ladder,
# sequentially, to avoid GPU OOM with the RL still running.
cd /root/go-ai
MODEL=/root/katago/kata1-tf3-b11c768-s12382M-d6533M.bin.gz
CKPT=ckpt_distill_b40_v2/sl_final.pt
PY=/usr/local/miniconda3/envs/py312/bin/python

for N in 1 5 10 50; do
  echo "=== ladder start: KataGo ${N}visit $(date) ==="
  PYTHONPATH=. $PY -m goai.eval \
    --model $CKPT --channels 256 --blocks 40 --sims 400 --games 10 --board-size 19 \
    --engine-cmd "/root/katago/katago.sh gtp -model $MODEL -config /root/katago/gtp_${N}visit.cfg" \
    --label "KataGo ${N}visit" \
    > eval_b40_v2_katago_${N}visit.log 2>&1
  echo "=== ladder done: KataGo ${N}visit exit=$? $(date) ==="
done
echo "LADDER COMPLETE $(date)"
