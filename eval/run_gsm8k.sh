#!/usr/bin/env bash
# Evaluate the three models on GSM8K in parallel, one GPU each, then summarize.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/lian/miniconda3/envs/verl-fork/bin/python
mkdir -p logs
i=0
for m in base noDS_step300 DS_step300; do
  CUDA_VISIBLE_DEVICES=$i $PY eval_gsm8k.py --model $m > logs/gsm8k_$m.log 2>&1 &
  i=$((i+1))
done
wait
grep -h "strict" logs/gsm8k_*.log
$PY eval_gsm8k.py --summarize | tee results/gsm8k/summary.txt
