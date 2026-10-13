#!/usr/bin/env bash
# Evaluate ONE model on all four benchmarks in parallel (one benchmark per GPU), then re-score
# every benchmark so the summaries include all models evaluated so far.
#   MODEL=PPO_step400 ./run_model.sh                          (8192 tokens -> results_8k/, the paper setting)
#   MODEL=PPO_step400 MAX_TOKENS=2048 EVAL_RESULTS=results ./run_model.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/lian/miniconda3/envs/verl-fork/bin/python
MODEL=${MODEL:?set MODEL to a key of MODELS in common.py}
MAX_TOKENS=${MAX_TOKENS:-8192}
export EVAL_RESULTS=${EVAL_RESULTS:-results_8k}
export HF_DATASETS_OFFLINE=1
LOG=logs/$EVAL_RESULTS
mkdir -p $LOG
i=0
for b in gsm8k math bbh mmlupro; do
  ( echo "[$(date +%T)] start $b $MODEL (max_tokens=$MAX_TOKENS, GPU $i)"
    CUDA_VISIBLE_DEVICES=$i $PY eval_$b.py --model $MODEL --max_tokens $MAX_TOKENS > $LOG/${b}_$MODEL.log 2>&1 \
      || echo "FAILED $b $MODEL (see $LOG/${b}_$MODEL.log)"
    echo "[$(date +%T)] done  $b $MODEL" ) &
  i=$((i+1))
done
wait
for b in gsm8k math bbh mmlupro; do
  $PY eval_$b.py --summarize > $LOG/${b}_summarize.log 2>&1 || echo "FAILED summarize $b"
  [ "$b" = gsm8k ] && cp $LOG/${b}_summarize.log $EVAL_RESULTS/gsm8k/summary.txt
done
echo "[$(date +%T)] all done"
