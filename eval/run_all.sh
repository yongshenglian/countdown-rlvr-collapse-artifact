#!/usr/bin/env bash
# Run the benchmarks for every model (one GPU per model, benchmarks in sequence), then score.
#   ./run_all.sh                                   GSM8K, MATH, BBH, MMLU-Pro at 2048 tokens -> results/
#   MAX_TOKENS=8192 EVAL_RESULTS=results_8k ./run_all.sh
#   BENCHES="bbh" MODELS_TO_RUN="DS_step300" ./run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/lian/miniconda3/envs/verl-fork/bin/python
BENCHES=${BENCHES:-"gsm8k math bbh mmlupro"}
MODELS_TO_RUN=${MODELS_TO_RUN:-"base noDS_step300 DS_step300"}
MAX_TOKENS=${MAX_TOKENS:-2048}
export EVAL_RESULTS=${EVAL_RESULTS:-results}
export HF_DATASETS_OFFLINE=1
LOG=logs/$EVAL_RESULTS
mkdir -p $LOG
i=0
for m in $MODELS_TO_RUN; do
  ( for b in $BENCHES; do
      echo "[$(date +%T)] start $b $m (max_tokens=$MAX_TOKENS)"
      CUDA_VISIBLE_DEVICES=$i $PY eval_$b.py --model $m --max_tokens $MAX_TOKENS > $LOG/${b}_$m.log 2>&1 \
        || echo "FAILED $b $m (see $LOG/${b}_$m.log)"
      echo "[$(date +%T)] done  $b $m"
    done ) &
  i=$((i+1))
done
wait
for b in $BENCHES; do
  echo "===== $b ====="
  $PY eval_$b.py --summarize > $LOG/${b}_summarize.log 2>&1 || echo "FAILED summarize $b (see $LOG/${b}_summarize.log)"
  [ "$b" = gsm8k ] && cp $LOG/${b}_summarize.log $EVAL_RESULTS/gsm8k/summary.txt
  head -30 $EVAL_RESULTS/$b/summary.txt
done
