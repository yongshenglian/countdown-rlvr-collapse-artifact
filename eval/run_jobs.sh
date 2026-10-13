#!/usr/bin/env bash
# Run individual (model, benchmark) jobs on all GPUs: each GPU takes the next job from a shared queue.
# Then scores the benchmarks touched and regenerates runs_registry.* and RESULTS.md.
#   JOBS="noDS_step100:gsm8k noDS_step100:math GRPO_n2_kl1e-3_step250:mmlupro" ./run_jobs.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/lian/miniconda3/envs/verl-fork/bin/python
JOBS=${JOBS:?"model:benchmark pairs"}
NGPU=${NGPU:-4}
MAX_TOKENS=${MAX_TOKENS:-8192}
export EVAL_RESULTS=${EVAL_RESULTS:-results_8k}
export HF_DATASETS_OFFLINE=1
LOG=logs/$EVAL_RESULTS; mkdir -p $LOG
Q=$(mktemp); for j in $JOBS; do echo $j >> $Q; done
worker() {
  local gpu=$1 job
  while true; do
    job=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    local m=${job%%:*} b=${job##*:}
    echo "[$(date '+%F %T')] GPU $gpu start $b $m"
    CUDA_VISIBLE_DEVICES=$gpu $PY eval_$b.py --model $m --max_tokens $MAX_TOKENS > $LOG/${b}_$m.log 2>&1 \
      || echo "FAILED $b $m (see $LOG/${b}_$m.log)"
    echo "[$(date '+%F %T')] GPU $gpu done  $b $m"
  done
}
for g in $(seq 0 $((NGPU-1))); do worker $g & done
wait; rm -f $Q $Q.lock
for b in $(for j in $JOBS; do echo ${j##*:}; done | sort -u); do
  $PY eval_$b.py --summarize > $LOG/${b}_summarize.log 2>&1 || echo "FAILED summarize $b"
  [ "$b" = gsm8k ] && cp $LOG/${b}_summarize.log $EVAL_RESULTS/gsm8k/summary.txt
done
/home/lian/miniconda3/bin/python run_registry.py > logs/registry_update.log 2>&1 || echo "registry update failed"
$PY make_results_md.py >> logs/registry_update.log 2>&1 && echo "RESULTS.md updated" || echo "RESULTS.md update failed"
echo "[$(date '+%F %T')] all jobs finished"
