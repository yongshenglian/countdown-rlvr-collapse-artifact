#!/usr/bin/env bash
# pass@k sampling jobs (model:bench pairs) spread over the GPUs, then pass@k summary + RESULTS.md.
#   JOBS="base:math500 base:gsm8k DS_step300:math500" ./run_passk_jobs.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/lian/miniconda3/envs/verl-fork/bin/python
JOBS=${JOBS:?"model:bench pairs (bench = math500 | gsm8k)"}
NGPU=${NGPU:-4}
export HF_DATASETS_OFFLINE=1
LOG=logs/passk; mkdir -p $LOG
Q=$(mktemp); for j in $JOBS; do echo $j >> $Q; done
worker() {
  local gpu=$1 job
  while true; do
    job=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"); [ -z "$job" ] && break
    local m=${job%%:*} b=${job##*:}
    echo "[$(date '+%F %T')] GPU $gpu start pass@k $b $m"
    CUDA_VISIBLE_DEVICES=$gpu $PY eval_passk.py --model $m --bench $b > $LOG/${b}_$m.log 2>&1 || echo "FAILED pass@k $b $m"
    echo "[$(date '+%F %T')] GPU $gpu done  pass@k $b $m"
  done
}
for g in $(seq 0 $((NGPU-1))); do worker $g & done
wait; rm -f $Q $Q.lock
$PY eval_passk.py --summarize > $LOG/summarize.log 2>&1 || echo "FAILED pass@k summarize"
$PY make_results_md.py >> logs/registry_update.log 2>&1 && echo "RESULTS.md updated"
echo "[$(date '+%F %T')] pass@k jobs finished"
