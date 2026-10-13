#!/usr/bin/env bash
# Collapse screen: 100-step runs of 4 recipes x seeds 3-6 (16 runs). Seeds 1-2 come from the existing full runs.
# Recipes: no DS (rung 2), DS (rung 3), A1 (correctness-only reward + overlong filtering), A1 + KL loss 0.01.
# Every collapse seen so far happened within ~25 steps, so 100 steps classify each run. Then summarise.
set -uo pipefail
B=/home/lian/miniconda3/envs/verl-fork/bin
R=$B/ray; A="--address=http://localhost:8265"
AB=/home/lian/AI_PPO_2026/ablation
if ! nvidia-smi > /dev/null 2>&1; then echo "GPUs not usable"; exit 1; fi
if ! $R status > /dev/null 2>&1; then PATH=$B:$PATH $R start --head --num-gpus=4 --dashboard-host=127.0.0.1 --dashboard-port=8265 --include-dashboard=true; sleep 10; fi
wait_job() {
  local st
  while true; do st=$($R job status "$1" $A 2>&1 | grep -o -i -E "succeeded|failed|stopped" | head -1 | tr a-z A-Z); [ -n "$st" ] && break; sleep 120; done
  echo "[$(date '+%F %T')] $1 finished: $st"; $R job logs "$1" $A > $AB/logs/$1.log 2>&1
}
export TOTAL_STEPS=100 SAVE_FREQ=100 NAME_SUFFIX=-s100
for SEED in 3 4 5 6; do
  for SCRIPT in rung2_clip0.28_noDS.sh rungA1.sh rungA1KL.sh rung3_DS_gen192.sh; do
    LOG=$AB/logs/submit_screen_${SCRIPT%.sh}_seed$SEED.log
    SEED=$SEED bash $AB/$SCRIPT > $LOG 2>&1
    S=$(grep -o "rung[0-9A-Z][^ ]*-[0-9]\{8\}-[0-9]\{6\}" $LOG | head -1)
    echo "[$(date '+%F %T')] seed $SEED $SCRIPT -> $S"
    [ -z "$S" ] && { echo "submission failed, see $LOG"; continue; }
    wait_job "$S"
  done
done
/home/lian/miniconda3/bin/python $AB/analysis_collapse_screen.py > $AB/logs/collapse_screen_summary.log 2>&1 && echo "summary written"
echo "[$(date '+%F %T')] chain finished"
