#!/usr/bin/env bash
# Evaluate several models one after another (each uses all 4 GPUs via run_model.sh).
# Waits for any evaluation already running before starting.
#   QUEUE="modelA modelB" ./run_queue.sh
# After each model, runs_registry.* and RESULTS.md are regenerated.
set -uo pipefail
cd "$(dirname "$0")"
QUEUE=${QUEUE:?list of MODELS keys}
while pgrep -f "eval_(gsm8k|math|bbh|mmlupro).py --model" >/dev/null; do sleep 60; done
for m in $QUEUE; do
  echo "[$(date '+%F %T')] start $m"
  MODEL=$m ./run_model.sh > logs/${EVAL_RESULTS:-results_8k}/run_$m.out 2>&1
  echo "[$(date '+%F %T')] done  $m ($(grep -c FAILED logs/${EVAL_RESULTS:-results_8k}/run_$m.out) failures)"
  # add the finished model to RESULTS.md (settings from wandb, then all results)
  /home/lian/miniconda3/bin/python run_registry.py > logs/registry_update.log 2>&1 || echo "  registry update failed"
  /home/lian/miniconda3/envs/verl-fork/bin/python make_results_md.py >> logs/registry_update.log 2>&1 \
    && echo "  RESULTS.md updated" || echo "  RESULTS.md update failed"
done
echo "[$(date '+%F %T')] queue finished"
