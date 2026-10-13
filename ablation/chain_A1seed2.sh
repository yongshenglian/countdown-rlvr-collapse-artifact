#!/usr/bin/env bash
# Option A1, training seed 2: train, then evaluate (step 300: all benchmarks; step 520: GSM8K + MATH) and rebuild
# RESULTS.md / results.html. Run after the GPUs work (e.g. after a reboot); starts the ray head if needed.
set -uo pipefail
B=/home/lian/miniconda3/envs/verl-fork/bin
R=$B/ray; A="--address=http://localhost:8265"
AB=/home/lian/AI_PPO_2026/ablation; EV=/home/lian/AI_PPO_2026/eval
if ! nvidia-smi > /dev/null 2>&1; then echo "GPUs not usable (nvidia-smi fails); reboot first"; exit 1; fi
if ! $R status > /dev/null 2>&1; then
  echo "[$(date '+%F %T')] starting ray head (verl-fork env)"
  PATH=$B:$PATH $R start --head --num-gpus=4 --dashboard-host=127.0.0.1 --dashboard-port=8265 --include-dashboard=true
  sleep 10
fi
wait_job() {
  local st
  while true; do st=$($R job status "$1" $A 2>&1 | grep -o -i -E "succeeded|failed|stopped" | head -1 | tr a-z A-Z); [ -n "$st" ] && break; sleep 300; done
  echo "[$(date '+%F %T')] $1 finished: $st"; $R job logs "$1" $A > $AB/logs/$1.log 2>&1
}
echo "[$(date '+%F %T')] submitting option A1 seed 2"
LOG=$AB/logs/submit_rungA1.sh_seed2.log
SEED=2 bash $AB/rungA1.sh > $LOG 2>&1
S=$(grep -o "rung[0-9A-Z][^ ]*-[0-9]\{8\}-[0-9]\{6\}" $LOG | head -1); echo "submission: $S"
[ -z "$S" ] && { echo "submission failed, see $LOG"; exit 1; }
wait_job "$S"
/home/lian/miniconda3/bin/python -c "
import wandb,pandas as pd; api=wandb.Api()
r=sorted([x for x in api.runs('yongshenglian-university-of-louisville/DAPO_ablation') if x.name=='rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2'],key=lambda x:x.created_at)[-1]
pd.DataFrame(list(r.scan_history())).to_csv('$AB/rungA1_seed2_history.csv',index=False); print('history', r.id)"
cd $EV && ./merge_checkpoints.sh 2>&1 | grep -E "converted|rror"
J=""; for b in mmlupro bbh math gsm8k; do J="$J rungA1s2_step300:$b"; done; J="$J rungA1s2_step520:math rungA1s2_step520:gsm8k"
JOBS="$J" ./run_jobs.sh
/home/lian/miniconda3/bin/python make_results_html.py
echo "[$(date '+%F %T')] chain finished"
