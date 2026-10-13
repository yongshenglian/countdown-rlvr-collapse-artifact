#!/usr/bin/env bash
# After rung 1: fetch its history, convert + evaluate it, then train second seeds:
#   rung 2 seed 2 (about 4-5 h), then the DS recipe seed 2 (rung 3; the original took about 75 h).
set -uo pipefail
R=/home/lian/miniconda3/envs/verl-fork/bin/ray; A="--address=http://localhost:8265"
AB=/home/lian/AI_PPO_2026/ablation; EV=/home/lian/AI_PPO_2026/eval
wait_job() {  # $1 = submission id
  local st
  while true; do st=$($R job status "$1" $A 2>&1 | grep -o -i -E "succeeded|failed|stopped" | head -1 | tr a-z A-Z); [ -n "$st" ] && break; sleep 300; done
  echo "[$(date '+%F %T')] $1 finished: $st"; $R job logs "$1" $A > $AB/logs/$1.log 2>&1
}
fetch_hist() {  # $1 = wandb run name, $2 = output csv
  /home/lian/miniconda3/bin/python -c "
import wandb,pandas as pd; api=wandb.Api()
r=sorted([x for x in api.runs('yongshenglian-university-of-louisville/DAPO_ablation') if x.name=='$1'],key=lambda x:x.created_at)[-1]
pd.DataFrame(list(r.scan_history())).to_csv('$2',index=False); print('history', r.id, '->', '$2')"
}
submit() {  # $1 = script, $2 = seed (empty = original seed)
  local log=$AB/logs/submit_$1${2:+_seed$2}.log
  SEED="${2:-}" bash $AB/$1 > $log 2>&1
  grep -o "rung[0-9][^ ]*-[0-9]\{8\}-[0-9]\{6\}" $log | head -1
}

wait_job rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096-20260929-151515
fetch_hist rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096 $AB/rung1_history.csv
cd $EV && ./merge_checkpoints.sh 2>&1 | grep -E "converted|rror"
J=""; for b in mmlupro bbh math gsm8k; do J="$J rung1_step300:$b"; done
for s in 100 520; do J="$J rung1_step$s:math rung1_step$s:gsm8k"; done
JOBS="$J" ./run_jobs.sh
/home/lian/miniconda3/bin/python make_results_html.py

echo "[$(date '+%F %T')] submitting rung 2 seed 2"
S=$(submit rung2_clip0.28_noDS.sh 2); echo "submission: $S"; wait_job "$S"
fetch_hist rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2 $AB/rung2_seed2_history.csv

echo "[$(date '+%F %T')] submitting DS recipe (rung 3) seed 2"
S=$(submit rung3_DS_gen192.sh 2); echo "submission: $S"; wait_job "$S"
fetch_hist rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed2 $AB/rung3_seed2_history.csv
echo "[$(date '+%F %T')] chain finished"
