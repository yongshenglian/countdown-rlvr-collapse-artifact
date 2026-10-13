#!/usr/bin/env bash
# 1) wait for option B (running); 2) train options C, D, A1; 3) evaluate rung 2 seed 2, A-D and A1
# (step 300: all benchmarks; step 520: GSM8K + MATH); 4) pass@k for the key models; 5) DS recipe seed 2.
set -uo pipefail
R=/home/lian/miniconda3/envs/verl-fork/bin/ray; A="--address=http://localhost:8265"
AB=/home/lian/AI_PPO_2026/ablation; EV=/home/lian/AI_PPO_2026/eval
wait_job() {
  local st
  while true; do st=$($R job status "$1" $A 2>&1 | grep -o -i -E "succeeded|failed|stopped" | head -1 | tr a-z A-Z); [ -n "$st" ] && break; sleep 300; done
  echo "[$(date '+%F %T')] $1 finished: $st"; $R job logs "$1" $A > $AB/logs/$1.log 2>&1
}
fetch_hist() {
  /home/lian/miniconda3/bin/python -c "
import wandb,pandas as pd; api=wandb.Api()
r=sorted([x for x in api.runs('yongshenglian-university-of-louisville/DAPO_ablation') if x.name=='$1'],key=lambda x:x.created_at)[-1]
pd.DataFrame(list(r.scan_history())).to_csv('$2',index=False); print('history', r.id, '->', '$2')"
}
submit() {  # $1 = script, $2 = seed (empty = original)
  local log=$AB/logs/submit_$1${2:+_seed$2}.log
  SEED="${2:-}" bash $AB/$1 > $log 2>&1
  grep -o "rung[0-9A-Z][^ ]*-[0-9]\{8\}-[0-9]\{6\}" $log | head -1
}
wait_job rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096-20261001-200756
fetch_hist rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096 $AB/rungB_history.csv
for X in C D A1; do
  echo "[$(date '+%F %T')] submitting option $X"
  S=$(submit rung$X.sh); echo "submission: $S"; wait_job "$S"
  N=$(grep -m1 "^EXP_NAME=" $AB/rung$X.sh | cut -d= -f2); fetch_hist "$N" $AB/rung${X}_history.csv
done
cd $EV && ./merge_checkpoints.sh 2>&1 | grep -E "converted|rror"
J=""; for m in rung2s2 rungA rungB rungC rungD rungA1; do for b in mmlupro bbh math gsm8k; do J="$J ${m}_step300:$b"; done; done
for m in rung2s2 rungA rungB rungC rungD rungA1; do J="$J ${m}_step520:math ${m}_step520:gsm8k"; done
JOBS="$J" ./run_jobs.sh
echo "[$(date '+%F %T')] pass@k"
P=""; for m in base DS_step300 noDS_step300 rung2_step300 rungA_step300 rungA1_step300 rungD_step300 GRPO_n2_kl4e-2_step520 GRPO_n2_noKL_step520 PPO_klF_ent0_lr1e-6_step520; do P="$P $m:math500 $m:gsm8k"; done
JOBS="$P" ./run_passk_jobs.sh
/home/lian/miniconda3/bin/python make_results_html.py
echo "[$(date '+%F %T')] submitting DS recipe (rung 3) seed 2"
S=$(submit rung3_DS_gen192.sh 2); echo "submission: $S"; wait_job "$S"
fetch_hist rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed2 $AB/rung3_seed2_history.csv
echo "[$(date '+%F %T')] chain finished"
