#!/usr/bin/env bash
# 1) evaluate rung 2 (step 300 on all benchmarks; steps 100/520 on GSM8K+MATH), updating RESULTS.md
# 2) then submit rung 1 training and wait until it finishes
set -uo pipefail
cd /home/lian/AI_PPO_2026/eval
J=""; for b in mmlupro bbh math gsm8k; do J="$J rung2_step300:$b"; done
for s in 100 520; do J="$J rung2_step$s:math rung2_step$s:gsm8k"; done
echo "[$(date '+%F %T')] evaluating rung 2: $J"
JOBS="$J" ./run_jobs.sh
cd /home/lian/AI_PPO_2026/ablation
echo "[$(date '+%F %T')] submitting rung 1"
bash rung1_grpo_clip0.2_noDS.sh > logs/rung1_submit.log 2>&1
SUB=$(grep -o "rung1-grpo[^ ]*-[0-9]\{8\}-[0-9]\{6\}" logs/rung1_submit.log | head -1)
echo "rung 1 submission: $SUB"
R=/home/lian/miniconda3/envs/verl-fork/bin/ray
while true; do
  st=$($R job status "$SUB" --address=http://localhost:8265 2>&1 | grep -o -E "SUCCEEDED|FAILED|STOPPED" | head -1)
  [ -n "$st" ] && break; sleep 300
done
echo "[$(date '+%F %T')] rung 1 finished: $st"
$R job logs "$SUB" --address=http://localhost:8265 > logs/rung1_job.log 2>&1
