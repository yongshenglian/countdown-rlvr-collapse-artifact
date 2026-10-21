#!/usr/bin/env bash
# Run all prospective matched comparisons serially on four GPUs.
set -euo pipefail

ROOT=/home/lian/AI_PPO_2026
RAY=/home/lian/miniconda3/envs/verl-fork/bin/ray
ADDRESS=http://localhost:8265
LOGDIR=$ROOT/replication/logs
mkdir -p "$LOGDIR"

wait_job() {
  local job=$1 status
  while true; do
    status=$("$RAY" job status "$job" --address="$ADDRESS" 2>&1 |
      grep -o -i -E "succeeded|failed|stopped" | head -1 | tr '[:lower:]' '[:upper:]' || true)
    if [[ -n "$status" ]]; then
      printf '[%s] %s %s\n' "$(date '+%F %T')" "$job" "$status"
      "$RAY" job logs "$job" --address="$ADDRESS" >"$LOGDIR/$job.log" 2>&1 || true
      [[ "$status" == "SUCCEEDED" ]]
      return
    fi
    sleep 60
  done
}

run_one() {
  local recipe=$1 seed=$2 job
  job=$(bash "$ROOT/replication/submit_matched_comparison.sh" "$recipe" "$seed")
  job=$(printf '%s\n' "$job" | tail -1)
  printf '[%s] submitted %s seed %s: %s\n' "$(date '+%F %T')" "$recipe" "$seed" "$job"
  wait_job "$job"
}

if [[ $# -gt 0 ]]; then
  printf '[%s] monitoring existing PPO seed 7: %s\n' "$(date '+%F %T')" "$1"
  wait_job "$1"
else
  run_one ppo 7
fi
run_one grpo 7
run_one dapo 7
run_one grpo 8
run_one dapo 8
run_one ppo 8
run_one dapo 9
run_one ppo 9
run_one grpo 9
printf '[%s] matched step-300 training queue complete\n' "$(date '+%F %T')"
