#!/usr/bin/env bash
# Run the seed-10--13 extension of the matched comparison serially on four GPUs
# (order frozen in MATCHED_COMPARISON_PROTOCOL.md, "Seed extension").
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

run_one ppo 10
run_one grpo 10
run_one dapo 10
run_one grpo 11
run_one dapo 11
run_one ppo 11
run_one dapo 12
run_one ppo 12
run_one grpo 12
run_one ppo 13
run_one dapo 13
run_one grpo 13
printf '[%s] matched seed-extension queue complete\n' "$(date '+%F %T')"
