#!/usr/bin/env bash
# Serial controller for all training runs in replication/PROTOCOL.md.
# It starts after the already-submitted no-DS seed-3 continuation.
set -euo pipefail

ROOT=/home/lian/AI_PPO_2026
RAY=/home/lian/miniconda3/envs/verl-fork/bin/ray
ADDRESS=http://localhost:8265
LOGDIR=$ROOT/replication/logs
CURRENT_JOB=${1:?current no-DS seed-3 Ray submission id is required}
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
    sleep 120
  done
}

submit_and_wait() {
  local recipe=$1 seed=$2 job
  job=$(bash "$ROOT/replication/submit_dapo_continuation.sh" "$recipe" "$seed")
  job=$(printf '%s\n' "$job" | tail -1)
  printf '[%s] submitted %s seed %s: %s\n' "$(date '+%F %T')" "$recipe" "$seed" "$job"
  wait_job "$job"
}

submit_main_and_wait() {
  local recipe=$1 seed=$2 job
  job=$(bash "$ROOT/replication/submit_main_ppo.sh" "$recipe" "$seed")
  job=$(printf '%s\n' "$job" | tail -1)
  printf '[%s] submitted %s seed %s: %s\n' "$(date '+%F %T')" "$recipe" "$seed" "$job"
  wait_job "$job"
}

printf '[%s] monitoring existing no-DS seed 3: %s\n' "$(date '+%F %T')" "$CURRENT_JOB"
wait_job "$CURRENT_JOB"
submit_and_wait nods 4
submit_and_wait nods 5
submit_and_wait ds 3
submit_and_wait ds 4
submit_and_wait ds 5
printf '[%s] DAPO-style continuation queue complete\n' "$(date '+%F %T')"

submit_main_and_wait grpo 3
submit_main_and_wait grpo 4
submit_main_and_wait grpo 5
submit_main_and_wait ppo 3
submit_main_and_wait ppo 4
submit_main_and_wait ppo 5
printf '[%s] complete training queue finished\n' "$(date '+%F %T')"
