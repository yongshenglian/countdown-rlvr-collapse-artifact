#!/usr/bin/env bash
# Submit one predeclared DAPO-style continuation from its exact step-100 prefix.
# Usage: bash replication/submit_dapo_continuation.sh nods 3
#        bash replication/submit_dapo_continuation.sh ds 3
set -euo pipefail

ROOT=/home/lian/AI_PPO_2026
AB=$ROOT/ablation
LOGDIR=$ROOT/replication/logs
RAY=/home/lian/miniconda3/envs/verl-fork/bin/ray
ADDRESS=http://localhost:8265

recipe=${1:?recipe must be nods or ds}
seed=${2:?seed must be 3, 4, or 5}
case "$seed" in
  3|4|5) ;;
  *) echo "Refusing non-protocol seed: $seed" >&2; exit 2 ;;
esac

case "$recipe" in
  nods)
    script=rung2_clip0.28_noDS.sh
    stem=rung2-clip0.28-noDS-gen32-train32-n8-resp4096
    ;;
  ds)
    script=rung3_DS_gen192.sh
    stem=rung3-DS-clip0.28-gen192-train32-n8-resp4096
    ;;
  *)
    echo "Refusing non-protocol recipe: $recipe" >&2
    exit 2
    ;;
esac

run_name=${stem}-seed${seed}-s100
checkpoint_dir=/mnt/sdb/verl/DAPO_ablation/$run_name
if [[ ! -d "$checkpoint_dir/global_step_100" ]]; then
  echo "Missing required prefix checkpoint: $checkpoint_dir/global_step_100" >&2
  exit 3
fi

if ! "$RAY" status >/dev/null 2>&1; then
  echo "Ray cluster is not available at $ADDRESS" >&2
  exit 4
fi

mkdir -p "$LOGDIR"
submit_log=$LOGDIR/submit_${recipe}_seed${seed}.log
echo "Submitting $run_name from global_step_100 to the fixed step-520 horizon"
SEED=$seed NAME_SUFFIX=-s100 TOTAL_STEPS=520 TOTAL_EPOCHS=10 SAVE_FREQ=50 \
  bash "$AB/$script" >"$submit_log" 2>&1

submission_id=$(grep -o "rung[0-9A-Z][^ ]*-[0-9]\{8\}-[0-9]\{6\}" "$submit_log" | head -1)
if [[ -z "$submission_id" ]]; then
  echo "Submission failed; inspect $submit_log" >&2
  exit 5
fi
printf '%s\n' "$submission_id"

