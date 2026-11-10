#!/usr/bin/env bash
# Experiment B (replication/ACTIVE_CLIP_PROTOCOL.md): GRPO (clip 0.2/0.2), DAPO-style (0.2/0.28) and PPO with an active clip.
# 1. Waits for the seed-extension queue to finish (and stops if it did not complete).
# 2. Applies the logging-only patch to /home/lian/verl (clip split metrics, per-step batch dumps).
# 3. Smoke test: 10 steps per arm (seed 99, separate W&B project). Proceeds only if all three arms show clipped
#    tokens (the clip binds) and finish without error.
# 4. Runs the 30 frozen runs (seeds 7-16 x three arms) serially in the frozen order. Checkpoints and dumps
#    go to /mnt/sda and are kept in full.
set -euo pipefail

ROOT=/home/lian/AI_PPO_2026
VERL=/home/lian/verl
RAY=/home/lian/miniconda3/envs/verl-fork/bin/ray
ADDRESS=http://localhost:8265
LOGDIR=$ROOT/replication/logs
EXT_LOG=$LOGDIR/matched_extension_queue.log
PATCH=$ROOT/replication/patches/experiment_b_logging.patch
PROJECT=Active_Clip_Comparison
mkdir -p "$LOGDIR"
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

wait_job() {
  local job=$1 status
  while true; do
    status=$("$RAY" job status "$job" --address="$ADDRESS" 2>&1 |
      grep -o -i -E "succeeded|failed|stopped" | head -1 | tr '[:lower:]' '[:upper:]' || true)
    if [[ -n "$status" ]]; then
      say "$job $status"
      "$RAY" job logs "$job" --address="$ADDRESS" >"$LOGDIR/$job.log" 2>&1 || true
      [[ "$status" == "SUCCEEDED" ]]
      return
    fi
    sleep 60
  done
}

submit() {  # submit <arm: g0.2|g0.28|ppo> <seed> <project> <total_steps> <save_freq> [name_suffix]
  local arm=$1 seed=$2 project=$3 steps=$4 save=$5 suffix=${6:-} slog job clip
  if [[ "$arm" == ppo ]]; then
    job=$(bash "$ROOT/replication/submit_active_clip_ppo.sh" "$seed" "$project" "$steps" "$save" "$suffix" | tail -1)
  else
    clip=${arm#g}
    slog=$LOGDIR/submit_b_clip${clip}_seed${seed}${suffix}.log
    PROJECT=$project SEED=$seed NAME_SUFFIX=$suffix TOTAL_STEPS=$steps TOTAL_EPOCHS=2 SAVE_FREQ=$save \
      bash "$ROOT/ablation/b_grpo_clip${clip}_mb4.sh" >"$slog" 2>&1
    grep -q "submitted successfully" "$slog" || { say "submission failed: $slog"; exit 5; }
    job=$(grep -o "b-grpo-clip${clip}-noDS-mb4-n8-resp4096-seed${seed}${suffix}-[0-9]\{8\}-[0-9]\{6\}" "$slog" | head -1)
  fi
  [[ -n "$job" ]] || { say "no submission id for $arm seed $seed"; exit 5; }
  printf '%s\n' "$job"
}

# 1. Wait for the extension queue.
while pgrep -f run_matched_extension_queue.sh >/dev/null; do sleep 300; done
grep -q "queue complete" "$EXT_LOG" || { say "extension queue did not complete; not starting B"; exit 3; }
say "extension queue complete"

# 2. Logging-only patch.
if git -C "$VERL" apply --check -p1 "$PATCH" 2>/dev/null; then
  git -C "$VERL" apply -p1 "$PATCH"; say "applied $PATCH"
elif git -C "$VERL" apply --check -R -p1 "$PATCH" 2>/dev/null; then
  say "patch already applied"
else
  say "patch does not apply cleanly; stopping"; exit 4
fi

# 3. Smoke test.
for arm in g0.2 g0.28 ppo; do
  job=$(submit "$arm" 99 Active_Clip_Smoke 10 1000 -smoke10)
  say "smoke submitted $arm: $job"
  wait_job "$job" || { say "smoke FAILED for $arm"; exit 6; }
  up=$(grep -oE "actor/pg_clipfrac_up_pct:[0-9.e-]+" "$LOGDIR/$job.log" | cut -d: -f2 | sort -g | tail -1)
  down=$(grep -oE "actor/pg_clipfrac_down_pct:[0-9.e-]+" "$LOGDIR/$job.log" | cut -d: -f2 | sort -g | tail -1)
  say "smoke $arm: max upper-clipped ${up:-none}% and lower-clipped ${down:-none}% of tokens per step"
  awk -v u="${up:-0}" -v d="${down:-0}" 'BEGIN{exit !(u + d > 0)}' || { say "clip never binds for $arm; stopping"; exit 7; }
  ndump=$(ls /mnt/sda/data/verl/Active_Clip_Smoke/${job%-*-*}/dumps/steps/step_*.pt 2>/dev/null | wc -l)
  nval=$(ls /mnt/sda/data/verl/Active_Clip_Smoke/${job%-*-*}/dumps/val/*.jsonl 2>/dev/null | wc -l)
  say "smoke $arm: $ndump step dumps, $nval validation dumps"
  [[ "$ndump" -ge 10 && "$nval" -ge 1 ]] || { say "dumps missing for $arm; stopping"; exit 8; }
done
say "smoke test passed; starting the 30 frozen runs"

# 4. Frozen order: the three arms rotate across seeds.
for row in "7 g0.2 g0.28 ppo" "8 g0.28 ppo g0.2" "9 ppo g0.2 g0.28" "10 g0.2 g0.28 ppo" "11 g0.28 ppo g0.2" \
           "12 ppo g0.2 g0.28" "13 g0.2 g0.28 ppo" "14 g0.28 ppo g0.2" "15 ppo g0.2 g0.28" "16 g0.2 g0.28 ppo"; do
  set -- $row
  seed=$1; shift
  for arm in "$@"; do
    job=$(submit "$arm" "$seed" "$PROJECT" 300 50)
    say "submitted $arm seed $seed: $job"
    wait_job "$job"
  done
done
say "active-clip queue complete"
