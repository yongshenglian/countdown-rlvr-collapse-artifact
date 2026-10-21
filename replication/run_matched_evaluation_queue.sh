#!/usr/bin/env bash
# Evaluate every seed from the step-300 matched PPO/GRPO/DAPO-style comparison.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export EVAL_RESULTS=${EVAL_RESULTS:-results_8k}
export QUEUE="matched_ppo_s7_step300 matched_ppo_s8_step300 matched_ppo_s9_step300 matched_grpo_s7_step300 matched_grpo_s8_step300 matched_grpo_s9_step300 matched_dapo_s7_step300 matched_dapo_s8_step300 matched_dapo_s9_step300"

exec "$ROOT/eval/run_queue.sh"
