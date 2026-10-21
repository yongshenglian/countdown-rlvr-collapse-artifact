#!/usr/bin/env bash
# Finish the matched comparison with benchmark/model tasks balanced across GPUs.
# Existing standard+think JSONL pairs are preserved and skipped.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/eval"

PY=/home/lian/miniconda3/envs/verl-fork/bin/python
export EVAL_RESULTS=${EVAL_RESULTS:-results_8k}
export HF_DATASETS_OFFLINE=1
MAX_TOKENS=${MAX_TOKENS:-8192}
LOG_DIR=logs/$EVAL_RESULTS
mkdir -p "$LOG_DIR"

run_task() {
  local gpu=$1 benchmark=$2 model=$3
  local standard=$EVAL_RESULTS/$benchmark/${model}__standard.jsonl
  local think=$EVAL_RESULTS/$benchmark/${model}__think.jsonl
  if [[ -s "$standard" && -s "$think" ]]; then
    echo "[$(date '+%F %T')] skip GPU $gpu $benchmark $model (already complete)"
    return
  fi
  echo "[$(date '+%F %T')] start GPU $gpu $benchmark $model"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "eval_${benchmark}.py" \
    --model "$model" --max_tokens "$MAX_TOKENS" \
    > "$LOG_DIR/${benchmark}_${model}.log" 2>&1
  echo "[$(date '+%F %T')] done  GPU $gpu $benchmark $model"
}

worker() {
  local gpu=$1
  shift
  local task benchmark model
  for task in "$@"; do
    benchmark=${task%%:*}
    model=${task#*:}
    run_task "$gpu" "$benchmark" "$model"
  done
}

# GPU 3 first waits for the preserved GRPO-seed-8 MMLU-Pro process.
(
  while pgrep -f 'eval_mmlupro.py --model matched_grpo_s8_step300' >/dev/null; do
    echo "[$(date '+%F %T')] GPU 3 waiting for matched_grpo_s8_step300 MMLU-Pro"
    sleep 30
  done
  worker 3 \
    mmlupro:matched_dapo_s9_step300 \
    math:matched_grpo_s9_step300 \
    bbh:matched_dapo_s7_step300 \
    gsm8k:matched_dapo_s8_step300
) &
p3=$!

worker 0 \
  mmlupro:matched_grpo_s9_step300 \
  math:matched_dapo_s7_step300 \
  bbh:matched_dapo_s8_step300 \
  gsm8k:matched_dapo_s9_step300 &
p0=$!

worker 1 \
  mmlupro:matched_dapo_s7_step300 \
  math:matched_dapo_s8_step300 \
  bbh:matched_dapo_s9_step300 \
  gsm8k:matched_grpo_s9_step300 &
p1=$!

worker 2 \
  mmlupro:matched_dapo_s8_step300 \
  math:matched_dapo_s9_step300 \
  bbh:matched_grpo_s9_step300 \
  gsm8k:matched_dapo_s7_step300 &
p2=$!

wait "$p0" "$p1" "$p2" "$p3"

for benchmark in gsm8k math bbh mmlupro; do
  "$PY" "eval_${benchmark}.py" --summarize \
    > "$LOG_DIR/${benchmark}_summarize.log" 2>&1
done
cp "$LOG_DIR/gsm8k_summarize.log" "$EVAL_RESULTS/gsm8k/summary.txt"

/home/lian/miniconda3/bin/python run_registry.py > "$LOG_DIR/registry_update.log" 2>&1 || true
"$PY" make_results_md.py >> "$LOG_DIR/registry_update.log" 2>&1
echo "[$(date '+%F %T')] balanced queue finished"
