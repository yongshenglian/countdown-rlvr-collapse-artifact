#!/usr/bin/env bash
# Merge the nine step-300 matched-comparison FSDP actors into HF format.
set -euo pipefail

PY=/home/lian/miniconda3/envs/verl-fork/bin/python
MERGER=/home/lian/verl/scripts/model_merger.py
ROOT=/mnt/sdb/verl/Matched_RLVR_Comparison
DST=$ROOT/merged_hf

declare -A CKPTS=(
  [matched_ppo_s7_step300]=$ROOT/matched-ppo-n8-b32-mini32-seed7-step300/global_step_300/actor
  [matched_ppo_s8_step300]=$ROOT/matched-ppo-n8-b32-mini32-seed8-step300/global_step_300/actor
  [matched_ppo_s9_step300]=$ROOT/matched-ppo-n8-b32-mini32-seed9-step300/global_step_300/actor
  [matched_grpo_s7_step300]=$ROOT/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096-seed7-matched300/global_step_300/actor
  [matched_grpo_s8_step300]=$ROOT/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096-seed8-matched300/global_step_300/actor
  [matched_grpo_s9_step300]=$ROOT/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096-seed9-matched300/global_step_300/actor
  [matched_dapo_s7_step300]=$ROOT/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed7-matched300/global_step_300/actor
  [matched_dapo_s8_step300]=$ROOT/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed8-matched300/global_step_300/actor
  [matched_dapo_s9_step300]=$ROOT/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed9-matched300/global_step_300/actor
)

mkdir -p "$DST"
cd /home/lian/verl

for key in \
  matched_ppo_s7_step300 matched_ppo_s8_step300 matched_ppo_s9_step300 \
  matched_grpo_s7_step300 matched_grpo_s8_step300 matched_grpo_s9_step300 \
  matched_dapo_s7_step300 matched_dapo_s8_step300 matched_dapo_s9_step300
do
  checkpoint=${CKPTS[$key]}
  output=$DST/$key
  if [[ -f "$output/config.json" ]]; then
    echo "exists: $output"
    continue
  fi
  if [[ ! -d "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  hf_source=$checkpoint
  [[ -f "$checkpoint/huggingface/config.json" ]] && hf_source=$checkpoint/huggingface
  "$PY" "$MERGER" \
    --backend fsdp \
    --local_dir "$checkpoint" \
    --hf_model_path "$hf_source" \
    --target_dir "$output"
  cp "$hf_source"/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,special_tokens_map.json,added_tokens.json,generation_config.json} "$output"/ 2>/dev/null || true
  echo "converted: $checkpoint -> $output"
done
