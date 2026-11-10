#!/usr/bin/env bash
# Merge the step-300 matched-comparison and seed-extension FSDP actors into HF format.
set -euo pipefail

PY=/home/lian/miniconda3/envs/verl-fork/bin/python
MERGER=/home/lian/verl/scripts/model_merger.py
ROOT=/mnt/sdb/verl/Matched_RLVR_Comparison
DST=$ROOT/merged_hf

# Seeds 7-9 (matched comparison) and 10-13 (seed extension). Runs whose step-300 actor is not
# complete yet are skipped, so the script can be rerun as runs finish.
declare -A RUN=(
  [ppo]=matched-ppo-n8-b32-mini32-seed%s-step300
  [grpo]=rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096-seed%s-matched300
  [dapo]=rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed%s-matched300
)

mkdir -p "$DST"
cd /home/lian/verl

for seed in 7 8 9 10 11 12 13; do
for method in ppo grpo dapo; do
  key=matched_${method}_s${seed}_step300
  checkpoint=$ROOT/$(printf "${RUN[$method]}" "$seed")/global_step_300/actor
  output=$DST/$key
  if [[ -f "$output/config.json" ]]; then
    echo "exists: $output"
    continue
  fi
  if [[ $(ls "$checkpoint"/model_world_size_4_rank_*.pt 2>/dev/null | wc -l) -ne 4 ]]; then
    echo "not finished: $checkpoint" >&2
    continue
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
done
