#!/usr/bin/env bash
# Delete optimizer-state shards from a FINISHED run's checkpoints, keeping model weights, tokenizer
# files and extra state. Optimizer state is only needed to resume training; merging and evaluation
# use the model shards. Refuses unless global_step_300 exists with all four actor model shards.
# Usage: bash replication/prune_optimizer_states.sh <checkpoint_run_dir> [...]
set -euo pipefail

for dir in "$@"; do
  final=$dir/global_step_300/actor
  n=$(ls "$final"/model_world_size_4_rank_*.pt 2>/dev/null | wc -l)
  if [[ "$n" -ne 4 ]]; then
    echo "skip (no complete step-300 actor): $dir" >&2
    continue
  fi
  files=$(find "$dir" -path '*/global_step_*/*' -name 'optim_world_size_*.pt' -type f)
  if [[ -z "$files" ]]; then
    echo "nothing to prune: $dir"
    continue
  fi
  bytes=$(du -cb $files | tail -1 | cut -f1)
  rm -f $files
  printf 'pruned %d optimizer shards (%.1f GB): %s\n' "$(wc -w <<<"$files")" "$(awk -v b="$bytes" 'BEGIN{print b/1e9}')" "$dir"
done
