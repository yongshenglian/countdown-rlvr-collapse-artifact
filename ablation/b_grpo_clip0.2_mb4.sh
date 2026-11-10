#!/usr/bin/env bash
# Experiment B: group-relative (GRPO) advantages with an ACTIVE clip. Mini-batch 4 prompts (32 responses), so each
# step makes 8 off-policy actor updates on its 256 responses and the clip can bind. Clip 0.2 upper / 0.2 lower.
# Everything else is the matched setup in common.sh. See replication/ACTIVE_CLIP_PROTOCOL.md.
EXP_NAME=b-grpo-clip0.2-noDS-mb4-n8-resp4096
CLIP_HIGH=0.2
FILTER=False
GEN_BSZ=32
MINI_BSZ=4
CKPT_ROOT=/mnt/sda/data/verl   # full checkpoints kept (no pruning)
DUMP=1
source "$(dirname "$0")/common.sh"
