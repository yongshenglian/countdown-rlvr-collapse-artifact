#!/usr/bin/env bash
# Option A: rung 2 (clip 0.2/0.28, no DS, 32 prompts generated and trained per step) + correctness-only reward (no 0.1 format credit).
# Everything else identical to rung 2 (common.sh). Compare with rung 2 (collapsed) and the DS run.
EXP_NAME=rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=False
GEN_BSZ=32
REWARD_FILE=/home/lian/AI_PPO_2026/ablation/reward_acc_only.py
source "$(dirname "$0")/common.sh"
