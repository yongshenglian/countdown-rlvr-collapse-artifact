#!/usr/bin/env bash
# Option A2: option A + DAPO "soft overlong punishment" (built into verl's dapo reward manager): linear penalty
# from 0 to -1 over the last 1,024 tokens before the 4,096-token cap (buffer/cap ratio as in the DAPO paper's
# 4,096 / 20,480 setting). No code change.
EXP_NAME=rungA2-accOnlyReward-overlongPenalty1024-clip0.28-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=False
GEN_BSZ=32
REWARD_FILE=/home/lian/AI_PPO_2026/ablation/reward_acc_only.py
OVERLONG=True
OVERLONG_LEN=1024
OVERLONG_PENALTY=1.0
source "$(dirname "$0")/common.sh"
