#!/usr/bin/env bash
# Option A1: option A (correctness-only reward, no DS, 32 prompts/step) + DAPO "overlong filtering": responses that
# hit the 4,096-token cap are removed from the actor loss (+algorithm.mask_truncated=True; trainer change, off by
# default). Motivation: option A collapsed after step ~400 as truncated responses rose from 25% to 59%.
EXP_NAME=rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=False
GEN_BSZ=32
REWARD_FILE=/home/lian/AI_PPO_2026/ablation/reward_acc_only.py
EXTRA_ARGS=+algorithm.mask_truncated=True
source "$(dirname "$0")/common.sh"
