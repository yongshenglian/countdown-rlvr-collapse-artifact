#!/usr/bin/env bash
# Option A1 + KL loss 0.01 (low_var_kl to the reference policy): candidate fix for the early short-answer collapse.
# In the GRPO sweep every run with KL >= 0.01 avoided the shortcut and every run with KL <= 0.005 took it.
EXP_NAME=rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=False
GEN_BSZ=32
REWARD_FILE=/home/lian/AI_PPO_2026/ablation/reward_acc_only.py
EXTRA_ARGS=+algorithm.mask_truncated=True
KL_LOSS=True
KL_COEF=0.01
source "$(dirname "$0")/common.sh"
