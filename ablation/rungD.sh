#!/usr/bin/env bash
# Option D: rung 2 (clip 0.2/0.28, no DS, 32 prompts generated and trained per step) + batch-level advantage
# normalisation (Lite PPO idea, arXiv:2508.08221): algorithm.adv_estimator=reinforce_plus_plus_baseline subtracts
# each group's mean reward, then whitens advantages over the whole batch (verl core_algos.py), instead of dividing
# by each group's std. Everything else identical to rung 2 (common.sh).
EXP_NAME=rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=False
GEN_BSZ=32
ADV_EST=reinforce_plus_plus_baseline
source "$(dirname "$0")/common.sh"
