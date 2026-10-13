#!/usr/bin/env bash
# Rung 3 = the DS recipe (dynamic sampling on, clip 0.2/0.28, 192 prompts generated per round, first 32 non-tied
# groups trained). Identical to the original DS run DAPO_new/f8h77wol except save_freq=50. Use with SEED=2 (etc.)
# for additional training seeds; SEED unset reproduces the original seed.
EXP_NAME=rung3-DS-clip0.28-gen192-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=True
GEN_BSZ=192
source "$(dirname "$0")/common.sh"
