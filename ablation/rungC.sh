#!/usr/bin/env bash
# Option C: rung 2 (clip 0.2/0.28, no DS, 32 prompts generated and trained per step) + all-right / all-wrong groups masked out of the loss instead of resampled.
# Everything else identical to rung 2 (common.sh). Compare with rung 2 (collapsed) and the DS run.
EXP_NAME=rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=False
GEN_BSZ=32
EXTRA_ARGS=+algorithm.mask_uninformative_groups=True
source "$(dirname "$0")/common.sh"
