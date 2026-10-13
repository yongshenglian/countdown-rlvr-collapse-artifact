#!/usr/bin/env bash
# Option B: rung 2 (clip 0.2/0.28, no DS, 32 prompts generated and trained per step) + advantage not divided by the group std (Dr. GRPO style).
# Everything else identical to rung 2 (common.sh). Compare with rung 2 (collapsed) and the DS run.
EXP_NAME=rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=False
GEN_BSZ=32
NORM_STD=False
source "$(dirname "$0")/common.sh"
