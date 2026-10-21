#!/usr/bin/env bash
# Rung 2: rung 1 + clip-higher (0.2 / 0.28), still no dynamic sampling, 32 prompts generated and trained
# per step. Differs from the existing DS run (rung 3, f8h77wol) only in dynamic sampling, with the same
# 32 trained prompts, 256 responses, and 8 actor updates per step, so rung 2 vs. rung 3 is the controlled DS comparison in retained training exposure.
EXP_NAME=rung2-clip0.28-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.28
FILTER=False
GEN_BSZ=32
source "$(dirname "$0")/common.sh"
