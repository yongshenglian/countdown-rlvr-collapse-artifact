#!/usr/bin/env bash
# Rung 1: plain GRPO in the shared setup. Symmetric clip (0.2 / 0.2), no dynamic sampling,
# 32 prompts generated and trained per step (256 responses and 8 actor updates per step). See common.sh / README.md.
EXP_NAME=rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096
CLIP_HIGH=0.2
FILTER=False
GEN_BSZ=32
source "$(dirname "$0")/common.sh"
