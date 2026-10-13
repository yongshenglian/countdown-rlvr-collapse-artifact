#!/usr/bin/env bash
# Optional: rung 1 with the 2048-token response cap used by the PPO/GRPO runs; rung 1 vs. 1b isolates the cap.
EXP_NAME=rung1b-grpo-clip0.2-noDS-gen32-train32-n8-resp2048
CLIP_HIGH=0.2
FILTER=False
GEN_BSZ=32
MAX_RESP=2048
source "$(dirname "$0")/common.sh"
