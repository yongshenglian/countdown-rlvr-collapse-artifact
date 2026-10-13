#!/usr/bin/env bash
# Merge FSDP actor shards (world size 4) into HF format for evaluation.
#   ./merge_checkpoints.sh            converts every entry below that is not converted yet
set -euo pipefail
PY=/home/lian/miniconda3/envs/verl-fork/bin/python
MERGER=/home/lian/verl/scripts/model_merger.py
DST=/mnt/sdb/verl/DAPO_new/merged_hf
declare -A CKPTS=(
  [DS_step300]=/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-filtering-tokenlevel-bsz-32-0910/global_step_300/actor
  [noDS_step300]=/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_300/actor
  [PPO_step400]=/home/lian/verl/checkpoints/PPO_Countdown_R1_0923/ppo_qwen2.5_1.5b-train192-mini32/global_step_400/actor
  [PPO_klF_ent0_lr3e-6_step520]=/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy0-3lr-6/global_step_520/actor
  [PPO_kl1e-3_ent1e-3_lr3e-6_step520]=/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-klloss1-entropy-3lr-6/global_step_520/actor
  [PPO_kl0_ent1e-3_lr3e-6_step520]=/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-klloss0-entropy1-3lr-6/global_step_520/actor
  [PPO_klF_ent1e-3_lr3e-6_step520]=/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy1-3lr-6/global_step_520/actor
  [PPO_klF_ent0_lr1e-6_step520]=/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy0-1lr-6/global_step_520/actor
  [GRPO_n2_kl1e-3_step520]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2/global_step_520/actor
  [GRPO_n2_noKL_step520]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2-noKL-Loss/global_step_520/actor
  [GRPO_n2_kl5e-3_step520]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.005-entropy0-rollout2/global_step_520/actor
  [GRPO_n2_kl1e-2_step250]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.01-entropy0-rollout2/global_step_250/actor
  [GRPO_n2_kl4e-2_step520]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.04-entropy0-rollout2/global_step_520/actor
  [GRPO_n4_kl1e-3_step520]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4/global_step_520/actor
  [GRPO_n8_kl1e-3_step520]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout8/global_step_520/actor
  [GRPO_n4_kl1e-6_step520]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4-equalReward/global_step_520/actor
  [GRPO_n4_kl1e-6_eqR_step520]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4-equalReward-checkagain/global_step_520/actor
  [noDS_step100]=/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_100/actor
  [noDS_step200]=/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_200/actor
  [noDS_step400]=/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_400/actor
  [noDS_step500]=/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_500/actor
  [noDS_step520]=/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_520/actor
  [GRPO_n2_noKL_step250]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2-noKL-Loss/global_step_250/actor
  [GRPO_n2_kl1e-3_step250]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2/global_step_250/actor
  [GRPO_n2_kl5e-3_step250]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.005-entropy0-rollout2/global_step_250/actor
  [GRPO_n2_kl4e-2_step250]=/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.04-entropy0-rollout2/global_step_250/actor
  [rung2_step100]=/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_100/actor
  [rung2_step300]=/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300/actor
  [rung2_step520]=/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520/actor
  [rung1_step100]=/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096/global_step_100/actor
  [rung1_step300]=/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096/global_step_300/actor
  [rung1_step520]=/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096/global_step_520/actor
  [rung2s2_step300]=/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2/global_step_300/actor
  [rung2s2_step520]=/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2/global_step_520/actor
  [rungA_step300]=/mnt/sdb/verl/DAPO_ablation/rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300/actor
  [rungA_step520]=/mnt/sdb/verl/DAPO_ablation/rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520/actor
  [rungB_step300]=/mnt/sdb/verl/DAPO_ablation/rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300/actor
  [rungB_step520]=/mnt/sdb/verl/DAPO_ablation/rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520/actor
  [rungC_step300]=/mnt/sdb/verl/DAPO_ablation/rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300/actor
  [rungC_step520]=/mnt/sdb/verl/DAPO_ablation/rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520/actor
  [rungD_step300]=/mnt/sdb/verl/DAPO_ablation/rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300/actor
  [rungD_step520]=/mnt/sdb/verl/DAPO_ablation/rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520/actor
  [rungA1_step300]=/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300/actor
  [rungA1_step520]=/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520/actor
  [DSs2_step200]=/mnt/sdb/verl/DAPO_ablation/rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed2/global_step_200/actor
  [rungA1s2_step300]=/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2/global_step_300/actor
  [rungA1s2_step520]=/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2/global_step_520/actor
)
cd /home/lian/verl
for k in "${!CKPTS[@]}"; do
  ckpt=${CKPTS[$k]}
  out=$DST/$k
  if [ -f "$out/config.json" ]; then echo "exists: $out"; continue; fi
  if [ ! -d "$ckpt" ]; then echo "not trained yet: $ckpt"; continue; fi
  # newer verl checkpoints keep the HF config/tokenizer in actor/huggingface/
  hf=$ckpt; [ -f "$ckpt/huggingface/config.json" ] && hf=$ckpt/huggingface
  $PY $MERGER --backend fsdp --local_dir "$ckpt" --hf_model_path "$hf" --target_dir "$out"
  cp "$hf"/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,special_tokens_map.json,added_tokens.json,generation_config.json} "$out"/ 2>/dev/null || true
  echo "converted: $ckpt -> $out"
done
