#!/usr/bin/env bash
# Experiment B PPO arm (replication/ACTIVE_CLIP_PROTOCOL.md): the matched PPO run with actor and critic
# mini-batch 4 prompts (32 responses), i.e. 8 actor and 8 critic updates per step, so the clip binds.
# Checkpoints go to /mnt/sda (kept in full) and every training batch and validation set is dumped
# (needs replication/patches/experiment_b_logging.patch). Everything else is copied verbatim from
# submit_matched_comparison.sh.
# Usage: bash replication/submit_active_clip_ppo.sh <seed> [project] [total_steps] [save_freq] [name_suffix]
set -euo pipefail

ROOT=/home/lian/AI_PPO_2026
VERL=/home/lian/verl
ENV_BIN=/home/lian/miniconda3/envs/verl-fork/bin
RAY=$ENV_BIN/ray
ADDRESS=http://localhost:8265
RUNTIME_ENV=$VERL/verl/trainer/runtime_env.yaml
MODEL_PATH=/home/lian/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
REWARD_FILE=$ROOT/ablation/reward_sep2025.py
LOGDIR=$ROOT/replication/logs

seed=${1:?seed required}
PROJECT=${2:-Active_Clip_Comparison}
TOTAL_STEPS=${3:-300}
SAVE_FREQ=${4:-50}
suffix=${5:-}
run_name=b-ppo-clip0.2-mb4-n8-resp4096-seed${seed}${suffix}
checkpoint_dir=/mnt/sda/data/verl/$PROJECT/$run_name   # full checkpoints kept (no pruning)
submit_log=$LOGDIR/submit_b_ppo_seed${seed}${suffix}.log
submission_id=${run_name}-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGDIR"

cd "$VERL"
"$RAY" job submit --address="$ADDRESS" --no-wait --runtime-env="$RUNTIME_ENV" \
  --working-dir "$VERL" --submission-id "$submission_id" \
  -- "$ENV_BIN/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files=$ROOT/replication/data/countdown_train.parquet \
  data.val_files=$ROOT/replication/data/countdown_test.parquet \
  data.prompt_key=prompt \
  data.truncation=left \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.train_batch_size=32 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.optim.total_training_steps=$TOTAL_STEPS \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6144 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6144 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=True \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  critic.ppo_mini_batch_size=4 \
  critic.ppo_epochs=1 \
  critic.use_dynamic_bsz=True \
  critic.ppo_max_token_len_per_gpu=6144 \
  critic.forward_max_token_len_per_gpu=6144 \
  critic.ulysses_sequence_parallel_size=4 \
  critic.optim.lr=1e-5 \
  +critic.optim.lr_warmup_steps=10 \
  critic.optim.weight_decay=0.1 \
  critic.optim.total_training_steps=$TOTAL_STEPS \
  reward_model.reward_manager=naive \
  custom_reward_function.path="$REWARD_FILE" \
  custom_reward_function.name=compute_score \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$PROJECT" \
  trainer.experiment_name="$run_name" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.test_freq=5 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.total_epochs=2 \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.default_local_dir="$checkpoint_dir" \
  trainer.resume_mode=auto \
  +data.seed="$seed" \
  +actor_rollout_ref.rollout.seed="$seed" \
  ++trainer.step_dump_dir="$checkpoint_dir/dumps/steps" \
  ++trainer.validation_data_dir="$checkpoint_dir/dumps/val" >"$submit_log" 2>&1
grep -q "submitted successfully" "$submit_log" || { echo "Submission failed; inspect $submit_log" >&2; exit 6; }
printf '%s\n' "$submission_id"
