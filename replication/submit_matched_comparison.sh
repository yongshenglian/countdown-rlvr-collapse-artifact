#!/usr/bin/env bash
# Submit one run from MATCHED_COMPARISON_PROTOCOL.md.
# Usage: bash replication/submit_matched_comparison.sh ppo|grpo|dapo 7|8|9
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
PROJECT=Matched_RLVR_Comparison
TOTAL_STEPS=300

recipe=${1:?recipe must be ppo, grpo, or dapo}
seed=${2:?seed must be 7, 8, or 9}
case "$recipe" in ppo|grpo|dapo) ;; *) echo "Invalid recipe: $recipe" >&2; exit 2 ;; esac
case "$seed" in 7|8|9) ;; *) echo "Invalid prospective seed: $seed" >&2; exit 2 ;; esac

if ! "$RAY" status >/dev/null 2>&1; then
  echo "Ray cluster is not available at $ADDRESS" >&2
  exit 3
fi
if [[ ! -f "$REWARD_FILE" ]]; then
  echo "Missing frozen reward implementation: $REWARD_FILE" >&2
  exit 4
fi
mkdir -p "$LOGDIR"

if [[ "$recipe" == grpo || "$recipe" == dapo ]]; then
  if [[ "$recipe" == grpo ]]; then
    rung=rung1_grpo_clip0.2_noDS.sh
    stem=rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096
  else
    rung=rung2_clip0.28_noDS.sh
    stem=rung2-clip0.28-noDS-gen32-train32-n8-resp4096
  fi
  run_name=${stem}-seed${seed}-matched300
  submit_log=$LOGDIR/submit_matched_${recipe}_seed${seed}.log
  PROJECT=$PROJECT SEED=$seed NAME_SUFFIX=-matched300 TOTAL_STEPS=$TOTAL_STEPS TOTAL_EPOCHS=2 SAVE_FREQ=50 \
    bash "$ROOT/ablation/$rung" >"$submit_log" 2>&1
  submission_id=$(grep -o "${run_name}-[0-9]\\{8\\}-[0-9]\\{6\\}" "$submit_log" | head -1)
else
  run_name=matched-ppo-n8-b32-mini32-seed${seed}-step300
  checkpoint_dir=/mnt/sdb/verl/$PROJECT/$run_name
  submit_log=$LOGDIR/submit_matched_ppo_seed${seed}.log
  submission_id=${run_name}-$(date +%Y%m%d-%H%M%S)

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
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    critic.ppo_mini_batch_size=32 \
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
    trainer.save_freq=50 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.default_local_dir="$checkpoint_dir" \
    trainer.resume_mode=auto \
    +data.seed="$seed" \
    +actor_rollout_ref.rollout.seed="$seed" >"$submit_log" 2>&1
fi

if [[ -z "${submission_id:-}" ]]; then
  echo "Submission failed; inspect $submit_log" >&2
  exit 5
fi
if ! grep -q "submitted successfully" "$submit_log"; then
  echo "Submission was not acknowledged; inspect $submit_log" >&2
  exit 6
fi
printf '%s\n' "$submission_id"
