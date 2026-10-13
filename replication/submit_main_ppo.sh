#!/usr/bin/env bash
# Submit one frozen strong-KL GRPO or PPO replication run.
# Usage: bash replication/submit_main_ppo.sh grpo 3
#        bash replication/submit_main_ppo.sh ppo 3
set -euo pipefail

ROOT=/home/lian/AI_PPO_2026
VERL=/home/lian/verl
ENV_BIN=/home/lian/miniconda3/envs/verl-fork/bin
RAY=$ENV_BIN/ray
ADDRESS=http://localhost:8265
RUNTIME_ENV=$VERL/verl/trainer/runtime_env.yaml
MODEL_PATH=/home/lian/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
REWARD_FILE=$ROOT/replication/reward_naive_countdown.py
LOGDIR=$ROOT/replication/logs

recipe=${1:?recipe must be grpo or ppo}
seed=${2:?seed must be 3, 4, or 5}
case "$seed" in
  3|4|5) ;;
  *) echo "Refusing non-protocol seed: $seed" >&2; exit 2 ;;
esac

case "$recipe" in
  grpo)
    project=GRPO_Countdown_replication
    run_name=grpo-n2-kl0.04-seed${seed}
    recipe_args=(
      algorithm.adv_estimator=grpo
      actor_rollout_ref.actor.ppo_mini_batch_size=64
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
      actor_rollout_ref.actor.use_kl_loss=True
      actor_rollout_ref.actor.kl_loss_coef=0.04
      actor_rollout_ref.actor.kl_loss_type=low_var_kl
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
      actor_rollout_ref.rollout.n=2
    )
    ;;
  ppo)
    project=PPO_Countdown_replication
    run_name=ppo-lr1e-6-kl-off-ent0-seed${seed}
    recipe_args=(
      algorithm.adv_estimator=gae
      actor_rollout_ref.actor.ppo_mini_batch_size=16
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
      actor_rollout_ref.actor.use_kl_loss=False
      actor_rollout_ref.actor.kl_loss_coef=0.001
      actor_rollout_ref.actor.kl_loss_type=low_var_kl
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
      actor_rollout_ref.rollout.n=1
      critic.optim.lr=1e-5
      critic.model.path=$MODEL_PATH
      critic.model.use_remove_padding=True
      critic.model.enable_gradient_checkpointing=False
      critic.ppo_micro_batch_size_per_gpu=4
      critic.model.fsdp_config.param_offload=False
      critic.model.fsdp_config.optimizer_offload=False
    )
    ;;
  *) echo "Refusing non-protocol recipe: $recipe" >&2; exit 2 ;;
esac

if ! "$RAY" status >/dev/null 2>&1; then
  echo "Ray cluster is not available at $ADDRESS" >&2
  exit 3
fi
if [[ ! -f "$REWARD_FILE" ]]; then
  echo "Missing frozen reward implementation: $REWARD_FILE" >&2
  exit 4
fi

checkpoint_dir=/mnt/sdb/verl/replication/$project/$run_name
mkdir -p "$LOGDIR"
submission_id=${run_name}-$(date +%Y%m%d-%H%M%S)
submit_log=$LOGDIR/submit_${recipe}_seed${seed}.log

cd "$VERL"
"$RAY" job submit --address="$ADDRESS" --no-wait --runtime-env="$RUNTIME_ENV" \
  --working-dir "$VERL" --submission-id "$submission_id" \
  -- "$ENV_BIN/python" -m verl.trainer.main_ppo \
  algorithm.use_kl_in_reward=False \
  data.train_files=$ROOT/replication/data/countdown_train.parquet \
  data.val_files=$ROOT/replication/data/countdown_test.parquet \
  data.train_batch_size=192 \
  data.val_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.total_training_steps=1024 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  reward_model.reward_manager=naive \
  custom_reward_function.path="$REWARD_FILE" \
  custom_reward_function.name=compute_score \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$project" \
  trainer.experiment_name="$run_name" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=5 \
  trainer.total_epochs=10 \
  trainer.total_training_steps=520 \
  trainer.default_local_dir="$checkpoint_dir" \
  trainer.resume_mode=auto \
  +data.seed="$seed" \
  +actor_rollout_ref.rollout.seed="$seed" \
  "${recipe_args[@]}" >"$submit_log" 2>&1

if ! grep -q "Job '.*' submitted successfully" "$submit_log"; then
  echo "Submission failed; inspect $submit_log" >&2
  exit 5
fi
printf '%s\n' "$submission_id"
