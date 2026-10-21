#!/usr/bin/env bash
# Shared settings for the matched DAPO ablation. Sourced by the rung scripts; do not run directly.
#
# Every value below is copied from the recorded wandb config of the existing DS run
# (DAPO_new/f8h77wol, Qwen2.5-1.5B-G8-acc-filtering-tokenlevel-bsz-32-0910), which is rung 3.
# A rung script sets only: EXP_NAME, CLIP_HIGH, FILTER, GEN_BSZ (and optionally MAX_RESP).
#
# Shared setup (all rungs):  32 prompts trained per step, 8 responses each, 8 actor optimizer updates per step
#   (256 response sequences / response mini-batch 32), 4096-token response cap, Sep-2025 reward (0.9 + 0.1 format), KL off, entropy 0,
#   lr 1e-6 with 10 warmup steps, constant after, a fixed requested horizon, token-mean loss.
set -xeuo pipefail

PROJECT=${PROJECT:-DAPO_ablation}
# Optional second/third training seed. verl 0.2 has two seeds: data.seed (prompt shuffling, default 1) and
# actor_rollout_ref.rollout.seed (vLLM sampling, default 0); the original runs set neither. SEED sets both.
SEED=${SEED:-}
SEED_ARGS=""
if [ -n "${SEED}" ]; then SEED_ARGS="+data.seed=${SEED} +actor_rollout_ref.rollout.seed=${SEED}"; EXP_NAME="${EXP_NAME}-seed${SEED}"; fi
# Collapse screen / other variants (defaults = the full-length runs, unchanged):
TOTAL_STEPS=${TOTAL_STEPS:-520}   # training steps (the lr schedule is constant after warmup, so stopping early = a prefix)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}   # increase only to keep the predeclared step horizon reachable after DS resampling
SAVE_FREQ=${SAVE_FREQ:-50}
KL_LOSS=${KL_LOSS:-False}         # KL loss to the reference policy (low_var_kl)
KL_COEF=${KL_COEF:-0.0}
NAME_SUFFIX=${NAME_SUFFIX:-}      # e.g. -s100 for the 100-step collapse screen
EXP_NAME="${EXP_NAME}${NAME_SUFFIX}"
# Training environment: verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1, transformers 4.51.1 -- the same
# versions the Sep-2025 runs recorded; the original `verl` env has since been downgraded to torch 2.0.1).
# The verl *code* comes from /home/lian/verl (ray working dir), as in the original runs.
ENV_BIN=/home/lian/miniconda3/envs/verl-fork/bin
# Base model: the original local copy (/home/lian/models/qwen2.5-1.5b-instruct) no longer exists; this is the
# Hugging Face release Qwen/Qwen2.5-1.5B-Instruct (revision 989aa79), the same checkpoint used as "base" in eval.
MODEL_PATH=/home/lian/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
MAX_RESP=${MAX_RESP:-4096}
CKPTS_DIR=/mnt/sdb/verl/${PROJECT}/${EXP_NAME}
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
RUNTIME_ENV=/home/lian/verl/verl/trainer/runtime_env.yaml
REWARD_FILE=${REWARD_FILE:-/home/lian/AI_PPO_2026/ablation/reward_sep2025.py}
ADV_EST=${ADV_EST:-grpo}        # option D: reinforce_plus_plus_baseline (group mean subtracted, then batch whitening)
NORM_STD=${NORM_STD:-True}      # option B sets False (advantage = score - group mean, no division by group std)
OVERLONG=${OVERLONG:-False}     # option A2: DAPO soft overlong punishment (reward manager, linear penalty)
OVERLONG_LEN=${OVERLONG_LEN:-0}
OVERLONG_PENALTY=${OVERLONG_PENALTY:-0.0}
EXTRA_ARGS=${EXTRA_ARGS:-}      # option C: +algorithm.mask_uninformative_groups=True

cd /home/lian/verl   # same working directory (and verl 0.2.0.dev code) as the original DAPO runs

${ENV_BIN}/ray job submit --address="${RAY_ADDRESS}" --no-wait --runtime-env="${RUNTIME_ENV}" \
  --working-dir /home/lian/verl --submission-id "${EXP_NAME}-$(date +%Y%m%d-%H%M%S)" \
  -- ${ENV_BIN}/python -m recipe.dapo.src.main_dapo \
  data.train_files=/home/lian/AI_PPO_2026/replication/data/countdown_train.parquet \
  data.val_files=/home/lian/AI_PPO_2026/replication/data/countdown_test.parquet \
  data.prompt_key=prompt \
  data.truncation=left \
  data.max_prompt_length=2048 \
  data.max_response_length=${MAX_RESP} \
  data.train_batch_size=32 \
  data.gen_batch_size=${GEN_BSZ} \
  actor_rollout_ref.rollout.n=8 \
  algorithm.adv_estimator=${ADV_EST} \
  algorithm.norm_adv_by_std_in_grpo=${NORM_STD} \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=${FILTER} \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=-1 \
  actor_rollout_ref.actor.use_kl_loss=${KL_LOSS} \
  actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
  actor_rollout_ref.actor.entropy_coeff=0 \
  +actor_rollout_ref.actor.log_entropy=True \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.optim.total_training_steps=${TOTAL_STEPS} \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2048 + MAX_RESP)) \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((2048 + MAX_RESP)) \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((2048 + MAX_RESP)) \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((2048 + MAX_RESP)) \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  reward_model.reward_manager=dapo \
  reward_model.overlong_buffer.enable=${OVERLONG} \
  reward_model.overlong_buffer.len=${OVERLONG_LEN} \
  reward_model.overlong_buffer.penalty_factor=${OVERLONG_PENALTY} \
  reward_model.overlong_buffer.log=${OVERLONG} \
  custom_reward_function.path=${REWARD_FILE} \
  custom_reward_function.name=compute_score \
  trainer.logger=['console','wandb'] \
  trainer.project_name=${PROJECT} \
  trainer.experiment_name=${EXP_NAME} \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.test_freq=5 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.default_local_dir=${CKPTS_DIR} \
  trainer.resume_mode=auto ${SEED_ARGS} ${EXTRA_ARGS}
