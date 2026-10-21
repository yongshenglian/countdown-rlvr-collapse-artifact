# Prospective Matched PPO/GRPO/DAPO-Style Comparison

Protocol frozen: 14 October 2026, before launching any seed-7--9 run in this comparison.

## Objective

Compare PPO, symmetric-clip GRPO, and an asymmetric-clip DAPO-style variant under matched actor-side
training exposure. This experiment is separate from the historical unequal-recipe checkpoints and from
the dynamic-sampling collapse screen.

The design matches sampled and retained responses, actor mini-batches, actor optimizer updates, actor
optimizer settings, reward, data, prompts, model, seeds, software environment, training horizon, and
evaluation checkpoint. It does not claim equal FLOPs: PPO additionally trains a critic. Dynamic sampling
is excluded from the primary comparison because rejection and resampling make generated-response compute
variable; it remains a separate ablation.

## Frozen environment and inputs

- Four NVIDIA RTX A6000 GPUs on one node.
- Python 3.10.0, PyTorch 2.6.0+cu124, vLLM 0.8.5.post1, Transformers 4.51.1, Ray 2.48.0.
- Verl source commit: `6abe50ab0a84237d9d8a733d365c3b983bcf647d`.
- Qwen2.5-1.5B-Instruct snapshot: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Countdown train SHA-256: `a8508c1531cac14162327f97033572949905bb5d24a6993666813fbde9d12f8d`.
- Countdown test SHA-256: `63b631a38143f0d79090919be96f40848178f1befcb9eb100b1f46d6d22b6898`.
- Frozen scalar reward: correct 0.9 plus 0.1 full-format credit; incorrect responses receive only
  the 0.1 format credit when valid; no equation and invalid format receive 0.

## Matched settings

| Setting | Frozen value |
|---|---:|
| Data seed / rollout seed | 7, 8, 9 (paired within each run) |
| Prompt groups per outer step | 32 |
| Responses per prompt | 8 |
| Response trajectories per outer step | 256 |
| Actor response mini-batch | 32 |
| Actor epochs | 1 |
| Actor optimizer updates per outer step | 8 |
| Actor learning rate | 1e-6 |
| Actor warmup | 10 steps, constant thereafter |
| Actor weight decay / gradient clipping | 0.1 / 1.0 |
| KL loss / KL reward / entropy coefficient | off / off / 0 |
| Loss aggregation | global token mean |
| Prompt / response cap | 2,048 / 4,096 tokens |
| Rollout sampling | temperature 1.0, top-p 1.0, top-k -1 |
| Validation sampling | temperature 1.0, top-p 0.7, top-k -1, n=1 |
| Outer training horizon | exactly 300 steps |
| Checkpoints | steps 50, 100, 150, 200, 250, 300 |
| Validation frequency | every 5 steps |

## Algorithm-defining differences

- PPO: GAE with gamma 1 and lambda 1, learned critic, symmetric actor clipping 0.2/0.2.
  The critic uses learning rate 1e-5, 10 warmup steps, weight decay 0.1, one epoch, and the same
  response mini-batch size 32. Critic compute is reported separately.
- GRPO: group-relative advantages with sample-standard-deviation normalization, no critic,
  symmetric clipping 0.2/0.2.
- DAPO-style: the same group-relative estimator and no dynamic sampling, but asymmetric clipping
  0.2/0.28. This isolates the asymmetric actor clip from symmetric GRPO under otherwise matched settings.

These labels describe the implemented recipes. The primary comparison does not represent every component
of published DAPO because dynamic sampling and overlong shaping are intentionally disabled.

## Fixed endpoints and reporting

- No run stops because of validation accuracy, response length, or downstream performance.
- Step 100 is the fixed early-collapse endpoint. Early short-response collapse uses the existing mean
  response length below 100 tokens over steps 90--100 definition.
- Step 300 is the fixed final in-domain and downstream comparison checkpoint.
- A failed or numerically invalid run remains in the denominator and is not replaced.
- All seeds are reported individually. Means are descriptive with only three seeds.
- Generated responses, retained responses, actor updates, critic updates, wall time, and GPU-hours are
  reported separately.
- Downstream evaluation uses the same standard and think prompts, strict and format-tolerant scorers,
  and complete benchmark sets used elsewhere in the artifact.

## Execution order

Runs execute serially on all four GPUs in a balanced recipe order:

1. PPO seed 7
2. GRPO seed 7
3. DAPO-style seed 7
4. GRPO seed 8
5. DAPO-style seed 8
6. PPO seed 8
7. DAPO-style seed 9
8. PPO seed 9
9. GRPO seed 9

The queue stops on a runtime failure so that any repair can be documented and applied consistently.

## Execution notes

- The first PPO seed-7 submission (`matched-ppo-n8-b32-mini32-seed7-step300-20261014-154327`)
  stopped during its pre-training validation, before step 1. The DAPO reward-manager options had been
  placed under `reward_model.overlong_buffer`, whereas the frozen verl revision passes manager-specific
  constructor arguments from `reward_model.reward_kwargs`. The resulting null constructor argument caused
  `AttributeError: 'NoneType' object has no attribute 'enable'`. No training result was produced, so this
  startup attempt is not an experimental run and is not included in the denominator.
- The first path correction reached the training reward manager but the generic trainer does not forward
  `reward_kwargs` to its validation reward manager. The retry
  (`matched-ppo-n8-b32-mini32-seed7-step300-20261014-160228`) therefore stopped at the same
  pre-training validation point and likewise produced no experimental result.
- PPO now uses VERL's `naive` reward-manager wrapper, the generic PPO trainer's supported path. With
  overlong shaping disabled, its reward tensor and returned score diagnostics are identical to the DAPO
  wrapper used by the DAPO-specific trainer: both invoke the same frozen `compute_score` function and place
  its scalar at the last valid response token. The GRPO and DAPO-style launchers retain their native DAPO
  wrapper with `overlong_buffer.enable=False`. This compatibility repair changes neither the reward function
  nor any frozen data, algorithm, seed, hyperparameter, endpoint, or run order.
