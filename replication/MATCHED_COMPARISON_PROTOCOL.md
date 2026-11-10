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

## Seed extension (frozen 27 October 2026, before launching any seed-10--13 run)

Motivation: with three seeds per method, no between-method difference in the early-short-response
rate can reach p < 0.05 (even 3/3 versus 0/3 gives one-sided Fisher p = 0.05). The extension adds
four seeds per method so that each method has seven.

- New seeds: 10, 11, 12, 13 for PPO, GRPO, and DAPO-style (12 runs). Data seed and rollout seed are
  again paired within each run.
- Everything else is unchanged from the frozen protocol above: the same machine, GPUs, software
  environment, verl commit, launch scripts, reward, data, hyperparameters, 300-step horizon,
  checkpoints, and validation schedule. The only script changes are accepting seeds 10--13 and a new
  queue order.
- Execution order (method order rotated across seeds):
  1. PPO 10, 2. GRPO 10, 3. DAPO-style 10,
  4. GRPO 11, 5. DAPO-style 11, 6. PPO 11,
  7. DAPO-style 12, 8. PPO 12, 9. GRPO 12,
  10. PPO 13, 11. DAPO-style 13, 12. GRPO 13.
- Endpoints are unchanged: the early-short-response criterion (mean response length below 100 tokens
  over steps 90--100) and the step-300 checkpoint for in-domain and downstream evaluation.
- No run is stopped early, dropped, or replaced; a failed run stays in the denominator. A startup
  failure before step 1 is repaired, documented here, and rerun, as for seed 7.
- Primary analysis, fixed now: the early-short-response rate of DAPO-style versus GRPO over seeds
  7--13 (7 versus 7), two-sided Fisher exact test. This contrast isolates asymmetric clipping. PPO
  versus GRPO and DAPO-style versus PPO are secondary and reported with the same test, without
  multiplicity correction; exact 95% Clopper--Pearson intervals are reported for each method's rate.
- Seeds 7--9 and 10--13 are pooled; per-seed results are reported for all 21 runs. Results for
  seeds 7--9 are already known, so the pooled analysis is not fully prospective; seeds 10--13 alone are
  also reported as a prospective replication.
- The prompt-conditional analysis (standard versus think prompt on GSM8K, persistent-short versus
  long-response checkpoints) is repeated on all 21 step-300 checkpoints with an exact two-sided
  Mann--Whitney test.

### Deviation: early-short threshold rounding (28 October 2026, after six extension runs)

GRPO seed 10 averaged 100.05 tokens over steps 90--100, so it narrowly missed the frozen "below 100
tokens" rule, although it stayed below 100 tokens for most of training after step 30 and ended at 66
tokens. After seeing this value, the criterion was changed to: the window mean, rounded to the nearest
token, is at most 100 tokens. The change is post hoc and is reported as such. It is applied to every run
(all comparison seeds, the extension, and the 22-run screen) and reclassifies only GRPO seed 10; the
nearest other run is 132 tokens. It works against the primary hypothesis because it adds an
early-short run to GRPO, the comparator for DAPO-style training. Results are also reported under the
original "below 100 tokens" rule as a sensitivity analysis.

### Addition: GSM8K prompt variants on the extension checkpoints (30 October 2026, before any extension checkpoint was evaluated downstream)

The 12 extension step-300 checkpoints get the same downstream evaluation as seeds 7--9: all four
benchmarks with the standard and think prompts. They also get the five GSM8K think-prompt variants
already run on the nine seed 7--9 checkpoints (`eval/eval_gsm8k_prompt_ablation.py`: think_no_example,
think_example17, think_no_prefill, clean_tags; and `eval/eval_gsm8k_prompt_open_tag_ablation.py`),
with the same 8,192-token limit, greedy decoding, and scorers. The variants are not run on the other
benchmarks: their wording assumes a numeric answer.
