# Matched DAPO ablation (draft, not launched)

Purpose: make the dynamic-sampling (DS) comparison controlled, and find which DAPO ingredient prevents the
think-prompt collapse seen in low-KL GRPO. All rungs share one setup; each rung adds one ingredient.

| Rung | Script | Clip (low / high) | Dynamic sampling | Prompts generated / trained per step | Updates per step | Status |
|---|---|---|---|---|---|---|
| 1 | `rung1_grpo_clip0.2_noDS.sh` | 0.2 / 0.2 | off | 32 / 32 | 8 | to train |
| 2 | `rung2_clip0.28_noDS.sh` | 0.2 / 0.28 | off | 32 / 32 | 8 | to train |
| 3 | existing run `DAPO_new/f8h77wol` (DS) | 0.2 / 0.28 | on | 192+ / 32 (first 32 non-tied groups) | 8 | done; checkpoint `global_step_300` only |
| 1b (optional) | `rung1b_grpo_clip0.2_noDS_resp2048.sh` | 0.2 / 0.2 | off | 32 / 32 | 8 | response cap 2048 instead of 4096 |

Comparisons:
- **Rung 2 vs. rung 3** = dynamic sampling only (same trained prompts, responses and actor updates per step). This
  replaces the confounded DS vs. old no-DS comparison (old no-DS trained 192 prompts and 48 actor updates per step).
- **Rung 1 vs. rung 2** = clip-higher only.
- **Rung 1 vs. rung 1b** = response cap only (4096 vs. 2048, the PPO/GRPO value).

## Shared setup (`common.sh`)

Every value is copied from the recorded wandb config of the DS run (f8h77wol), except the rung-specific
settings above and three changes needed for a fair ladder:
- `data.gen_batch_size = 32` for rungs 1/2 (the DS run oversampled 192).
- `trainer.total_training_steps = 520`: with 32 prompts per step, "2 epochs" would be about 3,100 steps.
  The optimizer schedule already used 520 (constant lr after 10 warmup steps).
- `trainer.save_freq = 50` (the DS run saved only step 300). Evaluate all rungs at step 300 to match rung 3.

Unchanged from f8h77wol: Qwen2.5-1.5B-Instruct, Countdown train/test parquet, n = 8, response cap 4096,
prompt cap 2048 (truncation left), GRPO advantages with std normalization, KL loss off, KL in reward off,
entropy 0, clip_c 10, token-mean loss, grad clip 1.0, 1 PPO epoch, mini-batch 32, lr 1e-6, warmup 10,
weight decay 0.1, rollout temperature 1.0 / top-p 1.0, validation top-p 0.7 sampled, `dapo` reward manager,
overlong buffer off, sequence parallel 4, dynamic batch size, FSDP param/optimizer offload, vLLM TP 4.
Verl expands the 32 prompt groups into 256 response sequences before the actor update; mini-batch 32 therefore yields 8 actor optimizer steps per outer training step, not one.

Reward: `reward_sep2025.py` wraps `countdown.compute_score_sep2025` (0.9 correct + 0.1 format; returns `acc`
for the DS filter). This is the reward the Sep-2025 DAPO runs used; the current default `countdown.compute_score`
is a later 0.5/0.5 version and must not be used here. It prints every scored response, so per-step p0/p1
(all-wrong / all-correct group fractions) can be recovered with `../parse_ds_log.py`.

## Before launching

1. **Environment.** The `verl` conda env currently has PyTorch 2.0.1 and vLLM fails to import
   (found during evaluation). Start the ray cluster from an environment that can train (the one the
   Sep-2025 runs used, or repair `verl`). The config composes with both `verl` and `verl-fork` Python
   (checked with `--cfg job`), but that does not test the GPU stack.
2. Start ray (`ray start --head`), then run e.g. `bash rung2_clip0.28_noDS.sh` (submits with `ray job submit --no-wait`).
3. Checkpoints go to `/mnt/sdb/verl/DAPO_ablation/<experiment name>`; wandb project `DAPO_ablation`.

Suggested order: rung 2 first (needed to keep the DS result), then rung 1, then second seeds of rungs 2 and 3
(rung 3 would be a re-run of f8h77wol with `trainer.save_freq=50`). Expected time: the DS run spent about
425 s per step generating for 192 prompts; with 32 prompts per step each rung should take roughly 10-20 hours
for 520 steps on the 4 A6000s (estimate, not measured).

## After training

Register each checkpoint in `../eval/run_registry.py`, `../eval/merge_checkpoints.sh` and `../eval/common.py`,
then `MODEL=<key> ../eval/run_model.sh` (or `run_queue.sh`); `RESULTS.md` is regenerated automatically.

## Launch log

| Rung | Started | Ray submission id | wandb project / checkpoints | Status |
|---|---|---|---|---|
| 2 | 2026-09-29 09:27 | `rung2-clip0.28-noDS-gen32-train32-n8-resp4096-20260929-092735` | `DAPO_ablation` / `/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096` | finished 14:13 (520 steps) |
| 1 | 2026-09-29 15:15 | `rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096-20260929-151515` | `DAPO_ablation` / `/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096` | running |

Environment actually used (checked with a test job before launching):
- Ray cluster started from the `verl-fork` conda env: torch 2.6.0+cu124, vLLM 0.8.5.post1, transformers 4.51.1
  (the Sep-2025 runs recorded torch 2.6.0, vLLM 0.8.5.post1, transformers 4.51.1; ray 2.48.0 vs. 2.43.0).
- verl code: `/home/lian/verl` (0.2.0.dev) uploaded as the ray working directory; driver and GPU workers both
  import it from there, not the `verl_fork` package installed in that env.
- Base model: Hugging Face `Qwen/Qwen2.5-1.5B-Instruct` revision 989aa79 (the original local copy was deleted).
- First step checked: 256 sequences trained (32 prompts x 8), 1 generation round, step-0 Countdown validation
  accuracy 0.030 (same as the original DAPO runs), about 60 s per step at the start.

Monitor: `/home/lian/miniconda3/envs/verl-fork/bin/ray job logs <submission id> --address=http://localhost:8265`,
the ray dashboard at http://127.0.0.1:8265, or the wandb project `DAPO_ablation`.
Stop: `/home/lian/miniconda3/envs/verl-fork/bin/ray job stop <submission id> --address=http://localhost:8265`.

## Training seeds

verl 0.2 has two random seeds: `data.seed` (shuffling of training prompts, `verl/trainer/main_ppo.py:237`, default 1)
and `actor_rollout_ref.rollout.seed` (vLLM sampling, `vllm_rollout_spmd.py:146`, default 0). The original runs set
neither. `SEED=<k>` in `common.sh` sets both to k (passed as `+data.seed=k +actor_rollout_ref.rollout.seed=k`) and
appends `-seed<k>` to the experiment name, checkpoint folder and submission id. `SEED` unset = original seeds.

- `rung3_DS_gen192.sh`: the DS recipe (identical to DAPO_new/f8h77wol except `save_freq=50`), for DS seeds.
- Planned: rung 2 seed 2, then DS (rung 3) seed 2. Chained by `chain_rung1eval_seeds.sh`
  (after evaluating rung 1). The DS run generates 192+ prompts per step; the original took about 75 hours.

## Options A, B, C: getting DS's benefit without resampling

Why DS works here (`analysis_why_ds.py` on rung 2's printed rewards): in rung 2's first 25 steps, 74-87% of the
training signal came from groups where all 8 answers were wrong and scores differed only by the 0.1 format credit;
std normalisation turns those 0.1 differences into full-size advantages, rewarding "write any equation", which short
answers do most reliably. DS filters on accuracy and so drops these groups. Each option below is rung 2 (no DS,
32 prompts generated and trained per step) plus one change:

| Option | Script | Change | Extra generation |
|---|---|---|---|
| A | `rungA.sh` | correctness-only reward `reward_acc_only.py` (1 if correct, else 0) | none |
| B | `rungB.sh` | `algorithm.norm_adv_by_std_in_grpo=False` (advantage = score - group mean) | none |
| C | `rungC.sh` | `+algorithm.mask_uninformative_groups=True`: groups whose 8 answers are all right or all wrong are masked out of the actor loss (and its token-mean denominator) instead of resampled | none |

Option C is a trainer change in `/home/lian/verl/recipe/dapo/src/dapo_ray_trainer.py` (after `compute_advantage`),
**off by default** so all other runs are unaffected; the original file is saved as
`dapo_ray_trainer.py.orig_before_mask`. It sets `loss_mask` and `meta_info["multi_turn"]`, which makes
`dp_actor.update_policy` use `loss_mask` instead of the attention mask (checked on a toy batch).

Queue (`chain_abc_then_dsseed.sh`): rung 2 seed 2 (running) -> A -> B -> C -> evaluate seed 2 and A/B/C
(step 300 all benchmarks, step 520 GSM8K+MATH) -> DS recipe seed 2 (about 3 days).

### Option D and pass@k (added 2026-10-01, from the literature review in `../literature/related_work.md`)

| Option | Script | Change |
|---|---|---|
| D | `rungD.sh` | `algorithm.adv_estimator=reinforce_plus_plus_baseline`: subtract each group's mean reward, then whiten over the whole batch (verl `core_algos.py`), instead of dividing by each group's std. Close to "Lite PPO" (arXiv:2508.08221). No code change. |

pass@k (`../eval/eval_passk.py`, `../eval/run_passk_jobs.sh`): MATH-500 x 64 samples and GSM8K x 32 samples,
standard prompts, temperature 0.6, top-p 0.95, up to 4,096 tokens, unbiased pass@k for k = 1..64, strict and
flexible scoring. Models: base, DS, old no-DS, rung 2, options A and D (step 300), GRPO KL 0.04 and KL off,
PPO lr 1e-6 (step 520). Results go to `../eval/results_passk/` and `RESULTS.md` section 3.12.

Queue (`chain_bcdA1_passk_dsseed.sh`, replaces the earlier chains): option B (running) -> C -> D -> A1 ->
evaluate rung 2 seed 2, A-D and A1 -> pass@k (incl. A1) -> DS recipe seed 2.

### Option A1 (added 2026-10-02)

Option A was stable to about step 400, then collapsed (Countdown 69% at step 300 -> 28% at step 500) while
responses hitting the 4,096-token cap rose from 25% to 59%. Under the correctness-only reward a truncated response
scores 0, like a wrong one, so nothing pushes against rambling.

| Option | Script | Change |
|---|---|---|
| A1 | `rungA1.sh` | option A + DAPO "overlong filtering": responses that hit the cap are removed from the actor loss (`+algorithm.mask_truncated=True`; trainer change in `dapo_ray_trainer.py`, off by default, tested on a toy batch) |
| A2 (prepared, not queued) | `rungA2.sh` | option A + DAPO soft overlong punishment (built into verl: `reward_model.overlong_buffer`, len 1024, factor 1.0) |

## Collapse screen (started 2026-10-08)

Every recipe run with two seeds collapsed on one (DS seed 2, no-DS seed 1, A1 seed 2), always within ~25 steps.
To estimate collapse rates, `chain_collapse_screen.sh` runs 100-step versions (`TOTAL_STEPS=100 SAVE_FREQ=100
NAME_SUFFIX=-s100`) of 4 recipes x seeds 3-6 (16 runs), seed by seed: no DS (rung 2), A1, A1 + KL loss 0.01
(`rungA1KL.sh`, candidate fix from the GRPO KL sweep), DS (rung 3). Seeds 1-2 come from the full-length runs.
`analysis_collapse_screen.py` classifies each run (collapsed = mean response length over steps 90-100 < 100
tokens) and writes `collapse_screen.md` / `.csv`. No benchmark evaluation needed.
