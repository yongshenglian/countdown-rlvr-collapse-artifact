# Experiment B: PPO, GRPO, and DAPO-Style Training With an Active Clip

Protocol frozen: 29 October 2026, before launching any run in this experiment.

## Why this experiment exists

The matched comparison (MATCHED_COMPARISON_PROTOCOL.md) was intended to make eight actor updates per step.
verl counts `actor.ppo_mini_batch_size` in prompts and multiplies it by `rollout.n`
(`verl/workers/fsdp_workers.py:127`), so mini-batch 32 meant 32 x 8 = 256 responses, which is the whole
batch. Every run therefore made one on-policy update per step, the importance ratio was 1, and the clip
never bound: `actor/pg_clipfrac` and `actor/ppo_kl` are 0.000 at every logged step of every matched,
extension, and screen run. Same-seed GRPO and DAPO-style runs have identical response lengths at steps 1
and 2 and diverge from step 3 through GPU nondeterminism. The "GRPO" versus "DAPO-style" contrast
therefore compared two sets of runs of the same algorithm.

This experiment makes the clip active and reruns the three-method comparison as it was intended: does
raising the upper clip bound from 0.2 to 0.28 change response-length collapse under group-relative
training, and does a learned critic (PPO) differ from group-relative advantages?

## Design

- Three arms: GRPO (group-relative advantages, clip 0.2 lower / 0.2 upper), DAPO-style (the same with
  0.2 lower / 0.28 upper), and PPO (GAE with a learned critic, clip 0.2 / 0.2).
- Seeds 7--16 in each arm (30 runs). Data seed and rollout seed are paired within each run.
- The only change from the matched group-relative setup is `ppo_mini_batch_size=4` prompts
  (32 responses). Each step's 256 responses give 8 sequential actor updates, so updates 2--8 are
  off-policy and the clip can bind. Everything else is unchanged: model, data, reward (0.9 correct +
  0.1 format), no dynamic sampling, no KL, entropy coefficient 0, token-mean loss, actor learning rate
  1e-6 with 10 warmup steps (the scheduler steps once per training step), 4,096-token response cap,
  rollout and validation sampling, 300 steps, checkpoints every 50 steps, validation every 5 steps,
  same machine, environment, and verl code. PPO also uses critic mini-batch 4 prompts (8 critic
  updates per step); its other settings are copied verbatim from the matched PPO launcher.
- Launchers: `ablation/b_grpo_clip0.2_mb4.sh`, `ablation/b_grpo_clip0.28_mb4.sh` (via `ablation/common.sh`
  with `MINI_BSZ=4`) and `replication/submit_active_clip_ppo.sh`. Queue: `replication/run_active_clip_queue.sh`. W&B project `Active_Clip_Comparison`.
- Run order: the three arms rotate by seed (seed 7: GRPO, DAPO-style, PPO; seed 8: DAPO-style, PPO, GRPO;
  seed 9: PPO, GRPO, DAPO-style; repeating).
- Logging-only patch `replication/patches/experiment_b_logging.patch`, applied after the
  seed-extension queue finishes. It adds `actor/pg_clipfrac_up_pct` (tokens with positive advantage and
  ratio above 1 + upper bound), `actor/pg_clipfrac_down_pct` (negative advantage, ratio below
  1 - lower bound), and `actor/ratio_max`. The loss is unchanged.

## Gate before the frozen runs

A 10-step smoke run per arm (seed 99, W&B project `Active_Clip_Smoke`, not analysed) must finish
without error and show clipped tokens (upper + lower percent > 0) in all three arms. Otherwise the queue stops, and the
problem is documented and fixed before any frozen run starts.

## Outcomes and analysis (fixed now)

Primary: mean response length over steps 90--100, compared with exact two-sided Mann--Whitney
(Wilcoxon rank-sum) tests, 10 versus 10, for two contrasts with Holm correction:
1. Clip: DAPO-style (0.28) versus GRPO (0.2). Only the upper bound differs.
2. Advantage estimator: PPO versus GRPO. Both use symmetric 0.2 clipping. A continuous outcome is primary because the
binary early-short rate may sit near its ceiling (10/12 single-update group-relative runs were early-short).

Secondary:
- Early-short rate (rounded window mean at most 100 tokens) for both contrasts, two-sided Fisher exact
  test, with exact Clopper--Pearson intervals for each arm.
- Step-300 Countdown validation accuracy (Mann--Whitney).
- First step at which the per-step mean response length falls below 100 tokens.
- Upper and lower clip fractions over steps 1--30 and 90--100, by arm and by eventual outcome.

Exploratory (not a controlled contrast): eight-update versus single-update runs of the same method
(seeds 7--13 of the matched comparison and extension). Eight updates also mean eight optimizer steps
per training step, so this contrast confounds clipping with update count.

Downstream: every step-300 checkpoint is evaluated on all four benchmarks (GSM8K, MATH, BBH,
MMLU-Pro) with the standard and think prompts, strict and flexible scoring, using the same complete test
sets, scorers, and greedy decoding as the main evaluation. The prompt-conditional analysis (think-prompt
GSM8K accuracy of persistent-short versus long-response checkpoints, exact Mann--Whitney) is repeated
within experiment B.
Each checkpoint also gets the five GSM8K think-prompt variants already run on the matched checkpoints
(`eval/eval_gsm8k_prompt_ablation.py`: think_no_example, think_example17, think_no_prefill,
clean_tags; and `eval/eval_gsm8k_prompt_open_tag_ablation.py`), with the same 8,192-token limit,
greedy decoding, and scorers. They run on GSM8K only, because their wording assumes a numeric answer.

## Rules

- No run is stopped early, dropped, or replaced; a failed run stays in the denominator. A startup failure
  before step 1 is repaired, documented here, and rerun.
- All 30 runs are reported individually. No interim analysis changes the design.

## Data retention (added before launch)

Experiment B keeps everything for post-hoc analysis, on /mnt/sda/data/verl/<project>/<run>:
- Full checkpoints (model, optimizer, and extra state) every 50 steps; nothing is pruned.
- `dumps/steps/step_NNNN.pt` for every training step: prompt and response token ids, response lengths,
  per-token old log-probs, advantages, returns, token-level scores and rewards, critic values (PPO), and
  the non-tensor batch fields (uid, reward extras, ground truth). Written by `verl/utils/step_dump.py`.
- `dumps/val/NNNN.jsonl` for every validation pass: inputs, generations, and scores.
- The Ray job log and W&B history, as for the other runs.

The logging patch is `replication/patches/experiment_b_logging.patch` (clip-split metrics and the dumps);
it does not change the loss, sampling, or data. The smoke gate also requires that step and validation
dumps are written.

The finished matched-comparison and extension runs (seeds 7--12) had their optimizer-state shards
deleted on 29 October 2026 to free disk space before /mnt/sda was used; their model weights are intact.
