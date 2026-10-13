# Downstream evaluation of Countdown-trained checkpoints

Generated 2026-10-12 by `make_results_md.py` from the saved responses. Do not edit by hand; rerun the script after adding models.

## 0. Summary

All models ranked by average strict accuracy (%) over the 4 benchmarks x 2 prompts, 8192 tokens, greedy. Labels are built from each run's recorded settings (section 2).

| rank | key | model (settings, step) | GSM8K standard | GSM8K think | MATH standard | MATH think | BBH standard | BBH think | MMLU-Pro standard | MMLU-Pro think | avg (8) | avg std | avg think |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noDS_step300 | DAPO w/o DS, n=8 (step 300) | 69.9 | 75.7 | 54.9 | 49.5 | 34.4 | 41.5 | 32.9 | 29.7 | 48.6 | 48.0 | 49.1 |
| 2 | rungA_step300 | Ablation rung 2 + correctness-only reward: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 61.6 | 72.2 | 55.7 | 52.3 | 35.8 | 41.2 | 33.0 | 29.7 | 47.7 | 46.5 | 48.8 |
| 3 | rungA1_step300 | Ablation rung 2 + correctness-only reward + overlong filtering: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 64.3 | 74.1 | 55.4 | 51.8 | 35.2 | 40.8 | 32.8 | 26.3 | 47.6 | 46.9 | 48.3 |
| 4 | GRPO_n2_kl4e-2_step520 | GRPO n=2, KL 0.04 (step 520) | 65.1 | 69.4 | 55.9 | 52.2 | 34.6 | 38.7 | 32.7 | 29.0 | 47.2 | 47.1 | 47.3 |
| 5 | DS_step300 | DAPO w/ DS, n=8 (step 300) | 66.1 | 74.4 | 55.1 | 46.5 | 35.1 | 40.2 | 33.1 | 26.1 | 47.1 | 47.4 | 46.8 |
| 6 | rungB_step300 | Ablation rung 2 + no std normalisation: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 61.1 | 71.6 | 55.5 | 48.7 | 35.5 | 40.9 | 33.9 | 29.1 | 47.0 | 46.5 | 47.6 |
| 7 | GRPO_n2_kl1e-2_step250 | GRPO n=2, KL 0.01 (step 250) | 56.5 | 72.9 | 56.0 | 52.5 | 34.8 | 40.5 | 33.1 | 29.6 | 47.0 | 45.1 | 48.9 |
| 8 | rungC_step300 | Ablation rung 2 + masked uninformative groups: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 55.0 | 75.0 | 55.6 | 52.6 | 35.1 | 40.6 | 33.1 | 26.5 | 46.7 | 44.7 | 48.7 |
| 9 | GRPO_n2_kl4e-2_step250 | GRPO n=2, KL 0.04 (step 250) | 69.5 | 57.7 | 55.3 | 51.7 | 35.0 | 37.5 | 33.9 | 29.7 | 46.3 | 48.4 | 44.1 |
| 10 | rungD_step300 | Ablation rung 2 + batch-level advantage normalisation: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 59.4 | 70.1 | 56.1 | 45.8 | 35.1 | 40.4 | 33.2 | 26.9 | 45.9 | 46.0 | 45.8 |
| 11 | rung2s2_step300 | Ablation rung 2, seed 2: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 47.1 | 70.6 | 55.0 | 47.4 | 34.4 | 40.7 | 33.3 | 28.8 | 44.7 | 42.5 | 46.9 |
| 12 | PPO_klF_ent0_lr1e-6_step520 | PPO lr 1e-06, KL off, entropy 0, mini 16 (step 520) | 62.2 | 62.4 | 46.2 | 42.3 | 24.3 | 36.4 | 31.4 | 27.0 | 41.5 | 41.0 | 42.0 |
| 13 | base | Base Qwen2.5-1.5B-Instruct | 33.4 | 48.4 | 55.1 | 47.4 | 35.2 | 36.2 | 33.1 | 27.5 | 39.5 | 39.2 | 39.9 |
| 14 | PPO_step400 | PPO lr 1e-06, KL off, entropy 0, mini 64 (step 400) | 40.1 | 50.0 | 53.6 | 28.0 | 35.6 | 36.1 | 33.1 | 25.3 | 37.7 | 40.6 | 34.9 |
| 15 | GRPO_n4_kl1e-6_eqR_step520 | GRPO n=4, KL 1e-06, equal 0.5/0.5 reward (step 520) | 66.4 | 18.7 | 53.8 | 18.4 | 35.2 | 34.1 | 33.7 | 24.2 | 35.5 | 47.3 | 23.8 |
| 16 | GRPO_n8_kl1e-3_step520 | GRPO n=8, KL 0.001 (step 520) | 63.5 | 11.0 | 52.7 | 15.9 | 35.8 | 36.0 | 33.3 | 25.4 | 34.2 | 46.3 | 22.1 |
| 17 | GRPO_n2_kl5e-3_step520 | GRPO n=2, KL 0.005 (step 520) | 56.1 | 11.2 | 53.9 | 16.6 | 35.7 | 33.1 | 33.6 | 25.6 | 33.2 | 44.8 | 21.6 |
| 18 | rung2_step300 | Ablation rung2: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 61.5 | 5.9 | 54.1 | 14.5 | 35.5 | 36.5 | 33.0 | 24.2 | 33.2 | 46.0 | 20.3 |
| 19 | GRPO_n4_kl1e-6_step520 | GRPO n=4, KL 1e-06 (step 520) | 51.1 | 12.0 | 53.4 | 15.5 | 34.3 | 35.0 | 33.4 | 25.4 | 32.5 | 43.0 | 22.0 |
| 20 | GRPO_n2_kl1e-3_step250 | GRPO n=2, KL 0.001 (step 250) | 49.6 | 9.8 | 54.6 | 14.6 | 35.4 | 35.4 | 33.0 | 25.2 | 32.2 | 43.1 | 21.2 |
| 21 | GRPO_n4_kl1e-3_step520 | GRPO n=4, KL 0.001 (step 520) | 50.0 | 10.4 | 52.1 | 15.6 | 34.4 | 35.3 | 33.5 | 25.7 | 32.1 | 42.5 | 21.8 |
| 22 | GRPO_n2_kl1e-3_step520 | GRPO n=2, KL 0.001 (step 520) | 45.3 | 11.0 | 55.2 | 15.0 | 35.2 | 35.7 | 33.6 | 25.0 | 32.0 | 42.3 | 21.7 |
| 23 | GRPO_n2_kl5e-3_step250 | GRPO n=2, KL 0.005 (step 250) | 47.2 | 9.6 | 54.3 | 15.9 | 36.0 | 34.8 | 33.2 | 25.1 | 32.0 | 42.6 | 21.3 |
| 24 | GRPO_n2_noKL_step250 | GRPO n=2, KL off (step 250) | 44.0 | 11.1 | 52.9 | 15.8 | 34.6 | 35.4 | 33.1 | 25.7 | 31.6 | 41.2 | 22.0 |
| 25 | rung1_step300 | Ablation rung1: clip 0.2/0.2, no DS, 32 prompts/step (step 300) | 56.1 | 5.2 | 53.7 | 11.7 | 33.7 | 35.2 | 33.1 | 20.5 | 31.1 | 44.2 | 18.1 |
| 26 | DSs2_step200 | DAPO w/ DS, training seed 2 (collapsed; step 200) | 39.2 | 9.9 | 54.4 | 14.7 | 34.9 | 35.7 | 32.7 | 24.7 | 30.8 | 40.3 | 21.3 |
| 27 | GRPO_n2_noKL_step520 | GRPO n=2, KL off (step 520) | 38.4 | 8.3 | 52.3 | 16.1 | 34.6 | 36.0 | 33.7 | 25.7 | 30.6 | 39.7 | 21.5 |
| 28 | rungA1s2_step300 | Ablation rung 2 + correctness-only reward + overlong filtering, seed 2: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 37.7 | 8.0 | 55.0 | 13.4 | 34.6 | 34.7 | 33.1 | 23.7 | 30.0 | 40.1 | 19.9 |
| 29 | PPO_kl1e-3_ent1e-3_lr3e-6_step520 | PPO lr 3e-06, KL 0.001, entropy 0.001, mini 16 (step 520) | 32.8 | 28.4 | 12.3 | 7.8 | 28.1 | 29.8 | 19.8 | 15.9 | 21.8 | 23.2 | 20.5 |
| 30 | PPO_kl0_ent1e-3_lr3e-6_step520 | PPO lr 3e-06, KL on, coef 0, entropy 0.001, mini 16 (step 520) | 20.4 | 21.6 | 11.9 | 7.5 | 22.9 | 33.8 | 18.5 | 20.5 | 19.6 | 18.4 | 20.8 |
| 31 | PPO_klF_ent0_lr3e-6_step520 | PPO lr 3e-06, KL off, entropy 0, mini 16 (step 520) | 28.7 | 25.3 | 12.1 | 10.7 | 18.0 | 20.9 | 18.6 | 11.4 | 18.2 | 19.3 | 17.1 |
| 32 | PPO_klF_ent1e-3_lr3e-6_step520 | PPO lr 3e-06, KL off, entropy 0.001, mini 16 (step 520) | 1.0 | 1.4 | 2.6 | 2.1 | 5.7 | 25.7 | 1.6 | 13.9 | 6.7 | 2.7 | 10.7 |

**What can be claimed about transfer** (standard prompts, flexible scoring so that answer formatting does not count; section 3.9)

| claim | evidence | strength |
|---|---|---|
| Best recipes slightly improve GSM8K | DAPO w/o DS +3.1 (p = 0.008); GRPO KL 0.04 +2.6 (p = 0.008) | small but significant (a real improvement); one training seed each |
| No measurable change on MATH, BBH, MMLU-Pro | best recipes within +/-1 point (only GRPO KL 0.04 on MMLU-Pro +0.8 is significant); no gain on the easiest MATH levels (base 75-86%) | consistent across models and difficulty levels |
| Some recipes hurt general ability | PPO lr 3e-6: -13 to -53 on MATH/BBH/MMLU-Pro; PPO lr 1e-6: -9 on MATH; weak-KL GRPO: -2 to -3 on MATH | large and clear |

Also supported, but not general reasoning: (1) large in-domain gains, Countdown validation accuracy 3% -> about 75-79% (DAPO); (2) answer-format compliance, GSM8K responses without a final-answer marker 54% -> about 8%; this is formatting, not reasoning; (3) the recipe decides whether general ability is kept (DAPO, GRPO KL >= 0.01) or damaged (PPO at high lr, GRPO KL <= 0.005); (4) the learned behaviour differs by recipe: the think prompt exposes a 20-token no-reasoning shortcut in weak-KL GRPO, linked in the training logs to early entropy collapse, while DAPO keeps reasoning in its training format.

Suggested wording: *RL on Countdown produces large in-domain gains and a small but statistically significant improvement on GSM8K: about 3 points for the best recipes (DAPO w/o DS +3.1, GRPO KL 0.04 +2.6, both p < 0.01), after accounting for answer-format compliance. MATH, BBH and MMLU-Pro are unchanged, and other recipes (high-learning-rate PPO, GRPO with KL <= 0.005) degrade general ability or learn a no-reasoning shortcut.*

Limits: the full downstream evaluations generally use one training seed per setting (McNemar p-values do not include seed variance, so even the +3 GSM8K gain may change with more seeds); only Qwen2.5-1.5B-Instruct and only Countdown are studied, so 'little transfer' should not be generalised beyond this setup. The matched DS comparison is complete but does not remove all environment differences.

**Open items**

- The matched DS/no-DS comparison is complete (`../ablation/README.md`); repeat it in one frozen software environment before making a causal claim about dynamic sampling.
- Full-length downstream replicate training remains incomplete for DS, matched no-DS, strong-KL GRPO, and PPO; the 22-run screen supports only claims about early collapse.
- **Follow-up paper (not this one):** train the same recipes on a math training set (e.g. the MATH or GSM8K training split, or a curated RL math set such as DAPO-Math-17k or DeepScaleR) with the same model, steps and evaluation pipeline, and compare transfer with the Countdown-trained models. Tests whether the small transfer seen here is due to the Countdown data rather than the RL recipes. Needs only a new data path and reward function in the training scripts; run after rung 2 and the seed runs.
- Paper: replace the dynamic-sampling filtered fraction f = 1 - b_train / (n_gen b_gen) with the measured per-step value (`../ds_run_p0_p1_per_step.csv`, column `f_true`); with 192 generated and 32 trained, the formula gives 0.83 at almost every step.

**Key findings**

- **These are comparisons of complete training recipes, not a single controlled setting.** DAPO, GRPO and PPO differ in response cap, responses and updates per step, reward, n, KL, clip-high, warmup, trainer code and evaluated step (section 2, 'Which comparisons are controlled').
- **Highest average among the legacy headline checkpoints: DAPO w/o DS (step 300)**, and highest on GSM8K. Its gap over the original DS checkpoint cannot be attributed to dynamic sampling: no-DS trained on 192 prompts and made 6 updates per step, whereas DS trained on 32 prompts and made 1 update. The completed matched comparison is reported separately in section 3.10 and remains subject to seed and environment caveats. Strong-KL GRPO (KL 0.01 / 0.04) is significantly better than DAPO w/o DS on MATH and worse on GSM8K.
- **Controlled dynamic-sampling test (ablation rung 2 vs. DS, both 32 prompts and 1 update per step, step 300; section 3.10):** without DS the model collapses into short answers on Countdown within 50 steps (mean response 54 tokens vs. 2,577 with DS; Countdown validation 25% vs. 65%). The old no-DS run avoided this only by training on 6x more prompts per step. On the benchmarks, DS makes little difference under standard prompts (GSM8K +4.6 strict but +1.3 flexible, n.s.; MATH +1.0, p = 0.10; BBH and MMLU-Pro 0) and a large one under the Countdown-style think prompt (GSM8K 74.4 vs. 5.9, MATH 46.5 vs. 14.5), where rung 2 writes a minimal think block (median about 100 characters). **Seed caveat:** a second training seed of rung 2 (`rung2s2`) dipped early (155 tokens at step 25) but then recovered (about 1,000 tokens, Countdown 51% at step 300 and 62% at step 495, vs. DS 67% and 72%). **A second seed of DS itself (`DSs2`) also collapsed**, to about 20-token answers by step 25 (Countdown 26% at step 100, 35% at step 240); it ended at step 245 because DS needed up to 3 generation rounds per step and used up the 2 epochs of data. So the collapse is seed-dependent with and without DS (1 of 2 seeds each): DS does not reliably prevent it.
- **Clip-higher alone does not prevent the collapse (ablation rung 1 vs. rung 2, no DS, 32 prompts/step):** rung 1 (clip 0.2) collapses like rung 2 (clip 0.28): mean response 47 vs. 54 tokens, Countdown validation 27% vs. 25% at step 300. Benchmark differences are small: under standard prompts GSM8K +5.4 strict for rung 2 but -0.8 flexible (format again), BBH +1.8, MATH and MMLU-Pro ~0; under the think prompt rung 2 is 1-4 points higher. Clip-higher did not help; with one seed each, rung 1 and rung 2 seed 1 both collapsed, while rung 2 seed 2 recovered (see the seed caveat above).
- **Early collapse is the usual outcome, and no recipe prevents it (collapse screen, section 3.13).** Collapsed (mean response < 100 tokens at steps 90-100) in 16 of 22 runs: DS 5/6, A1 5/6, A1 + KL 0.01 3/4, no DS 3/6; differences between recipes are not significant at these sample sizes, and KL 0.01 does not fix it. Collapses happen within about 25 steps, are seed-dependent, do not need the format credit (A1 has none), and are tied to the Countdown prompt (collapsed models stay at base level under standard prompts). The healthy DS and A1 runs used for the single-run comparisons are therefore the exceptions; DS seed 1 is also the only DS run from the original environment (see 3.13).
- **Single-run comparison of options (A-D, A1 seed 1; section 3.11).** On their first seed every option avoided the early collapse; given the result above this may be luck. In-domain at step 495, masked groups (C, 70%) and batch-level normalisation (D, 69%) end within 2-3 points of DS (72%); correctness-only reward (A) reaches 69% at step 300 but collapses after step 400 as truncated responses rise to 59-63%; adding overlong filtering (A1) removes that collapse (69% at step 515). On the benchmarks at step 300 the options match DS within noise on MATH, BBH and MMLU-Pro under standard prompts; A, A1, B and C score significantly higher than DS on MATH with the think prompt (+2 to +6); GSM8K standard-prompt strict differences are mostly answer format. A1 seed 1 was the closest to DS overall (no significant loss on any benchmark and prompt, at about one-sixth of DS's generation per step), but A1 seed 2 collapsed.
- **pass@k (section 3.12): finite-sample coverage is consistent with probability redistribution.** On MATH-500, base and several trained checkpoints are similar at pass@64 (base 91.0, DS 92.2, options 90.0-92.2). On GSM8K with flexible scoring, selected recipes gain about 3 points at pass@1 while reaching 96-97% at pass@32. These finite-k results do not prove equal latent support. The PPO lr 1e-6 checkpoint has lower empirical coverage (GSM8K pass@32 88.2; MATH pass@64 85.6). Matching pass@k runs for the headline strong-KL GRPO step-250 checkpoint reach 97.1 on GSM8K pass@32 and 91.6 on MATH-500 pass@64.
- With the standard prompt, Countdown RL barely changes MATH / BBH / MMLU-Pro relative to the base model; the large GSM8K gain is mostly answer-format compliance (compare flexible accuracy).
- The think prompt yields shorter reasoning and lower MATH / MMLU-Pro accuracy for every model, including the untrained base, so the gap is caused by the prompt, not truncation.
- PPO: the actor learning rate dominates transfer. At 1e-6 (no KL, no entropy) PPO stays near the base model and improves GSM8K; every 3e-6 run falls far below the base model on all benchmarks regardless of KL loss (off / coef 0 / 0.001) or entropy (0 / 0.001). The worst run (3e-6, KL off, entropy 0.001) collapsed into Countdown-style output. KL 0.001 limits but does not prevent the damage. 'KL on, coef 0' and 'KL off' should be equivalent but differ by up to 20 points, a sign of large seed-to-seed variance at 3e-6 (one seed per setting).
- GRPO (lr 1e-6, mini-batch 64): the KL-loss coefficient decides whether reasoning survives in the Countdown (think) format. With KL off or weak (0, 1e-6, 0.001, 0.005), every run -- whatever n (2/4/8) or reward -- learned to answer the think prompt with only an `<answer>` tag and no reasoning (median ~20 characters), so think-prompt accuracy collapses (GSM8K 8-19, MATH 15-18) while standard-prompt accuracy stays near the base model. With KL 0.01 (step 250) or 0.04 the models keep reasoning (median ~780 characters) and reach averages of 47.0 / 47.2, close to DAPO w/o DS (48.6).
- The best GRPO runs are significantly better than DAPO w/o DS on MATH (think +2.7 / +3.0; KL 0.01 also standard +1.2) and significantly worse on GSM8K (standard -4.8 / -13.4, think -6.2 / -2.8); BBH and MMLU-Pro are mostly within noise. The KL 0.01 run is at step 250 (it stopped early), so it is not fully comparable with the step-520 runs.
- The GRPO KL result holds at a matched step (section 3.7): at step 250, all three runs with KL <= 0.005 already answer the think prompt with a bare `<answer>` tag (median 20-22 characters; GSM8K 10-11, MATH 15-16), while KL 0.01 and 0.04 keep reasoning (median 730-1,900 characters; averages 47.0 and 46.3). Across both steps, all seven runs with KL <= 0.005 collapse and both runs with KL >= 0.01 do not.
- Longer in-domain training slightly reduces transfer (section 3.8). Over DAPO w/o DS checkpoints 100 -> 520, Countdown validation accuracy rises from 68.8 to 78.8, while MATH falls 55.9 -> 53.0 (standard, p < 0.001) and 51.8 -> 48.3 (think, p < 0.001); GSM8K changes little (standard 69.9 -> 67.6, p = 0.09; think unchanged). Which checkpoint is evaluated therefore shifts results by about 3 points on MATH.
- For weak-KL GRPO, standard-prompt GSM8K rises with n (n=2: 45.3, n=4: 50.0, n=8: 63.5 at KL 0.001) and with the KL coefficient (n=2: 38.4 no KL, 45.3, 56.1, 65.1 at 0.04). The equal 0.5/0.5 reward (n=4, KL 1e-6) gives 66.4 vs 51.1 with the standard reward, but still collapses on the think prompt.
- Countdown validation reward does not predict transfer (e.g. PPO lr 3e-6 with KL 0.001 had one of the highest Countdown rewards but lost 43 points on MATH).
- Caveats: one training seed per setting; DAPO evaluated at step 300 (the DS run saved only that checkpoint), PPO and GRPO runs at their final step except GRPO KL 0.01 (stopped at step 250); zero-shot BBH / MMLU-Pro; McNemar p-values compare checkpoints and include no seed variance.

## 1. Evaluation protocol (identical for every model)

- **Benchmarks (full test sets):** GSM8K (n=1,319), MATH (n=5,000), BBH (n=6,511), MMLU-Pro (n=12,032). MATH-500 is also reported inside `results_8k/math/summary.txt`.
- **Prompts:** *standard* = the benchmark's usual zero-shot chain-of-thought prompt (GSM8K: `####`; MATH: `\boxed{}`; BBH: "So the answer is"; MMLU-Pro: "the answer is (X)"). *think* = the Countdown training wrapper with `<think>`/`<answer>` tags (`common.THINK_TEMPLATE`). BBH and MMLU-Pro are zero-shot (official protocols are 3-shot / 5-shot).
- **Chat template:** Qwen2.5 template, one user message (identical files for all models).
- **Decoding:** vLLM, greedy (temperature 0), no repetition penalty, max 8192 output tokens (main setting, `results_8k/`); a 2048-token run of base/DS/noDS is in `results/`.
- **Scoring:** *strict* = answer inside the last explicit final-answer marker (`####`, `\boxed{}`, `<answer>`, "the answer is"); *flexible* = strict, else last number (GSM8K), math-verify over the whole response (MATH), last `(X)` (BBH multiple choice, MMLU-Pro). MATH answers are checked with math-verify (symbolic equivalence). All accuracies are in %.
- **Statistics:** paired exact McNemar test on the same problems. It tests whether two *checkpoints* differ on these test items. It does not include seed-to-seed training variance, so it does not test whether two *training procedures* differ; every setting here has one training seed.

## 2. Models and training configuration

Settings are read from each run's wandb config (`runs_registry.csv`, produced by `run_registry.py`), not from run names. Names are unreliable: all PPO runs are named "mini32" but used `ppo_mini_batch_size` 64 (PPO_step400) or 16 (all PPO_Countdown_R1 runs); "klloss0" means KL loss *enabled* with coefficient 0.

| model | wandb | step | adv | actor lr | critic lr | KL loss | KL coef | KL in reward | entropy | n | train bsz | gen bsz | mini bsz | epochs | clip low | clip high | loss agg | DS filter | max resp | logged reward values | Countdown val metric | Countdown val @step | Countdown val best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DS_step300 | DAPO_new/f8h77wol | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 192 | 32 | 1 | 0.2 | 0.28 | token-mean | True | 4096 | 0 1 | acc | 0.6602 | 0.7393 |
| noDS_step300 | DAPO_new/3md2pzp8 | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 192 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 1 | acc | 0.751 | 0.7949 |
| noDS_step100 | DAPO_new/3md2pzp8 | 100 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 192 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 1 | acc | 0.7012 | 0.7949 |
| noDS_step200 | DAPO_new/3md2pzp8 | 200 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 192 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 1 | acc | 0.7373 | 0.7949 |
| noDS_step400 | DAPO_new/3md2pzp8 | 400 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 192 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 1 | acc | 0.7725 | 0.7949 |
| noDS_step500 | DAPO_new/3md2pzp8 | 500 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 192 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 1 | acc | 0.7949 | 0.7949 |
| noDS_step520 | DAPO_new/3md2pzp8 | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 192 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 1 | acc | 0.7881 | 0.7949 |
| rung2_step100 | DAPO_ablation/5vmxmx4q | 100 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.2461 | 0.3291 |
| rung2_step300 | DAPO_ablation/5vmxmx4q | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.2666 | 0.3291 |
| rung2_step520 | DAPO_ablation/5vmxmx4q | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.3018 | 0.3291 |
| rung1_step100 | DAPO_ablation/8j7jwtkz | 100 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.2 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.2168 | 0.3027 |
| rung1_step300 | DAPO_ablation/8j7jwtkz | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.2 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.2646 | 0.3027 |
| rung1_step520 | DAPO_ablation/8j7jwtkz | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.2 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.2842 | 0.3027 |
| rung2s2_step300 | DAPO_ablation/iic956n7 | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.5098 | 0.6348 |
| rung2s2_step520 | DAPO_ablation/iic956n7 | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.6348 | 0.6348 |
| rungA_step300 | DAPO_ablation/01nbjk9i | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 1 | acc | 0.6846 | 0.7109 |
| rungA_step520 | DAPO_ablation/01nbjk9i | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 1 | acc | 0.2881 | 0.7109 |
| rungB_step300 | DAPO_ablation/hx0uekgu | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.6221 | 0.665 |
| rungB_step520 | DAPO_ablation/hx0uekgu | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0.0 0.1 0.9 1.0 | acc | 0.5957 | 0.665 |
| rungC_step300 | DAPO_ablation/l3q0775h | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 0.9 1.0 | acc | 0.6621 | 0.71 |
| rungC_step520 | DAPO_ablation/l3q0775h | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 0.9 1.0 | acc | 0.7012 | 0.71 |
| rungD_step300 | DAPO_ablation/dezinljh | 300 | reinforce_plus_plus_baseline | 1e-06 | 1e-05 | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 0.9 1.0 | acc | 0.6328 | 0.7031 |
| rungD_step520 | DAPO_ablation/dezinljh | 520 | reinforce_plus_plus_baseline | 1e-06 | 1e-05 | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 0.9 1.0 | acc | 0.7021 | 0.7031 |
| rungA1_step300 | DAPO_ablation/bynfpzjt | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 1 | acc | 0.6631 | 0.7246 |
| rungA1_step520 | DAPO_ablation/bynfpzjt | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 1 | acc | 0.7002 | 0.7246 |
| rungA1s2_step300 | DAPO_ablation/rsa2qzj9 | 300 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 1 | acc | 0.2744 | 0.3164 |
| rungA1s2_step520 | DAPO_ablation/rsa2qzj9 | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 32 | 32 | 1 | 0.2 | 0.28 | token-mean | False | 4096 | 0 1 | acc | 0.2852 | 0.3164 |
| DSs2_step200 | DAPO_ablation/359gx7xo | 200 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 8 | 32 | 192 | 32 | 1 | 0.2 | 0.28 | token-mean | True | 4096 | 0 0.9 1.0 | acc | 0.3398 | 0.3535 |
| PPO_step400 | PPO_Countdown_R1_0923/w1d766ut | 400 | gae | 1e-06 | 1e-05 | False | - (off) | False | 0.0 | 1 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 0.9 1.0 | reward | 0.4217 | 0.4217 |
| PPO_klF_ent0_lr1e-6_step520 | PPO_Countdown_R1/4y7yxx3y | 520 | gae | 1e-06 | 1e-05 | False | - (off) | False | 0.0 | 1 | 192 | - | 16 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0 0.1 1.0 | reward | 0.3959 | 0.5413 |
| PPO_klF_ent0_lr3e-6_step520 | PPO_Countdown_R1/ua0u61fm | 520 | gae | 3e-06 | 1e-05 | False | - (off) | False | 0.0 | 1 | 192 | - | 16 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1.0 | reward | 0.4766 | 0.5063 |
| PPO_klF_ent1e-3_lr3e-6_step520 | PPO_Countdown_R1/txx0vrw6 | 520 | gae | 3e-06 | 1e-05 | False | - (off) | False | 0.001 | 1 | 192 | - | 16 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1.0 | reward | 0.3565 | 0.3769 |
| PPO_kl0_ent1e-3_lr3e-6_step520 | PPO_Countdown_R1/bwimlgaz | 520 | gae | 3e-06 | 1e-05 | True | 0 | False | 0.001 | 1 | 192 | - | 16 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.4562 | 0.5214 |
| PPO_kl1e-3_ent1e-3_lr3e-6_step520 | PPO_Countdown_R1/dgwsksem | 520 | gae | 3e-06 | 1e-05 | True | 0.001 | False | 0.001 | 1 | 192 | - | 16 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.4769 | 0.5288 |
| GRPO_n2_noKL_step250 | GRPO_Countdown_R1/qde0tjh9 | 250 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.3707 | 0.4015 |
| GRPO_n2_kl1e-3_step250 | GRPO_Countdown_R1/djmdy7s6 | 250 | grpo | 1e-06 | - | True | 0.001 | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.3583 | 0.4032 |
| GRPO_n2_kl5e-3_step250 | GRPO_Countdown_R1/02o61vat | 250 | grpo | 1e-06 | - | True | 0.005 | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.3514 | 0.4199 |
| GRPO_n2_kl4e-2_step250 | GRPO_Countdown_R1/rq4tjmwu | 250 | grpo | 1e-06 | - | True | 0.04 | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0 1 | reward | 0.2318 | 0.3155 |
| GRPO_n2_noKL_step520 | GRPO_Countdown_R1/qde0tjh9 | 520 | grpo | 1e-06 | - | False | - (off) | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.4006 | 0.4015 |
| GRPO_n2_kl1e-3_step520 | GRPO_Countdown_R1/djmdy7s6 | 520 | grpo | 1e-06 | - | True | 0.001 | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.3918 | 0.4032 |
| GRPO_n2_kl5e-3_step520 | GRPO_Countdown_R1/02o61vat | 520 | grpo | 1e-06 | - | True | 0.005 | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.3909 | 0.4199 |
| GRPO_n2_kl1e-2_step250 | GRPO_Countdown_R1/q0b8qi84 | 250 | grpo | 1e-06 | - | True | 0.01 | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.4821 | 0.5061 |
| GRPO_n2_kl4e-2_step520 | GRPO_Countdown_R1/rq4tjmwu | 520 | grpo | 1e-06 | - | True | 0.04 | False | 0.0 | 2 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0 1 | reward | 0.2899 | 0.3155 |
| GRPO_n4_kl1e-3_step520 | GRPO_Countdown_R1/m7xc02to | 520 | grpo | 1e-06 | - | True | 0.001 | False | 0.0 | 4 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.4375 | 0.4393 |
| GRPO_n8_kl1e-3_step520 | GRPO_Countdown_R1/r2atlfqf | 520 | grpo | 1e-06 | - | True | 0.001 | False | 0.0 | 8 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.4419 | 0.4604 |
| GRPO_n4_kl1e-6_step520 | GRPO_Countdown_R1/cthf0kfn | 520 | grpo | 1e-06 | - | True | 1e-06 | False | 0.0 | 4 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.1 1 | reward | 0.4164 | 0.4489 |
| GRPO_n4_kl1e-6_eqR_step520 | GRPO_Countdown_R1/dzp5op8l | 520 | grpo | 1e-06 | - | True | 1e-06 | False | 0.0 | 4 | 192 | - | 64 | 1 | 0.2 | 0.2 | token-mean | - | 2048 | 0.0 0.5 1.0 | reward | 0.6553 | 0.6694 |

Common to all trained models: base `/home/lian/models/qwen2.5-1.5b-instruct` (Qwen2.5-1.5B-Instruct), Countdown data `/home/lian/data/countdown/train.parquet`, rollout temperature 1.0, 2 epochs, gamma = lam = 1, KL penalty in reward off. DAPO runs log Countdown *accuracy*; PPO and GRPO runs log Countdown *reward*, which includes format credit and depends on the reward scheme, so Countdown numbers are only comparable between runs with the same metric and reward.

Reward scheme, codebase and notes (not in the wandb config; established from each run's logged reward values and printed reward output, and from the location of its local wandb folder):

| model | training reward | codebase | note |
|---|---|---|---|
| DS_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, countdown.compute_score_sep2025) | /home/lian/verl |  |
| noDS_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, countdown.compute_score_sep2025) | /home/lian/verl |  |
| noDS_step100 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, countdown.compute_score_sep2025) | /home/lian/verl |  |
| noDS_step200 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, countdown.compute_score_sep2025) | /home/lian/verl |  |
| noDS_step400 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, countdown.compute_score_sep2025) | /home/lian/verl |  |
| noDS_step500 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, countdown.compute_score_sep2025) | /home/lian/verl |  |
| noDS_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, countdown.compute_score_sep2025) | /home/lian/verl |  |
| rung2_step100 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) |  |
| rung2_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) |  |
| rung2_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) |  |
| rung1_step100 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) |  |
| rung1_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) |  |
| rung1_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) |  |
| rung2s2_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | training seed 2 (data.seed=2, rollout.seed=2) |
| rung2s2_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | training seed 2 (data.seed=2, rollout.seed=2) |
| rungA_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | correctness-only reward (ablation/reward_acc_only.py: correct 1, else 0) |
| rungA_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | correctness-only reward (ablation/reward_acc_only.py: correct 1, else 0) |
| rungB_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | algorithm.norm_adv_by_std_in_grpo=False |
| rungB_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | algorithm.norm_adv_by_std_in_grpo=False |
| rungC_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | +algorithm.mask_uninformative_groups=True (trainer change, see ablation/README.md) |
| rungC_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | +algorithm.mask_uninformative_groups=True (trainer change, see ablation/README.md) |
| rungD_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | algorithm.adv_estimator=reinforce_plus_plus_baseline (group mean, batch whitening) |
| rungD_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | algorithm.adv_estimator=reinforce_plus_plus_baseline (group mean, batch whitening) |
| rungA1_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | correctness-only reward + +algorithm.mask_truncated=True (overlong filtering, trainer change) |
| rungA1_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | correctness-only reward + +algorithm.mask_truncated=True (overlong filtering, trainer change) |
| rungA1s2_step300 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | option A1 (correctness-only reward + overlong filtering), training seed 2 (data.seed=2, rollout.seed=2) |
| rungA1s2_step520 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | option A1 (correctness-only reward + overlong filtering), training seed 2 (data.seed=2, rollout.seed=2) |
| DSs2_step200 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (dapo manager, ablation/reward_sep2025.py) | /home/lian/verl code, run from the verl-fork conda env (torch 2.6.0, vLLM 0.8.5.post1) | DS recipe, training seed 2; collapsed by step 25; ended at step 245 (2 epochs of data used up by up to 3 generation rounds per step); last checkpoint step 200 |
| PPO_step400 | correct 0.9+0.1 fmt / wrong 0.1 fmt / no-eq 0 (naive manager; logged scores include 0.9 = correct without format credit) | /home/lian/verl |  |
| PPO_klF_ent0_lr1e-6_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| PPO_klF_ent0_lr3e-6_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| PPO_klF_ent1e-3_lr3e-6_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| PPO_kl0_ent1e-3_lr3e-6_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| PPO_kl1e-3_ent1e-3_lr3e-6_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n2_noKL_step250 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n2_kl1e-3_step250 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n2_kl5e-3_step250 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl | step 250 comes from 02o61vat |
| GRPO_n2_kl4e-2_step250 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n2_noKL_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n2_kl1e-3_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n2_kl5e-3_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl | a short resume run (hqg8norw) also exists; step 520 comes from 02o61vat |
| GRPO_n2_kl1e-2_step250 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl | run stopped early; last checkpoint is step 250 |
| GRPO_n2_kl4e-2_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n4_kl1e-3_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n8_kl1e-3_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) | /home/lian/verl |  |
| GRPO_n4_kl1e-6_step520 | correct 1.0 / eq. extracted but wrong 0.1 / no-eq 0 (naive manager) -- named 'equalReward' but logs show the standard reward | /home/lian/verl |  |
| GRPO_n4_kl1e-6_eqR_step520 | EQUAL: correct 0.5+0.5 fmt / wrong 0.5 fmt / no-eq 0 (naive manager, verl_fork countdown.py) | /home/lian/verl_fork/verl_fork |  |

Second source for the settings: the Hydra config saved when each run was launched (copied to `run_configs/<model>.yaml`), compared with the wandb config on every setting in the table above:

| model | launch config vs. wandb | launch config file |
|---|---|---|
| DS_step300 | no launch config found (launched via ray job submit) |  |
| noDS_step300 | no launch config found (launched via ray job submit) |  |
| noDS_step100 | no launch config found (launched via ray job submit) |  |
| noDS_step200 | no launch config found (launched via ray job submit) |  |
| noDS_step400 | no launch config found (launched via ray job submit) |  |
| noDS_step500 | no launch config found (launched via ray job submit) |  |
| noDS_step520 | no launch config found (launched via ray job submit) |  |
| rung2_step100 | no launch config found |  |
| rung2_step300 | no launch config found |  |
| rung2_step520 | no launch config found |  |
| rung1_step100 | no launch config found |  |
| rung1_step300 | no launch config found |  |
| rung1_step520 | no launch config found |  |
| rung2s2_step300 | no launch config found |  |
| rung2s2_step520 | no launch config found |  |
| rungA_step300 | no launch config found |  |
| rungA_step520 | no launch config found |  |
| rungB_step300 | no launch config found |  |
| rungB_step520 | no launch config found |  |
| rungC_step300 | no launch config found |  |
| rungC_step520 | no launch config found |  |
| rungD_step300 | no launch config found |  |
| rungD_step520 | no launch config found |  |
| rungA1_step300 | no launch config found |  |
| rungA1_step520 | no launch config found |  |
| rungA1s2_step300 | no launch config found |  |
| rungA1s2_step520 | no launch config found |  |
| DSs2_step200 | no launch config found |  |
| PPO_step400 | identical | /home/lian/verl/outputs/2025-09-23/12-48-21/.hydra/config.yaml |
| PPO_klF_ent0_lr1e-6_step520 | identical | /home/lian/verl/examples/ppo_trainer/outputs/2025-10-16/13-37-23/.hydra/config.yaml |
| PPO_klF_ent0_lr3e-6_step520 | identical | /home/lian/verl/examples/ppo_trainer/outputs/2025-10-13/09-27-47/.hydra/config.yaml |
| PPO_klF_ent1e-3_lr3e-6_step520 | identical | /home/lian/verl/examples/ppo_trainer/outputs/2025-10-10/15-45-24/.hydra/config.yaml |
| PPO_kl0_ent1e-3_lr3e-6_step520 | identical | /home/lian/verl/examples/ppo_trainer/outputs/2025-10-07/10-01-27/.hydra/config.yaml |
| PPO_kl1e-3_ent1e-3_lr3e-6_step520 | identical | /home/lian/verl/examples/ppo_trainer/outputs/2025-10-06/16-34-20/.hydra/config.yaml |
| GRPO_n2_noKL_step250 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-23/07-36-07/.hydra/config.yaml |
| GRPO_n2_kl1e-3_step250 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-22/11-33-52/.hydra/config.yaml |
| GRPO_n2_kl5e-3_step250 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-28/12-08-44/.hydra/config.yaml |
| GRPO_n2_kl4e-2_step250 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-24/09-50-35/.hydra/config.yaml |
| GRPO_n2_noKL_step520 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-23/07-36-07/.hydra/config.yaml |
| GRPO_n2_kl1e-3_step520 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-22/11-33-52/.hydra/config.yaml |
| GRPO_n2_kl5e-3_step520 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-28/12-08-44/.hydra/config.yaml |
| GRPO_n2_kl1e-2_step250 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-27/09-03-00/.hydra/config.yaml |
| GRPO_n2_kl4e-2_step520 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-24/09-50-35/.hydra/config.yaml |
| GRPO_n4_kl1e-3_step520 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-20/14-50-37/.hydra/config.yaml |
| GRPO_n8_kl1e-3_step520 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-10-23/13-04-58/.hydra/config.yaml |
| GRPO_n4_kl1e-6_step520 | identical | /home/lian/verl/examples/grpo_trainer/outputs/2025-11-07/15-06-50/.hydra/config.yaml |
| GRPO_n4_kl1e-6_eqR_step520 | identical | /home/lian/verl_fork/verl_fork/examples/grpo_trainer/outputs/2025-11-19/10-44-22/.hydra/config.yaml |

### What one training step means in each recipe

Steps are not comparable across recipes. Trained prompts per step follow from the trainer code: with DS off, the DAPO trainer trains on the whole generated batch (`batch = new_batch`, `recipe/dapo/src/dapo_ray_trainer.py` line 170), so `train_batch_size=32` is ignored; with DS on it cuts the kept groups to `train_batch_size` (lines 215-216). Both are confirmed by the logged token counts (1,536 sequences per step without DS, 256 with DS, at every step). Optimizer updates per step = trained prompts / mini-batch x PPO epochs (PPO also updates its critic, not counted).

| model | step | prompts generated / step | prompts trained / step | responses trained / step | updates / step | prompts trained at eval step | updates at eval step |
|---|---|---|---|---|---|---|---|
| DS_step300 | 300 | 200.3 | 32 | 256 | 1 | 9600 | 300 |
| noDS_step300 | 300 | 192.0 | 192 | 1536 | 6 | 57600 | 1800 |
| noDS_step100 | 100 | 192.0 | 192 | 1536 | 6 | 19200 | 600 |
| noDS_step200 | 200 | 192.0 | 192 | 1536 | 6 | 38400 | 1200 |
| noDS_step400 | 400 | 192.0 | 192 | 1536 | 6 | 76800 | 2400 |
| noDS_step500 | 500 | 192.0 | 192 | 1536 | 6 | 96000 | 3000 |
| noDS_step520 | 520 | 192.0 | 192 | 1536 | 6 | 99840 | 3120 |
| rung2_step100 | 100 | 32.0 | 32 | 256 | 1 | 3200 | 100 |
| rung2_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rung2_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| rung1_step100 | 100 | 32.0 | 32 | 256 | 1 | 3200 | 100 |
| rung1_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rung1_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| rung2s2_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rung2s2_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| rungA_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rungA_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| rungB_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rungB_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| rungC_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rungC_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| rungD_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rungD_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| rungA1_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rungA1_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| rungA1s2_step300 | 300 | 32.0 | 32 | 256 | 1 | 9600 | 300 |
| rungA1s2_step520 | 520 | 32.0 | 32 | 256 | 1 | 16640 | 520 |
| DSs2_step200 | 200 | 204.5 | 32 | 256 | 1 | 6400 | 200 |
| PPO_step400 | 400 | 192.0 | 192 | 192 | 3 | 76800 | 1200 |
| PPO_klF_ent0_lr1e-6_step520 | 520 | 192.0 | 192 | 192 | 12 | 99840 | 6240 |
| PPO_klF_ent0_lr3e-6_step520 | 520 | 192.0 | 192 | 192 | 12 | 99840 | 6240 |
| PPO_klF_ent1e-3_lr3e-6_step520 | 520 | 192.0 | 192 | 192 | 12 | 99840 | 6240 |
| PPO_kl0_ent1e-3_lr3e-6_step520 | 520 | 192.0 | 192 | 192 | 12 | 99840 | 6240 |
| PPO_kl1e-3_ent1e-3_lr3e-6_step520 | 520 | 192.0 | 192 | 192 | 12 | 99840 | 6240 |
| GRPO_n2_noKL_step250 | 250 | 192.0 | 192 | 384 | 3 | 48000 | 750 |
| GRPO_n2_kl1e-3_step250 | 250 | 192.0 | 192 | 384 | 3 | 48000 | 750 |
| GRPO_n2_kl5e-3_step250 | 250 | 192.0 | 192 | 384 | 3 | 48000 | 750 |
| GRPO_n2_kl4e-2_step250 | 250 | 192.0 | 192 | 384 | 3 | 48000 | 750 |
| GRPO_n2_noKL_step520 | 520 | 192.0 | 192 | 384 | 3 | 99840 | 1560 |
| GRPO_n2_kl1e-3_step520 | 520 | 192.0 | 192 | 384 | 3 | 99840 | 1560 |
| GRPO_n2_kl5e-3_step520 | 520 | 192.0 | 192 | 384 | 3 | 99840 | 1560 |
| GRPO_n2_kl1e-2_step250 | 250 | 192.0 | 192 | 384 | 3 | 48000 | 750 |
| GRPO_n2_kl4e-2_step520 | 520 | 192.0 | 192 | 384 | 3 | 99840 | 1560 |
| GRPO_n4_kl1e-3_step520 | 520 | 192.0 | 192 | 768 | 3 | 99840 | 1560 |
| GRPO_n8_kl1e-3_step520 | 520 | 192.0 | 192 | 1536 | 3 | 99840 | 1560 |
| GRPO_n4_kl1e-6_step520 | 520 | 192.0 | 192 | 768 | 3 | 99840 | 1560 |
| GRPO_n4_kl1e-6_eqR_step520 | 520 | 192.0 | 192 | 768 | 3 | 99840 | 1560 |

### Which comparisons are controlled

- **Original DAPO w/ DS vs. w/o DS: not controlled.** Without DS the trainer used all 192 generated prompts (1,536 responses, 6 updates per step); with DS it retained 32 prompts (256 responses, 1 update). The original benchmark gap therefore cannot be attributed to filtering. The completed matched comparison uses 32 prompts and one update per step in both conditions (section 3.10), but it has limited seed replication and the matched no-DS run was trained in a later software environment (`../ablation/README.md`).
- **GRPO KL sweep (n = 2, lr 1e-6, mini-batch 64):** only the KL coefficient differs. Step-matched at 250 for all five coefficients (section 3.7); the KL 0.01 run exists only at step 250.
- **GRPO group size (n = 2 / 4 / 8 at KL 0.001):** prompts and updates per step are equal (192, 3), but responses per step grow with n (384 / 768 / 1,536), so group size and sample budget change together.
- **GRPO reward (equal 0.5/0.5 vs. standard, n = 4, KL 1e-6): not controlled.** The equal-reward run (`checkagain`) was trained with `/home/lian/verl_fork` = verl 0.5.0.dev, the others with `/home/lian/verl` = verl 0.2.0.dev; 135 source files differ, including `trainer/ppo/core_algos.py`, `workers/actor/dp_actor.py` and `trainer/ppo/ray_trainer.py`.
- **PPO (October runs, mini-batch 16): not a full grid.** Learning rate is compared only with KL off and entropy 0; entropy only at lr 3e-6 with KL off; KL only at lr 3e-6 with entropy 0.001. 'KL off' and 'KL on, coefficient 0' should be equivalent but differ by up to about 20 points, so KL and entropy differences at 3e-6 are within seed noise. The learning-rate effect holds up because all four 3e-6 runs collapse (rough replicates). PPO_step400 is not part of these sweeps (mini-batch 64, Sep reward, step 400).
- **Across families (DAPO vs. GRPO vs. PPO): complete recipes, not controlled.** They differ in response cap (4,096 vs. 2,048), responses and updates per step (table above), reward, n, KL, clip-high (0.28 vs. 0.2), learning-rate warmup (10 vs. none), trainer code (`recipe/dapo` vs. `main_ppo`), critic (PPO) and the evaluated step.

Checkpoint paths:

- `DS_step300`: `/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-filtering-tokenlevel-bsz-32-0910/global_step_300` (run name `Qwen2.5-1.5B-G8-acc-filtering-tokenlevel-bsz-32-0910`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/DS_step300`
- `noDS_step300`: `/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_300` (run name `Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step300`
- `noDS_step100`: `/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_100` (run name `Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step100`
- `noDS_step200`: `/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_200` (run name `Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step200`
- `noDS_step400`: `/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_400` (run name `Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step400`
- `noDS_step500`: `/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_500` (run name `Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step500`
- `noDS_step520`: `/mnt/sdb/verl/DAPO_new/Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915/global_step_520` (run name `Qwen2.5-1.5B-G8-acc-no-filtering-tokenlevel-bsz-32-0915`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step520`
- `rung2_step100`: `/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_100` (run name `rung2-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rung2_step100`
- `rung2_step300`: `/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300` (run name `rung2-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rung2_step300`
- `rung2_step520`: `/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520` (run name `rung2-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rung2_step520`
- `rung1_step100`: `/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096/global_step_100` (run name `rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rung1_step100`
- `rung1_step300`: `/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096/global_step_300` (run name `rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rung1_step300`
- `rung1_step520`: `/mnt/sdb/verl/DAPO_ablation/rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096/global_step_520` (run name `rung1-grpo-clip0.2-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rung1_step520`
- `rung2s2_step300`: `/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2/global_step_300` (run name `rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rung2s2_step300`
- `rung2s2_step520`: `/mnt/sdb/verl/DAPO_ablation/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2/global_step_520` (run name `rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rung2s2_step520`
- `rungA_step300`: `/mnt/sdb/verl/DAPO_ablation/rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300` (run name `rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungA_step300`
- `rungA_step520`: `/mnt/sdb/verl/DAPO_ablation/rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520` (run name `rungA-accOnlyReward-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungA_step520`
- `rungB_step300`: `/mnt/sdb/verl/DAPO_ablation/rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300` (run name `rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungB_step300`
- `rungB_step520`: `/mnt/sdb/verl/DAPO_ablation/rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520` (run name `rungB-noStdNorm-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungB_step520`
- `rungC_step300`: `/mnt/sdb/verl/DAPO_ablation/rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300` (run name `rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungC_step300`
- `rungC_step520`: `/mnt/sdb/verl/DAPO_ablation/rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520` (run name `rungC-maskGroups-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungC_step520`
- `rungD_step300`: `/mnt/sdb/verl/DAPO_ablation/rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300` (run name `rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungD_step300`
- `rungD_step520`: `/mnt/sdb/verl/DAPO_ablation/rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520` (run name `rungD-batchNorm-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungD_step520`
- `rungA1_step300`: `/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_300` (run name `rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungA1_step300`
- `rungA1_step520`: `/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096/global_step_520` (run name `rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungA1_step520`
- `rungA1s2_step300`: `/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2/global_step_300` (run name `rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungA1s2_step300`
- `rungA1s2_step520`: `/mnt/sdb/verl/DAPO_ablation/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2/global_step_520` (run name `rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/rungA1s2_step520`
- `DSs2_step200`: `/mnt/sdb/verl/DAPO_ablation/rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed2/global_step_200` (run name `rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/DSs2_step200`
- `PPO_step400`: `/home/lian/verl/checkpoints/PPO_Countdown_R1_0923/ppo_qwen2.5_1.5b-train192-mini32/global_step_400` (run name `ppo_qwen2.5_1.5b-train192-mini32`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/PPO_step400`
- `PPO_klF_ent0_lr1e-6_step520`: `/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy0-1lr-6/global_step_520` (run name `ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy0-1lr-6`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/PPO_klF_ent0_lr1e-6_step520`
- `PPO_klF_ent0_lr3e-6_step520`: `/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy0-3lr-6/global_step_520` (run name `ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy0-3lr-6`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/PPO_klF_ent0_lr3e-6_step520`
- `PPO_klF_ent1e-3_lr3e-6_step520`: `/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy1-3lr-6/global_step_520` (run name `ppo_qwen2.5_1.5b-train192-mini32-kllossFalse-entropy1-3lr-6`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/PPO_klF_ent1e-3_lr3e-6_step520`
- `PPO_kl0_ent1e-3_lr3e-6_step520`: `/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-klloss0-entropy1-3lr-6/global_step_520` (run name `ppo_qwen2.5_1.5b-train192-mini32-klloss0-entropy1-3lr-6`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/PPO_kl0_ent1e-3_lr3e-6_step520`
- `PPO_kl1e-3_ent1e-3_lr3e-6_step520`: `/mnt/sdb/verl/ckpts/PPO_Countdown_R1/ppo_qwen2.5_1.5b-train192-mini32-klloss1-entropy-3lr-6/global_step_520` (run name `ppo_qwen2.5_1.5b-train192-mini32-klloss1-entropy-3lr-6`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/PPO_kl1e-3_ent1e-3_lr3e-6_step520`
- `GRPO_n2_noKL_step250`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2-noKL-Loss/global_step_250` (run name `grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2-noKL-Loss`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_noKL_step250`
- `GRPO_n2_kl1e-3_step250`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2/global_step_250` (run name `grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl1e-3_step250`
- `GRPO_n2_kl5e-3_step250`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.005-entropy0-rollout2/global_step_250` (run name `grpo_qwen2.5_1.5b-klloss0.005-entropy0-rollout2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl5e-3_step250`
- `GRPO_n2_kl4e-2_step250`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.04-entropy0-rollout2/global_step_250` (run name `grpo_qwen2.5_1.5b-klloss0.04-entropy0-rollout2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl4e-2_step250`
- `GRPO_n2_noKL_step520`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2-noKL-Loss/global_step_520` (run name `grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2-noKL-Loss`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_noKL_step520`
- `GRPO_n2_kl1e-3_step520`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2/global_step_520` (run name `grpo_qwen2.5_1.5b-klloss1-entropy0-rollout2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl1e-3_step520`
- `GRPO_n2_kl5e-3_step520`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.005-entropy0-rollout2/global_step_520` (run name `grpo_qwen2.5_1.5b-klloss0.005-entropy0-rollout2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl5e-3_step520`
- `GRPO_n2_kl1e-2_step250`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.01-entropy0-rollout2/global_step_250` (run name `grpo_qwen2.5_1.5b-klloss0.01-entropy0-rollout2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl1e-2_step250`
- `GRPO_n2_kl4e-2_step520`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss0.04-entropy0-rollout2/global_step_520` (run name `grpo_qwen2.5_1.5b-klloss0.04-entropy0-rollout2`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl4e-2_step520`
- `GRPO_n4_kl1e-3_step520`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4/global_step_520` (run name `grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n4_kl1e-3_step520`
- `GRPO_n8_kl1e-3_step520`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout8/global_step_520` (run name `grpo_qwen2.5_1.5b-klloss1-entropy0-rollout8`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n8_kl1e-3_step520`
- `GRPO_n4_kl1e-6_step520`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4-equalReward/global_step_520` (run name `grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4-equalReward`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n4_kl1e-6_step520`
- `GRPO_n4_kl1e-6_eqR_step520`: `/mnt/sdb/verl/ckpts/GRPO_Countdown_R1/grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4-equalReward-checkagain/global_step_520` (run name `grpo_qwen2.5_1.5b-klloss1-entropy0-rollout4-equalReward-checkagain`) -> merged `/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n4_kl1e-6_eqR_step520`
- `base`: `Qwen/Qwen2.5-1.5B-Instruct` (HF hub; the original local copy used for training no longer exists)

## 3. Results, 8192 output tokens (main setting)

### 3.1 Strict accuracy (%)

| model | GSM8K standard | GSM8K think | MATH standard | MATH think | BBH standard | BBH think | MMLU-Pro standard | MMLU-Pro think | avg (8) | avg std | avg think |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 33.4 | 48.4 | 55.1 | 47.4 | 35.2 | 36.2 | 33.1 | 27.5 | 39.5 | 39.2 | 39.9 |
| DS_step300 | 66.1 | 74.4 | 55.1 | 46.5 | 35.1 | 40.2 | 33.1 | 26.1 | 47.1 | 47.4 | 46.8 |
| noDS_step300 | 69.9 | 75.7 | 54.9 | 49.5 | 34.4 | 41.5 | 32.9 | 29.7 | 48.6 | 48.0 | 49.1 |
| rung2_step300 | 61.5 | 5.9 | 54.1 | 14.5 | 35.5 | 36.5 | 33.0 | 24.2 | 33.2 | 46.0 | 20.3 |
| rung1_step300 | 56.1 | 5.2 | 53.7 | 11.7 | 33.7 | 35.2 | 33.1 | 20.5 | 31.1 | 44.2 | 18.1 |
| rung2s2_step300 | 47.1 | 70.6 | 55.0 | 47.4 | 34.4 | 40.7 | 33.3 | 28.8 | 44.7 | 42.5 | 46.9 |
| rungA_step300 | 61.6 | 72.2 | 55.7 | 52.3 | 35.8 | 41.2 | 33.0 | 29.7 | 47.7 | 46.5 | 48.8 |
| rungB_step300 | 61.1 | 71.6 | 55.5 | 48.7 | 35.5 | 40.9 | 33.9 | 29.1 | 47.0 | 46.5 | 47.6 |
| rungC_step300 | 55.0 | 75.0 | 55.6 | 52.6 | 35.1 | 40.6 | 33.1 | 26.5 | 46.7 | 44.7 | 48.7 |
| rungD_step300 | 59.4 | 70.1 | 56.1 | 45.8 | 35.1 | 40.4 | 33.2 | 26.9 | 45.9 | 46.0 | 45.8 |
| rungA1_step300 | 64.3 | 74.1 | 55.4 | 51.8 | 35.2 | 40.8 | 32.8 | 26.3 | 47.6 | 46.9 | 48.3 |
| rungA1s2_step300 | 37.7 | 8.0 | 55.0 | 13.4 | 34.6 | 34.7 | 33.1 | 23.7 | 30.0 | 40.1 | 19.9 |
| DSs2_step200 | 39.2 | 9.9 | 54.4 | 14.7 | 34.9 | 35.7 | 32.7 | 24.7 | 30.8 | 40.3 | 21.3 |
| PPO_step400 | 40.1 | 50.0 | 53.6 | 28.0 | 35.6 | 36.1 | 33.1 | 25.3 | 37.7 | 40.6 | 34.9 |
| PPO_klF_ent0_lr1e-6_step520 | 62.2 | 62.4 | 46.2 | 42.3 | 24.3 | 36.4 | 31.4 | 27.0 | 41.5 | 41.0 | 42.0 |
| PPO_klF_ent0_lr3e-6_step520 | 28.7 | 25.3 | 12.1 | 10.7 | 18.0 | 20.9 | 18.6 | 11.4 | 18.2 | 19.3 | 17.1 |
| PPO_klF_ent1e-3_lr3e-6_step520 | 1.0 | 1.4 | 2.6 | 2.1 | 5.7 | 25.7 | 1.6 | 13.9 | 6.7 | 2.7 | 10.7 |
| PPO_kl0_ent1e-3_lr3e-6_step520 | 20.4 | 21.6 | 11.9 | 7.5 | 22.9 | 33.8 | 18.5 | 20.5 | 19.6 | 18.4 | 20.8 |
| PPO_kl1e-3_ent1e-3_lr3e-6_step520 | 32.8 | 28.4 | 12.3 | 7.8 | 28.1 | 29.8 | 19.8 | 15.9 | 21.8 | 23.2 | 20.5 |
| GRPO_n2_noKL_step250 | 44.0 | 11.1 | 52.9 | 15.8 | 34.6 | 35.4 | 33.1 | 25.7 | 31.6 | 41.2 | 22.0 |
| GRPO_n2_kl1e-3_step250 | 49.6 | 9.8 | 54.6 | 14.6 | 35.4 | 35.4 | 33.0 | 25.2 | 32.2 | 43.1 | 21.2 |
| GRPO_n2_kl5e-3_step250 | 47.2 | 9.6 | 54.3 | 15.9 | 36.0 | 34.8 | 33.2 | 25.1 | 32.0 | 42.6 | 21.3 |
| GRPO_n2_kl4e-2_step250 | 69.5 | 57.7 | 55.3 | 51.7 | 35.0 | 37.5 | 33.9 | 29.7 | 46.3 | 48.4 | 44.1 |
| GRPO_n2_noKL_step520 | 38.4 | 8.3 | 52.3 | 16.1 | 34.6 | 36.0 | 33.7 | 25.7 | 30.6 | 39.7 | 21.5 |
| GRPO_n2_kl1e-3_step520 | 45.3 | 11.0 | 55.2 | 15.0 | 35.2 | 35.7 | 33.6 | 25.0 | 32.0 | 42.3 | 21.7 |
| GRPO_n2_kl5e-3_step520 | 56.1 | 11.2 | 53.9 | 16.6 | 35.7 | 33.1 | 33.6 | 25.6 | 33.2 | 44.8 | 21.6 |
| GRPO_n2_kl1e-2_step250 | 56.5 | 72.9 | 56.0 | 52.5 | 34.8 | 40.5 | 33.1 | 29.6 | 47.0 | 45.1 | 48.9 |
| GRPO_n2_kl4e-2_step520 | 65.1 | 69.4 | 55.9 | 52.2 | 34.6 | 38.7 | 32.7 | 29.0 | 47.2 | 47.1 | 47.3 |
| GRPO_n4_kl1e-3_step520 | 50.0 | 10.4 | 52.1 | 15.6 | 34.4 | 35.3 | 33.5 | 25.7 | 32.1 | 42.5 | 21.8 |
| GRPO_n8_kl1e-3_step520 | 63.5 | 11.0 | 52.7 | 15.9 | 35.8 | 36.0 | 33.3 | 25.4 | 34.2 | 46.3 | 22.1 |
| GRPO_n4_kl1e-6_step520 | 51.1 | 12.0 | 53.4 | 15.5 | 34.3 | 35.0 | 33.4 | 25.4 | 32.5 | 43.0 | 22.0 |
| GRPO_n4_kl1e-6_eqR_step520 | 66.4 | 18.7 | 53.8 | 18.4 | 35.2 | 34.1 | 33.7 | 24.2 | 35.5 | 47.3 | 23.8 |

### 3.2 Ranking by average strict accuracy over all 4 benchmarks x 2 prompts

| rank | model | avg (8) | avg std | avg think |
|---|---|---|---|---|
| 1 | noDS_step300 | 48.6 | 48.0 | 49.1 |
| 2 | rungA_step300 | 47.7 | 46.5 | 48.8 |
| 3 | rungA1_step300 | 47.6 | 46.9 | 48.3 |
| 4 | GRPO_n2_kl4e-2_step520 | 47.2 | 47.1 | 47.3 |
| 5 | DS_step300 | 47.1 | 47.4 | 46.8 |
| 6 | rungB_step300 | 47.0 | 46.5 | 47.6 |
| 7 | GRPO_n2_kl1e-2_step250 | 47.0 | 45.1 | 48.9 |
| 8 | rungC_step300 | 46.7 | 44.7 | 48.7 |
| 9 | GRPO_n2_kl4e-2_step250 | 46.3 | 48.4 | 44.1 |
| 10 | rungD_step300 | 45.9 | 46.0 | 45.8 |
| 11 | rung2s2_step300 | 44.7 | 42.5 | 46.9 |
| 12 | PPO_klF_ent0_lr1e-6_step520 | 41.5 | 41.0 | 42.0 |
| 13 | base | 39.5 | 39.2 | 39.9 |
| 14 | PPO_step400 | 37.7 | 40.6 | 34.9 |
| 15 | GRPO_n4_kl1e-6_eqR_step520 | 35.5 | 47.3 | 23.8 |
| 16 | GRPO_n8_kl1e-3_step520 | 34.2 | 46.3 | 22.1 |
| 17 | GRPO_n2_kl5e-3_step520 | 33.2 | 44.8 | 21.6 |
| 18 | rung2_step300 | 33.2 | 46.0 | 20.3 |
| 19 | GRPO_n4_kl1e-6_step520 | 32.5 | 43.0 | 22.0 |
| 20 | GRPO_n2_kl1e-3_step250 | 32.2 | 43.1 | 21.2 |
| 21 | GRPO_n4_kl1e-3_step520 | 32.1 | 42.5 | 21.8 |
| 22 | GRPO_n2_kl1e-3_step520 | 32.0 | 42.3 | 21.7 |
| 23 | GRPO_n2_kl5e-3_step250 | 32.0 | 42.6 | 21.3 |
| 24 | GRPO_n2_noKL_step250 | 31.6 | 41.2 | 22.0 |
| 25 | rung1_step300 | 31.1 | 44.2 | 18.1 |
| 26 | DSs2_step200 | 30.8 | 40.3 | 21.3 |
| 27 | GRPO_n2_noKL_step520 | 30.6 | 39.7 | 21.5 |
| 28 | rungA1s2_step300 | 30.0 | 40.1 | 19.9 |
| 29 | PPO_kl1e-3_ent1e-3_lr3e-6_step520 | 21.8 | 23.2 | 20.5 |
| 30 | PPO_kl0_ent1e-3_lr3e-6_step520 | 19.6 | 18.4 | 20.8 |
| 31 | PPO_klF_ent0_lr3e-6_step520 | 18.2 | 19.3 | 17.1 |
| 32 | PPO_klF_ent1e-3_lr3e-6_step520 | 6.7 | 2.7 | 10.7 |

### 3.3 Flexible accuracy (%)

| model | bbh standard | bbh think | gsm8k standard | gsm8k think | math standard | math think | mmlupro standard | mmlupro think |
|---|---|---|---|---|---|---|---|---|
| base | 35.4 | 37.4 | 72.3 | 51.9 | 55.3 | 49.5 | 33.1 | 27.5 |
| DS_step300 | 35.3 | 40.2 | 73.0 | 74.4 | 55.2 | 47.2 | 33.1 | 26.2 |
| noDS_step300 | 34.5 | 41.5 | 75.6 | 75.7 | 54.9 | 50.9 | 32.9 | 29.7 |
| rung2_step300 | 36.0 | 36.5 | 72.1 | 5.9 | 54.2 | 14.8 | 33.1 | 24.2 |
| rung1_step300 | 34.2 | 35.2 | 72.6 | 5.2 | 53.7 | 12.6 | 33.1 | 20.5 |
| rung2s2_step300 | 34.9 | 40.7 | 71.6 | 70.6 | 55.1 | 47.7 | 33.3 | 28.8 |
| rungA_step300 | 36.1 | 41.2 | 73.4 | 72.2 | 55.8 | 52.9 | 33.0 | 29.7 |
| rungB_step300 | 35.8 | 41.0 | 73.1 | 71.6 | 55.5 | 49.3 | 33.9 | 29.1 |
| rungC_step300 | 35.3 | 40.7 | 72.0 | 75.0 | 55.6 | 53.5 | 33.2 | 26.5 |
| rungD_step300 | 35.3 | 40.4 | 74.0 | 70.4 | 56.2 | 47.7 | 33.2 | 27.0 |
| rungA1_step300 | 35.3 | 40.9 | 73.6 | 74.2 | 55.4 | 52.3 | 32.8 | 26.3 |
| rungA1s2_step300 | 34.8 | 34.7 | 71.2 | 8.0 | 55.1 | 13.5 | 33.2 | 23.7 |
| DSs2_step200 | 35.1 | 35.7 | 70.4 | 9.9 | 54.5 | 14.9 | 32.7 | 24.7 |
| PPO_step400 | 35.7 | 36.1 | 70.5 | 54.1 | 53.9 | 29.0 | 33.1 | 25.3 |
| PPO_klF_ent0_lr1e-6_step520 | 24.3 | 36.4 | 62.7 | 62.4 | 46.2 | 42.6 | 31.4 | 27.0 |
| PPO_klF_ent0_lr3e-6_step520 | 18.1 | 20.9 | 29.0 | 25.5 | 12.1 | 11.4 | 18.6 | 11.4 |
| PPO_klF_ent1e-3_lr3e-6_step520 | 14.2 | 25.7 | 1.2 | 1.4 | 2.7 | 3.0 | 5.6 | 13.9 |
| PPO_kl0_ent1e-3_lr3e-6_step520 | 23.0 | 33.8 | 20.8 | 21.6 | 11.9 | 8.0 | 18.5 | 20.5 |
| PPO_kl1e-3_ent1e-3_lr3e-6_step520 | 28.1 | 29.8 | 33.0 | 28.4 | 12.3 | 8.4 | 19.8 | 15.9 |
| GRPO_n2_noKL_step250 | 34.7 | 35.4 | 63.8 | 11.1 | 53.2 | 15.9 | 33.1 | 25.7 |
| GRPO_n2_kl1e-3_step250 | 35.5 | 35.4 | 72.5 | 9.8 | 54.7 | 14.7 | 33.0 | 25.2 |
| GRPO_n2_kl5e-3_step250 | 36.1 | 34.8 | 71.9 | 9.6 | 54.3 | 16.0 | 33.2 | 25.1 |
| GRPO_n2_kl4e-2_step250 | 35.2 | 37.5 | 75.4 | 57.7 | 55.4 | 52.6 | 33.9 | 29.7 |
| GRPO_n2_noKL_step520 | 34.6 | 36.0 | 56.1 | 8.3 | 52.6 | 16.2 | 33.7 | 25.7 |
| GRPO_n2_kl1e-3_step520 | 35.3 | 35.7 | 67.4 | 11.0 | 55.2 | 15.1 | 33.6 | 25.0 |
| GRPO_n2_kl5e-3_step520 | 35.9 | 33.1 | 71.4 | 11.2 | 54.0 | 16.6 | 33.6 | 25.6 |
| GRPO_n2_kl1e-2_step250 | 35.0 | 40.5 | 73.5 | 72.9 | 56.1 | 53.2 | 33.1 | 29.6 |
| GRPO_n2_kl4e-2_step520 | 34.9 | 38.7 | 74.0 | 69.4 | 55.9 | 53.0 | 32.7 | 29.1 |
| GRPO_n4_kl1e-3_step520 | 34.5 | 35.3 | 60.1 | 10.4 | 52.2 | 15.8 | 33.5 | 25.7 |
| GRPO_n8_kl1e-3_step520 | 35.8 | 36.0 | 72.0 | 11.0 | 52.8 | 16.0 | 33.3 | 25.4 |
| GRPO_n4_kl1e-6_step520 | 34.4 | 35.0 | 67.6 | 12.0 | 53.5 | 15.6 | 33.4 | 25.4 |
| GRPO_n4_kl1e-6_eqR_step520 | 35.4 | 34.1 | 72.3 | 24.0 | 53.8 | 19.2 | 33.7 | 24.2 |

### 3.4 Responses cut off at 8192 tokens (%)

| model | bbh standard | bbh think | gsm8k standard | gsm8k think | math standard | math think | mmlupro standard | mmlupro think |
|---|---|---|---|---|---|---|---|---|
| base | 3.1 | 4.2 | 0.2 | 0.1 | 4.7 | 5.3 | 3.8 | 8.0 |
| DS_step300 | 3.3 | 3.6 | 0.5 | 0.9 | 5.6 | 7.1 | 6.2 | 10.5 |
| noDS_step300 | 4.5 | 4.0 | 0.8 | 0.3 | 5.7 | 6.1 | 6.8 | 4.4 |
| rung2_step300 | 3.6 | 1.4 | 0.4 | 0.3 | 4.9 | 1.0 | 3.9 | 0.0 |
| rung1_step300 | 2.9 | 0.7 | 0.5 | 0.0 | 4.3 | 0.2 | 3.3 | 0.0 |
| rung2s2_step300 | 2.4 | 1.4 | 0.4 | 0.2 | 3.3 | 3.2 | 2.7 | 2.1 |
| rungA_step300 | 3.2 | 4.8 | 0.4 | 0.8 | 5.4 | 7.2 | 5.4 | 10.1 |
| rungB_step300 | 2.7 | 4.3 | 0.8 | 1.4 | 5.0 | 7.9 | 4.4 | 6.8 |
| rungC_step300 | 3.0 | 3.2 | 0.2 | 0.2 | 4.6 | 6.1 | 5.9 | 9.2 |
| rungD_step300 | 2.6 | 5.6 | 0.7 | 1.3 | 4.8 | 10.7 | 4.0 | 8.3 |
| rungA1_step300 | 4.1 | 8.2 | 0.9 | 1.2 | 8.1 | 12.2 | 8.2 | 18.0 |
| rungA1s2_step300 | 3.3 | 1.0 | 0.3 | 0.0 | 4.4 | 0.3 | 4.0 | 0.1 |
| DSs2_step200 | 2.7 | 1.6 | 0.2 | 0.0 | 3.8 | 0.8 | 3.6 | 0.3 |
| PPO_step400 | 3.1 | 1.8 | 0.5 | 0.6 | 5.0 | 4.2 | 3.4 | 2.5 |
| PPO_klF_ent0_lr1e-6_step520 | 4.7 | 9.1 | 0.8 | 2.8 | 8.7 | 17.7 | 9.7 | 19.0 |
| PPO_klF_ent0_lr3e-6_step520 | 12.4 | 11.8 | 10.8 | 7.9 | 35.7 | 33.0 | 27.0 | 22.8 |
| PPO_klF_ent1e-3_lr3e-6_step520 | 6.5 | 2.1 | 27.7 | 2.0 | 53.2 | 5.5 | 32.3 | 0.9 |
| PPO_kl0_ent1e-3_lr3e-6_step520 | 15.9 | 1.0 | 29.6 | 2.0 | 51.6 | 19.0 | 31.0 | 7.8 |
| PPO_kl1e-3_ent1e-3_lr3e-6_step520 | 15.7 | 17.0 | 14.4 | 18.6 | 39.3 | 44.1 | 29.1 | 36.6 |
| GRPO_n2_noKL_step250 | 2.7 | 1.3 | 0.1 | 0.1 | 4.1 | 0.2 | 3.4 | 0.0 |
| GRPO_n2_kl1e-3_step250 | 2.8 | 1.1 | 0.2 | 0.0 | 4.3 | 0.4 | 3.2 | 0.0 |
| GRPO_n2_kl5e-3_step250 | 2.7 | 1.0 | 0.4 | 0.0 | 3.8 | 0.2 | 3.2 | 0.1 |
| GRPO_n2_kl4e-2_step250 | 2.9 | 3.9 | 0.6 | 0.7 | 3.9 | 4.8 | 4.1 | 7.3 |
| GRPO_n2_noKL_step520 | 1.5 | 1.0 | 0.2 | 0.0 | 4.9 | 0.2 | 3.6 | 0.0 |
| GRPO_n2_kl1e-3_step520 | 2.1 | 0.9 | 0.5 | 0.0 | 4.1 | 0.4 | 2.9 | 0.0 |
| GRPO_n2_kl5e-3_step520 | 2.1 | 0.9 | 0.4 | 0.0 | 4.3 | 0.5 | 3.2 | 0.0 |
| GRPO_n2_kl1e-2_step250 | 2.7 | 4.8 | 0.5 | 1.1 | 5.1 | 6.6 | 6.2 | 9.4 |
| GRPO_n2_kl4e-2_step520 | 2.6 | 4.2 | 0.5 | 0.7 | 3.5 | 4.9 | 4.8 | 7.2 |
| GRPO_n4_kl1e-3_step520 | 1.4 | 0.8 | 0.4 | 0.0 | 3.7 | 0.0 | 2.7 | 0.0 |
| GRPO_n8_kl1e-3_step520 | 2.7 | 0.7 | 0.2 | 0.0 | 4.7 | 0.2 | 2.5 | 0.0 |
| GRPO_n4_kl1e-6_step520 | 2.3 | 0.9 | 0.2 | 0.0 | 3.6 | 0.0 | 2.7 | 0.0 |
| GRPO_n4_kl1e-6_eqR_step520 | 3.8 | 1.1 | 0.2 | 0.7 | 4.1 | 3.0 | 3.1 | 0.5 |

### 3.5 Difference from base model (strict, points) and McNemar p

| model | bbh standard | bbh think | gsm8k standard | gsm8k think | math standard | math think | mmlupro standard | mmlupro think |
|---|---|---|---|---|---|---|---|---|
| DS_step300 | -0.0 (p=0.961) | +4.0 (p=<0.001) | +32.8 (p=<0.001) | +25.9 (p=<0.001) | -0.0 (p=1.000) | -0.9 (p=0.244) | +0.0 (p=1.000) | -1.3 (p=0.002) |
| noDS_step300 | -0.8 (p=0.221) | +5.3 (p=<0.001) | +36.5 (p=<0.001) | +27.2 (p=<0.001) | -0.3 (p=0.675) | +2.2 (p=0.002) | -0.1 (p=0.742) | +2.2 (p=<0.001) |
| rung2_step300 | +0.4 (p=0.545) | +0.3 (p=0.675) | +28.1 (p=<0.001) | -42.5 (p=<0.001) | -1.0 (p=0.082) | -32.9 (p=<0.001) | -0.0 (p=0.967) | -3.2 (p=<0.001) |
| rung1_step300 | -1.4 (p=0.020) | -1.0 (p=0.140) | +22.7 (p=<0.001) | -43.3 (p=<0.001) | -1.5 (p=0.009) | -35.6 (p=<0.001) | +0.0 (p=0.933) | -6.9 (p=<0.001) |
| rung2s2_step300 | -0.7 (p=0.233) | +4.5 (p=<0.001) | +13.7 (p=<0.001) | +22.1 (p=<0.001) | -0.1 (p=0.877) | +0.0 (p=0.977) | +0.2 (p=0.554) | +1.3 (p=0.003) |
| rungA_step300 | +0.6 (p=0.335) | +4.9 (p=<0.001) | +28.2 (p=<0.001) | +23.7 (p=<0.001) | +0.6 (p=0.294) | +4.9 (p=<0.001) | -0.0 (p=0.983) | +2.2 (p=<0.001) |
| rungB_step300 | +0.3 (p=0.625) | +4.7 (p=<0.001) | +27.7 (p=<0.001) | +23.1 (p=<0.001) | +0.4 (p=0.513) | +1.4 (p=0.052) | +0.8 (p=0.032) | +1.6 (p=<0.001) |
| rungC_step300 | -0.0 (p=0.959) | +4.4 (p=<0.001) | +21.6 (p=<0.001) | +26.5 (p=<0.001) | +0.4 (p=0.453) | +5.3 (p=<0.001) | +0.1 (p=0.846) | -1.0 (p=0.021) |
| rungD_step300 | -0.0 (p=0.980) | +4.1 (p=<0.001) | +26.1 (p=<0.001) | +21.6 (p=<0.001) | +1.0 (p=0.059) | -1.5 (p=0.036) | +0.1 (p=0.728) | -0.5 (p=0.243) |
| rungA1_step300 | +0.0 (p=0.960) | +4.6 (p=<0.001) | +30.9 (p=<0.001) | +25.7 (p=<0.001) | +0.3 (p=0.666) | +4.4 (p=<0.001) | -0.3 (p=0.487) | -1.2 (p=0.004) |
| rungA1s2_step300 | -0.6 (p=0.323) | -1.5 (p=0.036) | +4.3 (p=0.006) | -40.4 (p=<0.001) | -0.1 (p=0.814) | -34.0 (p=<0.001) | +0.1 (p=0.828) | -3.8 (p=<0.001) |
| DSs2_step200 | -0.3 (p=0.643) | -0.5 (p=0.476) | +5.8 (p=<0.001) | -38.5 (p=<0.001) | -0.8 (p=0.163) | -32.6 (p=<0.001) | -0.4 (p=0.333) | -2.7 (p=<0.001) |
| PPO_step400 | +0.4 (p=0.543) | -0.1 (p=0.876) | +6.7 (p=<0.001) | +1.6 (p=0.390) | -1.5 (p=0.010) | -19.3 (p=<0.001) | +0.0 (p=0.967) | -2.1 (p=<0.001) |
| PPO_klF_ent0_lr1e-6_step520 | -10.9 (p=<0.001) | +0.2 (p=0.809) | +28.8 (p=<0.001) | +13.9 (p=<0.001) | -9.0 (p=<0.001) | -5.1 (p=<0.001) | -1.7 (p=<0.001) | -0.5 (p=0.283) |
| PPO_klF_ent0_lr3e-6_step520 | -17.2 (p=<0.001) | -15.3 (p=<0.001) | -4.6 (p=0.008) | -23.1 (p=<0.001) | -43.1 (p=<0.001) | -36.7 (p=<0.001) | -14.5 (p=<0.001) | -16.1 (p=<0.001) |
| PPO_klF_ent1e-3_lr3e-6_step520 | -29.4 (p=<0.001) | -10.5 (p=<0.001) | -32.4 (p=<0.001) | -47.1 (p=<0.001) | -52.6 (p=<0.001) | -45.3 (p=<0.001) | -31.4 (p=<0.001) | -13.6 (p=<0.001) |
| PPO_kl0_ent1e-3_lr3e-6_step520 | -12.2 (p=<0.001) | -2.4 (p=<0.001) | -13.0 (p=<0.001) | -26.8 (p=<0.001) | -43.2 (p=<0.001) | -39.9 (p=<0.001) | -14.6 (p=<0.001) | -7.0 (p=<0.001) |
| PPO_kl1e-3_ent1e-3_lr3e-6_step520 | -7.0 (p=<0.001) | -6.5 (p=<0.001) | -0.6 (p=0.756) | -20.1 (p=<0.001) | -42.8 (p=<0.001) | -39.6 (p=<0.001) | -13.3 (p=<0.001) | -11.5 (p=<0.001) |
| GRPO_n2_noKL_step250 | -0.6 (p=0.360) | -0.8 (p=0.273) | +10.7 (p=<0.001) | -37.4 (p=<0.001) | -2.2 (p=<0.001) | -31.6 (p=<0.001) | +0.0 (p=0.967) | -1.8 (p=<0.001) |
| GRPO_n2_kl1e-3_step250 | +0.2 (p=0.720) | -0.9 (p=0.240) | +16.2 (p=<0.001) | -38.7 (p=<0.001) | -0.5 (p=0.382) | -32.8 (p=<0.001) | -0.1 (p=0.880) | -2.3 (p=<0.001) |
| GRPO_n2_kl5e-3_step250 | +0.8 (p=0.171) | -1.4 (p=0.051) | +13.8 (p=<0.001) | -38.8 (p=<0.001) | -0.9 (p=0.129) | -31.5 (p=<0.001) | +0.1 (p=0.784) | -2.4 (p=<0.001) |
| GRPO_n2_kl4e-2_step250 | -0.2 (p=0.766) | +1.3 (p=0.033) | +36.2 (p=<0.001) | +9.2 (p=<0.001) | +0.2 (p=0.706) | +4.4 (p=<0.001) | +0.8 (p=0.026) | +2.2 (p=<0.001) |
| GRPO_n2_noKL_step520 | -0.6 (p=0.374) | -0.2 (p=0.772) | +5.1 (p=0.003) | -40.1 (p=<0.001) | -2.9 (p=<0.001) | -31.3 (p=<0.001) | +0.7 (p=0.117) | -1.8 (p=<0.001) |
| GRPO_n2_kl1e-3_step520 | +0.1 (p=0.922) | -0.5 (p=0.476) | +12.0 (p=<0.001) | -37.5 (p=<0.001) | +0.0 (p=0.971) | -32.4 (p=<0.001) | +0.5 (p=0.192) | -2.4 (p=<0.001) |
| GRPO_n2_kl5e-3_step520 | +0.5 (p=0.419) | -3.1 (p=<0.001) | +22.7 (p=<0.001) | -37.2 (p=<0.001) | -1.2 (p=0.037) | -30.8 (p=<0.001) | +0.5 (p=0.182) | -1.9 (p=<0.001) |
| GRPO_n2_kl1e-2_step250 | -0.3 (p=0.573) | +4.3 (p=<0.001) | +23.1 (p=<0.001) | +24.4 (p=<0.001) | +0.9 (p=0.075) | +5.2 (p=<0.001) | +0.1 (p=0.839) | +2.1 (p=<0.001) |
| GRPO_n2_kl4e-2_step520 | -0.5 (p=0.354) | +2.5 (p=<0.001) | +31.8 (p=<0.001) | +21.0 (p=<0.001) | +0.7 (p=0.173) | +4.8 (p=<0.001) | -0.4 (p=0.306) | +1.6 (p=<0.001) |
| GRPO_n4_kl1e-3_step520 | -0.8 (p=0.233) | -0.9 (p=0.199) | +16.6 (p=<0.001) | -38.1 (p=<0.001) | -3.0 (p=<0.001) | -31.7 (p=<0.001) | +0.5 (p=0.272) | -1.7 (p=<0.001) |
| GRPO_n8_kl1e-3_step520 | +0.6 (p=0.349) | -0.2 (p=0.757) | +30.2 (p=<0.001) | -37.5 (p=<0.001) | -2.5 (p=<0.001) | -31.5 (p=<0.001) | +0.2 (p=0.584) | -2.1 (p=<0.001) |
| GRPO_n4_kl1e-6_step520 | -0.8 (p=0.190) | -1.2 (p=0.095) | +17.7 (p=<0.001) | -36.5 (p=<0.001) | -1.7 (p=0.003) | -31.9 (p=<0.001) | +0.3 (p=0.429) | -2.0 (p=<0.001) |
| GRPO_n4_kl1e-6_eqR_step520 | +0.0 (p=0.960) | -2.1 (p=0.002) | +33.1 (p=<0.001) | -29.8 (p=<0.001) | -1.4 (p=0.017) | -29.0 (p=<0.001) | +0.6 (p=0.126) | -3.3 (p=<0.001) |

### 3.6 DAPO w/o DS vs. w/ DS (paired, strict)

| tokens | benchmark | prompt | noDS | DS | diff | noDS_only | DS_only | p |
|---|---|---|---|---|---|---|---|---|
| 8192 | GSM8K | standard | 69.9 | 66.1 | 3.8 | 185 | 135 | 0.006 |
| 8192 | GSM8K | think | 75.7 | 74.4 | 1.3 | 119 | 102 | 0.282 |
| 8192 | MATH | standard | 54.9 | 55.1 | -0.2 | 424 | 436 | 0.708 |
| 8192 | MATH | think | 49.5 | 46.5 | 3.0 | 662 | 511 | <0.001 |
| 8192 | BBH | standard | 34.4 | 35.1 | -0.7 | 803 | 851 | 0.248 |
| 8192 | BBH | think | 41.5 | 40.2 | 1.3 | 1047 | 963 | 0.064 |
| 8192 | MMLU-Pro | standard | 32.9 | 33.1 | -0.1 | 1146 | 1164 | 0.724 |
| 8192 | MMLU-Pro | think | 29.7 | 26.1 | 3.5 | 1530 | 1104 | <0.001 |
| 2048 | GSM8K | standard | 69.1 | 65.9 | 3.3 | 192 | 149 | 0.023 |
| 2048 | GSM8K | think | 75.9 | 73.3 | 2.6 | 128 | 94 | 0.027 |
| 2048 | MATH | standard | 54.8 | 55.3 | -0.5 | 410 | 437 | 0.372 |
| 2048 | MATH | think | 49.6 | 46.6 | 3.0 | 656 | 508 | <0.001 |
| 2048 | BBH | standard | 35.0 | 35.2 | -0.2 | 820 | 833 | 0.768 |
| 2048 | BBH | think | 40.8 | 40.3 | 0.5 | 1024 | 989 | 0.449 |
| 2048 | MMLU-Pro | standard | 32.3 | 33.4 | -1.0 | 1103 | 1227 | 0.011 |
| 2048 | MMLU-Pro | think | 28.7 | 26.2 | 2.5 | 1462 | 1160 | <0.001 |

### 3.7 GRPO KL sweep at a matched step (n = 2, step 250)

| model | KL coef | GSM8K standard | GSM8K think | MATH standard | MATH think | BBH standard | BBH think | MMLU-Pro standard | MMLU-Pro think | avg (8) | avg std | avg think |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRPO_n2_noKL_step250 | KL off | 44.0 | 11.1 | 52.9 | 15.8 | 34.6 | 35.4 | 33.1 | 25.7 | 31.6 | 41.2 | 22.0 |
| GRPO_n2_kl1e-3_step250 | KL 0.001 | 49.6 | 9.8 | 54.6 | 14.6 | 35.4 | 35.4 | 33.0 | 25.2 | 32.2 | 43.1 | 21.2 |
| GRPO_n2_kl5e-3_step250 | KL 0.005 | 47.2 | 9.6 | 54.3 | 15.9 | 36.0 | 34.8 | 33.2 | 25.1 | 32.0 | 42.6 | 21.3 |
| GRPO_n2_kl1e-2_step250 | KL 0.01 | 56.5 | 72.9 | 56.0 | 52.5 | 34.8 | 40.5 | 33.1 | 29.6 | 47.0 | 45.1 | 48.9 |
| GRPO_n2_kl4e-2_step250 | KL 0.04 | 69.5 | 57.7 | 55.3 | 51.7 | 35.0 | 37.5 | 33.9 | 29.7 | 46.3 | 48.4 | 44.1 |

Median response length (characters):

| model | bbh standard | bbh think | gsm8k standard | gsm8k think | math standard | math think | mmlupro standard | mmlupro think |
|---|---|---|---|---|---|---|---|---|
| GRPO_n2_noKL_step250 | 551 | 22 | 669 | 21 | 1271 | 21 | 1209 | 20 |
| GRPO_n2_kl1e-3_step250 | 650 | 22 | 790 | 21 | 1313 | 21 | 1587 | 20 |
| GRPO_n2_kl5e-3_step250 | 636 | 22 | 740 | 21 | 1299 | 21 | 1538 | 20 |
| GRPO_n2_kl1e-2_step250 | 760 | 907 | 862 | 741 | 1314 | 1231 | 1830 | 1895 |
| GRPO_n2_kl4e-2_step250 | 779 | 769 | 902 | 729 | 1345 | 1300 | 1790 | 1671 |

### 3.8 DAPO w/o DS checkpoint series (GSM8K and MATH)

Same run (3md2pzp8) evaluated at several steps, to relate in-domain Countdown accuracy to transfer. Countdown validation accuracy is the run's own logged value (sampled, top-p 0.7) at that step.

| model | step | Countdown val acc | GSM8K standard | GSM8K think | MATH standard | MATH think |
|---|---|---|---|---|---|---|
| noDS_step100 | 100 | 68.8 | 69.9 | 75.1 | 55.9 | 51.8 |
| noDS_step200 | 200 | 73.8 | 70.7 | 76.0 | 55.1 | 50.9 |
| noDS_step300 | 300 | 75.6 | 69.9 | 75.7 | 54.9 | 49.5 |
| noDS_step400 | 400 | 78.0 | 67.8 | 73.5 | 54.2 | 48.4 |
| noDS_step500 | 500 | 79.2 | 66.7 | 75.5 | 52.9 | 48.6 |
| noDS_step520 | 520 | 78.8 | 67.6 | 75.2 | 53.0 | 48.3 |

### 3.9 Why GSM8K gains a lot and MATH does not (standard prompt)

Each trained model's change over the base model is split item by item using tolerant correctness first. *Tolerant fix*: base wrong and trained correct under tolerant scoring. *Tolerant loss*: base correct and trained wrong under tolerant scoring. *Marker win/loss*: both responses are tolerant-correct, but the trained/base response alone satisfies the strict final-answer-marker rule. Percent of items. Full analysis, including an arithmetic-slip check: `analysis_gsm8k_vs_math.py` / `.txt`.

| benchmark | model | total gain | tolerant fixes | tolerant losses | net tolerant gain | net 95% CI | marker wins | marker losses | flexible p |
|---|---|---|---|---|---|---|---|---|---|
| GSM8K | DS_step300 | 32.8 | 7.1 | 6.3 | 0.8 | [-1.21, 2.73] | 30.9 | 1.5 | 0.498 |
| GSM8K | noDS_step300 | 36.5 | 10.3 | 7.0 | 3.3 | [1.10, 5.57] | 31.8 | 1.7 | 0.004 |
| GSM8K | GRPO_n2_kl4e-2_step250 | 36.2 | 7.1 | 4.0 | 3.1 | [1.31, 4.90] | 33.1 | 1.4 | <0.001 |
| GSM8K | GRPO_n2_kl1e-2_step250 | 23.1 | 7.1 | 5.8 | 1.2 | [-0.72, 3.15] | 22.9 | 3.0 | 0.250 |
| GSM8K | PPO_klF_ent0_lr1e-6_step520 | 28.8 | 7.9 | 17.4 | -9.6 | [-12.22, -6.89] | 28.8 | 0.2 | <0.001 |
| MATH | DS_step300 | -0.0 | 7.8 | 7.9 | -0.1 | [-1.22, 0.98] | 0.1 | 0.0 | 0.858 |
| MATH | noDS_step300 | -0.3 | 8.0 | 8.4 | -0.4 | [-1.50, 0.74] | 0.1 | 0.0 | 0.529 |
| MATH | GRPO_n2_kl4e-2_step250 | 0.2 | 5.7 | 5.6 | 0.1 | [-0.83, 1.03] | 0.1 | 0.0 | 0.866 |
| MATH | GRPO_n2_kl1e-2_step250 | 0.9 | 6.5 | 5.7 | 0.8 | [-0.17, 1.77] | 0.1 | 0.0 | 0.114 |
| MATH | PPO_klF_ent0_lr1e-6_step520 | -9.0 | 6.4 | 15.5 | -9.1 | [-10.39, -7.85] | 0.1 | 0.0 | <0.001 |

Final-answer markers written (standard prompt, % of responses):

| model | benchmark | writes #### | writes \boxed{} | neither |
|---|---|---|---|---|
| base | GSM8K | 17.8 | 28.2 | 54.4 |
| base | MATH | 0.3 | 95.2 | 4.8 |
| noDS_step300 | GSM8K | 70.7 | 24.5 | 8.0 |
| noDS_step300 | MATH | 0.2 | 94.3 | 5.7 |
| GRPO_n2_kl4e-2_step250 | GSM8K | 59.4 | 36.8 | 7.7 |
| GRPO_n2_kl4e-2_step250 | MATH | 0.4 | 96.0 | 3.9 |

The base model already writes `\boxed{}` in 95% of MATH responses (Qwen2.5-Instruct's math convention), so MATH has no format gain to collect, at any difficulty level. GSM8K's `####` is dataset-specific and the base model omits any marker in 54% of responses; Countdown RL, whose reward requires an explicit final answer, teaches the models to give one. Arithmetic slips are rare in base-model GSM8K errors (about 4%) and are not reduced by training (explicit calculations wrong 2.7% base vs. 2.0-2.3% for the best recipes).

### 3.10 Controlled dynamic-sampling test: ablation rung 2 vs. DS

Rung 2 has the DS run's exact settings (KL off, entropy 0, clip 0.2/0.28, lr 1e-6, n = 8, 4,096-token cap, Sep reward) and the same trained batch (32 prompts, 1 update per step); only dynamic sampling is off (`../ablation/README.md`). The old no-DS run is shown for reference (192 prompts, 6 updates per step).

Training curves (Countdown):

| step | Countdown val acc: DS (f8h77wol) | Countdown val acc: old no-DS (192/step) | Countdown val acc: rung 1 (no DS, clip 0.2) | Countdown val acc: rung 2 (no DS, 32/step) | mean response tokens: DS (f8h77wol) | mean response tokens: old no-DS (192/step) | mean response tokens: rung 1 (no DS, clip 0.2) | mean response tokens: rung 2 (no DS, 32/step) |
|---|---|---|---|---|---|---|---|---|
| 25 | 14 | 18 | 2 | 2 | 347 | 228 | 101 | 130 |
| 50 | 31 | 51 | 16 | 14 | 708 | 866 | 49 | 76 |
| 100 | 53 | 69 | 20 | 23 | 1649 | 1014 | 49 | 66 |
| 300 | 65 | 76 | 26 | 25 | 2577 | 1149 | 47 | 54 |
| 500 |  | 79 | 28 | 30 | 2276 | 1280 | 51 | 54 |

Benchmarks at step 300, paired (strict, 8,192 tokens; positive = DS higher):

| benchmark | prompt | DS | rung 2 | DS - rung 2 | DS only | rung 2 only | p |
|---|---|---|---|---|---|---|---|
| GSM8K | standard | 66.1 | 61.5 | 4.6 | 218 | 157 | 0.002 |
| GSM8K | think | 74.4 | 5.9 | 68.5 | 915 | 12 | <0.001 |
| MATH | standard | 55.1 | 54.1 | 1.0 | 466 | 416 | 0.099 |
| MATH | think | 46.5 | 14.5 | 32.0 | 1838 | 237 | <0.001 |
| BBH | standard | 35.1 | 35.5 | -0.4 | 796 | 824 | 0.502 |
| BBH | think | 40.2 | 36.5 | 3.7 | 1226 | 985 | <0.001 |
| MMLU-Pro | standard | 33.1 | 33.0 | 0.0 | 1196 | 1192 | 0.951 |
| MMLU-Pro | think | 26.1 | 24.2 | 1.9 | 1518 | 1289 | <0.001 |

Clip-higher without DS: rung 2 (clip 0.28) vs. rung 1 (clip 0.2), step 300, paired (strict; positive = rung 2 higher):

| benchmark | prompt | rung 2 | rung 1 | rung 2 - rung 1 | p |
|---|---|---|---|---|---|
| GSM8K | standard | 61.5 | 56.1 | 5.4 | <0.001 |
| GSM8K | think | 5.9 | 5.2 | 0.8 | 0.353 |
| MATH | standard | 54.1 | 53.7 | 0.4 | 0.446 |
| MATH | think | 14.5 | 11.7 | 2.8 | <0.001 |
| BBH | standard | 35.5 | 33.7 | 1.8 | 0.002 |
| BBH | think | 36.5 | 35.2 | 1.4 | 0.016 |
| MMLU-Pro | standard | 33.0 | 33.1 | -0.1 | 0.881 |
| MMLU-Pro | think | 24.2 | 20.5 | 3.7 | <0.001 |

### 3.11 Getting DS's benefit without resampling: options A, B, C (step 300)

Each option is ablation rung 2 (no DS, 32 prompts generated and trained per step) with one change: A = correctness-only reward (no 0.1 format credit); B = advantages not divided by the group std; C = groups whose 8 answers are all right or all wrong are masked out of the loss instead of resampled; D = batch-level advantage normalisation (group mean subtracted, then whitened over the batch; Lite PPO, arXiv:2508.08221). Motivation (`../ablation/analysis_why_ds.py`): in rung 2's first 25 steps, 74-87% of the training signal came from groups where every answer was wrong and scores differed only by the 0.1 format credit, which std normalisation turns into full-size advantages; DS drops those groups because it filters on accuracy.

| model | label | Countdown val acc (%) | mean response tokens | benchmark avg std | benchmark avg think |
|---|---|---|---|---|---|
| DS_step300 | DAPO w/ DS, n=8 (step 300) | 66.3 | 2590.5 | 47.4 | 46.8 |
| rung2_step300 | Ablation rung2: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 25.0 | 54.0 | 46.0 | 20.3 |
| rung2s2_step300 | Ablation rung 2, seed 2: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 50.7 | 718.1 | 42.5 | 46.9 |
| rungA_step300 | Ablation rung 2 + correctness-only reward: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 68.9 | 1503.7 | 46.5 | 48.8 |
| rungB_step300 | Ablation rung 2 + no std normalisation: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 62.1 | 1705.4 | 46.5 | 47.6 |
| rungC_step300 | Ablation rung 2 + masked uninformative groups: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 65.7 | 1605.0 | 44.7 | 48.7 |
| rungD_step300 | Ablation rung 2 + batch-level advantage normalisation: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 61.8 | 1796.8 | 46.0 | 45.8 |
| rungA1_step300 | Ablation rung 2 + correctness-only reward + overlong filtering: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 65.7 | 1712.7 | 46.9 | 48.3 |
| rungA1s2_step300 | Ablation rung 2 + correctness-only reward + overlong filtering, seed 2: clip 0.2/0.28, no DS, 32 prompts/step (step 300) | 27.2 | 18.8 | 40.1 | 19.9 |

### 3.12 pass@k (does RL solve more problems, or the same ones more reliably?)

Following Yue et al. (arXiv:2504.13837): standard prompts, temperature 0.6, top-p 0.95, up to 4,096 tokens; MATH-500 with 64 samples per problem, GSM8K with 32; unbiased pass@k estimator; strict and tolerant scoring (`eval_passk.py`, `results_passk/`). Similarity at the largest sampled k is evidence consistent with probability redistribution, not proof that the models have identical latent solution support.

MATH-500 (strict, %):

| model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 |
|---|---|---|---|---|---|---|---|
| DS_step300 | 55.2 | 64.8 | 72.8 | 79.4 | 84.8 | 89.2 | 92.2 |
| GRPO_n2_kl4e-2_step250 | 54.4 | 64.5 | 72.6 | 79.2 | 84.6 | 88.7 | 91.6 |
| GRPO_n2_kl4e-2_step520 | 53.8 | 64.0 | 72.3 | 79.1 | 84.4 | 88.4 | 91.2 |
| GRPO_n2_noKL_step520 | 50.6 | 61.6 | 70.3 | 77.3 | 82.8 | 86.9 | 90.2 |
| PPO_klF_ent0_lr1e-6_step520 | 45.3 | 54.2 | 62.2 | 69.6 | 76.2 | 81.8 | 85.6 |
| base | 54.0 | 64.1 | 72.2 | 78.8 | 83.9 | 88.0 | 91.0 |
| noDS_step300 | 54.1 | 63.8 | 71.7 | 78.0 | 83.1 | 87.3 | 90.8 |
| rung2_step300 | 52.9 | 63.5 | 72.0 | 78.7 | 84.2 | 88.8 | 92.2 |
| rungA1_step300 | 54.6 | 64.3 | 72.2 | 78.6 | 83.8 | 87.6 | 90.0 |
| rungA_step300 | 54.3 | 63.9 | 72.0 | 78.7 | 84.0 | 87.8 | 90.8 |
| rungD_step300 | 54.5 | 64.2 | 72.2 | 78.8 | 84.2 | 88.6 | 92.2 |

GSM8K (strict, %):

| model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
|---|---|---|---|---|---|---|
| DS_step300 | 63.8 | 77.9 | 86.5 | 91.5 | 94.6 | 96.4 |
| GRPO_n2_kl4e-2_step250 | 68.0 | 80.6 | 88.0 | 92.5 | 95.2 | 97.0 |
| GRPO_n2_kl4e-2_step520 | 64.7 | 78.6 | 86.8 | 91.7 | 94.6 | 96.7 |
| GRPO_n2_noKL_step520 | 40.3 | 58.2 | 73.9 | 84.8 | 91.2 | 94.5 |
| PPO_klF_ent0_lr1e-6_step520 | 63.0 | 71.6 | 77.8 | 82.2 | 85.6 | 88.4 |
| base | 36.7 | 55.8 | 73.6 | 85.2 | 91.1 | 94.1 |
| noDS_step300 | 67.8 | 80.5 | 87.7 | 91.8 | 94.6 | 96.3 |
| rung2_step300 | 60.2 | 76.1 | 86.0 | 91.3 | 94.4 | 96.1 |
| rungA1_step300 | 62.1 | 77.4 | 86.3 | 91.1 | 94.0 | 96.0 |
| rungA_step300 | 59.3 | 75.0 | 84.8 | 90.4 | 93.8 | 96.1 |
| rungD_step300 | 58.4 | 74.9 | 85.2 | 90.6 | 93.7 | 95.5 |


### 3.13 Collapse screen: how often does each recipe collapse? (first 100 steps)

Collapsed = mean response length over steps 90-100 < 100 tokens. Seeds 1-2: existing full-length runs; seeds 3-6: 100-step screen runs (`chain_collapse_screen.sh`).

| recipe | collapse rate |
|---|---|
| A1 | 5/6 collapsed |
| A1 + KL 0.01 | 3/4 collapsed |
| DS | 5/6 collapsed |
| no DS (rung 2) | 3/6 collapsed |

| recipe | seed | run | wandb | len_step25 | len_step50 | len_step90_100 | val_acc_step100 | status |
|---|---|---|---|---|---|---|---|---|
| no DS (rung 2) | 1 | rung2-clip0.28-noDS-gen32-train32-n8-resp4096 | 5vmxmx4q | 115.0 | 77.0 | 62.0 | 22.4 | collapsed |
| no DS (rung 2) | 2 | rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed2 | iic956n7 | 147.0 | 215.0 | 230.0 | 18.8 | healthy |
| no DS (rung 2) | 3 | rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed3-s100 | f346631t | 131.0 | 245.0 | 1007.0 | 21.8 | healthy |
| no DS (rung 2) | 4 | rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed4-s100 | x2djfjpf | 138.0 | 340.0 | 498.0 | 25.6 | healthy |
| no DS (rung 2) | 5 | rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed5-s100 | 2pc6931g | 127.0 | 55.0 | 42.0 | 20.3 | collapsed |
| no DS (rung 2) | 6 | rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed6-s100 | fx6ntgy6 | 173.0 | 75.0 | 53.0 | 20.1 | collapsed |
| DS | 1 | Qwen2.5-1.5B-G8-acc-filtering-tokenlevel-bsz-32-0910 | f8h77wol | 361.0 | 678.0 | 1569.0 | 50.8 | healthy |
| DS | 2 | rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed2 | 359gx7xo | 20.0 | 21.0 | 20.0 | 26.2 | collapsed |
| DS | 3 | rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed3-s100 | cuu3i1xj | 21.0 | 20.0 | 21.0 | 27.9 | collapsed |
| DS | 4 | rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed4-s100 | 4xma3zki | 19.0 | 20.0 | 21.0 | 26.6 | collapsed |
| DS | 5 | rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed5-s100 | x43gbqkj | 20.0 | 20.0 | 20.0 | 28.7 | collapsed |
| DS | 6 | rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed6-s100 | xa0ykms7 | 19.0 | 19.0 | 19.0 | 28.8 | collapsed |
| A1 | 1 | rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096 | bynfpzjt | 252.0 | 996.0 | 2224.0 | 41.6 | healthy |
| A1 | 2 | rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed2 | rsa2qzj9 | 23.0 | 24.0 | 20.0 | 24.6 | collapsed |
| A1 | 3 | rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed3-s100 | 00syzabd | 25.0 | 19.0 | 20.0 | 22.0 | collapsed |
| A1 | 4 | rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed4-s100 | q2cih0cw | 255.0 | 44.0 | 21.0 | 22.9 | collapsed |
| A1 | 5 | rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed5-s100 | m5lgjg52 | 23.0 | 22.0 | 20.0 | 19.6 | collapsed |
| A1 | 6 | rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed6-s100 | wghm4mct | 23.0 | 19.0 | 20.0 | 25.7 | collapsed |
| A1 + KL 0.01 | 1 | rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed1-s100 |  |  |  |  |  | not run |
| A1 + KL 0.01 | 2 | rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed2-s100 |  |  |  |  |  | not run |
| A1 + KL 0.01 | 3 | rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed3-s100 | xtwilh0l | 188.0 | 21.0 | 21.0 | 23.4 | collapsed |
| A1 + KL 0.01 | 4 | rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed4-s100 | wl6bdjat | 21.0 | 23.0 | 132.0 | 25.3 | healthy |
| A1 + KL 0.01 | 5 | rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed5-s100 | pc748q7c | 59.0 | 86.0 | 19.0 | 22.8 | collapsed |
| A1 + KL 0.01 | 6 | rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed6-s100 | mfreek0z | 95.0 | 44.0 | 21.0 | 23.5 | collapsed |

Note: DS seed 1 is the only DS run trained in the original Sep-2025 environment (old `verl` conda env, original local base-model copy, original reward code); all other ablation runs used the current setup (verl-fork env, Hugging Face Qwen2.5-1.5B-Instruct, reconstructed reward). Step-0 Countdown accuracy matched (3.0%), and healthy runs also occur in the current setup (A1 seed 1, no-DS seeds 2-4), but an environment effect on DS seed 1 cannot be ruled out.

## 4. Results, 2048 output tokens (base, DS, noDS)

| model | bbh standard | bbh think | gsm8k standard | gsm8k think | math standard | math think | mmlupro standard | mmlupro think |
|---|---|---|---|---|---|---|---|---|
| base | 35.5 | 35.6 | 32.9 | 48.4 | 55.2 | 47.3 | 33.0 | 27.9 |
| DS_step300 | 35.2 | 40.3 | 65.9 | 73.3 | 55.3 | 46.6 | 33.4 | 26.2 |
| noDS_step300 | 35.0 | 40.8 | 69.1 | 75.9 | 54.8 | 49.6 | 32.3 | 28.7 |

Truncated at 2048 tokens (%):

| model | bbh standard | bbh think | gsm8k standard | gsm8k think | math standard | math think | mmlupro standard | mmlupro think |
|---|---|---|---|---|---|---|---|---|
| base | 3.7 | 4.8 | 0.6 | 0.3 | 5.5 | 5.9 | 4.0 | 8.2 |
| DS_step300 | 3.5 | 4.1 | 0.8 | 0.8 | 6.7 | 8.4 | 6.4 | 11.3 |
| noDS_step300 | 5.1 | 7.9 | 0.9 | 0.8 | 7.4 | 11.8 | 7.1 | 11.6 |

## 5. Main findings

See section 0 (Summary) at the top of this file.

## 6. Files

- Per-response outputs: `results_8k/<bench>/<model>__<prompt>.jsonl` (GSM8K) and `results_8k/<bench>/scored/<model>__<prompt>.jsonl` (others, with extracted answers).
- Summaries: `results_8k/<bench>/summary.txt|csv`, per-group breakdowns `by_subject.csv`, `by_task.csv`, `by_category.csv`; all models in one file: `all_models_8k.csv`.
- Training settings: `runs_registry.md|csv` (`run_registry.py`).
- Paper table: `../paper_figs/table2.tex` (`make_table2.py`); 2048 vs 8192 comparison: `compare_lengths.txt` (`compare_lengths.py`).
- Scripts: `merge_checkpoints.sh`, `run_model'sh`, `run_queue.sh`, `run_all.sh`, `eval_{gsm8k,math,bbh,mmlupro}.py`, `common.py`.
