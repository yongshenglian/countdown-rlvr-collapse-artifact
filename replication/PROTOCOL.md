# Prospective Continuation Protocol: Frozen-Environment Multi-Seed Replication

Protocol frozen: 12 October 2026, before any run was continued beyond its archived step-100 prefix.

This is a **prospective continuation protocol**, not a preregistration of the first 100 steps. The step-100
outcomes for the DAPO-style seeds were already inspected in the collapse screen. No terminal-step downstream
results for seeds 3--5 had been generated when this protocol was written. That distinction will be stated in
the manuscript.

## Objective

Estimate seed-level stability and downstream outcomes for principal Countdown RL recipes in one frozen
software environment, without selecting only surviving runs.

## Frozen environment and inputs

- GPUs: four NVIDIA RTX A6000; driver 580.178.04.
- Python 3.10.0; PyTorch 2.6.0+cu124; vLLM 0.8.5.post1; Transformers 4.51.1; Ray 2.48.0.
- Verl source commit: `6abe50ab0a84237d9d8a733d365c3b983bcf647d`.
- Paper/evaluation repository starting commit: `380a25ff53afd660fb45d13daa860b5b27379150`,
  plus the documented uncommitted Round-2 revisions.
- Base model: Qwen2.5-1.5B-Instruct snapshot
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Base-model `config.json` SHA-256:
  `98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670`.
- Countdown train parquet SHA-256:
  `a8508c1531cac14162327f97033572949905bb5d24a6993666813fbde9d12f8d`.
- Countdown test parquet SHA-256:
  `63b631a38143f0d79090919be96f40848178f1befcb9eb100b1f46d6d22b6898`.
- Reward implementations, prompts, and trainer options are fixed by the versioned launch scripts. No software,
  prompt, reward, or scorer change is permitted after the first continuation is launched. A necessary runtime
  repair must be documented and applied identically to every not-yet-started run.

## Recipes and seeds

The replication seed set is 3, 4, and 5 for every recipe. Both `data.seed` and the vLLM rollout seed are set
to that value.

1. **Matched no-DS DAPO style**: `ablation/rung2_clip0.28_noDS.sh`.
2. **Dynamic-sampling DAPO style**: `ablation/rung3_DS_gen192.sh`.
3. **Strong-KL GRPO**: the recorded headline recipe with group size 2, KL-loss coefficient 0.04, actor
   learning rate 1e-6, global token-mean aggregation, 2,048-token response cap, and the archived naive
   Countdown reward (1.0 correct, 0.1 extracted-but-wrong, 0 otherwise).
4. **PPO**: the recorded conservative PPO recipe with GAE, actor learning rate 1e-6, KL loss off, entropy
   coefficient 0, 2,048-token response cap, and the same archived naive Countdown reward.

The DAPO-style seeds resume the exact `global_step_100` optimizer/model/extra checkpoints produced for the
collapse screen. Their learning-rate schedule was already configured for 520 steps. Increasing
`trainer.total_epochs` only prevents data-loader exhaustion before the fixed step horizon; it does not alter
the optimizer schedule or the first 100 steps. GRPO and PPO start from the same pinned base snapshot.

## Fixed horizon and checkpoint rule

- Planned horizon: **520 optimizer steps**.
- Primary checkpoint: exactly `global_step_520`.
- If the training procedure terminates before step 520 (including data exhaustion or numerical failure), that
  is an unconditional procedure failure. The last valid checkpoint is retained and reported as terminal; it
  is not replaced by another seed or an earlier best-performing checkpoint.
- Runtime interruption is resumed from the latest checkpoint without changing hyperparameters.
- No stopping based on validation reward, response length, or downstream benchmark performance.
- Checkpoints are saved every 50 steps.

## Outcomes

Primary outcomes, fixed before continuation:

1. terminal standard-prompt, format-tolerant GSM8K accuracy;
2. terminal mean standard-prompt, format-tolerant accuracy over GSM8K, MATH, BBH, and MMLU-Pro;
3. unconditional probability of reaching step 520 without early short-response collapse.

Secondary outcomes:

- strict accuracy and marker rate;
- Countdown validation accuracy/reward;
- response length, truncation, entropy, KL, and time to collapse;
- survivor-conditional downstream accuracy;
- empirical pass@k only for a separately identified fixed checkpoint, if compute permits.

Early short-response collapse retains the manuscript's primary definition: mean response length below 100
tokens over steps 90--100. Time to collapse is the first logged evaluation window ending at or after step 10
whose mean response length is below 100 tokens and remains below 100 for the next two logged windows. Runs that
never satisfy this rule are right-censored at their terminal step. Threshold sensitivity at 50, 100, and 200
tokens will also be reported.

## Analysis and reporting

- Show every seed; do not average away bimodality.
- Report unconditional outcomes for all three seeds and survivor-conditional outcomes separately.
- Report seed-level means and intervals as descriptive due to n=3; do not use benchmark items as the
  inferential unit for recipe-level claims.
- Item-level paired intervals/tests may be reported only as fixed-checkpoint diagnostics and must be labeled
  separately from seed uncertainty.
- Collapsed, failed, or early-terminated runs remain in denominators.
- No recipe ranking will be called confirmatory. The first 100 DAPO-style steps were previously observed.

## Execution order

To limit simultaneous sources of failure, runs execute one at a time on all four GPUs:

1. matched no-DS seeds 3, 4, 5 (resume step 100);
2. DS seeds 3, 4, 5 (resume step 100);
3. strong-KL GRPO seeds 3, 4, 5;
4. PPO seeds 3, 4, 5;
5. merge fixed checkpoints, run the frozen evaluation pipeline, and generate seed-level tables/figures.

The execution log and any deviations will be appended below without rewriting the frozen rules above.

## Execution log

- 12 October 2026: protocol created; all six DAPO-style step-100 prefix checkpoints verified present.
- 12 October 2026: matched no-DS seed 3 submitted as
  `rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed3-s100-20261012-062302`; checkpoint-100 model,
  optimizer, and extra-state restoration verified in the live log.
- 12 October 2026: persistent serial controller started. Before any GRPO/PPO replication launch, their
  scorer was vendored as `replication/reward_naive_countdown.py` because the live uncommitted Verl scorer
  had changed since the archived October-2025 runs. The frozen scorer values are established directly by
  those archived console logs; this prevents an unintended reward change rather than altering the recipe.
- 13 October 2026: matched no-DS seed 3 completed the fixed step-520 horizon successfully. Its terminal
  Countdown validation accuracy was 0.641, validation reward was 0.676, mean response length was 1,413.1
  tokens, and truncation fraction was 0.004. The run was endpoint-healthy under the already observed
  steps-90--100 rule. A compact 421-row training trajectory was extracted to
  `replication/results/noDS_seed3_training_metrics.csv`; downstream evaluation remains deferred until the
  predeclared training queue completes.
- 13 October 2026: after the completed seed-3 continuation, the author suspended the remaining serial queue
  to limit additional compute. No additional continuation is retained or included in the analysis.
- 13 October 2026: following reviewer concern that the five equal-seed current-environment DS runs could be
  separated from the healthy legacy DS run by environment rather than seed, one targeted 100-step replay was
  launched in the frozen current environment. It uses the recorded legacy DS configuration and the legacy
  defaults (`data.seed=1`, rollout seed 0), rather than the later screen convention that set both seed fields
  to the same value. The fixed run identifier is
  `rung3-DS-clip0.28-gen192-train32-n8-resp4096-legacy-replay-s100-20261013-080044`; no stopping rule depends
  on validation accuracy or response length. At step 50 the run remained above the collapse threshold (807.1
  mean response tokens; 0.269 Countdown validation accuracy). This interim value is not a terminal result.
- 13 October 2026: the targeted DS replay completed the fixed step-100 horizon successfully and saved
  `global_step_100`. Its mean response length over steps 90--100 was 1,527.4 tokens, classifying it as
  endpoint-healthy. At step 100, Countdown validation accuracy was 0.472, validation reward was 0.477,
  format-valid rate was 0.524, mean response length was 1,618.9 tokens, and truncation fraction was 0.223.
  The compact 101-row trajectory is `replication/results/DS_legacy_replay_training_metrics.csv`. This
  targeted post-screen run remains separate from the original 22-run denominator and does not reopen the
  suspended full continuation queue.
