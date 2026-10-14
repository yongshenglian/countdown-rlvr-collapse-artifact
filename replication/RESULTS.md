# Frozen-Environment Replication Results

This file records runs from the protocol in [PROTOCOL.md](PROTOCOL.md). The queue was stopped by author decision on 13 October 2026 before completion. These partial results must not be used to rank recipes or update manuscript-level conclusions.

## Completed training runs

| Recipe | Seed | Terminal status | Early-collapse endpoint | Step 520 validation accuracy | Step 520 validation reward | Step 520 response length | Step 520 truncation |
|---|---:|---|---|---:|---:|---:|---:|
| Matched no-DS DAPO style | 3 | Reached step 520 | Healthy (steps 90--100 mean: 1007.1 tokens) | 0.641 | 0.676 | 1413.1 tokens | 0.004 |

The seed-3 continuation ran from its frozen step-100 prefix to step 520 without a validation-based stopping rule. The exact Ray submission ID was 'rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed3-s100-20261012-062302'. The terminal checkpoint is under the corresponding training directory and the compact 421-row trajectory is in 'results/noDS_seed3_training_metrics.csv'.

Online token entropy was not instrumented for this first continuation, as disclosed prospectively. The common post-hoc checkpoint diagnostic is implemented and tested but was not executed before the queue was stopped.

The remaining queue and post-hoc evaluations were suspended after the completed seed-3 continuation. They can
be restarted from the original step-100 checkpoints if additional replication is requested during peer review.

## Targeted post-screen DS configuration replay

One targeted run replays the legacy DS configuration in the current frozen environment with its original
default seed pair (`data.seed=1`, rollout seed 0). This is not part of the stopped full continuation queue and
has a fixed 100-step endpoint. Run identifier:
`rung3-DS-clip0.28-gen192-train32-n8-resp4096-legacy-replay-s100-20261013-080044`.

| Step | Countdown validation accuracy | Validation reward | Format-valid rate | Mean response length | Truncation |
|---:|---:|---:|---:|---:|---:|
| 90 | 0.437 | 0.443 | 0.506 | 1445.2 | 0.148 |
| 95 | 0.453 | 0.459 | 0.515 | 1616.9 | 0.211 |
| 100 | 0.472 | 0.477 | 0.524 | 1618.9 | 0.223 |

The run completed successfully and saved `global_step_100`. Its mean response length over the predefined
steps 90--100 endpoint window was **1527.4 tokens**, so it was healthy under the manuscript's primary
100-token collapse rule. The compact 101-row trajectory is
`results/DS_legacy_replay_training_metrics.csv`.

This post-screen replay is reported separately and is not added to the original 22-run screen denominator.
Together with the five current-environment equal-seed-pair DS failures, it rules out deterministic failure
caused solely by the current environment. A single replay cannot identify whether the contrast arises from
the data seed, rollout seed, their interaction, or a residual difference in the reconstructed legacy setup.
