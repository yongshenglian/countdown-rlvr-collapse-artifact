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

## Reviewer-requested DS configuration replay (in progress)

One targeted run replays the legacy DS configuration in the current frozen environment with its original
default seed pair (`data.seed=1`, rollout seed 0). This is not part of the stopped full continuation queue and
has a fixed 100-step endpoint. Run identifier:
`rung3-DS-clip0.28-gen192-train32-n8-resp4096-legacy-replay-s100-20261013-080044`.

| Step | Status | Countdown validation accuracy | Mean response length |
|---:|---|---:|---:|
| 50 | Interim; run active | 0.269 | 807.1 tokens |

The manuscript will not use this run in endpoint counts until step 100 is complete.
