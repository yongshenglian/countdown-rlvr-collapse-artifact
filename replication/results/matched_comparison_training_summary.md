# Step-300 Matched PPO/GRPO/DAPO-Style Training Results

All nine runs use the matched protocol in
`replication/MATCHED_COMPARISON_PROTOCOL.md`. The early-collapse label uses the
prespecified mean response length below 100 tokens over steps 90--100 rule.

| Recipe | Seed | Step-300 Countdown accuracy | Step-300 reward | Step-300 format valid | Step-300 response tokens | Mean tokens, steps 90--100 | Early collapse |
|---|---:|---:|---:|---:|---:|---:|:---:|
| PPO | 7 | 55.7% | 0.560 | 58.9% | 1455.2 | 1374.1 | No |
| PPO | 8 | 53.5% | 0.545 | 63.6% | 1371.9 | 447.0 | No |
| PPO | 9 | 1.5% | 0.104 | 90.4% | 468.8 | 16.3 | Yes |
| GRPO | 7 | 52.0% | 0.568 | 100.0% | 1720.5 | 267.8 | No |
| GRPO | 8 | 59.1% | 0.628 | 96.1% | 1239.3 | 725.0 | No |
| GRPO | 9 | 25.9% | 0.333 | 100.0% | 22.3 | 22.4 | Yes |
| DAPO-style | 7 | 25.8% | 0.332 | 100.0% | 54.2 | 55.6 | Yes |
| DAPO-style | 8 | 24.8% | 0.323 | 100.0% | 60.0 | 54.7 | Yes |
| DAPO-style | 9 | 27.1% | 0.344 | 100.0% | 46.9 | 51.0 | Yes |

## Recipe-level descriptive summary

| Recipe | Step-300 accuracy, mean (sample SD) | Mean response tokens at step 300 | Early collapses |
|---|---:|---:|---:|
| PPO | 36.9% (30.7 pp) | 1098.6 | 1/3 |
| GRPO | 45.7% (17.5 pp) | 994.0 | 1/3 |
| DAPO-style | 25.9% (1.2 pp) | 53.7 | 3/3 |

These three-seed summaries are descriptive. The large within-recipe dispersion
and small sample size do not support precise ranking claims. The matched screen
does show that asymmetric clipping (0.20 lower, 0.28 upper) was associated with
short-response collapse in all three tested seeds, whereas symmetric clipping
and PPO each collapsed in one of three seeds under the same data, reward,
batching, update count, learning rate, and step budget.
