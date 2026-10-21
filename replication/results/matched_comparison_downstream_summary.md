# Matched Step-300 Downstream Evaluation

This report combines the matched training outcomes with the completed 8192-token
downstream evaluation. All values are strict accuracy. Each downstream model was
evaluated on the complete GSM8K (n=1319), MATH (n=5000), BBH (n=6511), and
MMLU-Pro (n=12032) test sets under both the standard and think prompts.

## Per-seed results

| Recipe | Seed | Early collapse | Countdown step 300 | GSM8K std / think | MATH std / think | BBH std / think | MMLU-Pro std / think | Mean std | Mean think | Mean all 8 |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 7 | No | 55.7 | 59.4 / 71.2 | 55.5 / 44.2 | 34.9 / 40.4 | 33.1 / 27.1 | 45.7 | 45.7 | 45.7 |
| PPO | 8 | No | 53.5 | 51.9 / 72.7 | 55.4 / 47.0 | 34.7 / 42.0 | 33.8 / 28.5 | 43.9 | 47.5 | 45.7 |
| PPO | 9 | Yes | 1.5 | 55.5 / 56.4 | 54.1 / 32.3 | 37.1 / 31.5 | 32.2 / 21.8 | 44.7 | 35.5 | 40.1 |
| GRPO | 7 | No | 52.0 | 56.8 / 55.6 | 55.0 / 38.3 | 34.2 / 37.7 | 33.4 / 18.6 | 44.9 | 37.6 | 41.2 |
| GRPO | 8 | No | 59.1 | 53.8 / 65.5 | 54.9 / 41.7 | 35.7 / 36.7 | 33.5 / 24.5 | 44.5 | 42.1 | 43.3 |
| GRPO | 9 | Yes | 25.9 | 62.5 / 7.1 | 54.6 / 10.9 | 34.0 / 34.5 | 33.7 / 20.1 | 46.2 | 18.2 | 32.2 |
| DAPO-style | 7 | Yes | 25.8 | 56.6 / 12.8 | 53.9 / 10.8 | 33.5 / 33.7 | 33.3 / 15.5 | 44.3 | 18.2 | 31.3 |
| DAPO-style | 8 | Yes | 24.8 | 48.9 / 8.0 | 54.5 / 15.1 | 34.1 / 36.1 | 33.9 / 25.6 | 42.8 | 21.2 | 32.0 |
| DAPO-style | 9 | Yes | 27.1 | 50.1 / 15.1 | 54.6 / 12.5 | 33.9 / 32.2 | 34.0 / 20.3 | 43.1 | 20.0 | 31.6 |

## Recipe-level descriptive summary

The mean is across the three matched seeds; the parenthetical value is the
sample standard deviation across seeds. The four-benchmark means weight each
benchmark equally and are descriptive, not formal inferential estimates.

| Recipe | Early collapses | Standard-prompt mean | Think-prompt mean | All-eight mean |
|---|---:|---:|---:|---:|
| PPO | 1/3 | 44.8 (0.9) | 42.9 (6.5) | 43.9 (3.3) |
| GRPO | 1/3 | 45.2 (0.9) | 32.6 (12.7) | 38.9 (5.9) |
| DAPO-style | 3/3 | 43.4 (0.8) | 19.8 (1.5) | 31.6 (0.4) |

For reference, the base model scores 39.2 on the standard-prompt mean, 39.9 on
the think-prompt mean, and 39.5 across all eight benchmark/prompt cells.

## Interpretation for the manuscript

1. The controlled comparison supports a recipe-dependent collapse result. With
   data, reward, batch size, group size, optimizer settings, update count, and
   300-step budget matched, early collapse occurred in 1/3 PPO seeds, 1/3
   symmetric-clip GRPO seeds, and 3/3 asymmetric-clip DAPO-style seeds.
2. The downstream damage is prompt-conditional. Standard-prompt means differ by
   less than two percentage points between PPO and GRPO and by less than two
   points between GRPO and DAPO-style, even though the collapse rates differ.
   Think-prompt performance exposes the damage: collapsed GRPO seed 9 falls to
   18.2 versus 39.8 for its two healthy seeds, and all DAPO-style seeds average
   19.8.
3. The strongest gains are concentrated on GSM8K, which is closest to the
   arithmetic training domain. MATH, BBH, and MMLU-Pro standard-prompt accuracy
   changes little. The paper should not describe this as broad capability gain.
4. PPO is the strongest matched recipe on the all-eight descriptive mean, but
   three seeds are insufficient for a general algorithm ranking. The defensible
   claim is that asymmetric clipping was less stable in this tested setting and
   that prompt robustness distinguishes apparently healthy standard-prompt
   checkpoints from collapsed ones.
5. Report all seeds individually. A standard-prompt-only average would conceal
   the main failure mode and could be read as selective reporting.

The exact generated responses and scorer outputs are in `eval/results_8k/`; the
source summaries are the four benchmark-specific `summary.csv` files there.
