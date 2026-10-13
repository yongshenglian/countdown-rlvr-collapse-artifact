# GSM8K Format-Only Control

The base model was evaluated on all 1,319 GSM8K test items with greedy 
decoding and the same 8,192-token limit, chat template, dataset, and 
scorer as the headline evaluation. The control retains the original 
zero-shot chain-of-thought prompt and appends a mandatory final-line 
contract: `#### N`, with `N` replaced by only the numeric answer.

## Accuracy and compliance

| Condition | n | Strict (%) | Tolerant (%) | Explicit marker (%) | Truncated (%) |
|---|---:|---:|---:|---:|---:|
| Base, standard prompt | 1319 | 33.36 | 72.25 | 45.56 | 0.15 |
| Base, strong-format prompt | 1319 | 58.53 | 67.10 | 88.93 | 0.45 |
| DS, step 300, standard prompt | 1319 | 66.11 | 73.01 | 89.01 | 0.45 |
| No-DS, step 300, standard prompt | 1319 | 69.90 | 75.59 | 91.28 | 0.83 |
| GRPO KL 0.04, step 250, standard prompt | 1319 | 69.52 | 75.36 | 92.19 | 0.61 |

## Prespecified paired base-model comparison

| Metric | Standard only | Strong-format only | Difference (points) | Paired 95% CI | Raw exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Strict | 124 | 456 | +25.17 | [+21.86, +28.48] | 1.454e-45 | 2.908e-45 |
| Flexible | 182 | 114 | -5.16 | [-7.70, -2.61] | 9.225e-05 | 9.225e-05 |

## Interpretation

The format-only instruction changed strict accuracy by +25.17 points and tolerant accuracy by -5.16 points. Strict-score movement without a corresponding tolerant-score movement is attributable to answer presentation rather than newly correct solutions. Any tolerant difference is a prompt effect and must not be described as a training gain.

This is a prompt-only control, not a format-only supervised-tuning control. It establishes what can be obtained from a stronger inference instruction without changing model weights.

## Reproduction metadata

- Model snapshot: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Dataset: `eval/data/gsm8k_test.parquet` (1,319 records)
- Decoding: greedy (`temperature=0`), maximum 8,192 generated tokens
- Output archive: `eval/results_8k/format_control/base__format_strong.jsonl`
- Analysis script: `eval/eval_format_control.py`
