# Dynamic-Sampling Run: Legacy versus Current Environment

This record separates settings recovered from the archived healthy DS run (`DAPO_new/f8h77wol`) from settings fixed for the targeted post-screen replay (`DAPO_ablation/rung3-DS-clip0.28-gen192-train32-n8-resp4096-legacy-replay-s100`). “Unknown” means the legacy artifact did not preserve enough information; no value is inferred retrospectively.

| Factor | Legacy healthy DS run | Current controlled replay | Assessment |
|---|---|---|---|
| Actor/model family | Qwen2.5-1.5B-Instruct local copy | Qwen2.5-1.5B-Instruct HF snapshot `989aa798...` | Same named release; legacy content hash unknown |
| Verl code location | `/home/lian/verl` | `/home/lian/verl`, commit `6abe50a...` | Legacy exact commit unknown |
| PyTorch | 2.6.0 recorded | 2.6.0+cu124 | Nominally matched |
| vLLM | 0.8.5.post1 recorded | 0.8.5.post1 | Matched |
| Training data | Countdown train parquet | Vendored Countdown train parquet, SHA-256 `a8508c15...` | Legacy file hash unknown |
| Validation data | Countdown test parquet | Vendored Countdown test parquet, SHA-256 `63b631a3...` | Legacy file hash unknown |
| Reward | September-2025 format (0.1) + correctness (0.9) reward | Reconstructed reward that reproduces the archived logged score | Logic matched; legacy source file itself unavailable |
| Data seed | Framework default (1) | Framework default (1) | Matched |
| Rollout seed | Framework default (0) | Framework default (0) | Matched |
| Group size | 8 | 8 | Matched |
| Generated prompts/round | 192 | 192 | Matched |
| Retained prompts/update | 32 | 32 | Matched |
| Dynamic sampling | Accuracy-variation filter on | Accuracy-variation filter on | Matched |
| Retry cap | Non-positive/no fixed cap | Non-positive/no fixed cap | Matched |
| Response cap | 4096 tokens | 4096 tokens | Matched |
| Clip interval | 0.20/0.28 | 0.20/0.28 | Matched |
| Loss aggregation | Global token mean | Global token mean | Matched |
| Actor learning rate | `1e-6` | `1e-6` | Matched |
| Warmup/schedule | 10-step warmup, then constant (recorded) | 10-step warmup, then constant | Matched |
| Weight decay | 0.1 (recorded) | 0.1 | Matched |
| KL loss / entropy bonus | Off / 0 | Off / 0 | Matched |
| Hardware | Four RTX A6000 GPUs | Four RTX A6000 GPUs | Matched at model level |

The replay is therefore a close reconstruction, not a byte-identical reproduction. The unverified legacy model hash, data hashes, reward source, code commit, driver/runtime details, and nondeterministic GPU kernels remain possible differences. Its fixed 100-step outcome should be interpreted together with these residual uncertainties.

## Step-100 outcome

| Run group | Environment | Data seed | Rollout seed | Steps 90--100 mean length | Step-100 validation accuracy | Endpoint |
|---|---|---:|---:|---:|---:|---|
| Archived healthy DS | Legacy | Default 1 | Default 0 | 1569.4 | 0.508 | Healthy |
| Original DS screen, five runs | Current | 2--6 | 2--6 | 19.5--20.6 | 0.262--0.288 | Collapsed |
| Targeted legacy-config replay | Current | 1 | 0 | 1527.4 | 0.472 | Healthy |

The targeted replay completed successfully at step 100. It demonstrates that the current frozen environment
does not deterministically produce DS collapse under the reconstructed legacy recipe. Because this is one
post-screen replay and both the data and rollout seeds differ from the five failed pairs, it does not identify
which random stream or interaction caused the contrast. Nor can it eliminate every residual difference listed
above. The original 22-run screen is therefore left unchanged; the targeted replay is reported separately.

Compact trajectory: `replication/results/DS_legacy_replay_training_metrics.csv`.
Ray job identifier:
`rung3-DS-clip0.28-gen192-train32-n8-resp4096-legacy-replay-s100-20261013-080044`.
