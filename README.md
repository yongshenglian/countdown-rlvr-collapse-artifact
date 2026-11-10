# Countdown RLVR Evaluation Artifact

This repository contains the manuscript source, supplementary materials,
compact results, training configurations, evaluation code, and reproducibility
records for:

> Yongsheng Lian, *What Standard Evaluation Misses in Small-Model RLVR:
> Evidence from 30 Countdown Training Runs*.

Qwen2.5-1.5B-Instruct was trained on Countdown in 30 runs: ten seeds (7--16)
each of PPO, GRPO, and DAPO-style training (GRPO with a wider upper clip), with
eight sequential actor updates per step. Every step-300 checkpoint is evaluated
under standard and training-style prompts, with strict and flexible scoring, and
with sampled pass@k.

## Release status

Archived on Zenodo: [10.5281/zenodo.24095520](https://doi.org/10.5281/zenodo.24095520)
(this DOI always resolves to the latest version).

## Where the 30-run results are

| Paper content | Files |
| --- | --- |
| Training protocol and analysis plan | `replication/ACTIVE_CLIP_PROTOCOL.md`, `replication/submit_active_clip_ppo.sh`, `replication/patches/experiment_b_logging.patch` |
| Per-run outcomes (window mean, early short, step-300 entropy and Countdown accuracy) | `replication/b_run_registry.csv` |
| Training trajectories (Figure 1) | `eval/revision_20261109/training_trajectories.csv` |
| Bootstrap intervals and coverage summary | `eval/revision_20261109/countdown_intervals.json`, `coverage_summary.json`, `math_coverage.csv` |
| Greedy benchmarks, standard and think prompts (Tables S1--S3) | `eval/results_4090/<benchmark>/summary.csv` |
| GSM8K think-prompt component controls | `eval/results_4090_prompt_ablation/gsm8k/summary.csv` |
| Countdown under the training wrapper and a plain prompt | `eval/results_countdown_prompt/scored.csv` |
| Sampled pass@k (Table S4) and five base-model sampling runs | `eval/results_passk_b/`, `eval/revision_20261109/base_math_repeats/` |
| GSM8K scorer sensitivity | `eval/revision_20261109/gsm8k_scorer_sensitivity.csv`, `gsm8k_scorer_groups.csv` |
| Compact summaries above | `eval/summarize_b_runs.py` |

The folder name `results_4090` is historical; the generations in it were
produced as described in Supplementary Note S1.

## Contents

- `ppo_comparison_revised.tex`/`.pdf` and `supplementary_materials.tex`/`.pdf`:
  manuscript and supplementary materials.
- `eval/`: evaluation and scoring code, compact summaries, manifests, and
  vendored evaluation inputs needed for offline checks.
- `replication/`: protocols, launch scripts, and compact training records.
- `ablation/`, `replication/results/`, `eval/results/`, `eval/results_8k/`,
  `eval/results_passk/`: records from earlier versions of the study, kept for
  provenance; the current manuscript does not use them.
- `environment/`: package snapshots, source patches, and provenance records.
- `paper_figs/`: publication figures and generation scripts.

Except for four base-model records needed to verify the exact benchmark inputs,
raw generations and model checkpoints are excluded because of their size.

## Verification

```bash
python -m pip install -r eval/requirements-scorers.txt
python eval/verify_dataset_manifest.py
python eval/test_scorers.py
(cd replication && python -m unittest test_eval_checkpoint_diagnostics.py)
tectonic ppo_comparison_revised.tex
tectonic supplementary_materials.tex
```

Some training and evaluation commands require Verl, vLLM, GPUs, and model
weights. See `environment/MANIFEST.md` and the recorded requirement snapshots.

## Licenses

Original code in this artifact is released under Apache License 2.0. Original
documentation and compact result data are released under CC BY 4.0. Third-party
templates, datasets, model artifacts, and source patches remain governed by
their respective upstream licenses. See `LICENSE.md`.
