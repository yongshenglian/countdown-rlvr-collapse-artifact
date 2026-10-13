# Countdown RLVR Collapse and Transfer Artifact

This repository contains the manuscript source, compact results, training
configurations, evaluation code, and reproducibility records for:

> Yongsheng Lian, *Environment- and Seed-Dependent Short-Response Collapse in
> Countdown RLVR: Format Effects and Limited Transfer*.

The study examines Qwen2.5-1.5B-Instruct fine-tuning on Countdown using a
DAPO-style collapse screen, with selected PPO- and GRPO-style checkpoints as
observational references. It reports training stability, prompt-format effects,
held-out transfer, and pass@k behavior.

## Release status

This public repository is the compact reproducibility artifact. The DOI-bearing
`v1.0.0` release will be created after the targeted dynamic-sampling replay and
manuscript consistency checks are complete. Until then, the default branch is a
transparent pre-release snapshot rather than the archival version cited by the
paper.

## Contents

- `ppo_comparison_revised.tex` and `ppo_comparison_revised.pdf`: manuscript
  source and compiled paper.
- `ablation/`: launch configurations, collapse-screen data, and analysis.
- `eval/`: evaluation/scoring code, compact summaries, manifests, and vendored
  evaluation inputs needed for offline checks.
- `replication/`: frozen protocol, execution record, scripts, and compact
  trajectories.
- `environment/`: package snapshots, source patches, and provenance records.
- `paper_figs/`: publication figures and generation scripts.

Except for four base-model records needed to verify the exact benchmark inputs,
raw generations and model checkpoints are intentionally excluded from normal
Git history because of their size. Their hashes and access locations will be
recorded in the DOI-bearing release metadata where redistribution permits.

## Verification

```bash
python -m pip install -r eval/requirements-scorers.txt
python eval/verify_dataset_manifest.py
python eval/test_scorers.py
(cd replication && python -m unittest test_eval_checkpoint_diagnostics.py)
tectonic ppo_comparison_revised.tex --keep-logs --keep-intermediates
```

Some training and evaluation commands require Verl, vLLM, GPUs, and model
weights. See `environment/MANIFEST.md` and the recorded requirement snapshots.

## Licenses

Original code in this artifact is released under Apache License 2.0. Original
documentation and compact result data are released under CC BY 4.0. Third-party
templates, datasets, model artifacts, and source patches remain governed by
their respective upstream licenses. See `LICENSE.md`.
