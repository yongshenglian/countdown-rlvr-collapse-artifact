# Reproducibility Environment Manifest

Captured on 2026-10-12 for the frozen multi-seed replication and paper-analysis artifact.

## Training environment

- Host OS: Ubuntu 24.04.5 LTS; Linux 7.0.0-38-generic x86_64.
- Accelerators: 4 x NVIDIA RTX A6000 (48 GB); driver 580.178.04.
- Python: 3.10.0.
- PyTorch: 2.6.0+cu124.
- vLLM: 0.8.5.post1.
- Transformers: 4.51.1.
- Ray: 2.48.0.
- Datasets: 4.0.0.
- NumPy: 1.26.4.
- Pandas: 2.3.1.
- Full package lock: `environment/requirements-training.txt`.
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`, immutable snapshot
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Verl source base commit:
  `6abe50ab0a84237d9d8a733d365c3b983bcf647d`.
- The Verl worktree used for job submission had tracked modifications. The exact
  seed-3 submission diff is archived as
  `environment/verl-training-working-tree-seed3.patch`. The source used by
  subsequently submitted jobs adds diagnostic-only entropy logging and is archived
  as `environment/verl-training-working-tree-prospective.patch`; the optimization
  coefficient remains zero. Untracked files reported by Git were launch-script
  variants and an `.orig_before_mask` backup; none is imported by the submitted
  entry points.
- The immutable custom reward implementations used by the replication are
  `ablation/reward_sep2025.py` and
  `replication/reward_naive_countdown.py`.
- Exact training and validation parquet files are vendored under
  `replication/data/` and verified by `eval/DATASET_MANIFEST.json`.

The active DAPO continuation is launched from `/home/lian/verl` with
`/home/lian/miniconda3/envs/verl-fork/bin/python`. Although `pip freeze`
records an editable/local Verl installation from another checkout, Python module
resolution for each Ray job begins with the submitted `/home/lian/verl`
working directory. The base commit plus archived tracked patch therefore records
the code actually packaged for those jobs.

## Evaluation and analysis environment

- Full package lock: `environment/requirements-analysis.txt`.
- Dataset revisions, exact file hashes, canonical input digests, and an offline
  verification command are in `eval/DATASET_MANIFEST.json` and
  `eval/verify_dataset_manifest.py`.
- Paper compiler: Tectonic 0.17.0.
- Artifact source is the Git commit containing this manifest; the earlier
  pre-revision baseline was
  `380a25ff53afd660fb45d13daa860b5b27379150`.

## Integrity hashes

A consolidated machine-readable list is stored in
`environment/SHA256SUMS`. Headline checkpoint file and aggregate hashes are
stored separately in `eval/CHECKPOINT_MANIFEST.json`.

- `requirements-training.txt`:
  `72c44162548a804e4c752f78f625448a06b506fd4801a0e1df55807b4cc08817`
- `requirements-analysis.txt`:
  `ace77b0b87e87dc7375544fb1bfe46f23bb0690b68c1b95eedb8cdeb4e69f49f`
- `verl-training-working-tree-seed3.patch`:
  `85586f66c4becc0d27792837ac978a23772cd4c72e7eb979acd08bda24cfc609`
- `verl-training-working-tree-prospective.patch`:
  `8254f03bdffed6b7d24a4e1150548849571c70270a1f3f676c21057a65a01301`

Run the offline data verification with:

```bash
/home/lian/miniconda3/envs/verl-fork/bin/python eval/verify_dataset_manifest.py
```

Add `--upstream` to reconstruct MATH, BBH, and MMLU-Pro from their immutable
Hugging Face revisions. That optional check may require network access.
