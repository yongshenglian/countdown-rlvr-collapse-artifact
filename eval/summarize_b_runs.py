"""Compact public summaries for the 30-run study.

  python eval/summarize_b_runs.py

replication/b_run_registry.csv: one row per run, from eval/revision_20261109/training_trajectories.csv.
  window_mean_length is the mean rollout length over steps 90-100; a run is early short when it rounds
  to at most 100 tokens. entropy_explosion is step-300 entropy above 1 nat (post hoc criterion).
eval/results_4090_prompt_ablation/gsm8k/summary.csv: GSM8K think-prompt component controls, one row
  per model and variant, from the scored rows written by handoff/import_full_eval_results.py.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

traj = pd.read_csv(ROOT / "eval/revision_20261109/training_trajectories.csv")
window = traj[traj.step.between(90, 100)].groupby(["method", "seed"]).length.mean()
final = traj[traj.step == 300].set_index(["method", "seed"])
reg = pd.DataFrame({"window_mean_length": window.round(1),
                    "step300_entropy": final.entropy.round(3),
                    "step300_countdown_val_acc": (100 * final.countdown).round(1)}).reset_index()
reg["early_short"] = reg.window_mean_length.round() <= 100
reg["entropy_explosion"] = reg.step300_entropy > 1
reg["checkpoint"] = [f"b_{m}_s{s}_step300" for m, s in zip(reg.method, reg.seed)]
reg.to_csv(ROOT / "replication/b_run_registry.csv", index=False)

rows = []
abl = ROOT / "eval/results_4090_prompt_ablation/gsm8k"
for path in sorted(abl.glob("*__*.jsonl")):
    model, variant = path.stem.split("__")
    if model != "base" and not model.startswith("b_"):
        continue  # older matched-study checkpoints share this folder
    items = [json.loads(line) for line in path.open()]
    assert len(items) == 1319, path
    rows.append(dict(model=model, variant=variant, n=len(items),
                     strict=100 * sum(r["correct_strict"] for r in items) / len(items),
                     flexible=100 * sum(r["correct_flexible"] for r in items) / len(items),
                     length_limit=100 * sum(r["finish"] == "length" for r in items) / len(items)))
pd.DataFrame(rows).round(2).to_csv(abl / "summary.csv", index=False)
print(f"{len(reg)} runs, {len(rows)} prompt-ablation rows")
