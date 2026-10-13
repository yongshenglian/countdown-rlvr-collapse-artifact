"""Collapse screen summary: for each recipe and training seed, did the run collapse into short answers?

Collapsed = mean response length over steps 90-100 below 100 tokens (healthy runs are at 250-1,700 tokens by
then; collapsed runs at about 20-70). Seeds 1-2 are the first 100 steps of the existing full-length runs; seeds 3-6
are the 100-step screen runs (name suffix -s100). Writes collapse_screen.csv and collapse_screen.md.

  /home/lian/miniconda3/bin/python analysis_collapse_screen.py
"""
import os

import pandas as pd
import wandb

HERE = os.path.dirname(os.path.abspath(__file__))
ENT = "yongshenglian-university-of-louisville"
THRESH = 100
BASE = {"no DS (rung 2)": "rung2-clip0.28-noDS-gen32-train32-n8-resp4096",
        "DS": "rung3-DS-clip0.28-gen192-train32-n8-resp4096",
        "A1": "rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096",
        "A1 + KL 0.01": "rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096"}
# seed 1 / seed 2 of the full-length runs (project, run name); DS seed 1 is the original DAPO_new run
FULL = {("no DS (rung 2)", 1): ("DAPO_ablation", BASE["no DS (rung 2)"]),
        ("no DS (rung 2)", 2): ("DAPO_ablation", BASE["no DS (rung 2)"] + "-seed2"),
        ("DS", 1): ("DAPO_new", "Qwen2.5-1.5B-G8-acc-filtering-tokenlevel-bsz-32-0910"),
        ("DS", 2): ("DAPO_ablation", BASE["DS"] + "-seed2"),
        ("A1", 1): ("DAPO_ablation", BASE["A1"]),
        ("A1", 2): ("DAPO_ablation", BASE["A1"] + "-seed2")}


def main():
    api = wandb.Api()
    runs = {}
    for proj in ["DAPO_ablation", "DAPO_new"]:
        for r in api.runs(f"{ENT}/{proj}"):
            runs.setdefault((proj, r.name), []).append(r)
    rows = []
    for recipe, base in BASE.items():
        for seed in range(1, 7):
            key = FULL.get((recipe, seed), ("DAPO_ablation", f"{base}-seed{seed}-s100"))
            cands = runs.get(key, [])
            if not cands:
                rows.append(dict(recipe=recipe, seed=seed, run=key[1], status="not run")); continue
            r = sorted(cands, key=lambda x: x.created_at)[-1]
            h = pd.DataFrame(list(r.scan_history(keys=["_step", "response_length/mean"]))).set_index("_step").sort_index()
            v = pd.DataFrame(list(r.scan_history(keys=["_step", "val-core/countdown/acc/mean@1"])))
            v = v.set_index("_step").sort_index()["val-core/countdown/acc/mean@1"] if len(v) else pd.Series(dtype=float)
            L = h["response_length/mean"]
            if L.index.max() < 95:
                rows.append(dict(recipe=recipe, seed=seed, run=r.name, wandb=r.id, status=f"incomplete (step {L.index.max()})")); continue
            len100 = L.loc[90:100].mean()
            rows.append(dict(recipe=recipe, seed=seed, run=r.name, wandb=r.id,
                             len_step25=round(L.loc[23:27].mean()), len_step50=round(L.loc[48:52].mean()),
                             len_step90_100=round(len100), val_acc_step100=round(100 * v.loc[90:100].mean(), 1) if len(v) else None,
                             status="collapsed" if len100 < THRESH else "healthy"))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "collapse_screen.csv"), index=False)
    done = d[d.status.isin(["collapsed", "healthy"])]
    rate = done.groupby("recipe").status.agg(lambda s: f"{(s == 'collapsed').sum()}/{len(s)} collapsed").rename("collapse rate")
    lines = ["# Collapse screen (first 100 steps)\n",
             f"Collapsed = mean response length over steps 90-100 < {THRESH} tokens. Seeds 1-2: existing full-length runs; "
             "seeds 3-6: 100-step screen runs (`chain_collapse_screen.sh`).\n",
             "| recipe | collapse rate |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rate.items()]
    lines += ["", "| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join("" if pd.isna(x) else str(x) for x in row) + " |" for row in d.itertuples(index=False)]
    open(os.path.join(HERE, "collapse_screen.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
