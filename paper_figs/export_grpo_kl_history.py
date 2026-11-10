"""Export the GRPO G=2 KL-sweep training histories from W&B to a CSV.

Run once with W&B access; make_grpo_kl_fig.py then plots from the CSV alone.
"""
from pathlib import Path

import pandas as pd
import wandb

PROJECT = "yongshenglian-university-of-louisville/GRPO_Countdown_R1"
RUNS = {  # KL-loss coefficient label -> W&B run id (all G=2, lr 1e-6, mini-batch 64)
    "off": "qde0tjh9",
    "0.001": "djmdy7s6",
    "0.005": "02o61vat",
    "0.0075": "1hrg4q75",
    "0.01": "q0b8qi84",
    "0.04": "rq4tjmwu",
}
KEYS = ["critic/score/mean", "actor/kl_loss", "response_length/mean", "actor/grad_norm"]
OUT = Path(__file__).resolve().parent / "grpo_kl_history.csv"

api = wandb.Api(timeout=120)
frames = []
for label, run_id in RUNS.items():
    run = api.run(f"{PROJECT}/{run_id}")
    h = run.history(samples=5000, pandas=True)
    h = h[["_step"] + [k for k in KEYS if k in h.columns]].dropna(subset=["response_length/mean"])
    h.insert(0, "run_id", run_id)
    h.insert(0, "kl_coef", label)
    frames.append(h)
pd.concat(frames).rename(columns={"_step": "step"}).to_csv(OUT, index=False)
print(OUT)
