"""Reconstruct time-to-collapse and joint diagnostics for the 22-run screen.

The binary screen was defined earlier. This descriptive post-hoc analysis uses
non-overlapping five-step endpoints (10, 15, ..., 100). An event occurs at the
first endpoint whose trailing five-step mean response length is below the
threshold and whose next two windows are also below it. Runs without an event
are right-censored at step 100.

Fetch/cache (environment with wandb):
  python analysis_collapse_survival.py --refresh --no-plot
Plot (environment with matplotlib):
  python analysis_collapse_survival.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGISTRY = HERE / "collapse_screen.csv"
TRAJECTORIES = HERE / "collapse_trajectories.csv"
TIMES = HERE / "collapse_times.csv"
SURVIVAL = HERE / "collapse_survival.csv"
FIG_PDF = ROOT / "paper_figs" / "fig5_collapse_survival_diagnostics.pdf"
FIG_PNG = ROOT / "paper_figs" / "fig5_collapse_survival_diagnostics.png"

ENTITY = "yongshenglian-university-of-louisville"
THRESHOLD = 100.0
WINDOW = 5
ENDPOINTS = list(range(10, 101, WINDOW))
RECIPES = ["no DS (rung 2)", "DS", "A1", "A1 + KL 0.01"]
COLORS = {
    "no DS (rung 2)": "#0072B2",
    "DS": "#D55E00",
    "A1": "#009E73",
    "A1 + KL 0.01": "#CC79A7",
}

METRICS = {
    "response_length": "response_length/mean",
    "train_score": "critic/score/mean",
    "train_reward": "critic/rewards/mean",
    "val_accuracy": "val-core/countdown/acc/mean@1",
    "ppo_kl": "actor/ppo_kl",
    "kl_loss": "actor/kl_loss",
    "truncation_rate": "response_length/clip_ratio",
    "entropy": "actor/entropy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refetch all W&B histories")
    parser.add_argument("--no-plot", action="store_true", help="write CSVs only")
    return parser.parse_args()


def project_for(row: pd.Series) -> str:
    if row["recipe"] == "DS" and int(row["seed"]) == 1:
        return "DAPO_new"
    return "DAPO_ablation"


def fetch_histories() -> pd.DataFrame:
    import wandb

    registry = pd.read_csv(REGISTRY)
    registry = registry[registry["wandb"].notna()].copy()
    api = wandb.Api()
    rows: list[dict] = []
    for item in registry.itertuples(index=False):
        project = "DAPO_new" if item.recipe == "DS" and int(item.seed) == 1 else "DAPO_ablation"
        print(f"fetch {item.recipe} seed {item.seed}: {project}/{item.wandb}", flush=True)
        run = api.run(f"{ENTITY}/{project}/{item.wandb}")
        history = list(run.scan_history(page_size=1000))
        for record in history:
            step = record.get("_step")
            if step is None or step > 100:
                continue
            out = {
                "recipe": item.recipe,
                "seed": int(item.seed),
                "run": item.run,
                "wandb": item.wandb,
                "project": project,
                "environment": "legacy" if project == "DAPO_new" else "current",
                "step": int(step),
            }
            for short, key in METRICS.items():
                out[short] = record.get(key)
            rows.append(out)
    data = pd.DataFrame(rows).sort_values(["recipe", "seed", "step"])
    data.to_csv(TRAJECTORIES, index=False)
    return data


def trailing_window(lengths: pd.Series, endpoint: int) -> float:
    values = lengths.loc[endpoint - WINDOW + 1 : endpoint].dropna()
    return float(values.mean()) if len(values) >= 3 else math.nan


def first_persistent_event(lengths: pd.Series, threshold: float = THRESHOLD) -> float:
    means = {endpoint: trailing_window(lengths, endpoint) for endpoint in ENDPOINTS}
    for endpoint in ENDPOINTS[:-2]:
        triple = [means[endpoint + offset] for offset in (0, WINDOW, 2 * WINDOW)]
        if all(np.isfinite(value) and value < threshold for value in triple):
            return float(endpoint)
    return math.nan


def nearest_value(group: pd.DataFrame, column: str, start: int, end: int) -> float:
    values = group.loc[group["step"].between(start, end), column].dropna()
    return float(values.mean()) if len(values) else math.nan


def make_times(data: pd.DataFrame) -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY)
    registry = registry[registry["wandb"].notna()].copy()
    rows = []
    for item in registry.itertuples(index=False):
        group = data[(data["wandb"] == item.wandb)].sort_values("step")
        lengths = group.dropna(subset=["response_length"]).set_index("step")["response_length"]
        event_time = first_persistent_event(lengths)
        event = np.isfinite(event_time)
        rows.append(
            {
                "recipe": item.recipe,
                "seed": int(item.seed),
                "run": item.run,
                "wandb": item.wandb,
                "environment": "legacy" if item.recipe == "DS" and int(item.seed) == 1 else "current",
                "max_logged_step": int(group["step"].max()),
                "event": bool(event),
                "collapse_time": event_time,
                "analysis_time": event_time if event else 100.0,
                "binary_step100_status": item.status,
                "length_steps90_100": nearest_value(group, "response_length", 90, 100),
                "val_accuracy_steps90_100": nearest_value(group, "val_accuracy", 90, 100),
                "train_score_steps90_100": nearest_value(group, "train_score", 90, 100),
                "ppo_kl_steps90_100": nearest_value(group, "ppo_kl", 90, 100),
                "kl_loss_steps90_100": nearest_value(group, "kl_loss", 90, 100),
                "truncation_steps90_100": nearest_value(group, "truncation_rate", 90, 100),
                "entropy_logged": bool(group["entropy"].notna().any()),
                "format_validity_logged": False,
            }
        )
    times = pd.DataFrame(rows).sort_values(["recipe", "seed"])
    times.to_csv(TIMES, index=False)
    return times


def km_table(times: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for recipe in RECIPES:
        group = times[times["recipe"] == recipe]
        survival = 1.0
        greenwood = 0.0
        for time in sorted(set(group["analysis_time"])):
            at_risk = int((group["analysis_time"] >= time).sum())
            events = int(((group["analysis_time"] == time) & group["event"]).sum())
            censored = int(((group["analysis_time"] == time) & ~group["event"]).sum())
            if events:
                survival *= 1.0 - events / at_risk
                if at_risk > events:
                    greenwood += events / (at_risk * (at_risk - events))
            lower, upper = survival, survival
            if 0.0 < survival < 1.0 and greenwood > 0:
                loglog = math.log(-math.log(survival))
                se = math.sqrt(greenwood) / abs(math.log(survival))
                lower = math.exp(-math.exp(loglog + 1.96 * se))
                upper = math.exp(-math.exp(loglog - 1.96 * se))
            elif survival == 0.0:
                lower = upper = 0.0
            rows.append(
                dict(recipe=recipe, time=time, n_at_risk=at_risk, events=events,
                     censored=censored, survival=survival, lower95=lower, upper95=upper)
            )
    result = pd.DataFrame(rows)
    result.to_csv(SURVIVAL, index=False)
    return result


def plot(data: pd.DataFrame, times: pd.DataFrame, survival: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
    })
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.2), constrained_layout=False)
    panels = [
        ("response_length", "(a) Response length", "Mean tokens", True, 1.0),
        ("val_accuracy", "(b) Countdown validation", "Accuracy (%)", False, 100.0),
        ("train_score", "(c) Retained-group training score", "Score", False, 1.0),
        ("kl_loss", "(d) KL-loss estimate", "KL recipe only", False, 1.0),
        ("truncation_rate", "(e) Length truncation", "Rate (%)", False, 100.0),
    ]
    status = times.set_index("wandb")["event"].to_dict()
    for ax, (column, title, ylabel, log_y, scale) in zip(axes.flat[:5], panels):
        for (recipe, seed, wandb_id), group in data.groupby(["recipe", "seed", "wandb"]):
            visible = group.dropna(subset=[column])
            if visible.empty:
                continue
            ax.plot(visible["step"], scale * visible[column], color=COLORS[recipe],
                    lw=0.9, alpha=0.66, ls="--" if status[wandb_id] else "-")
        if log_y:
            ax.set_yscale("log")
            ax.axhline(THRESHOLD, color="black", lw=0.8, ls=":")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.18)

    ax = axes.flat[5]
    for recipe in RECIPES:
        table = survival[survival["recipe"] == recipe]
        x = np.r_[0.0, table["time"].to_numpy()]
        y = np.r_[1.0, table["survival"].to_numpy()]
        ax.step(x, y, where="post", color=COLORS[recipe], lw=1.8)
    ax.set_title("(f) Time to sustained short response", loc="left", fontweight="bold")
    ax.set_ylabel("Fraction not persistently short")
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.18)
    ax.text(0.02, 0.03, "DS run 1: legacy; all others: current",
            transform=ax.transAxes, fontsize=7.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Optimizer step")
    handles = [Line2D([0], [0], color=COLORS[name], lw=2, label=name) for name in RECIPES]
    handles += [Line2D([0], [0], color="0.35", lw=1, ls="-", label="right-censored"),
                Line2D([0], [0], color="0.35", lw=1, ls="--", label="collapse event")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), h_pad=1.1, w_pad=1.0)
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.refresh or not TRAJECTORIES.exists():
        data = fetch_histories()
    else:
        data = pd.read_csv(TRAJECTORIES)
    times = make_times(data)
    survival = km_table(times)
    print(times[["recipe", "seed", "event", "collapse_time", "binary_step100_status"]].to_string(index=False))
    print(f"entropy logged for {int(times['entropy_logged'].sum())}/{len(times)} runs; format validity logged for 0/{len(times)}")
    if not args.no_plot:
        plot(data, times, survival)
        print(f"wrote {FIG_PDF} and {FIG_PNG}")


if __name__ == "__main__":
    main()
