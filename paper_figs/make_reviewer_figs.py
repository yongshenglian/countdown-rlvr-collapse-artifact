"""Generate publication seed trajectories and pass@k figures.

The script uses only archived CSV outputs and does not query external services.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

INK = "#242424"
GRID = "#dddddd"
COLORS = {
    "no DS (rung 2)": "#0072B2",
    "DS": "#D55E00",
    "A1": "#009E73",
    "A1 + KL 0.01": "#CC79A7",
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def collapse_figure():
    df = pd.read_csv(ROOT / "ablation" / "collapse_screen.csv")
    df = df[df["status"] != "not run"].copy()
    windows = [25, 50, 95]
    columns = ["len_step25", "len_step50", "len_step90_100"]
    recipes = ["no DS (rung 2)", "DS", "A1", "A1 + KL 0.01"]
    titles = ["Matched no-DS", "Dynamic sampling", "A1", "A1 + KL 0.01"]

    seed_markers = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.55), sharex=True, sharey=True)
    for ax, recipe, title in zip(axes, recipes, titles):
        group = df[df["recipe"] == recipe].sort_values("seed")
        for _, row in group.iterrows():
            values = row[columns].astype(float).to_numpy()
            seed = int(row["seed"])
            jittered_windows = [x + (seed - 3.5) * 0.45 for x in windows]
            legacy = recipe == "DS" and seed == 1
            ax.plot(
                jittered_windows,
                values,
                color=COLORS[recipe],
                alpha=1.0 if legacy else 0.72,
                lw=2.1 if legacy else 1.05,
                ls="--" if legacy else "-",
                marker=seed_markers[seed - 1],
                ms=3.2,
            )
        ax.axhline(100, color=INK, lw=0.9, ls=":")
        ax.set_title(title)
        ax.set_xticks(windows, ["25", "50", "90--100"])
        ax.set_yscale("log")
        ax.set_ylim(15, 3200)
        ax.set_xlim(20, 105)
        ax.set_xlabel("Training step/window")
    axes[0].set_ylabel("Mean response length (tokens, log scale)")
    axes[1].text(23, 108, "100-token screen", fontsize=6.5, va="bottom")
    axes[1].text(22, 1750, "dashed = legacy environment", fontsize=6.3, color=COLORS["DS"])
    handles = [
        Line2D([], [], color="#666666", marker=marker, lw=1, ms=3.5, label=f"Seed {seed}")
        for seed, marker in enumerate(seed_markers, start=1)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.7)
    save(fig, "fig3_collapse_seeds")


def passk_figure():
    """Flexible pass@k of the 30 experiment-B checkpoints minus the same-machine base model."""
    df = pd.read_csv(ROOT / "eval" / "results_passk_b" / "summary.csv")
    df = df[df["metric"] == "flexible"].copy()
    # Early-short runs: rounded mean response length over steps 90-100 <= 100 tokens.
    short = {f"ppo_s{s}" for s in [8, 10, 16]} | {f"grpo_s{s}" for s in [8, 9, 10, 12, 14, 15, 16]} \
        | {f"dapo_s{s}" for s in [8, 9, 10, 12, 13, 14, 15, 16]}
    methods = {"ppo": ("PPO", "o"), "grpo": ("GRPO", "s"), "dapo": ("DAPO-style", "^")}
    group_colors = {True: "#D55E00", False: "#0072B2"}
    ylim = (-22, 4)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), sharey=True)
    for ax, bench, title, xmax in zip(
        axes,
        ["math500", "gsm8k"],
        ["(a) MATH-500", "(b) GSM8K"],
        [64, 32],
    ):
        part = df[df["bench"] == bench]
        base = part[part["model"] == "base"].set_index("k")["pass_at_k"]
        for method, (_, marker) in methods.items():
            for seed in range(7, 17):
                run = f"{method}_s{seed}"
                curve = part[part["model"] == f"b_{run}_step300"].set_index("k")["pass_at_k"]
                delta = (curve - base).sort_index()
                if delta.min() < ylim[0]:  # off-scale run: give its range instead of compressing the axis
                    ax.annotate(f"Not drawn: {methods[method][0]} s{seed},\n{delta.min():.1f} to {delta.max():.1f}",
                                xy=(0.97, 0.04), xycoords="axes fraction", ha="right", fontsize=6.5,
                                color=group_colors[run in short])
                    continue
                ax.plot(
                    delta.index,
                    delta.values,
                    color=group_colors[run in short],
                    marker=marker,
                    ms=2.4,
                    lw=0.8,
                    alpha=0.6,
                )
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_xscale("log", base=2)
        ticks = [1, 2, 4, 8, 16, 32] + ([64] if xmax == 64 else [])
        ax.set_xticks(ticks, [str(k) for k in ticks])
        ax.set_xlim(0.85, xmax * 1.12)
        ax.set_xlabel("Number of samples, k")
        ax.set_title(f"{title}: base pass@1 {base[1]:.1f}%, pass@{xmax} {base[xmax]:.1f}%", loc="left")
    axes[0].set_ylabel("Flexible pass@k minus base\n(percentage points)")
    axes[0].set_ylim(*ylim)
    handles = [
        Line2D([], [], color=group_colors[True], lw=1.4, label="Early-short-response run"),
        Line2D([], [], color=group_colors[False], lw=1.4, label="Other run"),
    ] + [Line2D([], [], color="#666666", marker=m, lw=0, ms=3.5, label=name) for name, m in methods.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.6)
    save(fig, "fig4_passk")


if __name__ == "__main__":
    collapse_figure()
    passk_figure()
