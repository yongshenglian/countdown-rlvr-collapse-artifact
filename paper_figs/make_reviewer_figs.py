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
    df = pd.read_csv(ROOT / "eval" / "results_passk" / "summary.csv")
    df = df[df["metric"] == "flexible"].copy()
    order = [
        "base",
        "DS_step300",
        "noDS_step300",
        "GRPO_n2_kl4e-2_step250",
        "PPO_klF_ent0_lr1e-6_step520",
    ]
    labels = {
        "base": "Base",
        "DS_step300": "DS, step 300",
        "noDS_step300": "Original no-DS, step 300",
        "GRPO_n2_kl4e-2_step250": "GRPO KL 0.04, step 250",
        "PPO_klF_ent0_lr1e-6_step520": r"PPO lr $10^{-6}$, step 520",
    }
    colors = ["#000000", "#D55E00", "#0072B2", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D", "v"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), sharey=False)
    for ax, bench, title, xmax in zip(
        axes,
        ["math500", "gsm8k"],
        ["(a) MATH-500", "(b) GSM8K"],
        [64, 32],
    ):
        for model, color, marker in zip(order, colors, markers):
            part = df[(df["bench"] == bench) & (df["model"] == model)].sort_values("k")
            ax.plot(
                part["k"],
                part["pass_at_k"],
                label=labels[model],
                color=color,
                marker=marker,
                ms=3.5,
                lw=1.4,
            )
        ax.set_xscale("log", base=2)
        ticks = [1, 2, 4, 8, 16, 32] + ([64] if xmax == 64 else [])
        ax.set_xticks(ticks, [str(k) for k in ticks])
        ax.set_xlim(0.85, xmax * 1.12)
        ax.set_xlabel("Number of samples, k")
        ax.set_ylabel("Format-tolerant pass@k (%)")
        ax.set_title(title, loc="left")
    axes[0].set_ylim(42, 96)
    axes[1].set_ylim(55, 100)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.6)
    save(fig, "fig4_passk")


if __name__ == "__main__":
    collapse_figure()
    passk_figure()
