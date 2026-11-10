"""Plot GRPO KL-sweep training diagnostics (G=2) from grpo_kl_history.csv."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ["off", "0.001", "0.005", "0.0075", "0.01", "0.04"]
COLORS = {"off": "#555555", "0.001": "#E69F00", "0.005": "#D55E00",
          "0.0075": "#009E73", "0.01": "#0072B2", "0.04": "#CC79A7"}
STYLES = {"off": ":", "0.001": "--", "0.005": "-.", "0.0075": "-", "0.01": "-", "0.04": "-"}
PANELS = [
    ("critic/score/mean", "Training reward", "(a) Training reward"),
    ("actor/kl_loss", "KL to reference", "(b) KL divergence"),
    ("response_length/mean", "Tokens", "(c) Mean response length"),
    ("actor/grad_norm", "Gradient norm", "(d) Actor gradient norm"),
]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#666666", "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

df = pd.read_csv(HERE / "grpo_kl_history.csv", dtype={"kl_coef": str})
fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.9))
for ax, (col, ylabel, title) in zip(axes.flat, PANELS):
    for label in ORDER:
        part = df[df["kl_coef"] == label].sort_values("step").dropna(subset=[col])
        if part.empty:
            continue
        y = part[col].rolling(10, min_periods=1).mean()
        ax.plot(part["step"], y, color=COLORS[label], ls=STYLES[label], lw=1.3,
                label="KL off" if label == "off" else rf"$\beta={label}$")
    ax.set_title(title, loc="left")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 520)
axes[0, 1].set_yscale("log")
axes[1, 1].set_yscale("log")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2, w_pad=1.6)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig6_grpo_kl_dynamics.{ext}", bbox_inches="tight")
