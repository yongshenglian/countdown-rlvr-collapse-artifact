"""Figures and table for the DAPO dynamic-sampling (DS) vs no-DS comparison.

Inputs (all in ROOT):
  dapo_DS_f8h77wol_history.csv     wandb history, DAPO with DS   (run f8h77wol, 0910)
  dapo_noDS_3md2pzp8_history.csv   wandb history, DAPO without DS (run 3md2pzp8, 0915)
  ds_run_p0_p1_per_step.csv        per-step group stats of the DS run, from parse_ds_log.py

Outputs (in OUT):
  fig1_val_acc.pdf/.png   held-out val accuracy vs training step / wall-clock hours / optimizer updates
  fig2_mechanism.pdf/.png what DS filters per step, and the train-reward shift that filtering causes
  table1.tex              setup + results table (LaTeX, booktabs)
  stats.txt               every number used in the table/captions, plus a paired test

Run:  python make_paper_figs.py            (uses the CSVs above)
      python make_paper_figs.py --refresh  (re-download histories from wandb first)

Later, to add results on a new problem set, put a CSV at EXTERNAL_EVAL with columns
run,step,acc   (run in {"DS","noDS"}); it is added to table1.tex as extra columns.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_EVAL = os.path.join(OUT, "external_eval.csv")
ENTITY_PROJECT = "yongshenglian-university-of-louisville/DAPO_new"

# ---- run metadata (edit here) --------------------------------------------------
RUNS = {
    "noDS": dict(label="DAPO w/o DS", id="3md2pzp8", csv="dapo_noDS_3md2pzp8_history.csv", color="#2a78d6"),
    "DS":   dict(label="DAPO w/ DS",  id="f8h77wol", csv="dapo_DS_f8h77wol_history.csv",  color="#eb6834"),
}
N_RESP = 8            # actor_rollout_ref.rollout.n
PPO_MINI_PROMPTS = 32 # actor.ppo_mini_batch_size (prompts); ppo_epochs = 1
B_GEN, B_TRAIN = 192, 32
MAX_STEP = 495        # last step both runs evaluated (DS last val = 495; noDS crashed at 518)
REPORT_STEPS = [100, 200, 300, 400]
THRESHOLDS = [0.6, 0.7]
SMOOTH_EVALS = 5      # rolling window (in evaluations, one every 5 steps) for the solid lines

VAL = "val-core/countdown/acc/mean@1"
N_VAL = 1024          # size of the held-out validation set

# ---- style: thin marks, recessive axes, one y-axis per panel -------------------
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def refresh():
    import wandb
    api = wandb.Api()
    for r in RUNS.values():
        run = api.run(f"{ENTITY_PROJECT}/{r['id']}")
        pd.DataFrame(list(run.scan_history())).to_csv(os.path.join(ROOT, r["csv"]), index=False)
        print("downloaded", r["id"])


def load(key):
    """Per-step training frame and per-eval validation frame, with derived x-axes."""
    h = pd.read_csv(os.path.join(ROOT, RUNS[key]["csv"]))
    t = h[h["critic/score/mean"].notna()].set_index("_step").sort_index()
    # sequences actually trained per step (DS: 256, noDS: 1536), from logged token totals
    seqs = (t["perf/total_num_tokens"] / (t["prompt_length/mean"] + t["response_length/mean"])).round()
    t["prompts_trained"] = seqs / N_RESP
    t["updates"] = (t["prompts_trained"] / PPO_MINI_PROMPTS).cumsum()
    t["hours"] = t["timing_s/step"].cumsum() / 3600
    t["responses_generated"] = (t["train/num_gen_batches"] * B_GEN * N_RESP).cumsum()
    v = h[h[VAL].notna()].set_index("_step")[VAL].sort_index().rename("acc").to_frame()
    v = v[v.index <= MAX_STEP]
    x0 = pd.DataFrame({"updates": [0.0], "hours": [0.0]}, index=[0])
    xs = pd.concat([x0, t[["updates", "hours"]]])
    v = v.join(xs, how="left")
    v["smooth"] = v["acc"].rolling(SMOOTH_EVALS, min_periods=1, center=True).mean()
    return t, v


def first_crossing(v, thr, col):
    s = v["acc"].rolling(3, min_periods=3).mean()  # 3 consecutive evals, avoids one lucky eval
    hit = s[s >= thr]
    return np.nan if hit.empty else v.loc[hit.index[0], col] if col != "step" else hit.index[0]


def paired_signflip(d, n=200_000, seed=0):
    """Two-sided sign-flip permutation p-value for mean(d) == 0 (numpy only)."""
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return (np.abs((flips * d).mean(1)) >= obs).mean()


def fig1(data):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    specs = [("step", "Training step", "(a) Same number of steps"),
             ("hours", "Wall-clock hours", "(b) Same wall-clock time"),
             ("updates", "Optimizer updates", "(c) Same number of updates")]
    for ax, (col, xl, title) in zip(axes, specs):
        for key in ["noDS", "DS"]:
            v = data[key][1]
            if col == "updates":
                v = v[v.index > 0]  # 0 updates cannot sit on a log axis
            x = v.index.values if col == "step" else v[col].values
            c = RUNS[key]["color"]
            ax.plot(x, v["acc"], color=c, lw=0.7, alpha=0.35)
            ax.plot(x, v["smooth"], color=c, lw=2.0, label=RUNS[key]["label"])
        ax.set_xlabel(xl)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(0, 0.85)
        ax.margins(x=0.01)
    axes[2].set_xscale("log")
    axes[2].set_xlim(left=5)
    axes[0].set_ylabel("Val. accuracy (1024 held-out)")
    # direct labels at line ends of panel (a); legend below for identity
    for key in ["noDS", "DS"]:
        v = data[key][1]
        axes[0].annotate(RUNS[key]["label"], (v.index[-1], v["smooth"].iloc[-1]), xytext=(-2, 5 if key == "noDS" else -9),
                         textcoords="offset points", ha="right", fontsize=7, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_val_acc.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig2(g):
    g = g[g.step <= MAX_STEP].set_index("step")
    r = lambda s: s.rolling(5, min_periods=1, center=True).mean()
    p0, p1 = r(g.p0), r(g.p1)
    mixed = 1 - p0 - p1
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.45))

    cols = {"p0": "#4a3aa7", "mixed": "#1baf7a", "p1": "#eda100"}
    a.stackplot(g.index, p0, mixed, p1, colors=[cols["p0"], cols["mixed"], cols["p1"]],
                edgecolor="white", linewidth=0.6)
    a.plot(g.index, np.full(len(g), B_TRAIN / B_GEN), color=INK, lw=1.0, ls="--")
    a.text(g.index[-1] - 4, B_TRAIN / B_GEN - 0.012, f"{B_TRAIN}/{B_GEN} trained", fontsize=7, color="white",
           ha="right", va="top")
    # direct labels inside the bands (fills below 3:1 contrast need visible labels)
    for txt, x, y, col in [("all incorrect ($p_0$)", 8, 0.09, "white"), ("mixed (kept)", 230, 0.36, INK),
                           ("all correct ($p_1$)", 330, 0.85, INK)]:
        a.text(x, y, txt, fontsize=7, color=col, ha="left", va="center")
    a.set_xlim(g.index.min(), g.index.max()); a.set_ylim(0, 1)
    a.set_xlabel("Training step"); a.set_ylabel("Fraction of prompt groups")
    a.set_title("(a) Groups generated per step by the DS run", loc="left")

    c = RUNS["DS"]["color"]
    b.plot(g.index, r(g.acc_all), color=c, lw=2.0, label="before filtering (all generated)")
    b.plot(g.index, r(g.acc_kept), color=c, lw=1.2, ls="--", label="after filtering (trained batch)")
    b.plot(g.index[::10], r(g.acc_pred)[::10], ls="none", marker="o", ms=3, mfc="white", mec=INK2, mew=0.8,
           label=r"predicted $(R-p_1)/(1-p_0-p_1)$")
    b.set_xlim(g.index.min(), g.index.max()); b.set_ylim(0, 0.85)
    b.set_xlabel("Training step"); b.set_ylabel("Training accuracy")
    b.set_title("(b) DS training accuracy, before vs. after filtering", loc="left")
    b.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_mechanism.{ext}"), bbox_inches="tight")
    plt.close(fig)


def table_and_stats(data, g):
    lines, rows = [], {}
    ext = pd.read_csv(EXTERNAL_EVAL) if os.path.exists(EXTERNAL_EVAL) else None
    for key in ["noDS", "DS"]:
        t, v = data[key]
        t = t[t.index <= MAX_STEP]
        near = lambda s: v["acc"].loc[s - 10:s + 10].mean()  # mean of the 5 evals centred on s
        row = {
            "gen": f"{B_GEN}",
            "trained": f"{int(t.prompts_trained.median())}",
            "upd": f"{int(t.prompts_trained.median() / PPO_MINI_PROMPTS)}",
            **{f"s{s}": f"{near(s):.3f}" for s in REPORT_STEPS},
            "last": f"{v['acc'].loc[MAX_STEP - 45:MAX_STEP].mean():.3f}",
            "best": f"{v['acc'].max():.3f}",
            **{f"step{th}": f"{first_crossing(v, th, 'step'):.0f}" if not np.isnan(first_crossing(v, th, 'step')) else "--" for th in THRESHOLDS},
            **{f"h{th}": f"{first_crossing(v, th, 'hours'):.1f}" if not np.isnan(first_crossing(v, th, 'hours')) else "--" for th in THRESHOLDS},
            "hours": f"{t['hours'].iloc[-1]:.1f}",
        }
        if ext is not None:
            e = ext[ext.run == key]
            for _, er in e.iterrows():
                row[f"ext{int(er.step)}"] = f"{er.acc:.3f}"
        rows[key] = row
        lines.append(f"[{key}] {RUNS[key]['label']} run={RUNS[key]['id']}: {row}")

    # paired comparison at matched eval steps (5..MAX_STEP)
    va, vb = data["noDS"][1]["acc"], data["DS"][1]["acc"]
    common = va.index.intersection(vb.index)
    common = common[common > 0]
    d = (va.loc[common] - vb.loc[common]).values
    se = np.sqrt(2 * 0.75 * 0.25 / N_VAL)
    lines += [
        f"matched evals: {len(d)}  noDS>DS in {int((d > 0).sum())}/{len(d)}",
        f"mean diff (noDS-DS) {d.mean():.4f}  min {d.min():.4f}  max {d.max():.4f}",
        f"sign-flip permutation p (two-sided): {paired_signflip(d):.2e}   (note: evals are autocorrelated; treat as descriptive)",
        f"binomial SE of one eval ~{np.sqrt(0.75*0.25/N_VAL):.4f}; SE of a single-eval difference ~{se:.4f}",
        f"DS group stats (steps<= {MAX_STEP}): mean p0 {g.p0.mean():.3f}, p1 {g.p1.mean():.3f}, "
        f"mixed {1-g.f_true.mean():.3f} (= {192*(1-g.f_true.mean()):.0f} of {B_GEN} groups; {B_TRAIN} trained)",
        f"DS within-run check: acc_pred - acc_kept mean {(g.acc_pred-g.acc_kept).mean():.4f}, "
        f"mean |.| {(g.acc_pred-g.acc_kept).abs().mean():.4f}",
    ]
    for key in ["noDS", "DS"]:
        t = data[key][0]
        lines.append(f"[{key}] mean s/step {t['timing_s/step'].mean():.1f}, gen {t['timing_s/gen'].mean():.1f}, "
                     f"update {t['timing_s/update_actor'].mean():.1f}")
    open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    ext_cols = sorted({k for r in rows.values() for k in r if k.startswith("ext")}, key=lambda k: int(k[3:]))
    cols = (["gen", "trained", "upd"] + [f"s{s}" for s in REPORT_STEPS] + ["last", "best"]
            + [f"step{th}" for th in THRESHOLDS] + [f"h{th}" for th in THRESHOLDS] + ["hours"] + ext_cols)
    head1 = (r"& \multicolumn{3}{c}{Prompts / updates per step} & \multicolumn{%d}{c}{Val.\ accuracy} "
             r"& \multicolumn{%d}{c}{Steps to acc.} & \multicolumn{%d}{c}{Hours to acc.} & Total"
             % (len(REPORT_STEPS) + 2, len(THRESHOLDS), len(THRESHOLDS)))
    if ext_cols:
        head1 += r" & \multicolumn{%d}{c}{New problems}" % len(ext_cols)
    head2 = ("Method & Gen. & Trained & Updates & "
             + " & ".join(f"@{s}" for s in REPORT_STEPS) + r" & Last 10 & Best & "
             + " & ".join(f"{th}" for th in THRESHOLDS) + " & "
             + " & ".join(f"{th}" for th in THRESHOLDS) + " & hours"
             + "".join(f" & @{k[3:]}" for k in ext_cols))
    body = []
    for key in ["noDS", "DS"]:
        vals = [rows[key].get(c, "--") for c in cols]
        body.append(RUNS[key]["label"] + " & " + " & ".join(vals) + r" \\")
    tex = "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{DAPO with and without dynamic sampling (DS) on Countdown, Qwen2.5-1.5B-Instruct. "
        r"The two runs differ only in \texttt{filter\_groups.enable}. Both generate "
        rf"{B_GEN}$\times${N_RESP} responses per step; DS trains on the first {B_TRAIN} groups that survive filtering. "
        rf"Val.\ accuracy is on {N_VAL} held-out problems (one sample, $T{{=}}1.0$, top-$p{{=}}0.7$); "
        r"@$s$ is the mean of the five evaluations centred on step $s$; ``Last 10'' averages steps "
        rf"{MAX_STEP-45}--{MAX_STEP}. Steps/hours to accuracy use the first three consecutive evaluations "
        r"averaging at or above the threshold.}",
        r"\label{tab:ds_vs_nods}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", head1 + r" \\", head2 + r" \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    open(os.path.join(OUT, "table1.tex"), "w").write(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download wandb histories first")
    if ap.parse_args().refresh:
        refresh()
    data = {k: load(k) for k in RUNS}
    g = pd.read_csv(os.path.join(ROOT, "ds_run_p0_p1_per_step.csv"))
    fig1(data)
    fig2(g)
    table_and_stats(data, g[g.step <= MAX_STEP])


if __name__ == "__main__":
    main()
