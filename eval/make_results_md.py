"""Write RESULTS.md: every evaluated model with its training configuration and all downstream results.

Sources (all regenerated from saved responses, nothing hand-typed):
  runs_registry.csv                 training settings from wandb configs (run_registry.py)
  results_8k/<bench>/...            8192-token run (main setting)
  results/<bench>/...               2048-token run (base, DS, noDS only)
Also rewrites all_models_8k.csv.

  /home/lian/miniconda3/envs/verl-fork/bin/python make_results_md.py
"""
import datetime
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODELS, mcnemar_p  # noqa: E402

BENCHES = [("gsm8k", "GSM8K", 1319), ("math", "MATH", 5000), ("bbh", "BBH", 6511), ("mmlupro", "MMLU-Pro", 12032)]
STYLES = ["standard", "think"]
ORDER = ["base", "DS_step300", "noDS_step300", "noDS_step100", "noDS_step200", "noDS_step400", "noDS_step500",
         "noDS_step520", "rung2_step100", "rung2_step300", "rung2_step520", "rung1_step100", "rung1_step300",
         "rung1_step520", "rung2s2_step300", "rung2s2_step520", "rungA_step300", "rungA_step520", "rungB_step300",
         "rungB_step520", "rungC_step300", "rungC_step520", "rungD_step300", "rungD_step520", "rungA1_step300", "rungA1_step520", "rungA1s2_step300", "rungA1s2_step520", "DSs2_step200", "PPO_step400", "PPO_klF_ent0_lr1e-6_step520",
         "PPO_klF_ent0_lr3e-6_step520", "PPO_klF_ent1e-3_lr3e-6_step520", "PPO_kl0_ent1e-3_lr3e-6_step520",
         "PPO_kl1e-3_ent1e-3_lr3e-6_step520",
         "GRPO_n2_noKL_step250", "GRPO_n2_kl1e-3_step250", "GRPO_n2_kl5e-3_step250", "GRPO_n2_kl4e-2_step250",
         "GRPO_n2_noKL_step520", "GRPO_n2_kl1e-3_step520", "GRPO_n2_kl5e-3_step520", "GRPO_n2_kl1e-2_step250",
         "GRPO_n2_kl4e-2_step520", "GRPO_n4_kl1e-3_step520", "GRPO_n8_kl1e-3_step520", "GRPO_n4_kl1e-6_step520",
         "GRPO_n4_kl1e-6_eqR_step520"]
ORDER = [m for m in ORDER if m in MODELS] + [m for m in MODELS if m not in ORDER]


def paired_diff_ci(trained, base, z=1.96):
    """Asymptotic 95% CI for a paired accuracy difference, in percentage points."""
    fixes = int((~base & trained).sum())
    losses = int((base & ~trained).sum())
    n = len(base)
    diff = (fixes - losses) / n
    variance = ((fixes + losses) / n - diff ** 2) / n
    half_width = z * math.sqrt(max(0.0, variance))
    return 100 * (diff - half_width), 100 * (diff + half_width)


def path(res, b, m, s):
    d = os.path.join(HERE, res, b)
    return os.path.join(d, f"{m}__{s}.jsonl") if b == "gsm8k" else os.path.join(d, "scored", f"{m}__{s}.jsonl")


def available(res, m):
    return all(os.path.exists(path(res, b, m, s)) for b, _, _ in BENCHES for s in STYLES)


def collect(res, models):
    rows, frames = [], {}
    for b, _, _ in BENCHES:
        for s in STYLES:
            for m in models:
                r = pd.read_json(path(res, b, m, s), lines=True, dtype=False)
                frames[(b, s, m)] = r
            base = frames[(b, s, "base")].correct_strict.values.astype(bool)
            for m in models:
                r = frames[(b, s, m)]
                a = r.correct_strict.values.astype(bool)
                rows.append(dict(bench=b, prompt=s, model=m, n=len(r), acc=100 * a.mean(),
                                 flex=100 * r.correct_flexible.astype(bool).mean(),
                                 truncated=100 * (r.finish == "length").mean(),
                                 median_chars=int(r.response.str.len().median()),
                                 vs_base=100 * (a.mean() - base.mean()),
                                 p_vs_base=mcnemar_p(a, base)[2] if m != "base" else None))
    return pd.DataFrame(rows), frames


def label(m, reg):
    """Readable name from the recorded settings (not from the run name)."""
    if m == "base":
        return "Base Qwen2.5-1.5B-Instruct"
    if m not in reg.index:
        return m
    r = reg.loc[m]
    step = int(r.eval_step)
    kl = "KL off" if not bool(r.use_kl_loss) else (
        "KL on, coef 0" if float(r.kl_loss_coef) == 0 else f"KL {r.kl_loss_coef:g}")
    if m.startswith("DS_") or m.startswith("noDS_"):
        return f"DAPO {'w/' if bool(r.filter_groups) else 'w/o'} DS, n={int(r.rollout_n)} (step {step})"
    extra = {"rung2s2": ", seed 2", "rungA": " + correctness-only reward", "rungB": " + no std normalisation",
             "rungC": " + masked uninformative groups", "rungD": " + batch-level advantage normalisation",
             "rungA1": " + correctness-only reward + overlong filtering",
             "rungA1s2": " + correctness-only reward + overlong filtering, seed 2"}.get(m.split("_")[0], "")
    if m.startswith("DSs2"):
        return f"DAPO w/ DS, training seed 2 (collapsed; step {step})"
    if r.wandb_project == "DAPO_ablation" and extra:
        return (f"Ablation rung 2{extra}: clip {r.clip_ratio_low:g}/{r.clip_ratio_high:g}, no DS, "
                f"{int(r.train_batch_size)} prompts/step (step {step})")
    if r.wandb_project == "DAPO_ablation":
        return (f"Ablation {m.split('_')[0]}: clip {r.clip_ratio_low:g}/{r.clip_ratio_high:g}, "
                f"{'DS' if str(r.filter_groups) == 'True' else 'no DS'}, {int(r.train_batch_size)} prompts/step (step {step})")
    if r.adv_estimator == "gae":
        return (f"PPO lr {r.actor_lr:g}, {kl}, entropy {r.entropy_coeff:g}, mini {int(r.ppo_mini_batch_size)} "
                f"(step {step})")
    eq = ", equal 0.5/0.5 reward" if str(r.get("reward", "")).startswith("EQUAL") else ""
    return f"GRPO n={int(r.rollout_n)}, {kl}{eq} (step {step})"


def steps_table(reg, order):
    """Prompts / responses / optimizer updates per step and at the evaluated checkpoint."""
    ngen = None
    h = os.path.join(os.path.dirname(HERE), "dapo_DS_f8h77wol_history.csv")
    if os.path.exists(h):
        hh = pd.read_csv(h)
        ngen = hh[hh["train/num_gen_batches"].notna()].set_index("_step")["train/num_gen_batches"]
    rows = []
    for m in order:
        if m not in reg.index:
            continue
        r = reg.loc[m]
        n, step, mini = int(r.rollout_n), int(r.eval_step), int(r.ppo_mini_batch_size)
        ds = str(r.filter_groups) == "True"
        if r.wandb_project in ("DAPO_new", "DAPO_ablation"):
            gen_b = int(r.gen_batch_size)
            trained = int(r.train_batch_size) if ds else gen_b
            gen = gen_b * (ngen.loc[:step].mean() if ds and ngen is not None else 1.0)
        else:
            trained = int(r.train_batch_size)
            gen = trained
        upd = trained / mini * int(r.ppo_epochs)
        rows.append({"model": m, "step": step, "prompts generated / step": round(gen, 1),
                     "prompts trained / step": trained, "responses trained / step": trained * n,
                     "updates / step": int(upd), "prompts trained at eval step": trained * step,
                     "updates at eval step": int(upd * step)})
    return pd.DataFrame(rows).astype(str)


CONTROLLED = [
    "- **Original DAPO w/ DS vs. w/o DS: not controlled.** Without DS the trainer used all 192 generated prompts "
    "(1,536 responses, 6 updates per step); with DS it retained 32 prompts (256 responses, 1 update). The original "
    "benchmark gap therefore cannot be attributed to filtering. The completed matched comparison uses 32 prompts "
    "and one update per step in both conditions (section 3.10), but it has limited seed replication and the matched "
    "no-DS run was trained in a later software environment (`../ablation/README.md`).",
    "- **GRPO KL sweep (n = 2, lr 1e-6, mini-batch 64):** only the KL coefficient differs. Step-matched at 250 for "
    "all five coefficients (section 3.7); the KL 0.01 run exists only at step 250.",
    "- **GRPO group size (n = 2 / 4 / 8 at KL 0.001):** prompts and updates per step are equal (192, 3), but "
    "responses per step grow with n (384 / 768 / 1,536), so group size and sample budget change together.",
    "- **GRPO reward (equal 0.5/0.5 vs. standard, n = 4, KL 1e-6): not controlled.** The equal-reward run "
    "(`checkagain`) was trained with `/home/lian/verl_fork` = verl 0.5.0.dev, the others with `/home/lian/verl` = "
    "verl 0.2.0.dev; 135 source files differ, including `trainer/ppo/core_algos.py`, `workers/actor/dp_actor.py` "
    "and `trainer/ppo/ray_trainer.py`.",
    "- **PPO (October runs, mini-batch 16): not a full grid.** Learning rate is compared only with KL off and "
    "entropy 0; entropy only at lr 3e-6 with KL off; KL only at lr 3e-6 with entropy 0.001. 'KL off' and 'KL on, "
    "coefficient 0' should be equivalent but differ by up to about 20 points, so KL and entropy differences at "
    "3e-6 are within seed noise. The learning-rate effect holds up because all four 3e-6 runs collapse (rough "
    "replicates). PPO_step400 is not part of these sweeps (mini-batch 64, Sep reward, step 400).",
    "- **Across families (DAPO vs. GRPO vs. PPO): complete recipes, not controlled.** They differ in response cap "
    "(4,096 vs. 2,048), responses and updates per step (table above), reward, n, KL, clip-high (0.28 vs. 0.2), "
    "learning-rate warmup (10 vs. none), trainer code (`recipe/dapo` vs. `main_ppo`), critic (PPO) and the "
    "evaluated step.",
]


def md_table(df, floatfmt="{:.1f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pfmt(p):
    return "" if p is None or pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def main():
    reg = pd.read_csv(os.path.join(HERE, "runs_registry.csv")).set_index("model_key")
    models8 = [m for m in ORDER if available("results_8k", m)]
    d8, f8 = collect("results_8k", models8)
    d8.round(4).to_csv(os.path.join(HERE, "all_models_8k.csv"), index=False)
    models2 = [m for m in ORDER if available("results", m)]
    d2, f2 = collect("results", models2)

    L = []
    L.append("# Downstream evaluation of Countdown-trained checkpoints")
    L.append(f"\nGenerated {datetime.date.today()} by `make_results_md.py` from the saved responses. "
             "Do not edit by hand; rerun the script after adding models.\n")

    L.append("## 1. Evaluation protocol (identical for every model)\n")
    L.append("- **Benchmarks (full test sets):** " + ", ".join(f"{n} (n={k:,})" for _, n, k in BENCHES)
             + ". MATH-500 is also reported inside `results_8k/math/summary.txt`.")
    L.append("- **Prompts:** *standard* = the benchmark's usual zero-shot chain-of-thought prompt "
             "(GSM8K: `####`; MATH: `\\boxed{}`; BBH: \"So the answer is\"; MMLU-Pro: \"the answer is (X)\"). "
             "*think* = the Countdown training wrapper with `<think>`/`<answer>` tags (`common.THINK_TEMPLATE`). "
             "BBH and MMLU-Pro are zero-shot (official protocols are 3-shot / 5-shot).")
    L.append("- **Chat template:** Qwen2.5 template, one user message (identical files for all models).")
    L.append("- **Decoding:** vLLM, greedy (temperature 0), no repetition penalty, max 8192 output tokens "
             "(main setting, `results_8k/`); a 2048-token run of base/DS/noDS is in `results/`.")
    L.append("- **Scoring:** *strict* = answer inside the last explicit final-answer marker "
             "(`####`, `\\boxed{}`, `<answer>`, \"the answer is\"); *flexible* = strict, else last number "
             "(GSM8K), math-verify over the whole response (MATH), last `(X)` (BBH multiple choice, MMLU-Pro). "
             "MATH answers are checked with math-verify (symbolic equivalence). All accuracies are in %.")
    L.append("- **Statistics:** paired exact McNemar test on the same problems. It tests whether two *checkpoints* "
             "differ on these test items. It does not include seed-to-seed training variance, so it does not test "
             "whether two *training procedures* differ; every setting here has one training seed.\n")

    L.append("## 2. Models and training configuration\n")
    L.append("Settings are read from each run's wandb config (`runs_registry.csv`, produced by `run_registry.py`), "
             "not from run names. Names are unreliable: all PPO runs are named \"mini32\" but used "
             "`ppo_mini_batch_size` 64 (PPO_step400) or 16 (all PPO_Countdown_R1 runs); \"klloss0\" means KL loss "
             "*enabled* with coefficient 0.\n")
    rows = [dict(model="base", source="Qwen/Qwen2.5-1.5B-Instruct (Hugging Face)", run="-", step="-")]
    cfg_cols = [("adv_estimator", "adv"), ("actor_lr", "actor lr"), ("critic_lr", "critic lr"),
                ("use_kl_loss", "KL loss"), ("kl_loss_coef", "KL coef"), ("use_kl_in_reward", "KL in reward"),
                ("entropy_coeff", "entropy"), ("rollout_n", "n"), ("train_batch_size", "train bsz"),
                ("gen_batch_size", "gen bsz"), ("ppo_mini_batch_size", "mini bsz"), ("ppo_epochs", "epochs"),
                ("clip_ratio_low", "clip low"), ("clip_ratio_high", "clip high"), ("loss_agg_mode", "loss agg"),
                ("filter_groups", "DS filter"), ("max_response_length", "max resp"),
                ("logged_score_values", "logged reward values"),
                ("countdown_val_metric", "Countdown val metric"), ("countdown_val_at_eval_step", "Countdown val @step"),
                ("countdown_val_best", "Countdown val best")]
    cfg = []
    for m in ORDER:
        if m == "base":
            continue
        if m not in reg.index:
            continue  # not trained / not registered yet
        r = reg.loc[m]
        row = {"model": m, "wandb": f"{r.wandb_project}/{r.wandb_run}", "step": int(r.eval_step)}
        for c, lab in cfg_cols:
            v = r.get(c)
            if c == "critic_lr" and r.adv_estimator == "grpo":
                v = "-"
            if c == "kl_loss_coef" and not bool(r.use_kl_loss):
                v = "- (off)"  # coefficient present in config but unused when KL loss is disabled
            if c == "gen_batch_size":
                v = "-" if pd.isna(v) else int(v)
            if c == "filter_groups" and pd.isna(v):
                v = "-"
            if isinstance(v, float) and not pd.isna(v) and v.is_integer() and c not in ("actor_lr", "critic_lr"):
                v = int(v)
            row[lab] = v if not isinstance(v, float) or not pd.isna(v) else "-"
        cfg.append(row)
    notes = [(m, reg.loc[m].get("reward"), reg.loc[m].get("codebase"), reg.loc[m].get("note"))
             for m in ORDER if m in reg.index]
    cfgdf = pd.DataFrame(cfg).astype(str)
    L.append(md_table(cfgdf))
    L.append("\nCommon to all trained models: base `/home/lian/models/qwen2.5-1.5b-instruct` (Qwen2.5-1.5B-Instruct), "
             "Countdown data `/home/lian/data/countdown/train.parquet`, rollout temperature 1.0, 2 epochs, "
             "gamma = lam = 1, KL penalty in reward off. DAPO runs log Countdown *accuracy*; PPO and GRPO runs log "
             "Countdown *reward*, which includes format credit and depends on the reward scheme, so Countdown "
             "numbers are only comparable between runs with the same metric and reward.\n")
    L.append("Reward scheme, codebase and notes (not in the wandb config; established from each run's logged "
             "reward values and printed reward output, and from the location of its local wandb folder):\n")
    L.append(md_table(pd.DataFrame(notes, columns=["model", "training reward", "codebase", "note"]).fillna("")))
    L.append("")
    L.append("Second source for the settings: the Hydra config saved when each run was launched (copied to "
             "`run_configs/<model>.yaml`), compared with the wandb config on every setting in the table above:\n")
    prov = [(m, reg.loc[m].get("launch_config_vs_wandb"), reg.loc[m].get("launch_config"))
            for m in ORDER if m in reg.index]
    L.append(md_table(pd.DataFrame(prov, columns=["model", "launch config vs. wandb", "launch config file"]).fillna("")))
    L.append("")
    L.append("### What one training step means in each recipe\n")
    L.append("Steps are not comparable across recipes. Trained prompts per step follow from the trainer code: with "
             "DS off, the DAPO trainer trains on the whole generated batch (`batch = new_batch`, "
             "`recipe/dapo/src/dapo_ray_trainer.py` line 170), so `train_batch_size=32` is ignored; with DS on it "
             "cuts the kept groups to `train_batch_size` (lines 215-216). Both are confirmed by the logged token counts "
             "(1,536 sequences per step without DS, 256 with DS, at every step). Optimizer updates per step = trained "
             "prompts / mini-batch x PPO epochs (PPO also updates its critic, not counted).\n")
    L.append(md_table(steps_table(reg, ORDER)))
    L.append("")
    L.append("### Which comparisons are controlled\n")
    L += CONTROLLED
    L.append("")
    L.append("Checkpoint paths:\n")
    for m in ORDER:
        if m in reg.index:
            L.append(f"- `{m}`: `{reg.loc[m].checkpoint}` (run name `{reg.loc[m].run_name}`) "
                     f"-> merged `{MODELS[m]}`")
    L.append("- `base`: `Qwen/Qwen2.5-1.5B-Instruct` (HF hub; the original local copy used for training no longer exists)\n")

    L.append("## 3. Results, 8192 output tokens (main setting)\n")
    L.append("### 3.1 Strict accuracy (%)\n")
    pv = d8.pivot_table(index="model", columns=["bench", "prompt"], values="acc").reindex(models8)
    pv.columns = [f"{dict((b, n) for b, n, _ in BENCHES)[b]} {s}" for b, s in pv.columns]
    pv = pv[[f"{n} {s}" for _, n, _ in BENCHES for s in STYLES]]
    pv["avg (8)"] = pv.mean(axis=1)
    pv["avg std"] = pv[[c for c in pv.columns if c.endswith("standard")]].mean(axis=1)
    pv["avg think"] = pv[[c for c in pv.columns if c.endswith("think")]].mean(axis=1)
    L.append(md_table(pv.reset_index()))
    L.append("\n### 3.2 Ranking by average strict accuracy over all 4 benchmarks x 2 prompts\n")
    rk = pv[["avg (8)", "avg std", "avg think"]].sort_values("avg (8)", ascending=False).reset_index()
    rk.insert(0, "rank", range(1, len(rk) + 1))
    L.append(md_table(rk))
    L.append("\n### 3.3 Flexible accuracy (%)\n")
    fv = d8.pivot_table(index="model", columns=["bench", "prompt"], values="flex").reindex(models8)
    fv.columns = [f"{b} {s}" for b, s in fv.columns]
    L.append(md_table(fv.reset_index()))
    L.append("\n### 3.4 Responses cut off at 8192 tokens (%)\n")
    tv = d8.pivot_table(index="model", columns=["bench", "prompt"], values="truncated").reindex(models8)
    tv.columns = [f"{b} {s}" for b, s in tv.columns]
    L.append(md_table(tv.reset_index()))
    L.append("\n### 3.5 Difference from base model (strict, points) and McNemar p\n")
    dd = d8[d8.model != "base"].copy()
    dd["cell"] = dd.apply(lambda r: f"{r.vs_base:+.1f} (p={pfmt(r.p_vs_base)})", axis=1)
    cv = dd.pivot_table(index="model", columns=["bench", "prompt"], values="cell", aggfunc="first").reindex(
        [m for m in models8 if m != "base"])
    cv.columns = [f"{b} {s}" for b, s in cv.columns]
    L.append(md_table(cv.reset_index()))
    L.append("\n### 3.6 DAPO w/o DS vs. w/ DS (paired, strict)\n")
    t = []
    for res, fr, lab in [("results_8k", f8, "8192"), ("results", f2, "2048")]:
        for b, name, _ in BENCHES:
            for s in STYLES:
                if (b, s, "noDS_step300") not in fr:
                    continue
                a = fr[(b, s, "noDS_step300")].correct_strict.values.astype(bool)
                c = fr[(b, s, "DS_step300")].correct_strict.values.astype(bool)
                n10, n01, p = mcnemar_p(a, c)
                t.append(dict(tokens=lab, benchmark=name, prompt=s, noDS=100 * a.mean(), DS=100 * c.mean(),
                              diff=100 * (a.mean() - c.mean()), noDS_only=n10, DS_only=n01, p=pfmt(p)))
    L.append(md_table(pd.DataFrame(t)))

    L.append("\n### 3.7 GRPO KL sweep at a matched step (n = 2, step 250)\n")
    k250 = [m for m in ["GRPO_n2_noKL_step250", "GRPO_n2_kl1e-3_step250", "GRPO_n2_kl5e-3_step250",
                        "GRPO_n2_kl1e-2_step250", "GRPO_n2_kl4e-2_step250"] if m in pv.index]
    if len(k250) > 1:
        t7 = pv.loc[k250].copy()
        t7.insert(0, "KL coef", [label(m, reg).split(", ")[1].split(" (")[0] for m in k250])
        L.append(md_table(t7.reset_index()))
        mc = d8[d8.model.isin(k250)].pivot_table(index="model", columns=["bench", "prompt"], values="median_chars").reindex(k250)
        mc.columns = [f"{b} {p}" for b, p in mc.columns]
        L.append("\nMedian response length (characters):\n")
        L.append(md_table(mc.reset_index(), floatfmt="{:.0f}"))
    else:
        L.append("Pending: step-250 evaluations of the other n = 2 runs.")

    L.append("\n### 3.8 DAPO w/o DS checkpoint series (GSM8K and MATH)\n")
    L.append("Same run (3md2pzp8) evaluated at several steps, to relate in-domain Countdown accuracy to transfer. "
             "Countdown validation accuracy is the run's own logged value (sampled, top-p 0.7) at that step.\n")
    series = [m for m in ["noDS_step100", "noDS_step200", "noDS_step300", "noDS_step400", "noDS_step500", "noDS_step520"]
              if all(os.path.exists(path("results_8k", b, m, s_)) for b in ["gsm8k", "math"] for s_ in STYLES)]
    if len(series) > 1:
        srows = []
        for m in series:
            row = {"model": m, "step": int(m.split("step")[1])}
            if m in reg.index:
                pass
            for b, name in [("gsm8k", "GSM8K"), ("math", "MATH")]:
                for s_ in STYLES:
                    r = pd.read_json(path("results_8k", b, m, s_), lines=True, dtype=False)
                    row[f"{name} {s_}"] = 100 * r.correct_strict.astype(bool).mean()
            srows.append(row)
        sd = pd.DataFrame(srows)
        hf = os.path.join(os.path.dirname(HERE), "dapo_noDS_3md2pzp8_history.csv")
        if os.path.exists(hf):
            hh = pd.read_csv(hf)
            v = hh[hh["val-core/countdown/acc/mean@1"].notna()].set_index("_step")["val-core/countdown/acc/mean@1"]
            sd.insert(2, "Countdown val acc", [100 * v.loc[max(0, st - 5):st + 5].mean() for st in sd.step])
        L.append(md_table(sd))
    else:
        L.append("Pending: evaluations at steps 100, 200, 400, 500, 520.")

    L.append("\n### 3.9 Why GSM8K gains a lot and MATH does not (standard prompt)\n")
    L.append("Each trained model's change over the base model is split item by item using tolerant correctness first. "
             "*Tolerant fix*: base wrong and trained correct under tolerant scoring. *Tolerant loss*: base correct and "
             "trained wrong under tolerant scoring. *Marker win/loss*: both responses are tolerant-correct, but the "
             "trained/base response alone satisfies the strict final-answer-marker rule. Percent of items. Full analysis, including an arithmetic-"
             "slip check: `analysis_gsm8k_vs_math.py` / `.txt`.\n")
    dec = []
    for bench, name in [("gsm8k", "GSM8K"), ("math", "MATH")]:
        if "base" not in models8:
            break
        b = pd.read_json(path("results_8k", bench, "base", "standard"), lines=True, dtype=False)
        bs, bf = b.correct_strict.astype(bool), b.correct_flexible.astype(bool)
        for m in ["DS_step300", "noDS_step300", "GRPO_n2_kl4e-2_step250", "GRPO_n2_kl1e-2_step250",
                  "PPO_klF_ent0_lr1e-6_step520"]:
            if m not in models8:
                continue
            t = pd.read_json(path("results_8k", bench, m, "standard"), lines=True, dtype=False)
            ts, tf = t.correct_strict.astype(bool), t.correct_flexible.astype(bool)
            fix, loss = 100 * (~bf & tf).mean(), 100 * (bf & ~tf).mean()
            ci_low, ci_high = paired_diff_ci(tf, bf)
            dec.append({"benchmark": name, "model": m, "total gain": 100 * (ts.mean() - bs.mean()),
                        "tolerant fixes": fix, "tolerant losses": loss, "net tolerant gain": fix - loss,
                        "net 95% CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
                        "marker wins": 100 * (bf & tf & ~bs & ts).mean(),
                        "marker losses": 100 * (bf & tf & bs & ~ts).mean(),
                        "flexible p": pfmt(mcnemar_p(tf.values, bf.values)[2])})
    if dec:
        L.append(md_table(pd.DataFrame(dec)))
    mk = []
    for m in ["base", "noDS_step300", "GRPO_n2_kl4e-2_step250"]:
        for bench, name in [("gsm8k", "GSM8K"), ("math", "MATH")]:
            if m not in models8:
                continue
            t = pd.read_json(path("results_8k", bench, m, "standard"), lines=True, dtype=False).response
            h, bx = t.str.contains("####", regex=False), t.str.contains("\\boxed", regex=False)
            mk.append({"model": m, "benchmark": name, "writes ####": 100 * h.mean(), "writes \\boxed{}": 100 * bx.mean(),
                       "neither": 100 * (~h & ~bx).mean()})
    L.append("\nFinal-answer markers written (standard prompt, % of responses):\n")
    L.append(md_table(pd.DataFrame(mk)))
    L.append("\nThe base model already writes `\\boxed{}` in 95% of MATH responses (Qwen2.5-Instruct's math convention), "
             "so MATH has no format gain to collect, at any difficulty level. GSM8K's `####` is dataset-specific and the "
             "base model omits any marker in 54% of responses; Countdown RL, whose reward requires an explicit final "
             "answer, teaches the models to give one. Arithmetic slips are rare in base-model GSM8K errors (about 4%) "
             "and are not reduced by training (explicit calculations wrong 2.7% base vs. 2.0-2.3% for the best recipes).")

    L.append("\n### 3.10 Controlled dynamic-sampling test: ablation rung 2 vs. DS\n")
    L.append("Rung 2 has the DS run's exact settings (KL off, entropy 0, clip 0.2/0.28, lr 1e-6, n = 8, 4,096-token cap, "
             "Sep reward) and the same trained batch (32 prompts, 1 update per step); only dynamic sampling is off "
             "(`../ablation/README.md`). The old no-DS run is shown for reference (192 prompts, 6 updates per step).\n")
    root = os.path.dirname(HERE)
    hist = {"DS (f8h77wol)": os.path.join(root, "dapo_DS_f8h77wol_history.csv"),
            "rung 2 (no DS, 32/step)": os.path.join(root, "ablation", "rung2_5vmxmx4q_history.csv"),
            "old no-DS (192/step)": os.path.join(root, "dapo_noDS_3md2pzp8_history.csv"),
            "rung 1 (no DS, clip 0.2)": os.path.join(root, "ablation", "rung1_history.csv")}
    hist = {k: v for k, v in hist.items() if os.path.exists(v)}
    if len(hist) >= 3:
        tc = []
        for lab, f in hist.items():
            hh = pd.read_csv(f)
            ser = lambda k: hh[hh[k].notna()].set_index("_step")[k] if k in hh else pd.Series(dtype=float)
            ln, va = ser("response_length/mean"), ser("val-core/countdown/acc/mean@1")
            for st in [25, 50, 100, 300, 500]:
                near = lambda x: x.loc[max(0, st - 4):st + 4].mean() if len(x) else float("nan")
                tc.append({"run": lab, "step": st, "mean response tokens": near(ln), "Countdown val acc": 100 * near(va)})
        tcd = pd.DataFrame(tc)
        L.append("Training curves (Countdown):\n")
        piv = tcd.pivot_table(index="step", columns="run", values=["mean response tokens", "Countdown val acc"])
        piv.columns = [f"{v}: {r}" for v, r in piv.columns]
        L.append(md_table(piv.reset_index(), floatfmt="{:.0f}"))
    pair = [("rung2_step300", "DS_step300")]
    if all(m in models8 for p_ in pair for m in p_):
        L.append("\nBenchmarks at step 300, paired (strict, 8,192 tokens; positive = DS higher):\n")
        t = []
        for b, name, _ in BENCHES:
            for s_ in STYLES:
                a = f8[(b, s_, "DS_step300")].correct_strict.values.astype(bool)
                c = f8[(b, s_, "rung2_step300")].correct_strict.values.astype(bool)
                n10, n01, pv_ = mcnemar_p(a, c)
                t.append({"benchmark": name, "prompt": s_, "DS": 100 * a.mean(), "rung 2": 100 * c.mean(),
                          "DS - rung 2": 100 * (a.mean() - c.mean()), "DS only": n10, "rung 2 only": n01, "p": pfmt(pv_)})
        L.append(md_table(pd.DataFrame(t)))
    else:
        L.append("\nPending: benchmark evaluation of rung 2 at step 300.")
    if all(m in models8 for m in ["rung1_step300", "rung2_step300"]):
        L.append("\nClip-higher without DS: rung 2 (clip 0.28) vs. rung 1 (clip 0.2), step 300, paired "
                 "(strict; positive = rung 2 higher):\n")
        t = []
        for b, name, _ in BENCHES:
            for s_ in STYLES:
                a = f8[(b, s_, "rung2_step300")].correct_strict.values.astype(bool)
                c = f8[(b, s_, "rung1_step300")].correct_strict.values.astype(bool)
                n10, n01, pv_ = mcnemar_p(a, c)
                t.append({"benchmark": name, "prompt": s_, "rung 2": 100 * a.mean(), "rung 1": 100 * c.mean(),
                          "rung 2 - rung 1": 100 * (a.mean() - c.mean()), "p": pfmt(pv_)})
        L.append(md_table(pd.DataFrame(t)))

    L.append("\n### 3.11 Getting DS's benefit without resampling: options A, B, C (step 300)\n")
    L.append("Each option is ablation rung 2 (no DS, 32 prompts generated and trained per step) with one change: "
             "A = correctness-only reward (no 0.1 format credit); B = advantages not divided by the group std; "
             "C = groups whose 8 answers are all right or all wrong are masked out of the loss instead of resampled; "
             "D = batch-level advantage normalisation (group mean subtracted, then whitened over the batch; "
             "Lite PPO, arXiv:2508.08221). "
             "Motivation (`../ablation/analysis_why_ds.py`): in rung 2's first 25 steps, 74-87% of the training signal "
             "came from groups where every answer was wrong and scores differed only by the 0.1 format credit, which "
             "std normalisation turns into full-size advantages; DS drops those groups because it filters on accuracy.\n")
    root = os.path.dirname(HERE)
    hmap = {"DS_step300": os.path.join(root, "dapo_DS_f8h77wol_history.csv"),
            "rung2_step300": os.path.join(root, "ablation", "rung2_5vmxmx4q_history.csv"),
            "rung2s2_step300": os.path.join(root, "ablation", "rung2_seed2_history.csv"),
            "rungA_step300": os.path.join(root, "ablation", "rungA_history.csv"),
            "rungB_step300": os.path.join(root, "ablation", "rungB_history.csv"),
            "rungC_step300": os.path.join(root, "ablation", "rungC_history.csv"),
            "rungD_step300": os.path.join(root, "ablation", "rungD_history.csv"),
            "rungA1_step300": os.path.join(root, "ablation", "rungA1_history.csv"),
            "rungA1s2_step300": os.path.join(root, "ablation", "rungA1_seed2_history.csv")}
    rows = []
    for m, hf in hmap.items():
        row = {"model": m, "label": label(m, reg) if m in reg.index else m}
        if os.path.exists(hf):
            hh = pd.read_csv(hf)
            for k, lab in [("val-core/countdown/acc/mean@1", "Countdown val acc (%)"), ("response_length/mean", "mean response tokens")]:
                if k in hh:
                    x = hh[hh[k].notna()].set_index("_step")[k]
                    v = x.loc[295:305].mean()
                    row[lab] = 100 * v if "acc" in k else v
        if m in pv.index:
            row["benchmark avg std"] = pv.loc[m, "avg std"]; row["benchmark avg think"] = pv.loc[m, "avg think"]
        if len(row) > 2:
            rows.append(row)
    if len(rows) > 2:
        L.append(md_table(pd.DataFrame(rows)))
    else:
        L.append("Pending: options A, B, C are queued after ablation rung 2 seed 2.")

    L.append("\n### 3.12 pass@k (does RL solve more problems, or the same ones more reliably?)\n")
    L.append("Following Yue et al. (arXiv:2504.13837): standard prompts, temperature 0.6, top-p 0.95, up to 4,096 tokens; "
             "MATH-500 with 64 samples per problem, GSM8K with 32; unbiased pass@k estimator; strict and tolerant scoring "
             "(`eval_passk.py`, `results_passk/`). Similarity at the largest sampled k is evidence consistent with "
             "probability redistribution, not proof that the models have identical latent solution support.\n")
    pk = os.path.join(HERE, "results_passk", "summary.csv")
    if os.path.exists(pk):
        P = pd.read_csv(pk)
        for bench, name in [("math500", "MATH-500"), ("gsm8k", "GSM8K")]:
            t = P[(P.bench == bench) & (P.metric == "strict")]
            if len(t):
                tt = t.pivot_table(index="model", columns="k", values="pass_at_k")
                tt.columns = [f"pass@{k}" for k in tt.columns]
                L.append(f"{name} (strict, %):\n")
                L.append(md_table(tt.reset_index()))
                L.append("")
    else:
        L.append("Pending.")

    L.append("\n### 3.13 Collapse screen: how often does each recipe collapse? (first 100 steps)\n")
    cs = os.path.join(os.path.dirname(HERE), "ablation", "collapse_screen.md")
    if os.path.exists(cs):
        body = open(cs).read().split("\n", 2)[2]  # drop the file's own title line
        L.append(body.strip())
        L.append("\nNote: DS seed 1 is the only DS run trained in the original Sep-2025 environment (old `verl` conda env, "
                 "original local base-model copy, original reward code); all other ablation runs used the current setup "
                 "(verl-fork env, Hugging Face Qwen2.5-1.5B-Instruct, reconstructed reward). Step-0 Countdown accuracy "
                 "matched (3.0%), and healthy runs also occur in the current setup (A1 seed 1, no-DS seeds 2-4), but an "
                 "environment effect on DS seed 1 cannot be ruled out.")
    else:
        L.append("Pending.")

    L.append("\n## 4. Results, 2048 output tokens (base, DS, noDS)\n")
    p2 = d2.pivot_table(index="model", columns=["bench", "prompt"], values="acc").reindex(models2)
    p2.columns = [f"{b} {s}" for b, s in p2.columns]
    L.append(md_table(p2.reset_index()))
    tr2 = d2.pivot_table(index="model", columns=["bench", "prompt"], values="truncated").reindex(models2)
    tr2.columns = [f"{b} {s}" for b, s in tr2.columns]
    L.append("\nTruncated at 2048 tokens (%):\n")
    L.append(md_table(tr2.reset_index()))

    L.append("\n## 5. Main findings\n")
    L.append("See section 0 (Summary) at the top of this file.\n")

    L.append("## 6. Files\n")
    L.append("- Per-response outputs: `results_8k/<bench>/<model>__<prompt>.jsonl` (GSM8K) and "
             "`results_8k/<bench>/scored/<model>__<prompt>.jsonl` (others, with extracted answers).")
    L.append("- Summaries: `results_8k/<bench>/summary.txt|csv`, per-group breakdowns `by_subject.csv`, "
             "`by_task.csv`, `by_category.csv`; all models in one file: `all_models_8k.csv`.")
    L.append("- Training settings: `runs_registry.md|csv` (`run_registry.py`).")
    L.append("- Paper table: `../paper_figs/table2.tex` (`make_table2.py`); 2048 vs 8192 comparison: "
             "`compare_lengths.txt` (`compare_lengths.py`).")
    L.append("- Scripts: `merge_checkpoints.sh`, `run_model'sh`, `run_queue.sh`, `run_all.sh`, "
             "`eval_{gsm8k,math,bbh,mmlupro}.py`, `common.py`.")

    F = []
    F.append("- **These are comparisons of complete training recipes, not a single controlled setting.** DAPO, "
             "GRPO and PPO differ in response cap, responses and updates per step, reward, n, KL, clip-high, warmup, "
             "trainer code and evaluated step (section 2, 'Which comparisons are controlled').")
    F.append("- **Highest average among the legacy headline checkpoints: DAPO w/o DS (step 300)**, and highest on GSM8K. "
             "Its gap over the original DS checkpoint cannot be attributed to dynamic sampling: no-DS trained on 192 "
             "prompts and made 6 updates per step, whereas DS trained on 32 prompts and made 1 update. The completed "
             "matched comparison is reported separately in section 3.10 and remains subject to seed and environment "
             "caveats. Strong-KL GRPO (KL 0.01 / 0.04) is significantly better than DAPO w/o DS on MATH and worse on "
             "GSM8K.")
    F.append("- **Controlled dynamic-sampling test (ablation rung 2 vs. DS, both 32 prompts and 1 update per step, "
             "step 300; section 3.10):** without DS the model collapses into short answers on Countdown within 50 steps "
             "(mean response 54 tokens vs. 2,577 with DS; Countdown validation 25% vs. 65%). The old no-DS run avoided "
             "this only by training on 6x more prompts per step. On the benchmarks, DS makes little difference under "
             "standard prompts (GSM8K +4.6 strict but +1.3 flexible, n.s.; MATH +1.0, p = 0.10; BBH and MMLU-Pro 0) and a "
             "large one under the Countdown-style think prompt (GSM8K 74.4 vs. 5.9, MATH 46.5 vs. 14.5), where rung 2 "
             "writes a minimal think block (median about 100 characters). **Seed caveat:** a second training seed of rung 2 "
             "(`rung2s2`) dipped early (155 tokens at step 25) but then recovered (about 1,000 tokens, Countdown 51% at "
             "step 300 and 62% at step 495, vs. DS 67% and 72%). **A second seed of DS itself (`DSs2`) also collapsed**, "
             "to about 20-token answers by step 25 (Countdown 26% at step 100, 35% at step 240); it ended at step 245 because "
             "DS needed up to 3 generation rounds per step and used up the 2 epochs of data. So the collapse is "
             "seed-dependent with and without DS (1 of 2 seeds each): DS does not reliably prevent it.")
    F.append("- **Clip-higher alone does not prevent the collapse (ablation rung 1 vs. rung 2, no DS, 32 prompts/step):** "
             "rung 1 (clip 0.2) collapses like rung 2 (clip 0.28): mean response 47 vs. 54 tokens, Countdown validation "
             "27% vs. 25% at step 300. Benchmark differences are small: under standard prompts GSM8K +5.4 strict for "
             "rung 2 but -0.8 flexible (format again), BBH +1.8, MATH and MMLU-Pro ~0; under the think prompt rung 2 is "
             "1-4 points higher. Clip-higher did not help; with one seed each, rung 1 and rung 2 seed 1 both collapsed, "
             "while rung 2 seed 2 recovered (see the seed caveat above).")
    F.append("- **Early collapse is the usual outcome, and no recipe prevents it (collapse screen, section 3.13).** "
             "Collapsed (mean response < 100 tokens at steps 90-100) in 16 of 22 runs: DS 5/6, A1 5/6, A1 + KL 0.01 3/4, "
             "no DS 3/6; differences between recipes are not significant at these sample sizes, and KL 0.01 does not fix "
             "it. Collapses happen within about 25 steps, are seed-dependent, do not need the format credit (A1 has "
             "none), and are tied to the Countdown prompt (collapsed models stay at base level under standard prompts). "
             "The healthy DS and A1 runs used for the single-run comparisons are therefore the exceptions; DS seed 1 is "
             "also the only DS run from the original environment (see 3.13).")
    F.append("- **Single-run comparison of options (A-D, A1 seed 1; section 3.11).** On their first seed every option "
             "avoided the early collapse; given the result above this may be luck. In-domain at step 495, masked "
             "groups (C, 70%) and batch-level normalisation (D, 69%) end within 2-3 points of DS (72%); "
             "correctness-only reward (A) reaches 69% at step 300 but collapses after step 400 as truncated responses "
             "rise to 59-63%; adding overlong filtering (A1) removes that collapse (69% at step 515). On the benchmarks "
             "at step 300 the options match DS within noise on MATH, BBH and MMLU-Pro under standard prompts; A, A1, B "
             "and C score significantly higher than DS on MATH with the think prompt (+2 to +6); GSM8K standard-prompt "
             "strict differences are mostly answer format. A1 seed 1 was the closest to DS overall (no significant loss on "
             "any benchmark and prompt, at about one-sixth of DS's generation per step), but A1 seed 2 collapsed.")
    F.append("- **pass@k (section 3.12): finite-sample coverage is consistent with probability redistribution.** "
             "On MATH-500, base and several trained checkpoints are similar at pass@64 (base 91.0, DS 92.2, options "
             "90.0-92.2). On GSM8K with flexible scoring, selected recipes gain about 3 points at pass@1 while reaching "
             "96-97% at pass@32. These finite-k results do not prove equal latent support. The PPO lr 1e-6 checkpoint has "
             "lower empirical coverage (GSM8K pass@32 88.2; MATH pass@64 85.6). Matching pass@k runs for the headline "
             "strong-KL GRPO step-250 checkpoint reach 97.1 on GSM8K pass@32 and 91.6 on MATH-500 pass@64.")
    F.append("- With the standard prompt, Countdown RL barely changes MATH / BBH / MMLU-Pro relative to the base "
             "model; the large GSM8K gain is mostly answer-format compliance (compare flexible accuracy).")
    F.append("- The think prompt yields shorter reasoning and lower MATH / MMLU-Pro accuracy for every model, "
             "including the untrained base, so the gap is caused by the prompt, not truncation.")
    F.append("- PPO: the actor learning rate dominates transfer. At 1e-6 (no KL, no entropy) PPO stays near the "
             "base model and improves GSM8K; every 3e-6 run falls far below the base model on all benchmarks "
             "regardless of KL loss (off / coef 0 / 0.001) or entropy (0 / 0.001). The worst run "
             "(3e-6, KL off, entropy 0.001) collapsed into Countdown-style output. KL 0.001 limits but does not "
             "prevent the damage. 'KL on, coef 0' and 'KL off' should be equivalent but differ by up to 20 points, "
             "a sign of large seed-to-seed variance at 3e-6 (one seed per setting).")
    F.append("- GRPO (lr 1e-6, mini-batch 64): the KL-loss coefficient decides whether reasoning survives in the "
             "Countdown (think) format. With KL off or weak (0, 1e-6, 0.001, 0.005), every run -- whatever n (2/4/8) "
             "or reward -- learned to answer the think prompt with only an `<answer>` tag and no reasoning "
             "(median ~20 characters), so think-prompt accuracy collapses (GSM8K 8-19, MATH 15-18) while "
             "standard-prompt accuracy stays near the base model. With KL 0.01 (step 250) or 0.04 the models keep "
             "reasoning (median ~780 characters) and reach averages of 47.0 / 47.2, close to DAPO w/o DS (48.6).")
    F.append("- The best GRPO runs are significantly better than DAPO w/o DS on MATH (think +2.7 / +3.0; KL 0.01 "
             "also standard +1.2) and significantly worse on GSM8K (standard -4.8 / -13.4, think -6.2 / -2.8); "
             "BBH and MMLU-Pro are mostly within noise. The KL 0.01 run is at step 250 (it stopped early), "
             "so it is not fully comparable with the step-520 runs.")
    F.append("- The GRPO KL result holds at a matched step (section 3.7): at step 250, all three runs with KL <= 0.005 "
             "already answer the think prompt with a bare `<answer>` tag (median 20-22 characters; GSM8K 10-11, MATH "
             "15-16), while KL 0.01 and 0.04 keep reasoning (median 730-1,900 characters; averages 47.0 and 46.3). "
             "Across both steps, all seven runs with KL <= 0.005 collapse and both runs with KL >= 0.01 do not.")
    F.append("- Longer in-domain training slightly reduces transfer (section 3.8). Over DAPO w/o DS checkpoints 100 -> 520, "
             "Countdown validation accuracy rises from 68.8 to 78.8, while MATH falls 55.9 -> 53.0 (standard, p < 0.001) "
             "and 51.8 -> 48.3 (think, p < 0.001); GSM8K changes little (standard 69.9 -> 67.6, p = 0.09; think "
             "unchanged). Which checkpoint is evaluated therefore shifts results by about 3 points on MATH.")
    F.append("- For weak-KL GRPO, standard-prompt GSM8K rises with n (n=2: 45.3, n=4: 50.0, n=8: 63.5 at KL 0.001) "
             "and with the KL coefficient (n=2: 38.4 no KL, 45.3, 56.1, 65.1 at 0.04). The equal 0.5/0.5 reward "
             "(n=4, KL 1e-6) gives 66.4 vs 51.1 with the standard reward, but still collapses on the think prompt.")
    F.append("- Countdown validation reward does not predict transfer (e.g. PPO lr 3e-6 with KL 0.001 had one of the "
             "highest Countdown rewards but lost 43 points on MATH).")
    F.append("- Caveats: one training seed per setting; DAPO evaluated at step 300 (the DS run saved only that "
             "checkpoint), PPO and GRPO runs at their final step except GRPO KL 0.01 (stopped at step 250); "
             "zero-shot BBH / MMLU-Pro; McNemar p-values compare checkpoints and include no seed variance.")


    S = ["## 0. Summary\n",
         "All models ranked by average strict accuracy (%) over the 4 benchmarks x 2 prompts, 8192 tokens, greedy. "
         "Labels are built from each run's recorded settings (section 2).\n"]
    lab = pd.Series({m: label(m, reg) for m in pv.index})
    sm = pv.copy()
    sm.insert(0, "model (settings, step)", lab)
    sm = sm.sort_values("avg (8)", ascending=False).reset_index()
    sm.insert(0, "rank", range(1, len(sm) + 1))
    sm = sm.rename(columns={"model": "key"})
    S.append(md_table(sm))
    S.append("\n**What can be claimed about transfer** (standard prompts, flexible scoring so that answer formatting "
             "does not count; section 3.9)\n")
    S.append(md_table(pd.DataFrame([
        {"claim": "Best recipes slightly improve GSM8K",
         "evidence": "DAPO w/o DS +3.1 (p = 0.008); GRPO KL 0.04 +2.6 (p = 0.008)",
         "strength": "small but significant (a real improvement); one training seed each"},
        {"claim": "No measurable change on MATH, BBH, MMLU-Pro",
         "evidence": "best recipes within +/-1 point (only GRPO KL 0.04 on MMLU-Pro +0.8 is significant); no gain on "
                     "the easiest MATH levels (base 75-86%)",
         "strength": "consistent across models and difficulty levels"},
        {"claim": "Some recipes hurt general ability",
         "evidence": "PPO lr 3e-6: -13 to -53 on MATH/BBH/MMLU-Pro; PPO lr 1e-6: -9 on MATH; weak-KL GRPO: -2 to -3 on MATH",
         "strength": "large and clear"},
    ])))
    S.append("")
    S.append("Also supported, but not general reasoning: (1) large in-domain gains, Countdown validation accuracy 3% -> "
             "about 75-79% (DAPO); (2) answer-format compliance, GSM8K responses without a final-answer marker 54% -> "
             "about 8%; this is formatting, not reasoning; (3) the recipe decides whether general ability is kept "
             "(DAPO, GRPO KL >= 0.01) or damaged (PPO at high lr, GRPO KL <= 0.005); (4) the learned behaviour "
             "differs by recipe: the think prompt exposes a 20-token no-reasoning shortcut in weak-KL GRPO, linked in "
             "the training logs to early entropy collapse, while DAPO keeps reasoning in its training format.\n")
    S.append("Suggested wording: *RL on Countdown produces large in-domain gains and a small but statistically "
             "significant improvement on GSM8K: about 3 points for the best recipes (DAPO w/o DS +3.1, GRPO KL 0.04 "
             "+2.6, both p < 0.01), after accounting for answer-format compliance. MATH, BBH and MMLU-Pro are unchanged, "
             "and other recipes (high-learning-rate PPO, GRPO with KL <= 0.005) degrade general ability or learn a "
             "no-reasoning shortcut.*\n")
    S.append("Limits: the full downstream evaluations generally use one training seed per setting (McNemar p-values "
             "do not include seed variance, so even the +3 GSM8K gain may change with more seeds); only "
             "Qwen2.5-1.5B-Instruct and only Countdown are studied, so 'little transfer' should not be generalised "
             "beyond this setup. The matched DS comparison is complete but does not remove all environment differences.\n")
    S.append("**Open items**\n")
    S.append("- The matched DS/no-DS comparison is complete (`../ablation/README.md`); repeat it in one frozen software "
             "environment before making a causal claim about dynamic sampling.")
    S.append("- Full-length downstream replicate training remains incomplete for DS, matched no-DS, strong-KL GRPO, "
             "and PPO; the 22-run screen supports only claims about early collapse.")
    S.append("- **Follow-up paper (not this one):** train the same recipes on a math training set (e.g. the MATH or GSM8K "
             "training split, or a curated RL math set such as DAPO-Math-17k or DeepScaleR) with the same model, "
             "steps and evaluation pipeline, and compare transfer with the Countdown-trained models. Tests whether the "
             "small transfer seen here is due to the Countdown data rather than the RL recipes. Needs only a new data "
             "path and reward function in the training scripts; run after rung 2 and the seed runs.")
    S.append("- Paper: replace the dynamic-sampling filtered fraction f = 1 - b_train / (n_gen b_gen) with the measured "
             "per-step value (`../ds_run_p0_p1_per_step.csv`, column `f_true`); with 192 generated and 32 trained, "
             "the formula gives 0.83 at almost every step.")
    S.append("\n**Key findings**\n")
    S += F
    S.append("")
    L = L[:2] + S + L[2:]
    out = os.path.join(HERE, "RESULTS.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out, "|", len(models8), "models at 8192 tokens,", len(models2), "at 2048")


if __name__ == "__main__":
    main()
