"""LaTeX table of downstream accuracy (8192-token run) -> ../paper_figs/table2.tex

Rows: base model, DAPO w/ DS, DAPO w/o DS (step 300), and the paired difference
(w/o DS - w/ DS) with an exact McNemar test. Columns: benchmark x prompt, strict accuracy (%).
  python make_table2.py                      # uses results_8k/
  EVAL_RESULTS=results python make_table2.py # the 2048-token run instead
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import mcnemar_p  # noqa: E402

RES = os.environ.get("EVAL_RESULTS", "results_8k")
OUT = os.path.join(os.path.dirname(HERE), "paper_figs", "table2.tex" if RES == "results_8k" else f"table2_{RES}.tex")
BENCHES = [("gsm8k", "GSM8K", 1319), ("math", "MATH", 5000), ("bbh", "BBH", 6511), ("mmlupro", "MMLU-Pro", 12032)]
STYLES = [("standard", "Std."), ("think", "Think")]
ROWS = [("base", "Qwen2.5-1.5B-Instruct"), ("DS_step300", "DAPO w/ DS"), ("noDS_step300", "DAPO w/o DS"),
        # PPO labels: actor lr / KL-loss coef (-- = off) / entropy coef / step; see runs_registry.md
        ("PPO_step400", r"PPO $1{\times}10^{-6}$ / -- / 0 / 400$^\dagger$"),
        ("PPO_klF_ent0_lr1e-6_step520", r"PPO $1{\times}10^{-6}$ / -- / 0 / 520"),
        ("PPO_klF_ent0_lr3e-6_step520", r"PPO $3{\times}10^{-6}$ / -- / 0 / 520"),
        ("PPO_klF_ent1e-3_lr3e-6_step520", r"PPO $3{\times}10^{-6}$ / -- / 0.001 / 520"),
        ("PPO_kl0_ent1e-3_lr3e-6_step520", r"PPO $3{\times}10^{-6}$ / 0$^\ddagger$ / 0.001 / 520"),
        ("PPO_kl1e-3_ent1e-3_lr3e-6_step520", r"PPO $3{\times}10^{-6}$ / 0.001 / 0.001 / 520")]
# models without complete results (every benchmark and prompt, scored) are left out of the table
def _complete(m):
    for b in ["gsm8k", "math", "bbh", "mmlupro"]:
        for s in ["standard", "think"]:
            f = f"{m}__{s}.jsonl" if b == "gsm8k" else os.path.join("scored", f"{m}__{s}.jsonl")
            if not os.path.exists(os.path.join(HERE, RES, b, f)):
                return False
    return True


ROWS = [(m, l) for m, l in ROWS if _complete(m)]
MAX_TOKENS = {"results_8k": 8192, "results": 2048}.get(RES, "?")


def load(bench, model, style):
    d = os.path.join(HERE, RES, bench)
    f = os.path.join(d, f"{model}__{style}.jsonl") if bench == "gsm8k" else os.path.join(d, "scored", f"{model}__{style}.jsonl")
    return pd.read_json(f, lines=True, dtype=False).correct_strict.values.astype(bool)


acc, delta = {}, {}
for b, _, _ in BENCHES:
    for s, _ in STYLES:
        for m, _ in ROWS:
            acc[(b, s, m)] = load(b, m, s)
        a, c = acc[(b, s, "noDS_step300")], acc[(b, s, "DS_step300")]
        _, _, p = mcnemar_p(a, c)
        delta[(b, s)] = (100 * (a.mean() - c.mean()), p)


def stars(p):
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


cols = [(b, s) for b, _, _ in BENCHES for s, _ in STYLES]
lines = [
    r"\begin{table*}[t]", r"\centering", r"\small",
    r"\caption{Downstream accuracy (\%) of the step-300 checkpoints of DAPO with and without dynamic "
    r"sampling (DS), trained only on Countdown, and of the untrained base model"
    + (r"; PPO rows are GAE baselines trained on the same Countdown data with one response per prompt and "
       r"a critic learning rate of $10^{-5}$, labelled actor lr / KL-loss coefficient (-- = KL loss off) / "
       r"entropy coefficient / step. All use PPO mini-batch 16 except $^\dagger$ (mini-batch 64, separate "
       r"project); $^\ddagger$ KL loss enabled with coefficient 0. Full settings in runs\_registry.md"
       if any(m.startswith("PPO") for m, _ in ROWS) else "") + r". Greedy decoding, "
    rf"up to {MAX_TOKENS} output tokens, zero-shot. \emph{{Std.}}: the benchmark's usual chain-of-thought "
    r"prompt; \emph{Think}: the Countdown training prompt with \texttt{<think>}/\texttt{<answer>} tags. "
    r"An answer counts only if it appears in an explicit final-answer marker (\texttt{\#\#\#\#}, "
    r"\texttt{\textbackslash boxed\{\}}, \texttt{<answer>}, or ``the answer is''). "
    r"$\Delta$ = w/o DS $-$ w/ DS on the same problems; exact McNemar test, "
    r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. Bold: the better of the two DAPO variants when $p<0.05$.}",
    r"\label{tab:downstream}",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{l" + "cc" * len(BENCHES) + "}",
    r"\toprule",
    " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name} ($n$={n:,})}}" for _, name, n in BENCHES) + r" \\",
    "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(BENCHES))),
    "Model & " + " & ".join(lab for _ in BENCHES for _, lab in STYLES) + r" \\",
    r"\midrule",
]
for m, label in ROWS:
    cells = []
    for b, s in cols:
        v = f"{100 * acc[(b, s, m)].mean():.1f}"
        d, p = delta[(b, s)]
        if p < 0.05 and ((m == "noDS_step300" and d > 0) or (m == "DS_step300" and d < 0)):
            v = rf"\textbf{{{v}}}"
        cells.append(v)
    lines.append(label + " & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
lines.append(r"$\Delta$ (w/o DS $-$ w/ DS) & " + " & ".join(
    rf"${delta[c][0]:+.1f}{stars(delta[c][1])}$" for c in cols) + r" \\")
lines.append(r"McNemar $p$ & " + " & ".join(
    ("$<$0.001" if delta[c][1] < 0.001 else f"{delta[c][1]:.3f}") for c in cols) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
tex = "\n".join(lines)
open(OUT, "w").write(tex)
print(tex)
print("wrote", OUT)
