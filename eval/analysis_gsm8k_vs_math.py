"""Why does Countdown RL raise GSM8K a lot but MATH hardly at all (standard prompt)?

1. Decompose each trained model's GSM8K / MATH change over the base model, item by item:
     tolerant fix  : base flexible-wrong, trained model flexible-correct
     tolerant loss : base flexible-correct, trained model flexible-wrong
     marker win    : both flexible-correct, base strict-wrong, trained model strict-correct
     marker loss   : both flexible-correct, base strict-correct, trained model strict-wrong
2. Arithmetic accuracy: every explicit calculation "a (+-*/) b [...] = c" written in a response is recomputed.
   Reports the share of wrong calculations, and the share of wrong answers containing at least one.
3. Where base-model errors come from: share of base errors that contain an arithmetic slip.

  python analysis_gsm8k_vs_math.py      (reads results_8k/, writes analysis_gsm8k_vs_math.txt)
"""
import ast
import operator as op
import math
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import mcnemar_p  # noqa: E402

MODELS = ["base", "DS_step300", "noDS_step300", "GRPO_n2_kl4e-2_step250", "GRPO_n2_kl1e-2_step250",
          "GRPO_n8_kl1e-3_step520", "PPO_klF_ent0_lr1e-6_step520"]
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}
NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
# right-hand side of "=": a plain number; left-hand side: the arithmetic run just before "=" (bounded, linear time)
RHS = re.compile(rf"=\s*\$?(-?{NUM})(?![\d.]*\s*[-+*/^])")
LHS_CHARS = set("0123456789.,$()+-*/ ")


def paired_diff_ci(trained, base, z=1.96):
    """Asymptotic 95% CI for a paired accuracy difference, in percentage points."""
    fixes = int((~base & trained).sum())
    losses = int((base & ~trained).sum())
    n = len(base)
    diff = (fixes - losses) / n
    variance = ((fixes + losses) / n - diff ** 2) / n
    half_width = z * math.sqrt(max(0.0, variance))
    return 100 * (diff - half_width), 100 * (diff + half_width)


def load(bench, m, s="standard"):
    f = os.path.join(HERE, "results_8k", bench, f"{m}__{s}.jsonl" if bench == "gsm8k" else f"scored/{m}__{s}.jsonl")
    return pd.read_json(f, lines=True, dtype=False)


def ev(expr):
    def f(n):
        if isinstance(n, ast.Expression):
            return f(n.body)
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.BinOp):
            return OPS[type(n.op)](f(n.left), f(n.right))
        if isinstance(n, ast.UnaryOp):
            return OPS[type(n.op)](f(n.operand))
        raise ValueError
    return f(ast.parse(expr, mode="eval"))


def calcs(text):
    """(n_calculations, n_wrong) over explicit numeric calculations in the text (LaTeX \\times, \\div normalised)."""
    t = text.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("×", "*").replace("÷", "/")
    t = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", t)
    n = wrong = 0
    for m in RHS.finditer(t):
        i = m.start() - 1
        while i >= 0 and m.start() - i < 80 and t[i] in LHS_CHARS:
            i -= 1
        lhs = t[i + 1:m.start()].strip().replace(",", "").replace("$", "").strip("+-*/ ")
        rhs = m.group(1).replace(",", "")
        if not re.search(r"\d\s*[-+*/]\s*\(?\s*\d", lhs):
            continue  # not a calculation (e.g. "x = 5")
        if lhs.count("(") != lhs.count(")"):
            lhs = lhs.replace("(", "").replace(")", "")
        try:
            v, r = ev(lhs), float(rhs)
        except Exception:
            continue
        n += 1
        tol = max(1e-6, 0.51 * 10 ** -len(rhs.split(".")[1])) if "." in rhs else 1e-6
        if abs(v - r) > tol * max(1, abs(r)) and abs(v - r) > tol:
            wrong += 1
    return n, wrong


def main():
    out = []
    P = lambda *a: out.append(" ".join(str(x) for x in a))

    P("== 1. Tolerant-correctness and marker-compliance transitions (standard prompt, % of items) ==")
    rows = []
    for bench in ["gsm8k", "math"]:
        b = load(bench, "base")
        bs, bf = b.correct_strict.astype(bool), b.correct_flexible.astype(bool)
        for m in MODELS[1:]:
            t = load(bench, m)
            ts, tf = t.correct_strict.astype(bool), t.correct_flexible.astype(bool)
            tolerant_fixes = (~bf & tf).mean() * 100
            tolerant_losses = (bf & ~tf).mean() * 100
            marker_wins = (bf & tf & ~bs & ts).mean() * 100
            marker_losses = (bf & tf & bs & ~ts).mean() * 100
            ci_low, ci_high = paired_diff_ci(tf, bf)
            rows.append(dict(bench=bench, model=m, base_strict=bs.mean() * 100, base_no_marker=b.pred_strict.isna().mean() * 100,
                             model_strict=ts.mean() * 100, total_gain=(ts.mean() - bs.mean()) * 100,
                             tolerant_fixes=tolerant_fixes, tolerant_losses=tolerant_losses,
                             net_tolerant=tolerant_fixes - tolerant_losses,
                             net_ci=f"[{ci_low:.2f}, {ci_high:.2f}]",
                             marker_wins=marker_wins, marker_losses=marker_losses,
                             p_flexible=mcnemar_p(tf.values, bf.values)[2]))
    P(pd.DataFrame(rows).round(2).to_string(index=False))

    P("\n== 2. Arithmetic accuracy: explicit calculations recomputed (standard prompt) ==")
    rows = []
    for bench in ["gsm8k", "math"]:
        for m in MODELS:
            t = load(bench, m)
            c = t.response.map(calcs)
            n = sum(x[0] for x in c); w = sum(x[1] for x in c)
            has_err = c.map(lambda x: x[1] > 0)
            wrong = ~t.correct_strict.astype(bool)
            rows.append(dict(bench=bench, model=m, calcs_per_response=n / len(t), calc_error_rate_pct=100 * w / max(n, 1),
                             responses_with_error_pct=100 * has_err.mean(),
                             wrong_answers_with_error_pct=100 * has_err[wrong].mean(),
                             correct_answers_with_error_pct=100 * has_err[~wrong].mean()))
    P(pd.DataFrame(rows).round(2).to_string(index=False))

    P("\n== 3. Base-model errors that contain an arithmetic slip, and whether trained models fix them ==")
    for bench in ["gsm8k", "math"]:
        b = load(bench, "base")
        err = b.response.map(lambda x: calcs(x)[1] > 0)
        bw = ~b.correct_flexible.astype(bool)
        P(f"{bench}: base wrong (flexible) {100 * bw.mean():.1f}% of items; of those, {100 * err[bw].mean():.1f}% contain "
          f"an arithmetic slip")
        for m in MODELS[1:]:
            t = load(bench, m).correct_flexible.astype(bool)
            P(f"   {m:30s} fixes {100 * t[bw & err].mean():5.1f}% of base errors WITH a slip, "
              f"{100 * t[bw & ~err].mean():5.1f}% of base errors WITHOUT a slip")
    if True:
        P("\n== 4. MATH by subject: gain over base (standard, strict, points) ==")
        b = load("math", "base")
        for m in ["noDS_step300", "GRPO_n2_kl4e-2_step250"]:
            t = load("math", m)
            g = (t.correct_strict.astype(bool).groupby(b.subject).mean() - b.correct_strict.astype(bool).groupby(b.subject).mean()) * 100
            P(f"   {m}: " + ", ".join(f"{k} {v:+.1f}" for k, v in g.round(1).items()))
    txt = "\n".join(out)
    open(os.path.join(HERE, "analysis_gsm8k_vs_math.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
