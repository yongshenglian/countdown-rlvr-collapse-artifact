"""Why does dynamic sampling (DS) prevent the short-answer collapse? Evidence from the printed reward logs.

Reward (Sep-2025): correct 0.9 + 0.1 format; wrong/invalid = 0.1 format credit (or 0); no equation = 0.
GRPO advantage per response = (score - group mean) / (group std + eps), with std normalisation on.
DS filters groups on *accuracy*, so it drops groups whose 8 answers are all right or all wrong, even when their
*scores* differ (e.g. all wrong, but some earned the 0.1 format credit). Without DS those "format-only" groups
stay in, and std normalisation blows the 0.1 differences up to full-size advantages.

For each training step (first 256 printed responses = 32 groups x 8; the rest is validation) this reports:
  groups by type: mixed-correctness / format-only (acc all 0 but scores differ) / zero-signal (scores all equal)
  length signal: mean advantage-weighted response length change, i.e. corr(advantage, length) within each group type
Positive advantage on shorter responses = the update pushes toward shorter answers.

  python analysis_why_ds.py <ray job log> [max_step]      e.g. logs/rung2_job.log 60
"""
import re
import sys

import numpy as np
import pandas as pd

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PFX = re.compile(r"^\(\w+ pid=\d+\)\s?")
N, B = 8, 32


def parse(path, max_step):
    steps, cur, resp, state, sol = [], [], None, None, []
    with open(path, errors="replace") as fh:
        for raw in fh:
            ln = PFX.sub("", ANSI.sub("", raw.rstrip("\n")))
            m = re.match(r"^step:(\d+) - ", ln)
            if m:
                steps.append((int(m.group(1)), cur)); cur = []
                if int(m.group(1)) >= max_step:
                    break
                continue
            if re.match(r"^\s*Target: \d+ \| Numbers: \[", ln):
                resp = {"prompt": ln.strip(), "fmt": None, "out": None, "len": 0}; cur.append(resp); state = "sol"; sol = []
                continue
            if resp is None:
                continue
            if state == "sol":
                m = re.match(r"^Format score: ([\d.]+)\s*$", ln)
                if m:
                    resp["fmt"] = float(m.group(1)); resp["len"] = len("\n".join(sol)); state = "post"
                else:
                    sol.append(ln.replace("Solution string: ", "", 1))
            elif state == "post":
                for pre, tag in [("Correct equation", "correct"), ("Wrong result", "wrong"), ("Invalid equation", "invalid"),
                                 ("equation does not exist", "noeq"), ("Could not evaluate", "noeval"), ("Error evaluating", "noeval")]:
                    if ln.startswith(pre):
                        resp["out"] = tag; state = None; break
    return steps


def analyse(steps):
    rows = []
    for st, rs in steps:
        tr = rs[:B * N]
        if len(tr) < B * N or not all(r["out"] for r in tr):
            continue
        acc = np.array([r["out"] == "correct" for r in tr], float).reshape(B, N)
        fmt = np.array([r["fmt"] for r in tr], float).reshape(B, N)
        noeq = np.array([r["out"] == "noeq" for r in tr], float).reshape(B, N)
        score = np.where(acc == 1, 0.9 + fmt, np.where(noeq == 1, 0.0, fmt))
        length = np.array([r["len"] for r in tr], float).reshape(B, N)
        std = score.std(1, ddof=1)
        adv = (score - score.mean(1, keepdims=True)) / (std[:, None] + 1e-6)
        k = acc.sum(1)
        mixed = (k > 0) & (k < N)
        fmt_only = (~mixed) & (std > 1e-9)
        row = dict(step=st, mixed_groups=int(mixed.sum()), format_only_groups=int(fmt_only.sum()),
                   zero_signal_groups=int((std <= 1e-9).sum()), mean_len=length.mean(), accuracy=acc.mean())
        for name, sel in [("mixed", mixed), ("format_only", fmt_only)]:
            if sel.any():
                a, l = adv[sel].ravel(), length[sel].ravel()
                row[f"corr_adv_len_{name}"] = np.corrcoef(a, l)[0, 1] if l.std() > 0 else np.nan
                # share of the total |advantage| mass (gradient weight) coming from this group type
                row[f"adv_mass_{name}"] = np.abs(adv[sel]).sum()
        rows.append(row)
    d = pd.DataFrame(rows).fillna(0)
    tot = d.get("adv_mass_mixed", 0) + d.get("adv_mass_format_only", 0)
    d["share_of_signal_from_format_only_%"] = 100 * d.get("adv_mass_format_only", 0) / tot.replace(0, np.nan)
    return d


if __name__ == "__main__":
    path = sys.argv[1]
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    d = analyse(parse(path, mx))
    cols = ["step", "accuracy", "mean_len", "mixed_groups", "format_only_groups", "zero_signal_groups",
            "share_of_signal_from_format_only_%", "corr_adv_len_mixed", "corr_adv_len_format_only"]
    print(d[cols].round(3).to_string(index=False))
    for a, b in [(1, 10), (11, 25), (26, 60)]:
        s = d[(d.step >= a) & (d.step <= b)]
        if len(s):
            print(f"steps {a}-{b}: mixed {s.mixed_groups.mean():.1f}, format-only {s.format_only_groups.mean():.1f}, "
                  f"zero-signal {s.zero_signal_groups.mean():.1f} groups of 32 | signal from format-only "
                  f"{s['share_of_signal_from_format_only_%'].mean():.0f}% | corr(adv, length): mixed "
                  f"{s.corr_adv_len_mixed.mean():+.2f}, format-only {s.corr_adv_len_format_only.mean():+.2f} | "
                  f"mean length {s.mean_len.mean():.0f} chars")
