"""Side-by-side comparison of the 2048-token (results/) and 8192-token (results_8k/) runs.

For every benchmark x prompt x model: strict accuracy and truncation rate at both limits,
plus the paired DS-vs-noDS McNemar test at each limit. Writes compare_lengths.csv / .txt.
  python compare_lengths.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import mcnemar_p  # noqa: E402

RUNS = {"2k": "results", "8k": "results_8k"}
BENCHES = ["gsm8k", "math", "bbh", "mmlupro"]
MODELS = ["base", "DS_step300", "noDS_step300"]
STYLES = ["standard", "think"]


def load(run, bench, model, style):
    d = os.path.join(HERE, RUNS[run], bench)
    f = os.path.join(d, f"{model}__{style}.jsonl") if bench == "gsm8k" else os.path.join(d, "scored", f"{model}__{style}.jsonl")
    return pd.read_json(f, lines=True, dtype=False)


rows, tests = [], []
for b in BENCHES:
    for s in STYLES:
        frames = {}
        for run in RUNS:
            for m in MODELS:
                r = load(run, b, m, s)
                frames[(run, m)] = r
                rows.append(dict(bench=b, prompt=s, model=m, limit=run, acc=r.correct_strict.mean(),
                                 truncated=(r.finish == "length").mean(),
                                 median_chars=int(r.response.str.len().median())))
            a, c = frames[(run, "noDS_step300")], frames[(run, "DS_step300")]
            n10, n01, p = mcnemar_p(a.correct_strict.values.astype(bool), c.correct_strict.values.astype(bool))
            tests.append(dict(bench=b, prompt=s, limit=run, noDS_only=n10, DS_only=n01, p=p,
                              diff=a.correct_strict.mean() - c.correct_strict.mean()))

df = pd.DataFrame(rows)
wide = df.pivot_table(index=["bench", "prompt", "model"], columns="limit", values=["acc", "truncated"]).round(3)
wide.columns = [f"{v}_{l}" for v, l in wide.columns]
wide = wide.reindex(pd.MultiIndex.from_product([BENCHES, STYLES, MODELS])).reset_index()
wide.columns = ["bench", "prompt", "model"] + list(wide.columns[3:])
wide["acc_change"] = (wide.acc_8k - wide.acc_2k).round(3)
t = pd.DataFrame(tests)
out = os.path.join(HERE, "compare_lengths")
wide.to_csv(out + ".csv", index=False)
t.round(4).to_csv(out + "_mcnemar.csv", index=False)
txt = wide.to_string(index=False) + "\n\nDS vs noDS (strict; diff = noDS - DS)\n" + \
    t.pivot_table(index=["bench", "prompt"], columns="limit", values=["diff", "p"]).round(4).to_string()
open(out + ".txt", "w").write(txt + "\n")
print(txt)
