"""Recover per-step p0 / p1 (all-incorrect / all-correct group fractions) from the
DS run's console log (verl countdown reward fn prints every scored response).

Per scored response the reward fn prints, in order:
  Target: .. | Numbers: ..   -> Solution string: <model text> -> Format score: x
  -> (equation does not exist | Extracted equation: ..) -> outcome line
The outcome is the first reward-fn line after "Format score:", so model text
(which can contain look-alike strings) is skipped.
"""
import re, sys
import numpy as np, pandas as pd

LOG = sys.argv[1] if len(sys.argv) > 1 else "ds_log/output.log"
HIST = sys.argv[2] if len(sys.argv) > 2 else "dapo_DS_f8h77wol_history.csv"
N, B_TRAIN, B_GEN = 8, 32, 192

step_re = re.compile(r"^(?:\(\w+ pid=\d+\) )?step:(\d+) - ")
tgt_re = re.compile(r"^(?:\(\w+ pid=\d+\) )?\s*Target: (\d+) \| Numbers: (\[.*\])\s*$")
fmt_re = re.compile(r"^(?:\(\w+ pid=\d+\) )?Format score: ([\d.]+)\s*$")
OUT = [("Correct equation: ", "correct"), ("Wrong result: equation = ", "wrong"),
       ("Invalid equation", "invalid"), ("equation does not exist", "noeq"),
       ("Could not evaluate equation", "noeval")]

segments = []          # one list of responses per logged step
cur, resp, state = [], None, None
with open(LOG, errors="replace") as fh:
    for line in fh:
        line = line.rstrip("\n")
        m = step_re.match(line)
        if m:
            segments.append((int(m.group(1)), cur)); cur = []; continue
        m = tgt_re.match(line)
        if m:
            resp = {"prompt": m.group(1) + m.group(2), "fmt": None, "out": None}
            cur.append(resp); state = "sol"; continue
        if resp is None:
            continue
        if state == "sol":
            m = fmt_re.match(line)
            if m:
                resp["fmt"] = float(m.group(1)); state = "post"
        elif state == "post":
            s = re.sub(r"^\(\w+ pid=\d+\) ", "", line)
            for pre, tag in OUT:
                if s.startswith(pre):
                    resp["out"] = tag; state = None; break

hist = pd.read_csv(HIST)
hist = hist[hist["critic/score/mean"].notna()].set_index("_step")

rows = []
for step, resps in segments:
    if step == 0 or step not in hist.index:
        continue
    n_gen = int(hist.loc[step, "train/num_gen_batches"])
    n_train = n_gen * B_GEN * N
    tr = resps[:n_train]
    assert len(tr) == n_train and all(r["out"] for r in tr), (step, len(resps), n_train)
    acc = np.array([r["out"] == "correct" for r in tr], float).reshape(-1, N)
    fmt = np.array([r["fmt"] for r in tr], float).reshape(-1, N)
    noeq = np.array([r["out"] == "noeq" for r in tr], float).reshape(-1, N)
    sc = np.where(acc == 1, 0.9 + fmt, np.where(noeq == 1, 0.0, fmt))   # run-time reward: correct=0.9+fmt, wrong/invalid=fmt (0.1 or 0), no equation=0
    prompts = np.array([r["prompt"] for r in tr]).reshape(-1, N)
    assert (prompts == prompts[:, :1]).all(), f"group misalignment at step {step}"
    k = acc.sum(1)
    p0, p1 = (k == 0).mean(), (k == N).mean()
    kept = np.where((k > 0) & (k < N))[0][:B_TRAIN]          # DS keeps first b_train mixed groups
    rows.append(dict(step=step, n_gen=n_gen, n_groups=len(k), p0=p0, p1=p1, f_true=p0 + p1,
                     f_ngen=1 - B_TRAIN / (n_gen * B_GEN),
                     acc_all=acc.mean(), acc_kept=acc[kept].mean(),
                     acc_pred=(acc.mean() - p1) / (1 - p0 - p1),
                     score_all=sc.mean(), score_kept=sc[kept].mean(),
                     score_kept_scorefilter=sc[np.where(sc.std(1) > 0)[0][:B_TRAIN]].mean(),
                     logged_score_mean=hist.loc[step, "critic/score/mean"],
                     n_val=len(resps) - n_train))
out = pd.DataFrame(rows)
out.to_csv("ds_run_p0_p1_per_step.csv", index=False)
print(out.describe().T.to_string())
