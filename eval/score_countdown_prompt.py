"""Score the Countdown prompt test (handoff/countdown_prompt) and compare the two prompts per checkpoint.

  python eval/score_countdown_prompt.py <results dir>      (the bundle's results/, copied back)

Correctness (both prompts, one rule set): the candidate is whichever occurs last of the last <answer>...</answer>,
the last \\boxed{...}, and the last "Answer:" (anywhere on a line, e.g. inside \\[ \\]) (models box equations under the standard prompt
although it does not ask for a box). Markdown/LaTeX wrappers are removed, x/÷ become * and /, and anything from the first
"=" on is dropped. It is then checked with the training reward's own validate_equation and evaluate_equation
(every given number used exactly once, value within 1e-5 of the target).
train_reward_acc is the unchanged training reward (compute_score_sep2025 "acc") for reference.
Writes <results dir>/scored.csv (per model and prompt) and scored_items.jsonl.gz (per item).
"""
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("countdown", "/home/lian/verl/verl/utils/reward_score/countdown.py")
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)
TASKS = ["countdown__train_wrapper", "countdown__standard"]
PROMPTS = Path(__file__).resolve().parents[1] / "handoff" / "countdown_prompt" / "prompts"


def last_boxed(text):
    """(start, content) of the last \\boxed{...}, with nested braces; None if absent or unclosed."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed")
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return i, text[j + 1:k]
    return None


def candidate(text):
    cands = []
    m = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if m:
        cands.append((m[-1].start(), m[-1].group(1)))
    b = last_boxed(text)
    if b:
        cands.append(b)
    m = list(re.finditer(r"\bAnswer[ \t]*[*_`]*[ \t]*:(.*)$", text, re.I | re.M))
    if m:
        cands.append((m[-1].start(), m[-1].group(1)))
    if not cands:
        return None
    s = max(cands)[1]
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\[()\[\]]|\$|`|\*\*|\\text\{[^}]*\}", " ", s)
    s = s.split("=")[0].strip().rstrip(".").strip()
    return s or None


def correct(expr, nums, target):
    if expr is None or not cd.validate_equation(expr, nums):
        return False
    v = cd.evaluate_equation(expr)
    return v is not None and abs(v - target) < 1e-5


def score_one(args):
    res, t, m = args
    items = [json.loads(l) for l in open(PROMPTS / f"{t}.jsonl")]
    gen = [json.loads(l) for l in open(res / t / f"{m}.jsonl")]
    assert [g["idx"] for g in gen] == [it["idx"] for it in items]
    acc, tr, has, trunc, ntok, lines = [], [], [], [], [], []
    for g, it in zip(gen, items):
        e = candidate(g["response"])
        c = correct(e, it["nums"], it["target"])
        r = cd.compute_score_sep2025(g["response"], dict(numbers=it["nums"], target=it["target"]), do_print=False)
        acc.append(c); tr.append(bool(r["acc"])); has.append(e is not None)
        trunc.append(g["finish"] == "length"); ntok.append(g["response_tokens"])
        lines.append(json.dumps(dict(task=t, model=m, idx=g["idx"], candidate=e, correct=c, train_reward_acc=bool(r["acc"]))) + "\n")
    row = dict(task=t, model=m, n=len(gen), acc=100 * np.mean(acc), train_reward_acc=100 * np.mean(tr),
               answer_found=100 * np.mean(has), truncated=100 * np.mean(trunc), median_tokens=float(np.median(ntok)))
    return row, lines


def main(res):
    import csv
    from multiprocessing import Pool
    models = sorted({p.stem for p in (res / TASKS[0]).glob("*.jsonl")})
    jobs = [(res, t, m) for t in TASKS for m in models]
    with Pool(16) as pool:
        results = pool.map(score_one, jobs)
    rows = [r for r, _ in results]
    with gzip.open(res / "scored_items.jsonl.gz", "wt") as out:
        for _, lines in results:
            out.writelines(lines)
    with open(res / "scored.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['task']:26s} {r['model']:22s} acc {r['acc']:5.1f}  train_reward {r['train_reward_acc']:5.1f}  "
              f"answer {r['answer_found']:5.1f}  trunc {r['truncated']:4.1f}  med_tok {r['median_tokens']:.0f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
