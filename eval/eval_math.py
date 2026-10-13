"""MATH (Hendrycks) full test set, 5000 problems, 7 subjects; MATH-500 reported as a subset.

Prompts:
  standard : problem + "Please reason step by step, and put your final answer within \\boxed{}."
  think    : Countdown training wrapper with <think>/<answer> tags
Scoring (math-verify, symbolic equivalence against the \\boxed{} answer of the reference solution):
  strict   : content of the explicit final-answer marker that appears last (\\boxed{} or <answer>)
  flexible : strict, or math-verify extraction over a completed response

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_math.py --model noDS_step300
  python eval_math.py --summarize          # CPU only; needs .pydeps (math-verify) on the path
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODELS, THINK_TEMPLATE, run_generation, summarize_benchmark  # noqa: E402

BENCH = "math"
DATASET_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
STANDARD_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_items():
    import datasets
    items = []
    for s in SUBJECTS:
        for ex in datasets.load_dataset(
                "EleutherAI/hendrycks_math", s, split="test", revision=DATASET_REVISION):
            items.append(dict(problem=ex["problem"], solution=ex["solution"], level=ex["level"],
                              subject=s, gold=last_boxed(ex["solution"])))
    return items


def math500_idx(items):
    keys = {json.loads(l)["problem"].strip() for l in open(os.path.join(HERE, "data", "math500.jsonl"))}
    idx = {i for i, it in enumerate(items) if it["problem"].strip() in keys}
    return idx


def build_prompt(it, style):
    if style == "standard":
        return it["problem"] + STANDARD_SUFFIX
    return THINK_TEMPLATE.format(question=it["problem"], example="\\frac{3}{4}")


def last_boxed(text):
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if i < 0:
        return None
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def final_marker(text):
    """Content of whichever final-answer marker occurs last: \\boxed{} or <answer>."""
    cands = []
    i = text.rfind("\\boxed{")
    if i >= 0:
        cands.append((i, last_boxed(text)))
    a = list(re.finditer(r"<answer>(.*?)</answer>", text, re.S))
    if a:
        c = a[-1].group(1).strip()
        cands.append((a[-1].start(), last_boxed(c) or c))
    return max(cands)[1] if cands else None


_GOLD_CACHE = {}


def score(row, style):
    from math_verify import parse, verify
    if row["gold"] not in _GOLD_CACHE:
        _GOLD_CACHE[row["gold"]] = parse("$" + row["gold"] + "$")
    gold = _GOLD_CACHE[row["gold"]]
    m = final_marker(row["response"])
    ok_s = bool(m is not None and verify(gold, parse("$" + m + "$")))
    if ok_s:
        return m, True, m, True
    if row.get("finish") == "length":
        return m, False, None, False
    pf = parse(row["response"])
    ok_f = bool(pf and verify(gold, pf))
    return m, False, str(pf[-1]) if pf else None, ok_f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    items = load_items()
    if a.summarize:
        sys.path.insert(0, os.path.join(HERE, ".pydeps"))
        summarize_benchmark(BENCH, score, group_col="subject", subsets={"math500": math500_idx(items)})
    else:
        run_generation(BENCH, a.model, items, build_prompt, a.max_tokens)
