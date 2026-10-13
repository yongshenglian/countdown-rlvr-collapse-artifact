"""GSM8K test set (1319 problems), greedy decoding, two prompt styles.

  standard : question + 'Let's think step by step and output the final answer after "####".'
             (the verl GSM8K prompt)
  think    : the Countdown training wrapper with <think>/<answer> tags

Scoring (both reported):
  strict   : number in the explicit final-answer marker that appears last in the response:
             "#### N", "\\boxed{N}" or "<answer> N </answer>" (models use all three regardless of prompt)
  flexible : correct strict marker, or the last number in a completed response

Usage (one model per process/GPU):
  CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py --model noDS_step300
  python eval_gsm8k.py --summarize
"""
import argparse
import glob
import json
import os

import re

import pandas as pd

from common import (MODELS, RESULTS_DIR, THINK_TEMPLATE, last_answer_tag, last_number,
                    numbers_equal, to_chat, write_jsonl, generate)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_test.parquet")
OUT = os.path.join(RESULTS_DIR, "gsm8k")
STANDARD_SUFFIX = ' Let\'s think step by step and output the final answer after "####".'


def build_prompts(question, style):
    if style == "standard":
        return question + STANDARD_SUFFIX
    return THINK_TEMPLATE.format(question=question, example="42")


def strict_pred(text, style=None):
    """Number inside whichever explicit final-answer marker occurs last; None if no marker."""
    cands = []
    i = text.rfind("####")
    if i >= 0:
        cands.append((i, last_number(text[i + 4:].split("\n")[0]) or last_number(text[i + 4:])))
    b = [m for m in re.finditer(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)]
    if b:
        cands.append((b[-1].start(), last_number(b[-1].group(1))))
    a = [m for m in re.finditer(r"<answer>(.*?)</answer>", text, re.S)]
    if a:
        cands.append((a[-1].start(), last_number(a[-1].group(1))))
    return max(cands)[1] if cands else None


def score_response(text, gold, finish):
    """Return strict/flexible predictions and labels.

    Flexible correctness is a true relaxation of strict correctness. The
    marker-free last-number fallback is disabled for length-truncated outputs,
    where an incidental intermediate number is not a completed final answer.
    """
    sp = strict_pred(text)
    lp = last_number(text)
    ok_s = numbers_equal(sp, gold)
    ok_last = numbers_equal(lp, gold)
    ok_f = ok_s or (finish != "length" and ok_last)
    fp = sp if ok_s else lp
    return sp, ok_s, fp, ok_f


def run(model_key, max_tokens):
    df = pd.read_parquet(DATA)
    qs = [r["question"] for r in df["extra_info"]]
    gold = [r["ground_truth"] for r in df["reward_model"]]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    styles = ["standard", "think"]
    prompts = [to_chat(tok, build_prompts(q, s)) for s in styles for q in qs]
    gens, _ = generate(MODELS[model_key], prompts, max_tokens=max_tokens)
    n = len(qs)
    for i, s in enumerate(styles):
        rows = []
        for j in range(n):
            text, finish = gens[i * n + j]
            sp, ok_s, fp, ok_f = score_response(text, gold[j], finish)
            rows.append(dict(idx=j, question=qs[j], gold=gold[j], response=text, finish=finish,
                             pred_strict=sp, pred_flexible=fp,
                             correct_strict=ok_s, correct_flexible=ok_f))
        write_jsonl(os.path.join(OUT, f"{model_key}__{s}.jsonl"), rows)
        r = pd.DataFrame(rows)
        print(f"{model_key:14s} {s:9s} strict {r.correct_strict.mean():.4f}  flexible {r.correct_flexible.mean():.4f}  "
              f"truncated {(r.finish == 'length').mean():.3f}")


def rescore(r):
    scored = [score_response(t, g, f) for t, g, f in zip(r.response, r.gold, r.finish)]
    r["pred_strict"], r["correct_strict"], r["pred_flexible"], r["correct_flexible"] = map(list, zip(*scored))
    return r


def summarize():
    recs = []
    for f in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
        model, style = os.path.basename(f)[:-6].split("__")
        r = pd.read_json(f, lines=True, dtype={"gold": str})
        r = rescore(r)
        write_jsonl(f, r.to_dict("records"))
        n = len(r)
        for metric in ["strict", "flexible"]:
            p = r[f"correct_{metric}"].mean()
            recs.append(dict(model=model, prompt=style, metric=metric, n=n, acc=round(p, 4),
                             se=round((p * (1 - p) / n) ** 0.5, 4)))
        recs[-1]["truncated"] = recs[-2]["truncated"] = round((r.finish == "length").mean(), 4)
    s = pd.DataFrame(recs)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(s.pivot_table(index="model", columns=["prompt", "metric"], values="acc").to_string())

    # paired comparison DS vs noDS on the same problems (exact McNemar via binomial on discordant pairs)
    from math import comb
    for style in ["standard", "think"]:
        for metric in ["strict", "flexible"]:
            a = os.path.join(OUT, f"noDS_step300__{style}.jsonl"); b = os.path.join(OUT, f"DS_step300__{style}.jsonl")
            if not (os.path.exists(a) and os.path.exists(b)):
                continue
            ca = pd.read_json(a, lines=True)[f"correct_{metric}"].values
            cb = pd.read_json(b, lines=True)[f"correct_{metric}"].values
            n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
            k, m = min(n01, n10), n01 + n10
            p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
            print(f"McNemar {style:8s} {metric:8s}: noDS-only {n10}, DS-only {n01}, p={p:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run(a.model, a.max_tokens)
