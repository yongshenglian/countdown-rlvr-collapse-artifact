"""pass@k evaluation (Yue et al., arXiv:2504.13837): does RL make the model solve *more* problems, or only solve
the problems it could already solve more reliably?

  MATH-500 : 500 problems, N=64 samples each      GSM8K : 1319 problems, N=32 samples each
  standard prompts (same as eval_math.py / eval_gsm8k.py), temperature 0.6, top-p 0.95, max 4096 output tokens
  scoring  : same strict / flexible rules as the main evaluation
  pass@k   : unbiased estimator 1 - C(n-c, k) / C(n, k), averaged over problems, k = 1, 2, 4, ..., N

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_passk.py --model DS_step300 --bench math500
  python eval_passk.py --summarize
"""
import argparse
import glob
import json
import os
import sys
from math import comb

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODELS, to_chat, write_jsonl  # noqa: E402

OUT = os.path.join(HERE, os.environ.get("PASSK_RESULTS", "results_passk"))
SETTINGS = {"math500": dict(n=64), "gsm8k": dict(n=32)}
TEMPERATURE, TOP_P, MAX_TOKENS = 0.6, 0.95, 4096


def items(bench):
    if bench == "math500":
        import eval_math as M
        rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "math500.jsonl"))]
        return [dict(question=r["problem"], gold=r["answer"], subject=r.get("subject"), level=r.get("level")) for r in rows], \
            (lambda it: M.build_prompt({"problem": it["question"]}, "standard"))
    import eval_gsm8k as G
    df = pd.read_parquet(G.DATA)
    return [dict(question=r["question"], gold=g["ground_truth"]) for r, g in zip(df["extra_info"], df["reward_model"])], \
        (lambda it: G.build_prompts(it["question"], "standard"))


def run(model_key, bench):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    its, build = items(bench)
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    prompts = [to_chat(tok, build(it)) for it in its]
    llm = LLM(model=MODELS[model_key], dtype="bfloat16", max_model_len=MAX_TOKENS + 2048, gpu_memory_utilization=0.85, seed=0)
    sp = SamplingParams(n=SETTINGS[bench]["n"], temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=0)
    outs = llm.generate(prompts, sp)
    rows = [dict(it, idx=i, responses=[o.text for o in out.outputs], finish=[o.finish_reason for o in out.outputs])
            for i, (it, out) in enumerate(zip(its, outs))]
    write_jsonl(os.path.join(OUT, bench, f"{model_key}.jsonl"), rows)
    print(f"{bench} {model_key}: {len(rows)} problems x {SETTINGS[bench]['n']} samples")


def score_all(bench, row):
    """(strict, flexible) correctness lists for one problem's samples."""
    if bench == "math500":
        import eval_math as M
        sys.path.insert(0, os.path.join(HERE, ".pydeps"))
        res = [M.score(dict(gold=row["gold"], response=r, finish=f), "standard")
               for r, f in zip(row["responses"], row["finish"])]
        return [x[1] for x in res], [x[3] for x in res]
    import eval_gsm8k as G
    res = [G.score_response(r, row["gold"], f) for r, f in zip(row["responses"], row["finish"])]
    return [x[1] for x in res], [x[3] for x in res]


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def summarize():
    recs = []
    for bench in SETTINGS:
        for f in sorted(glob.glob(os.path.join(OUT, bench, "*.jsonl"))):
            model = os.path.basename(f)[:-6]
            cache = os.path.join(OUT, bench, "scored", model + ".json")
            if os.environ.get("RESCORE") != "1" and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(f):
                counts = json.load(open(cache))
            else:
                counts = {"strict": [], "flexible": [], "n": [], "truncated": 0, "total": 0}
                for line in open(f):
                    row = json.loads(line)
                    s, fl = score_all(bench, row)
                    counts["strict"].append(sum(map(bool, s))); counts["flexible"].append(sum(map(bool, fl)))
                    counts["n"].append(len(s))
                    counts["truncated"] += sum(x == "length" for x in row["finish"]); counts["total"] += len(row["finish"])
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                json.dump(counts, open(cache, "w"))
            n = counts["n"][0]
            ks = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= n]
            for metric in ["strict", "flexible"]:
                for k in ks:
                    v = sum(pass_at_k(nn, c, k) for nn, c in zip(counts["n"], counts[metric])) / len(counts["n"])
                    recs.append(dict(bench=bench, model=model, metric=metric, k=k, pass_at_k=round(100 * v, 2),
                                     truncated_pct=round(100 * counts["truncated"] / counts["total"], 2)))
    s = pd.DataFrame(recs)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    lines = []
    for bench in SETTINGS:
        for metric in ["strict", "flexible"]:
            t = s[(s.bench == bench) & (s.metric == metric)]
            if len(t):
                lines.append(f"== {bench} pass@k ({metric}) ==")
                lines.append(t.pivot_table(index="model", columns="k", values="pass_at_k").to_string())
    txt = "\n".join(lines)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--bench", choices=list(SETTINGS))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run(a.model, a.bench)
