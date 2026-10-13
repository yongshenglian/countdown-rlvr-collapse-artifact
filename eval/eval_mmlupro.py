"""MMLU-Pro test set: 12032 questions, 14 categories, up to 10 options (A-J), zero-shot CoT.

(The official protocol is 5-shot CoT; zero-shot is used for consistency with the other
benchmarks here. DS vs. no-DS vs. base are compared under identical conditions.)

Prompts:
  standard : official MMLU-Pro instruction, 'finish your answer with "the answer is (X)"'
  think    : Countdown training wrapper, example <answer> C </answer>
Scoring:
  strict   : letter in the final-answer marker that appears last ("answer is (X)", <answer>, \\boxed{})
  flexible : strict, else the last "(X)" anywhere in the response
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODELS, THINK_TEMPLATE, run_generation, summarize_benchmark  # noqa: E402

BENCH = "mmlupro"
DATASET_REVISION = "dd36ce4b34827164989f100331f82c5a29741747"
LETTERS = "ABCDEFGHIJ"


def load_items():
    import datasets
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id="TIGER-Lab/MMLU-Pro",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    data_dir = os.path.join(snapshot, "data")
    files = sorted(os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.startswith("test-") and name.endswith(".parquet"))
    ds = datasets.load_dataset("parquet", data_files={"test": files}, split="test")
    return [dict(question_id=ex["question_id"], category=ex["category"], question=ex["question"],
                 options=ex["options"], gold=ex["answer"]) for ex in ds]


def body(it):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["options"]))
    return f"Question: {it['question']}\nOptions:\n{opts}"


def build_prompt(it, style):
    if style == "standard":
        return (f"The following is a multiple choice question about {it['category']}. Think step by step and "
                f'then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n'
                + body(it) + "\nAnswer: Let's think step by step.")
    return THINK_TEMPLATE.format(question=body(it) + "\nAnswer with the letter of the correct option.", example="C")


MARKERS = [r"answer is:?\s*\(?\*?\*?([A-J])\b", r"<answer>\s*\(?([A-J])\b[^<]*</answer>", r"\\boxed\{\(?([A-J])\)?\}"]


def final_marker(text):
    cands = []
    for pat in MARKERS:
        ms = list(re.finditer(pat, text, re.I if pat.startswith("answer") else 0))
        if ms:
            cands.append((ms[-1].start(), ms[-1].group(1).upper()))
    return max(cands)[1] if cands else None


def score(row, style):
    m = final_marker(row["response"])
    ok_s = m == row["gold"]
    pf = m
    if m is None and row.get("finish") != "length":
        ls = re.findall(r"\(([A-J])\)", row["response"])
        pf = ls[-1] if ls else None
    return m, ok_s, pf, pf == row["gold"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    items = load_items()
    if a.summarize:
        summarize_benchmark(BENCH, score, group_col="category")
    else:
        run_generation(BENCH, a.model, items, build_prompt, a.max_tokens)
