"""BIG-Bench Hard: 27 tasks, 6511 examples, zero-shot chain of thought.

(The original BBH protocol is 3-shot CoT; zero-shot is used here so all benchmarks share
the same two prompt styles. Scores are therefore not directly comparable to 3-shot numbers
in the literature, but DS vs. no-DS vs. base are compared under identical conditions.)

Prompts:
  standard : input + 'Let's think step by step, and end your response with "So the answer is <answer>."'
  think    : Countdown training wrapper; the <answer> example is the target of another item of
             the same task, so the model sees the answer *format* but not this item's answer
Scoring:
  strict   : text of the final-answer marker that appears last ("the answer is ...", <answer>, \\boxed{})
  flexible : strict, else for multiple-choice tasks the last "(X)" anywhere in the response
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MODELS, THINK_TEMPLATE, run_generation, summarize_benchmark  # noqa: E402

BENCH = "bbh"
DATASET_REVISION = "b5306be6f827cfafbb545ff5a51f96916029b0fd"
TASKS = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "dyck_languages",
         "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
         "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation",
         "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table",
         "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks",
         "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
         "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", "web_of_lies",
         "word_sorting"]
STANDARD_SUFFIX = '\n\nLet\'s think step by step, and end your response with "So the answer is <answer>."'


def load_items():
    import datasets
    items = []
    for t in TASKS:
        ds = datasets.load_dataset("SaylorTwift/bbh", t, split="test", revision=DATASET_REVISION)
        targets = ds["target"]
        for j, ex in enumerate(ds):
            items.append(dict(task=t, input=ex["input"], gold=ex["target"],
                              example=targets[1] if j == 0 else targets[0]))
    return items


def build_prompt(it, style):
    if style == "standard":
        return it["input"] + STANDARD_SUFFIX
    return THINK_TEMPLATE.format(question=it["input"], example=it["example"])


MARKERS = [r"[Tt]he answer is[:\s]*(.+)", r"<answer>(.*?)</answer>", r"\\boxed\{([^{}]*)\}"]


def final_marker(text):
    cands = []
    for pat in MARKERS:
        ms = list(re.finditer(pat, text, re.S if "answer>" in pat else 0))
        if ms:
            cands.append((ms[-1].start(), ms[-1].group(1)))
    return max(cands)[1].strip() if cands else None


def clean(s):
    s = s.strip().split("\n")[0].strip()
    s = s.strip("*\"'`. ")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return s


def is_mc(gold):
    return re.fullmatch(r"\([A-R]\)", gold) is not None


def match(pred, gold, task, question=""):
    if pred is None:
        return False
    p = clean(pred)
    if is_mc(gold):
        m = re.search(r"\(([A-R])\)", p) or re.fullmatch(r"([A-R])(?:[).:\s].*)?", p)
        return bool(m) and m.group(1) == gold[1]
    if task == "dyck_languages":
        # gold is only the closing suffix; models often answer with the full completed sequence
        ns = lambda x: re.sub(r"\s+", "", x)
        seq = question.split("Input:")[-1].strip()
        return ns(p) in (ns(gold), ns(seq) + ns(gold))
    if task == "word_sorting":
        return " ".join(re.findall(r"[a-z']+", p.lower())) == " ".join(gold.lower().split())
    try:
        return abs(float(p.replace(",", "")) - float(gold)) < 1e-6
    except ValueError:
        pass
    return p.lower() == gold.lower()


def score(row, style):
    m = final_marker(row["response"])
    ok_s = match(m, row["gold"], row["task"], row["input"])
    pf, ok_f = m, ok_s
    if not ok_s and m is None and row.get("finish") != "length" and is_mc(row["gold"]):
        ls = re.findall(r"\(([A-R])\)", row["response"])
        pf = f"({ls[-1]})" if ls else None
        ok_f = match(pf, row["gold"], row["task"], row["input"])
    return m, ok_s, pf, ok_f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    items = load_items()
    if a.summarize:
        summarize_benchmark(BENCH, score, group_col="task")
    else:
        run_generation(BENCH, a.model, items, build_prompt, a.max_tokens)
