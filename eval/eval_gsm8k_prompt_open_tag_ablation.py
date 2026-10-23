"""Narrow GSM8K control: remove only the prefilled opening <think> tag."""
import argparse
import json
import os

import pandas as pd
from transformers import AutoTokenizer

from common import MODELS, generate, to_chat, write_jsonl
from eval_gsm8k import score_response


HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_RECORDS = os.path.join(HERE, "results_8k", "gsm8k", "base__standard.jsonl")
OUT = os.path.join(HERE, "results_prompt_ablation", "gsm8k")

TRANSCRIPT_PREFIX = (
    'system: You are a helpful assistant. You first thinks about the reasoning process in the mind '
    'and then provides the user with the answer."\n'
)


def build_prompt(question):
    return (
        TRANSCRIPT_PREFIX
        + f"User: {question} Show your work in <think> </think> tags. And return the final answer in "
        '<answer> </answer> tags, for example <answer> 42 </answer>."\n'
        "Assistant: Let me solve this step by step.\n"
    )


def run(model_key, max_tokens):
    path = os.path.join(OUT, f"{model_key}__think_no_open_tag.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            if sum(1 for _ in handle) == 1319:
                print(f"{model_key}: narrow opening-tag control already complete")
                return

    with open(SOURCE_RECORDS, encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle]
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])
    prompts = [to_chat(tokenizer, build_prompt(row["question"])) for row in source]
    generations, _ = generate(MODELS[model_key], prompts, max_tokens=max_tokens)

    rows = []
    for row, (response, finish) in zip(source, generations):
        strict, strict_ok, flexible, flexible_ok = score_response(
            response, str(row["gold"]), finish
        )
        rows.append({
            "idx": row["idx"],
            "question": row["question"],
            "gold": str(row["gold"]),
            "prompt_variant": "think_no_open_tag",
            "response": response,
            "finish": finish,
            "pred_strict": strict,
            "pred_flexible": flexible,
            "correct_strict": strict_ok,
            "correct_flexible": flexible_ok,
        })
    write_jsonl(path, rows)
    result = pd.DataFrame(rows)
    print(
        f"{model_key} think_no_open_tag: n={len(rows)} "
        f"strict={result.correct_strict.mean():.4f} "
        f"flexible={result.correct_flexible.mean():.4f} "
        f"truncated={(result.finish == 'length').mean():.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=[key for key in MODELS if key.startswith("matched_")])
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()
    run(args.model, args.max_tokens)
