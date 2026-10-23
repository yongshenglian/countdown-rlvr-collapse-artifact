"""Evaluation-only ablation of the Countdown-style GSM8K prompt.

The original standard- and think-prompt outputs are left untouched.  This
script evaluates four variants that isolate the literal answer example, the
embedded transcript, and the prefilled opening <think> tag.  Generation and
scoring otherwise match eval_gsm8k.py (greedy vLLM decoding, one deterministic
response per GSM8K test item).

Example:
  CUDA_VISIBLE_DEVICES=0 /home/lian/miniconda3/envs/vllm-serve/bin/python \
      eval_gsm8k_prompt_ablation.py --model matched_grpo_s9_step300
"""
import argparse
import json
import os

import pandas as pd
from transformers import AutoTokenizer

from common import MODELS, THINK_TEMPLATE, generate, to_chat, write_jsonl
from eval_gsm8k import DATA, score_response


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_prompt_ablation", "gsm8k")
SOURCE_RECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results_8k", "gsm8k", "base__standard.jsonl")

TRANSCRIPT_PREFIX = (
    'system: You are a helpful assistant. You first thinks about the reasoning process in the mind '
    'and then provides the user with the answer."\n'
)


def build_prompt(question, variant):
    if variant == "think_no_example":
        return (
            TRANSCRIPT_PREFIX
            + f"User: {question} Show your work in <think> </think> tags. And return the final answer in "
            '<answer> </answer> tags."\n'
            "Assistant: Let me solve this step by step.\n<think>\n"
        )
    if variant == "think_example17":
        return THINK_TEMPLATE.format(question=question, example="17")
    if variant == "think_no_prefill":
        return (
            TRANSCRIPT_PREFIX
            + f"User: {question} Show your work in <think> </think> tags. And return the final answer in "
            '<answer> </answer> tags, for example <answer> 42 </answer>."'
        )
    if variant == "clean_tags":
        return (
            f"{question}\nShow your work between <think> and </think> tags, then put only the final "
            "numeric answer between <answer> and </answer> tags."
        )
    raise ValueError(f"unknown prompt variant: {variant}")


VARIANTS = ("think_no_example", "think_example17", "think_no_prefill", "clean_tags")


def complete(path, expected):
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle) == expected


def run(model_key, max_tokens, only_variant=None):
    # Reuse the canonical saved records from the original 8k GSM8K evaluation.
    # This is the same 1,319-example test set as DATA, while avoiding an otherwise
    # unnecessary pyarrow dependency in the vLLM inference environment.
    with open(SOURCE_RECORDS, encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle]
    questions = [row["question"] for row in source]
    gold = [row["gold"] for row in source]
    requested = (only_variant,) if only_variant else VARIANTS
    pending = [
        variant for variant in requested
        if not complete(os.path.join(OUT, f"{model_key}__{variant}.jsonl"), len(questions))
    ]
    if not pending:
        print(f"{model_key}: all prompt variants already complete")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_key])
    prompts = [
        to_chat(tokenizer, build_prompt(question, variant))
        for variant in pending for question in questions
    ]
    generations, _ = generate(MODELS[model_key], prompts, max_tokens=max_tokens)
    n = len(questions)
    for i, variant in enumerate(pending):
        rows = []
        for j in range(n):
            response, finish = generations[i * n + j]
            strict, strict_ok, flexible, flexible_ok = score_response(response, gold[j], finish)
            rows.append({
                "idx": j,
                "question": questions[j],
                "gold": gold[j],
                "prompt_variant": variant,
                "response": response,
                "finish": finish,
                "pred_strict": strict,
                "pred_flexible": flexible,
                "correct_strict": strict_ok,
                "correct_flexible": flexible_ok,
            })
        path = os.path.join(OUT, f"{model_key}__{variant}.jsonl")
        write_jsonl(path, rows)
        result = pd.DataFrame(rows)
        print(
            f"{model_key} {variant}: n={n} strict={result.correct_strict.mean():.4f} "
            f"flexible={result.correct_flexible.mean():.4f} "
            f"truncated={(result.finish == 'length').mean():.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[key for key in MODELS if key.startswith("matched_")])
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--variant", choices=VARIANTS,
                        help="evaluate one condition; by default run every incomplete condition")
    args = parser.parse_args()
    run(args.model, args.max_tokens, args.variant)
