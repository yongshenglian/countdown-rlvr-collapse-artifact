"""Summarize the GSM8K evaluation-only prompt ablation.

The script combines the existing original think-prompt outputs with the four
ablation variants, re-applies the current GSM8K scorer, and reports accuracy,
response length, immediate reasoning-tag closure, and wrong copying of a
literal example answer.
"""
import json
import os
import re

import pandas as pd

from common import numbers_equal
from eval_gsm8k import score_response


HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(HERE, "results_8k", "gsm8k")
ABLATION = os.path.join(HERE, "results_prompt_ablation", "gsm8k")

MODELS = [
    f"matched_{method}_s{seed}_step300"
    for method in ("ppo", "grpo", "dapo")
    for seed in (7, 8, 9)
]
EARLY_SHORT = {
    "matched_ppo_s9_step300",
    "matched_grpo_s9_step300",
    "matched_dapo_s7_step300",
    "matched_dapo_s8_step300",
    "matched_dapo_s9_step300",
}
PERSISTENT_SHORT = {
    "matched_grpo_s9_step300",
    "matched_dapo_s7_step300",
    "matched_dapo_s8_step300",
    "matched_dapo_s9_step300",
}
VARIANTS = {
    "original_think": (ORIGINAL, "think", 42),
    "no_example": (ABLATION, "think_no_example", None),
    "example_17": (ABLATION, "think_example17", 17),
    "no_open_tag": (ABLATION, "think_no_open_tag", 42),
    "no_assistant_prefill": (ABLATION, "think_no_prefill", 42),
    "clean_tags": (ABLATION, "clean_tags", None),
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def summarize_file(model, variant, path, anchor):
    rows = read_jsonl(path)
    strict = []
    flexible = []
    predictions = []
    for row in rows:
        sp, ok_s, fp, ok_f = score_response(
            row["response"], str(row["gold"]), row["finish"]
        )
        strict.append(ok_s)
        flexible.append(ok_f)
        predictions.append(fp)
    lengths = pd.Series([len(row["response"]) for row in rows])
    immediate = [bool(re.match(r"^\s*</think>", row["response"])) for row in rows]
    if anchor is None:
        wrong_anchor = [False] * len(rows)
    else:
        wrong_anchor = [
            numbers_equal(str(pred), str(anchor))
            and not numbers_equal(str(row["gold"]), str(anchor))
            for pred, row in zip(predictions, rows)
        ]
    return {
        "model": model,
        "method": model.split("_")[1].upper(),
        "seed": int(re.search(r"_s(\d+)_", model).group(1)),
        "early_short": model in EARLY_SHORT,
        "persistent_short": model in PERSISTENT_SHORT,
        "variant": variant,
        "n": len(rows),
        "strict_accuracy": sum(strict) / len(rows),
        "flexible_accuracy": sum(flexible) / len(rows),
        "mean_output_chars": lengths.mean(),
        "median_output_chars": lengths.median(),
        "truncation_rate": sum(row["finish"] == "length" for row in rows) / len(rows),
        "immediate_close_rate": sum(immediate) / len(rows),
        "wrong_anchor_rate": sum(wrong_anchor) / len(rows),
    }


def main():
    records = []
    missing = []
    for model in MODELS:
        for variant, (directory, suffix, anchor) in VARIANTS.items():
            path = os.path.join(directory, f"{model}__{suffix}.jsonl")
            if not os.path.exists(path):
                missing.append(path)
                continue
            records.append(summarize_file(model, variant, path, anchor))

    out = pd.DataFrame(records)
    os.makedirs(ABLATION, exist_ok=True)
    out.to_csv(os.path.join(ABLATION, "summary_by_checkpoint.csv"), index=False)

    metrics = [
        "strict_accuracy", "flexible_accuracy", "mean_output_chars",
        "median_output_chars", "truncation_rate", "immediate_close_rate",
        "wrong_anchor_rate",
    ]
    group_rows = []
    for variant, frame in out.groupby("variant", sort=False):
        for group, selected in (
            ("all", frame),
            ("early_short", frame[frame.early_short]),
            ("other", frame[~frame.early_short]),
            ("persistent_short", frame[frame.persistent_short]),
            ("not_persistent_short", frame[~frame.persistent_short]),
            ("recovered_ppo9", frame[frame.model.eq("matched_ppo_s9_step300")]),
            ("long_throughout", frame[
                ~frame.persistent_short
                & ~frame.model.eq("matched_ppo_s9_step300")
            ]),
        ):
            row = {"variant": variant, "group": group, "checkpoints": len(selected)}
            row.update({metric: selected[metric].mean() for metric in metrics})
            group_rows.append(row)
        for method, selected in frame.groupby("method"):
            row = {"variant": variant, "group": method, "checkpoints": len(selected)}
            row.update({metric: selected[metric].mean() for metric in metrics})
            group_rows.append(row)
    groups = pd.DataFrame(group_rows)
    groups.to_csv(os.path.join(ABLATION, "summary_by_group.csv"), index=False)

    print(out.pivot(index="model", columns="variant", values="flexible_accuracy").to_string())
    print("\nGroup means (flexible accuracy):")
    print(groups.pivot(index="group", columns="variant", values="flexible_accuracy").to_string())
    if missing:
        print(f"\nIncomplete: {len(missing)} files are not yet available.")


if __name__ == "__main__":
    main()
