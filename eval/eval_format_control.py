"""Run and analyse a strong format-only GSM8K prompt control.

The control changes only the instruction appended to the standard GSM8K
question.  It makes the final ``#### <number>`` line mandatory, while retaining
the same zero-shot chain-of-thought request, base model, chat template, greedy
decoding, token limit, dataset, and scorer used by the headline evaluation.

Generation (one local GPU):

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      python eval_format_control.py --generate

Analysis (CPU; can be repeated without generation):

    python eval_format_control.py --analyse
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd

from common import generate, mcnemar_p, to_chat, write_jsonl
from eval_gsm8k import DATA, STANDARD_SUFFIX, score_response


HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results_8k"
OUT_DIR = RESULTS_ROOT / "format_control"
OUT_FILE = OUT_DIR / "base__format_strong.jsonl"
SUMMARY_CSV = OUT_DIR / "summary.csv"
REPORT_MD = OUT_DIR / "FORMAT_CONTROL.md"
BASE_SNAPSHOT = Path(
    "/home/lian/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

# Preserve the original prompt verbatim, then add a stronger output contract.
# The angle-bracket placeholder is deliberately explained rather than shown as
# a literal target response so that it cannot be mistaken for the answer.
FORMAT_SUFFIX = (
    STANDARD_SUFFIX
    + " Your response must end with exactly one separate final line of the form "
    '"#### N", where N is replaced by only the final numeric answer. '
    + "This final line is mandatory and must be the last line of your response."
)


def _load_dataset() -> tuple[list[str], list[str]]:
    df = pd.read_parquet(DATA)
    questions = [row["question"] for row in df["extra_info"]]
    gold = [row["ground_truth"] for row in df["reward_model"]]
    return questions, gold


def run_generation(max_tokens: int) -> None:
    if not BASE_SNAPSHOT.is_dir():
        raise FileNotFoundError(f"Missing pinned base-model snapshot: {BASE_SNAPSHOT}")

    from transformers import AutoTokenizer

    questions, gold = _load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_SNAPSHOT), local_files_only=True)
    prompts = [to_chat(tokenizer, question + FORMAT_SUFFIX) for question in questions]
    generations, _ = generate(str(BASE_SNAPSHOT), prompts, max_tokens=max_tokens)

    rows = []
    for idx, ((response, finish), question, answer) in enumerate(
        zip(generations, questions, gold)
    ):
        pred_strict, correct_strict, pred_flexible, correct_flexible = score_response(
            response, answer, finish
        )
        rows.append(
            {
                "idx": idx,
                "question": question,
                "gold": answer,
                "response": response,
                "finish": finish,
                "pred_strict": pred_strict,
                "pred_flexible": pred_flexible,
                "correct_strict": correct_strict,
                "correct_flexible": correct_flexible,
            }
        )

    write_jsonl(str(OUT_FILE), rows)
    frame = pd.DataFrame(rows)
    print(
        f"format_strong n={len(frame)} "
        f"strict={frame.correct_strict.mean():.4f} "
        f"flexible={frame.correct_flexible.mean():.4f} "
        f"truncated={(frame.finish == 'length').mean():.4f}"
    )


def _load_scored(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True, dtype={"gold": str})
    scored = [
        score_response(text, gold, finish)
        for text, gold, finish in zip(frame.response, frame.gold, frame.finish)
    ]
    (
        frame["pred_strict"],
        frame["correct_strict"],
        frame["pred_flexible"],
        frame["correct_flexible"],
    ) = map(list, zip(*scored))
    return frame.sort_values("idx").reset_index(drop=True)


def _paired_difference(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """Return b-a percentage-point difference and normal paired 95% CI."""
    delta = b.astype(float).to_numpy() - a.astype(float).to_numpy()
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / math.sqrt(len(delta)))
    return 100 * mean, 100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se)


def _holm_two(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def analyse() -> None:
    if not OUT_FILE.exists():
        raise FileNotFoundError(f"Run --generate first; missing {OUT_FILE}")

    comparison_paths = {
        "Base, standard prompt": RESULTS_ROOT / "gsm8k/base__standard.jsonl",
        "Base, strong-format prompt": OUT_FILE,
        "DS, step 300, standard prompt": RESULTS_ROOT / "gsm8k/DS_step300__standard.jsonl",
        "No-DS, step 300, standard prompt": RESULTS_ROOT / "gsm8k/noDS_step300__standard.jsonl",
        "GRPO KL 0.04, step 250, standard prompt": (
            RESULTS_ROOT / "gsm8k/GRPO_n2_kl4e-2_step250__standard.jsonl"
        ),
    }
    missing = [str(path) for path in comparison_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison files:\n" + "\n".join(missing))

    frames = {name: _load_scored(path) for name, path in comparison_paths.items()}
    expected_idx = frames["Base, standard prompt"].idx.tolist()
    for name, frame in frames.items():
        if frame.idx.tolist() != expected_idx:
            raise ValueError(f"Item alignment differs for {name}")

    summary_rows = []
    for name, frame in frames.items():
        marked = frame.pred_strict.notna()
        summary_rows.append(
            {
                "condition": name,
                "n": len(frame),
                "strict_accuracy": frame.correct_strict.mean(),
                "flexible_accuracy": frame.correct_flexible.mean(),
                "explicit_marker_rate": marked.mean(),
                "truncation_rate": (frame.finish == "length").mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    standard = frames["Base, standard prompt"]
    strong = frames["Base, strong-format prompt"]
    tests = []
    raw_ps = []
    for metric in ("strict", "flexible"):
        a = standard[f"correct_{metric}"]
        b = strong[f"correct_{metric}"]
        diff, low, high = _paired_difference(a, b)
        standard_only, strong_only, p_value = mcnemar_p(a.to_numpy(), b.to_numpy())
        raw_ps.append(p_value)
        tests.append(
            {
                "metric": metric,
                "standard_only": standard_only,
                "strong_only": strong_only,
                "difference_pp": diff,
                "ci_low_pp": low,
                "ci_high_pp": high,
                "p_raw": p_value,
            }
        )
    for test, p_adjusted in zip(tests, _holm_two(raw_ps)):
        test["p_holm"] = p_adjusted

    def pct(value: float) -> str:
        return f"{100 * value:.2f}"

    lines = [
        "# GSM8K Format-Only Control",
        "",
        "The base model was evaluated on all 1,319 GSM8K test items with greedy ",
        "decoding and the same 8,192-token limit, chat template, dataset, and ",
        "scorer as the headline evaluation. The control retains the original ",
        "zero-shot chain-of-thought prompt and appends a mandatory final-line ",
        "contract: `#### N`, with `N` replaced by only the numeric answer.",
        "",
        "## Accuracy and compliance",
        "",
        "| Condition | n | Strict (%) | Tolerant (%) | Explicit marker (%) | Truncated (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['n']} | {pct(row['strict_accuracy'])} | "
            f"{pct(row['flexible_accuracy'])} | {pct(row['explicit_marker_rate'])} | "
            f"{pct(row['truncation_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Prespecified paired base-model comparison",
            "",
            "| Metric | Standard only | Strong-format only | Difference (points) | Paired 95% CI | Raw exact p | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for test in tests:
        lines.append(
            f"| {test['metric'].capitalize()} | {test['standard_only']} | "
            f"{test['strong_only']} | {test['difference_pp']:+.2f} | "
            f"[{test['ci_low_pp']:+.2f}, {test['ci_high_pp']:+.2f}] | "
            f"{test['p_raw']:.4g} | {test['p_holm']:.4g} |"
        )

    strict_gain = tests[0]["difference_pp"]
    tolerant_gain = tests[1]["difference_pp"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The format-only instruction changed strict accuracy by {strict_gain:+.2f} "
            f"points and tolerant accuracy by {tolerant_gain:+.2f} points. Strict-score "
            "movement without a corresponding tolerant-score movement is attributable "
            "to answer presentation rather than newly correct solutions. Any tolerant "
            "difference is a prompt effect and must not be described as a training gain.",
            "",
            "This is a prompt-only control, not a format-only supervised-tuning control. "
            "It establishes what can be obtained from a stronger inference instruction "
            "without changing model weights.",
            "",
            "## Reproduction metadata",
            "",
            f"- Model snapshot: `{BASE_SNAPSHOT.name}`",
            "- Dataset: `eval/data/gsm8k_test.parquet` (1,319 records)",
            "- Decoding: greedy (`temperature=0`), maximum 8,192 generated tokens",
            "- Output archive: `eval/results_8k/format_control/base__format_strong.jsonl`",
            "- Analysis script: `eval/eval_format_control.py`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(REPORT_MD.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if not args.generate and not args.analyse:
        parser.error("select --generate and/or --analyse")
    if args.generate:
        run_generation(args.max_tokens)
    if args.analyse:
        analyse()


if __name__ == "__main__":
    main()
