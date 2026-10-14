"""Create a deterministic 40-record audit sample from saved GSM8K predictions.

The sample targets the last-number fallback that matters most for the reported
GSM8K comparison: 24 base-model responses correct only by fallback, eight
base-model responses with an incorrect fallback result, four base strict-correct
controls, and one fallback-only correct response from each of four trained
checkpoints. Selection is by SHA-256 rank so it is reproducible without
depending on file order or Python RNG versions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_8k"
BENCHMARK = "gsm8k"
MODELS = [
    "base",
    "DS_step300",
    "noDS_step300",
    "GRPO_n2_kl4e-2_step250",
    "PPO_klF_ent0_lr1e-6_step520",
]


def source_path(benchmark: str, model: str) -> Path:
    plain = RESULTS / benchmark / f"{model}__standard.jsonl"
    scored = RESULTS / benchmark / "scored" / f"{model}__standard.jsonl"
    return scored if scored.exists() else plain


def rank(benchmark: str, model: str, idx: object, stratum: str) -> str:
    key = f"scorer-audit-v2|{benchmark}|{model}|{idx}|{stratum}"
    return hashlib.sha256(key.encode()).hexdigest()


def select_many(path: Path, model: str, stratum: str, count: int, predicate):
    candidates = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if predicate(row):
                candidates.append((rank(BENCHMARK, model, row["idx"], stratum), row))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < count:
        raise ValueError(
            f"{model}/{stratum}: requested {count}, found {len(candidates)}"
        )
    return [row for _, row in candidates[:count]]


def prompt_text(row: dict) -> str:
    return row.get("question") or row.get("problem") or row.get("input") or ""


def main():
    base_path = source_path(BENCHMARK, "base")
    selections = [
        (
            "base",
            "base_fallback_only_correct",
            select_many(
                base_path,
                "base",
                "base_fallback_only_correct",
                24,
                lambda row: row.get("pred_strict") is None
                and bool(row.get("correct_flexible"))
                and row.get("finish") != "length",
            ),
        ),
        (
            "base",
            "base_fallback_incorrect",
            select_many(
                base_path,
                "base",
                "base_fallback_incorrect",
                8,
                lambda row: row.get("pred_strict") is None
                and row.get("pred_flexible") is not None
                and not bool(row.get("correct_flexible"))
                and row.get("finish") != "length",
            ),
        ),
        (
            "base",
            "base_strict_correct_control",
            select_many(
                base_path,
                "base",
                "base_strict_correct_control",
                4,
                lambda row: bool(row.get("correct_strict")),
            ),
        ),
    ]
    for model in MODELS[1:]:
        path = source_path(BENCHMARK, model)
        selections.append(
            (
                model,
                "trained_fallback_only_correct",
                select_many(
                    path,
                    model,
                    "trained_fallback_only_correct",
                    1,
                    lambda row: row.get("pred_strict") is None
                    and bool(row.get("correct_flexible"))
                    and row.get("finish") != "length",
                ),
            )
        )

    output = []
    for model, stratum, rows in selections:
        for row in rows:
            output.append(
                {
                    "audit_id": f"{BENCHMARK}-{model}-{stratum}-{row['idx']}",
                    "benchmark": BENCHMARK,
                    "model": model,
                    "stratum": stratum,
                    "idx": row["idx"],
                    "prompt": prompt_text(row),
                    "options": row.get("options"),
                    "gold": row.get("gold"),
                    "response": row.get("response"),
                    "finish": row.get("finish"),
                    "pred_strict": row.get("pred_strict"),
                    "pred_flexible": row.get("pred_flexible"),
                    "correct_strict": bool(row.get("correct_strict")),
                    "correct_flexible": bool(row.get("correct_flexible")),
                    "audit_extraction_correct": None,
                    "audit_correctness_label_correct": None,
                    "audit_error_category": None,
                    "audit_note": None,
                }
            )
    if len(output) != 40:
        raise AssertionError(f"expected 40 records, generated {len(output)}")

    out = HERE / "scorer_audit_sample.jsonl"
    with out.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in output:
        key = (row["model"], row["stratum"])
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {len(output)} records to {out}")
    for key, count in sorted(counts.items()):
        print(f"{key}: {count}")

if __name__ == "__main__":
    main()
