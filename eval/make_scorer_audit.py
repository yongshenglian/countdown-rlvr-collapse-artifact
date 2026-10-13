"""Create a deterministic, stratified author-audit sample from saved predictions.

The sample contains one strict/flexible disagreement when available (otherwise
a matched control) and one general case for each headline checkpoint and
benchmark. Selection is by SHA-256 rank so it can
be regenerated without depending on file ordering or Python RNG versions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_8k"
MODELS = [
    "base",
    "DS_step300",
    "noDS_step300",
    "GRPO_n2_kl4e-2_step250",
    "PPO_klF_ent0_lr1e-6_step520",
]
BENCHMARKS = ["gsm8k", "math", "bbh", "mmlupro"]


def source_path(benchmark: str, model: str) -> Path:
    plain = RESULTS / benchmark / f"{model}__standard.jsonl"
    scored = RESULTS / benchmark / "scored" / f"{model}__standard.jsonl"
    return scored if scored.exists() else plain


def rank(benchmark: str, model: str, idx: object, stratum: str) -> str:
    key = f"scorer-audit-v1|{benchmark}|{model}|{idx}|{stratum}"
    return hashlib.sha256(key.encode()).hexdigest()


def select_one(
    path: Path, benchmark: str, model: str, disagreement: bool, stratum: str, exclude_idx=None
):
    best = None
    best_rank = None
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["idx"] == exclude_idx:
                continue
            differs = bool(row["correct_strict"]) != bool(row["correct_flexible"])
            if disagreement != differs:
                continue
            value = rank(benchmark, model, row["idx"], stratum)
            if best_rank is None or value < best_rank:
                best, best_rank = row, value
    return best


def prompt_text(row: dict) -> str:
    return row.get("question") or row.get("problem") or row.get("input") or ""


def main():
    output = []
    for benchmark in BENCHMARKS:
        for model in MODELS:
            path = source_path(benchmark, model)
            if not path.exists():
                raise FileNotFoundError(path)
            edge = select_one(path, benchmark, model, True, "strict_flexible_disagreement")
            edge_stratum = "strict_flexible_disagreement"
            if edge is None:
                edge_stratum = "no_disagreement_control"
                edge = select_one(path, benchmark, model, False, edge_stratum)
            general = select_one(path, benchmark, model, False, "general", exclude_idx=edge["idx"])
            for stratum, row in ((edge_stratum, edge), ("general", general)):
                output.append(
                    {
                        "audit_id": f"{benchmark}-{model}-{stratum}",
                        "benchmark": benchmark,
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
    out = HERE / "scorer_audit_sample.jsonl"
    with out.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in output:
        counts[row["benchmark"]] = counts.get(row["benchmark"], 0) + 1
    print(f"wrote {len(output)} records to {out}")
    print("counts:", counts)


if __name__ == "__main__":
    main()
