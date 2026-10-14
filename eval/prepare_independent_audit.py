#!/usr/bin/env python3
"""Prepare a blinded packet for the required second-person scorer audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "scorer_audit_sample.jsonl"
OUTDIR = HERE / "independent_audit"
ALLOWED_CATEGORIES = [
    "no_error",
    "extraction_error",
    "incidental_match",
    "ambiguous_expression",
    "reference_error",
    "truncation_policy_error",
    "other",
]


def rank(record: dict) -> str:
    value = "independent-scorer-audit-v2|" + record["audit_id"]
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=rank)
    OUTDIR.mkdir(exist_ok=True)

    blinded = []
    key = []
    for number, row in enumerate(rows, 1):
        blind_id = f"AUD-{number:03d}"
        blinded.append(
            {
                "blind_id": blind_id,
                "benchmark": row["benchmark"],
                "prompt": row["prompt"],
                "options": row["options"],
                "gold": row["gold"],
                "response": row["response"],
                "finish": row["finish"],
                "human_strict_prediction": None,
                "human_tolerant_prediction": None,
                "human_strict_correct": None,
                "human_tolerant_correct": None,
                "human_error_category": None,
                "human_note": None,
            }
        )
        key.append(
            {
                "blind_id": blind_id,
                "audit_id": row["audit_id"],
                "benchmark": row["benchmark"],
                "model": row["model"],
                "stratum": row["stratum"],
                "idx": row["idx"],
                "automatic_pred_strict": row["pred_strict"],
                "automatic_pred_tolerant": row["pred_flexible"],
                "automatic_strict_correct": row["correct_strict"],
                "automatic_tolerant_correct": row["correct_flexible"],
            }
        )

    for path, records in (
        (OUTDIR / "records_to_complete.jsonl", blinded),
        (OUTDIR / "coordinator_key.jsonl", key),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "auditor_name_or_id": None,
        "audit_date_utc": None,
        "independent_of_authors": None,
        "records_file_sha256_before_annotation": hashlib.sha256(
            (OUTDIR / "records_to_complete.jsonl").read_bytes()
        ).hexdigest(),
        "adjudication_complete": False,
        "notes": None,
    }
    (OUTDIR / "auditor_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(blinded)} blinded records to {OUTDIR}")


if __name__ == "__main__":
    main()
