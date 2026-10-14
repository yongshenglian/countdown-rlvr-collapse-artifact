#!/usr/bin/env python3
"""Validate and report a completed independent scorer audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY_PATH = HERE / "independent_audit" / "coordinator_key.jsonl"
ALLOWED_CATEGORIES = {
    "no_error",
    "extraction_error",
    "incidental_match",
    "ambiguous_expression",
    "reference_error",
    "truncation_policy_error",
    "other",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def validate(rows: list[dict], metadata: dict, keys: list[dict]) -> None:
    if len(rows) != 40:
        raise ValueError(f"expected 40 records, found {len(rows)}")
    if len(keys) != 40:
        raise ValueError(f"expected 40 key records, found {len(keys)}")
    ids = [row.get("blind_id") for row in rows]
    if len(set(ids)) != 40 or set(ids) != {row["blind_id"] for row in keys}:
        raise ValueError("completed record IDs do not match the coordinator key")
    for row in rows:
        for field in ("human_strict_correct", "human_tolerant_correct"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"{row.get('blind_id')}: {field} must be true or false")
        category = row.get("human_error_category")
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"{row.get('blind_id')}: invalid error category {category!r}")
        if category != "no_error" and not row.get("human_note"):
            raise ValueError(f"{row.get('blind_id')}: non-no_error category requires a note")
    if not metadata.get("auditor_name_or_id"):
        raise ValueError("auditor_name_or_id is required")
    if not metadata.get("audit_date_utc"):
        raise ValueError("audit_date_utc is required")
    if metadata.get("independent_of_authors") is not True:
        raise ValueError("independent_of_authors must be true for an independent audit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "independent_audit" / "INDEPENDENT_AUDIT_RESULTS.md",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.completed)
    keys = read_jsonl(KEY_PATH)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    validate(rows, metadata, keys)

    by_id = {row["blind_id"]: row for row in rows}
    strict_agree = 0
    tolerant_agree = 0
    disagreements = []
    categories = Counter()
    stratum_total = Counter()
    stratum_strict_agree = Counter()
    stratum_tolerant_agree = Counter()
    for key in keys:
        human = by_id[key["blind_id"]]
        categories[human["human_error_category"]] += 1
        strict_match = human["human_strict_correct"] == key["automatic_strict_correct"]
        tolerant_match = human["human_tolerant_correct"] == key["automatic_tolerant_correct"]
        strict_agree += strict_match
        tolerant_agree += tolerant_match
        stratum = key["stratum"]
        stratum_total[stratum] += 1
        stratum_strict_agree[stratum] += strict_match
        stratum_tolerant_agree[stratum] += tolerant_match
        if not (strict_match and tolerant_match):
            disagreements.append((key, human, strict_match, tolerant_match))

    strict_ci = wilson(strict_agree, 40)
    tolerant_ci = wilson(tolerant_agree, 40)
    final = metadata.get("adjudication_complete") is True
    lines = [
        "# Independent Scorer Audit Results",
        "",
        f"Status: **{'final, adjudicated' if final else 'draft; adjudication not marked complete'}**",
        "",
        f"- Auditor: {metadata['auditor_name_or_id']}",
        f"- Audit date (UTC): {metadata['audit_date_utc']}",
        f"- Independent of authors: {metadata['independent_of_authors']}",
        f"- Completed-record SHA-256: `{sha256(args.completed)}`",
        f"- Metadata SHA-256: `{sha256(args.metadata)}`",
        f"- Strict-label agreement: {strict_agree}/40 ({strict_agree / 40:.1%}; Wilson 95% CI {strict_ci[0]:.1%}--{strict_ci[1]:.1%})",
        f"- Tolerant-label agreement: {tolerant_agree}/40 ({tolerant_agree / 40:.1%}; Wilson 95% CI {tolerant_ci[0]:.1%}--{tolerant_ci[1]:.1%})",
        f"- Records with any pre-adjudication label disagreement: {len(disagreements)}/40",
        "",
        "## Error-category counts",
        "",
    ]
    lines.extend(f"- {category}: {categories[category]}" for category in sorted(categories))
    lines.extend(["", "## Label disagreements", ""])
    if disagreements:
        lines.append("| Blind ID | Benchmark | Strict auto/human | Tolerant auto/human | Category | Note |")
        lines.append("|---|---|---:|---:|---|---|")
        for key, human, _, _ in disagreements:
            note = str(human.get("human_note") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {key['blind_id']} | {key['benchmark']} | "
                f"{key['automatic_strict_correct']}/{human['human_strict_correct']} | "
                f"{key['automatic_tolerant_correct']}/{human['human_tolerant_correct']} | "
                f"{human['human_error_category']} | {note} |"
            )
    else:
        lines.append("No strict- or tolerant-correctness label disagreements.")
    lines.extend(
        [
            "",
            "## Sign-off note",
            "",
            str(metadata.get("notes") or "No additional note supplied."),
            "",
        ]
    )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    if not final:
        print("WARNING: metadata does not mark adjudication_complete=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
