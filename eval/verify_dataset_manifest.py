#!/usr/bin/env python3
"""Verify exact dataset bytes and canonical benchmark inputs used in the paper.

The default check is offline and reads only repository artifacts. Pass
--upstream to additionally reconstruct MATH, BBH, and MMLU-Pro from their
immutable Hugging Face revisions; this may require network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("DATASET_MANIFEST.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(rows: Iterable[dict]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        encoded = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        digest.update((encoded + "\n").encode("utf-8"))
        count += 1
    return count, digest.hexdigest()


def jsonl_rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def selected_rows(spec: dict) -> Iterable[dict]:
    fields = spec["canonical_fields"]
    for row in jsonl_rows(ROOT / spec["result_source"]):
        yield {field: row[field] for field in fields}


def check_equal(label: str, observed, expected, failures: list[str]) -> None:
    if observed == expected:
        print(f"PASS {label}: {observed}")
    else:
        failures.append(f"{label}: observed {observed!r}, expected {expected!r}")
        print(f"FAIL {failures[-1]}")


def verify_local(manifest: dict) -> list[str]:
    failures: list[str] = []
    for name, spec in manifest["datasets"].items():
        if "path" in spec:
            path = ROOT / spec["path"]
            if not path.is_file():
                failures.append(f"{name}: missing {path}")
                print(f"FAIL {failures[-1]}")
                continue
            check_equal(
                f"{name} file SHA-256",
                file_sha256(path),
                spec["file_sha256"],
                failures,
            )
            if path.suffix == ".parquet":
                import pandas as pd

                check_equal(
                    f"{name} rows",
                    len(pd.read_parquet(path)),
                    spec["rows"],
                    failures,
                )
            elif path.suffix == ".jsonl":
                check_equal(
                    f"{name} rows",
                    sum(1 for _ in jsonl_rows(path)),
                    spec["rows"],
                    failures,
                )

        if "result_source" in spec:
            count, digest = canonical_sha256(selected_rows(spec))
            check_equal(f"{name} canonical rows", count, spec["rows"], failures)
            check_equal(
                f"{name} canonical SHA-256",
                digest,
                spec["canonical_sha256"],
                failures,
            )

        if "subset_of_result" in spec:
            subset = list(jsonl_rows(ROOT / spec["path"]))
            superset = list(jsonl_rows(ROOT / spec["subset_of_result"]))
            key = spec["subset_key"]
            available = {str(row[key]).strip() for row in superset}
            matched = sum(str(row[key]).strip() in available for row in subset)
            check_equal(
                f"{name} rows matched to pinned MATH inputs",
                matched,
                spec["rows"],
                failures,
            )
    return failures


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]


def verify_upstream(manifest: dict) -> list[str]:
    import datasets

    failures: list[str] = []

    spec = manifest["datasets"]["math_test"]
    math_rows = []
    for subject in MATH_SUBJECTS:
        dataset = datasets.load_dataset(
            spec["repository"],
            subject,
            split="test",
            revision=spec["upstream_revision"],
        )
        math_rows.extend(
            {
                "problem": row["problem"],
                "solution": row["solution"],
                "level": row["level"],
                "subject": subject,
            }
            for row in dataset
        )
    count, digest = canonical_sha256(math_rows)
    check_equal("upstream MATH rows", count, spec["rows"], failures)
    check_equal("upstream MATH SHA-256", digest, spec["canonical_sha256"], failures)

    spec = manifest["datasets"]["bbh_test"]
    bbh_rows = []
    for task in BBH_TASKS:
        dataset = datasets.load_dataset(
            spec["repository"],
            task,
            split="test",
            revision=spec["upstream_revision"],
        )
        targets = dataset["target"]
        bbh_rows.extend(
            {
                "task": task,
                "input": row["input"],
                "gold": row["target"],
                "example": targets[1] if index == 0 else targets[0],
            }
            for index, row in enumerate(dataset)
        )
    count, digest = canonical_sha256(bbh_rows)
    check_equal("upstream BBH rows", count, spec["rows"], failures)
    check_equal("upstream BBH SHA-256", digest, spec["canonical_sha256"], failures)

    spec = manifest["datasets"]["mmlu_pro_test"]
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=spec["repository"],
        repo_type="dataset",
        revision=spec["upstream_revision"],
    ))
    parquet_files = sorted((snapshot / "data").glob("test-*.parquet"))
    if not parquet_files:
        failures.append(f"MMLU-Pro: no test parquet in immutable snapshot {snapshot}")
        return failures
    dataset = datasets.load_dataset(
        "parquet", data_files={"test": [str(path) for path in parquet_files]}, split="test"
    )
    mmlu_rows = (
        {
            "question_id": row["question_id"],
            "category": row["category"],
            "question": row["question"],
            "options": row["options"],
            "gold": row["answer"],
        }
        for row in dataset
    )
    count, digest = canonical_sha256(mmlu_rows)
    check_equal("upstream MMLU-Pro rows", count, spec["rows"], failures)
    check_equal(
        "upstream MMLU-Pro SHA-256", digest, spec["canonical_sha256"], failures
    )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream",
        action="store_true",
        help="also reconstruct datasets from immutable upstream revisions",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = verify_local(manifest)
    if args.upstream:
        failures.extend(verify_upstream(manifest))
    if failures:
        print(f"\n{len(failures)} verification failure(s).")
        return 1
    print("\nAll requested dataset checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
