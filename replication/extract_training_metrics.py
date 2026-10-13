#!/usr/bin/env python3
"""Extract compact per-step metrics from a Verl/Ray driver log."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)$")

METRICS = [
    "val-aux/countdown/reward/mean@1",
    "val-aux/countdown/score/mean@1",
    "val-core/countdown/acc/mean@1",
    "val-aux/countdown/format_valid/mean@1",
    "val-aux/countdown/answer_extracted/mean@1",
    "critic/score/mean",
    "critic/rewards/mean",
    "actor/entropy_loss",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/grad_norm",
    "response_length/mean",
    "response_length/min",
    "response_length/max",
    "response_length/clip_ratio",
    "timing_s/step",
    "perf/throughput",
    "train/num_gen_batches",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def parse_metric_blob(blob: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in blob.split(" - "):
        if ":" not in field:
            continue
        key, raw_value = field.rsplit(":", 1)
        try:
            values[key.strip()] = float(raw_value.strip())
        except ValueError:
            continue
    return values


def main() -> None:
    args = parse_args()
    rows: dict[int, dict[str, float]] = {}

    with args.log.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line).rstrip()
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            rows[step] = parse_metric_blob(match.group(2))

    if not rows:
        raise SystemExit(f"No step metrics found in {args.log}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "recipe", "seed", "step", *METRICS]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(rows):
            record = {
                "run_id": args.run_id,
                "recipe": args.recipe,
                "seed": args.seed,
                "step": step,
            }
            record.update({name: rows[step].get(name, "") for name in METRICS})
            writer.writerow(record)

    terminal_step = max(rows)
    summary = {
        "source_log": str(args.log),
        "output": str(args.output),
        "rows": len(rows),
        "terminal_step": terminal_step,
        "terminal_metrics": {
            name: rows[terminal_step].get(name) for name in METRICS
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
