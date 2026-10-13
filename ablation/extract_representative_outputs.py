"""Extract rule-based examples from the archived 100-step screen logs.

Selection is fixed by metadata, not output content. Within seeds 3--6 (the
locally archived screen logs), each recipe contributes the lowest-numbered
endpoint-collapsed seed at its first sustained short-response event and, when
available, the lowest-numbered endpoint-healthy seed at step 100. If that
healthy seed had an earlier event and recovered, both its event and endpoint
examples are retained. Validation prints occur at steps 0, 5, ..., 100.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
TIMES = HERE / "collapse_times.csv"
OUT_JSON = HERE / "representative_outputs.json"
OUT_CSV = HERE / "representative_outputs_summary.csv"
OUT_MD = HERE / "representative_outputs.md"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RAY_PREFIX = re.compile(r"^\([^)]*pid=\d+\)\s*")


def clean_line(line: str) -> str:
    return RAY_PREFIX.sub("", ANSI.sub("", line)).rstrip("\n")


def parse_log(path: Path) -> list[dict]:
    lines = [clean_line(line) for line in path.read_text(errors="replace").splitlines()]
    response_positions = [i for i, line in enumerate(lines) if "[response]" in line]
    examples = []
    for ordinal, start in enumerate(response_positions):
        end = response_positions[ordinal + 1] if ordinal + 1 < len(response_positions) else len(lines)
        ground_index = next((i for i in range(start + 1, end) if "[ground_truth]" in lines[i]), None)
        score_index = next((i for i in range(start + 1, end) if "[score]" in lines[i]), None)
        if ground_index is None or score_index is None:
            raise RuntimeError(f"incomplete validation block {ordinal} in {path}")

        first = lines[start].split("[response]", 1)[1].lstrip()
        response = "\n".join([first, *lines[start + 1 : ground_index]]).strip()
        ground_text = lines[ground_index].split("[ground_truth]", 1)[1].strip()
        score_text = lines[score_index].split("[score]", 1)[1].strip()
        examples.append(
            {
                "validation_step": ordinal * 5,
                "ground_truth": ast.literal_eval(ground_text),
                "score": float(score_text),
                "response": response,
            }
        )
    if len(examples) != 21:
        raise RuntimeError(f"expected 21 validation examples in {path}, found {len(examples)}")
    return examples


def selection_specs(times: pd.DataFrame) -> list[dict]:
    specs = []
    for recipe in ["no DS (rung 2)", "DS", "A1", "A1 + KL 0.01"]:
        local = times[(times["recipe"] == recipe) & times["seed"].between(3, 6)].copy()
        collapsed = local[(local["event"]) & (local["binary_step100_status"] == "collapsed")]
        if len(collapsed):
            row = collapsed.sort_values("seed").iloc[0]
            specs.append(dict(row, selection_role="first_event_lowest_collapsed_seed",
                              selected_step=int(row["collapse_time"])))

        healthy = local[local["binary_step100_status"] == "healthy"]
        if len(healthy):
            row = healthy.sort_values("seed").iloc[0]
            if bool(row["event"]):
                specs.append(dict(row, selection_role="recovered_seed_at_first_event",
                                  selected_step=int(row["collapse_time"])))
                endpoint_role = "same_recovered_seed_at_endpoint"
            else:
                endpoint_role = "lowest_healthy_seed_at_endpoint"
            specs.append(dict(row, selection_role=endpoint_role, selected_step=100))
    return specs


def excerpt(response: str, width: int = 100) -> str:
    compact = " ".join(response.split())
    if len(compact) <= 2 * width + 3:
        return compact
    return compact[:width] + " ... " + compact[-width:]


def main() -> None:
    times = pd.read_csv(TIMES)
    records = []
    parsed: dict[Path, list[dict]] = {}
    for spec in selection_specs(times):
        matches = sorted(LOGS.glob(f"{spec['run']}-*.log"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one archived log for {spec['run']}, found {matches}")
        path = matches[0]
        if path not in parsed:
            parsed[path] = parse_log(path)
        selected = next(item for item in parsed[path] if item["validation_step"] == spec["selected_step"])
        response = selected["response"]
        records.append(
            {
                "recipe": spec["recipe"],
                "seed": int(spec["seed"]),
                "endpoint_status": spec["binary_step100_status"],
                "selection_role": spec["selection_role"],
                "validation_step": selected["validation_step"],
                "ground_truth": selected["ground_truth"],
                "score": selected["score"],
                "response_chars": len(response),
                "has_answer_tag": "<answer>" in response and "</answer>" in response,
                "excerpt": excerpt(response),
                "response": response,
                "source_log": str(path.relative_to(HERE.parent)),
            }
        )

    OUT_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    summary_fields = [key for key in records[0] if key != "response"]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in summary_fields} for row in records)

    lines = [
        "# Systematically selected collapse-screen outputs",
        "",
        "Selection uses recipe, seed, endpoint status, and event time only; response content is never a selection criterion. "
        "Within locally archived seeds 3--6, each recipe contributes the lowest-numbered endpoint-collapsed seed at its "
        "first sustained event and the lowest-numbered endpoint-healthy seed at step 100 when one exists. A recovered "
        "healthy seed is shown both at its earlier event and at step 100.",
        "",
    ]
    for index, row in enumerate(records, 1):
        lines += [
            f"## Example {index}: {row['recipe']}, seed {row['seed']}, step {row['validation_step']}",
            "",
            f"- Selection role: `{row['selection_role']}`",
            f"- Endpoint status: `{row['endpoint_status']}`",
            f"- Ground truth: `{row['ground_truth']}`",
            f"- Training scorer value: `{row['score']}`",
            f"- Response characters: `{row['response_chars']}`",
            f"- Source: `{row['source_log']}`",
            "",
            "````text",
            row["response"],
            "````",
            "",
        ]
    OUT_MD.write_text("\n".join(lines))
    print(pd.DataFrame(records)[summary_fields].to_string(index=False))
    print(f"wrote {OUT_JSON}, {OUT_CSV}, and {OUT_MD}")


if __name__ == "__main__":
    main()
