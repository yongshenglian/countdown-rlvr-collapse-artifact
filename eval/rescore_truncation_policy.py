"""Migrate archived scorer artifacts to the completed-response fallback rule.

No generation is repeated. Main-evaluation scored JSONL files are updated from
their saved labels. Pass@k caches are rebuilt for GSM8K and corrected only for
truncated MATH samples, avoiding an unnecessary full symbolic rescore.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / ".pydeps"))

import eval_gsm8k as G  # noqa: E402
import eval_math as M  # noqa: E402


def atomic_jsonl(path: Path, rows):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def migrate_main_scored():
    total_changed = 0
    for result_set in ("results", "results_8k"):
        for benchmark in ("math", "bbh", "mmlupro"):
            pattern = HERE / result_set / benchmark / "scored" / "*.jsonl"
            for name in sorted(glob.glob(str(pattern))):
                path = Path(name)
                changed = 0

                def rows():
                    nonlocal changed
                    with path.open() as source:
                        for line in source:
                            row = json.loads(line)
                            if row.get("finish") == "length" and not bool(row.get("correct_strict")):
                                if bool(row.get("correct_flexible")):
                                    changed += 1
                                row["correct_flexible"] = False
                            yield row

                atomic_jsonl(path, rows())
                total_changed += changed
                if changed:
                    print(
                        f"main {result_set:10s} {benchmark:8s} {path.name}: "
                        f"invalidated {changed} truncated fallback labels"
                    )
    print(f"main total invalidated: {total_changed}")


def write_cache(path: Path, counts: dict):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(counts, handle)
    os.replace(tmp, path)


def rebuild_gsm8k_passk():
    root = HERE / "results_passk" / "gsm8k"
    for raw in sorted(root.glob("*.jsonl")):
        counts = {"strict": [], "flexible": [], "n": [], "truncated": 0, "total": 0}
        for line in raw.open():
            row = json.loads(line)
            scored = [G.score_response(text, row["gold"], finish)
                      for text, finish in zip(row["responses"], row["finish"])]
            counts["strict"].append(sum(bool(x[1]) for x in scored))
            counts["flexible"].append(sum(bool(x[3]) for x in scored))
            counts["n"].append(len(scored))
            counts["truncated"] += sum(x == "length" for x in row["finish"])
            counts["total"] += len(scored)
        cache = root / "scored" / f"{raw.stem}.json"
        write_cache(cache, counts)
        print(f"passk gsm8k   {raw.stem}: rebuilt {len(counts['n'])} items")


def correct_math_passk():
    from math_verify import parse, verify

    root = HERE / "results_passk" / "math500"
    for raw in sorted(root.glob("*.jsonl")):
        cache = root / "scored" / f"{raw.stem}.json"
        counts = json.loads(cache.read_text())
        if counts.get("scorer_policy") == "completed_response_fallback_v1":
            print(f"passk math500 {raw.stem}: already migrated")
            continue
        invalidated = 0
        for item_index, line in enumerate(raw.open()):
            row = json.loads(line)
            gold = parse("$" + row["gold"] + "$")
            for text, finish in zip(row["responses"], row["finish"]):
                if finish != "length":
                    continue
                marker = M.final_marker(text)
                strict_ok = bool(marker is not None and verify(gold, parse("$" + marker + "$")))
                if strict_ok:
                    continue
                parsed = parse(text)
                legacy_flexible_ok = bool(parsed and verify(gold, parsed))
                if legacy_flexible_ok:
                    counts["flexible"][item_index] -= 1
                    invalidated += 1
        counts["scorer_policy"] = "completed_response_fallback_v1"
        write_cache(cache, counts)
        print(f"passk math500 {raw.stem}: invalidated {invalidated} truncated fallback samples")


if __name__ == "__main__":
    migrate_main_scored()
    rebuild_gsm8k_passk()
    correct_math_passk()
