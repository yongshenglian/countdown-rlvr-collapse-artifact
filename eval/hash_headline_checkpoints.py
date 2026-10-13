#!/usr/bin/env python3
"""Generate stable file and aggregate hashes for headline merged checkpoints."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKPOINTS = {
    "DS_step300": Path("/mnt/sdb/verl/DAPO_new/merged_hf/DS_step300"),
    "noDS_step300": Path("/mnt/sdb/verl/DAPO_new/merged_hf/noDS_step300"),
    "GRPO_n2_kl4e-2_step250": Path(
        "/mnt/sdb/verl/DAPO_new/merged_hf/GRPO_n2_kl4e-2_step250"
    ),
    "PPO_klF_ent0_lr1e-6_step520": Path(
        "/mnt/sdb/verl/DAPO_new/merged_hf/PPO_klF_ent0_lr1e-6_step520"
    ),
}
OUTPUT = HERE / "CHECKPOINT_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_directory(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "local_path": str(path),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "aggregate_definition": "SHA-256 of canonical JSON file list (path, bytes, sha256)",
        "files": files,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_model": {
            "repository": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        },
        "checkpoints": {},
    }
    for name, path in CHECKPOINTS.items():
        print(f"Hashing {name}: {path}", flush=True)
        payload["checkpoints"][name] = manifest_directory(path)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
