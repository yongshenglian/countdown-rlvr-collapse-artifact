#!/usr/bin/env python3
"""Create or verify a deterministic SHA-256 manifest for an artifact tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, output: Path) -> dict:
    output_resolved = output.resolve()
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if ".git" in path.relative_to(root).parts:
            continue
        if not path.is_file() or path.is_symlink() or path.resolve() == output_resolved:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "root_name": root.name,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "aggregate_definition": "SHA-256 of canonical JSON file list (path, bytes, sha256)",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"artifact root is not a directory: {root}")
    current = inventory(root, args.output)
    if args.verify:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        if current != expected:
            raise SystemExit("release manifest verification FAILED")
        print(f"release manifest verified: {current['file_count']} files")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: {current['file_count']} files, "
        f"{current['total_bytes']} bytes, aggregate {current['aggregate_sha256']}"
    )


if __name__ == "__main__":
    main()
