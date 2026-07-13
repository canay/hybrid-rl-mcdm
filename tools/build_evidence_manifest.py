#!/usr/bin/env python3
"""Build the deterministic evidence inventory used by verify_evidence.py."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_FILES = {"EVIDENCE_MANIFEST.json", "SHA256SUMS.txt"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.name in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--operation-id", required=True)
    args = parser.parse_args()

    rows = inventory()
    manifest = {
        "schema_version": "hre.public_evidence_manifest.v1",
        "generated_at": args.generated_at,
        "tool": "Codex",
        "model": "GPT-5",
        "operation_id": args.operation_id,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }
    (ROOT / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sums = "".join(f'{row["sha256"]}  {row["path"]}\n' for row in rows)
    (ROOT / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
    print(json.dumps({"file_count": len(rows), "total_bytes": manifest["total_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
