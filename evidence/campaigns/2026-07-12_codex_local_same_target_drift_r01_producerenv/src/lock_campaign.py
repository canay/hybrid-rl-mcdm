"""Create or verify the immutable same-target drift pre-compute manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_LOCK = CAMPAIGN_ROOT / "PROTOCOL_LOCK.md"
MANIFEST_PATH = CAMPAIGN_ROOT / "RUN_MANIFEST.json"
INPUT_MANIFEST = CAMPAIGN_ROOT / "inputs" / "data" / "processed" / "manifest.json"
OPERATION_ID = "HRE_R1_PRODUCERENV_DRIFT_LOCK_20260713_CODEX_08"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def load_runner():
    path = CAMPAIGN_ROOT / "src" / "drift_main.py"
    spec = importlib.util.spec_from_file_location("drift_lock_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_paths() -> list[Path]:
    runner = load_runner()
    runner.assert_exact_environment()
    runner.assert_frozen_inputs()
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if len(runs) != 50:
        raise ValueError("Expected 50 frozen catalogs")
    paths = [
        PROTOCOL_LOCK,
        CAMPAIGN_ROOT / "src" / "drift_main.py",
        CAMPAIGN_ROOT / "src" / "lock_campaign.py",
        CAMPAIGN_ROOT / "src" / "original_hybrid_core.py",
        CAMPAIGN_ROOT / "verify_drift.py",
        CAMPAIGN_ROOT / "tests" / "test_drift_contract.py",
        CAMPAIGN_ROOT / "tests" / "test_verify_drift.py",
        CAMPAIGN_ROOT / "inputs" / "design" / "SAME_TARGET_BRIDGE_PROTOCOL_2026-07-12.md",
        INPUT_MANIFEST,
        CAMPAIGN_ROOT / "inputs" / "results" / "amazon_primary.json",
        CAMPAIGN_ROOT / "inputs" / "results" / "amazon_drift.json",
        CAMPAIGN_ROOT / "inputs" / "results" / "validation_extensions.json",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "code" / "hybrid_core.py",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "code" / "run_amazon_experiments.py",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "code" / "validation_extensions.py",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "requirements.txt",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "COMMIT_METADATA.json",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "PUBLIC_TAG_EXACT_HASHES.json",
        CAMPAIGN_ROOT / "inputs" / "exact_replay_provenance" / "EXACT_REPLAY_AUDIT.json",
        CAMPAIGN_ROOT / "inputs" / "historical_reward_provenance" / "hybrid_rl_mcdm_v2.py",
        CAMPAIGN_ROOT / "inputs" / "historical_reward_provenance" / "supplementary_runs.py",
    ]
    seen: set[str] = set()
    for entry in runs:
        relative = str(entry["path"]).replace("\\", "/")
        if relative in seen:
            raise ValueError(f"Duplicate catalog: {relative}")
        seen.add(relative)
        path = CAMPAIGN_ROOT / "inputs" / Path(relative)
        if sha256_file(path) != str(entry["sha256"]).lower():
            raise ValueError(f"Catalog hash mismatch: {relative}")
        paths.append(path)
    relative_paths = [path.relative_to(CAMPAIGN_ROOT).as_posix() for path in paths]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Duplicate required lock path")
    expected = runner.expected_run_manifest_paths()
    if set(relative_paths) != expected:
        raise ValueError(
            f"Lock required-path contract mismatch: missing={sorted(expected-set(relative_paths))} "
            f"extra={sorted(set(relative_paths)-expected)}"
        )
    return paths


def create() -> None:
    protocol = PROTOCOL_LOCK.read_text(encoding="utf-8")
    if "Status: `LOCKED_BEFORE_COMPUTE`" not in protocol:
        raise ValueError("Local drift protocol must be independently approved and explicitly LOCKED_BEFORE_COMPUTE")
    if MANIFEST_PATH.exists():
        raise FileExistsError("RUN_MANIFEST already exists and is immutable")
    output_root = CAMPAIGN_ROOT / "outputs"
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise FileExistsError("Scientific outputs already exist; refusing to lock/relock")
    runner = load_runner()
    environment = runner.assert_exact_environment()
    runner.assert_frozen_inputs()
    paths = required_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    files = [
        {"path": path.relative_to(CAMPAIGN_ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(set(paths), key=lambda item: item.relative_to(CAMPAIGN_ROOT).as_posix())
    ]
    now = datetime.now(ZoneInfo("Europe/Istanbul"))
    payload = {
        "schema_version": "same_target_drift.run_manifest.v1",
        "campaign_id": CAMPAIGN_ROOT.name,
        "status": "LOCKED_BEFORE_COMPUTE",
        "created_at": now.isoformat(timespec="seconds"),
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "operation_id": OPERATION_ID,
        "hash_algorithm": "SHA-256",
        "contract": runner.RUN_MANIFEST_CONTRACT,
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": environment},
        "producer_provenance": {"tag": "v1.0-submission", "commit_sha1": runner.PRODUCER_COMMIT},
        "files": files,
    }
    atomic_write(MANIFEST_PATH, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
    print(f"DRIFT_RUN_MANIFEST_LOCKED files={len(files)} sha256={sha256_file(MANIFEST_PATH)}", flush=True)


def verify() -> None:
    runner = load_runner()
    digest = runner.assert_run_manifest()
    print(f"DRIFT_RUN_MANIFEST_VERIFY PASS sha256={digest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else create()


if __name__ == "__main__":
    main()
