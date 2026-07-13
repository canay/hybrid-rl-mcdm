"""Create or verify the immutable pre-compute manifest for the bridge."""

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
PROJECT_ROOT = CAMPAIGN_ROOT.parents[1]
PROTOCOL_SOURCE = PROJECT_ROOT / "MD" / "02_design" / "SAME_TARGET_BRIDGE_PROTOCOL_2026-07-12.md"
PROTOCOL_LOCK = CAMPAIGN_ROOT / "PROTOCOL_LOCK.md"
MANIFEST_PATH = CAMPAIGN_ROOT / "RUN_MANIFEST.json"
INPUT_MANIFEST = CAMPAIGN_ROOT / "inputs" / "data" / "processed" / "manifest.json"
OPERATION_ID = "HRE_R1_PRODUCERENV_BRIDGE_LOCK_20260713_CODEX_04"


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


def load_bridge():
    path = CAMPAIGN_ROOT / "src" / "bridge_main.py"
    spec = importlib.util.spec_from_file_location("bridge_lock_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required_paths() -> list[Path]:
    bridge = load_bridge()
    bridge.assert_exact_environment()
    bridge.assert_historical_reward_provenance()
    bridge.assert_producer_provenance()
    bridge.assert_exact_replay_provenance()
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if len(runs) != 50:
        raise ValueError(f"Expected 50 frozen catalogs, got {len(runs)}")
    paths = [
        PROTOCOL_LOCK,
        CAMPAIGN_ROOT / "src" / "bridge_main.py",
        CAMPAIGN_ROOT / "src" / "lock_campaign.py",
        CAMPAIGN_ROOT / "src" / "original_hybrid_core.py",
        CAMPAIGN_ROOT / "verify_bridge.py",
        CAMPAIGN_ROOT / "tests" / "test_bridge_contract.py",
        CAMPAIGN_ROOT / "tests" / "test_verify_bridge.py",
        INPUT_MANIFEST,
        CAMPAIGN_ROOT / "inputs" / "results" / "amazon_primary.json",
        CAMPAIGN_ROOT / "inputs" / "historical_reward_provenance" / "hybrid_rl_mcdm_v2.py",
        CAMPAIGN_ROOT / "inputs" / "historical_reward_provenance" / "supplementary_runs.py",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "code" / "hybrid_core.py",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "code" / "run_amazon_experiments.py",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "requirements.txt",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "COMMIT_METADATA.json",
        CAMPAIGN_ROOT / "inputs" / "producer_provenance" / "v1.0-submission" / "PUBLIC_TAG_EXACT_HASHES.json",
        CAMPAIGN_ROOT / "inputs" / "exact_replay_provenance" / "EXACT_REPLAY_AUDIT.json",
    ]
    seen_catalogs: set[str] = set()
    for run in runs:
        rel = str(run["path"]).replace("\\", "/")
        if rel in seen_catalogs:
            raise ValueError(f"Duplicate catalog in input manifest: {rel}")
        seen_catalogs.add(rel)
        catalog = CAMPAIGN_ROOT / "inputs" / Path(rel)
        if sha256_file(catalog) != str(run["sha256"]).lower():
            raise ValueError(f"Frozen catalog hash mismatch: {rel}")
        paths.append(catalog)
    return paths


def create() -> None:
    if not PROTOCOL_SOURCE.is_file():
        raise FileNotFoundError(PROTOCOL_SOURCE)
    protocol_text = PROTOCOL_SOURCE.read_text(encoding="utf-8")
    if "Status: `LOCKED_BEFORE_COMPUTE`" not in protocol_text:
        raise ValueError("Protocol must be explicitly LOCKED_BEFORE_COMPUTE")
    if MANIFEST_PATH.exists() or PROTOCOL_LOCK.exists():
        raise FileExistsError("Pre-compute lock already exists and is immutable")
    output_root = CAMPAIGN_ROOT / "outputs"
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise FileExistsError("Scientific outputs already exist; refusing to relock")
    bridge = load_bridge()
    environment_versions = bridge.assert_exact_environment()
    producer_provenance = bridge.assert_producer_provenance()
    exact_replay_provenance = bridge.assert_exact_replay_provenance()
    atomic_write(PROTOCOL_LOCK, protocol_text.encode("utf-8"))
    paths = required_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required lock files missing: {missing}")
    files = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(CAMPAIGN_ROOT).as_posix()):
        files.append(
            {
                "path": path.relative_to(CAMPAIGN_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    now = datetime.now(ZoneInfo("Europe/Istanbul"))
    payload = {
        "schema_version": "same_target_bridge.run_manifest.v1",
        "campaign_id": CAMPAIGN_ROOT.name,
        "status": "LOCKED_BEFORE_COMPUTE",
        "created_at": now.isoformat(timespec="seconds"),
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "operation_id": OPERATION_ID,
        "hash_algorithm": "SHA-256",
        "contract": {
            "catalogs": 50,
            "profiles": 5,
            "episodes": 30000,
            "checkpoints": list(bridge.CHECKPOINTS),
            "mandatory_arms": len(bridge.mandatory_arms()),
            "sensitivity_arms": len(bridge.sensitivity_arms()),
            "total_arms": len(bridge.ARMS),
            "exact_replay_arm": bridge.EXACT_ARM_ID,
            "primary_corrected_arm": bridge.PRIMARY_CORRECTED_ARM_ID,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": environment_versions,
        },
        "producer_provenance": {
            "tag": "v1.0-submission",
            "commit_sha1": bridge.PRODUCER_COMMIT_SHA1,
            "file_sha256": producer_provenance,
        },
        "exact_replay_provenance": exact_replay_provenance,
        "files": files,
    }
    atomic_write(MANIFEST_PATH, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
    print(f"RUN_MANIFEST_LOCKED files={len(files)} sha256={sha256_file(MANIFEST_PATH)}", flush=True)


def verify() -> None:
    bridge = load_bridge()
    digest = bridge.assert_run_manifest()
    print(f"RUN_MANIFEST_VERIFY PASS sha256={digest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        create()


if __name__ == "__main__":
    main()
