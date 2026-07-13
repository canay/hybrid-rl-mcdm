"""Build or check the locked Exact-XAI canonical run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy


CAMPAIGN_ID = "2026-07-13_codex_local_exact_xai_r1_producerenv"
OPERATION_ID = "HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19"
DATA_CAMPAIGN_ID = "2026-07-12_codex_local_same_target_bridge_r01_producerenv"
EVIDENCE_CAMPAIGN_ID = "2026-07-13_codex_static_verifier_topsis_gatefix"
PRIMARY_ARM_ID = "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix"
DATA_RUN_SHA = "0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222"
CORE_SHA = "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f"
EVIDENCE_RUN_SHA = "ab29bb4782daf44c856f79ef8ad83559d4b59b637409476a43ed02dac8672ea7"
DATA_HASHES = {
    "main_catalogs.jsonl": "803eebfe09be8d62b5f446955f2106fe7ef8b220a979b68c9f9d71acb4827ecd",
    "main_results.json": "48677825f4446e2df427a0940dc8c0947b99aef1373ca5dfb6933f35728ad861",
    "status.json": "6470b05e83827637e34983511359a9eda24d26d0977d7976eb517dfe156ec2f3",
}
EVIDENCE_HASHES = {
    "FULL_VERIFICATION.json": "a3112da73a28e9c68f3148b0a8668cc834472c7d5c765870ad1fed25e09fcd97",
    "verification_status.json": "2513c8922a38e5f1a413d081b607225a55a620d7ad1cf540db208c48373b811f",
}
FILES = (
    "PROTOCOL_LOCK.md",
    "src/xai_main.py",
    "verify_xai.py",
    "build_run_manifest.py",
    "tests/test_xai_contract.py",
    "tests/test_verify_xai.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root: Path) -> dict[str, Any]:
    environment = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
    if environment != {"numpy": "1.26.0", "pandas": "2.2.3", "scipy": "1.16.3"}:
        raise RuntimeError(f"Producer environment mismatch: {environment}")
    return {
        "schema_version": "exact_xai.run_manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "operation_id": OPERATION_ID,
        "status": "LOCKED",
        "canonical_authorized": True,
        "execution_policy": {
            "lifecycle_status": "LOCKED",
            "allowed_modes": ["smoke", "canonical"],
            "canonical_authorized": True,
            "authorization_operation_id": OPERATION_ID,
        },
        "environment": environment,
        "source_bindings": {
            "data": {
                "campaign_id": DATA_CAMPAIGN_ID,
                "canonical_root": "outputs/canonical_main",
                "run_manifest_sha256": DATA_RUN_SHA,
                "core_sha256": CORE_SHA,
                "canonical_output_hashes": DATA_HASHES,
            },
            "evidence": {
                "campaign_id": EVIDENCE_CAMPAIGN_ID,
                "canonical_root": "outputs/canonical_overlay",
                "run_manifest_sha256": EVIDENCE_RUN_SHA,
                "canonical_file_hashes": EVIDENCE_HASHES,
            },
            "primary_arm_id": PRIMARY_ARM_ID,
            "canonical_shape": {"catalogs": 50, "profiles_per_catalog": 5},
        },
        "files": [{"path": relative, "sha256": sha256(root / relative)} for relative in FILES],
        "full_verification_contract": {
            "binds": [
                "RUN_MANIFEST.json",
                "PROTOCOL_LOCK.md",
                "src/xai_main.py",
                "verify_xai.py",
                "build_run_manifest.py",
                "tests/test_xai_contract.py",
                "tests/test_verify_xai.py",
                "producer_environment",
            ],
            "terminal_verdict": "completed_verified/PASS",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    expected = build_manifest(root)
    path = root / "RUN_MANIFEST.json"
    if args.write:
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("RUN_MANIFEST.json is stale; rebuild only before applying ReadOnly lock")


if __name__ == "__main__":
    main()
