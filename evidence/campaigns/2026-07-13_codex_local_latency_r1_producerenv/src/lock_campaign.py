from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CAMPAIGN_ROOT.parents[1]
OPERATION_ID = "HRE_R1_LATENCY_CANONICAL_LOCK_20260713_CODEX_20"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> int:
    lock_paths = {
        "fixture_builder": CAMPAIGN_ROOT / "src" / "prepare_verified_fixture.py",
        "lock_script": CAMPAIGN_ROOT / "src" / "lock_campaign.py",
        "protocol": CAMPAIGN_ROOT / "PROTOCOL_LOCK.md",
        "runner": CAMPAIGN_ROOT / "src" / "latency_benchmark.py",
        "verifier": CAMPAIGN_ROOT / "verify_latency.py",
        "test_latency_contract": CAMPAIGN_ROOT / "tests" / "test_latency_contract.py",
        "test_verify_latency": CAMPAIGN_ROOT / "tests" / "test_verify_latency.py",
        "fixture_manifest": CAMPAIGN_ROOT / "inputs" / "fixture_manifest.json",
        "fixture_vectors": CAMPAIGN_ROOT / "inputs" / "verified_vectors.npz",
    }
    payload = {
        "schema_version": "hre.latency.run_manifest.v1",
        "status": "LOCKED",
        "campaign_id": CAMPAIGN_ROOT.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "operation_id": OPERATION_ID,
        "execution_policy": {
            "allowed_modes": ["smoke", "canonical"],
            "canonical_launch_authorized": True,
            "canonical_timing_executed_at_lock": False,
            "authorization_operation_id": OPERATION_ID,
        },
        "environment": {
            "python_version": "3.12.12",
            "python_implementation": "CPython",
            "python_executable": str(
                (
                    PROJECT_ROOT
                    / "experiments"
                    / "_runtime"
                    / "hre_submission_py312_numpy1260_pandas223"
                    / "Scripts"
                    / "python.exe"
                ).relative_to(PROJECT_ROOT)
            ),
            "numpy_version": "1.26.0",
            "timer_name": "perf_counter_ns",
            "timer_implementation": "QueryPerformanceCounter()",
        },
        "source_bindings": {
            "canonical_main_catalogs_sha256": "803eebfe09be8d62b5f446955f2106fe7ef8b220a979b68c9f9d71acb4827ecd",
            "canonical_terminal_sha256": "48677825f4446e2df427a0940dc8c0947b99aef1373ca5dfb6933f35728ad861",
            "canonical_status_sha256": "6470b05e83827637e34983511359a9eda24d26d0977d7976eb517dfe156ec2f3",
            "source_run_manifest_sha256": "0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222",
            "overlay_full_sha256": "a3112da73a28e9c68f3148b0a8668cc834472c7d5c765870ad1fed25e09fcd97",
            "overlay_status_sha256": "2513c8922a38e5f1a413d081b607225a55a620d7ad1cf540db208c48373b811f",
            "overlay_run_manifest_sha256": "ab29bb4782daf44c856f79ef8ad83559d4b59b637409476a43ed02dac8672ea7",
        },
        "lock_files": {
            name: {
                "path": str(path.resolve().relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in lock_paths.items()
        },
    }
    output = CAMPAIGN_ROOT / "RUN_MANIFEST.json"
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "canonical_launch_authorized": True,
                "lock_file_count": len(lock_paths),
                "sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
