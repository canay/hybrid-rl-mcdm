from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CAMPAIGN_ROOT.parents[1]
SOURCE_CAMPAIGN_NAME = "2026-07-12_codex_local_same_target_bridge_r01_producerenv"
EVIDENCE_CAMPAIGN_NAME = "2026-07-13_codex_static_verifier_topsis_gatefix"
SOURCE_CAMPAIGN = PROJECT_ROOT / "experiments" / SOURCE_CAMPAIGN_NAME
EVIDENCE_CAMPAIGN = PROJECT_ROOT / "experiments" / EVIDENCE_CAMPAIGN_NAME

CANONICAL_CATALOGS_SHA256 = "803eebfe09be8d62b5f446955f2106fe7ef8b220a979b68c9f9d71acb4827ecd"
CANONICAL_TERMINAL_SHA256 = "48677825f4446e2df427a0940dc8c0947b99aef1373ca5dfb6933f35728ad861"
CANONICAL_STATUS_SHA256 = "6470b05e83827637e34983511359a9eda24d26d0977d7976eb517dfe156ec2f3"
SOURCE_RUN_MANIFEST_SHA256 = "0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222"
OVERLAY_FULL_SHA256 = "a3112da73a28e9c68f3148b0a8668cc834472c7d5c765870ad1fed25e09fcd97"
OVERLAY_STATUS_SHA256 = "2513c8922a38e5f1a413d081b607225a55a620d7ad1cf540db208c48373b811f"
OVERLAY_RUN_MANIFEST_SHA256 = "ab29bb4782daf44c856f79ef8ad83559d4b59b637409476a43ed02dac8672ea7"
LOCKED_CORE_SHA256 = "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f"
PRIMARY_ARM_ID = "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix"
PROFILE_ORDER = ["budget", "quality_seeker", "explorer", "loyal", "balanced"]
RUN_INDICES = list(range(50))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def typed_array_contract(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    header = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return {**header, "typed_sha256": digest.hexdigest()}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("latency_locked_core_v2", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load locked core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_paths() -> dict[str, Path]:
    return {
        "catalogs": SOURCE_CAMPAIGN / "outputs" / "canonical_main" / "main_catalogs.jsonl",
        "terminal": SOURCE_CAMPAIGN / "outputs" / "canonical_main" / "main_results.json",
        "status": SOURCE_CAMPAIGN / "outputs" / "canonical_main" / "status.json",
        "source_run_manifest": SOURCE_CAMPAIGN / "RUN_MANIFEST.json",
        "core": SOURCE_CAMPAIGN / "src" / "original_hybrid_core.py",
        "overlay_full": EVIDENCE_CAMPAIGN / "outputs" / "canonical_overlay" / "FULL_VERIFICATION.json",
        "overlay_status": EVIDENCE_CAMPAIGN / "outputs" / "canonical_overlay" / "verification_status.json",
        "overlay_run_manifest": EVIDENCE_CAMPAIGN / "RUN_MANIFEST.json",
    }


def validate_distinct_canonical_evidence() -> dict[str, Path]:
    paths = canonical_paths()
    expected_hashes = {
        "catalogs": CANONICAL_CATALOGS_SHA256,
        "terminal": CANONICAL_TERMINAL_SHA256,
        "status": CANONICAL_STATUS_SHA256,
        "source_run_manifest": SOURCE_RUN_MANIFEST_SHA256,
        "core": LOCKED_CORE_SHA256,
        "overlay_full": OVERLAY_FULL_SHA256,
        "overlay_status": OVERLAY_STATUS_SHA256,
        "overlay_run_manifest": OVERLAY_RUN_MANIFEST_SHA256,
    }
    for key, expected in expected_hashes.items():
        if sha256_file(paths[key]) != expected:
            raise AssertionError(f"Canonical data/evidence hash mismatch: {key}")

    source_status = json.loads(paths["status"].read_text(encoding="utf-8"))
    if (
        source_status.get("status") != "completed_unverified"
        or source_status.get("mode") != "canonical"
        or source_status.get("campaign_id") != SOURCE_CAMPAIGN_NAME
        or source_status.get("run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
        or source_status.get("terminal_sha256") != CANONICAL_TERMINAL_SHA256
        or source_status.get("runs_completed") != 50
        or source_status.get("runs_total") != 50
    ):
        raise AssertionError("Canonical scientific-data runner status contract mismatch")

    overlay_status = json.loads(paths["overlay_status"].read_text(encoding="utf-8"))
    if (
        overlay_status.get("status") != "completed_verified"
        or overlay_status.get("mode") != "canonical"
        or overlay_status.get("campaign_id") != EVIDENCE_CAMPAIGN_NAME
        or overlay_status.get("source_campaign_id") != SOURCE_CAMPAIGN_NAME
        or overlay_status.get("full_verification_sha256") != OVERLAY_FULL_SHA256
        or overlay_status.get("scientific_values_exposed") is not False
    ):
        raise AssertionError("Distinct canonical overlay status contract mismatch")

    overlay_full = json.loads(paths["overlay_full"].read_text(encoding="utf-8"))
    expected_outputs = {
        "main_catalogs.jsonl": CANONICAL_CATALOGS_SHA256,
        "main_results.json": CANONICAL_TERMINAL_SHA256,
        "status.json": CANONICAL_STATUS_SHA256,
    }
    contract = overlay_full.get("overlay_contract", {})
    if (
        overlay_full.get("status") != "completed_verified"
        or overlay_full.get("verdict") != "PASS"
        or overlay_full.get("mode") != "canonical"
        or overlay_full.get("campaign_id") != EVIDENCE_CAMPAIGN_NAME
        or overlay_full.get("source_campaign_id") != SOURCE_CAMPAIGN_NAME
        or overlay_full.get("run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
        or overlay_full.get("output_hashes") != expected_outputs
        or contract.get("overlay_manifest_sha256") != OVERLAY_RUN_MANIFEST_SHA256
        or contract.get("source_canonical_checkpoint_sha256") != CANONICAL_CATALOGS_SHA256
        or contract.get("source_canonical_terminal_sha256") != CANONICAL_TERMINAL_SHA256
        or contract.get("source_canonical_runner_status_sha256") != CANONICAL_STATUS_SHA256
        or contract.get("source_run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
    ):
        raise AssertionError("Distinct canonical overlay FULL contract mismatch")
    return paths


def build_fixture(output_dir: Path, force: bool = False) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    fixture_path = output_dir / "verified_vectors.npz"
    manifest_path = output_dir / "fixture_manifest.json"
    if not force and (fixture_path.exists() or manifest_path.exists()):
        raise AssertionError("Fixture output exists; use a fresh directory or --force")

    paths = validate_distinct_canonical_evidence()
    lines = [line for line in paths["catalogs"].read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 50:
        raise AssertionError("Canonical fixture source must contain exactly 50 runs")

    core = load_module(paths["core"])
    topsis = np.empty((50, 400), dtype=np.float64)
    q_scores = np.empty((50, 5, 400), dtype=np.float64)
    expected_top7 = np.empty((50, 5, 7), dtype=np.int64)
    dataset_records: list[dict[str, Any]] = []

    for expected_run_index, line in enumerate(lines):
        payload = json.loads(line)
        if payload.get("run_index") != expected_run_index:
            raise AssertionError("Canonical run order/index mismatch")
        if payload.get("campaign_id") != SOURCE_CAMPAIGN_NAME:
            raise AssertionError("Canonical source campaign mismatch")
        if PRIMARY_ARM_ID not in payload.get("arms", {}):
            raise AssertionError("Primary corrected arm missing from canonical source")

        dataset_path = SOURCE_CAMPAIGN / "inputs" / Path(str(payload["dataset_path"]))
        dataset_sha = sha256_file(dataset_path)
        if dataset_sha != payload.get("dataset_sha256"):
            raise AssertionError(f"Canonical catalog hash mismatch: run {expected_run_index}")
        frame = pd.read_csv(dataset_path)
        run_topsis = np.asarray(core.topsis_artifacts(frame)["scores"], dtype=np.float64)
        if run_topsis.shape != (400,):
            raise AssertionError("Canonical TOPSIS vector shape mismatch")
        topsis[expected_run_index] = run_topsis

        arm = payload["arms"][PRIMARY_ARM_ID]
        if arm.get("arm", {}).get("arm_id") != PRIMARY_ARM_ID:
            raise AssertionError("Canonical primary-arm metadata mismatch")
        profiles = arm.get("profiles", [])
        if [item.get("profile_name") for item in profiles] != PROFILE_ORDER:
            raise AssertionError("Canonical profile order/count mismatch")
        for profile_index, profile in enumerate(profiles):
            q = np.asarray(profile.get("q_scores"), dtype=np.float64)
            expected = np.asarray(profile.get("final_rankings", {}).get("hybrid"), dtype=np.int64)
            if q.shape != (400,) or expected.shape != (7,):
                raise AssertionError("Canonical Q/ranking shape mismatch")
            actual = np.asarray(
                np.argsort(core.static_hybrid_score(q, run_topsis, lambda_q=0.50))[::-1][:7],
                dtype=np.int64,
            )
            if not np.array_equal(actual, expected):
                raise AssertionError("Locked core does not reproduce canonical expected top-7")
            q_scores[expected_run_index, profile_index] = q
            expected_top7[expected_run_index, profile_index] = expected
        dataset_records.append(
            {
                "run_index": expected_run_index,
                "path": str(dataset_path.relative_to(PROJECT_ROOT)),
                "sha256": dataset_sha,
            }
        )

    arrays = {
        "topsis": topsis,
        "q_scores": q_scores,
        "expected_top7": expected_top7,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="verified_vectors.", suffix=".npz", dir=output_dir)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, fixture_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    manifest: dict[str, Any] = {
        "schema_version": "hre.latency_fixture.v2",
        "campaign_id": CAMPAIGN_ROOT.name,
        "data_source": {
            "kind": "canonical_scientific_payload",
            "campaign": SOURCE_CAMPAIGN_NAME,
            "mode": "canonical",
            "main_catalogs_path": str(paths["catalogs"].relative_to(PROJECT_ROOT)),
            "main_catalogs_sha256": CANONICAL_CATALOGS_SHA256,
            "terminal_path": str(paths["terminal"].relative_to(PROJECT_ROOT)),
            "terminal_sha256": CANONICAL_TERMINAL_SHA256,
            "status_path": str(paths["status"].relative_to(PROJECT_ROOT)),
            "status_sha256": CANONICAL_STATUS_SHA256,
            "run_manifest_path": str(paths["source_run_manifest"].relative_to(PROJECT_ROOT)),
            "run_manifest_sha256": SOURCE_RUN_MANIFEST_SHA256,
            "arm_id": PRIMARY_ARM_ID,
            "run_indices": RUN_INDICES,
            "profile_order": PROFILE_ORDER,
            "datasets": dataset_records,
        },
        "verification_evidence": {
            "kind": "distinct_independent_canonical_overlay",
            "campaign": EVIDENCE_CAMPAIGN_NAME,
            "source_campaign": SOURCE_CAMPAIGN_NAME,
            "full_path": str(paths["overlay_full"].relative_to(PROJECT_ROOT)),
            "full_sha256": OVERLAY_FULL_SHA256,
            "status_path": str(paths["overlay_status"].relative_to(PROJECT_ROOT)),
            "status_sha256": OVERLAY_STATUS_SHA256,
            "run_manifest_path": str(paths["overlay_run_manifest"].relative_to(PROJECT_ROOT)),
            "run_manifest_sha256": OVERLAY_RUN_MANIFEST_SHA256,
            "status": "completed_verified",
            "verdict": "PASS",
        },
        "locked_core": {
            "path": str(paths["core"].relative_to(PROJECT_ROOT)),
            "sha256": LOCKED_CORE_SHA256,
        },
        "fixture_file": fixture_path.name,
        "fixture_sha256": sha256_file(fixture_path),
        "arrays": {name: typed_array_contract(values) for name, values in arrays.items()},
        "pair_schedule": {
            "pair_count": 250,
            "order": "run_index major, PROFILE_ORDER minor",
            "canonical_calls_per_pair_per_block": 20,
            "canonical_calls_per_pair_total": 1200,
        },
        "lambda_q": 0.50,
        "top_k": 7,
        "excluded_from_fixture": [
            "ground_truth",
            "gt_rank",
            "final_metrics",
            "checkpoint_metrics",
            "reward_diagnostics",
            "visits",
        ],
        "material_passport": {
            "content_type": "canonical-verified cached score/ranking vectors",
            "verification_status": "VERIFIED_BY_DISTINCT_OVERLAY",
            "allowed_use": "latency smoke and future separately authorized canonical timing",
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=CAMPAIGN_ROOT / "inputs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build_fixture(args.output_dir, args.force)
    print(
        json.dumps(
            {
                "status": "fixture_v2_ready",
                "fixture_sha256": manifest["fixture_sha256"],
                "arrays": manifest["arrays"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
