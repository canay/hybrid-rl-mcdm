from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


CAMPAIGN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CAMPAIGN_ROOT.parents[1]
BOOTSTRAP_SEED = 20_260_713
THRESHOLD_NS = 1_000_000
CANONICAL_CONTRACT = {
    "warmup": 10_000,
    "passes": 3,
    "blocks": 20,
    "calls_per_block": 5_000,
    "bootstrap_replicates": 2_000,
    "timer_overhead_samples": 50_000,
}
REQUIRED_CORE_SHA256 = "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f"
SOURCE_CAMPAIGN_NAME = "2026-07-12_codex_local_same_target_bridge_r01_producerenv"
EVIDENCE_CAMPAIGN_NAME = "2026-07-13_codex_static_verifier_topsis_gatefix"
CANONICAL_CATALOGS_SHA256 = "803eebfe09be8d62b5f446955f2106fe7ef8b220a979b68c9f9d71acb4827ecd"
CANONICAL_TERMINAL_SHA256 = "48677825f4446e2df427a0940dc8c0947b99aef1373ca5dfb6933f35728ad861"
CANONICAL_STATUS_SHA256 = "6470b05e83827637e34983511359a9eda24d26d0977d7976eb517dfe156ec2f3"
SOURCE_RUN_MANIFEST_SHA256 = "0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222"
OVERLAY_FULL_SHA256 = "a3112da73a28e9c68f3148b0a8668cc834472c7d5c765870ad1fed25e09fcd97"
OVERLAY_STATUS_SHA256 = "2513c8922a38e5f1a413d081b607225a55a620d7ad1cf540db208c48373b811f"
OVERLAY_RUN_MANIFEST_SHA256 = "ab29bb4782daf44c856f79ef8ad83559d4b59b637409476a43ed02dac8672ea7"
PRIMARY_ARM_ID = "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix"
PROFILE_ORDER = ["budget", "quality_seeker", "explorer", "loyal", "balanced"]
RUN_INDICES = list(range(50))
PAIR_COUNT = 250
EXPECTED_PYTHON_VERSION = [3, 12, 12]
EXPECTED_PYTHON_IMPLEMENTATION = "CPython"
EXPECTED_NUMPY_VERSION = "1.26.0"
EXPECTED_TIMER_IMPLEMENTATION = "QueryPerformanceCounter()"
EXPECTED_CLAIM_BOUNDARY = "cached 400-item Q/TOPSIS static_hybrid_score plus full argsort top-7 only"


class Audit:
    def __init__(self) -> None:
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise AssertionError(message)


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def typed_array_contract(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    header = {"dtype": array.dtype.str, "shape": list(array.shape), "nbytes": int(array.nbytes)}
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


def independent_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo + 1e-10)


def independent_top7(q_scores: np.ndarray, topsis_scores: np.ndarray) -> list[int]:
    scores = 0.5 * independent_norm(q_scores) + 0.5 * independent_norm(topsis_scores)
    return [int(x) for x in np.argsort(scores)[::-1][:7]]


def independent_topsis(frame: pd.DataFrame) -> np.ndarray:
    columns = ["price_pct", "quality_pct", "popularity_pct", "rating_pct"]
    matrix = frame[columns].to_numpy(dtype=float).copy()
    matrix = np.clip(matrix, 1e-10, None)
    proportions = matrix / matrix.sum(axis=0, keepdims=True)
    proportions = np.clip(proportions, 1e-10, 1.0)
    entropy = -np.sum(proportions * np.log(proportions), axis=0) / np.log(len(matrix))
    diversification = 1.0 - entropy
    floor = 0.10
    remaining = 1.0 - floor * len(columns)
    weights = floor + (diversification / diversification.sum()) * remaining
    weights = np.clip(weights, floor, None)
    weights = weights / weights.sum()
    norms = np.sqrt((matrix**2).sum(axis=0))
    norms[norms == 0] = 1.0
    weighted = (matrix / norms) * weights
    ideal_plus = weighted.max(axis=0)
    ideal_minus = weighted.min(axis=0)
    d_plus = np.sqrt(((weighted - ideal_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - ideal_minus) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    denom[denom == 0] = 1e-10
    return np.asarray(d_minus / denom, dtype=np.float64)


def summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise AssertionError("Cannot summarize empty values")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    return {
        "n": int(arr.size),
        "mean_ns": mean,
        "std_ns": std,
        "cv": float(std / mean) if mean > 0 else None,
        "min_ns": float(np.min(arr)),
        "median_ns": float(np.percentile(arr, 50, method="linear")),
        "p95_ns": float(np.percentile(arr, 95, method="linear")),
        "p99_ns": float(np.percentile(arr, 99, method="linear")),
        "max_ns": float(np.max(arr)),
    }


def block_bootstrap_ci(
    block_summaries: Sequence[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    if not block_summaries:
        raise AssertionError("No stable blocks for bootstrap")
    rng = np.random.RandomState(seed)
    result: dict[str, Any] = {
        "method": "nonparametric bootstrap over blocks; estimator is the median of the selected block-level statistic",
        "replicates": int(replicates),
        "seed": int(seed),
        "n_blocks": len(block_summaries),
        "confidence_level": 0.95,
        "statistics": {},
    }
    for metric in ("median_ns", "p95_ns", "p99_ns"):
        values = np.asarray([float(block[metric]) for block in block_summaries], dtype=float)
        estimates = np.empty(replicates, dtype=float)
        for index in range(replicates):
            sample = values[rng.randint(0, len(values), size=len(values))]
            estimates[index] = float(np.median(sample))
        result["statistics"][metric] = {
            "estimate_ns": float(np.median(values)),
            "ci_lo_ns": float(np.percentile(estimates, 2.5, method="linear")),
            "ci_hi_ns": float(np.percentile(estimates, 97.5, method="linear")),
        }
    return result


def analyze_durations(
    durations: np.ndarray, calls_per_block: int, bootstrap_replicates: int
) -> dict[str, Any]:
    if durations.ndim != 3 or durations.shape[2] != calls_per_block:
        raise AssertionError("Raw duration shape mismatch")
    passes, blocks, _ = durations.shape
    block_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    all_stable_blocks: list[dict[str, Any]] = []
    for pass_index in range(passes):
        summaries = [summary(durations[pass_index, block]) for block in range(blocks)]
        medians = np.asarray([item["median_ns"] for item in summaries], dtype=float)
        center = float(np.median(medians))
        mad = float(np.median(np.abs(medians - center)))
        tolerance = float(max(0.20 * center, 6.0 * 1.4826 * mad, 500.0))
        stable_indices: list[int] = []
        for block_index, item in enumerate(summaries):
            stable = bool(item["n"] == calls_per_block and abs(float(item["median_ns"]) - center) <= tolerance)
            row = {
                "pass_index": pass_index,
                "block_index": block_index,
                **item,
                "stable": stable,
                "stability_center_ns": center,
                "stability_mad_ns": mad,
                "stability_tolerance_ns": tolerance,
            }
            block_rows.append(row)
            if stable:
                stable_indices.append(block_index)
                all_stable_blocks.append(row)
        stable_raw = np.concatenate([durations[pass_index, idx] for idx in stable_indices])
        pass_rows.append(
            {
                "pass_index": pass_index,
                "all_samples": summary(durations[pass_index].reshape(-1)),
                "stable_samples": summary(stable_raw),
                "stable_blocks": len(stable_indices),
                "total_blocks": blocks,
                "minimum_stable_blocks": int(math.ceil(0.90 * blocks)),
                "stability_sufficient": len(stable_indices) >= int(math.ceil(0.90 * blocks)),
                "bootstrap": block_bootstrap_ci(
                    [block_rows[pass_index * blocks + idx] for idx in stable_indices],
                    bootstrap_replicates,
                    BOOTSTRAP_SEED + pass_index + 1,
                ),
            }
        )
    stable_raw_all = np.concatenate(
        [durations[row["pass_index"], row["block_index"]] for row in all_stable_blocks]
    )
    all_block_ok = all(float(row["p99_ns"]) < THRESHOLD_NS for row in all_stable_blocks)
    every_pass_stable_ok = all(float(row["stable_samples"]["p99_ns"]) < THRESHOLD_NS for row in pass_rows)
    stability_ok = all(bool(row["stability_sufficient"]) for row in pass_rows)
    pooled_raw_ok = float(summary(durations.reshape(-1))["p99_ns"]) < THRESHOLD_NS
    every_pass_all_ok = all(float(row["all_samples"]["p99_ns"]) < THRESHOLD_NS for row in pass_rows)
    return {
        "raw_all_samples": summary(durations.reshape(-1)),
        "stable_all_samples": summary(stable_raw_all),
        "blocks": block_rows,
        "passes": pass_rows,
        "bootstrap_all_stable_blocks": block_bootstrap_ci(
            all_stable_blocks, bootstrap_replicates, BOOTSTRAP_SEED
        ),
        "threshold_diagnostics": {
            "threshold_ns": THRESHOLD_NS,
            "primary_gate_basis": "all raw timed calls; no stability filtering",
            "pooled_raw_p99_strictly_below_threshold": pooled_raw_ok,
            "every_pass_all_sample_p99_strictly_below_threshold": every_pass_all_ok,
            "primary_condition_met": bool(pooled_raw_ok and every_pass_all_ok),
            "equality_to_threshold_fails": True,
            "stable_block_diagnostics_are_secondary": True,
            "stability_sufficient": stability_ok,
            "all_stable_block_p99_strictly_below_threshold": all_block_ok,
            "every_pass_stable_sample_p99_strictly_below_threshold": every_pass_stable_ok,
        },
    }


def compare(audit: Audit, expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, dict):
        audit.require(isinstance(actual, dict), f"{path} must be an object")
        audit.require(set(expected) == set(actual), f"{path} keys differ")
        for key in expected:
            compare(audit, expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        audit.require(isinstance(actual, list) and len(expected) == len(actual), f"{path} list differs")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(audit, left, right, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        audit.require(
            isinstance(actual, (int, float)) and np.isclose(float(expected), float(actual), rtol=1e-12, atol=1e-9),
            f"{path} numeric mismatch",
        )
    else:
        audit.require(expected == actual, f"{path} mismatch")


def quarantine_verification_artifacts(output_dir: Path, reason: str) -> list[str]:
    moved: list[str] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for name in ("FULL_VERIFICATION.json", "verification_status.json"):
        source = output_dir / name
        if source.exists():
            target = output_dir / f"REJECTED_{reason}_{stamp}__{name}"
            os.replace(source, target)
            moved.append(target.name)
    return moved


def reject_preexisting_verification_artifacts(output_dir: Path) -> None:
    found = [name for name in ("FULL_VERIFICATION.json", "verification_status.json") if (output_dir / name).exists()]
    if found:
        quarantine_verification_artifacts(output_dir, "PREEXISTING")
        raise AssertionError("Pre-existing verification artifacts rejected fail-closed: " + ", ".join(found))


def require_verified_source_terminal(audit: Audit, payload: dict[str, Any]) -> None:
    audit.require(payload.get("status") == "completed_verified", "Source status is not completed_verified")
    audit.require(payload.get("verdict") == "PASS", "Source verdict is not PASS")
    audit.require(
        payload.get("output_hashes")
        == {
            "main_catalogs.jsonl": CANONICAL_CATALOGS_SHA256,
            "main_results.json": CANONICAL_TERMINAL_SHA256,
            "status.json": CANONICAL_STATUS_SHA256,
        },
        "Source FULL does not bind main_catalogs",
    )


def validate_fixture_lineage(
    audit: Audit, manifest_path: Path, fixture_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("data_source", {})
    evidence = manifest.get("verification_evidence", {})
    expected_source_root = PROJECT_ROOT / "experiments" / SOURCE_CAMPAIGN_NAME
    expected_evidence_root = PROJECT_ROOT / "experiments" / EVIDENCE_CAMPAIGN_NAME
    expected_catalogs = expected_source_root / "outputs" / "canonical_main" / "main_catalogs.jsonl"
    expected_terminal = expected_source_root / "outputs" / "canonical_main" / "main_results.json"
    expected_status = expected_source_root / "outputs" / "canonical_main" / "status.json"
    expected_source_manifest = expected_source_root / "RUN_MANIFEST.json"
    expected_overlay_full = expected_evidence_root / "outputs" / "canonical_overlay" / "FULL_VERIFICATION.json"
    expected_overlay_status = expected_evidence_root / "outputs" / "canonical_overlay" / "verification_status.json"
    expected_overlay_manifest = expected_evidence_root / "RUN_MANIFEST.json"
    expected_core = expected_source_root / "src" / "original_hybrid_core.py"
    required = {
        "schema_version": "hre.latency_fixture.v2",
        "campaign_id": CAMPAIGN_ROOT.name,
        "fixture_file": "verified_vectors.npz",
        "lambda_q": 0.5,
        "top_k": 7,
    }
    for key, expected in required.items():
        audit.require(manifest.get(key) == expected, f"Fixture manifest {key} mismatch")
    audit.require(sha256_file(fixture_path) == manifest.get("fixture_sha256"), "Fixture NPZ manifest hash mismatch")
    audit.require(source.get("kind") == "canonical_scientific_payload", "Fixture data-source kind mismatch")
    audit.require(source.get("campaign") == SOURCE_CAMPAIGN_NAME, "Fixture source campaign mismatch")
    audit.require(source.get("mode") == "canonical", "Fixture source mode mismatch")
    audit.require(source.get("arm_id") == PRIMARY_ARM_ID, "Fixture primary arm mismatch")
    audit.require(source.get("run_indices") == RUN_INDICES, "Fixture run order/index mismatch")
    audit.require(source.get("profile_order") == PROFILE_ORDER, "Fixture profile order mismatch")
    audit.require(source.get("main_catalogs_sha256") == CANONICAL_CATALOGS_SHA256, "Source catalogs allowlist mismatch")
    audit.require(source.get("terminal_sha256") == CANONICAL_TERMINAL_SHA256, "Source terminal allowlist mismatch")
    audit.require(source.get("status_sha256") == CANONICAL_STATUS_SHA256, "Source status allowlist mismatch")
    audit.require(source.get("run_manifest_sha256") == SOURCE_RUN_MANIFEST_SHA256, "Source manifest allowlist mismatch")
    source_catalogs = (PROJECT_ROOT / Path(str(source.get("main_catalogs_path", "")))).resolve()
    source_terminal = (PROJECT_ROOT / Path(str(source.get("terminal_path", "")))).resolve()
    source_status = (PROJECT_ROOT / Path(str(source.get("status_path", "")))).resolve()
    source_manifest = (PROJECT_ROOT / Path(str(source.get("run_manifest_path", "")))).resolve()
    source_core = (PROJECT_ROOT / Path(str(manifest.get("locked_core", {}).get("path", "")))).resolve()
    audit.require(source_catalogs == expected_catalogs.resolve(), "Source catalogs path mismatch")
    audit.require(source_terminal == expected_terminal.resolve(), "Source terminal path mismatch")
    audit.require(source_status == expected_status.resolve(), "Source status path mismatch")
    audit.require(source_manifest == expected_source_manifest.resolve(), "Source manifest path mismatch")
    audit.require(source_core == expected_core.resolve(), "Source locked core path mismatch")
    audit.require(manifest.get("locked_core", {}).get("sha256") == REQUIRED_CORE_SHA256, "Core allowlist mismatch")
    audit.require(sha256_file(source_catalogs) == CANONICAL_CATALOGS_SHA256, "Canonical source catalogs changed")
    audit.require(sha256_file(source_terminal) == CANONICAL_TERMINAL_SHA256, "Canonical source terminal changed")
    audit.require(sha256_file(source_status) == CANONICAL_STATUS_SHA256, "Canonical source status changed")
    audit.require(sha256_file(source_manifest) == SOURCE_RUN_MANIFEST_SHA256, "Canonical source manifest changed")
    audit.require(sha256_file(source_core) == REQUIRED_CORE_SHA256, "Verified source core changed")
    source_status_payload = json.loads(source_status.read_text(encoding="utf-8"))
    audit.require(source_status_payload.get("status") == "completed_unverified", "Canonical data status mismatch")
    audit.require(source_status_payload.get("mode") == "canonical", "Canonical data mode mismatch")
    audit.require(source_status_payload.get("campaign_id") == SOURCE_CAMPAIGN_NAME, "Canonical data campaign mismatch")
    audit.require(source_status_payload.get("terminal_sha256") == CANONICAL_TERMINAL_SHA256, "Canonical terminal binding mismatch")
    audit.require(source_status_payload.get("run_manifest_sha256") == SOURCE_RUN_MANIFEST_SHA256, "Canonical manifest binding mismatch")
    audit.require(source_status_payload.get("runs_completed") == 50, "Canonical run completion mismatch")

    audit.require(evidence.get("kind") == "distinct_independent_canonical_overlay", "Evidence kind mismatch")
    audit.require(evidence.get("campaign") == EVIDENCE_CAMPAIGN_NAME, "Evidence campaign mismatch")
    audit.require(evidence.get("source_campaign") == SOURCE_CAMPAIGN_NAME, "Evidence source campaign mismatch")
    audit.require(evidence.get("full_sha256") == OVERLAY_FULL_SHA256, "Overlay FULL allowlist mismatch")
    audit.require(evidence.get("status_sha256") == OVERLAY_STATUS_SHA256, "Overlay status allowlist mismatch")
    audit.require(evidence.get("run_manifest_sha256") == OVERLAY_RUN_MANIFEST_SHA256, "Overlay manifest allowlist mismatch")
    overlay_full = (PROJECT_ROOT / Path(str(evidence.get("full_path", "")))).resolve()
    overlay_status = (PROJECT_ROOT / Path(str(evidence.get("status_path", "")))).resolve()
    overlay_manifest = (PROJECT_ROOT / Path(str(evidence.get("run_manifest_path", "")))).resolve()
    audit.require(overlay_full == expected_overlay_full.resolve(), "Overlay FULL path mismatch")
    audit.require(overlay_status == expected_overlay_status.resolve(), "Overlay status path mismatch")
    audit.require(overlay_manifest == expected_overlay_manifest.resolve(), "Overlay manifest path mismatch")
    audit.require(sha256_file(overlay_full) == OVERLAY_FULL_SHA256, "Overlay FULL changed")
    audit.require(sha256_file(overlay_status) == OVERLAY_STATUS_SHA256, "Overlay status changed")
    audit.require(sha256_file(overlay_manifest) == OVERLAY_RUN_MANIFEST_SHA256, "Overlay manifest changed")
    overlay_status_payload = json.loads(overlay_status.read_text(encoding="utf-8"))
    audit.require(overlay_status_payload.get("status") == "completed_verified", "Overlay status not completed_verified")
    audit.require(overlay_status_payload.get("source_campaign_id") == SOURCE_CAMPAIGN_NAME, "Overlay status source mismatch")
    audit.require(overlay_status_payload.get("full_verification_sha256") == OVERLAY_FULL_SHA256, "Overlay status FULL binding mismatch")
    overlay_full_payload = json.loads(overlay_full.read_text(encoding="utf-8"))
    require_verified_source_terminal(audit, overlay_full_payload)
    audit.require(overlay_full_payload.get("campaign_id") == EVIDENCE_CAMPAIGN_NAME, "Overlay FULL campaign mismatch")
    audit.require(overlay_full_payload.get("source_campaign_id") == SOURCE_CAMPAIGN_NAME, "Overlay FULL source mismatch")
    audit.require(
        overlay_full_payload.get("overlay_contract", {}).get("overlay_manifest_sha256") == OVERLAY_RUN_MANIFEST_SHA256,
        "Overlay FULL manifest binding mismatch",
    )

    source_lines = [line for line in source_catalogs.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit.require(len(source_lines) == 50, "Canonical source must contain exactly 50 ordered runs")
    reconstructed_topsis = np.empty((50, 400), dtype=np.float64)
    reconstructed_q = np.empty((50, 5, 400), dtype=np.float64)
    reconstructed_expected = np.empty((50, 5, 7), dtype=np.int64)
    dataset_records = source.get("datasets", [])
    audit.require(len(dataset_records) == 50, "Fixture dataset lineage count mismatch")
    expected_accuracy: list[dict[str, Any]] = []
    for run_index, line in enumerate(source_lines):
        payload = json.loads(line)
        audit.require(payload.get("run_index") == run_index, "Canonical source run order mismatch")
        audit.require(payload.get("campaign_id") == SOURCE_CAMPAIGN_NAME, "Canonical source campaign mismatch")
        audit.require(PRIMARY_ARM_ID in payload.get("arms", {}), "Canonical primary arm missing")
        dataset_path = (expected_source_root / "inputs" / Path(str(payload["dataset_path"]))).resolve()
        dataset_sha = sha256_file(dataset_path)
        audit.require(dataset_sha == payload.get("dataset_sha256"), "Canonical catalog input hash mismatch")
        audit.require(dataset_records[run_index] == {
            "run_index": run_index,
            "path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "sha256": dataset_sha,
        }, "Fixture dataset lineage record mismatch")
        run_topsis = independent_topsis(pd.read_csv(dataset_path))
        reconstructed_topsis[run_index] = run_topsis
        arm = payload["arms"][PRIMARY_ARM_ID]
        audit.require(arm.get("arm", {}).get("arm_id") == PRIMARY_ARM_ID, "Canonical arm metadata mismatch")
        profiles = arm.get("profiles", [])
        audit.require([item.get("profile_name") for item in profiles] == PROFILE_ORDER, "Canonical profile order mismatch")
        for profile_index, profile in enumerate(profiles):
            q = np.asarray(profile.get("q_scores"), dtype=np.float64)
            expected = np.asarray(profile.get("final_rankings", {}).get("hybrid"), dtype=np.int64)
            audit.require(q.shape == (400,), "Canonical Q shape mismatch")
            audit.require(expected.shape == (7,), "Canonical expected top-7 shape mismatch")
            actual = independent_top7(q, run_topsis)
            audit.require(actual == [int(x) for x in expected], "Independent canonical top-7 mismatch")
            reconstructed_q[run_index, profile_index] = q
            reconstructed_expected[run_index, profile_index] = expected
            expected_accuracy.append({
                "pair_index": run_index * 5 + profile_index,
                "run_index": run_index,
                "profile_index": profile_index,
                "profile_name": PROFILE_ORDER[profile_index],
                "expected_top7": [int(x) for x in expected],
                "actual_top7": actual,
                "exact_match": True,
            })

    with np.load(fixture_path, allow_pickle=False) as data:
        audit.require(set(data.files) == {"topsis", "q_scores", "expected_top7"}, "Fixture v2 NPZ keys mismatch")
        fixture_arrays = {
            "topsis": np.asarray(data["topsis"]),
            "q_scores": np.asarray(data["q_scores"]),
            "expected_top7": np.asarray(data["expected_top7"]),
        }
    reconstructed = {
        "topsis": reconstructed_topsis,
        "q_scores": reconstructed_q,
        "expected_top7": reconstructed_expected,
    }
    expected_shapes = {"topsis": (50, 400), "q_scores": (50, 5, 400), "expected_top7": (50, 5, 7)}
    expected_dtypes = {"topsis": np.dtype("<f8"), "q_scores": np.dtype("<f8"), "expected_top7": np.dtype("<i8")}
    for name, array in fixture_arrays.items():
        audit.require(array.shape == expected_shapes[name], f"Fixture {name} shape mismatch")
        audit.require(array.dtype == expected_dtypes[name], f"Fixture {name} dtype mismatch")
        audit.require(typed_array_contract(array) == manifest.get("arrays", {}).get(name), f"Fixture {name} typed hash mismatch")
        audit.require(np.array_equal(array, reconstructed[name]), f"Fixture {name} canonical lineage mismatch")
    audit.require(len(expected_accuracy) == PAIR_COUNT, "Fixture accuracy coverage must be 250 pairs")
    audit.require(manifest.get("pair_schedule") == {
        "pair_count": 250,
        "order": "run_index major, PROFILE_ORDER minor",
        "canonical_calls_per_pair_per_block": 20,
        "canonical_calls_per_pair_total": 1200,
    }, "Fixture pair schedule mismatch")
    return manifest, expected_accuracy


def runtime_contract_ok(environment: dict[str, Any]) -> bool:
    runtime_pre = environment["runtime_pre"]
    runtime_post = environment["runtime_post"]
    threadpool_pre = environment["threadpool_pre"]
    threadpool_post = environment["threadpool_post"]
    threads_pre = environment["thread_environment"]["forced_before_numpy_import"]
    threads_post = environment["thread_environment"]["post_timing_values"]
    snapshot_ok = lambda item: item.get("logical_processors") == 1 and item.get("priority_name") in {"Normal", "Normal/default"}
    pool_ok = lambda item: item.get("all_reported_threads_one") is True if item.get("available") else True
    return bool(
        snapshot_ok(runtime_pre)
        and snapshot_ok(runtime_post)
        and runtime_pre.get("priority_class") == runtime_post.get("priority_class")
        and runtime_pre.get("affinity_mask") == runtime_post.get("affinity_mask")
        and pool_ok(threadpool_pre)
        and pool_ok(threadpool_post)
        and all(value == "1" for value in threads_pre.values())
        and all(value == "1" for value in threads_post.values())
    )


def expected_producer_executable() -> Path:
    return (
        PROJECT_ROOT
        / "experiments"
        / "_runtime"
        / "hre_submission_py312_numpy1260_pandas223"
        / "Scripts"
        / "python.exe"
    ).resolve()


def producer_environment_record_ok(environment: dict[str, Any]) -> bool:
    python = environment.get("python", {})
    libraries = environment.get("libraries", {})
    try:
        executable_ok = Path(str(python.get("executable", ""))).resolve() == expected_producer_executable()
    except Exception:
        executable_ok = False
    return bool(
        python.get("version_info") == EXPECTED_PYTHON_VERSION
        and python.get("implementation") == EXPECTED_PYTHON_IMPLEMENTATION
        and executable_ok
        and libraries.get("numpy") == EXPECTED_NUMPY_VERSION
    )


def require_current_producer_environment(audit: Audit) -> None:
    audit.require(list(sys.version_info[:3]) == EXPECTED_PYTHON_VERSION, "Verifier requires exact Python 3.12.12")
    audit.require(platform.python_implementation() == EXPECTED_PYTHON_IMPLEMENTATION, "Verifier requires CPython")
    audit.require(Path(sys.executable).resolve() == expected_producer_executable(), "Verifier requires exact producer venv executable")
    audit.require(np.__version__ == EXPECTED_NUMPY_VERSION, "Verifier requires exact NumPy 1.26.0")


def timer_record_ok(timer: dict[str, Any]) -> bool:
    return bool(
        timer.get("name") == "perf_counter_ns"
        and timer.get("implementation") == EXPECTED_TIMER_IMPLEMENTATION
        and timer.get("monotonic") is True
        and float(timer.get("resolution_seconds", 0)) > 0
    )


def validate_run_manifest(audit: Audit, result: dict[str, Any]) -> dict[str, Any]:
    declared = result.get("run_manifest", {})
    manifest_path = (PROJECT_ROOT / Path(str(declared.get("path", "")))).resolve()
    expected_manifest_path = (CAMPAIGN_ROOT / "RUN_MANIFEST.json").resolve()
    audit.require(manifest_path == expected_manifest_path, "Locked run manifest path mismatch")
    audit.require(sha256_file(manifest_path) == declared.get("sha256"), "Locked run manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit.require(manifest.get("schema_version") == "hre.latency.run_manifest.v1", "Locked manifest schema mismatch")
    audit.require(manifest.get("status") == "LOCKED", "Locked manifest status mismatch")
    audit.require(declared.get("status") == manifest.get("status"), "Result locked manifest status mismatch")
    audit.require(manifest.get("campaign_id") == CAMPAIGN_ROOT.name, "Locked manifest campaign mismatch")
    audit.require(manifest.get("operation_id") == "HRE_R1_LATENCY_CANONICAL_LOCK_20260713_CODEX_20", "Lock operation mismatch")
    audit.require(manifest.get("execution_policy") == {
        "allowed_modes": ["smoke", "canonical"],
        "canonical_launch_authorized": True,
        "canonical_timing_executed_at_lock": False,
        "authorization_operation_id": "HRE_R1_LATENCY_CANONICAL_LOCK_20260713_CODEX_20",
    }, "Locked execution policy mismatch")
    audit.require(manifest.get("environment") == {
        "python_version": "3.12.12",
        "python_implementation": "CPython",
        "python_executable": str(expected_producer_executable().relative_to(PROJECT_ROOT)),
        "numpy_version": "1.26.0",
        "timer_name": "perf_counter_ns",
        "timer_implementation": "QueryPerformanceCounter()",
    }, "Locked environment mismatch")
    audit.require(manifest.get("source_bindings") == {
        "canonical_main_catalogs_sha256": CANONICAL_CATALOGS_SHA256,
        "canonical_terminal_sha256": CANONICAL_TERMINAL_SHA256,
        "canonical_status_sha256": CANONICAL_STATUS_SHA256,
        "source_run_manifest_sha256": SOURCE_RUN_MANIFEST_SHA256,
        "overlay_full_sha256": OVERLAY_FULL_SHA256,
        "overlay_status_sha256": OVERLAY_STATUS_SHA256,
        "overlay_run_manifest_sha256": OVERLAY_RUN_MANIFEST_SHA256,
    }, "Locked source/evidence bindings mismatch")
    expected_paths = {
        "fixture_builder": CAMPAIGN_ROOT / "src" / "prepare_verified_fixture.py",
        "lock_script": CAMPAIGN_ROOT / "src" / "lock_campaign.py",
        "runner": CAMPAIGN_ROOT / "src" / "latency_benchmark.py",
        "verifier": CAMPAIGN_ROOT / "verify_latency.py",
        "protocol": CAMPAIGN_ROOT / "PROTOCOL_LOCK.md",
        "fixture_manifest": CAMPAIGN_ROOT / "inputs" / "fixture_manifest.json",
        "fixture_vectors": CAMPAIGN_ROOT / "inputs" / "verified_vectors.npz",
        "test_latency_contract": CAMPAIGN_ROOT / "tests" / "test_latency_contract.py",
        "test_verify_latency": CAMPAIGN_ROOT / "tests" / "test_verify_latency.py",
    }
    files = manifest.get("lock_files", {})
    audit.require(set(files) == set(expected_paths), "Locked manifest file allowlist mismatch")
    for key, expected_path in expected_paths.items():
        resolved = (PROJECT_ROOT / Path(str(files[key].get("path", "")))).resolve()
        audit.require(resolved == expected_path.resolve(), f"Locked manifest {key} path mismatch")
        audit.require(sha256_file(resolved) == files[key].get("sha256"), f"Locked manifest {key} hash mismatch")
    runner = result.get("runner", {})
    runner_path = (PROJECT_ROOT / Path(str(runner.get("path", "")))).resolve()
    audit.require(runner_path == expected_paths["runner"].resolve(), "Result runner path is not exact locked runner")
    audit.require(runner.get("sha256") == files["runner"]["sha256"], "Result runner hash is not manifest-bound")
    return manifest


def validate_result_header(audit: Audit, result: dict[str, Any]) -> None:
    audit.require(result.get("schema_version") == "hre.latency_result.v1", "Result schema mismatch")
    audit.require(result.get("campaign_id") == CAMPAIGN_ROOT.name, "Result campaign mismatch")
    audit.require(result.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY, "Result claim boundary mismatch")
    audit.require(result.get("batch_size") == 1, "Result batch size mismatch")


def validate_artifact_declaration(
    audit: Audit,
    output_dir: Path,
    declaration: dict[str, Any],
    expected_basename: str,
    array: np.ndarray,
) -> Path:
    declared_path = Path(str(declaration.get("path", "")))
    audit.require(declared_path.name == expected_basename and str(declared_path) == expected_basename, f"{expected_basename} path must be a basename")
    resolved = (output_dir / declared_path).resolve()
    audit.require(resolved.parent == output_dir.resolve(), f"{expected_basename} path escapes output directory")
    audit.require(declaration.get("dtype") == str(array.dtype), f"{expected_basename} declared dtype mismatch")
    audit.require(declaration.get("shape") == list(array.shape), f"{expected_basename} declared shape mismatch")
    return resolved


def recomputed_retention_gate(
    mode: str,
    primary_condition: bool,
    accuracy_ok: bool,
    runtime_ok: bool,
) -> dict[str, Any]:
    canonical_condition = bool(
        mode == "canonical" and primary_condition and accuracy_ok and runtime_ok
    )
    if mode == "canonical":
        status = "PASS_RETAIN_CACHED_PATH_CLAIM" if canonical_condition else "FAIL_REMOVE_OR_NARROW_CLAIM"
    else:
        status = "SMOKE_ONLY_NOT_CLAIMABLE"
    return {
        "threshold_ns": THRESHOLD_NS,
        "status": status,
        "canonical_condition_met": canonical_condition,
        "accuracy_ok": bool(accuracy_ok),
        "runtime_ok": bool(runtime_ok),
        "verifier_pass_required": True,
    }


def verify(output_dir: Path, expected_mode: str) -> dict[str, Any]:
    audit = Audit()
    output_dir = output_dir.resolve()
    reject_preexisting_verification_artifacts(output_dir)
    require_current_producer_environment(audit)
    status_path = output_dir / "status.json"
    result_path = output_dir / "latency_results.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_result_header(audit, result)
    audit.require(status.get("status") == "completed_unverified", "Runner is not completed_unverified")
    audit.require(result.get("status") == "completed_unverified", "Result is not completed_unverified")
    audit.require(status.get("terminal_path") == result_path.name, "Runner terminal path mismatch")
    audit.require(status.get("terminal_sha256") == sha256_file(result_path), "Runner status does not bind terminal result hash")
    audit.require(status.get("mode") == expected_mode == result.get("mode"), "Mode mismatch")
    config = result["config"]
    if expected_mode == "canonical":
        audit.require(config == CANONICAL_CONTRACT, "Canonical configuration drift")
    else:
        audit.require(all(int(config[key]) > 0 for key in CANONICAL_CONTRACT), "Invalid smoke configuration")
    audit.require(int(config["warmup"]) % PAIR_COUNT == 0, "Warmup must balance all 250 pairs")
    audit.require(int(config["calls_per_block"]) % PAIR_COUNT == 0, "Each block must balance all 250 pairs")
    expected_pair_schedule = {
        "pair_count": PAIR_COUNT,
        "order": "run_index major, PROFILE_ORDER minor",
        "calls_per_pair_per_block": int(config["calls_per_block"]) // PAIR_COUNT,
        "calls_per_pair_total": (
            int(config["passes"]) * int(config["blocks"]) * int(config["calls_per_block"]) // PAIR_COUNT
        ),
        "warmup_calls_per_pair": int(config["warmup"]) // PAIR_COUNT,
    }
    compare(audit, expected_pair_schedule, result.get("pair_schedule"), "pair_schedule")
    if expected_mode == "canonical":
        audit.require(expected_pair_schedule["calls_per_pair_per_block"] == 20, "Canonical pair/block count mismatch")
        audit.require(expected_pair_schedule["calls_per_pair_total"] == 1200, "Canonical total/pair count mismatch")

    run_manifest = validate_run_manifest(audit, result)
    runner_path = (CAMPAIGN_ROOT / "src" / "latency_benchmark.py").resolve()
    runner_source = runner_path.read_text(encoding="utf-8")
    numpy_import_at = runner_source.index("import numpy as np")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        audit.require(runner_source.index(f'"{key}"') < numpy_import_at, f"{key} is not declared before NumPy import")

    manifest_path = PROJECT_ROOT / Path(result["fixture"]["manifest_path"])
    fixture_path = PROJECT_ROOT / Path(result["fixture"]["fixture_path"])
    core_path = PROJECT_ROOT / Path(result["fixture"]["locked_core_path"])
    audit.require(sha256_file(manifest_path) == result["fixture"]["manifest_sha256"], "Fixture manifest hash mismatch")
    audit.require(sha256_file(fixture_path) == result["fixture"]["fixture_sha256"], "Fixture NPZ hash mismatch")
    audit.require(sha256_file(core_path) == result["fixture"]["locked_core_sha256"] == REQUIRED_CORE_SHA256, "Core hash mismatch")
    manifest, expected_accuracy = validate_fixture_lineage(audit, manifest_path, fixture_path)
    source = manifest["data_source"]
    evidence = manifest["verification_evidence"]
    source_catalogs = PROJECT_ROOT / Path(source["main_catalogs_path"])
    source_terminal = PROJECT_ROOT / Path(source["terminal_path"])
    source_status = PROJECT_ROOT / Path(source["status_path"])
    source_run_manifest = PROJECT_ROOT / Path(source["run_manifest_path"])
    overlay_full = PROJECT_ROOT / Path(evidence["full_path"])
    overlay_status = PROJECT_ROOT / Path(evidence["status_path"])
    overlay_run_manifest = PROJECT_ROOT / Path(evidence["run_manifest_path"])
    expected_accuracy_payload = {
        "before": expected_accuracy,
        "after": expected_accuracy,
        "all_exact": True,
    }
    compare(audit, expected_accuracy_payload, result["accuracy"], "accuracy")
    accuracy_ok = True

    raw_path = output_dir / "raw_durations_ns.npy"
    overhead_path = output_dir / "timer_overhead_ns.npy"
    audit.require(sha256_file(raw_path) == result["raw_artifact"]["sha256"], "Raw duration hash mismatch")
    audit.require(sha256_file(overhead_path) == result["timer_overhead"]["raw_artifact"]["sha256"], "Timer overhead hash mismatch")
    raw = np.load(raw_path, allow_pickle=False)
    overhead = np.load(overhead_path, allow_pickle=False)
    validate_artifact_declaration(audit, output_dir, result["raw_artifact"], "raw_durations_ns.npy", raw)
    validate_artifact_declaration(audit, output_dir, result["timer_overhead"]["raw_artifact"], "timer_overhead_ns.npy", overhead)
    expected_shape = (int(config["passes"]), int(config["blocks"]), int(config["calls_per_block"]))
    audit.require(raw.dtype == np.uint64 and raw.shape == expected_shape, "Raw array dtype/shape mismatch")
    audit.require(overhead.dtype == np.uint64 and overhead.shape == (int(config["timer_overhead_samples"]),), "Timer overhead dtype/shape mismatch")
    audit.require(bool(np.all(raw > 0)), "Timed durations must all be positive")
    audit.require(bool(np.all(overhead > 0)), "Timer-overhead durations must all be positive")
    recomputed = analyze_durations(raw, int(config["calls_per_block"]), int(config["bootstrap_replicates"]))
    compare(audit, recomputed, result["analysis"], "analysis")
    expected_overhead = {"measurement": "back-to-back perf_counter_ns calls", **summary(overhead), "raw_artifact": result["timer_overhead"]["raw_artifact"]}
    compare(audit, expected_overhead, result["timer_overhead"], "timer_overhead")

    timer = result["environment"]["timer"]
    audit.require(timer_record_ok(timer), "Exact QueryPerformanceCounter timer contract failed")
    audit.require(producer_environment_record_ok(result["environment"]), "Recorded producer environment contract failed")
    runtime_ok = runtime_contract_ok(result["environment"])
    audit.require(runtime_ok, "Runtime control contract failed")
    expected_gate = recomputed_retention_gate(
        expected_mode,
        bool(recomputed["threshold_diagnostics"]["primary_condition_met"]),
        accuracy_ok,
        runtime_ok,
    )
    compare(audit, expected_gate, result["retention_gate"], "retention_gate")

    report = {
        "schema_version": "hre.latency_full_verification.v2",
        "campaign_id": CAMPAIGN_ROOT.name,
        "mode": expected_mode,
        "status": "completed_verified",
        "verdict": "PASS",
        "verified_at": local_now(),
        "checks_executed": audit.checks,
        "input_hashes": {
            "fixture_manifest.json": sha256_file(manifest_path),
            "verified_vectors.npz": sha256_file(fixture_path),
            "locked_core.py": sha256_file(core_path),
            "canonical_main_catalogs.jsonl": sha256_file(source_catalogs),
            "canonical_main_results.json": sha256_file(source_terminal),
            "canonical_status.json": sha256_file(source_status),
            "canonical_source_RUN_MANIFEST.json": sha256_file(source_run_manifest),
            "canonical_overlay_FULL_VERIFICATION.json": sha256_file(overlay_full),
            "canonical_overlay_verification_status.json": sha256_file(overlay_status),
            "canonical_overlay_RUN_MANIFEST.json": sha256_file(overlay_run_manifest),
            "RUN_MANIFEST.json": sha256_file(CAMPAIGN_ROOT / "RUN_MANIFEST.json"),
        },
        "output_hashes": {
            "latency_results.json": sha256_file(result_path),
            "raw_durations_ns.npy": sha256_file(raw_path),
            "timer_overhead_ns.npy": sha256_file(overhead_path),
        },
        "gate": expected_gate,
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "independence": "Verifier does not import the runner or locked core; score fusion/statistics/gates are independently implemented.",
    }
    atomic_json(output_dir / "FULL_VERIFICATION.json", report)
    terminal_sha256 = sha256_file(output_dir / "FULL_VERIFICATION.json")
    atomic_json(
        output_dir / "verification_status.json",
        {
            "schema_version": "hre.latency_verification_status.v1",
            "status": "completed_verified",
            "verdict": "PASS",
            "mode": expected_mode,
            "verified_at": report["verified_at"],
            "checks_executed": audit.checks,
            "terminal_path": "FULL_VERIFICATION.json",
            "terminal_sha256": terminal_sha256,
        },
    )
    atomic_json(
        status_path,
        {
            "schema_version": "hre.latency_status.v1",
            "status": "completed_verified",
            "mode": expected_mode,
            "verified_at": report["verified_at"],
            "progress_percent": 100.0,
            "terminal_path": "FULL_VERIFICATION.json",
            "terminal_sha256": terminal_sha256,
            "verdict": "PASS",
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "canonical"), required=True)
    args = parser.parse_args()
    try:
        report = verify(args.output_dir, args.mode)
        print(json.dumps({"status": report["status"], "verdict": report["verdict"], "checks": report["checks_executed"]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        quarantined = quarantine_verification_artifacts(args.output_dir, "FAILED_ATTEMPT")
        failure_path = args.output_dir / "VERIFICATION_FAILURE.json"
        atomic_json(
            failure_path,
            {
                "schema_version": "hre.latency_verification_failure.v1",
                "status": "failed_closed",
                "verdict": "FAIL",
                "mode": args.mode,
                "failed_at": local_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "quarantined_artifacts": quarantined,
            },
        )
        terminal_sha256 = sha256_file(failure_path)
        failure_status = {
            "schema_version": "hre.latency_verification_status.v1",
            "status": "failed_closed",
            "verdict": "FAIL",
            "mode": args.mode,
            "failed_at": local_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "terminal_path": failure_path.name,
            "terminal_sha256": terminal_sha256,
        }
        atomic_json(args.output_dir / "verification_status.json", failure_status)
        top_failure_status = dict(failure_status)
        top_failure_status["schema_version"] = "hre.latency_status.v1"
        atomic_json(args.output_dir / "status.json", top_failure_status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
