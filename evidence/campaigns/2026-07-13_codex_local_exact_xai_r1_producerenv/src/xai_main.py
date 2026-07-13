"""Exact mechanism-aligned XAI runner for the verified corrected bridge arm.

Launch with ``python -u``.  Console output is deliberately limited to progress
telemetry; scientific arrays are written only to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy


CAMPAIGN_ID = "2026-07-13_codex_local_exact_xai_r1_producerenv"
SOURCE_CAMPAIGN_ID = "2026-07-12_codex_local_same_target_bridge_r01_producerenv"
SOURCE_EVIDENCE_CAMPAIGN_ID = "2026-07-13_codex_static_verifier_topsis_gatefix"
PRIMARY_ARM_ID = "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix"
PROFILE_ORDER = ("budget", "quality_seeker", "explorer", "loyal", "balanced")
CRITERIA = ("price_pct", "quality_pct", "popularity_pct", "rating_pct")
FORBIDDEN_INPUT_KEY_PARTS = (
    "ground_truth",
    "target",
    "label",
    "metric",
    "relevance",
    "ndcg",
    "f1",
    "gt_",
    "_gt",
)
FORBIDDEN_INPUT_KEYS_EXACT = ("gt",)
TOL = 1e-12
EPISODES = 30_000
SOURCE_RUN_MANIFEST_SHA256 = "0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222"
SOURCE_CORE_SHA256 = "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f"
SOURCE_EVIDENCE_RUN_MANIFEST_SHA256 = "ab29bb4782daf44c856f79ef8ad83559d4b59b637409476a43ed02dac8672ea7"
CANONICAL_DATA_HASHES = {
    "main_catalogs.jsonl": "803eebfe09be8d62b5f446955f2106fe7ef8b220a979b68c9f9d71acb4827ecd",
    "main_results.json": "48677825f4446e2df427a0940dc8c0947b99aef1373ca5dfb6933f35728ad861",
    "status.json": "6470b05e83827637e34983511359a9eda24d26d0977d7976eb517dfe156ec2f3",
}
CANONICAL_EVIDENCE_HASHES = {
    "FULL_VERIFICATION.json": "a3112da73a28e9c68f3148b0a8668cc834472c7d5c765870ad1fed25e09fcd97",
    "verification_status.json": "2513c8922a38e5f1a413d081b607225a55a620d7ad1cf540db208c48373b811f",
}
EXPECTED_ENVIRONMENT = {"numpy": "1.26.0", "pandas": "2.2.3", "scipy": "1.16.3"}
EXPECTED_COLUMNS = (
    "product_id",
    "product_name",
    "category_raw",
    "category",
    "brand_label",
    "brand",
    "price",
    "actual_price",
    "discount_pct",
    "rating",
    "rating_count",
    "inferred_reviewer_count",
    "review_text_richness",
    "quality",
    "popularity",
    "recency",
    "price_pct",
    "quality_pct",
    "popularity_pct",
    "rating_pct",
    "recency_pct",
)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant rejected: {value}")


def environment_versions() -> dict[str, str]:
    return {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}


def ensure_contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes allowed root: {resolved}") from exc
    return resolved


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_finite(payload, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            assert_finite(row, str(path))
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"), parse_constant=reject_json_constant
    )
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    assert_finite(value, str(path))
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=reject_json_constant)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            assert_finite(value, f"{path}:{line_number}")
            rows.append(value)
    return rows


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("exact_xai_frozen_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_finite(value: Any, where: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite value at {where}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            assert_finite(child, f"{where}[{idx}]")


def assert_label_free_keys(value: Any, where: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_INPUT_KEYS_EXACT or any(
                part in lowered for part in FORBIDDEN_INPUT_KEY_PARTS
            ):
                raise ValueError(f"Forbidden evaluation/label key at {where}.{key}")
            assert_label_free_keys(child, f"{where}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            assert_label_free_keys(child, f"{where}[{idx}]")


def prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_source_dataset(source_campaign: Path, relative_value: str) -> Path:
    relative = Path(relative_value.replace("/", "\\"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source dataset path: {relative_value}")
    allowed = source_campaign / "inputs" / "data" / "processed" / "bootstrap_catalogs"
    return ensure_contained(source_campaign / "inputs" / relative, allowed, "dataset")


def validate_run_manifest(campaign_root: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    manifest_path = campaign_root / "RUN_MANIFEST.json"
    manifest = read_json(manifest_path)
    expected_keys = {
        "schema_version",
        "campaign_id",
        "status",
        "operation_id",
        "canonical_authorized",
        "execution_policy",
        "environment",
        "source_bindings",
        "files",
        "full_verification_contract",
    }
    if set(manifest) != expected_keys:
        raise RuntimeError("RUN manifest schema mismatch")
    if (
        manifest["schema_version"] != "exact_xai.run_manifest.v1"
        or manifest["campaign_id"] != CAMPAIGN_ID
        or manifest["operation_id"] != "HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19"
        or manifest["status"] != "LOCKED"
        or manifest["canonical_authorized"] is not True
        or manifest["environment"] != EXPECTED_ENVIRONMENT
    ):
        raise RuntimeError("RUN manifest identity/status/environment mismatch")
    expected_policy = {
        "lifecycle_status": "LOCKED",
        "allowed_modes": ["smoke", "canonical"],
        "canonical_authorized": True,
        "authorization_operation_id": "HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19",
    }
    if manifest["execution_policy"] != expected_policy:
        raise RuntimeError("RUN manifest execution policy mismatch")
    expected_bindings = {
        "data": {
            "campaign_id": SOURCE_CAMPAIGN_ID,
            "canonical_root": "outputs/canonical_main",
            "run_manifest_sha256": SOURCE_RUN_MANIFEST_SHA256,
            "core_sha256": SOURCE_CORE_SHA256,
            "canonical_output_hashes": CANONICAL_DATA_HASHES,
        },
        "evidence": {
            "campaign_id": SOURCE_EVIDENCE_CAMPAIGN_ID,
            "canonical_root": "outputs/canonical_overlay",
            "run_manifest_sha256": SOURCE_EVIDENCE_RUN_MANIFEST_SHA256,
            "canonical_file_hashes": CANONICAL_EVIDENCE_HASHES,
        },
        "primary_arm_id": PRIMARY_ARM_ID,
        "canonical_shape": {"catalogs": 50, "profiles_per_catalog": 5},
    }
    if manifest["source_bindings"] != expected_bindings:
        raise RuntimeError("RUN manifest source bindings mismatch")
    expected_paths = {
        "PROTOCOL_LOCK.md",
        "src/xai_main.py",
        "verify_xai.py",
        "build_run_manifest.py",
        "tests/test_xai_contract.py",
        "tests/test_verify_xai.py",
    }
    rows = manifest["files"]
    if not isinstance(rows, list) or {row.get("path") for row in rows} != expected_paths:
        raise RuntimeError("RUN manifest file-set mismatch")
    hashes: dict[str, str] = {}
    for row in rows:
        if set(row) != {"path", "sha256"}:
            raise RuntimeError("RUN manifest file entry schema mismatch")
        relative = Path(str(row["path"]))
        candidate = ensure_contained(campaign_root / relative, campaign_root, "run-manifest file")
        actual = sha256_file(candidate)
        if actual != row["sha256"]:
            raise RuntimeError(f"RUN manifest hash mismatch: {relative}")
        hashes[str(row["path"])] = actual
    contract = manifest["full_verification_contract"]
    if contract != {
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
    }:
        raise RuntimeError("RUN manifest FULL contract mismatch")
    return sha256_file(manifest_path), hashes, expected_policy


def validate_source(
    source_data_campaign: Path,
    source_data_root: Path,
    source_evidence_campaign: Path,
    source_evidence_root: Path,
    mode: str,
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    if source_data_campaign.name != SOURCE_CAMPAIGN_ID:
        raise ValueError("Unexpected source data campaign")
    if source_evidence_campaign.name != SOURCE_EVIDENCE_CAMPAIGN_ID:
        raise ValueError("Unexpected source evidence campaign")
    expected_data_name = "canonical_main" if mode == "canonical" else "smoke"
    expected_evidence_name = "canonical_overlay" if mode == "canonical" else "smoke_overlay"
    for root, campaign, expected_name, label in (
        (source_data_root, source_data_campaign, expected_data_name, "data"),
        (source_evidence_root, source_evidence_campaign, expected_evidence_name, "evidence"),
    ):
        ensure_contained(root, campaign / "outputs", f"source {label} root")
        if root.parent.resolve() != (campaign / "outputs").resolve() or root.name != expected_name:
            raise ValueError(f"Source {label} root containment/name mismatch")
    if environment_versions() != EXPECTED_ENVIRONMENT:
        raise RuntimeError("Producer environment mismatch")

    data_manifest = source_data_campaign / "RUN_MANIFEST.json"
    evidence_manifest = source_evidence_campaign / "RUN_MANIFEST.json"
    if sha256_file(data_manifest) != SOURCE_RUN_MANIFEST_SHA256:
        raise RuntimeError("Source data RUN_MANIFEST hash mismatch")
    if sha256_file(evidence_manifest) != SOURCE_EVIDENCE_RUN_MANIFEST_SHA256:
        raise RuntimeError("Source evidence RUN_MANIFEST hash mismatch")
    evidence_manifest_payload = read_json(evidence_manifest)
    if (
        evidence_manifest_payload.get("campaign_id") != SOURCE_EVIDENCE_CAMPAIGN_ID
        or evidence_manifest_payload.get("status") != "LOCKED"
        or evidence_manifest_payload.get("source_campaign", {}).get("campaign_id") != SOURCE_CAMPAIGN_ID
        or evidence_manifest_payload.get("execution_policy", {}).get("canonical_authorized") is not True
    ):
        raise RuntimeError("Source evidence manifest contract mismatch")

    data_paths = {name: source_data_root / name for name in CANONICAL_DATA_HASHES}
    full_path = source_evidence_root / "FULL_VERIFICATION.json"
    evidence_status_path = source_evidence_root / "verification_status.json"
    for path in (*data_paths.values(), full_path, evidence_status_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    full = read_json(full_path)
    evidence_status = read_json(evidence_status_path)
    data_status = read_json(data_paths["status.json"])
    if mode == "canonical":
        for name, expected in CANONICAL_DATA_HASHES.items():
            if sha256_file(data_paths[name]) != expected:
                raise RuntimeError(f"Canonical data hash mismatch: {name}")
        for name, expected in CANONICAL_EVIDENCE_HASHES.items():
            if sha256_file(source_evidence_root / name) != expected:
                raise RuntimeError(f"Canonical evidence hash mismatch: {name}")

    expected_runs = 50 if mode == "canonical" else 1
    expected_cells = expected_runs * 100
    if (
        full.get("campaign_id") != SOURCE_EVIDENCE_CAMPAIGN_ID
        or full.get("source_campaign_id") != SOURCE_CAMPAIGN_ID
        or full.get("mode") != mode
        or full.get("status") != "completed_verified"
        or full.get("verdict") != "PASS"
        or full.get("run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
        or any(gate.get("status") != "PASS" for gate in full.get("gates", []))
    ):
        raise RuntimeError("Source overlay FULL identity/verdict mismatch")
    if set(full.get("output_hashes", {})) != set(data_paths):
        raise RuntimeError("Source overlay output-hash schema mismatch")
    for name, path in data_paths.items():
        if full["output_hashes"][name] != sha256_file(path):
            raise RuntimeError(f"Source overlay/data hash mismatch: {name}")
    overlay_contract = full.get("overlay_contract", {})
    if (
        overlay_contract.get("overlay_manifest_sha256") != SOURCE_EVIDENCE_RUN_MANIFEST_SHA256
        or overlay_contract.get("source_run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
        or overlay_contract.get("source_canonical_checkpoint_sha256") != CANONICAL_DATA_HASHES["main_catalogs.jsonl"]
        or overlay_contract.get("source_canonical_terminal_sha256") != CANONICAL_DATA_HASHES["main_results.json"]
        or overlay_contract.get("source_canonical_runner_status_sha256") != CANONICAL_DATA_HASHES["status.json"]
    ):
        raise RuntimeError("Source overlay contract mismatch")
    if full.get("producer_provenance", {}).get("environment") != EXPECTED_ENVIRONMENT:
        raise RuntimeError("Source producer environment evidence mismatch")

    required_evidence_status = {
        "schema_version": "same_target_bridge.verification_status.v1",
        "campaign_id": SOURCE_EVIDENCE_CAMPAIGN_ID,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "mode": mode,
        "status": "completed_verified",
        "catalogs_completed": expected_runs,
        "catalogs_total": expected_runs,
        "cells_completed": expected_cells,
        "cells_total": expected_cells,
        "progress_percent": 100.0,
        "python_unbuffered_required": True,
        "scientific_values_exposed": False,
        "full_verification_path": "FULL_VERIFICATION.json",
        "full_verification_sha256": sha256_file(full_path),
    }
    for key, expected in required_evidence_status.items():
        if evidence_status.get(key) != expected:
            raise RuntimeError(f"Source evidence status mismatch: {key}")

    required_data_status = {
        "campaign_id": SOURCE_CAMPAIGN_ID,
        "mode": mode,
        "status": "completed_unverified",
        "progress_percent": 100.0,
        "runs_completed": expected_runs,
        "runs_total": expected_runs,
        "trajectories_completed": expected_cells,
        "trajectories_total": expected_cells,
        "run_manifest_sha256": SOURCE_RUN_MANIFEST_SHA256,
    }
    for key, expected in required_data_status.items():
        if data_status.get(key) != expected:
            raise RuntimeError(f"Source data status mismatch: {key}")
    source_terminal = ensure_contained(
        source_data_campaign / Path(str(data_status.get("terminal_path", ""))),
        source_data_root,
        "source data terminal",
    )
    if (
        source_terminal != data_paths["main_results.json"].resolve()
        or data_status.get("terminal_sha256") != sha256_file(source_terminal)
    ):
        raise RuntimeError("Source data terminal path/hash mismatch")

    core_path = source_data_campaign / "src" / "original_hybrid_core.py"
    if (
        full.get("input_hashes", {}).get("src/original_hybrid_core.py") != SOURCE_CORE_SHA256
        or sha256_file(core_path) != SOURCE_CORE_SHA256
    ):
        raise RuntimeError("Frozen producer core hash mismatch")
    rows = read_jsonl(data_paths["main_catalogs.jsonl"])
    if len(rows) != expected_runs or [int(row.get("run_index", -1)) for row in rows] != list(range(expected_runs)):
        raise RuntimeError("Source run count/order mismatch")
    return rows, load_module(core_path), {
        "data_campaign_id": SOURCE_CAMPAIGN_ID,
        "evidence_campaign_id": SOURCE_EVIDENCE_CAMPAIGN_ID,
        "data_main_catalogs_sha256": sha256_file(data_paths["main_catalogs.jsonl"]),
        "data_terminal_sha256": sha256_file(data_paths["main_results.json"]),
        "data_status_sha256": sha256_file(data_paths["status.json"]),
        "evidence_full_verification_sha256": sha256_file(full_path),
        "evidence_verification_status_sha256": sha256_file(evidence_status_path),
        "data_run_manifest_sha256": SOURCE_RUN_MANIFEST_SHA256,
        "evidence_run_manifest_sha256": SOURCE_EVIDENCE_RUN_MANIFEST_SHA256,
        "core_sha256": SOURCE_CORE_SHA256,
        "environment": EXPECTED_ENVIRONMENT,
    }


def extract_allowlisted_inputs(
    rows: list[dict[str, Any]], source_campaign: Path
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for row in rows:
        arm_bundle = row.get("arms", {}).get(PRIMARY_ARM_ID)
        if not isinstance(arm_bundle, dict):
            raise KeyError(f"Missing primary arm in run {row.get('run_index')}")
        arm = arm_bundle.get("arm")
        if arm != {
            "arm_id": PRIMARY_ARM_ID,
            "candidate": "full_catalog",
            "gt_bonus": 0.0,
            "reward_model": "component_continuous_fix",
            "role": "mandatory_factorial",
        }:
            raise RuntimeError("Primary arm definition mismatch")
        profiles = arm_bundle.get("profiles")
        if not isinstance(profiles, list) or [p.get("profile_name") for p in profiles] != list(PROFILE_ORDER):
            raise RuntimeError("Primary profile order/completeness mismatch")
        selected_profiles: list[dict[str, Any]] = []
        for profile in profiles:
            q = profile.get("q_scores")
            visits = profile.get("visits")
            if not isinstance(q, list) or len(q) != 400 or not np.all(np.isfinite(np.asarray(q, dtype=float))):
                raise RuntimeError("Invalid primary Q vector")
            if (
                not isinstance(visits, list)
                or len(visits) != 400
                or any(isinstance(x, bool) or int(x) != x or int(x) < 0 for x in visits)
                or sum(int(x) for x in visits) != EPISODES
            ):
                raise RuntimeError("Invalid primary visits vector")
            selected_profiles.append(
                {
                    "profile_name": str(profile["profile_name"]),
                    "q_scores": [float(x) for x in q],
                    "visits": [int(x) for x in visits],
                }
            )
        dataset_relative = str(row["dataset_path"]).replace("/", "\\")
        dataset_path = resolve_source_dataset(source_campaign, dataset_relative)
        if not dataset_path.is_file() or sha256_file(dataset_path) != str(row["dataset_sha256"]):
            raise RuntimeError("Catalog path/hash mismatch")
        selected = {
            "schema_version": "exact_xai.label_free_input.v1",
            "run_index": int(row["run_index"]),
            "run_seed": int(row["run_seed"]),
            "dataset_path": dataset_relative,
            "dataset_sha256": str(row["dataset_sha256"]),
            "profiles": selected_profiles,
        }
        selected["allowlisted_payload_sha256"] = canonical_sha256(selected)
        assert_label_free_keys(selected)
        assert_finite(selected)
        extracted.append(selected)
    return extracted


def minmax_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo + 1e-10)


def fixed_topsis_bundle(df: pd.DataFrame) -> dict[str, np.ndarray | float]:
    matrix = np.clip(df[list(CRITERIA)].to_numpy(dtype=float).copy(), 1e-10, None)
    proportions = np.clip(matrix / matrix.sum(axis=0, keepdims=True), 1e-10, 1.0)
    entropy = -np.sum(proportions * np.log(proportions), axis=0) / np.log(len(matrix))
    diversification = 1.0 - entropy
    weights = 0.10 + (diversification / diversification.sum()) * 0.60
    weights = np.clip(weights, 0.10, None)
    weights = weights / weights.sum()
    vector_norms = np.sqrt((matrix**2).sum(axis=0))
    vector_norms[vector_norms == 0] = 1.0
    weighted = (matrix / vector_norms) * weights
    ideal_plus = weighted.max(axis=0)
    ideal_minus = weighted.min(axis=0)
    d_plus = np.sqrt(((weighted - ideal_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - ideal_minus) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    denom[denom == 0] = 1e-10
    raw_scores = d_minus / denom
    return {
        "matrix": matrix,
        "weights": weights,
        "vector_norms": vector_norms,
        "ideal_plus": ideal_plus,
        "ideal_minus": ideal_minus,
        "reference": np.median(matrix, axis=0),
        "raw_scores": raw_scores,
        "normalized_scores": minmax_norm(raw_scores),
        "score_min": float(np.min(raw_scores)),
        "score_max": float(np.max(raw_scores)),
    }


def fixed_coalition_value(item: np.ndarray, mask: int, bundle: Mapping[str, Any]) -> float:
    reference = np.asarray(bundle["reference"], dtype=float)
    present = np.asarray([(mask >> idx) & 1 for idx in range(4)], dtype=bool)
    value = np.where(present, item, reference)
    weighted = (value / np.asarray(bundle["vector_norms"], dtype=float)) * np.asarray(bundle["weights"], dtype=float)
    d_plus = float(np.sqrt(np.sum((weighted - np.asarray(bundle["ideal_plus"], dtype=float)) ** 2)))
    d_minus = float(np.sqrt(np.sum((weighted - np.asarray(bundle["ideal_minus"], dtype=float)) ** 2)))
    denom = d_plus + d_minus
    raw = d_minus / (denom if denom != 0.0 else 1e-10)
    lo = float(bundle["score_min"])
    hi = float(bundle["score_max"])
    if hi <= lo:
        return 0.5
    return float((raw - lo) / (hi - lo + 1e-10))


def exact_shapley_from_values(values: Mapping[int, float], n_players: int) -> np.ndarray:
    phi = np.zeros(n_players, dtype=float)
    factorial_n = math.factorial(n_players)
    for player in range(n_players):
        bit = 1 << player
        for mask in range(1 << n_players):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(n_players - size - 1) / factorial_n
            phi[player] += weight * (float(values[mask | bit]) - float(values[mask]))
    return phi


def exact_topsis_shapley(bundle: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(bundle["matrix"], dtype=float)
    baselines = np.zeros(len(matrix), dtype=float)
    shapley = np.zeros((len(matrix), 4), dtype=float)
    max_error = 0.0
    for item_idx, item in enumerate(matrix):
        values = {mask: fixed_coalition_value(item, mask, bundle) for mask in range(16)}
        phi = exact_shapley_from_values(values, 4)
        error = abs(values[0] + float(np.sum(phi)) - values[15])
        max_error = max(max_error, error)
        baselines[item_idx] = values[0]
        shapley[item_idx] = phi
    if max_error > TOL:
        raise AssertionError(f"TOPSIS Shapley efficiency error {max_error}")
    return baselines, shapley, max_error


def corrected_reward_probabilities(df: pd.DataFrame, profile: Mapping[str, Any], core: Any) -> tuple[np.ndarray, np.ndarray]:
    brand = np.asarray([profile["brand_pref"].get(value, 0.10) for value in df["brand"]], dtype=float)
    recency_weight = float(profile["recency_weight"])
    recency = df["recency_pct"].to_numpy(dtype=float) * recency_weight + (1.0 - recency_weight) * 0.5
    components = core.hidden_components(df, profile)
    category = np.asarray(components["cat_score"], dtype=float)
    price_fit = np.asarray(components["price_fit"], dtype=float)
    p_engage = np.clip(0.40 * brand + 0.35 * price_fit + 0.15 * category + 0.10 * recency, 0.0, 1.0)
    p_convert = np.clip(0.50 * brand + 0.30 * price_fit + 0.20 * category, 0.0, 1.0)
    return p_engage, p_convert


def replay_reward_components(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    run_seed: int,
    expected_q: np.ndarray,
    expected_visits: np.ndarray,
    core: Any,
) -> dict[str, Any]:
    profile = core.PROFILE_HIDDEN[profile_name]
    p_engage, p_convert = corrected_reward_probabilities(df, profile, core)
    pool = np.arange(len(df), dtype=int)
    q_total = np.zeros(len(df), dtype=float)
    q_base = np.zeros(len(df), dtype=float)
    q_engage = np.zeros(len(df), dtype=float)
    q_convert = np.zeros(len(df), dtype=float)
    visits = np.zeros(len(df), dtype=np.int32)
    action_rng = np.random.RandomState(run_seed + profile_idx * 13)
    reward_rng = np.random.RandomState(run_seed + profile_idx * 997)
    epsilon = 0.30
    action_hash = hashlib.sha256()
    event_hash = hashlib.sha256()
    engage_count = 0
    convert_count = 0
    for _episode in range(1, EPISODES + 1):
        if action_rng.random() < epsilon:
            action = int(action_rng.choice(pool))
        else:
            action = int(pool[np.argmax(q_total[pool])])
        engaged = bool(reward_rng.random() < p_engage[action])
        converted = False
        if engaged:
            engage_count += 1
            converted = bool(reward_rng.random() < p_convert[action])
            if converted:
                convert_count += 1
        components = (-0.02, 0.30 if engaged else 0.0, 1.00 if converted else 0.0)
        total_reward = sum(components)
        visits[action] += 1
        q_total[action] += 0.05 * (total_reward - q_total[action])
        q_base[action] += 0.05 * (components[0] - q_base[action])
        q_engage[action] += 0.05 * (components[1] - q_engage[action])
        q_convert[action] += 0.05 * (components[2] - q_convert[action])
        epsilon = max(0.05, epsilon * 0.9997)
        action_hash.update(struct.pack("<i", action))
        event_hash.update(bytes((int(engaged), int(converted))))
    component_sum = q_base + q_engage + q_convert
    component_error = float(np.max(np.abs(q_total - component_sum)))
    source_error = float(np.max(np.abs(q_total - expected_q)))
    if component_error > TOL or source_error > TOL or not np.array_equal(visits, expected_visits):
        raise AssertionError(
            f"Q replay gate failed profile={profile_name} component={component_error} source={source_error}"
        )
    return {
        "q_total": q_total,
        "q_base": q_base,
        "q_engage": q_engage,
        "q_convert": q_convert,
        "visits": visits,
        "component_error": component_error,
        "source_error": source_error,
        "action_trace_sha256": action_hash.hexdigest(),
        "reward_event_trace_sha256": event_hash.hexdigest(),
        "engage_count": engage_count,
        "convert_count": convert_count,
    }


def array_list(value: np.ndarray) -> list[Any]:
    if np.issubdtype(value.dtype, np.integer):
        return [int(x) for x in value.tolist()]
    if value.ndim == 1:
        return [float(x) for x in value.tolist()]
    return [[float(x) for x in row] for row in value.tolist()]


def validate_catalog_frame(df: pd.DataFrame) -> None:
    if tuple(str(column) for column in df.columns) != EXPECTED_COLUMNS:
        raise RuntimeError("Catalog column schema/order mismatch")
    if len(df) != 400:
        raise RuntimeError("Expected 400-item catalog")
    numeric_columns = (
        "price",
        "actual_price",
        "discount_pct",
        "rating",
        "rating_count",
        "inferred_reviewer_count",
        "review_text_richness",
        "quality",
        "popularity",
        "recency",
        "price_pct",
        "quality_pct",
        "popularity_pct",
        "rating_pct",
        "recency_pct",
    )
    numeric = df[list(numeric_columns)].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("Catalog contains nonfinite numeric values")
    if df[["product_id", "category", "brand"]].isna().any().any():
        raise RuntimeError("Catalog contains missing identity/category/proxy values")


def cq_affine_decomposition(replay: Mapping[str, Any]) -> dict[str, Any]:
    q = np.asarray(replay["q_total"], dtype=float)
    lo, hi = float(np.min(q)), float(np.max(q))
    if hi <= lo:
        reference = 0.25
        base = np.zeros_like(q)
        engage = np.zeros_like(q)
        convert = np.zeros_like(q)
        denominator = 0.0
        is_constant = True
    else:
        denominator = hi - lo + 1e-10
        reference = -0.50 * lo / denominator
        base = 0.50 * np.asarray(replay["q_base"], dtype=float) / denominator
        engage = 0.50 * np.asarray(replay["q_engage"], dtype=float) / denominator
        convert = 0.50 * np.asarray(replay["q_convert"], dtype=float) / denominator
        is_constant = False
    c_q = 0.50 * minmax_norm(q)
    reconstructed = reference + base + engage + convert
    error = float(np.max(np.abs(c_q - reconstructed)))
    if error > TOL:
        raise AssertionError(f"cQ affine decomposition error {error}")
    return {
        "c_q": c_q,
        "reference": float(reference),
        "base": base,
        "engage": engage,
        "convert": convert,
        "error": error,
        "q_min": lo,
        "q_max": hi,
        "denominator": float(denominator),
        "is_constant": is_constant,
    }


def build_catalog_attribution(row: Mapping[str, Any], source_campaign: Path, core: Any) -> dict[str, Any]:
    dataset = resolve_source_dataset(source_campaign, str(row["dataset_path"]))
    if sha256_file(dataset) != row["dataset_sha256"]:
        raise RuntimeError("Catalog changed after extraction")
    df = pd.read_csv(dataset)
    validate_catalog_frame(df)
    topsis = fixed_topsis_bundle(df)
    producer_topsis = core.topsis_artifacts(df)
    if float(np.max(np.abs(np.asarray(topsis["raw_scores"]) - producer_topsis["scores"]))) > TOL:
        raise AssertionError("TOPSIS implementation mismatch against frozen producer core")
    baselines, shapley, shapley_error = exact_topsis_shapley(topsis)
    normalized_t = np.asarray(topsis["normalized_scores"], dtype=float)
    t_reconstruction = baselines + shapley.sum(axis=1)
    t_error = float(np.max(np.abs(t_reconstruction - normalized_t)))
    if t_error > TOL:
        raise AssertionError(f"Normalized TOPSIS reconstruction error {t_error}")

    profile_outputs: list[dict[str, Any]] = []
    for profile_idx, profile_row in enumerate(row["profiles"]):
        name = str(profile_row["profile_name"])
        replay = replay_reward_components(
            df,
            name,
            profile_idx,
            int(row["run_seed"]),
            np.asarray(profile_row["q_scores"], dtype=float),
            np.asarray(profile_row["visits"], dtype=np.int32),
            core,
        )
        cq_parts = cq_affine_decomposition(replay)
        q_norm = minmax_norm(replay["q_total"])
        c_q = np.asarray(cq_parts["c_q"], dtype=float)
        c_t = 0.50 * normalized_t
        score = c_q + c_t
        reconstruction = (
            float(cq_parts["reference"])
            + np.asarray(cq_parts["base"])
            + np.asarray(cq_parts["engage"])
            + np.asarray(cq_parts["convert"])
            + 0.50 * (baselines + shapley.sum(axis=1))
        )
        score_error = float(np.max(np.abs(score - reconstruction)))
        if score_error > TOL:
            raise AssertionError(f"Hybrid reconstruction error {score_error}")
        top7 = [int(x) for x in np.argsort(score)[::-1][:7]]
        reconstructed_top7 = [int(x) for x in np.argsort(reconstruction)[::-1][:7]]
        if top7 != reconstructed_top7:
            raise AssertionError("Top-7 reconstruction mismatch")
        profile_outputs.append(
            {
                "profile_name": name,
                "action_trace_sha256": replay["action_trace_sha256"],
                "reward_event_trace_sha256": replay["reward_event_trace_sha256"],
                "rank_definition": "descending hybrid score via numpy.argsort(score)[::-1][:7], preserving producer tie semantics",
                "hybrid_top7_rank": top7,
                "diagnostics": {
                    "engage_count": int(replay["engage_count"]),
                    "convert_count": int(replay["convert_count"]),
                    "q_component_raw_reconstruction_max_abs_error": float(replay["component_error"]),
                    "source_q_replay_max_abs_error": float(replay["source_error"]),
                    "c_q_affine_reconstruction_max_abs_error": float(cq_parts["error"]),
                    "hybrid_reconstruction_max_abs_error": score_error,
                    "q_score_min": float(cq_parts["q_min"]),
                    "q_score_max": float(cq_parts["q_max"]),
                    "q_normalization_denominator": float(cq_parts["denominator"]),
                    "q_is_constant": bool(cq_parts["is_constant"]),
                },
                "q_total": array_list(replay["q_total"]),
                "q_base": array_list(replay["q_base"]),
                "q_engage": array_list(replay["q_engage"]),
                "q_convert": array_list(replay["q_convert"]),
                "visits": array_list(replay["visits"]),
                "c_q_reference": float(cq_parts["reference"]),
                "c_q_base": array_list(np.asarray(cq_parts["base"])),
                "c_q_engage": array_list(np.asarray(cq_parts["engage"])),
                "c_q_convert": array_list(np.asarray(cq_parts["convert"])),
                "c_q": array_list(c_q),
                "c_t": array_list(c_t),
                "hybrid_score": array_list(score),
            }
        )
    output = {
        "schema_version": "exact_xai.catalog_attribution.v1",
        "run_index": int(row["run_index"]),
        "run_seed": int(row["run_seed"]),
        "dataset_sha256": str(row["dataset_sha256"]),
        "allowlisted_payload_sha256": str(row["allowlisted_payload_sha256"]),
        "topsis": {
            "criteria": list(CRITERIA),
            "weights": array_list(np.asarray(topsis["weights"])),
            "vector_norms": array_list(np.asarray(topsis["vector_norms"])),
            "ideal_plus": array_list(np.asarray(topsis["ideal_plus"])),
            "ideal_minus": array_list(np.asarray(topsis["ideal_minus"])),
            "median_reference": array_list(np.asarray(topsis["reference"])),
            "score_minmax": {
                "min": float(topsis["score_min"]),
                "max": float(topsis["score_max"]),
                "normalization_epsilon": 1e-10,
            },
            "raw_scores": array_list(np.asarray(topsis["raw_scores"])),
            "normalized_scores": array_list(normalized_t),
            "shapley_baseline_normalized": array_list(baselines),
            "shapley_values_normalized": array_list(shapley),
            "diagnostics": {
                "shapley_efficiency_max_abs_error": float(shapley_error),
                "normalized_reconstruction_max_abs_error": t_error,
            },
        },
        "profiles": profile_outputs,
    }
    assert_finite(output)
    assert_label_free_keys(output)
    return output


def run(args: argparse.Namespace) -> None:
    started = time.time()
    campaign_root = Path(__file__).resolve().parents[1]
    source_data_campaign = args.source_data_campaign.resolve()
    source_data_root = args.source_data_root.resolve()
    source_evidence_campaign = args.source_evidence_campaign.resolve()
    source_evidence_root = args.source_evidence_root.resolve()
    output_dir = args.output_dir.resolve()
    if source_data_campaign != (campaign_root.parent / SOURCE_CAMPAIGN_ID).resolve():
        raise RuntimeError("Source data campaign is not the expected sibling campaign")
    if source_evidence_campaign != (campaign_root.parent / SOURCE_EVIDENCE_CAMPAIGN_ID).resolve():
        raise RuntimeError("Source evidence campaign is not the expected sibling campaign")
    ensure_contained(output_dir, campaign_root / "outputs", "XAI output")
    if output_dir.parent.resolve() != (campaign_root / "outputs").resolve():
        raise RuntimeError("XAI output must be a direct child of campaign outputs")
    run_manifest_sha256, campaign_hashes, execution_policy = validate_run_manifest(campaign_root)
    if args.mode not in execution_policy["allowed_modes"]:
        raise PermissionError("LOCKED execution policy does not allow the requested mode")
    prepare_output(output_dir)
    rows, core, provenance = validate_source(
        source_data_campaign,
        source_data_root,
        source_evidence_campaign,
        source_evidence_root,
        args.mode,
    )
    extracted = extract_allowlisted_inputs(rows, source_data_campaign)
    input_path = output_dir / "xai_inputs.jsonl"
    atomic_jsonl(input_path, extracted)
    reread = read_jsonl(input_path)
    for row in reread:
        assert_label_free_keys(row)
    total_cells = len(extracted) * len(PROFILE_ORDER)
    status = {
        "campaign_id": CAMPAIGN_ID,
        "mode": args.mode,
        "status": "running",
        "pid": os.getpid(),
        "catalogs_total": len(extracted),
        "catalogs_completed": 0,
        "profile_cells_total": total_cells,
        "profile_cells_completed": 0,
        "progress_percent": 0.0,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "run_manifest_sha256": run_manifest_sha256,
        "environment": EXPECTED_ENVIRONMENT,
    }
    assert_label_free_keys(status)
    atomic_json(output_dir / "status.json", status)
    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(extracted, start=1):
        outputs.append(build_catalog_attribution(row, source_data_campaign, core))
        completed_cells = index * len(PROFILE_ORDER)
        elapsed = time.time() - started
        rate = elapsed / completed_cells
        status.update(
            {
                "catalogs_completed": index,
                "profile_cells_completed": completed_cells,
                "progress_percent": round(100.0 * completed_cells / total_cells, 3),
                "elapsed_seconds": elapsed,
                "eta_seconds": max(0.0, rate * (total_cells - completed_cells)),
            }
        )
        atomic_json(output_dir / "status.json", status)
        print(
            f"progress={status['progress_percent']:.3f}% cells={completed_cells}/{total_cells} "
            f"eta_seconds={status['eta_seconds']:.1f}",
            flush=True,
        )
    attribution_path = output_dir / "xai_attributions.jsonl"
    atomic_jsonl(attribution_path, outputs)
    terminal = {
        "schema_version": "exact_xai.results.v1",
        "campaign_id": CAMPAIGN_ID,
        "mode": args.mode,
        "status": "completed_unverified",
        "source_data_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_evidence_campaign_id": SOURCE_EVIDENCE_CAMPAIGN_ID,
        "primary_arm_id": PRIMARY_ARM_ID,
        "catalogs": len(outputs),
        "profile_cells": total_cells,
        "items_per_cell": 400,
        "episodes_per_cell": EPISODES,
        "tolerance": TOL,
        "run_manifest_sha256": run_manifest_sha256,
        "campaign_artifact_hashes": campaign_hashes,
        "environment": EXPECTED_ENVIRONMENT,
        "source_provenance": provenance,
        "output_hashes": {
            "xai_inputs.jsonl": sha256_file(input_path),
            "xai_attributions.jsonl": sha256_file(attribution_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    assert_label_free_keys(terminal)
    assert_finite(terminal)
    terminal_path = output_dir / "xai_results.json"
    atomic_json(terminal_path, terminal)
    status.update(
        {
            "status": "completed_unverified",
            "progress_percent": 100.0,
            "elapsed_seconds": terminal["elapsed_seconds"],
            "eta_seconds": 0.0,
            "terminal_path": str(terminal_path),
            "terminal_sha256": sha256_file(terminal_path),
        }
    )
    assert_label_free_keys(status)
    atomic_json(output_dir / "status.json", status)
    print(f"progress=100.000% cells={total_cells}/{total_cells} status=completed_unverified", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-campaign", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--source-evidence-campaign", type=Path, required=True)
    parser.add_argument("--source-evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "canonical"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    output_existed_before_launch = parsed.output_dir.resolve().exists()
    try:
        run(parsed)
    except Exception as exc:
        try:
            failure_path = parsed.output_dir.resolve() / "status.json"
            if not output_existed_before_launch and parsed.output_dir.resolve().is_dir():
                prior = read_json(failure_path) if failure_path.is_file() else {
                    "campaign_id": CAMPAIGN_ID,
                    "mode": parsed.mode,
                    "pid": os.getpid(),
                }
                prior.update(
                    {
                        "status": "failed_closed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                atomic_json(failure_path, prior)
        except Exception:
            pass
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
