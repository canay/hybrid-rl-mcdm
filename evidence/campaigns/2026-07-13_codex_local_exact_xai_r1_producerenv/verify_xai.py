"""Independent verifier for the exact XAI campaign.

This module intentionally does not import ``src/xai_main.py``.  It reimplements
the extraction, stochastic replay, TOPSIS coalition game, Shapley enumeration,
fusion, ranks, and hashes.
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
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy


CAMPAIGN_ID = "2026-07-13_codex_local_exact_xai_r1_producerenv"
SOURCE_CAMPAIGN_ID = "2026-07-12_codex_local_same_target_bridge_r01_producerenv"
SOURCE_EVIDENCE_CAMPAIGN_ID = "2026-07-13_codex_static_verifier_topsis_gatefix"
ARM_ID = "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix"
PROFILES = ("budget", "quality_seeker", "explorer", "loyal", "balanced")
COLS = ("price_pct", "quality_pct", "popularity_pct", "rating_pct")
DENIED = ("ground_truth", "target", "label", "metric", "relevance", "ndcg", "f1", "gt_", "_gt")
DENIED_EXACT = ("gt",)
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
    "product_id", "product_name", "category_raw", "category", "brand_label", "brand",
    "price", "actual_price", "discount_pct", "rating", "rating_count",
    "inferred_reviewer_count", "review_text_richness", "quality", "popularity",
    "recency", "price_pct", "quality_pct", "popularity_pct", "rating_pct", "recency_pct",
)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant rejected: {value}")


def versions() -> dict[str, str]:
    return {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}


def contained(path: Path, root: Path, label: str) -> Path:
    value, base = path.resolve(), root.resolve()
    try:
        value.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{label} escapes allowed root: {value}") from exc
    return value


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    finite_tree(value, str(path))
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def object_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"), parse_constant=reject_json_constant
    )
    if not isinstance(value, dict):
        raise TypeError(path)
    finite_tree(value, str(path))
    return value


def lines_json(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=reject_json_constant)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{number}")
            finite_tree(value, f"{path}:{number}")
            values.append(value)
    return values


def dynamic_module(path: Path):
    spec = importlib.util.spec_from_file_location("exact_xai_verifier_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reject_denied_keys(value: Any, trail: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in DENIED_EXACT or any(token in lowered for token in DENIED):
                raise ValueError(f"Forbidden label/evaluation key {trail}.{key}")
            reject_denied_keys(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            reject_denied_keys(child, f"{trail}[{idx}]")


def finite_tree(value: Any, trail: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite {trail}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            finite_tree(child, f"{trail}[{idx}]")


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise AssertionError(f"{label} keys {sorted(value)} != {sorted(expected)}")


def source_dataset(source_campaign: Path, relative_value: str) -> Path:
    relative = Path(relative_value.replace("/", "\\"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe source dataset path")
    return contained(
        source_campaign / "inputs" / relative,
        source_campaign / "inputs" / "data" / "processed" / "bootstrap_catalogs",
        "dataset",
    )


def validate_frame(frame: pd.DataFrame) -> None:
    if tuple(str(column) for column in frame.columns) != EXPECTED_COLUMNS or len(frame) != 400:
        raise RuntimeError("Catalog schema/row-count mismatch")
    numeric_columns = EXPECTED_COLUMNS[6:]
    if not np.all(np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float))):
        raise RuntimeError("Catalog contains nonfinite numeric values")
    if frame[["product_id", "category", "brand"]].isna().any().any():
        raise RuntimeError("Catalog contains missing identity/category/proxy values")


def validate_run_manifest(campaign: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    path = campaign / "RUN_MANIFEST.json"
    value = object_json(path)
    exact_keys(
        value,
        {
            "schema_version", "campaign_id", "operation_id", "status", "canonical_authorized",
            "execution_policy", "environment", "source_bindings", "files",
            "full_verification_contract",
        },
        "RUN manifest",
    )
    if (
        value["schema_version"] != "exact_xai.run_manifest.v1"
        or value["campaign_id"] != CAMPAIGN_ID
        or value["operation_id"] != "HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19"
        or value["status"] != "LOCKED"
        or value["canonical_authorized"] is not True
        or value["environment"] != EXPECTED_ENVIRONMENT
    ):
        raise RuntimeError("RUN manifest identity/status/environment mismatch")
    expected_policy = {
        "lifecycle_status": "LOCKED",
        "allowed_modes": ["smoke", "canonical"],
        "canonical_authorized": True,
        "authorization_operation_id": "HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19",
    }
    if value["execution_policy"] != expected_policy:
        raise RuntimeError("RUN manifest execution policy mismatch")
    if value["source_bindings"] != {
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
        "primary_arm_id": ARM_ID,
        "canonical_shape": {"catalogs": 50, "profiles_per_catalog": 5},
    }:
        raise RuntimeError("RUN manifest source bindings mismatch")
    expected_paths = {
        "PROTOCOL_LOCK.md", "src/xai_main.py", "verify_xai.py", "build_run_manifest.py",
        "tests/test_xai_contract.py", "tests/test_verify_xai.py",
    }
    rows = value["files"]
    if not isinstance(rows, list) or {row.get("path") for row in rows} != expected_paths:
        raise RuntimeError("RUN manifest file-set mismatch")
    hashes: dict[str, str] = {}
    for row in rows:
        exact_keys(row, {"path", "sha256"}, "RUN manifest file")
        candidate = contained(campaign / Path(row["path"]), campaign, "lock file")
        actual = file_hash(candidate)
        if actual != row["sha256"]:
            raise RuntimeError(f"RUN manifest file hash mismatch: {row['path']}")
        hashes[row["path"]] = actual
    if value["full_verification_contract"] != {
        "binds": [
            "RUN_MANIFEST.json", "PROTOCOL_LOCK.md", "src/xai_main.py",
            "verify_xai.py", "build_run_manifest.py", "tests/test_xai_contract.py", "tests/test_verify_xai.py",
            "producer_environment",
        ],
        "terminal_verdict": "completed_verified/PASS",
    }:
        raise RuntimeError("RUN manifest FULL contract mismatch")
    return file_hash(path), hashes, expected_policy


def independent_source_extract(source_records: list[dict[str, Any]], source_campaign: Path) -> list[dict[str, Any]]:
    result = []
    for record in source_records:
        bundle = record["arms"][ARM_ID]
        arm = bundle["arm"]
        if arm != {
            "arm_id": ARM_ID,
            "candidate": "full_catalog",
            "gt_bonus": 0.0,
            "reward_model": "component_continuous_fix",
            "role": "mandatory_factorial",
        }:
            raise RuntimeError("Primary arm contract mismatch")
        source_profiles = bundle["profiles"]
        if [item["profile_name"] for item in source_profiles] != list(PROFILES):
            raise RuntimeError("Profile order mismatch")
        selected_profiles = []
        for item in source_profiles:
            q = np.asarray(item["q_scores"], dtype=float)
            visits = np.asarray(item["visits"], dtype=np.int64)
            if (
                q.shape != (400,)
                or not np.all(np.isfinite(q))
                or visits.shape != (400,)
                or any(isinstance(x, bool) or int(x) != x or int(x) < 0 for x in item["visits"])
                or int(visits.sum()) != EPISODES
            ):
                raise RuntimeError("Malformed source primary vector")
            selected_profiles.append(
                {
                    "profile_name": str(item["profile_name"]),
                    "q_scores": [float(x) for x in q],
                    "visits": [int(x) for x in visits],
                }
            )
        rel = str(record["dataset_path"]).replace("/", "\\")
        data_path = source_dataset(source_campaign, rel)
        if file_hash(data_path) != record["dataset_sha256"]:
            raise RuntimeError("Source catalog hash mismatch")
        selected = {
            "schema_version": "exact_xai.label_free_input.v1",
            "run_index": int(record["run_index"]),
            "run_seed": int(record["run_seed"]),
            "dataset_path": rel,
            "dataset_sha256": str(record["dataset_sha256"]),
            "profiles": selected_profiles,
        }
        selected["allowlisted_payload_sha256"] = object_hash(selected)
        exact_keys(
            selected,
            {
                "schema_version", "run_index", "run_seed", "dataset_path",
                "dataset_sha256", "profiles", "allowlisted_payload_sha256",
            },
            "label-free input",
        )
        for selected_profile in selected_profiles:
            exact_keys(selected_profile, {"profile_name", "q_scores", "visits"}, "label-free profile")
        reject_denied_keys(selected)
        finite_tree(selected)
        result.append(selected)
    return result


def norm2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.full_like(values, 0.5)
    return (values - low) / (high - low + 1e-10)


def topsis_independent(frame: pd.DataFrame) -> dict[str, Any]:
    x = np.maximum(frame[list(COLS)].to_numpy(dtype=float), 1e-10)
    p = np.minimum(np.maximum(x / x.sum(axis=0, keepdims=True), 1e-10), 1.0)
    entropy = -(p * np.log(p)).sum(axis=0) / np.log(x.shape[0])
    div = 1.0 - entropy
    w = 0.10 + 0.60 * div / div.sum()
    w = np.maximum(w, 0.10)
    w = w / w.sum()
    norms = np.sqrt(np.sum(x * x, axis=0))
    norms[norms == 0] = 1.0
    weighted = x / norms * w
    high = weighted.max(axis=0)
    low = weighted.min(axis=0)
    dp = np.sqrt(np.sum((weighted - high) ** 2, axis=1))
    dm = np.sqrt(np.sum((weighted - low) ** 2, axis=1))
    den = dp + dm
    den[den == 0] = 1e-10
    scores = dm / den
    return {
        "matrix": x,
        "weights": w,
        "norms": norms,
        "plus": high,
        "minus": low,
        "reference": np.median(x, axis=0),
        "raw": scores,
        "normalized": norm2(scores),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def game_value(item: np.ndarray, coalition: int, fixed: Mapping[str, Any]) -> float:
    x = np.array(
        [item[j] if coalition & (1 << j) else fixed["reference"][j] for j in range(4)],
        dtype=float,
    )
    z = x / fixed["norms"] * fixed["weights"]
    dp = float(np.sqrt(np.sum((z - fixed["plus"]) ** 2)))
    dm = float(np.sqrt(np.sum((z - fixed["minus"]) ** 2)))
    raw = dm / (dp + dm if dp + dm != 0 else 1e-10)
    lo, hi = fixed["score_min"], fixed["score_max"]
    return 0.5 if hi <= lo else float((raw - lo) / (hi - lo + 1e-10))


def shapley4(item: np.ndarray, fixed: Mapping[str, Any]) -> tuple[float, np.ndarray, float]:
    v = [game_value(item, mask, fixed) for mask in range(16)]
    phi = np.zeros(4)
    for i in range(4):
        bit = 1 << i
        for mask in range(16):
            if mask & bit:
                continue
            s = mask.bit_count()
            coefficient = math.factorial(s) * math.factorial(3 - s) / math.factorial(4)
            phi[i] += coefficient * (v[mask | bit] - v[mask])
    error = abs(v[0] + phi.sum() - v[15])
    return v[0], phi, error


def hidden_parts_independent(frame: pd.DataFrame, profile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    brands = np.asarray([profile["brand_pref"].get(x, 0.10) for x in frame["brand"]], dtype=float)
    lo, hi = profile["price_range"]
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 + 1.0
    price = np.clip(1.0 - np.abs(frame["price"].to_numpy(float) - center) / half, 0.0, 1.0)
    affinity = profile["cat_affinity"]
    category = np.asarray([affinity.get(x, 0.05) for x in frame["category"]], dtype=float)
    category = category / category.max()
    rw = float(profile["recency_weight"])
    recent = frame["recency_pct"].to_numpy(float) * rw + (1.0 - rw) * 0.5
    pe = np.clip(0.40 * brands + 0.35 * price + 0.15 * category + 0.10 * recent, 0.0, 1.0)
    pc = np.clip(0.50 * brands + 0.30 * price + 0.20 * category, 0.0, 1.0)
    return pe, pc


def replay_independent(
    frame: pd.DataFrame,
    profile: Mapping[str, Any],
    profile_index: int,
    seed: int,
) -> dict[str, Any]:
    pe, pc = hidden_parts_independent(frame, profile)
    q = np.zeros(400)
    qb = np.zeros(400)
    qe = np.zeros(400)
    qc = np.zeros(400)
    visits = np.zeros(400, dtype=np.int32)
    actions = np.random.RandomState(seed + profile_index * 13)
    rewards = np.random.RandomState(seed + profile_index * 997)
    pool = np.arange(400)
    epsilon = 0.30
    action_digest = hashlib.sha256()
    reward_digest = hashlib.sha256()
    engage_count = convert_count = 0
    for _ in range(EPISODES):
        action = int(actions.choice(pool)) if actions.random() < epsilon else int(pool[np.argmax(q[pool])])
        engage = bool(rewards.random() < pe[action])
        convert = False
        if engage:
            engage_count += 1
            convert = bool(rewards.random() < pc[action])
            if convert:
                convert_count += 1
        rb, re, rc = -0.02, 0.30 if engage else 0.0, 1.0 if convert else 0.0
        visits[action] += 1
        q[action] += 0.05 * (rb + re + rc - q[action])
        qb[action] += 0.05 * (rb - qb[action])
        qe[action] += 0.05 * (re - qe[action])
        qc[action] += 0.05 * (rc - qc[action])
        epsilon = max(0.05, epsilon * 0.9997)
        action_digest.update(struct.pack("<i", action))
        reward_digest.update(bytes((int(engage), int(convert))))
    return {
        "q": q,
        "qb": qb,
        "qe": qe,
        "qc": qc,
        "visits": visits,
        "action_hash": action_digest.hexdigest(),
        "reward_hash": reward_digest.hexdigest(),
        "engage_count": engage_count,
        "convert_count": convert_count,
    }


def cq_parts_independent(replay: Mapping[str, Any]) -> dict[str, Any]:
    q = np.asarray(replay["q"], dtype=float)
    lo, hi = float(np.min(q)), float(np.max(q))
    if hi <= lo:
        reference = 0.25
        base = engage = convert = np.zeros_like(q)
        denominator = 0.0
        is_constant = True
    else:
        denominator = hi - lo + 1e-10
        reference = -0.50 * lo / denominator
        base = 0.50 * np.asarray(replay["qb"]) / denominator
        engage = 0.50 * np.asarray(replay["qe"]) / denominator
        convert = 0.50 * np.asarray(replay["qc"]) / denominator
        is_constant = False
    c_q = 0.50 * norm2(q)
    error = float(np.max(np.abs(c_q - (reference + base + engage + convert))))
    if error > TOL:
        raise AssertionError("Independent cQ affine decomposition failure")
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


def close_array(actual: Any, expected: np.ndarray, name: str) -> int:
    arr = np.asarray(actual, dtype=float)
    if not np.all(np.isfinite(arr)) or not np.all(np.isfinite(expected)):
        raise AssertionError(f"{name} contains nonfinite values")
    if arr.shape != expected.shape:
        raise AssertionError(f"{name} shape {arr.shape} != {expected.shape}")
    error = float(np.max(np.abs(arr - expected))) if arr.size else 0.0
    if error > TOL:
        raise AssertionError(f"{name} max abs error {error}")
    return int(arr.size)


def close_scalar(actual: Any, expected: float, name: str, tolerance: float = TOL) -> int:
    value = float(actual)
    if not math.isfinite(value) or not math.isfinite(float(expected)):
        raise AssertionError(f"{name} is nonfinite")
    if abs(value - float(expected)) > tolerance:
        raise AssertionError(f"{name} mismatch: {value} != {expected}")
    return 1


def validate_bound_source(
    data_campaign: Path,
    data_root: Path,
    evidence_campaign: Path,
    evidence_root: Path,
    mode: str,
) -> tuple[list[dict[str, Any]], Any, dict[str, Any], dict[str, Any]]:
    if data_campaign.name != SOURCE_CAMPAIGN_ID or evidence_campaign.name != SOURCE_EVIDENCE_CAMPAIGN_ID:
        raise RuntimeError("Bound source campaign identity mismatch")
    expected_data_name = "canonical_main" if mode == "canonical" else "smoke"
    expected_evidence_name = "canonical_overlay" if mode == "canonical" else "smoke_overlay"
    for root, campaign, expected_name, label in (
        (data_root, data_campaign, expected_data_name, "data"),
        (evidence_root, evidence_campaign, expected_evidence_name, "evidence"),
    ):
        contained(root, campaign / "outputs", f"source {label}")
        if root.parent.resolve() != (campaign / "outputs").resolve() or root.name != expected_name:
            raise RuntimeError(f"Bound source {label} containment/name mismatch")
    if versions() != EXPECTED_ENVIRONMENT:
        raise RuntimeError("Bound source producer environment mismatch")
    if file_hash(data_campaign / "RUN_MANIFEST.json") != SOURCE_RUN_MANIFEST_SHA256:
        raise RuntimeError("Bound data manifest mismatch")
    evidence_manifest_path = evidence_campaign / "RUN_MANIFEST.json"
    if file_hash(evidence_manifest_path) != SOURCE_EVIDENCE_RUN_MANIFEST_SHA256:
        raise RuntimeError("Bound evidence manifest mismatch")
    evidence_manifest = object_json(evidence_manifest_path)
    if (
        evidence_manifest.get("campaign_id") != SOURCE_EVIDENCE_CAMPAIGN_ID
        or evidence_manifest.get("status") != "LOCKED"
        or evidence_manifest.get("source_campaign", {}).get("campaign_id") != SOURCE_CAMPAIGN_ID
        or evidence_manifest.get("execution_policy", {}).get("canonical_authorized") is not True
    ):
        raise RuntimeError("Bound evidence manifest contract mismatch")

    data_paths = {name: data_root / name for name in CANONICAL_DATA_HASHES}
    full_path = evidence_root / "FULL_VERIFICATION.json"
    verification_path = evidence_root / "verification_status.json"
    for path in (*data_paths.values(), full_path, verification_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    full = object_json(full_path)
    verification = object_json(verification_path)
    data_status = object_json(data_paths["status.json"])
    if mode == "canonical":
        for name, expected in CANONICAL_DATA_HASHES.items():
            if file_hash(data_paths[name]) != expected:
                raise RuntimeError(f"Canonical data pin mismatch: {name}")
        for name, expected in CANONICAL_EVIDENCE_HASHES.items():
            if file_hash(evidence_root / name) != expected:
                raise RuntimeError(f"Canonical evidence pin mismatch: {name}")
    expected_runs = 50 if mode == "canonical" else 1
    expected_cells = expected_runs * 100
    if (
        full.get("campaign_id") != SOURCE_EVIDENCE_CAMPAIGN_ID
        or full.get("source_campaign_id") != SOURCE_CAMPAIGN_ID
        or full.get("mode") != mode
        or full.get("status") != "completed_verified"
        or full.get("verdict") != "PASS"
        or full.get("run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
        or full.get("producer_provenance", {}).get("environment") != EXPECTED_ENVIRONMENT
        or any(gate.get("status") != "PASS" for gate in full.get("gates", []))
    ):
        raise RuntimeError("Bound evidence FULL identity/verdict mismatch")
    if set(full.get("output_hashes", {})) != set(data_paths):
        raise RuntimeError("Bound evidence output-hash schema mismatch")
    for name, path in data_paths.items():
        if full["output_hashes"][name] != file_hash(path):
            raise RuntimeError(f"Bound evidence/data mismatch: {name}")
    overlay = full.get("overlay_contract", {})
    if (
        overlay.get("overlay_manifest_sha256") != SOURCE_EVIDENCE_RUN_MANIFEST_SHA256
        or overlay.get("source_run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
        or overlay.get("source_canonical_checkpoint_sha256") != CANONICAL_DATA_HASHES["main_catalogs.jsonl"]
        or overlay.get("source_canonical_terminal_sha256") != CANONICAL_DATA_HASHES["main_results.json"]
        or overlay.get("source_canonical_runner_status_sha256") != CANONICAL_DATA_HASHES["status.json"]
    ):
        raise RuntimeError("Bound overlay contract mismatch")
    required_verification = {
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
        "full_verification_sha256": file_hash(full_path),
    }
    for key, expected in required_verification.items():
        if verification.get(key) != expected:
            raise RuntimeError(f"Bound verification status mismatch: {key}")
    required_status = {
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
    for key, expected in required_status.items():
        if data_status.get(key) != expected:
            raise RuntimeError(f"Bound data status mismatch: {key}")
    terminal = contained(
        data_campaign / Path(str(data_status.get("terminal_path", ""))), data_root, "bound terminal"
    )
    if terminal != data_paths["main_results.json"].resolve() or data_status.get("terminal_sha256") != file_hash(terminal):
        raise RuntimeError("Bound data terminal mismatch")
    core_path = data_campaign / "src" / "original_hybrid_core.py"
    if full.get("input_hashes", {}).get("src/original_hybrid_core.py") != SOURCE_CORE_SHA256 or file_hash(core_path) != SOURCE_CORE_SHA256:
        raise RuntimeError("Bound core mismatch")
    records = lines_json(data_paths["main_catalogs.jsonl"])
    if len(records) != expected_runs or [int(row.get("run_index", -1)) for row in records] != list(range(expected_runs)):
        raise RuntimeError("Bound source run count/order mismatch")
    provenance = {
        "data_campaign_id": SOURCE_CAMPAIGN_ID,
        "evidence_campaign_id": SOURCE_EVIDENCE_CAMPAIGN_ID,
        "data_main_catalogs_sha256": file_hash(data_paths["main_catalogs.jsonl"]),
        "data_terminal_sha256": file_hash(data_paths["main_results.json"]),
        "data_status_sha256": file_hash(data_paths["status.json"]),
        "evidence_full_verification_sha256": file_hash(full_path),
        "evidence_verification_status_sha256": file_hash(verification_path),
        "data_run_manifest_sha256": SOURCE_RUN_MANIFEST_SHA256,
        "evidence_run_manifest_sha256": SOURCE_EVIDENCE_RUN_MANIFEST_SHA256,
        "core_sha256": SOURCE_CORE_SHA256,
        "environment": EXPECTED_ENVIRONMENT,
    }
    return records, dynamic_module(core_path), provenance, full


def verify(args: argparse.Namespace) -> None:
    began = time.time()
    campaign = args.campaign.resolve()
    source_data_campaign = args.source_data_campaign.resolve()
    source_data_root = args.source_data_root.resolve()
    source_evidence_campaign = args.source_evidence_campaign.resolve()
    source_evidence_root = args.source_evidence_root.resolve()
    output = args.output_dir.resolve()
    if (
        campaign.name != CAMPAIGN_ID
        or source_data_campaign.name != SOURCE_CAMPAIGN_ID
        or source_evidence_campaign.name != SOURCE_EVIDENCE_CAMPAIGN_ID
    ):
        raise RuntimeError("Campaign identity mismatch")
    if campaign != Path(__file__).resolve().parent:
        raise RuntimeError("Verifier is not executing from the declared campaign")
    if source_data_campaign != (campaign.parent / SOURCE_CAMPAIGN_ID).resolve():
        raise RuntimeError("Source data campaign is not the expected sibling campaign")
    if source_evidence_campaign != (campaign.parent / SOURCE_EVIDENCE_CAMPAIGN_ID).resolve():
        raise RuntimeError("Source evidence campaign is not the expected sibling campaign")
    contained(output, campaign / "outputs", "XAI output")
    if output.parent.resolve() != (campaign / "outputs").resolve():
        raise RuntimeError("XAI output must be a direct child of campaign outputs")
    if versions() != EXPECTED_ENVIRONMENT:
        raise RuntimeError("Verifier producer environment mismatch")
    run_manifest_sha256, campaign_hashes, execution_policy = validate_run_manifest(campaign)
    if args.mode not in execution_policy["allowed_modes"]:
        raise PermissionError("LOCKED execution policy does not allow the requested mode")
    full_path = output / "FULL_VERIFICATION.json"
    if full_path.exists():
        raise FileExistsError("Stale FULL_VERIFICATION.json exists")
    initial_names = {item.name for item in output.iterdir()}
    expected_initial_names = {"xai_inputs.jsonl", "xai_attributions.jsonl", "xai_results.json", "status.json"}
    if initial_names != expected_initial_names or any(not (output / name).is_file() for name in initial_names):
        raise RuntimeError("XAI runner output file-set mismatch")
    terminal = object_json(output / "xai_results.json")
    runner_status = object_json(output / "status.json")
    exact_keys(
        terminal,
        {
            "schema_version", "campaign_id", "mode", "status", "source_data_campaign_id",
            "source_evidence_campaign_id",
            "primary_arm_id", "catalogs", "profile_cells", "items_per_cell",
            "episodes_per_cell", "tolerance", "run_manifest_sha256",
            "campaign_artifact_hashes", "environment", "source_provenance",
            "output_hashes", "elapsed_seconds",
        },
        "runner terminal",
    )
    exact_keys(
        runner_status,
        {
            "campaign_id", "mode", "status", "pid", "catalogs_total",
            "catalogs_completed", "profile_cells_total", "profile_cells_completed",
            "progress_percent", "elapsed_seconds", "eta_seconds",
            "run_manifest_sha256", "environment", "terminal_path", "terminal_sha256",
        },
        "runner status",
    )
    if terminal.get("status") != "completed_unverified" or runner_status.get("status") != "completed_unverified":
        raise RuntimeError("Runner not terminal completed_unverified")
    expected_runs = 1 if args.mode == "smoke" else 50
    total_cells = expected_runs * 5
    if (
        terminal.get("schema_version") != "exact_xai.results.v1"
        or terminal.get("campaign_id") != CAMPAIGN_ID
        or terminal.get("mode") != args.mode
        or terminal.get("source_data_campaign_id") != SOURCE_CAMPAIGN_ID
        or terminal.get("source_evidence_campaign_id") != SOURCE_EVIDENCE_CAMPAIGN_ID
        or terminal.get("primary_arm_id") != ARM_ID
        or terminal.get("catalogs") != expected_runs
        or terminal.get("profile_cells") != total_cells
        or terminal.get("items_per_cell") != 400
        or terminal.get("episodes_per_cell") != EPISODES
        or terminal.get("tolerance") != TOL
        or terminal.get("run_manifest_sha256") != run_manifest_sha256
        or terminal.get("campaign_artifact_hashes") != campaign_hashes
        or terminal.get("environment") != EXPECTED_ENVIRONMENT
    ):
        raise RuntimeError("Runner terminal identity/mode mismatch")
    expected_runner_status = {
        "campaign_id": CAMPAIGN_ID,
        "mode": args.mode,
        "status": "completed_unverified",
        "catalogs_total": expected_runs,
        "catalogs_completed": expected_runs,
        "profile_cells_total": total_cells,
        "profile_cells_completed": total_cells,
        "progress_percent": 100.0,
        "eta_seconds": 0.0,
        "run_manifest_sha256": run_manifest_sha256,
        "environment": EXPECTED_ENVIRONMENT,
    }
    for key, expected in expected_runner_status.items():
        if runner_status.get(key) != expected:
            raise RuntimeError(f"Runner status mismatch: {key}")
    if not math.isfinite(float(runner_status["elapsed_seconds"])) or float(runner_status["elapsed_seconds"]) < 0:
        raise RuntimeError("Runner status elapsed time invalid")
    terminal_path = contained(Path(str(runner_status["terminal_path"])), output, "runner terminal")
    if terminal_path != (output / "xai_results.json").resolve() or runner_status["terminal_sha256"] != file_hash(terminal_path):
        raise RuntimeError("Runner terminal path/hash mismatch")
    if set(terminal["output_hashes"]) != {"xai_inputs.jsonl", "xai_attributions.jsonl"}:
        raise RuntimeError("Runner terminal output hash schema mismatch")
    for name in sorted(terminal["output_hashes"]):
        if terminal["output_hashes"].get(name) != file_hash(output / name):
            raise RuntimeError(f"Runner output hash mismatch: {name}")
    reject_denied_keys(terminal)
    reject_denied_keys(runner_status)

    source_records, core, expected_source_provenance, source_full = validate_bound_source(
        source_data_campaign,
        source_data_root,
        source_evidence_campaign,
        source_evidence_root,
        args.mode,
    )
    if terminal.get("source_provenance") != expected_source_provenance:
        raise RuntimeError("Runner bound-source provenance mismatch")
    expected_inputs = independent_source_extract(source_records, source_data_campaign)
    actual_inputs = lines_json(output / "xai_inputs.jsonl")
    for row in actual_inputs:
        reject_denied_keys(row)
        finite_tree(row)
    if actual_inputs != expected_inputs:
        raise AssertionError("Allowlist extraction does not match independent extraction")
    actual_outputs = lines_json(output / "xai_attributions.jsonl")
    for row in actual_outputs:
        reject_denied_keys(row)
        finite_tree(row)
    if len(actual_outputs) != len(actual_inputs):
        raise AssertionError("Attribution catalog count mismatch")
    if len(actual_inputs) != expected_runs:
        raise AssertionError("Mode run count mismatch")

    status_path = output / "verification_status.json"
    status = {
        "campaign_id": CAMPAIGN_ID,
        "mode": args.mode,
        "status": "running",
        "pid": os.getpid(),
        "profile_cells_total": total_cells,
        "profile_cells_verified": 0,
        "progress_percent": 0.0,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "run_manifest_sha256": run_manifest_sha256,
        "environment": EXPECTED_ENVIRONMENT,
    }
    atomic(status_path, status)
    checks = 0
    maximum_error = 0.0
    for run_pos, (input_row, reported) in enumerate(zip(actual_inputs, actual_outputs)):
        exact_keys(
            input_row,
            {
                "schema_version", "run_index", "run_seed", "dataset_path",
                "dataset_sha256", "profiles", "allowlisted_payload_sha256",
            },
            "label-free input row",
        )
        if input_row["schema_version"] != "exact_xai.label_free_input.v1":
            raise AssertionError("Label-free input schema version mismatch")
        exact_keys(
            reported,
            {
                "schema_version", "run_index", "run_seed", "dataset_sha256",
                "allowlisted_payload_sha256", "topsis", "profiles",
            },
            "attribution row",
        )
        if (
            reported.get("schema_version") != "exact_xai.catalog_attribution.v1"
            or int(reported.get("run_index", -1)) != int(input_row["run_index"])
            or int(reported.get("run_seed", -1)) != int(input_row["run_seed"])
            or reported.get("dataset_sha256") != input_row["dataset_sha256"]
            or reported.get("allowlisted_payload_sha256") != input_row["allowlisted_payload_sha256"]
        ):
            raise AssertionError("Attribution record identity mismatch")
        dataset = source_dataset(source_data_campaign, input_row["dataset_path"])
        if file_hash(dataset) != input_row["dataset_sha256"]:
            raise RuntimeError("Catalog hash changed")
        frame = pd.read_csv(dataset)
        validate_frame(frame)
        fixed = topsis_independent(frame)
        baseline = np.zeros(400)
        phi = np.zeros((400, 4))
        efficiency_max = 0.0
        for item_index, item in enumerate(fixed["matrix"]):
            baseline[item_index], phi[item_index], error = shapley4(item, fixed)
            efficiency_max = max(efficiency_max, error)
        if efficiency_max > TOL:
            raise AssertionError("Independent Shapley efficiency failure")
        topsis_out = reported["topsis"]
        exact_keys(
            topsis_out,
            {
                "criteria", "weights", "vector_norms", "ideal_plus", "ideal_minus",
                "median_reference", "score_minmax", "raw_scores", "normalized_scores",
                "shapley_baseline_normalized", "shapley_values_normalized", "diagnostics",
            },
            "TOPSIS attribution",
        )
        if topsis_out["criteria"] != list(COLS):
            raise AssertionError("TOPSIS criterion order mismatch")
        exact_keys(topsis_out["score_minmax"], {"min", "max", "normalization_epsilon"}, "TOPSIS score minmax")
        checks += close_scalar(topsis_out["score_minmax"]["min"], fixed["score_min"], "score_min")
        checks += close_scalar(topsis_out["score_minmax"]["max"], fixed["score_max"], "score_max")
        checks += close_scalar(topsis_out["score_minmax"]["normalization_epsilon"], 1e-10, "normalization epsilon", 0.0)
        if float(topsis_out["score_minmax"]["max"]) < float(topsis_out["score_minmax"]["min"]):
            raise AssertionError("TOPSIS score minmax ordering invalid")
        exact_keys(
            topsis_out["diagnostics"],
            {"shapley_efficiency_max_abs_error", "normalized_reconstruction_max_abs_error"},
            "TOPSIS diagnostics",
        )
        checks += close_array(topsis_out["weights"], fixed["weights"], "weights")
        checks += close_array(topsis_out["vector_norms"], fixed["norms"], "vector_norms")
        checks += close_array(topsis_out["ideal_plus"], fixed["plus"], "ideal_plus")
        checks += close_array(topsis_out["ideal_minus"], fixed["minus"], "ideal_minus")
        checks += close_array(topsis_out["median_reference"], fixed["reference"], "median_reference")
        checks += close_array(topsis_out["raw_scores"], fixed["raw"], "raw_scores")
        checks += close_array(topsis_out["normalized_scores"], fixed["normalized"], "normalized_scores")
        checks += close_array(topsis_out["shapley_baseline_normalized"], baseline, "shapley_baseline")
        checks += close_array(topsis_out["shapley_values_normalized"], phi, "shapley_values")
        maximum_error = max(
            maximum_error,
            float(topsis_out["diagnostics"]["shapley_efficiency_max_abs_error"]),
            float(topsis_out["diagnostics"]["normalized_reconstruction_max_abs_error"]),
            efficiency_max,
            float(np.max(np.abs(baseline + phi.sum(axis=1) - fixed["normalized"]))),
        )
        if maximum_error > TOL:
            raise AssertionError(f"Global reconstruction tolerance exceeded: {maximum_error}")
        if [p["profile_name"] for p in reported["profiles"]] != list(PROFILES):
            raise AssertionError("Reported profile order mismatch")
        for profile_index, (source_profile, reported_profile) in enumerate(zip(input_row["profiles"], reported["profiles"])):
            exact_keys(source_profile, {"profile_name", "q_scores", "visits"}, "input profile")
            exact_keys(
                reported_profile,
                {
                    "profile_name", "action_trace_sha256", "reward_event_trace_sha256",
                    "rank_definition", "hybrid_top7_rank", "diagnostics", "q_total",
                    "q_base", "q_engage", "q_convert", "visits", "c_q_reference",
                    "c_q_base", "c_q_engage", "c_q_convert", "c_q", "c_t", "hybrid_score",
                },
                "profile attribution",
            )
            name = source_profile["profile_name"]
            replay = replay_independent(frame, core.PROFILE_HIDDEN[name], profile_index, int(input_row["run_seed"]))
            checks += close_array(source_profile["q_scores"], replay["q"], "source_q")
            source_q_error = float(
                np.max(np.abs(np.asarray(source_profile["q_scores"], dtype=float) - replay["q"]))
            )
            if not np.array_equal(np.asarray(source_profile["visits"], dtype=np.int32), replay["visits"]):
                raise AssertionError("Source visits mismatch")
            checks += close_array(reported_profile["q_total"], replay["q"], "q_total")
            checks += close_array(reported_profile["q_base"], replay["qb"], "q_base")
            checks += close_array(reported_profile["q_engage"], replay["qe"], "q_engage")
            checks += close_array(reported_profile["q_convert"], replay["qc"], "q_convert")
            if not np.array_equal(np.asarray(reported_profile["visits"], dtype=np.int32), replay["visits"]):
                raise AssertionError("Reported visits mismatch")
            if reported_profile["action_trace_sha256"] != replay["action_hash"]:
                raise AssertionError("Action trace hash mismatch")
            if reported_profile["reward_event_trace_sha256"] != replay["reward_hash"]:
                raise AssertionError("Reward trace hash mismatch")
            diagnostics = reported_profile["diagnostics"]
            exact_keys(
                diagnostics,
                {
                    "engage_count", "convert_count", "q_component_raw_reconstruction_max_abs_error",
                    "source_q_replay_max_abs_error", "c_q_affine_reconstruction_max_abs_error",
                    "hybrid_reconstruction_max_abs_error", "q_score_min", "q_score_max",
                    "q_normalization_denominator", "q_is_constant",
                },
                "profile diagnostics",
            )
            if int(diagnostics["engage_count"]) != replay["engage_count"] or int(diagnostics["convert_count"]) != replay["convert_count"]:
                raise AssertionError("Reward event count mismatch")
            q_component_error = float(np.max(np.abs(replay["q"] - replay["qb"] - replay["qe"] - replay["qc"])))
            cq_parts = cq_parts_independent(replay)
            cq = np.asarray(cq_parts["c_q"])
            ct = 0.5 * fixed["normalized"]
            hybrid = cq + ct
            reconstruction = (
                cq_parts["reference"] + cq_parts["base"] + cq_parts["engage"]
                + cq_parts["convert"] + 0.5 * (baseline + phi.sum(axis=1))
            )
            score_error = float(np.max(np.abs(hybrid - reconstruction)))
            maximum_error = max(maximum_error, q_component_error, cq_parts["error"], score_error)
            if maximum_error > TOL:
                raise AssertionError("Q/fusion reconstruction tolerance exceeded")
            checks += close_scalar(reported_profile["c_q_reference"], cq_parts["reference"], "c_q_reference")
            checks += close_array(reported_profile["c_q_base"], cq_parts["base"], "c_q_base")
            checks += close_array(reported_profile["c_q_engage"], cq_parts["engage"], "c_q_engage")
            checks += close_array(reported_profile["c_q_convert"], cq_parts["convert"], "c_q_convert")
            checks += close_array(reported_profile["c_q"], cq, "c_q")
            checks += close_array(reported_profile["c_t"], ct, "c_t")
            checks += close_array(reported_profile["hybrid_score"], hybrid, "hybrid_score")
            top7 = [int(x) for x in np.argsort(hybrid)[::-1][:7]]
            reconstructed_top7 = [int(x) for x in np.argsort(reconstruction)[::-1][:7]]
            verified_bridge_top7 = [
                int(x)
                for x in source_records[run_pos]["arms"][ARM_ID]["profiles"][profile_index][
                    "final_rankings"
                ]["hybrid"]
            ]
            expected_rank_definition = "descending hybrid score via numpy.argsort(score)[::-1][:7], preserving producer tie semantics"
            if (
                reported_profile["rank_definition"] != expected_rank_definition
                or reported_profile["hybrid_top7_rank"] != top7
                or top7 != reconstructed_top7
                or top7 != verified_bridge_top7
            ):
                raise AssertionError("Exact top-7 mismatch")
            checks += close_scalar(diagnostics["q_component_raw_reconstruction_max_abs_error"], q_component_error, "raw Q component diagnostic")
            checks += close_scalar(diagnostics["source_q_replay_max_abs_error"], source_q_error, "source Q replay diagnostic")
            checks += close_scalar(diagnostics["c_q_affine_reconstruction_max_abs_error"], cq_parts["error"], "cQ affine diagnostic")
            checks += close_scalar(diagnostics["hybrid_reconstruction_max_abs_error"], score_error, "hybrid reconstruction diagnostic")
            checks += close_scalar(diagnostics["q_score_min"], cq_parts["q_min"], "q score min")
            checks += close_scalar(diagnostics["q_score_max"], cq_parts["q_max"], "q score max")
            checks += close_scalar(diagnostics["q_normalization_denominator"], cq_parts["denominator"], "q denominator")
            if diagnostics["q_is_constant"] is not cq_parts["is_constant"]:
                raise AssertionError("q_is_constant diagnostic mismatch")
            completed = run_pos * 5 + profile_index + 1
            elapsed = time.time() - began
            status.update(
                {
                    "profile_cells_verified": completed,
                    "progress_percent": round(100.0 * completed / total_cells, 3),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": max(0.0, elapsed / completed * (total_cells - completed)),
                }
            )
            atomic(status_path, status)
            print(
                f"verify_progress={status['progress_percent']:.3f}% cells={completed}/{total_cells} "
                f"eta_seconds={status['eta_seconds']:.1f}",
                flush=True,
            )

    status.update({"status": "completed_verified", "progress_percent": 100.0, "eta_seconds": 0.0, "elapsed_seconds": time.time() - began})
    reject_denied_keys(status)
    finite_tree(status)
    atomic(status_path, status)
    full = {
        "schema_version": "exact_xai.full_verification.v1",
        "campaign_id": CAMPAIGN_ID,
        "mode": args.mode,
        "status": "completed_verified",
        "verdict": "PASS",
        "verified_at_epoch_seconds": time.time(),
        "source_data_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_evidence_campaign_id": SOURCE_EVIDENCE_CAMPAIGN_ID,
        "source_evidence_full_verification_sha256": file_hash(source_evidence_root / "FULL_VERIFICATION.json"),
        "source_data_run_manifest_sha256": SOURCE_RUN_MANIFEST_SHA256,
        "source_evidence_run_manifest_sha256": SOURCE_EVIDENCE_RUN_MANIFEST_SHA256,
        "source_core_sha256": SOURCE_CORE_SHA256,
        "run_manifest_sha256": run_manifest_sha256,
        "campaign_artifact_hashes": campaign_hashes,
        "environment": EXPECTED_ENVIRONMENT,
        "counts": {
            "catalogs": expected_runs,
            "profiles_per_catalog": 5,
            "profile_cells": total_cells,
            "items_per_cell": 400,
            "reward_replay_episodes": total_cells * EPISODES,
            "topsis_coalition_evaluations": expected_runs * 400 * 16,
            "numeric_values_checked": checks,
        },
        "gates": [
            {"gate_id": "G01_verified_bridge_and_hash_chain", "status": "PASS"},
            {"gate_id": "G02_allowlist_extraction", "status": "PASS"},
            {"gate_id": "G03_same_rng_action_reward_q_replay", "status": "PASS"},
            {"gate_id": "G04_q_reward_component_efficiency", "status": "PASS", "tolerance": TOL},
            {"gate_id": "G05_score_space_cq_affine_decomposition", "status": "PASS", "tolerance": TOL},
            {"gate_id": "G06_exact_16_coalition_topsis_shapley", "status": "PASS", "tolerance": TOL},
            {"gate_id": "G07_full_score_and_top7_rank_reconstruction", "status": "PASS", "tolerance": TOL},
            {"gate_id": "G08_strict_schema_finite_and_containment", "status": "PASS"},
            {"gate_id": "G09_independent_verifier_code_path", "status": "PASS", "runner_imported": False},
        ],
        "maximum_reconstruction_abs_error": maximum_error,
        "output_hashes": {
            "xai_inputs.jsonl": file_hash(output / "xai_inputs.jsonl"),
            "xai_attributions.jsonl": file_hash(output / "xai_attributions.jsonl"),
            "xai_results.json": file_hash(output / "xai_results.json"),
            "status.json": file_hash(output / "status.json"),
            "verification_status.json": file_hash(status_path),
        },
        "verifier_sha256": file_hash(Path(__file__)),
    }
    reject_denied_keys(full)
    finite_tree(full)
    atomic(full_path, full)
    print(f"verify_progress=100.000% cells={total_cells}/{total_cells} status=completed_verified", flush=True)


def options() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--source-data-campaign", type=Path, required=True)
    p.add_argument("--source-data-root", type=Path, required=True)
    p.add_argument("--source-evidence-campaign", type=Path, required=True)
    p.add_argument("--source-evidence-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "canonical"), required=True)
    return p.parse_args()


if __name__ == "__main__":
    parsed = options()
    try:
        verify(parsed)
    except Exception as exc:
        try:
            failure_path = parsed.output_dir.resolve() / "verification_status.json"
            if not (parsed.output_dir.resolve() / "FULL_VERIFICATION.json").exists():
                prior = object_json(failure_path) if failure_path.is_file() else {
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
                atomic(failure_path, prior)
        except Exception:
            pass
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
