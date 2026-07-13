"""Producer-environment, same-target drift bridge.

Legacy arms reproduce the public submission artifacts exactly. Corrected arms
train on the full 400-item catalog with no evaluation-label reward. Their
training functions have no target argument and create no ground-truth object;
targets are constructed only by the separate evaluation functions. Console
output is progress/health only, never a partial scientific result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = CAMPAIGN_ROOT / "inputs"
CORE_PATH = CAMPAIGN_ROOT / "src" / "original_hybrid_core.py"
MANIFEST_PATH = INPUT_ROOT / "data" / "processed" / "manifest.json"
PRIMARY_RESULT_PATH = INPUT_ROOT / "results" / "amazon_primary.json"
SUDDEN_RESULT_PATH = INPUT_ROOT / "results" / "amazon_drift.json"
GRADUAL_RESULT_PATH = INPUT_ROOT / "results" / "validation_extensions.json"
PRODUCER_ROOT = INPUT_ROOT / "producer_provenance" / "v1.0-submission"
EXACT_REPLAY_PATH = INPUT_ROOT / "exact_replay_provenance" / "EXACT_REPLAY_AUDIT.json"
HISTORICAL_ROOT = INPUT_ROOT / "historical_reward_provenance"
RUN_MANIFEST_PATH = CAMPAIGN_ROOT / "RUN_MANIFEST.json"

EXPECTED_ENVIRONMENT = {"numpy": "1.26.0", "pandas": "2.2.3", "scipy": "1.16.3"}
PRODUCER_COMMIT = "3b92f6485d20d1a45ac03b60077d20af08060885"
EXPECTED_HASHES = {
    "inputs/design/SAME_TARGET_BRIDGE_PROTOCOL_2026-07-12.md": "da8bb62b606efcd670214cfeb7509fe452fbdfd36823ee8c850eb9e5871c128f",
    "src/original_hybrid_core.py": "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f",
    "inputs/data/processed/manifest.json": "81b01f5580109552fc6086c67441159ddad40c1d1447f1061a092a88c6c89652",
    "inputs/results/amazon_primary.json": "cfeaff03084df0d3f0a07a5c8c40308027ca7980288a89cf3616c588d0791ce4",
    "inputs/results/amazon_drift.json": "bd8e3523938219186c0c92a963a7c5392f0aa3710db30d8fabf07497444c7454",
    "inputs/results/validation_extensions.json": "1c1700c98a2991ac37c4ba424ec175d0ed17166c5d136259c5a9a3e3af242c5e",
    "inputs/producer_provenance/v1.0-submission/code/hybrid_core.py": "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f",
    "inputs/producer_provenance/v1.0-submission/code/run_amazon_experiments.py": "361f29012b1618c164d9688bd4887ca6187fc04ec67bc8dd6a9de54fd0a2f15d",
    "inputs/producer_provenance/v1.0-submission/code/validation_extensions.py": "4f751381eb9d05578f08c32ffa1e2d5ec47cfd380866f175752660b87704c5a9",
    "inputs/producer_provenance/v1.0-submission/requirements.txt": "5241d0abaccd86ffad73f36592acbabb1bf9331be83dd4678b4ef5d6be71f391",
    "inputs/producer_provenance/v1.0-submission/COMMIT_METADATA.json": "35b41ec3c5a9f4dd1325839aa90563767f71c9624815c302f4a1ce18892f2dad",
    "inputs/producer_provenance/v1.0-submission/PUBLIC_TAG_EXACT_HASHES.json": "9351463c48ea2d7da9df37b230f8cac5ce5dd29ea091cc5ac7a78787298f3315",
    "inputs/exact_replay_provenance/EXACT_REPLAY_AUDIT.json": "ba76de6b8043ac4baf34f4f5763c96b2c778d458ec497febc97b6f9135aa655d",
    "inputs/historical_reward_provenance/hybrid_rl_mcdm_v2.py": "90af7d4d3150099d840c510f5ff420b8773659c2a7a579d04e7b6e711da65e4f",
    "inputs/historical_reward_provenance/supplementary_runs.py": "a18af10d9d7c2c81e400910ec1d0dae4071322dc1fc24aa0f3b6e022984d8bdc",
}

SUDDEN_CHECKPOINTS = (2000, 5000, 10000, 14000, 15000, 16000, 20000, 25000, 30000)
GRADUAL_CHECKPOINTS = (5000, 10000, 15000, 20000, 25000, 30000)
SUDDEN_AUC_GRID = (15000, 16000, 20000, 25000, 30000)
GRADUAL_AUC_GRID = (10000, 15000, 20000, 25000, 30000)
REWARD_MODELS = (
    "inclusive_range_fix",
    "component_continuous_fix",
    "historical_funnel_coefficients_on_may_h",
)
PRIMARY_REWARD_MODEL = "component_continuous_fix"
METHODS = ("rl_only", "hybrid")
DRIFT_START = 10000
DRIFT_END = 25000
SUDDEN_BOUNDARY = 15000
EPS_INIT = 0.30
EPS_DECAY = 0.9997
EPS_MIN = 0.05
LEARNING_RATE = 0.05
BOOTSTRAP_REPS = 20_000
PAIR_TOLERANCE = 1e-12
RUN_MANIFEST_CONTRACT = {
    "sudden_catalogs": 50,
    "gradual_catalogs": 30,
    "profiles": 5,
    "episodes": 30000,
    "sudden_boundary": {"pre_final": 15000, "post_first": 15001},
    "gradual_steps": 41,
    "sudden_checkpoints": list(SUDDEN_CHECKPOINTS),
    "gradual_checkpoints": list(GRADUAL_CHECKPOINTS),
    "exact_legacy_raw_cells": {"sudden": 900, "gradual": 540},
    "corrected_candidate": "full_catalog_400",
    "corrected_gt_bonus": 0.0,
    "corrected_reward_models": list(REWARD_MODELS),
    "primary_corrected_reward": PRIMARY_REWARD_MODEL,
    "targets": "evaluation_only_in_corrected_arms",
    "verifier": "independent_full_stochastic_replay",
}
LOCK_REQUIRED_STATIC_PATHS = frozenset(
    {
        "PROTOCOL_LOCK.md",
        "src/drift_main.py",
        "src/lock_campaign.py",
        "src/original_hybrid_core.py",
        "verify_drift.py",
        "tests/test_drift_contract.py",
        "tests/test_verify_drift.py",
        "inputs/design/SAME_TARGET_BRIDGE_PROTOCOL_2026-07-12.md",
        "inputs/data/processed/manifest.json",
        "inputs/results/amazon_primary.json",
        "inputs/results/amazon_drift.json",
        "inputs/results/validation_extensions.json",
        "inputs/producer_provenance/v1.0-submission/code/hybrid_core.py",
        "inputs/producer_provenance/v1.0-submission/code/run_amazon_experiments.py",
        "inputs/producer_provenance/v1.0-submission/code/validation_extensions.py",
        "inputs/producer_provenance/v1.0-submission/requirements.txt",
        "inputs/producer_provenance/v1.0-submission/COMMIT_METADATA.json",
        "inputs/producer_provenance/v1.0-submission/PUBLIC_TAG_EXACT_HASHES.json",
        "inputs/exact_replay_provenance/EXACT_REPLAY_AUDIT.json",
        "inputs/historical_reward_provenance/hybrid_rl_mcdm_v2.py",
        "inputs/historical_reward_provenance/supplementary_runs.py",
    }
)
RUNNER_RESUME_ALLOWLIST = frozenset({"sealed_records.jsonl", "STATUS.json", "PROGRESS.json"})


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_module("drift_frozen_core", CORE_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt checkpoint JSONL line {number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Checkpoint line {number} is not an object")
        rows.append(value)
    return rows


def assert_exact_environment() -> dict[str, str]:
    actual = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
    if actual != EXPECTED_ENVIRONMENT:
        raise RuntimeError(f"Producer environment mismatch: actual={actual} expected={EXPECTED_ENVIRONMENT}")
    return actual


def assert_frozen_inputs() -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = CAMPAIGN_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen hash mismatch: {relative}: {actual} != {expected}")
        verified[relative] = actual
    if EXPECTED_HASHES["src/original_hybrid_core.py"] != EXPECTED_HASHES[
        "inputs/producer_provenance/v1.0-submission/code/hybrid_core.py"
    ]:
        raise AssertionError("Executable core is not the exact public-tag core")
    commit = json.loads((PRODUCER_ROOT / "COMMIT_METADATA.json").read_text(encoding="utf-8"))
    public = json.loads((PRODUCER_ROOT / "PUBLIC_TAG_EXACT_HASHES.json").read_text(encoding="utf-8"))
    if commit.get("commit_sha1") != PRODUCER_COMMIT or public.get("public_ls_remote_commit_sha1") != PRODUCER_COMMIT:
        raise ValueError("Producer public tag/commit mismatch")
    public_map = {entry["path"]: entry for entry in public.get("files", [])}
    public_expected = {
        "code/hybrid_core.py": EXPECTED_HASHES["inputs/producer_provenance/v1.0-submission/code/hybrid_core.py"],
        "code/run_amazon_experiments.py": EXPECTED_HASHES["inputs/producer_provenance/v1.0-submission/code/run_amazon_experiments.py"],
        "code/validation_extensions.py": EXPECTED_HASHES["inputs/producer_provenance/v1.0-submission/code/validation_extensions.py"],
        "requirements.txt": EXPECTED_HASHES["inputs/producer_provenance/v1.0-submission/requirements.txt"],
        "results/amazon_drift.json": EXPECTED_HASHES["inputs/results/amazon_drift.json"],
        "results/validation_extensions.json": EXPECTED_HASHES["inputs/results/validation_extensions.json"],
    }
    for relative, expected in public_expected.items():
        if public_map.get(relative, {}).get("sha256") != expected:
            raise ValueError(f"Public-tag provenance metadata mismatch: {relative}")
    replay = json.loads(EXACT_REPLAY_PATH.read_text(encoding="utf-8"))
    if not (
        replay.get("status") == "completed"
        and replay.get("all_exact") is True
        and replay.get("cells_total") == 250
        and replay.get("cells_exact") == 250
        and replay.get("cells_mismatched") == 0
        and replay.get("numpy") == "1.26.0"
        and replay.get("pandas") == "2.2.3"
    ):
        raise ValueError("250-cell exact producer gate is not terminal PASS")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 50:
        raise ValueError("Frozen catalog manifest must contain 50 runs")
    for expected_index, entry in enumerate(runs):
        if entry.get("run_index") != expected_index:
            raise ValueError("Frozen catalog ordering mismatch")
        relative = str(entry["path"]).replace("\\", "/")
        path = INPUT_ROOT / Path(relative)
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]).lower():
            raise ValueError(f"Frozen catalog mismatch: {relative}")
    return verified


def expected_run_manifest_paths() -> set[str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 50:
        raise ValueError("Cannot derive lock paths without exactly 50 catalogs")
    catalog_paths = {
        "inputs/" + str(entry["path"]).replace("\\", "/")
        for entry in runs
    }
    if len(catalog_paths) != 50:
        raise ValueError("Catalog lock paths are not unique")
    return set(LOCK_REQUIRED_STATIC_PATHS) | catalog_paths


def validate_run_manifest_payload(payload: Mapping[str, Any], check_files: bool = True) -> None:
    expected_top_level = {
        "schema_version", "campaign_id", "status", "created_at", "tool", "model",
        "operation_id", "hash_algorithm", "contract", "environment",
        "producer_provenance", "files",
    }
    if set(payload) != expected_top_level:
        raise ValueError("Run-manifest top-level schema mismatch")
    if payload.get("schema_version") != "same_target_drift.run_manifest.v1":
        raise ValueError("Unexpected run-manifest schema")
    if payload.get("hash_algorithm") != "SHA-256" or payload.get("tool") != "Codex" or payload.get("model") != "GPT-5 Codex":
        raise ValueError("Run-manifest writer/hash metadata mismatch")
    if not isinstance(payload.get("created_at"), str) or not isinstance(payload.get("operation_id"), str):
        raise ValueError("Run-manifest timestamp/operation metadata mismatch")
    if payload.get("campaign_id") != CAMPAIGN_ROOT.name or payload.get("status") != "LOCKED_BEFORE_COMPUTE":
        raise ValueError("Run-manifest campaign/status mismatch")
    if payload.get("contract") != RUN_MANIFEST_CONTRACT:
        raise ValueError("Run-manifest scientific contract mismatch")
    environment = payload.get("environment")
    if (
        not isinstance(environment, dict)
        or set(environment) != {"python", "platform", "packages"}
        or environment.get("python") != sys.version
        or environment.get("platform") != platform.platform()
        or environment.get("packages") != EXPECTED_ENVIRONMENT
    ):
        raise ValueError("Run-manifest environment mismatch")
    if payload.get("producer_provenance") != {"tag": "v1.0-submission", "commit_sha1": PRODUCER_COMMIT}:
        raise ValueError("Run-manifest producer provenance mismatch")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Run-manifest file list missing")
    relative_paths = [str(entry.get("path")) for entry in entries]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Duplicate run-manifest file path")
    expected_paths = expected_run_manifest_paths()
    if set(relative_paths) != expected_paths:
        missing = sorted(expected_paths - set(relative_paths))
        extra = sorted(set(relative_paths) - expected_paths)
        raise ValueError(f"Run-manifest exact path-set mismatch: missing={missing} extra={extra}")
    if not check_files:
        return
    root = CAMPAIGN_ROOT.resolve()
    for entry in entries:
        if set(entry) != {"path", "sha256", "bytes"}:
            raise ValueError(f"Run-manifest file entry schema mismatch: {entry.get('path')}")
        path = (CAMPAIGN_ROOT / str(entry["path"])).resolve()
        if root != path and root not in path.parents:
            raise ValueError("Run-manifest path escapes campaign")
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Run-manifest file missing/size mismatch: {entry['path']}")
        if sha256_file(path) != str(entry["sha256"]).lower():
            raise ValueError(f"Run-manifest file hash mismatch: {entry['path']}")


def assert_run_manifest() -> str:
    if not RUN_MANIFEST_PATH.is_file():
        raise FileNotFoundError("Missing immutable pre-compute RUN_MANIFEST.json")
    payload = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_run_manifest_payload(payload, check_files=True)
    return sha256_file(RUN_MANIFEST_PATH)


def load_catalog(run_entry: Mapping[str, Any]) -> pd.DataFrame:
    relative = str(run_entry["path"]).replace("\\", "/")
    path = INPUT_ROOT / Path(relative)
    if sha256_file(path) != str(run_entry["sha256"]).lower():
        raise ValueError(f"Catalog hash drift: {relative}")
    df = pd.read_csv(path)
    if len(df) != 400:
        raise ValueError(f"Catalog must contain 400 products: {relative}")
    return df


def array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def state_digest(q: np.ndarray, visits: np.ndarray) -> str:
    return hashlib.sha256((array_digest(q) + array_digest(visits)).encode("ascii")).hexdigest()


def shifted_profile(profile: Mapping[str, object]) -> dict[str, Any]:
    shifted = CORE.flip_brand_preferences(profile)
    ranked = sorted(shifted["cat_affinity"].items(), key=lambda item: item[1])
    shifted["cat_affinity"] = {
        ranked[0][0]: ranked[2][1],
        ranked[1][0]: ranked[1][1],
        ranked[2][0]: ranked[0][1],
    }
    lo, hi = profile["price_range"]
    width = hi - lo
    center = (lo + hi) / 2.0
    new_center = center + 250.0 if center < 350.0 else center - 250.0
    shifted["price_range"] = (max(10.0, new_center - width / 2.0), min(1000.0, new_center + width / 2.0))
    shifted["recency_weight"] = float(np.clip(1.0 - float(profile["recency_weight"]), 0.10, 0.90))
    return shifted


def interpolate_profile(pre: Mapping[str, object], post: Mapping[str, object], frac: float) -> dict[str, Any]:
    frac = float(np.clip(frac, 0.0, 1.0))
    brand_keys = sorted(set(pre["brand_pref"]) | set(post["brand_pref"]))
    cat_keys = sorted(set(pre["cat_affinity"]) | set(post["cat_affinity"]))
    pre_lo, pre_hi = pre["price_range"]
    post_lo, post_hi = post["price_range"]
    return {
        "brand_pref": {k: (1-frac)*float(pre["brand_pref"].get(k, 0.0)) + frac*float(post["brand_pref"].get(k, 0.0)) for k in brand_keys},
        "cat_affinity": {k: (1-frac)*float(pre["cat_affinity"].get(k, 0.0)) + frac*float(post["cat_affinity"].get(k, 0.0)) for k in cat_keys},
        "price_range": ((1-frac)*pre_lo + frac*post_lo, (1-frac)*pre_hi + frac*post_hi),
        "recency_weight": (1-frac)*float(pre["recency_weight"]) + frac*float(post["recency_weight"]),
    }


def gradual_fraction(episode: int) -> float:
    if episode <= DRIFT_START:
        return 0.0
    if episode >= DRIFT_END:
        return 1.0
    return (episode - DRIFT_START) / (DRIFT_END - DRIFT_START)


def reward_probabilities(df: pd.DataFrame, profile: Mapping[str, object], model: str) -> tuple[np.ndarray, np.ndarray]:
    brand = np.asarray([profile["brand_pref"].get(value, 0.10) for value in df["brand"]], dtype=float)
    lo, hi = profile["price_range"]
    recency_weight = float(profile["recency_weight"])
    recency = df["recency_pct"].to_numpy(dtype=float) * recency_weight + (1.0 - recency_weight) * 0.5
    if model == "inclusive_range_fix":
        price = df["price"].to_numpy(dtype=float)
        price_component = ((price >= float(lo)) & (price <= float(hi))).astype(float)
        scale = max(float(v) for v in profile["cat_affinity"].values())
        category = np.asarray([float(profile["cat_affinity"].get(v, 0.0))/scale for v in df["category"]], dtype=float)
        p_engage = np.clip(0.40*brand + 0.35*price_component + 0.15*category + 0.10*recency, 0, 1)
        p_convert = np.clip(0.50*brand + 0.30*price_component + 0.20*category, 0, 1)
    elif model == "component_continuous_fix":
        components = CORE.hidden_components(df, profile)
        category = np.asarray(components["cat_score"], dtype=float)
        price_component = np.asarray(components["price_fit"], dtype=float)
        p_engage = np.clip(0.40*brand + 0.35*price_component + 0.15*category + 0.10*recency, 0, 1)
        p_convert = np.clip(0.50*brand + 0.30*price_component + 0.20*category, 0, 1)
    elif model == "historical_funnel_coefficients_on_may_h":
        hidden = CORE.hidden_utility(df, profile)
        p_engage = np.clip(0.70*hidden + 0.10, 0.05, 0.95)
        p_convert = np.clip(0.50*hidden, 0.02, 0.80)
    else:
        raise KeyError(model)
    if not np.all(np.isfinite(p_engage)) or not np.all(np.isfinite(p_convert)):
        raise ValueError("Nonfinite reward probability")
    return p_engage, p_convert


def legacy_reward_probabilities(df: pd.DataFrame, profile: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    brand = np.asarray([profile["brand_pref"].get(v, 0.10) for v in df["brand"]], dtype=float)
    lo, hi = profile["price_range"]
    center = (lo + hi) / 2.0
    half_range = (hi - lo) / 2.0 + 1.0
    price_fit = np.clip(1.0 - np.abs(df["price"].to_numpy(dtype=float) - center) / half_range, 0, 1)
    in_range = (price_fit > 0.999999).astype(float)
    category = np.ones(len(df), dtype=float)
    recency_weight = float(profile["recency_weight"])
    recency = df["recency_pct"].to_numpy(dtype=float) * recency_weight + (1.0 - recency_weight) * 0.5
    return (
        np.clip(0.40*brand + 0.35*in_range + 0.15*category + 0.10*recency, 0, 1),
        np.clip(0.50*brand + 0.30*in_range + 0.20*category, 0, 1),
    )


def _state_payload(q: np.ndarray, visits: np.ndarray, topsis: np.ndarray) -> dict[str, Any]:
    rl_rank = [int(x) for x in CORE.top_k_ranking(q)]
    hybrid_rank = [int(x) for x in CORE.top_k_ranking(CORE.static_hybrid_score(q, topsis, lambda_q=0.50))]
    return {
        "q_sha256": array_digest(q),
        "visits_sha256": array_digest(visits),
        "state_sha256": state_digest(q, visits),
        "rl_rank": rl_rank,
        "hybrid_rank": hybrid_rank,
    }


def legacy_sudden_profile(df: pd.DataFrame, profile_name: str, profile_idx: int, seed: int, topsis: np.ndarray) -> dict[str, Any]:
    pre = CORE.PROFILE_HIDDEN[profile_name]
    post = CORE.flip_brand_preferences(pre)
    pre_seed = CORE.profile_seed(seed, profile_name)
    pre_gt = CORE.top_k_set(CORE.build_ground_truth(df, pre, pre_seed, observable_alpha=CORE.MAIN_ALPHA))
    post_gt = CORE.top_k_set(CORE.build_ground_truth(df, post, pre_seed + 5000, observable_alpha=CORE.MAIN_ALPHA))
    pool = CORE.build_candidate_pool(df, [pre, post], [pre_gt, post_gt])
    q = np.zeros(len(df), dtype=float)
    visits = np.zeros(len(df), dtype=np.int32)
    eps = EPS_INIT
    reward_rng = np.random.RandomState(seed + profile_idx * 997)
    act_rng = np.random.RandomState(seed + profile_idx * 13)
    pre_engage, pre_convert = legacy_reward_probabilities(df, pre)
    post_engage, post_convert = legacy_reward_probabilities(df, post)
    pre_mask = np.zeros(len(df), dtype=bool); pre_mask[list(pre_gt)] = True
    post_mask = np.zeros(len(df), dtype=bool); post_mask[list(post_gt)] = True
    checkpoints: dict[str, Any] = {}
    for episode in range(1, SUDDEN_CHECKPOINTS[-1] + 1):
        action = int(act_rng.choice(pool)) if act_rng.random() < eps else int(pool[np.argmax(q[pool])])
        if episode > SUDDEN_BOUNDARY:
            p_engage, p_convert, mask, eval_gt = post_engage, post_convert, post_mask, post_gt
        else:
            p_engage, p_convert, mask, eval_gt = pre_engage, pre_convert, pre_mask, pre_gt
        reward = -0.02
        if reward_rng.random() < p_engage[action]:
            reward += 0.30
            if reward_rng.random() < p_convert[action]: reward += 1.00
        if mask[action]: reward += 0.20
        visits[action] += 1
        q[action] += LEARNING_RATE * (reward - q[action])
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if episode in SUDDEN_CHECKPOINTS:
            state = _state_payload(q.copy(), visits.copy(), topsis)
            state["f1"] = {
                "rl_only": float(CORE.f1_score(set(state["rl_rank"]), eval_gt)),
                "hybrid": float(CORE.f1_score(set(state["hybrid_rank"]), eval_gt)),
            }
            checkpoints[str(episode)] = state
    return {"profile_name": profile_name, "checkpoints": checkpoints, "epsilon_final": float(eps)}


def gradual_eval_profile(pre: Mapping[str, object], post: Mapping[str, object], episode: int) -> tuple[int, dict[str, Any]]:
    key = int(round(gradual_fraction(episode) * 40))
    return key, interpolate_profile(pre, post, key / 40.0)


def legacy_gradual_profile(df: pd.DataFrame, profile_name: str, profile_idx: int, seed: int, topsis: np.ndarray) -> dict[str, Any]:
    pre = CORE.PROFILE_HIDDEN[profile_name]
    post = shifted_profile(pre)
    q = np.zeros(len(df), dtype=float)
    visits = np.zeros(len(df), dtype=np.int32)
    eps = EPS_INIT
    reward_rng = np.random.RandomState(seed + profile_idx * 2221)
    act_rng = np.random.RandomState(seed + profile_idx * 4441)
    pre_gt = CORE.top_k_set(CORE.build_ground_truth(df, pre, CORE.profile_seed(seed, profile_name), observable_alpha=CORE.MAIN_ALPHA))
    post_gt = CORE.top_k_set(CORE.build_ground_truth(df, post, CORE.profile_seed(seed, profile_name) + 5000, observable_alpha=CORE.MAIN_ALPHA))
    pool = CORE.build_candidate_pool(df, [pre, post], [pre_gt, post_gt])
    topsis_rank = [int(x) for x in CORE.top_k_ranking(topsis)]
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]] = {}
    checkpoints: dict[str, Any] = {}
    for episode in range(1, GRADUAL_CHECKPOINTS[-1] + 1):
        key, profile = gradual_eval_profile(pre, post, episode)
        if key not in cache:
            gt_seed = CORE.profile_seed(seed, profile_name) + 7000 + key
            gt = CORE.top_k_set(CORE.build_ground_truth(df, profile, gt_seed, observable_alpha=CORE.MAIN_ALPHA))
            p_engage, p_convert = legacy_reward_probabilities(df, profile)
            mask = np.zeros(len(df), dtype=bool); mask[list(gt)] = True
            cache[key] = (p_engage, p_convert, mask, gt)
        p_engage, p_convert, mask, eval_gt = cache[key]
        action = int(act_rng.choice(pool)) if act_rng.random() < eps else int(pool[np.argmax(q[pool])])
        reward = -0.02
        if reward_rng.random() < p_engage[action]:
            reward += 0.30
            if reward_rng.random() < p_convert[action]: reward += 1.00
        if mask[action]: reward += 0.20
        visits[action] += 1
        q[action] += LEARNING_RATE * (reward - q[action])
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if episode in GRADUAL_CHECKPOINTS:
            state = _state_payload(q.copy(), visits.copy(), topsis)
            state["topsis_rank"] = topsis_rank
            state["target_key"] = key
            state["f1"] = {
                "topsis_only": float(CORE.f1_score(set(topsis_rank), eval_gt)),
                "rl_only": float(CORE.f1_score(set(state["rl_rank"]), eval_gt)),
                "hybrid": float(CORE.f1_score(set(state["hybrid_rank"]), eval_gt)),
            }
            checkpoints[str(episode)] = state
    return {"profile_name": profile_name, "checkpoints": checkpoints, "epsilon_final": float(eps)}


def corrected_sudden_train(df: pd.DataFrame, profile_name: str, profile_idx: int, seed: int, topsis: np.ndarray, reward_model: str, checkpoints: Sequence[int] = SUDDEN_CHECKPOINTS) -> dict[str, Any]:
    """Future-blind training: no ground-truth/target object is accepted or built."""
    pre = CORE.PROFILE_HIDDEN[profile_name]
    post = CORE.flip_brand_preferences(pre)
    q = np.zeros(len(df), dtype=float)
    visits = np.zeros(len(df), dtype=np.int32)
    eps = EPS_INIT
    reward_rng = np.random.RandomState(seed + profile_idx * 997)
    act_rng = np.random.RandomState(seed + profile_idx * 13)
    pool = np.arange(len(df), dtype=int)
    pre_engage, pre_convert = reward_probabilities(df, pre, reward_model)
    post_engage, post_convert = reward_probabilities(df, post, reward_model)
    states: dict[str, Any] = {}
    checkpoint_set = set(int(x) for x in checkpoints)
    for episode in range(1, max(checkpoints) + 1):
        action = int(act_rng.choice(pool)) if act_rng.random() < eps else int(pool[np.argmax(q[pool])])
        p_engage, p_convert = (post_engage, post_convert) if episode > SUDDEN_BOUNDARY else (pre_engage, pre_convert)
        reward = -0.02
        if reward_rng.random() < p_engage[action]:
            reward += 0.30
            if reward_rng.random() < p_convert[action]: reward += 1.00
        visits[action] += 1
        q[action] += LEARNING_RATE * (reward - q[action])
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if episode in checkpoint_set:
            states[str(episode)] = _state_payload(q.copy(), visits.copy(), topsis)
    return {"profile_name": profile_name, "states": states, "epsilon_final": float(eps)}


def corrected_gradual_train(df: pd.DataFrame, profile_name: str, profile_idx: int, seed: int, topsis: np.ndarray, reward_model: str, checkpoints: Sequence[int] = GRADUAL_CHECKPOINTS) -> dict[str, Any]:
    """Future-blind training: phase profiles affect reward, targets never do."""
    pre = CORE.PROFILE_HIDDEN[profile_name]
    post = shifted_profile(pre)
    q = np.zeros(len(df), dtype=float)
    visits = np.zeros(len(df), dtype=np.int32)
    eps = EPS_INIT
    reward_rng = np.random.RandomState(seed + profile_idx * 2221)
    act_rng = np.random.RandomState(seed + profile_idx * 4441)
    pool = np.arange(len(df), dtype=int)
    probability_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    states: dict[str, Any] = {}
    checkpoint_set = set(int(x) for x in checkpoints)
    for episode in range(1, max(checkpoints) + 1):
        key, profile = gradual_eval_profile(pre, post, episode)
        if key not in probability_cache:
            probability_cache[key] = reward_probabilities(df, profile, reward_model)
        p_engage, p_convert = probability_cache[key]
        action = int(act_rng.choice(pool)) if act_rng.random() < eps else int(pool[np.argmax(q[pool])])
        reward = -0.02
        if reward_rng.random() < p_engage[action]:
            reward += 0.30
            if reward_rng.random() < p_convert[action]: reward += 1.00
        visits[action] += 1
        q[action] += LEARNING_RATE * (reward - q[action])
        eps = max(EPS_MIN, eps * EPS_DECAY)
        if episode in checkpoint_set:
            state = _state_payload(q.copy(), visits.copy(), topsis)
            state["target_key"] = key
            states[str(episode)] = state
    return {"profile_name": profile_name, "states": states, "epsilon_final": float(eps)}


def sudden_targets(df: pd.DataFrame, profile_name: str, seed: int, variant_offset: int = 0) -> dict[str, set[int]]:
    pre = CORE.PROFILE_HIDDEN[profile_name]
    post = CORE.flip_brand_preferences(pre)
    base = CORE.profile_seed(seed, profile_name)
    return {
        "pre": CORE.top_k_set(CORE.build_ground_truth(df, pre, base, observable_alpha=CORE.MAIN_ALPHA)),
        "post": CORE.top_k_set(CORE.build_ground_truth(df, post, base + 5000 + variant_offset, observable_alpha=CORE.MAIN_ALPHA)),
    }


def gradual_target(df: pd.DataFrame, profile_name: str, seed: int, episode: int, variant_offset: int = 0) -> tuple[int, set[int]]:
    pre = CORE.PROFILE_HIDDEN[profile_name]
    post = shifted_profile(pre)
    key, profile = gradual_eval_profile(pre, post, episode)
    gt_seed = CORE.profile_seed(seed, profile_name) + 7000 + key + variant_offset
    return key, CORE.top_k_set(CORE.build_ground_truth(df, profile, gt_seed, observable_alpha=CORE.MAIN_ALPHA))


def evaluate_sudden(df: pd.DataFrame, profile_name: str, seed: int, trained: Mapping[str, Any]) -> dict[str, Any]:
    targets = sudden_targets(df, profile_name, seed)
    output: dict[str, Any] = {}
    for cp_text, state in trained["states"].items():
        cp = int(cp_text)
        target = targets["post"] if cp > SUDDEN_BOUNDARY else targets["pre"]
        row = dict(state)
        row["target_phase"] = "post" if cp > SUDDEN_BOUNDARY else "pre"
        row["target_sha256"] = canonical_sha256({"set": sorted(target)})
        row["f1"] = {m: float(CORE.f1_score(set(state[f"{m.split('_')[0]}_rank"]), target)) for m in METHODS}
        row["ndcg"] = {m: float(CORE.ndcg_at_k(state[f"{m.split('_')[0]}_rank"], target)) for m in METHODS}
        output[cp_text] = row
    return {"profile_name": profile_name, "checkpoints": output, "epsilon_final": trained["epsilon_final"]}


def evaluate_gradual(df: pd.DataFrame, profile_name: str, seed: int, trained: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for cp_text, state in trained["states"].items():
        key, target = gradual_target(df, profile_name, seed, int(cp_text))
        if state.get("target_key") != key:
            raise AssertionError("Gradual key drift between training profile and evaluation")
        row = dict(state)
        row["target_sha256"] = canonical_sha256({"set": sorted(target)})
        row["f1"] = {m: float(CORE.f1_score(set(state[f"{m.split('_')[0]}_rank"]), target)) for m in METHODS}
        row["ndcg"] = {m: float(CORE.ndcg_at_k(state[f"{m.split('_')[0]}_rank"], target)) for m in METHODS}
        output[cp_text] = row
    return {"profile_name": profile_name, "checkpoints": output, "epsilon_final": trained["epsilon_final"]}


def future_blind_source_contract() -> dict[str, Any]:
    forbidden = {"build_ground_truth", "sudden_targets", "gradual_target"}
    checked: dict[str, Any] = {}
    for function in (corrected_sudden_train, corrected_gradual_train):
        tree = ast.parse(inspect.getsource(function))
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))}
        bad = sorted(forbidden & calls)
        signature = inspect.signature(function)
        bad_params = [name for name in signature.parameters if "target" in name.lower() or name.lower().startswith("gt")]
        if bad or bad_params:
            raise AssertionError(f"Corrected training is not future blind: {function.__name__}: {bad}/{bad_params}")
        checked[function.__name__] = {"forbidden_calls_absent": True, "target_parameters_absent": True}
    return checked


def metamorphic_preflight(df: pd.DataFrame, seed: int, topsis: np.ndarray) -> dict[str, Any]:
    proof: dict[str, Any] = {}
    profile_name = CORE.PROFILE_ORDER[0]
    for model in REWARD_MODELS:
        a = corrected_sudden_train(df, profile_name, 0, seed, topsis, model, checkpoints=(128,))
        b = corrected_sudden_train(df, profile_name, 0, seed, topsis, model, checkpoints=(128,))
        if a["states"]["128"]["state_sha256"] != b["states"]["128"]["state_sha256"]:
            raise AssertionError("Sudden deterministic prefix invariance failed")
        original = sudden_targets(df, profile_name, seed, 0)["post"]
        altered = sudden_targets(df, profile_name, seed, 100000)["post"]
        if original == altered:
            raise AssertionError("Sudden metamorphic future target did not change")
        c = corrected_gradual_train(df, profile_name, 0, seed, topsis, model, checkpoints=(128,))
        d = corrected_gradual_train(df, profile_name, 0, seed, topsis, model, checkpoints=(128,))
        if c["states"]["128"]["state_sha256"] != d["states"]["128"]["state_sha256"]:
            raise AssertionError("Gradual deterministic prefix invariance failed")
        _, gradual_original = gradual_target(df, profile_name, seed, 30000, 0)
        _, gradual_altered = gradual_target(df, profile_name, seed, 30000, 100000)
        if gradual_original == gradual_altered:
            raise AssertionError("Gradual metamorphic future target did not change")
        proof[model] = {
            "sudden_prefix_state_equal": True,
            "sudden_future_target_changed": True,
            "gradual_prefix_state_equal": True,
            "gradual_future_target_changed": True,
        }
    return proof


def aggregate_f1(profiles: Sequence[Mapping[str, Any]], methods: Sequence[str], checkpoints: Sequence[int]) -> dict[str, dict[str, float]]:
    return {
        method: {
            str(cp): float(np.mean([profile["checkpoints"][str(cp)]["f1"][method] for profile in profiles]))
            for cp in checkpoints
        }
        for method in methods
    }


def assert_legacy_raw_gate(run_index: int, sudden_profiles: Sequence[Mapping[str, Any]] | None, gradual_profiles: Sequence[Mapping[str, Any]] | None, sudden_stored: Mapping[str, Any], gradual_stored: Mapping[str, Any]) -> dict[str, int]:
    counts = {"sudden": 0, "gradual": 0}
    if sudden_profiles is not None:
        aggregate = aggregate_f1(sudden_profiles, METHODS, SUDDEN_CHECKPOINTS)
        for method in METHODS:
            for cp in SUDDEN_CHECKPOINTS:
                expected = float(sudden_stored["summary"][method][str(cp)]["raw"][run_index])
                if aggregate[method][str(cp)] != expected:
                    raise AssertionError(f"Legacy sudden raw-cell mismatch run={run_index} method={method} cp={cp}")
                counts["sudden"] += 1
    if gradual_profiles is not None:
        methods = ("topsis_only", "rl_only", "hybrid")
        aggregate = aggregate_f1(gradual_profiles, methods, GRADUAL_CHECKPOINTS)
        stored = gradual_stored["gradual_multidim_drift"]["summary"]
        for method in methods:
            for cp in GRADUAL_CHECKPOINTS:
                expected = float(stored[method][str(cp)]["raw"][run_index])
                if aggregate[method][str(cp)] != expected:
                    raise AssertionError(f"Legacy gradual raw-cell mismatch run={run_index} method={method} cp={cp}")
                counts["gradual"] += 1
    return counts


def build_run_record(run_index: int, run_entry: Mapping[str, Any], df: pd.DataFrame, include_sudden: bool, include_gradual: bool, sudden_stored: Mapping[str, Any], gradual_stored: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(run_entry["seed"])
    topsis = CORE.topsis_artifacts(df)["scores"]
    payload: dict[str, Any] = {
        "schema_version": "same_target_drift.catalog_record.v1",
        "campaign_id": CAMPAIGN_ROOT.name,
        "run_index": run_index,
        "run_seed": seed,
        "dataset_path": str(run_entry["path"]).replace("\\", "/"),
        "dataset_sha256": str(run_entry["sha256"]).lower(),
        "sudden": None,
        "gradual": None,
    }
    sudden_legacy = None
    gradual_legacy = None
    if include_sudden:
        sudden_legacy = [legacy_sudden_profile(df, name, idx, seed, topsis) for idx, name in enumerate(CORE.PROFILE_ORDER)]
        corrected = {}
        for model in REWARD_MODELS:
            corrected[model] = [
                evaluate_sudden(df, name, seed, corrected_sudden_train(df, name, idx, seed, topsis, model))
                for idx, name in enumerate(CORE.PROFILE_ORDER)
            ]
        payload["sudden"] = {"legacy_exact": sudden_legacy, "corrected_future_blind": corrected}
    if include_gradual:
        gradual_legacy = [legacy_gradual_profile(df, name, idx, seed, topsis) for idx, name in enumerate(CORE.PROFILE_ORDER)]
        corrected = {}
        for model in REWARD_MODELS:
            corrected[model] = [
                evaluate_gradual(df, name, seed, corrected_gradual_train(df, name, idx, seed, topsis, model))
                for idx, name in enumerate(CORE.PROFILE_ORDER)
            ]
        payload["gradual"] = {"legacy_exact": gradual_legacy, "corrected_future_blind": corrected}
    payload["legacy_gate_cells"] = assert_legacy_raw_gate(run_index, sudden_legacy, gradual_legacy, sudden_stored, gradual_stored)
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def validate_resume(records: Sequence[Mapping[str, Any]], manifest_runs: Sequence[Mapping[str, Any]], sudden_runs: int, gradual_runs: int) -> None:
    if len(records) > max(sudden_runs, gradual_runs):
        raise ValueError("Resume checkpoint contains too many catalog records")
    for expected_index, record in enumerate(records):
        if record.get("run_index") != expected_index or record.get("campaign_id") != CAMPAIGN_ROOT.name:
            raise ValueError("Resume record ordering/campaign mismatch")
        unsigned = dict(record); digest = unsigned.pop("payload_sha256", None)
        if digest != canonical_sha256(unsigned):
            raise ValueError(f"Resume record payload hash mismatch at run {expected_index}")
        entry = manifest_runs[expected_index]
        if record.get("run_seed") != entry["seed"] or record.get("dataset_sha256") != str(entry["sha256"]).lower():
            raise ValueError(f"Resume record input identity mismatch at run {expected_index}")
        expect_sudden = expected_index < sudden_runs
        expect_gradual = expected_index < gradual_runs
        if (record.get("sudden") is not None) != expect_sudden or (record.get("gradual") is not None) != expect_gradual:
            raise ValueError(f"Resume scenario coverage mismatch at run {expected_index}")
        for scenario, checkpoints, legacy_methods in (
            ("sudden", SUDDEN_CHECKPOINTS, METHODS),
            ("gradual", GRADUAL_CHECKPOINTS, ("topsis_only", "rl_only", "hybrid")),
        ):
            block = record.get(scenario)
            if block is None:
                continue
            legacy = block.get("legacy_exact")
            corrected = block.get("corrected_future_blind")
            if not isinstance(legacy, list) or len(legacy) != 5 or set(corrected or {}) != set(REWARD_MODELS):
                raise ValueError(f"Resume {scenario} arm/profile structure mismatch")
            for profile in legacy:
                if set(profile.get("checkpoints", {})) != {str(cp) for cp in checkpoints}:
                    raise ValueError(f"Resume {scenario} legacy checkpoint mismatch")
                for cp in checkpoints:
                    if set(profile["checkpoints"][str(cp)].get("f1", {})) != set(legacy_methods):
                        raise ValueError(f"Resume {scenario} legacy method mismatch")
            for profiles in corrected.values():
                if not isinstance(profiles, list) or len(profiles) != 5:
                    raise ValueError(f"Resume {scenario} corrected profile count mismatch")


def validate_output_start(output_dir: Path, resume: bool, mode: str, total: int) -> int:
    """Enforce an empty new run or an exact, progress-only resume checkpoint."""
    if not resume:
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError("New-run output directory must be truly empty")
        return 0
    if not output_dir.is_dir():
        raise FileNotFoundError("Resume requires an existing output directory")
    items = list(output_dir.iterdir())
    if any(not item.is_file() for item in items):
        raise FileExistsError("Resume output may not contain subdirectories")
    names = {item.name for item in items}
    if names != set(RUNNER_RESUME_ALLOWLIST):
        missing = sorted(set(RUNNER_RESUME_ALLOWLIST) - names)
        unexpected = sorted(names - set(RUNNER_RESUME_ALLOWLIST))
        raise FileExistsError(f"Resume artifact allowlist mismatch: missing={missing} unexpected={unexpected}")
    status = json.loads((output_dir / "STATUS.json").read_text(encoding="utf-8"))
    progress = json.loads((output_dir / "PROGRESS.json").read_text(encoding="utf-8"))
    sealed_hash = sha256_file(output_dir / "sealed_records.jsonl")
    for label, payload, schema in (
        ("STATUS", status, "same_target_drift.status.v1"),
        ("PROGRESS", progress, "same_target_drift.progress.v1"),
    ):
        if payload.get("schema_version") != schema or payload.get("campaign_id") != CAMPAIGN_ROOT.name:
            raise ValueError(f"Resume {label} identity/schema mismatch")
        if payload.get("mode") != mode or payload.get("total_catalogs") != total:
            raise ValueError(f"Resume {label} mode/total mismatch")
        if payload.get("scientific_metrics_exposed") is not False:
            raise ValueError(f"Resume {label} violates checkpoint blindness")
        if payload.get("sealed_records_sha256") != sealed_hash:
            raise ValueError(f"Resume {label} sealed-record hash mismatch")
    if status.get("status") != "running":
        raise ValueError("Resume STATUS must be running")
    completed = int(progress.get("completed_catalogs", -1))
    if completed <= 0 or completed >= total or status.get("completed_catalogs") != completed:
        raise ValueError("Resume completed-catalog count is not a strict partial checkpoint")
    expected_percent = 100.0 * completed / total
    reported_percent = float(progress.get("percent", -1.0))
    if not math.isfinite(reported_percent) or abs(reported_percent - expected_percent) > 1e-12:
        raise ValueError("Resume progress percent mismatch")
    return completed


def normalized_auc(values: Sequence[float], grid: Sequence[int]) -> float:
    return float(np.trapz(np.asarray(values, dtype=float), np.asarray(grid, dtype=float)) / (grid[-1] - grid[0]))


def analysis_seed(label: str) -> int:
    digest = hashlib.sha256(f"same_target_drift_analysis_v1|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def summarize(values: Sequence[float], label: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"Invalid analysis vector: {label}")
    seed = analysis_seed(label)
    rng = np.random.default_rng(seed)
    samples = arr[rng.integers(0, len(arr), size=(BOOTSTRAP_REPS, len(arr)))].mean(axis=1)
    return {
        "analysis_label": label,
        "bootstrap_seed": seed,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "mean": float(arr.mean()),
        "sample_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "bootstrap_ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n_catalog_resamples": int(len(arr)),
    }


def paired_summary(hybrid_minus_rl: Sequence[float], label: str) -> dict[str, Any]:
    arr = np.asarray(hybrid_minus_rl, dtype=float)
    summary = summarize(arr, label)
    summary.update(
        {
            "direction": "hybrid_minus_rl",
            "raw_catalog_resample_vector": [float(value) for value in arr],
            "wins": int(np.sum(arr > PAIR_TOLERANCE)),
            "ties": int(np.sum(np.abs(arr) <= PAIR_TOLERANCE)),
            "losses": int(np.sum(arr < -PAIR_TOLERANCE)),
            "tie_tolerance": PAIR_TOLERANCE,
        }
    )
    return summary


def analyze(records: Sequence[Mapping[str, Any]], sudden_runs: int, gradual_runs: int) -> dict[str, Any]:
    report: dict[str, Any] = {"unit": "paired catalog-resample/Monte Carlo run; five profiles averaged within run", "primary_reward_model": PRIMARY_REWARD_MODEL, "sudden": {}, "gradual": {}}
    for scenario, runs, auc_grid in (("sudden", sudden_runs, SUDDEN_AUC_GRID), ("gradual", gradual_runs, GRADUAL_AUC_GRID)):
        for model in REWARD_MODELS:
            final_by_method = {m: [] for m in METHODS}
            auc_by_method = {m: [] for m in METHODS}
            for record in records[:runs]:
                profiles = record[scenario]["corrected_future_blind"][model]
                for method in METHODS:
                    profile_mean_by_cp = [
                        float(np.mean([p["checkpoints"][str(cp)]["f1"][method] for p in profiles]))
                        for cp in auc_grid
                    ]
                    final_by_method[method].append(profile_mean_by_cp[-1])
                    auc_by_method[method].append(normalized_auc(profile_mean_by_cp, auc_grid))
            final_difference = np.asarray(final_by_method["hybrid"], dtype=float) - np.asarray(final_by_method["rl_only"], dtype=float)
            auc_difference = np.asarray(auc_by_method["hybrid"], dtype=float) - np.asarray(auc_by_method["rl_only"], dtype=float)
            report[scenario][model] = {
                method: {
                    "final_f1": summarize(final_by_method[method], f"{scenario}|{model}|{method}|final_f1"),
                    "checkpoint_normalized_post_change_auc": summarize(auc_by_method[method], f"{scenario}|{model}|{method}|auc"),
                }
                for method in METHODS
            }
            report[scenario][model]["paired_hybrid_minus_rl"] = {
                "final_f1": paired_summary(final_difference, f"{scenario}|{model}|hybrid_minus_rl|final_f1"),
                "checkpoint_normalized_post_change_auc": paired_summary(auc_difference, f"{scenario}|{model}|hybrid_minus_rl|auc"),
            }
    return report


def run(args: argparse.Namespace) -> None:
    started = time.time()
    environment = assert_exact_environment()
    frozen = assert_frozen_inputs()
    run_manifest_sha = assert_run_manifest()
    if args.mode == "canonical":
        sudden_runs, gradual_runs = 50, 30
    else:
        sudden_runs = int(args.sudden_runs)
        gradual_runs = int(args.gradual_runs)
        if not (1 <= sudden_runs <= 50 and 1 <= gradual_runs <= 30):
            raise ValueError("Smoke run counts must be sudden 1..50 and gradual 1..30")
    output_dir = Path(args.output_dir).resolve()
    campaign_resolved = CAMPAIGN_ROOT.resolve()
    if campaign_resolved != output_dir and campaign_resolved not in output_dir.parents:
        raise ValueError("Output directory must remain inside the campaign")
    records_path = output_dir / "sealed_records.jsonl"
    terminal_path = output_dir / "TERMINAL.json"
    total = max(sudden_runs, gradual_runs)
    resume_completed = validate_output_start(output_dir, bool(args.resume), args.mode, total)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_runs = manifest["runs"]
    sudden_stored = json.loads(SUDDEN_RESULT_PATH.read_text(encoding="utf-8"))
    gradual_stored = json.loads(GRADUAL_RESULT_PATH.read_text(encoding="utf-8"))
    records = read_jsonl(records_path)
    if len(records) != resume_completed:
        raise ValueError("Resume sealed-record count disagrees with STATUS/PROGRESS")
    validate_resume(records, manifest_runs, sudden_runs, gradual_runs)
    if records:
        # A digest alone is not an authenticity proof because it could be
        # recomputed after tampering. Rebuild every completed catalog from the
        # frozen sources and require byte-equivalent scientific payloads before
        # accepting a resume point.
        for resume_index, existing in enumerate(records):
            resume_entry = manifest_runs[resume_index]
            resume_df = load_catalog(resume_entry)
            rebuilt = build_run_record(
                resume_index,
                resume_entry,
                resume_df,
                resume_index < sudden_runs,
                resume_index < gradual_runs,
                sudden_stored,
                gradual_stored,
            )
            if rebuilt != existing:
                raise ValueError(f"Resume full-replay integrity mismatch at run {resume_index}")
        print(f"RESUME_INTEGRITY_PASS replayed_catalogs={len(records)}", flush=True)
    source_contract = future_blind_source_contract()
    first_df = load_catalog(manifest_runs[0])
    preflight = metamorphic_preflight(first_df, int(manifest_runs[0]["seed"]), CORE.topsis_artifacts(first_df)["scores"])
    if not args.resume:
        atomic_json(output_dir / "STATUS.json", {
            "schema_version": "same_target_drift.status.v1", "campaign_id": CAMPAIGN_ROOT.name,
            "status": "running", "mode": args.mode, "completed_catalogs": 0, "total_catalogs": total,
            "run_manifest_sha256": run_manifest_sha,
            "scientific_metrics_exposed": False,
            "blindness": "scientific_payload_sealed_until_terminal_independent_verifier_pass",
        })
    print(f"DRIFT_START mode={args.mode} catalogs={total} resume_from={len(records)}", flush=True)
    sudden_cells = sum(int(r["legacy_gate_cells"]["sudden"]) for r in records)
    gradual_cells = sum(int(r["legacy_gate_cells"]["gradual"]) for r in records)
    for run_index in range(len(records), total):
        entry = manifest_runs[run_index]
        df = load_catalog(entry)
        record = build_run_record(run_index, entry, df, run_index < sudden_runs, run_index < gradual_runs, sudden_stored, gradual_stored)
        append_jsonl(records_path, record)
        records.append(record)
        sudden_cells += int(record["legacy_gate_cells"]["sudden"])
        gradual_cells += int(record["legacy_gate_cells"]["gradual"])
        elapsed = time.time() - started
        eta = elapsed / len(records) * (total - len(records)) if records else None
        sealed_hash = sha256_file(records_path)
        atomic_json(output_dir / "PROGRESS.json", {
            "schema_version": "same_target_drift.progress.v1", "campaign_id": CAMPAIGN_ROOT.name,
            "mode": args.mode,
            "completed_catalogs": len(records), "total_catalogs": total,
            "percent": 100.0 * len(records) / total, "elapsed_seconds": elapsed, "eta_seconds": eta,
            "legacy_gate_cells_checked": {"sudden": sudden_cells, "gradual": gradual_cells},
            "sealed_records_sha256": sealed_hash,
            "run_manifest_sha256": run_manifest_sha,
            "scientific_metrics_exposed": False,
        })
        atomic_json(output_dir / "STATUS.json", {
            "schema_version": "same_target_drift.status.v1", "campaign_id": CAMPAIGN_ROOT.name,
            "status": "running", "mode": args.mode, "completed_catalogs": len(records), "total_catalogs": total,
            "sealed_records_sha256": sealed_hash, "run_manifest_sha256": run_manifest_sha,
            "scientific_metrics_exposed": False,
            "blindness": "scientific_payload_sealed_until_terminal_independent_verifier_pass",
        })
        print(f"PROGRESS {len(records)}/{total} percent={100.0*len(records)/total:.1f} eta_seconds={eta:.0f}", flush=True)
    expected_sudden = sudden_runs * len(METHODS) * len(SUDDEN_CHECKPOINTS)
    expected_gradual = gradual_runs * 3 * len(GRADUAL_CHECKPOINTS)
    if sudden_cells != expected_sudden or gradual_cells != expected_gradual:
        raise AssertionError("Legacy exact raw-cell gate count mismatch")
    terminal = {
        "schema_version": "same_target_drift.terminal.v1", "campaign_id": CAMPAIGN_ROOT.name,
        "status": "completed_unverified", "mode": args.mode,
        "environment": environment, "run_manifest_sha256": run_manifest_sha,
        "frozen_input_hashes": frozen, "producer_commit": PRODUCER_COMMIT,
        "exact_legacy_raw_cells": {"sudden": sudden_cells, "gradual": gradual_cells},
        "future_blind_source_contract": source_contract, "prefix_invariance_metamorphic_preflight": preflight,
        "sealed_records_sha256": sha256_file(records_path), "analysis": analyze(records, sudden_runs, gradual_runs),
        "elapsed_seconds": time.time() - started,
        "blindness": "do_not_interpret_analysis_before_FULL_VERIFICATION_PASS",
    }
    atomic_json(terminal_path, terminal)
    terminal_sha = sha256_file(terminal_path)
    atomic_json(output_dir / "STATUS.json", {
        "schema_version": "same_target_drift.status.v1", "campaign_id": CAMPAIGN_ROOT.name,
        "status": "completed_unverified", "mode": args.mode, "completed_catalogs": total, "total_catalogs": total,
        "terminal_sha256": terminal_sha,
        "sealed_records_sha256": terminal["sealed_records_sha256"],
        "run_manifest_sha256": run_manifest_sha,
        "scientific_metrics_exposed": False,
        "blindness": "do_not_interpret_before_FULL_VERIFICATION_PASS",
    })
    print(f"DRIFT_COMPLETE_UNVERIFIED catalogs={total} sudden_cells={sudden_cells} gradual_cells={gradual_cells}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "canonical"), required=True)
    parser.add_argument("--sudden-runs", type=int, default=1)
    parser.add_argument("--gradual-runs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
