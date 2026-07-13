"""Same-target factorial bridge for the restored ECRA r1 experiment.

The bridge freezes the original catalogs, stored evaluation targets, seeds,
method definitions, checkpoints, and metric. It varies only candidate support,
the GT reward bonus, and the reward implementation. Console output is progress
only. A partial scientific checkpoint exists but remains unopened until terminal
verification; publication-facing interpretation uses only verified terminal data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy
from scipy import stats


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = CAMPAIGN_ROOT / "inputs"
CORE_PATH = CAMPAIGN_ROOT / "src" / "original_hybrid_core.py"
MANIFEST_PATH = INPUT_ROOT / "data" / "processed" / "manifest.json"
ORIGINAL_RESULT_PATH = INPUT_ROOT / "results" / "amazon_primary.json"
RUN_MANIFEST_PATH = CAMPAIGN_ROOT / "RUN_MANIFEST.json"
HISTORICAL_PROVENANCE_ROOT = INPUT_ROOT / "historical_reward_provenance"
PRODUCER_PROVENANCE_ROOT = INPUT_ROOT / "producer_provenance" / "v1.0-submission"
EXACT_REPLAY_REPORT_PATH = INPUT_ROOT / "exact_replay_provenance" / "EXACT_REPLAY_AUDIT.json"
EXPECTED_ENVIRONMENT = {
    "numpy": "1.26.0",
    "pandas": "2.2.3",
    "scipy": "1.16.3",
}
PRODUCER_COMMIT_SHA1 = "3b92f6485d20d1a45ac03b60077d20af08060885"
PRODUCER_FILE_SHA256 = {
    "code/hybrid_core.py": "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f",
    "code/run_amazon_experiments.py": "361f29012b1618c164d9688bd4887ca6187fc04ec67bc8dd6a9de54fd0a2f15d",
    "requirements.txt": "5241d0abaccd86ffad73f36592acbabb1bf9331be83dd4678b4ef5d6be71f391",
    "COMMIT_METADATA.json": "35b41ec3c5a9f4dd1325839aa90563767f71c9624815c302f4a1ce18892f2dad",
    "PUBLIC_TAG_EXACT_HASHES.json": "2e128e8b9fabe328ce8577cb56407c5bf90c2f7fbed6314ca110930f9bcefe8b",
}
EXACT_REPLAY_REPORT_SHA256 = "ba76de6b8043ac4baf34f4f5763c96b2c778d458ec497febc97b6f9135aa655d"
HISTORICAL_V2_SHA256 = (
    "90AF7D4D3150099D840C510F5FF420B8773659C2A7A579D04E7B6E711DA65E4F"
)
HISTORICAL_SUPPLEMENTARY_SHA256 = (
    "A18AF10D9D7C2C81E400910EC1D0DAE4071322DC1FC24AA0F3B6E022984D8BDC"
)
HISTORICAL_REWARD_SOURCE_SHA256 = {
    "hybrid_rl_mcdm_v2.py": HISTORICAL_V2_SHA256,
    "supplementary_runs.py": HISTORICAL_SUPPLEMENTARY_SHA256,
}
CHECKPOINTS = (500, 1000, 2000, 5000, 10000, 20000, 30000)
METHODS = ("random", "popularity", "topsis_only", "rl_only", "hybrid")
ANALYSIS_SEED = 2026071201
BOOTSTRAP_REPS = 20_000
PAIR_TOLERANCE = 1e-12


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("bridge_original_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_module(CORE_PATH)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_exact_environment() -> dict[str, str]:
    """Fail before any run artifact is created unless producer versions match."""
    actual = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    if actual != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            f"Producer environment mismatch: actual={actual} expected={EXPECTED_ENVIRONMENT}"
        )
    return actual


def assert_run_manifest() -> str:
    """Fail before compute if any hash-locked campaign source/input drifted."""
    if not RUN_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing pre-compute lock: {RUN_MANIFEST_PATH}")
    payload = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "same_target_bridge.run_manifest.v1":
        raise ValueError("Unexpected RUN_MANIFEST schema")
    if payload.get("campaign_id") != CAMPAIGN_ROOT.name:
        raise ValueError("RUN_MANIFEST campaign ID mismatch")
    locked_packages = payload.get("environment", {}).get("packages")
    if locked_packages != EXPECTED_ENVIRONMENT:
        raise ValueError(
            f"RUN_MANIFEST environment lock mismatch: {locked_packages}"
        )
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("RUN_MANIFEST files must be a non-empty list")
    root = CAMPAIGN_ROOT.resolve()
    seen: set[str] = set()
    for entry in entries:
        rel = str(entry["path"])
        if rel in seen:
            raise ValueError(f"Duplicate RUN_MANIFEST path: {rel}")
        seen.add(rel)
        path = (CAMPAIGN_ROOT / rel).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"RUN_MANIFEST path escapes campaign: {rel}")
        if not path.is_file():
            raise FileNotFoundError(f"Locked file missing: {rel}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Locked file size mismatch: {rel}")
        actual = sha256_file(path)
        if actual != str(entry["sha256"]).lower():
            raise ValueError(f"Locked file hash mismatch: {rel}")
    required = {
        "PROTOCOL_LOCK.md",
        "verify_bridge.py",
        "tests/test_bridge_contract.py",
        "tests/test_verify_bridge.py",
        "src/bridge_main.py",
        "src/lock_campaign.py",
        "src/original_hybrid_core.py",
        "inputs/data/processed/manifest.json",
        "inputs/results/amazon_primary.json",
        "inputs/historical_reward_provenance/hybrid_rl_mcdm_v2.py",
        "inputs/historical_reward_provenance/supplementary_runs.py",
        "inputs/producer_provenance/v1.0-submission/code/hybrid_core.py",
        "inputs/producer_provenance/v1.0-submission/code/run_amazon_experiments.py",
        "inputs/producer_provenance/v1.0-submission/requirements.txt",
        "inputs/producer_provenance/v1.0-submission/COMMIT_METADATA.json",
        "inputs/producer_provenance/v1.0-submission/PUBLIC_TAG_EXACT_HASHES.json",
        "inputs/exact_replay_provenance/EXACT_REPLAY_AUDIT.json",
    }
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"RUN_MANIFEST missing required files: {missing}")
    return sha256_file(RUN_MANIFEST_PATH)


def assert_producer_provenance() -> dict[str, str]:
    """Fail closed unless the frozen public-tag producer files are exact."""
    verified: dict[str, str] = {}
    for relative, expected_sha256 in PRODUCER_FILE_SHA256.items():
        path = PRODUCER_PROVENANCE_ROOT / Path(relative)
        if not path.is_file():
            raise AssertionError(f"Missing producer provenance file: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise AssertionError(
                f"Producer provenance hash mismatch for {relative}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        verified[relative] = actual_sha256
    if sha256_file(CORE_PATH) != PRODUCER_FILE_SHA256["code/hybrid_core.py"]:
        raise AssertionError("Executable frozen core is not the exact public-tag core")
    metadata = json.loads(
        (PRODUCER_PROVENANCE_ROOT / "COMMIT_METADATA.json").read_text(encoding="utf-8")
    )
    public_hashes = json.loads(
        (PRODUCER_PROVENANCE_ROOT / "PUBLIC_TAG_EXACT_HASHES.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata.get("commit_sha1") != PRODUCER_COMMIT_SHA1:
        raise AssertionError("Producer commit metadata does not identify the frozen tag")
    if public_hashes.get("public_ls_remote_commit_sha1") != PRODUCER_COMMIT_SHA1:
        raise AssertionError("Public tag commit does not match the local producer tag")
    file_map = {str(item.get("path")): item for item in public_hashes.get("files", [])}
    for relative in ("code/hybrid_core.py", "code/run_amazon_experiments.py", "requirements.txt"):
        if file_map.get(relative, {}).get("sha256") != PRODUCER_FILE_SHA256[relative]:
            raise AssertionError(f"Public-tag file metadata mismatch: {relative}")
    return verified


def assert_exact_replay_provenance() -> dict[str, Any]:
    """Require the terminal producer-environment 250-cell exact replay gate."""
    if not EXACT_REPLAY_REPORT_PATH.is_file():
        raise AssertionError(f"Missing exact replay report: {EXACT_REPLAY_REPORT_PATH}")
    if sha256_file(EXACT_REPLAY_REPORT_PATH) != EXACT_REPLAY_REPORT_SHA256:
        raise AssertionError("Exact replay report hash mismatch")
    report = json.loads(EXACT_REPLAY_REPORT_PATH.read_text(encoding="utf-8"))
    if not (
        report.get("status") == "completed"
        and report.get("all_exact") is True
        and int(report.get("cells_total", -1)) == 250
        and int(report.get("cells_exact", -1)) == 250
        and int(report.get("cells_mismatched", -1)) == 0
    ):
        raise AssertionError("Exact replay report is not terminal all_exact 250/250")
    if (
        report.get("numpy") != EXPECTED_ENVIRONMENT["numpy"]
        or report.get("pandas") != EXPECTED_ENVIRONMENT["pandas"]
    ):
        raise AssertionError("Exact replay report was not produced in the locked environment")
    return {
        "sha256": EXACT_REPLAY_REPORT_SHA256,
        "cells_total": 250,
        "cells_exact": 250,
        "all_exact": True,
    }


def assert_historical_reward_provenance() -> dict[str, str]:
    """Fail closed unless both frozen historical reward sources match exactly."""
    verified: dict[str, str] = {}
    for filename, expected_sha256 in HISTORICAL_REWARD_SOURCE_SHA256.items():
        path = HISTORICAL_PROVENANCE_ROOT / filename
        if not path.is_file():
            raise AssertionError(f"Missing historical reward provenance source: {path}")
        actual_sha256 = sha256_file(path).upper()
        if actual_sha256 != expected_sha256:
            raise AssertionError(
                f"Historical reward provenance hash mismatch for {filename}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        verified[filename] = actual_sha256
    return verified


def assert_json_finite(value: Any, path: str = "root") -> None:
    """Reject non-finite JSON numbers while permitting explicit null values."""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite JSON number at {path}: {value!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_json_finite(item, f"{path}[{index}]")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_json_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def arm_id(candidate: str, bonus: float, reward_model: str) -> str:
    return f"candidate={candidate}__bonus={bonus:.2f}__reward={reward_model}"


def mandatory_arms() -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog"):
        for bonus in (0.20, 0.00):
            for reward_model in (
                "implemented_r0",
                "inclusive_range_fix",
                "component_continuous_fix",
            ):
                arms.append(
                    {
                        "arm_id": arm_id(candidate, bonus, reward_model),
                        "candidate": candidate,
                        "gt_bonus": bonus,
                        "reward_model": reward_model,
                        "role": "mandatory_factorial",
                    }
                )
    return arms


def sensitivity_arms() -> list[dict[str, Any]]:
    return [
        {
            "arm_id": arm_id("oracle_gt_hidden30", 0.20, "historical_funnel_coefficients_on_may_h"),
            "candidate": "oracle_gt_hidden30",
            "gt_bonus": 0.20,
            "reward_model": "historical_funnel_coefficients_on_may_h",
            "role": "secondary_reward_specification_sensitivity",
        },
        {
            "arm_id": arm_id("full_catalog", 0.00, "historical_funnel_coefficients_on_may_h"),
            "candidate": "full_catalog",
            "gt_bonus": 0.00,
            "reward_model": "historical_funnel_coefficients_on_may_h",
            "role": "secondary_reward_specification_sensitivity",
        },
    ]


ARMS = mandatory_arms() + sensitivity_arms()
EXACT_ARM_ID = arm_id("oracle_gt_hidden30", 0.20, "implemented_r0")
PRIMARY_CORRECTED_ARM_ID = arm_id("full_catalog", 0.00, "component_continuous_fix")


def candidate_pool(
    df: pd.DataFrame,
    profile: Mapping[str, object],
    kind: str,
    gt_set: set[int] | None = None,
) -> np.ndarray:
    hidden30 = CORE.top_k_set(CORE.hidden_utility(df, profile), k=30)
    if kind == "oracle_gt_hidden30":
        if gt_set is None:
            raise ValueError("oracle candidate arm requires the frozen GT set")
        return np.asarray(sorted(hidden30 | gt_set), dtype=int)
    if kind == "hidden30_only":
        if gt_set is not None:
            raise ValueError("hidden30_only candidate path must not receive GT")
        return np.asarray(sorted(hidden30), dtype=int)
    if kind == "full_catalog":
        if gt_set is not None:
            raise ValueError("full_catalog candidate path must not receive GT")
        return np.arange(len(df), dtype=int)
    raise KeyError(kind)


def reward_probabilities(
    df: pd.DataFrame,
    profile: Mapping[str, object],
    model: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    brand_pref = profile["brand_pref"]
    price_lo, price_hi = profile["price_range"]
    recency_weight = float(profile["recency_weight"])

    brand = np.asarray([brand_pref.get(value, 0.10) for value in df["brand"]], dtype=float)
    recency = (
        df["recency_pct"].to_numpy(dtype=float) * recency_weight
        + (1.0 - recency_weight) * 0.5
    )

    if model == "implemented_r0":
        center = (price_lo + price_hi) / 2.0
        half_range = (price_hi - price_lo) / 2.0 + 1.0
        price_fit = np.clip(
            1.0 - np.abs(df["price"].to_numpy(dtype=float) - center) / half_range,
            0.0,
            1.0,
        )
        in_range = (price_fit > 0.999999).astype(float)
        category = np.ones(len(df), dtype=float)
        p_engage = np.clip(0.40 * brand + 0.35 * in_range + 0.15 * category + 0.10 * recency, 0, 1)
        p_convert = np.clip(0.50 * brand + 0.30 * in_range + 0.20 * category, 0, 1)
        diagnostics = {
            "category_distinct": int(np.unique(category).size),
            "price_match_count": int(in_range.sum()),
            "definition": "exact original fast path",
        }
    elif model == "inclusive_range_fix":
        prices = df["price"].to_numpy(dtype=float)
        in_range = ((prices >= float(price_lo)) & (prices <= float(price_hi))).astype(float)
        cat_affinity = profile["cat_affinity"]
        cat_scale = max(float(value) for value in cat_affinity.values())
        category = np.asarray(
            [float(cat_affinity.get(value, 0.0)) / cat_scale for value in df["category"]],
            dtype=float,
        )
        p_engage = np.clip(0.40 * brand + 0.35 * in_range + 0.15 * category + 0.10 * recency, 0, 1)
        p_convert = np.clip(0.50 * brand + 0.30 * in_range + 0.20 * category, 0, 1)
        diagnostics = {
            "category_distinct": int(np.unique(category).size),
            "price_match_count": int(in_range.sum()),
            "definition": "original coefficients; category profile-scaled; inclusive binary price interval",
        }
    elif model == "component_continuous_fix":
        components = CORE.hidden_components(df, profile)
        category = np.asarray(components["cat_score"], dtype=float)
        price_fit = np.asarray(components["price_fit"], dtype=float)
        p_engage = np.clip(
            0.40 * brand + 0.35 * price_fit + 0.15 * category + 0.10 * recency,
            0,
            1,
        )
        p_convert = np.clip(
            0.50 * brand + 0.30 * price_fit + 0.20 * category,
            0,
            1,
        )
        diagnostics = {
            "category_distinct": int(np.unique(category).size),
            "price_match_count": int(np.sum(price_fit > 0.0)),
            "definition": "original coefficients; full-vector category and continuous triangular price fit",
        }
    elif model == "historical_funnel_coefficients_on_may_h":
        hidden = CORE.hidden_utility(df, profile)
        p_engage = np.clip(0.70 * hidden + 0.10, 0.05, 0.95)
        p_convert = np.clip(0.50 * hidden, 0.02, 0.80)
        diagnostics = {
            "category_distinct": int(
                np.unique(CORE.hidden_components(df, profile)["cat_score"]).size
            ),
            "price_match_count": int(
                np.sum(CORE.hidden_components(df, profile)["price_fit"] > 0.0)
            ),
            "definition": (
                "secondary historical sensitivity: "
                "P(engage)=clip(0.7H+0.1,0.05,0.95); "
                "P(convert|engage)=clip(0.5H,0.02,0.80)"
            ),
            "historical_source_sha256": HISTORICAL_REWARD_SOURCE_SHA256,
        }
    else:
        raise KeyError(model)

    if not np.all(np.isfinite(p_engage)) or not np.all(np.isfinite(p_convert)):
        raise ValueError(f"Nonfinite reward probabilities for {model}")
    if np.any((p_engage < 0) | (p_engage > 1)) or np.any((p_convert < 0) | (p_convert > 1)):
        raise ValueError(f"Out-of-range reward probabilities for {model}")
    return p_engage, p_convert, diagnostics


def train_arm_profile(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    run_seed: int,
    gt_set: set[int],
    gt_rank: list[int],
    topsis_scores: np.ndarray,
    arm: Mapping[str, Any],
    episodes: int,
) -> dict[str, Any]:
    profile = CORE.PROFILE_HIDDEN[profile_name]
    candidate_kind = str(arm["candidate"])
    pool = candidate_pool(
        df,
        profile,
        candidate_kind,
        gt_set if candidate_kind == "oracle_gt_hidden30" else None,
    )
    p_engage, p_convert, reward_diag = reward_probabilities(
        df, profile, str(arm["reward_model"])
    )

    n_products = len(df)
    q_scores = np.zeros(n_products, dtype=float)
    visits = np.zeros(n_products, dtype=np.int32)
    eps = 0.30
    act_rng = np.random.RandomState(run_seed + profile_idx * 13)
    reward_rng = np.random.RandomState(run_seed + profile_idx * 997)
    gt_mask: np.ndarray | None = None
    if float(arm["gt_bonus"]) != 0.0:
        gt_mask = np.zeros(n_products, dtype=bool)
        gt_mask[list(gt_set)] = True
    checkpoints = set(cp for cp in CHECKPOINTS if cp <= episodes)

    random_rank = CORE.random_ranking(np.random.RandomState(run_seed + 5555), n_products)
    popularity_rank = CORE.popularity_ranking(df)
    topsis_rank = CORE.top_k_ranking(topsis_scores)
    checkpoint_metrics: dict[str, dict[str, float]] = {}
    final_rankings: dict[str, list[int]] | None = None

    for episode in range(1, episodes + 1):
        if act_rng.random() < eps:
            action = int(act_rng.choice(pool))
        else:
            action = int(pool[np.argmax(q_scores[pool])])

        reward = -0.02
        if reward_rng.random() < p_engage[action]:
            reward += 0.30
            if reward_rng.random() < p_convert[action]:
                reward += 1.00
        if gt_mask is not None and gt_mask[action]:
            reward += float(arm["gt_bonus"])

        visits[action] += 1
        q_scores[action] += 0.05 * (reward - q_scores[action])
        eps = max(0.05, eps * 0.9997)

        if episode in checkpoints:
            rl_rank = CORE.top_k_ranking(q_scores)
            hybrid_rank = CORE.top_k_ranking(
                CORE.static_hybrid_score(q_scores, topsis_scores, lambda_q=0.50)
            )
            rankings = {
                "random": random_rank,
                "popularity": popularity_rank,
                "topsis_only": topsis_rank,
                "rl_only": rl_rank,
                "hybrid": hybrid_rank,
            }
            checkpoint_metrics[str(episode)] = {
                method: float(CORE.f1_score(set(rank), gt_set))
                for method, rank in rankings.items()
            }
            if episode == episodes:
                final_rankings = {method: [int(x) for x in rank] for method, rank in rankings.items()}

    if final_rankings is None:
        raise RuntimeError("Final rankings were not produced")
    final_metrics = {
        method: {
            "f1_at_7": float(CORE.f1_score(set(rank), gt_set)),
            "ndcg_at_7": float(CORE.ndcg_at_k(rank, gt_set)),
        }
        for method, rank in final_rankings.items()
    }
    return {
        "profile_name": profile_name,
        "candidate_count": int(pool.size),
        "candidate_sha256": hashlib.sha256(pool.astype("<i8").tobytes()).hexdigest(),
        "reward_diagnostics": reward_diag,
        "checkpoint_f1": checkpoint_metrics,
        "final_metrics": final_metrics,
        "final_rankings": final_rankings,
        "gt_rank": [int(x) for x in gt_rank],
        "q_scores": [float(x) for x in q_scores],
        "visits": [int(x) for x in visits],
        "epsilon_final": float(eps),
    }


def original_profile_map(original: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(catalog["run_index"]), str(profile["profile_name"])): profile
        for catalog in original["artifacts"]
        for profile in catalog["profile_results"]
    }


def validate_resume_records(
    records: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    original: Mapping[str, Any],
    run_manifest_sha256: str,
    requested_runs: int,
) -> None:
    """Reject stale or structurally incomplete catalog checkpoints before skip."""
    expected_arms = {arm["arm_id"]: arm for arm in ARMS}
    original_map = original_profile_map(original)
    original_sha256 = sha256_file(ORIGINAL_RESULT_PATH)
    seen: set[int] = set()
    for record in records:
        if record.get("schema_version") != "same_target_bridge.catalog.v1":
            raise ValueError("Resume checkpoint schema mismatch")
        if record.get("campaign_id") != CAMPAIGN_ROOT.name:
            raise ValueError("Resume checkpoint campaign mismatch")
        run_index = int(record.get("run_index", -1))
        if run_index in seen or not 0 <= run_index < requested_runs:
            raise ValueError(f"Duplicate/out-of-scope resume run_index: {run_index}")
        seen.add(run_index)
        meta = manifest["runs"][run_index]
        if int(record.get("run_seed", -1)) != int(meta["seed"]):
            raise ValueError(f"Resume seed mismatch run={run_index}")
        if str(record.get("dataset_path", "")).replace("\\", "/") != str(
            meta["path"]
        ).replace("\\", "/"):
            raise ValueError(f"Resume dataset path mismatch run={run_index}")
        if record.get("dataset_sha256") != meta["sha256"]:
            raise ValueError(f"Resume dataset hash mismatch run={run_index}")
        if record.get("target_source_sha256") != original_sha256:
            raise ValueError(f"Resume target-source hash mismatch run={run_index}")
        if record.get("run_manifest_sha256") != run_manifest_sha256:
            raise ValueError(f"Resume run-manifest mismatch run={run_index}")
        arms = record.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(expected_arms):
            raise ValueError(f"Resume arm completeness mismatch run={run_index}")
        for arm_id_value, expected_arm in expected_arms.items():
            cell = arms[arm_id_value]
            if not isinstance(cell, dict) or cell.get("arm") != expected_arm:
                raise ValueError(f"Resume arm metadata mismatch run={run_index} arm={arm_id_value}")
            profiles = cell.get("profiles")
            if not isinstance(profiles, list) or [
                profile.get("profile_name") for profile in profiles
            ] != list(CORE.PROFILE_ORDER):
                raise ValueError(f"Resume profile completeness mismatch run={run_index} arm={arm_id_value}")
            for profile in profiles:
                q_scores = profile.get("q_scores")
                visits = profile.get("visits")
                if not isinstance(q_scores, list) or len(q_scores) != 400:
                    raise ValueError(f"Resume Q-vector mismatch run={run_index} arm={arm_id_value}")
                if (
                    not isinstance(visits, list)
                    or len(visits) != 400
                    or sum(int(value) for value in visits) != 30000
                ):
                    raise ValueError(f"Resume visit-vector mismatch run={run_index} arm={arm_id_value}")
        exact_profiles = arms[EXACT_ARM_ID]["profiles"]
        for profile in exact_profiles:
            stored = original_map[(run_index, str(profile["profile_name"]))]
            assert_exact_profile(profile, stored, run_index)


def assert_exact_profile(
    computed: Mapping[str, Any], stored: Mapping[str, Any], run_index: int
) -> None:
    final = stored["final"]
    for method, stored_key in (
        ("random", "random_rank"),
        ("popularity", "popularity_rank"),
        ("topsis_only", "topsis_rank"),
        ("rl_only", "rl_rank"),
        ("hybrid", "hybrid_rank"),
    ):
        if computed["final_rankings"][method] != [int(x) for x in final[stored_key]]:
            raise AssertionError(
                f"Exact replay ranking mismatch run={run_index} profile={stored['profile_name']} method={method}"
            )
    for checkpoint in CHECKPOINTS:
        for method in METHODS:
            got = float(computed["checkpoint_f1"][str(checkpoint)][method])
            expected = float(stored["f1"][method][str(checkpoint)])
            if got != expected:
                raise AssertionError(
                    f"Exact replay F1 mismatch run={run_index} profile={stored['profile_name']} "
                    f"checkpoint={checkpoint} method={method}: {got!r} != {expected!r}"
                )
    if not np.array_equal(
        np.asarray(computed["q_scores"], dtype=float),
        np.asarray(final["q_scores"], dtype=float),
    ):
        raise AssertionError(
            f"Exact replay Q mismatch run={run_index} profile={stored['profile_name']}"
        )
    if computed["visits"] != [int(x) for x in final["visits"]]:
        raise AssertionError(
            f"Exact replay visits mismatch run={run_index} profile={stored['profile_name']}"
        )


def summarize_vector(values: Iterable[float], rng: np.random.Generator) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if arr.shape != (50,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"Expected 50 finite catalog values, got {arr.shape}")
    indices = rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))
    boot = arr[indices].mean(axis=1)
    if not np.all(np.isfinite(boot)):
        raise ValueError("Bootstrap generated non-finite catalog means")
    result = {
        "mean": float(arr.mean()),
        "sample_sd": float(arr.std(ddof=1)),
        "bootstrap_ci95_lo": float(np.percentile(boot, 2.5)),
        "bootstrap_ci95_hi": float(np.percentile(boot, 97.5)),
        "n_catalogs": int(arr.size),
        "raw_catalog_means": [float(x) for x in arr],
    }
    assert_json_finite(result, "summary")
    return result


def paired_summary(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    base = summarize_vector(diff, rng)
    sd = float(diff.std(ddof=1))
    constant_difference = bool(np.all(diff == diff[0]))
    t_defined = not constant_difference
    if t_defined:
        t = stats.ttest_rel(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        t_stat, t_p = float(t.statistic), float(t.pvalue)
        if not math.isfinite(t_stat) or not math.isfinite(t_p):
            raise ValueError("Defined paired t-test returned a non-finite result")
    else:
        t_stat, t_p = None, None

    wilcoxon_defined = bool(np.any(diff != 0.0))
    if wilcoxon_defined:
        w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
        if not math.isfinite(wilcoxon_stat) or not math.isfinite(wilcoxon_p):
            raise ValueError("Defined Wilcoxon test returned a non-finite result")
    else:
        wilcoxon_stat, wilcoxon_p = None, None

    cohen_dz_defined = not constant_difference
    base.update(
        {
            "paired_t_defined": t_defined,
            "paired_t_stat": t_stat,
            "paired_t_p_two_sided": t_p,
            "cohen_dz_defined": cohen_dz_defined,
            "cohen_dz": float(diff.mean() / sd) if cohen_dz_defined else None,
            "wilcoxon_defined": wilcoxon_defined,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p_two_sided": wilcoxon_p,
            "wins": int(np.sum(diff > PAIR_TOLERANCE)),
            "ties": int(np.sum(np.abs(diff) <= PAIR_TOLERANCE)),
            "losses": int(np.sum(diff < -PAIR_TOLERANCE)),
        }
    )
    assert_json_finite(base, "paired_summary")
    return base


def analyze(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 50:
        raise ValueError("Canonical analysis requires exactly 50 catalogs")
    rng = np.random.default_rng(ANALYSIS_SEED)
    metrics = ("f1_at_7", "ndcg_at_7")
    by_arm: dict[str, dict[str, dict[str, list[float]]]] = {
        arm["arm_id"]: {
            metric: {method: [] for method in METHODS} for metric in metrics
        }
        for arm in ARMS
    }
    for record in sorted(records, key=lambda item: int(item["run_index"])):
        for arm in ARMS:
            profiles = record["arms"][arm["arm_id"]]["profiles"]
            for metric in metrics:
                for method in METHODS:
                    by_arm[arm["arm_id"]][metric][method].append(
                        float(
                            np.mean(
                                [
                                    profile["final_metrics"][method][metric]
                                    for profile in profiles
                                ]
                            )
                        )
                    )

    summaries: dict[str, Any] = {}
    vectors_by_arm: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for arm in ARMS:
        aid = arm["arm_id"]
        vectors_by_arm[aid] = {
            metric: {
                method: np.asarray(by_arm[aid][metric][method], dtype=float)
                for method in METHODS
            }
            for metric in metrics
        }
        f1_vectors = vectors_by_arm[aid]["f1_at_7"]
        ndcg_vectors = vectors_by_arm[aid]["ndcg_at_7"]
        summaries[aid] = {
            "arm": arm,
            # Backward-compatible top-level keys are the primary F1 summaries.
            "methods": {
                method: summarize_vector(f1_vectors[method], rng) for method in METHODS
            },
            "hybrid_minus_rl": paired_summary(
                f1_vectors["hybrid"], f1_vectors["rl_only"], rng
            ),
            "hybrid_minus_topsis": paired_summary(
                f1_vectors["hybrid"], f1_vectors["topsis_only"], rng
            ),
            "ndcg_at_7": {
                "methods": {
                    method: summarize_vector(ndcg_vectors[method], rng)
                    for method in METHODS
                },
                "hybrid_minus_rl": paired_summary(
                    ndcg_vectors["hybrid"], ndcg_vectors["rl_only"], rng
                ),
                "hybrid_minus_topsis": paired_summary(
                    ndcg_vectors["hybrid"], ndcg_vectors["topsis_only"], rng
                ),
            },
        }

    contrast_specs: list[dict[str, str]] = []
    core_rewards = (
        "implemented_r0",
        "inclusive_range_fix",
        "component_continuous_fix",
    )
    for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog"):
        for reward_model in core_rewards:
            contrast_specs.append(
                {
                    "contrast_id": f"bonus_removal__{candidate}__{reward_model}",
                    "factor": "gt_bonus",
                    "arm_a": arm_id(candidate, 0.00, reward_model),
                    "arm_b": arm_id(candidate, 0.20, reward_model),
                    "direction": "bonus_0.00_minus_0.20",
                }
            )
    candidate_pairs = (
        ("hidden30_only", "oracle_gt_hidden30"),
        ("full_catalog", "hidden30_only"),
        ("full_catalog", "oracle_gt_hidden30"),
    )
    for bonus in (0.20, 0.00):
        for reward_model in core_rewards:
            for candidate_a, candidate_b in candidate_pairs:
                contrast_specs.append(
                    {
                        "contrast_id": (
                            f"candidate__{candidate_a}_minus_{candidate_b}__"
                            f"bonus={bonus:.2f}__{reward_model}"
                        ),
                        "factor": "candidate_support",
                        "arm_a": arm_id(candidate_a, bonus, reward_model),
                        "arm_b": arm_id(candidate_b, bonus, reward_model),
                        "direction": f"{candidate_a}_minus_{candidate_b}",
                    }
                )
    reward_pairs = (
        ("inclusive_range_fix", "implemented_r0"),
        ("component_continuous_fix", "implemented_r0"),
        ("component_continuous_fix", "inclusive_range_fix"),
    )
    for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog"):
        for bonus in (0.20, 0.00):
            for reward_a, reward_b in reward_pairs:
                contrast_specs.append(
                    {
                        "contrast_id": (
                            f"reward__{reward_a}_minus_{reward_b}__{candidate}__"
                            f"bonus={bonus:.2f}"
                        ),
                        "factor": "reward_implementation",
                        "arm_a": arm_id(candidate, bonus, reward_a),
                        "arm_b": arm_id(candidate, bonus, reward_b),
                        "direction": f"{reward_a}_minus_{reward_b}",
                    }
                )

    factorial_contrasts: dict[str, Any] = {}
    for metric in metrics:
        metric_contrasts: dict[str, Any] = {}
        for spec in contrast_specs:
            arm_a = spec["arm_a"]
            arm_b = spec["arm_b"]
            metric_contrasts[spec["contrast_id"]] = {
                "specification": spec,
                "hybrid": paired_summary(
                    vectors_by_arm[arm_a][metric]["hybrid"],
                    vectors_by_arm[arm_b][metric]["hybrid"],
                    rng,
                ),
                "rl_only": paired_summary(
                    vectors_by_arm[arm_a][metric]["rl_only"],
                    vectors_by_arm[arm_b][metric]["rl_only"],
                    rng,
                ),
            }
        factorial_contrasts[metric] = metric_contrasts

    historical_reward = "historical_funnel_coefficients_on_may_h"
    sensitivity_specs = (
        {
            "contrast_id": "historical_funnel_minus_continuous__oracle_bonus",
            "arm_a": arm_id("oracle_gt_hidden30", 0.20, historical_reward),
            "arm_b": arm_id("oracle_gt_hidden30", 0.20, "component_continuous_fix"),
        },
        {
            "contrast_id": "historical_funnel_minus_continuous__full_no_bonus",
            "arm_a": arm_id("full_catalog", 0.00, historical_reward),
            "arm_b": arm_id("full_catalog", 0.00, "component_continuous_fix"),
        },
    )
    sensitivity_contrasts: dict[str, Any] = {}
    for metric in metrics:
        sensitivity_contrasts[metric] = {}
        for spec in sensitivity_specs:
            sensitivity_contrasts[metric][spec["contrast_id"]] = {
                "specification": spec,
                "hybrid": paired_summary(
                    vectors_by_arm[spec["arm_a"]][metric]["hybrid"],
                    vectors_by_arm[spec["arm_b"]][metric]["hybrid"],
                    rng,
                ),
                "rl_only": paired_summary(
                    vectors_by_arm[spec["arm_a"]][metric]["rl_only"],
                    vectors_by_arm[spec["arm_b"]][metric]["rl_only"],
                    rng,
                ),
            }
    analysis_payload = {
        "schema_version": "same_target_bridge.main.v1",
        "campaign_id": CAMPAIGN_ROOT.name,
        "analysis_seed": ANALYSIS_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "undefined_test_policy": {
            "paired_t": "null with paired_t_defined=false when paired differences are constant",
            "cohen_dz": "null with cohen_dz_defined=false when paired differences are constant",
            "wilcoxon": "null with wilcoxon_defined=false when every paired difference is exactly zero",
            "win_tie_loss_tolerance": PAIR_TOLERANCE,
            "json_nonfinite": "fail before write; JSON serialization uses allow_nan=false",
        },
        "inference_unit": (
            "paired catalog-resample/Monte Carlo run; five profiles averaged "
            "within each run; not an independent source-population catalog"
        ),
        "uncertainty_scope": (
            "fixed-seed diagnostic uncertainty across 50 paired catalog-resample "
            "runs; p-values are sensitivity diagnostics, not population-generalization claims"
        ),
        "arm_count": len(ARMS),
        "mandatory_factorial_arm_count": len(mandatory_arms()),
        "summaries": summaries,
        "factorial_contrasts": factorial_contrasts,
        "sensitivity_contrasts": sensitivity_contrasts,
    }
    assert_json_finite(analysis_payload, "analysis")
    return analysis_payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid checkpoint JSONL line {line_number}") from exc
    return records


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    assert_json_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def prepare_output_directory(output_dir: Path, resume: bool) -> None:
    """Fail closed on stale or non-canonical scientific output contents."""
    output_dir = output_dir.resolve()
    checkpoint_path = output_dir / "main_catalogs.jsonl"
    status_path = output_dir / "status.json"
    terminal_path = output_dir / "main_results.json"
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError("--resume requires an existing output directory")
        entries = {entry.name: entry for entry in output_dir.iterdir()}
        allowed = {"main_catalogs.jsonl", "status.json"}
        unexpected = sorted(set(entries) - allowed)
        if unexpected:
            raise FileExistsError(
                f"Resume output directory contains stale/unexpected entries: {unexpected}"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires an existing catalog checkpoint")
        if status_path.exists() and not status_path.is_file():
            raise FileExistsError("Resume status path is not a regular file")
        if terminal_path.exists():
            raise FileExistsError(
                "A terminal output already exists; completed runs are immutable"
            )
        return

    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError("New-run output path exists and is not a directory")
        stale = [entry.name for entry in output_dir.iterdir()]
        if stale:
            raise FileExistsError(
                f"A new run requires a truly empty output directory; found: {sorted(stale)}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def run(args: argparse.Namespace) -> None:
    environment_versions = assert_exact_environment()
    if not args.smoke and (args.runs != 50 or args.episodes != 30000):
        raise ValueError("Canonical mode is locked to --runs 50 --episodes 30000")
    if args.episodes != 30000:
        raise ValueError("The exact replay gate requires 30,000 episodes")

    run_manifest_sha256 = assert_run_manifest()
    historical_provenance_hashes = assert_historical_reward_provenance()
    producer_provenance_hashes = assert_producer_provenance()
    exact_replay_provenance = assert_exact_replay_provenance()

    output_dir = (CAMPAIGN_ROOT / args.output_dir).resolve()
    campaign_root = CAMPAIGN_ROOT.resolve()
    if campaign_root != output_dir and campaign_root not in output_dir.parents:
        raise ValueError(f"Output directory escapes campaign: {output_dir}")
    checkpoint_path = output_dir / "main_catalogs.jsonl"
    status_path = output_dir / "status.json"
    terminal_path = output_dir / "main_results.json"
    prepare_output_directory(output_dir, args.resume)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_RESULT_PATH.read_text(encoding="utf-8"))
    original_map = original_profile_map(original)
    existing = read_jsonl(checkpoint_path) if args.resume else []
    validate_resume_records(
        existing, manifest, original, run_manifest_sha256, args.runs
    )
    completed = {int(record["run_index"]) for record in existing}
    if len(completed) != len(existing):
        raise ValueError("Duplicate run_index in checkpoint")
    if any(index >= args.runs for index in completed):
        raise ValueError("Checkpoint contains run outside requested scope")

    started = time.time()
    trajectories_per_catalog = len(CORE.PROFILE_ORDER) * len(ARMS)
    trajectories_total = args.runs * trajectories_per_catalog
    trajectories_existing = len(existing) * trajectories_per_catalog
    atomic_json(
        status_path,
        {
            "status": "running",
            "campaign_id": CAMPAIGN_ROOT.name,
            "mode": "smoke" if args.smoke else "canonical",
            "pid": os.getpid(),
            "run_manifest_sha256": run_manifest_sha256,
            "runs_total": args.runs,
            "runs_completed": len(completed),
            "trajectories_total": trajectories_total,
            "trajectories_completed": trajectories_existing,
            "progress_percent": 100.0 * len(completed) / args.runs,
            "current_run": None,
            "stderr_health": "inspect wrapper stderr",
            "python_unbuffered_required": True,
        },
    )

    for run_index in range(args.runs):
        if run_index in completed:
            continue
        run_started = time.time()
        catalog_meta = manifest["runs"][run_index]
        catalog_path = INPUT_ROOT / catalog_meta["path"]
        if sha256_file(catalog_path) != catalog_meta["sha256"]:
            raise AssertionError(f"Catalog hash mismatch run={run_index}")
        stored_catalog = original["artifacts"][run_index]
        if stored_catalog["dataset_sha256"] != catalog_meta["sha256"]:
            raise AssertionError(f"Original result/catalog hash mismatch run={run_index}")

        df = pd.read_csv(catalog_path)
        topsis_bundle = CORE.topsis_artifacts(df)
        arm_payloads: dict[str, Any] = {
            arm["arm_id"]: {"arm": arm, "profiles": []} for arm in ARMS
        }

        for profile_idx, profile_name in enumerate(CORE.PROFILE_ORDER):
            stored = original_map[(run_index, profile_name)]
            gt_set = set(int(x) for x in stored["final"]["gt_set"])
            gt_rank = [int(x) for x in stored["final"]["gt_rank"]]
            if len(gt_set) != 7 or len(gt_rank) != 7:
                raise AssertionError("Frozen target must contain seven items")
            if CORE.top_k_ranking(topsis_bundle["scores"]) != [
                int(x) for x in stored["final"]["topsis_rank"]
            ]:
                raise AssertionError(f"TOPSIS invariant mismatch run={run_index} profile={profile_name}")

            for arm in ARMS:
                result = train_arm_profile(
                    df,
                    profile_name,
                    profile_idx,
                    int(catalog_meta["seed"]),
                    gt_set,
                    gt_rank,
                    topsis_bundle["scores"],
                    arm,
                    args.episodes,
                )
                if arm["arm_id"] == EXACT_ARM_ID:
                    assert_exact_profile(result, stored, run_index)
                arm_payloads[arm["arm_id"]]["profiles"].append(result)

            trajectories_completed = (
                len(completed) * trajectories_per_catalog
                + (profile_idx + 1) * len(ARMS)
            )
            trajectory_rate = (time.time() - started) / max(
                1, trajectories_completed - trajectories_existing
            )
            trajectory_remaining = trajectory_rate * (
                trajectories_total - trajectories_completed
            )
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "campaign_id": CAMPAIGN_ROOT.name,
                    "mode": "smoke" if args.smoke else "canonical",
                    "pid": os.getpid(),
                    "run_manifest_sha256": run_manifest_sha256,
                    "runs_total": args.runs,
                    "runs_completed": len(completed),
                    "current_run": run_index,
                    "current_profile": profile_name,
                    "trajectories_total": trajectories_total,
                    "trajectories_completed": trajectories_completed,
                    "progress_percent": 100.0
                    * trajectories_completed
                    / trajectories_total,
                    "estimated_remaining_seconds": trajectory_remaining,
                    "python_unbuffered_required": True,
                },
            )
            print(
                f"BRIDGE_PROGRESS trajectories={trajectories_completed}/"
                f"{trajectories_total} percent="
                f"{100.0 * trajectories_completed / trajectories_total:.1f} "
                f"eta_seconds={trajectory_remaining:.0f}",
                flush=True,
            )

        record = {
            "schema_version": "same_target_bridge.catalog.v1",
            "campaign_id": CAMPAIGN_ROOT.name,
            "run_index": run_index,
            "run_seed": int(catalog_meta["seed"]),
            "dataset_path": str(catalog_meta["path"]),
            "dataset_sha256": str(catalog_meta["sha256"]),
            "target_source_sha256": sha256_file(ORIGINAL_RESULT_PATH),
            "run_manifest_sha256": run_manifest_sha256,
            "arms": arm_payloads,
        }
        append_jsonl(checkpoint_path, record)
        completed.add(run_index)
        elapsed = time.time() - started
        rate = elapsed / max(1, len(completed) - len(existing))
        remaining = rate * (args.runs - len(completed))
        atomic_json(
            status_path,
            {
                "status": "running",
                "campaign_id": CAMPAIGN_ROOT.name,
                "mode": "smoke" if args.smoke else "canonical",
                "pid": os.getpid(),
                "run_manifest_sha256": run_manifest_sha256,
                "runs_total": args.runs,
                "runs_completed": len(completed),
                "trajectories_total": trajectories_total,
                "trajectories_completed": len(completed) * trajectories_per_catalog,
                "progress_percent": 100.0 * len(completed) / args.runs,
                "current_run": run_index,
                "last_catalog_seconds": time.time() - run_started,
                "estimated_remaining_seconds": remaining,
                "python_unbuffered_required": True,
            },
        )
        print(
            f"BRIDGE_PROGRESS catalogs={len(completed)}/{args.runs} "
            f"percent={100.0 * len(completed) / args.runs:.1f} eta_seconds={remaining:.0f}",
            flush=True,
        )

    records = read_jsonl(checkpoint_path)
    records = [record for record in records if int(record["run_index"]) < args.runs]
    if len(records) != args.runs:
        raise AssertionError("Incomplete terminal checkpoint")
    if not args.smoke:
        analysis = analyze(records)
    else:
        analysis = {
            "schema_version": "same_target_bridge.smoke.v1",
            "campaign_id": CAMPAIGN_ROOT.name,
            "status": "exact_replay_and_all_arm_execution_pass",
            "runs": args.runs,
            "arm_count": len(ARMS),
        }
    terminal = {
        "schema_version": "same_target_bridge.terminal.v1",
        "campaign_id": CAMPAIGN_ROOT.name,
        "mode": "smoke" if args.smoke else "canonical",
        "status": "completed_unverified",
        "config": {
            "runs": args.runs,
            "profiles": list(CORE.PROFILE_ORDER),
            "episodes": args.episodes,
            "checkpoints": list(CHECKPOINTS),
            "arm_count": len(ARMS),
            "mandatory_factorial_arm_count": len(mandatory_arms()),
            "exact_arm_id": EXACT_ARM_ID,
            "primary_corrected_arm_id": PRIMARY_CORRECTED_ARM_ID,
        },
        "input_hashes": {
            "run_manifest_sha256": run_manifest_sha256,
            "frozen_core_sha256": sha256_file(CORE_PATH),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "original_result_sha256": sha256_file(ORIGINAL_RESULT_PATH),
            "historical_reward_provenance_sha256": historical_provenance_hashes,
            "producer_provenance_sha256": producer_provenance_hashes,
            "exact_replay_provenance": exact_replay_provenance,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            **environment_versions,
        },
        "analysis": analysis,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(terminal_path, terminal)
    atomic_json(
        status_path,
        {
            "status": "completed_unverified",
            "campaign_id": CAMPAIGN_ROOT.name,
            "mode": terminal["mode"],
            "pid": os.getpid(),
            "run_manifest_sha256": run_manifest_sha256,
            "runs_total": args.runs,
            "runs_completed": args.runs,
            "trajectories_total": trajectories_total,
            "trajectories_completed": trajectories_total,
            "progress_percent": 100.0,
            "terminal_path": str(terminal_path.relative_to(CAMPAIGN_ROOT)),
            "terminal_sha256": sha256_file(terminal_path),
            "elapsed_seconds": terminal["elapsed_seconds"],
        },
    )
    print("BRIDGE_COMPUTE_COMPLETE status=completed_unverified", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--output-dir", default="outputs/canonical_main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
