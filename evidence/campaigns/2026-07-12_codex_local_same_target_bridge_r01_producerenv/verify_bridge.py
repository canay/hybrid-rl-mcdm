"""Independent fail-closed verifier for the same-target bridge campaign.

This module deliberately does not import ``bridge_main.py`` or
``original_hybrid_core.py``.  It reconstructs the frozen target, TOPSIS,
candidate support, rankings, and metrics from the frozen inputs.  A successful
CLI invocation is the only code path that writes ``FULL_VERIFICATION.json``.
A partial scientific checkpoint exists but remains unopened until terminal
verification; this verifier reads it only after terminal artifacts are present.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats


CAMPAIGN_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = CAMPAIGN_ROOT / "inputs"
MANIFEST_PATH = INPUT_ROOT / "data" / "processed" / "manifest.json"
ORIGINAL_RESULT_PATH = INPUT_ROOT / "results" / "amazon_primary.json"
RUN_MANIFEST_PATH = CAMPAIGN_ROOT / "RUN_MANIFEST.json"
CORE_PATH = CAMPAIGN_ROOT / "src" / "original_hybrid_core.py"
RUNNER_PATH = CAMPAIGN_ROOT / "src" / "bridge_main.py"

CAMPAIGN_ID = CAMPAIGN_ROOT.name
CHECKPOINTS = (500, 1000, 2000, 5000, 10000, 20000, 30000)
METHODS = ("random", "popularity", "topsis_only", "rl_only", "hybrid")
PROFILE_ORDER = ("budget", "quality_seeker", "explorer", "loyal", "balanced")
PROFILE_SEED_OFFSETS = {name: 101 + index * 97 for index, name in enumerate(PROFILE_ORDER)}
TOPSIS_COLUMNS = ("price_pct", "quality_pct", "popularity_pct", "rating_pct")
TOP_K = 7
EPISODES = 30_000
EXPECTED_CATALOGS = 50
EXPECTED_EPSILON_FINAL = max(0.05, 0.30 * (0.9997**EPISODES))
ANALYSIS_SEED = 2026071201
BOOTSTRAP_REPS = 20_000
PAIR_TOLERANCE = 1e-12
EXPECTED_ENVIRONMENT = {
    "numpy": "1.26.0",
    "pandas": "2.2.3",
    "scipy": "1.16.3",
}
PRODUCER_COMMIT_SHA1 = "3b92f6485d20d1a45ac03b60077d20af08060885"
PRODUCER_GIT_BLOBS = {
    "code/hybrid_core.py": "1ffcd66d8b668ca8cbd03f447a24f49daedddfb1",
    "code/run_amazon_experiments.py": "d846056c6dcc3ee21df667a7798876fd2da4a603",
    "requirements.txt": "2bb03a798dcd60d48bd63db03e0d9cee70960b89",
}
HISTORICAL_REWARD_SOURCE_SHA256 = {
    "hybrid_rl_mcdm_v2.py": "90AF7D4D3150099D840C510F5FF420B8773659C2A7A579D04E7B6E711DA65E4F",
    "supplementary_runs.py": "A18AF10D9D7C2C81E400910EC1D0DAE4071322DC1FC24AA0F3B6E022984D8BDC",
}

EXPECTED_HASHES = {
    "src/original_hybrid_core.py": "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f",
    "inputs/data/processed/manifest.json": "81b01f5580109552fc6086c67441159ddad40c1d1447f1061a092a88c6c89652",
    "inputs/results/amazon_primary.json": "cfeaff03084df0d3f0a07a5c8c40308027ca7980288a89cf3616c588d0791ce4",
    "inputs/historical_reward_provenance/hybrid_rl_mcdm_v2.py": "90af7d4d3150099d840c510f5ff420b8773659c2a7a579d04e7b6e711da65e4f",
    "inputs/historical_reward_provenance/supplementary_runs.py": "a18af10d9d7c2c81e400910ec1d0dae4071322dc1fc24aa0f3b6e022984d8bdc",
    "inputs/producer_provenance/v1.0-submission/code/hybrid_core.py": "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f",
    "inputs/producer_provenance/v1.0-submission/code/run_amazon_experiments.py": "361f29012b1618c164d9688bd4887ca6187fc04ec67bc8dd6a9de54fd0a2f15d",
    "inputs/producer_provenance/v1.0-submission/requirements.txt": "5241d0abaccd86ffad73f36592acbabb1bf9331be83dd4678b4ef5d6be71f391",
    "inputs/producer_provenance/v1.0-submission/COMMIT_METADATA.json": "35b41ec3c5a9f4dd1325839aa90563767f71c9624815c302f4a1ce18892f2dad",
    "inputs/producer_provenance/v1.0-submission/PUBLIC_TAG_EXACT_HASHES.json": "2e128e8b9fabe328ce8577cb56407c5bf90c2f7fbed6314ca110930f9bcefe8b",
    "inputs/exact_replay_provenance/EXACT_REPLAY_AUDIT.json": "ba76de6b8043ac4baf34f4f5763c96b2c778d458ec497febc97b6f9135aa655d",
}

PROFILE_HIDDEN: dict[str, dict[str, Any]] = {
    "budget": {
        "brand_pref": {"budget_brand": 0.80, "mid_brand": 0.15, "premium_brand": 0.05},
        "price_range": (10, 160),
        "cat_affinity": {"Electronics": 0.20, "Computers": 0.20, "HomeKitchen": 0.60},
        "recency_weight": 0.10,
    },
    "quality_seeker": {
        "brand_pref": {"budget_brand": 0.05, "mid_brand": 0.20, "premium_brand": 0.75},
        "price_range": (300, 1000),
        "cat_affinity": {"Electronics": 0.55, "Computers": 0.35, "HomeKitchen": 0.10},
        "recency_weight": 0.75,
    },
    "explorer": {
        "brand_pref": {"budget_brand": 0.25, "mid_brand": 0.50, "premium_brand": 0.25},
        "price_range": (50, 700),
        "cat_affinity": {"Electronics": 0.35, "Computers": 0.30, "HomeKitchen": 0.35},
        "recency_weight": 0.65,
    },
    "loyal": {
        "brand_pref": {"budget_brand": 0.05, "mid_brand": 0.30, "premium_brand": 0.65},
        "price_range": (100, 800),
        "cat_affinity": {"Electronics": 0.70, "Computers": 0.20, "HomeKitchen": 0.10},
        "recency_weight": 0.25,
    },
    "balanced": {
        "brand_pref": {"budget_brand": 0.20, "mid_brand": 0.50, "premium_brand": 0.30},
        "price_range": (80, 500),
        "cat_affinity": {"Electronics": 0.35, "Computers": 0.30, "HomeKitchen": 0.35},
        "recency_weight": 0.45,
    },
}


class VerificationError(RuntimeError):
    """A fail-closed verification failure."""


@dataclass
class Audit:
    checks: int = 0
    gates: list[dict[str, Any]] = field(default_factory=list)

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise VerificationError(message)

    def gate(self, gate_id: str, evidence: Mapping[str, Any]) -> None:
        self.gates.append({"gate_id": gate_id, "status": "PASS", "evidence": dict(evidence)})


@dataclass
class VerificationProgress:
    output_dir: Path
    mode: str
    catalogs_total: int
    cells_total: int
    started_monotonic: float = field(default_factory=time.monotonic)

    @property
    def status_path(self) -> Path:
        return self.output_dir / "verification_status.json"

    def _payload(
        self, status: str, catalogs_completed: int, cells_completed: int
    ) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        if cells_completed > 0:
            eta = elapsed / cells_completed * (self.cells_total - cells_completed)
        else:
            eta = None
        return {
            "schema_version": "same_target_bridge.verification_status.v1",
            "status": status,
            "campaign_id": CAMPAIGN_ID,
            "mode": self.mode,
            "pid": os.getpid(),
            "catalogs_total": self.catalogs_total,
            "catalogs_completed": catalogs_completed,
            "cells_total": self.cells_total,
            "cells_completed": cells_completed,
            "progress_percent": (
                100.0 * cells_completed / self.cells_total
                if self.cells_total
                else 0.0
            ),
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": eta,
            "stderr_health": "external runtime log; inspect wrapper-managed stderr",
            "runtime_logs_policy": "campaign-sibling runtime_logs/; wrapper-managed",
            "scientific_values_exposed": False,
            "python_unbuffered_required": True,
        }

    def start(self) -> None:
        atomic_write_json(self.status_path, self._payload("running", 0, 0))

    def update(self, catalogs_completed: int, cells_completed: int) -> None:
        payload = self._payload("running", catalogs_completed, cells_completed)
        atomic_write_json(self.status_path, payload)
        eta = payload["estimated_remaining_seconds"]
        eta_text = "unknown" if eta is None else f"{float(eta):.0f}"
        print(
            "VERIFY_PROGRESS "
            f"catalogs={catalogs_completed}/{self.catalogs_total} "
            f"cells={cells_completed}/{self.cells_total} "
            f"percent={float(payload['progress_percent']):.1f} "
            f"eta_seconds={eta_text}",
            flush=True,
        )

    def complete(self, report_path: Path) -> None:
        payload = self._payload(
            "completed_verified", self.catalogs_total, self.cells_total
        )
        payload["full_verification_path"] = report_path.name
        payload["full_verification_sha256"] = sha256_file(report_path)
        payload["estimated_remaining_seconds"] = 0.0
        atomic_write_json(self.status_path, payload)

    def fail(self, exc: Exception) -> None:
        current: dict[str, Any] = {}
        if self.status_path.is_file():
            try:
                loaded = strict_load(self.status_path)
                if isinstance(loaded, dict):
                    current = loaded
            except Exception:
                current = {}
        current.update(
            {
                "schema_version": "same_target_bridge.verification_status.v1",
                "status": "failed",
                "campaign_id": CAMPAIGN_ID,
                "mode": self.mode,
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stderr_health": "external runtime log; inspect wrapper-managed stderr",
                "scientific_values_exposed": False,
            }
        )
        atomic_write_json(self.status_path, current)


def assert_verification_start_clean(output_dir: Path) -> None:
    """Reject stale verifier state or a pre-existing terminal verification."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Verification output directory missing: {output_dir}")
    stale = [
        name
        for name in ("verification_status.json", "FULL_VERIFICATION.json")
        if (output_dir / name).exists()
    ]
    if stale:
        raise FileExistsError(
            f"Stale verification artifacts require archival before rerun: {stale}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise VerificationError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def strict_load(path: Path) -> Any:
    try:
        return strict_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Cannot read strict JSON {path}: {exc}") from exc


def strict_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                item = strict_loads(line)
                if not isinstance(item, dict):
                    raise VerificationError(f"JSONL line {line_no} is not an object")
                records.append(item)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Cannot read strict JSONL {path}: {exc}") from exc
    return records


def ensure_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise VerificationError(f"Non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            ensure_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_finite(item, f"{path}[{index}]")


def safe_path(root: Path, relative: str) -> Path:
    cleaned = str(relative).replace("\\", "/")
    candidate = (root / cleaned).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise VerificationError(f"Path escapes campaign/input root: {relative}") from exc
    return candidate


def arm_id(candidate: str, bonus: float, reward: str) -> str:
    return f"candidate={candidate}__bonus={bonus:.2f}__reward={reward}"


def expected_arms() -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog"):
        for bonus in (0.20, 0.00):
            for reward in ("implemented_r0", "inclusive_range_fix", "component_continuous_fix"):
                arms.append({
                    "arm_id": arm_id(candidate, bonus, reward),
                    "candidate": candidate,
                    "gt_bonus": bonus,
                    "reward_model": reward,
                    "role": "mandatory_factorial",
                })
    for candidate, bonus in (("oracle_gt_hidden30", 0.20), ("full_catalog", 0.00)):
        arms.append({
            "arm_id": arm_id(candidate, bonus, "historical_funnel_coefficients_on_may_h"),
            "candidate": candidate,
            "gt_bonus": bonus,
            "reward_model": "historical_funnel_coefficients_on_may_h",
            "role": "secondary_reward_specification_sensitivity",
        })
    return arms


def expected_factorial_contrast_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    core_rewards = ("implemented_r0", "inclusive_range_fix", "component_continuous_fix")
    for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog"):
        for reward in core_rewards:
            specs.append({
                "contrast_id": f"bonus_removal__{candidate}__{reward}",
                "factor": "gt_bonus",
                "arm_a": arm_id(candidate, 0.00, reward),
                "arm_b": arm_id(candidate, 0.20, reward),
                "direction": "bonus_0.00_minus_0.20",
            })
    for bonus in (0.20, 0.00):
        for reward in core_rewards:
            for a, b in (
                ("hidden30_only", "oracle_gt_hidden30"),
                ("full_catalog", "hidden30_only"),
                ("full_catalog", "oracle_gt_hidden30"),
            ):
                specs.append({
                    "contrast_id": f"candidate__{a}_minus_{b}__bonus={bonus:.2f}__{reward}",
                    "factor": "candidate_support",
                    "arm_a": arm_id(a, bonus, reward),
                    "arm_b": arm_id(b, bonus, reward),
                    "direction": f"{a}_minus_{b}",
                })
    for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog"):
        for bonus in (0.20, 0.00):
            for a, b in (
                ("inclusive_range_fix", "implemented_r0"),
                ("component_continuous_fix", "implemented_r0"),
                ("component_continuous_fix", "inclusive_range_fix"),
            ):
                specs.append({
                    "contrast_id": f"reward__{a}_minus_{b}__{candidate}__bonus={bonus:.2f}",
                    "factor": "reward_implementation",
                    "arm_a": arm_id(candidate, bonus, a),
                    "arm_b": arm_id(candidate, bonus, b),
                    "direction": f"{a}_minus_{b}",
                })
    return specs


def expected_sensitivity_contrast_specs() -> list[dict[str, str]]:
    reward = "historical_funnel_coefficients_on_may_h"
    return [
        {
            "contrast_id": "historical_funnel_minus_continuous__oracle_bonus",
            "arm_a": arm_id("oracle_gt_hidden30", 0.20, reward),
            "arm_b": arm_id("oracle_gt_hidden30", 0.20, "component_continuous_fix"),
        },
        {
            "contrast_id": "historical_funnel_minus_continuous__full_no_bonus",
            "arm_a": arm_id("full_catalog", 0.00, reward),
            "arm_b": arm_id("full_catalog", 0.00, "component_continuous_fix"),
        },
    ]


ARMS = expected_arms()
ARM_MAP = {arm["arm_id"]: arm for arm in ARMS}
EXACT_ARM_ID = arm_id("oracle_gt_hidden30", 0.20, "implemented_r0")
PRIMARY_CORRECTED_ARM_ID = arm_id("full_catalog", 0.00, "component_continuous_fix")


def normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo + 1e-10)


def top_rank(scores: np.ndarray, k: int = TOP_K) -> list[int]:
    return [int(item) for item in np.argsort(np.asarray(scores))[::-1][:k]]


def f1_at_7(rank: Sequence[int], truth: set[int]) -> float:
    predicted = set(int(item) for item in rank[:TOP_K])
    return 0.0 if not predicted or not truth else 2.0 * len(predicted & truth) / (len(predicted) + len(truth))


def ndcg_at_7(rank: Sequence[int], truth: set[int]) -> float:
    dcg = sum(1.0 / np.log2(index + 2.0) for index, item in enumerate(rank[:TOP_K]) if int(item) in truth)
    idcg = sum(1.0 / np.log2(index + 2.0) for index in range(min(TOP_K, len(truth))))
    return float(dcg / idcg) if idcg else 0.0


def read_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise VerificationError(f"Empty catalog: {path}")
    required = set(TOPSIS_COLUMNS) | {"brand", "category", "price", "recency_pct"}
    if not required.issubset(rows[0]):
        raise VerificationError(f"Catalog columns missing in {path}")
    return {
        "n": len(rows),
        "brand": np.asarray([row["brand"] for row in rows], dtype=object),
        "category": np.asarray([row["category"] for row in rows], dtype=object),
        "price": np.asarray([float(row["price"]) for row in rows], dtype=float),
        "recency_pct": np.asarray([float(row["recency_pct"]) for row in rows], dtype=float),
        **{column: np.asarray([float(row[column]) for row in rows], dtype=float) for column in TOPSIS_COLUMNS},
    }


def independent_topsis(data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack([data[column] for column in TOPSIS_COLUMNS]).astype(float)
    matrix = np.clip(matrix, 1e-10, None)
    proportions = np.clip(matrix / matrix.sum(axis=0, keepdims=True), 1e-10, 1.0)
    entropy = -np.sum(proportions * np.log(proportions), axis=0) / np.log(len(matrix))
    diversification = 1.0 - entropy
    weights = 0.10 + (diversification / diversification.sum()) * 0.60
    weights = np.clip(weights, 0.10, None)
    weights = weights / weights.sum()
    norms = np.sqrt((matrix**2).sum(axis=0))
    norms[norms == 0] = 1.0
    weighted = (matrix / norms) * weights
    plus, minus = weighted.max(axis=0), weighted.min(axis=0)
    d_plus = np.sqrt(((weighted - plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - minus) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    denom[denom == 0] = 1e-10
    return d_minus / denom, weights


def hidden_components(data: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, np.ndarray]:
    brand = np.asarray([profile["brand_pref"].get(value, 0.10) for value in data["brand"]], dtype=float)
    lo, hi = profile["price_range"]
    center, half_range = (lo + hi) / 2.0, (hi - lo) / 2.0 + 1.0
    price = np.clip(1.0 - np.abs(data["price"] - center) / half_range, 0.0, 1.0)
    category = np.asarray([profile["cat_affinity"].get(value, 0.05) for value in data["category"]], dtype=float)
    category = category / category.max()
    rw = float(profile["recency_weight"])
    recency = data["recency_pct"] * rw + (1.0 - rw) * 0.5
    return {"brand": brand, "price": price, "category": category, "recency": recency}


def hidden_utility(data: Mapping[str, Any], profile: Mapping[str, Any]) -> np.ndarray:
    comp = hidden_components(data, profile)
    return normalize(0.45 * comp["brand"] + 0.30 * comp["price"] + 0.15 * comp["category"] + 0.10 * comp["recency"])


def observable_utility(data: Mapping[str, Any]) -> np.ndarray:
    return normalize(sum(np.asarray(data[column], dtype=float) for column in TOPSIS_COLUMNS) / 4.0)


def frozen_gt(data: Mapping[str, Any], profile_name: str, run_seed: int) -> list[int]:
    profile = PROFILE_HIDDEN[profile_name]
    gt_seed = run_seed + PROFILE_SEED_OFFSETS[profile_name]
    rng = np.random.RandomState(gt_seed + 7777)
    scores = 0.50 * observable_utility(data) + 0.50 * hidden_utility(data, profile)
    scores = normalize(scores + rng.normal(0.0, 0.015, int(data["n"])))
    return top_rank(scores)


def expected_candidate(data: Mapping[str, Any], profile_name: str, kind: str, gt: set[int]) -> np.ndarray:
    hidden30 = set(top_rank(hidden_utility(data, PROFILE_HIDDEN[profile_name]), 30))
    if kind == "oracle_gt_hidden30":
        return np.asarray(sorted(hidden30 | gt), dtype=np.int64)
    if kind == "hidden30_only":
        return np.asarray(sorted(hidden30), dtype=np.int64)
    if kind == "full_catalog":
        return np.arange(int(data["n"]), dtype=np.int64)
    raise VerificationError(f"Unknown candidate kind: {kind}")


def candidate_hash(pool: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest()


def independent_reward_probabilities(
    data: Mapping[str, Any], profile_name: str, model: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    profile = PROFILE_HIDDEN[profile_name]
    comp = hidden_components(data, profile)
    brand = comp["brand"]
    recency = comp["recency"]
    if model == "implemented_r0":
        category = np.ones(int(data["n"]), dtype=float)
        price = (comp["price"] > 0.999999).astype(float)
        engage = np.clip(
            0.40 * brand + 0.35 * price + 0.15 * category + 0.10 * recency,
            0.0,
            1.0,
        )
        convert = np.clip(
            0.50 * brand + 0.30 * price + 0.20 * category, 0.0, 1.0
        )
        diagnostics = {
            "category_distinct": int(np.unique(category).size),
            "price_match_count": int(price.sum()),
            "definition": "exact original fast path",
        }
    elif model == "inclusive_range_fix":
        lo, hi = profile["price_range"]
        price = ((data["price"] >= lo) & (data["price"] <= hi)).astype(float)
        category_scale = max(float(value) for value in profile["cat_affinity"].values())
        category = np.asarray(
            [
                float(profile["cat_affinity"].get(value, 0.0)) / category_scale
                for value in data["category"]
            ],
            dtype=float,
        )
        engage = np.clip(
            0.40 * brand + 0.35 * price + 0.15 * category + 0.10 * recency,
            0.0,
            1.0,
        )
        convert = np.clip(
            0.50 * brand + 0.30 * price + 0.20 * category, 0.0, 1.0
        )
        diagnostics = {
            "category_distinct": int(np.unique(category).size),
            "price_match_count": int(price.sum()),
            "definition": "original coefficients; category profile-scaled; inclusive binary price interval",
        }
    elif model == "component_continuous_fix":
        category, price = comp["category"], comp["price"]
        engage = np.clip(
            0.40 * brand + 0.35 * price + 0.15 * category + 0.10 * recency,
            0.0,
            1.0,
        )
        convert = np.clip(
            0.50 * brand + 0.30 * price + 0.20 * category, 0.0, 1.0
        )
        diagnostics = {
            "category_distinct": int(np.unique(category).size),
            "price_match_count": int(np.sum(price > 0.0)),
            "definition": "original coefficients; full-vector category and continuous triangular price fit",
        }
    elif model == "historical_funnel_coefficients_on_may_h":
        category, price = comp["category"], comp["price"]
        hidden = hidden_utility(data, profile)
        engage = np.clip(0.70 * hidden + 0.10, 0.05, 0.95)
        convert = np.clip(0.50 * hidden, 0.02, 0.80)
        diagnostics = {
            "category_distinct": int(np.unique(category).size),
            "price_match_count": int(np.sum(price > 0.0)),
            "definition": (
                "secondary historical sensitivity: "
                "P(engage)=clip(0.7H+0.1,0.05,0.95); "
                "P(convert|engage)=clip(0.5H,0.02,0.80)"
            ),
            "historical_source_sha256": HISTORICAL_REWARD_SOURCE_SHA256,
        }
    else:
        raise VerificationError(f"Unknown reward model: {model}")
    if not np.all(np.isfinite(engage)) or not np.all(np.isfinite(convert)):
        raise VerificationError(f"Non-finite independent reward probabilities: {model}")
    if np.any((engage < 0.0) | (engage > 1.0)) or np.any(
        (convert < 0.0) | (convert > 1.0)
    ):
        raise VerificationError(f"Out-of-range independent reward probabilities: {model}")
    return engage, convert, diagnostics


def expected_reward_diagnostics(
    data: Mapping[str, Any], profile_name: str, model: str
) -> tuple[int, int]:
    _, _, diagnostics = independent_reward_probabilities(data, profile_name, model)
    return int(diagnostics["category_distinct"]), int(
        diagnostics["price_match_count"]
    )


def independent_train_profile(
    data: Mapping[str, Any],
    profile_name: str,
    profile_idx: int,
    run_seed: int,
    gt_set: set[int],
    gt_rank: Sequence[int],
    topsis_scores: np.ndarray,
    arm: Mapping[str, Any],
    episodes: int = EPISODES,
) -> dict[str, Any]:
    """Independent exact replay of one stochastic arm/profile trajectory."""
    n = int(data["n"])
    pool = expected_candidate(data, profile_name, str(arm["candidate"]), gt_set)
    p_engage, p_convert, diagnostics = independent_reward_probabilities(
        data, profile_name, str(arm["reward_model"])
    )
    q_scores = np.zeros(n, dtype=float)
    visits = np.zeros(n, dtype=np.int32)
    epsilon = 0.30
    action_rng = np.random.RandomState(run_seed + profile_idx * 13)
    reward_rng = np.random.RandomState(run_seed + profile_idx * 997)
    gt_mask: np.ndarray | None = None
    bonus = float(arm["gt_bonus"])
    if bonus != 0.0:
        gt_mask = np.zeros(n, dtype=bool)
        gt_mask[list(gt_set)] = True

    random_rank = [
        int(item)
        for item in np.random.RandomState(run_seed + 5555).choice(
            n, size=TOP_K, replace=False
        )
    ]
    popularity_rank = [
        int(item)
        for item in np.argsort(
            -np.asarray(data["popularity_pct"], dtype=float), kind="stable"
        )[:TOP_K]
    ]
    topsis_rank = top_rank(topsis_scores)
    checkpoints = {checkpoint for checkpoint in CHECKPOINTS if checkpoint <= episodes}
    checkpoint_f1: dict[str, dict[str, float]] = {}
    final_rankings: dict[str, list[int]] | None = None

    for episode in range(1, episodes + 1):
        if action_rng.random() < epsilon:
            action = int(action_rng.choice(pool))
        else:
            action = int(pool[np.argmax(q_scores[pool])])
        reward = -0.02
        if reward_rng.random() < p_engage[action]:
            reward += 0.30
            if reward_rng.random() < p_convert[action]:
                reward += 1.00
        if gt_mask is not None and gt_mask[action]:
            reward += bonus
        visits[action] += 1
        q_scores[action] += 0.05 * (reward - q_scores[action])
        epsilon = max(0.05, epsilon * 0.9997)

        if episode in checkpoints:
            rankings = {
                "random": random_rank,
                "popularity": popularity_rank,
                "topsis_only": topsis_rank,
                "rl_only": top_rank(q_scores),
                "hybrid": top_rank(
                    0.50 * normalize(q_scores)
                    + 0.50 * normalize(np.asarray(topsis_scores, dtype=float))
                ),
            }
            checkpoint_f1[str(episode)] = {
                method: float(f1_at_7(rank, gt_set))
                for method, rank in rankings.items()
            }
            if episode == episodes:
                final_rankings = {
                    method: [int(item) for item in rank]
                    for method, rank in rankings.items()
                }
    if final_rankings is None:
        raise VerificationError("Independent replay did not produce final rankings")
    final_metrics = {
        method: {
            "f1_at_7": float(f1_at_7(rank, gt_set)),
            "ndcg_at_7": float(ndcg_at_7(rank, gt_set)),
        }
        for method, rank in final_rankings.items()
    }
    return {
        "profile_name": profile_name,
        "candidate_count": int(pool.size),
        "candidate_sha256": candidate_hash(pool),
        "reward_diagnostics": diagnostics,
        "checkpoint_f1": checkpoint_f1,
        "final_metrics": final_metrics,
        "final_rankings": final_rankings,
        "gt_rank": [int(item) for item in gt_rank],
        "q_scores": [float(value) for value in q_scores],
        "visits": [int(value) for value in visits],
        "epsilon_final": float(epsilon),
    }


def verify_run_manifest(campaign_root: Path, required_paths: set[str] | None = None) -> tuple[dict[str, Any], str]:
    manifest_path = campaign_root / "RUN_MANIFEST.json"
    payload = strict_load(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != "same_target_bridge.run_manifest.v1":
        raise VerificationError("RUN_MANIFEST schema mismatch")
    locked_packages = payload.get("environment", {}).get("packages")
    if locked_packages != EXPECTED_ENVIRONMENT:
        raise VerificationError(
            f"RUN_MANIFEST producer-environment lock mismatch: {locked_packages}"
        )
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise VerificationError("RUN_MANIFEST.files must be a non-empty list")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(("path", "sha256", "bytes")) - set(entry):
            raise VerificationError("Malformed RUN_MANIFEST file entry")
        relative = str(entry["path"]).replace("\\", "/")
        if relative in seen:
            raise VerificationError(f"Duplicate RUN_MANIFEST path: {relative}")
        seen.add(relative)
        path = safe_path(campaign_root, relative)
        if not path.is_file():
            raise VerificationError(f"RUN_MANIFEST file missing: {relative}")
        if sha256_file(path) != str(entry["sha256"]).lower():
            raise VerificationError(f"RUN_MANIFEST hash mismatch: {relative}")
        if path.stat().st_size != int(entry["bytes"]):
            raise VerificationError(f"RUN_MANIFEST byte count mismatch: {relative}")
    if required_paths is not None and not required_paths.issubset(seen):
        missing = sorted(required_paths - seen)
        raise VerificationError(f"RUN_MANIFEST required paths missing: {missing}")
    return payload, sha256_file(manifest_path)


def required_manifest_paths(input_manifest: Mapping[str, Any]) -> set[str]:
    paths = {
        "PROTOCOL_LOCK.md",
        "src/bridge_main.py",
        "src/lock_campaign.py",
        "src/original_hybrid_core.py",
        "verify_bridge.py",
        "tests/test_bridge_contract.py",
        "tests/test_verify_bridge.py",
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
    for item in input_manifest.get("runs", []):
        paths.add("inputs/" + str(item["path"]).replace("\\", "/"))
    return paths


def installed_environment() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def verify_producer_and_replay_provenance(audit: Audit) -> dict[str, Any]:
    """Independently verify the public producer tag and terminal replay gate."""
    actual_environment = installed_environment()
    audit.require(
        actual_environment == EXPECTED_ENVIRONMENT,
        f"Verifier is not running in the producer environment: {actual_environment}",
    )
    producer_root = INPUT_ROOT / "producer_provenance" / "v1.0-submission"
    metadata = strict_load(producer_root / "COMMIT_METADATA.json")
    public_hashes = strict_load(producer_root / "PUBLIC_TAG_EXACT_HASHES.json")
    audit.require(
        metadata.get("schema_version") == "producer_provenance.commit_metadata.v1"
        and metadata.get("tag") == "v1.0-submission"
        and metadata.get("commit_sha1") == PRODUCER_COMMIT_SHA1
        and metadata.get("tree_sha1") == "39a0899c2ccdbb9bb3b65de0bebe225c741bd30f",
        "Producer commit metadata mismatch",
    )
    audit.require(
        public_hashes.get("schema_version") == "producer_provenance.public_tag_hashes.v1"
        and public_hashes.get("tag_ref") == "refs/tags/v1.0-submission"
        and public_hashes.get("public_ls_remote_commit_sha1") == PRODUCER_COMMIT_SHA1
        and public_hashes.get("local_tag_commit_sha1") == PRODUCER_COMMIT_SHA1,
        "Public-tag exact commit metadata mismatch",
    )
    file_map = {str(item.get("path")): item for item in public_hashes.get("files", [])}
    expected_source_paths = {
        "code/hybrid_core.py": "inputs/producer_provenance/v1.0-submission/code/hybrid_core.py",
        "code/run_amazon_experiments.py": "inputs/producer_provenance/v1.0-submission/code/run_amazon_experiments.py",
        "requirements.txt": "inputs/producer_provenance/v1.0-submission/requirements.txt",
    }
    for repository_path, campaign_path in expected_source_paths.items():
        item = file_map.get(repository_path, {})
        audit.require(
            item.get("git_blob_sha1") == PRODUCER_GIT_BLOBS[repository_path]
            and item.get("sha256") == EXPECTED_HASHES[campaign_path]
            and int(item.get("bytes", -1)) == safe_path(CAMPAIGN_ROOT, campaign_path).stat().st_size,
            f"Public-tag file evidence mismatch: {repository_path}",
        )
    audit.require(
        EXPECTED_HASHES["src/original_hybrid_core.py"]
        == EXPECTED_HASHES["inputs/producer_provenance/v1.0-submission/code/hybrid_core.py"],
        "Executable core is not the exact producer-tag core",
    )

    replay = strict_load(INPUT_ROOT / "exact_replay_provenance" / "EXACT_REPLAY_AUDIT.json")
    replay_cells = replay.get("cells")
    audit.require(
        replay.get("status") == "completed"
        and replay.get("all_exact") is True
        and int(replay.get("cells_total", -1)) == 250
        and int(replay.get("cells_exact", -1)) == 250
        and int(replay.get("cells_mismatched", -1)) == 0
        and isinstance(replay_cells, list)
        and len(replay_cells) == 250
        and all(cell.get("exact") is True for cell in replay_cells),
        "Terminal exact replay provenance is not all_exact 250/250",
    )
    audit.require(
        replay.get("numpy") == EXPECTED_ENVIRONMENT["numpy"]
        and replay.get("pandas") == EXPECTED_ENVIRONMENT["pandas"],
        "Exact replay provenance environment mismatch",
    )
    return {
        "environment": actual_environment,
        "producer_commit_sha1": PRODUCER_COMMIT_SHA1,
        "exact_replay_sha256": EXPECTED_HASHES[
            "inputs/exact_replay_provenance/EXACT_REPLAY_AUDIT.json"
        ],
        "exact_replay_cells": 250,
    }


def verify_runner_source_contract(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    compact = re.sub(r"\s+", "", source).replace('"', "'")
    required = (
        "ifkind=='hidden30_only':",
        "ifkind=='full_catalog':",
        "ifgt_setisnotNone:",
        "iffloat(arm['gt_bonus'])!=0.0:",
        "gt_mask[list(gt_set)]=True",
        "ifgt_maskisnotNoneandgt_mask[action]:",
        "reward+=float(arm['gt_bonus'])",
        "(price_fit>0.999999).astype(float)",
        "category=np.ones(len(df),dtype=float)",
        "(prices>=float(price_lo))&(prices<=float(price_hi))",
        "components=CORE.hidden_components(df,profile)",
        "allow_nan=False",
        "assert_json_finite(payload)",
        "'paired_t_defined':t_defined",
        "'wilcoxon_defined':wilcoxon_defined",
        "ifstale:",
        "output_dir.mkdir(parents=True,exist_ok=False)",
        "prepare_output_directory(output_dir,args.resume)",
    )
    for fragment in required:
        if fragment not in compact:
            raise VerificationError(f"Runner source contract missing: {fragment}")
    environment_gate = compact.find("environment_versions=assert_exact_environment()")
    output_preparation = compact.find("prepare_output_directory(output_dir,args.resume)")
    if (
        environment_gate < 0
        or output_preparation < 0
        or environment_gate > output_preparation
    ):
        raise VerificationError(
            "Exact producer environment is not gated before output preparation"
        )
    # Historical equation-locked H funnel: 0.7H+0.1 and 0.5H with stated clips.
    h_engage = re.search(r"p_engage=np\.clip\((?:0\.70|0\.7)\*hidden\+(?:0\.10|0\.1),(?:0\.05|0\.050),(?:0\.95|0\.950)\)", compact)
    h_convert = re.search(r"p_convert=np\.clip\((?:0\.50|0\.5)\*hidden,(?:0\.02|0\.020),(?:0\.80|0\.8)\)", compact)
    if h_engage is None or h_convert is None:
        raise VerificationError("Historical H-funnel implementation does not match 0.7H+0.1 / 0.5H provenance")


def close(a: float, b: float, atol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def validate_rank(rank: Any, n: int, label: str) -> list[int]:
    if not isinstance(rank, list) or len(rank) != TOP_K:
        raise VerificationError(f"{label} must contain exactly seven items")
    values = [int(item) for item in rank]
    if len(set(values)) != TOP_K or min(values) < 0 or max(values) >= n:
        raise VerificationError(f"{label} contains duplicate/out-of-range items")
    return values


def original_profile_map(original: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for catalog in original["artifacts"]:
        run_index = int(catalog["run_index"])
        for profile in catalog["profile_results"]:
            key = (run_index, str(profile["profile_name"]))
            if key in result:
                raise VerificationError(f"Duplicate original profile cell: {key}")
            result[key] = profile
    return result


def signature(q: np.ndarray, visits: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(q, dtype="<f8").tobytes())
    digest.update(np.asarray(visits, dtype="<i8").tobytes())
    return digest.hexdigest()


def verify_profile(
    audit: Audit,
    payload: Mapping[str, Any],
    arm: Mapping[str, Any],
    data: Mapping[str, Any],
    profile_name: str,
    run_seed: int,
    topsis_scores: np.ndarray,
    topsis_rank: list[int],
    stored: Mapping[str, Any],
    exact: bool,
) -> str:
    label = f"profile={profile_name} arm={arm['arm_id']}"
    n = int(data["n"])
    audit.require(payload.get("profile_name") == profile_name, f"Profile-name mismatch: {label}")
    gt_rank = validate_rank(payload.get("gt_rank"), n, f"{label}.gt_rank")
    stored_gt_rank = [int(item) for item in stored["final"]["gt_rank"]]
    stored_gt = set(int(item) for item in stored["final"]["gt_set"])
    audit.require(gt_rank == stored_gt_rank and set(gt_rank) == stored_gt, f"Frozen GT mismatch: {label}")
    audit.require(gt_rank == frozen_gt(data, profile_name, run_seed), f"Independent GT reconstruction mismatch: {label}")

    pool = expected_candidate(data, profile_name, str(arm["candidate"]), stored_gt)
    audit.require(int(payload.get("candidate_count", -1)) == len(pool), f"Candidate count mismatch: {label}")
    audit.require(payload.get("candidate_sha256") == candidate_hash(pool), f"Candidate hash mismatch: {label}")

    diagnostics = payload.get("reward_diagnostics")
    audit.require(isinstance(diagnostics, dict), f"Reward diagnostics missing: {label}")
    distinct, price_count = expected_reward_diagnostics(data, profile_name, str(arm["reward_model"]))
    audit.require(int(diagnostics.get("category_distinct", -1)) == distinct, f"Category diagnostic mismatch: {label}")
    audit.require(int(diagnostics.get("price_match_count", -1)) == price_count, f"Price diagnostic mismatch: {label}")
    audit.require(isinstance(diagnostics.get("definition"), str) and diagnostics["definition"], f"Reward definition missing: {label}")

    q = np.asarray(payload.get("q_scores"), dtype=float)
    raw_visits = payload.get("visits")
    audit.require(q.shape == (n,) and np.all(np.isfinite(q)), f"Invalid Q vector: {label}")
    audit.require(isinstance(raw_visits, list) and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_visits), f"Visits must be integer JSON values: {label}")
    visits = np.asarray(raw_visits, dtype=np.int64)
    audit.require(visits.shape == (n,) and np.all(visits >= 0), f"Invalid visit vector: {label}")
    audit.require(int(visits.sum()) == EPISODES, f"Visit sum mismatch: {label}")
    outside = np.ones(n, dtype=bool)
    outside[pool] = False
    audit.require(np.all(visits[outside] == 0) and np.all(q[outside] == 0.0), f"Training escaped candidate support: {label}")

    rankings = payload.get("final_rankings")
    metrics = payload.get("final_metrics")
    checkpoints = payload.get("checkpoint_f1")
    audit.require(isinstance(rankings, dict) and set(rankings) == set(METHODS), f"Final ranking methods mismatch: {label}")
    audit.require(isinstance(metrics, dict) and set(metrics) == set(METHODS), f"Final metric methods mismatch: {label}")
    audit.require(isinstance(checkpoints, dict) and set(checkpoints) == {str(cp) for cp in CHECKPOINTS}, f"Checkpoint set mismatch: {label}")

    expected_random = [int(item) for item in np.random.RandomState(run_seed + 5555).choice(n, size=TOP_K, replace=False)]
    popularity = np.asarray(data["popularity_pct"], dtype=float)
    expected_popularity = [int(item) for item in np.argsort(-popularity, kind="stable")[:TOP_K]]
    expected_rl = top_rank(q)
    expected_hybrid = top_rank(0.50 * normalize(q) + 0.50 * normalize(topsis_scores))
    expected_rankings = {
        "random": expected_random,
        "popularity": expected_popularity,
        "topsis_only": topsis_rank,
        "rl_only": expected_rl,
        "hybrid": expected_hybrid,
    }
    for method in METHODS:
        rank = validate_rank(rankings[method], n, f"{label}.{method}")
        audit.require(rank == expected_rankings[method], f"Independent final-ranking mismatch: {label} {method}")
        got = metrics[method]
        audit.require(isinstance(got, dict), f"Metric payload malformed: {label} {method}")
        expected_f1, expected_ndcg = f1_at_7(rank, stored_gt), ndcg_at_7(rank, stored_gt)
        audit.require(close(got.get("f1_at_7"), expected_f1), f"Raw-ranking F1 mismatch: {label} {method}")
        audit.require(close(got.get("ndcg_at_7"), expected_ndcg), f"Raw-ranking NDCG mismatch: {label} {method}")
        for checkpoint in CHECKPOINTS:
            cell = checkpoints[str(checkpoint)]
            audit.require(isinstance(cell, dict) and set(cell) == set(METHODS), f"Checkpoint method set mismatch: {label} cp={checkpoint}")
            value = float(cell[method])
            audit.require(math.isfinite(value) and 0.0 <= value <= 1.0, f"Invalid checkpoint F1: {label} cp={checkpoint} {method}")
            if method in {"random", "popularity", "topsis_only"}:
                audit.require(close(value, expected_f1), f"Static checkpoint drift: {label} cp={checkpoint} {method}")
            if checkpoint == EPISODES:
                audit.require(close(value, expected_f1), f"Final checkpoint/metric mismatch: {label} {method}")

    audit.require(close(payload.get("epsilon_final"), EXPECTED_EPSILON_FINAL), f"Final epsilon mismatch: {label}")

    if exact:
        final = stored["final"]
        stored_keys = {
            "random": "random_rank", "popularity": "popularity_rank", "topsis_only": "topsis_rank",
            "rl_only": "rl_rank", "hybrid": "hybrid_rank",
        }
        for method, key in stored_keys.items():
            audit.require(rankings[method] == [int(item) for item in final[key]], f"Exact-r0 ranking mismatch: {label} {method}")
            for checkpoint in CHECKPOINTS:
                audit.require(float(checkpoints[str(checkpoint)][method]) == float(stored["f1"][method][str(checkpoint)]), f"Exact-r0 checkpoint mismatch: {label} cp={checkpoint} {method}")
        audit.require(np.array_equal(q, np.asarray(final["q_scores"], dtype=float)), f"Exact-r0 Q mismatch: {label}")
        audit.require(raw_visits == [int(item) for item in final["visits"]], f"Exact-r0 visits mismatch: {label}")

    replayed = independent_train_profile(
        data=data,
        profile_name=profile_name,
        profile_idx=PROFILE_ORDER.index(profile_name),
        run_seed=run_seed,
        gt_set=stored_gt,
        gt_rank=stored_gt_rank,
        topsis_scores=topsis_scores,
        arm=arm,
        episodes=EPISODES,
    )
    audit.require(
        diagnostics == replayed["reward_diagnostics"],
        f"Full replay reward diagnostics mismatch: {label}",
    )
    audit.require(
        int(payload.get("candidate_count", -1)) == replayed["candidate_count"]
        and payload.get("candidate_sha256") == replayed["candidate_sha256"],
        f"Full replay candidate mismatch: {label}",
    )
    audit.require(
        np.array_equal(q, np.asarray(replayed["q_scores"], dtype=float)),
        f"Full stochastic replay Q mismatch: {label}",
    )
    audit.require(
        raw_visits == replayed["visits"],
        f"Full stochastic replay visits mismatch: {label}",
    )
    audit.require(
        checkpoints == replayed["checkpoint_f1"],
        f"Full stochastic replay checkpoints mismatch: {label}",
    )
    audit.require(
        rankings == replayed["final_rankings"],
        f"Full stochastic replay rankings mismatch: {label}",
    )
    audit.require(
        metrics == replayed["final_metrics"]
        and float(payload.get("epsilon_final")) == replayed["epsilon_final"],
        f"Full stochastic replay metrics/epsilon mismatch: {label}",
    )
    return signature(q, visits)


def verify_factor_effects(signatures: Mapping[tuple[int, str, str], str], run_indices: Sequence[int]) -> None:
    cells = [(run, profile) for run in run_indices for profile in PROFILE_ORDER]
    core_rewards = ("implemented_r0", "inclusive_range_fix", "component_continuous_fix")
    for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog"):
        for reward in core_rewards:
            a = arm_id(candidate, 0.20, reward)
            b = arm_id(candidate, 0.00, reward)
            if not any(signatures[(run, profile, a)] != signatures[(run, profile, b)] for run, profile in cells):
                raise VerificationError(f"GT-bonus factor is a no-op: {candidate}/{reward}")
    for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog"):
        for bonus in (0.20, 0.00):
            ids = [arm_id(candidate, bonus, reward) for reward in core_rewards]
            if not any(len({signatures[(run, profile, aid)] for aid in ids}) > 1 for run, profile in cells):
                raise VerificationError(f"Reward-model factor is a no-op: {candidate}/{bonus}")
    for bonus in (0.20, 0.00):
        for reward in core_rewards:
            ids = [arm_id(candidate, bonus, reward) for candidate in ("oracle_gt_hidden30", "hidden30_only", "full_catalog")]
            if not any(len({signatures[(run, profile, aid)] for aid in ids}) > 1 for run, profile in cells):
                raise VerificationError(f"Candidate factor is a no-op: {bonus}/{reward}")


def independent_vector_summary(
    vector: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    arr = np.asarray(vector, dtype=float)
    if arr.shape != (EXPECTED_CATALOGS,) or not np.all(np.isfinite(arr)):
        raise VerificationError(
            f"Independent summary requires {EXPECTED_CATALOGS} finite values"
        )
    indices = rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))
    bootstrap_means = arr[indices].mean(axis=1)
    if not np.all(np.isfinite(bootstrap_means)):
        raise VerificationError("Independent bootstrap produced non-finite means")
    return {
        "mean": float(arr.mean()),
        "sample_sd": float(arr.std(ddof=1)),
        "bootstrap_ci95_lo": float(np.percentile(bootstrap_means, 2.5)),
        "bootstrap_ci95_hi": float(np.percentile(bootstrap_means, 97.5)),
        "n_catalogs": int(arr.size),
        "raw_catalog_means": [float(value) for value in arr],
    }


def independent_paired_summary(
    difference: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    diff = np.asarray(difference, dtype=float)
    result = independent_vector_summary(diff, rng)
    constant_difference = bool(np.all(diff == diff[0]))
    paired_t_defined = not constant_difference
    if paired_t_defined:
        t_result = stats.ttest_1samp(diff, popmean=0.0)
        t_stat = float(t_result.statistic)
        t_p = float(t_result.pvalue)
        if not math.isfinite(t_stat) or not math.isfinite(t_p):
            raise VerificationError("Independent paired t-test is non-finite")
    else:
        t_stat, t_p = None, None

    wilcoxon_defined = bool(np.any(diff != 0.0))
    if wilcoxon_defined:
        w_result = stats.wilcoxon(
            diff, zero_method="wilcox", alternative="two-sided"
        )
        w_stat = float(w_result.statistic)
        w_p = float(w_result.pvalue)
        if not math.isfinite(w_stat) or not math.isfinite(w_p):
            raise VerificationError("Independent Wilcoxon result is non-finite")
    else:
        w_stat, w_p = None, None

    sample_sd = float(diff.std(ddof=1))
    cohen_dz_defined = not constant_difference
    result.update(
        {
            "paired_t_defined": paired_t_defined,
            "paired_t_stat": t_stat,
            "paired_t_p_two_sided": t_p,
            "cohen_dz_defined": cohen_dz_defined,
            "cohen_dz": (
                float(diff.mean() / sample_sd) if cohen_dz_defined else None
            ),
            "wilcoxon_defined": wilcoxon_defined,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_two_sided": w_p,
            "wins": int(np.sum(diff > PAIR_TOLERANCE)),
            "ties": int(np.sum(np.abs(diff) <= PAIR_TOLERANCE)),
            "losses": int(np.sum(diff < -PAIR_TOLERANCE)),
        }
    )
    ensure_finite(result, "independent_paired_summary")
    return result


def _same_optional_number(reported: Any, expected: Any) -> bool:
    if expected is None:
        return reported is None
    return reported is not None and close(reported, expected, atol=1e-12)


def verify_reported_vector(
    audit: Audit,
    reported: Mapping[str, Any],
    vector: np.ndarray,
    label: str,
    rng: np.random.Generator,
    paired: bool = False,
) -> None:
    expected = (
        independent_paired_summary(vector, rng)
        if paired
        else independent_vector_summary(vector, rng)
    )
    audit.require(
        np.allclose(
            np.asarray(reported.get("raw_catalog_means"), dtype=float),
            vector,
            rtol=0.0,
            atol=1e-15,
        ),
        f"Raw catalog-resample vector mismatch: {label}",
    )
    audit.require(
        close(reported.get("mean"), expected["mean"])
        and close(reported.get("sample_sd"), expected["sample_sd"]),
        f"Mean/SD mismatch: {label}",
    )
    audit.require(
        close(
            reported.get("bootstrap_ci95_lo"), expected["bootstrap_ci95_lo"]
        )
        and close(
            reported.get("bootstrap_ci95_hi"), expected["bootstrap_ci95_hi"]
        )
        and int(reported.get("n_catalogs", -1)) == EXPECTED_CATALOGS,
        f"Bootstrap interval/n mismatch: {label}",
    )
    if paired:
        audit.require(
            reported.get("paired_t_defined") is expected["paired_t_defined"]
            and _same_optional_number(
                reported.get("paired_t_stat"), expected["paired_t_stat"]
            )
            and _same_optional_number(
                reported.get("paired_t_p_two_sided"),
                expected["paired_t_p_two_sided"],
            ),
            f"Paired t-test mismatch: {label}",
        )
        audit.require(
            reported.get("cohen_dz_defined") is expected["cohen_dz_defined"]
            and _same_optional_number(
                reported.get("cohen_dz"), expected["cohen_dz"]
            ),
            f"Cohen dz mismatch: {label}",
        )
        audit.require(
            reported.get("wilcoxon_defined") is expected["wilcoxon_defined"]
            and _same_optional_number(
                reported.get("wilcoxon_stat"), expected["wilcoxon_stat"]
            )
            and _same_optional_number(
                reported.get("wilcoxon_p_two_sided"),
                expected["wilcoxon_p_two_sided"],
            ),
            f"Wilcoxon mismatch: {label}",
        )
        audit.require(
            int(reported.get("wins", -1)) == expected["wins"]
            and int(reported.get("ties", -1)) == expected["ties"]
            and int(reported.get("losses", -1)) == expected["losses"],
            f"Paired win/tie/loss mismatch: {label}",
        )


def verify_campaign(
    output_dir: Path,
    mode: str = "auto",
    progress: VerificationProgress | None = None,
) -> dict[str, Any]:
    audit = Audit()
    provenance_evidence = verify_producer_and_replay_provenance(audit)
    for relative, expected in EXPECTED_HASHES.items():
        path = safe_path(CAMPAIGN_ROOT, relative)
        audit.require(path.is_file() and sha256_file(path) == expected, f"Frozen input hash mismatch: {relative}")

    input_manifest = strict_load(MANIFEST_PATH)
    original = strict_load(ORIGINAL_RESULT_PATH)
    ensure_finite(input_manifest, "input_manifest")
    ensure_finite(original, "original_result")
    audit.require(isinstance(input_manifest.get("runs"), list) and len(input_manifest["runs"]) == EXPECTED_CATALOGS, "Frozen manifest must contain 50 catalogs")
    audit.require(isinstance(original.get("artifacts"), list) and len(original["artifacts"]) == EXPECTED_CATALOGS, "Original result must contain 50 catalogs")
    run_manifest, run_manifest_sha = verify_run_manifest(
        CAMPAIGN_ROOT, required_manifest_paths(input_manifest)
    )
    audit.require(
        run_manifest.get("campaign_id") == CAMPAIGN_ID,
        "RUN_MANIFEST campaign ID mismatch",
    )
    verify_runner_source_contract(RUNNER_PATH)
    audit.gate("G01_source_and_run_manifest", {"run_manifest_sha256": run_manifest_sha, "files": len(required_manifest_paths(input_manifest))})

    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(CAMPAIGN_ROOT.resolve())
    except ValueError as exc:
        raise VerificationError("Output directory must remain inside the campaign") from exc
    checkpoint_path = output_dir / "main_catalogs.jsonl"
    terminal_path = output_dir / "main_results.json"
    status_path = output_dir / "status.json"
    audit.require(checkpoint_path.is_file() and terminal_path.is_file() and status_path.is_file(), "Terminal/checkpoint/status artifact missing")
    checkpoint_sha = sha256_file(checkpoint_path)
    terminal_sha = sha256_file(terminal_path)
    status_sha = sha256_file(status_path)
    terminal = strict_load(terminal_path)
    status = strict_load(status_path)
    records = strict_jsonl(checkpoint_path)
    ensure_finite(terminal, "terminal")
    ensure_finite(status, "status")
    ensure_finite(records, "checkpoint")
    audit.require(terminal.get("status") == "completed_unverified", "Terminal is not completed_unverified")
    terminal_mode = str(terminal.get("mode"))
    if mode != "auto":
        audit.require(terminal_mode == mode, f"Requested mode {mode} != terminal mode {terminal_mode}")
    audit.require(terminal_mode in {"smoke", "canonical"}, "Unknown terminal mode")
    config = terminal.get("config")
    audit.require(isinstance(config, dict), "Terminal config missing")
    runs = int(config.get("runs", -1))
    audit.require((terminal_mode == "canonical" and runs == EXPECTED_CATALOGS) or (terminal_mode == "smoke" and 1 <= runs <= EXPECTED_CATALOGS), "Mode/run-count contract mismatch")
    audit.require(config.get("profiles") == list(PROFILE_ORDER), "Profile order mismatch")
    audit.require(int(config.get("episodes", -1)) == EPISODES and config.get("checkpoints") == list(CHECKPOINTS), "Episode/checkpoint config mismatch")
    audit.require(int(config.get("arm_count", -1)) == 20 and int(config.get("mandatory_factorial_arm_count", -1)) == 18, "Arm-count config mismatch")
    audit.require(config.get("exact_arm_id") == EXACT_ARM_ID and config.get("primary_corrected_arm_id") == PRIMARY_CORRECTED_ARM_ID, "Anchor arm mismatch")
    audit.require(terminal.get("campaign_id") == CAMPAIGN_ID and status.get("campaign_id") == CAMPAIGN_ID, "Campaign ID mismatch")
    terminal_environment = terminal.get("environment", {})
    audit.require(
        all(terminal_environment.get(name) == version for name, version in EXPECTED_ENVIRONMENT.items()),
        f"Terminal producer-environment versions mismatch: {terminal_environment}",
    )
    audit.require(terminal.get("checkpoint_sha256") == checkpoint_sha, "Terminal/checkpoint hash mismatch")
    audit.require(status.get("terminal_sha256") == terminal_sha, "Status/terminal hash mismatch")
    audit.require(status.get("status") == "completed_unverified" and int(status.get("runs_completed", -1)) == runs, "Terminal status is incomplete")
    terminal_rel = str(status.get("terminal_path", "")).replace("\\", "/")
    audit.require(safe_path(CAMPAIGN_ROOT, terminal_rel) == terminal_path.resolve(), "Status terminal path mismatch")
    terminal_hashes = terminal.get("input_hashes", {})
    audit.require(terminal_hashes.get("run_manifest_sha256") == run_manifest_sha, "Terminal run-manifest hash mismatch")
    audit.require(terminal_hashes.get("frozen_core_sha256") == EXPECTED_HASHES["src/original_hybrid_core.py"], "Terminal frozen-core hash mismatch")
    audit.require(terminal_hashes.get("manifest_sha256") == EXPECTED_HASHES["inputs/data/processed/manifest.json"], "Terminal manifest hash mismatch")
    audit.require(terminal_hashes.get("original_result_sha256") == EXPECTED_HASHES["inputs/results/amazon_primary.json"], "Terminal original-result hash mismatch")
    historical = terminal_hashes.get("historical_reward_provenance_sha256")
    audit.require(
        isinstance(historical, dict)
        and historical.get("hybrid_rl_mcdm_v2.py", "").lower()
        == EXPECTED_HASHES["inputs/historical_reward_provenance/hybrid_rl_mcdm_v2.py"]
        and historical.get("supplementary_runs.py", "").lower()
        == EXPECTED_HASHES["inputs/historical_reward_provenance/supplementary_runs.py"],
        "Terminal historical-provenance hash mismatch",
    )
    terminal_producer = terminal_hashes.get("producer_provenance_sha256")
    expected_terminal_producer = {
        "code/hybrid_core.py": EXPECTED_HASHES[
            "inputs/producer_provenance/v1.0-submission/code/hybrid_core.py"
        ],
        "code/run_amazon_experiments.py": EXPECTED_HASHES[
            "inputs/producer_provenance/v1.0-submission/code/run_amazon_experiments.py"
        ],
        "requirements.txt": EXPECTED_HASHES[
            "inputs/producer_provenance/v1.0-submission/requirements.txt"
        ],
        "COMMIT_METADATA.json": EXPECTED_HASHES[
            "inputs/producer_provenance/v1.0-submission/COMMIT_METADATA.json"
        ],
        "PUBLIC_TAG_EXACT_HASHES.json": EXPECTED_HASHES[
            "inputs/producer_provenance/v1.0-submission/PUBLIC_TAG_EXACT_HASHES.json"
        ],
    }
    audit.require(
        terminal_producer == expected_terminal_producer,
        "Terminal producer-provenance hashes mismatch",
    )
    exact_replay_terminal = terminal_hashes.get("exact_replay_provenance")
    audit.require(
        exact_replay_terminal
        == {
            "sha256": EXPECTED_HASHES[
                "inputs/exact_replay_provenance/EXACT_REPLAY_AUDIT.json"
            ],
            "cells_total": 250,
            "cells_exact": 250,
            "all_exact": True,
        },
        "Terminal exact-replay provenance mismatch",
    )
    audit.require(status.get("run_manifest_sha256") == run_manifest_sha, "Status run-manifest hash mismatch")
    audit.require(len(records) == runs, "Checkpoint record count mismatch")
    audit.gate("G02_terminal_hash_chain", {"checkpoint_sha256": checkpoint_sha, "terminal_sha256": terminal_sha, "status_sha256": status_sha})
    audit.gate("G02b_producer_environment_and_replay", provenance_evidence)

    manifest_runs = {int(item["run_index"]): item for item in input_manifest["runs"]}
    audit.require(set(manifest_runs) == set(range(EXPECTED_CATALOGS)), "Frozen manifest run indices are not 0..49")
    original_map = original_profile_map(original)
    record_map: dict[int, Mapping[str, Any]] = {}
    signatures: dict[tuple[int, str, str], str] = {}
    original_sha = EXPECTED_HASHES["inputs/results/amazon_primary.json"]

    for record_number, record in enumerate(records, 1):
        run_index = int(record.get("run_index", -1))
        audit.require(run_index not in record_map and 0 <= run_index < runs, f"Duplicate/out-of-scope checkpoint run: {run_index}")
        record_map[run_index] = record
        meta = manifest_runs[run_index]
        catalog_path = safe_path(INPUT_ROOT, str(meta["path"]))
        audit.require(catalog_path.is_file() and sha256_file(catalog_path) == str(meta["sha256"]), f"Catalog hash mismatch: run={run_index}")
        audit.require(int(record.get("run_seed", -1)) == int(meta["seed"]), f"Run seed mismatch: run={run_index}")
        audit.require(str(record.get("dataset_path", "")).replace("\\", "/") == str(meta["path"]).replace("\\", "/"), f"Dataset path mismatch: run={run_index}")
        audit.require(record.get("dataset_sha256") == meta["sha256"] and record.get("target_source_sha256") == original_sha, f"Dataset/target source mismatch: run={run_index}")
        audit.require(record.get("run_manifest_sha256") == run_manifest_sha, f"Record run-manifest hash mismatch: run={run_index}")
        stored_catalog = original["artifacts"][run_index]
        audit.require(int(stored_catalog["run_index"]) == run_index and stored_catalog["dataset_sha256"] == meta["sha256"], f"Original artifact alignment mismatch: run={run_index}")
        data = read_catalog(catalog_path)
        audit.require(int(data["n"]) == 400, f"Catalog size is not 400: run={run_index}")
        topsis_scores, topsis_weights = independent_topsis(data)
        topsis_rank = top_rank(topsis_scores)
        arms = record.get("arms")
        audit.require(isinstance(arms, dict) and set(arms) == set(ARM_MAP), f"20-arm completeness mismatch: run={run_index}")
        for aid, expected_arm in ARM_MAP.items():
            cell = arms[aid]
            audit.require(isinstance(cell, dict) and cell.get("arm") == expected_arm, f"Arm metadata mismatch: run={run_index} arm={aid}")
            profiles = cell.get("profiles")
            audit.require(isinstance(profiles, list) and [item.get("profile_name") for item in profiles] == list(PROFILE_ORDER), f"Five-profile completeness/order mismatch: run={run_index} arm={aid}")
            for payload in profiles:
                profile_name = str(payload["profile_name"])
                stored = original_map[(run_index, profile_name)]
                final = stored["final"]
                audit.require(top_rank(np.asarray(final["topsis_scores"], dtype=float)) == topsis_rank, f"Stored TOPSIS score/rank mismatch: run={run_index} profile={profile_name}")
                # Manual CSV parsing and independently ordered float reductions can
                # differ from pandas by one binary64 ULP; the rank gate remains exact.
                audit.require(np.allclose(np.asarray(final["topsis_scores"], dtype=float), topsis_scores, rtol=0.0, atol=1e-15), f"Independent TOPSIS scores mismatch: run={run_index} profile={profile_name}")
                audit.require(np.allclose(np.asarray(final["topsis_weights"], dtype=float), topsis_weights, rtol=0.0, atol=1e-15), f"Independent TOPSIS weights mismatch: run={run_index} profile={profile_name}")
                signatures[(run_index, profile_name, aid)] = verify_profile(
                    audit, payload, expected_arm, data, profile_name, int(meta["seed"]),
                    topsis_scores, topsis_rank, stored, aid == EXACT_ARM_ID,
                )
        if progress is not None:
            progress.update(
                catalogs_completed=record_number,
                cells_completed=record_number * len(PROFILE_ORDER) * len(ARMS),
            )

    audit.require(set(record_map) == set(range(runs)), "Checkpoint run indices are not complete")
    verify_factor_effects(signatures, list(range(runs)))
    audit.gate("G03_20_arm_50x5_raw_verification", {
        "catalogs": runs, "profiles_per_catalog": 5, "arms": 20,
        "verified_cells": runs * 5 * 20, "canonical_cells_expected": 5000 if terminal_mode == "canonical" else None,
    })
    audit.gate("G04_exact_r0_replay", {"catalog_profile_cells": runs * 5, "checkpoints_per_cell": 7, "q_and_visits": "exact"})
    audit.gate(
        "G04b_full_corrected_stochastic_replay",
        {
            "catalog_profile_arm_cells": runs * 5 * 20,
            "episodes_per_cell": EPISODES,
            "q": "exact",
            "visits": "exact",
            "checkpoint_f1": "exact",
            "final_rankings": "exact",
            "reward_diagnostics": "exact",
            "runner_imported": False,
        },
    )

    analysis = terminal.get("analysis")
    audit.require(isinstance(analysis, dict), "Terminal analysis missing")
    if terminal_mode == "smoke":
        audit.require(analysis.get("status") == "exact_replay_and_all_arm_execution_pass" and int(analysis.get("arm_count", -1)) == 20, "Smoke analysis marker mismatch")
    else:
        audit.require(analysis.get("schema_version") == "same_target_bridge.main.v1", "Canonical analysis schema mismatch")
        audit.require(int(analysis.get("arm_count", -1)) == 20 and int(analysis.get("mandatory_factorial_arm_count", -1)) == 18, "Canonical analysis arm count mismatch")
        audit.require(
            int(analysis.get("analysis_seed", -1)) == ANALYSIS_SEED
            and int(analysis.get("bootstrap_reps", -1)) == BOOTSTRAP_REPS,
            "Canonical analysis seed/bootstrap lock mismatch",
        )
        audit.require(
            analysis.get("undefined_test_policy")
            == {
                "paired_t": "null with paired_t_defined=false when paired differences are constant",
                "cohen_dz": "null with cohen_dz_defined=false when paired differences are constant",
                "wilcoxon": "null with wilcoxon_defined=false when every paired difference is exactly zero",
                "win_tie_loss_tolerance": PAIR_TOLERANCE,
                "json_nonfinite": "fail before write; JSON serialization uses allow_nan=false",
            },
            "Undefined-statistic/JSON-finite policy mismatch",
        )
        analysis_rng = np.random.default_rng(ANALYSIS_SEED)
        summaries = analysis.get("summaries")
        audit.require(isinstance(summaries, dict) and set(summaries) == set(ARM_MAP), "Canonical analysis summaries incomplete")
        # Verify F1 and NDCG summaries from raw rankings/metrics.
        vectors_by_arm: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for aid in ARM_MAP:
            vectors_by_arm[aid] = {}
            for metric in ("f1_at_7", "ndcg_at_7"):
                vectors: dict[str, np.ndarray] = {}
                section = summaries[aid] if metric == "f1_at_7" else summaries[aid].get("ndcg_at_7")
                audit.require(isinstance(section, dict), f"Missing {metric} summary: {aid}")
                for method in METHODS:
                    raw = []
                    for run_index in range(EXPECTED_CATALOGS):
                        profiles = record_map[run_index]["arms"][aid]["profiles"]
                        raw.append(
                            float(
                                np.mean(
                                    [
                                        profile["final_metrics"][method][metric]
                                        for profile in profiles
                                    ]
                                )
                            )
                        )
                    vectors[method] = np.asarray(raw, dtype=float)
                    verify_reported_vector(
                        audit,
                        section["methods"][method],
                        vectors[method],
                        f"{aid}/{metric}/{method}",
                        analysis_rng,
                    )
                for name, baseline in (
                    ("hybrid_minus_rl", "rl_only"),
                    ("hybrid_minus_topsis", "topsis_only"),
                ):
                    diff = vectors["hybrid"] - vectors[baseline]
                    verify_reported_vector(
                        audit,
                        section[name],
                        diff,
                        f"{aid}/{metric}/{name}",
                        analysis_rng,
                        paired=True,
                    )
                vectors_by_arm[aid][metric] = vectors

        factorial = analysis.get("factorial_contrasts")
        audit.require(
            isinstance(factorial, dict)
            and set(factorial) == {"f1_at_7", "ndcg_at_7"},
            "Factorial contrast metrics missing",
        )
        expected_specs = {
            spec["contrast_id"]: spec for spec in expected_factorial_contrast_specs()
        }
        for metric in ("f1_at_7", "ndcg_at_7"):
            audit.require(
                set(factorial[metric]) == set(expected_specs),
                f"Factorial contrast set mismatch: {metric}",
            )
            for contrast_id, spec in expected_specs.items():
                reported = factorial[metric][contrast_id]
                audit.require(
                    reported.get("specification") == spec,
                    f"Factorial specification mismatch: {contrast_id}",
                )
                for method in ("hybrid", "rl_only"):
                    diff = (
                        vectors_by_arm[spec["arm_a"]][metric][method]
                        - vectors_by_arm[spec["arm_b"]][metric][method]
                    )
                    verify_reported_vector(
                        audit,
                        reported[method],
                        diff,
                        f"factorial/{metric}/{contrast_id}/{method}",
                        analysis_rng,
                        paired=True,
                    )

        sensitivity = analysis.get("sensitivity_contrasts")
        audit.require(
            isinstance(sensitivity, dict)
            and set(sensitivity) == {"f1_at_7", "ndcg_at_7"},
            "Sensitivity contrast metrics missing",
        )
        expected_sensitivity = {
            spec["contrast_id"]: spec for spec in expected_sensitivity_contrast_specs()
        }
        for metric in ("f1_at_7", "ndcg_at_7"):
            audit.require(
                set(sensitivity[metric]) == set(expected_sensitivity),
                f"Sensitivity contrast set mismatch: {metric}",
            )
            for contrast_id, spec in expected_sensitivity.items():
                reported = sensitivity[metric][contrast_id]
                audit.require(
                    reported.get("specification") == spec,
                    f"Sensitivity specification mismatch: {contrast_id}",
                )
                for method in ("hybrid", "rl_only"):
                    diff = (
                        vectors_by_arm[spec["arm_a"]][metric][method]
                        - vectors_by_arm[spec["arm_b"]][metric][method]
                    )
                    verify_reported_vector(
                        audit,
                        reported[method],
                        diff,
                        f"sensitivity/{metric}/{contrast_id}/{method}",
                        analysis_rng,
                        paired=True,
                    )
    audit.gate(
        "G05_analysis_from_raw_rankings",
        {
            "mode": terminal_mode,
            "finite": True,
            "analysis_seed": ANALYSIS_SEED if terminal_mode == "canonical" else None,
            "bootstrap_reps": BOOTSTRAP_REPS if terminal_mode == "canonical" else None,
            "reported_statistics_recomputed": terminal_mode == "canonical",
        },
    )

    return {
        "schema_version": "same_target_bridge.full_verification.v1",
        "campaign_id": CAMPAIGN_ID,
        "mode": terminal_mode,
        "verdict": "PASS",
        "status": "completed_verified",
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "Codex independent verifier",
        "verifier_sha256": sha256_file(Path(__file__)),
        "run_manifest_sha256": run_manifest_sha,
        "input_hashes": dict(EXPECTED_HASHES),
        "producer_provenance": provenance_evidence,
        "output_hashes": {
            "main_catalogs.jsonl": checkpoint_sha,
            "main_results.json": terminal_sha,
            "status.json": status_sha,
        },
        "counts": {
            "catalogs": runs,
            "profiles_per_catalog": 5,
            "arms": 20,
            "catalog_profile_arm_cells": runs * 5 * 20,
            "full_stochastic_replay_cells": runs * 5 * 20,
            "full_stochastic_replay_episodes": runs * 5 * 20 * EPISODES,
            "exact_replay_checkpoint_values": runs * 5 * 7 * 5,
        },
        "checks_executed": audit.checks,
        "gates": audit.gates,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_finite(payload, "atomic_json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/canonical_main", help="Campaign-relative terminal output directory")
    parser.add_argument("--mode", choices=("auto", "smoke", "canonical"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = safe_path(CAMPAIGN_ROOT, args.output_dir)
    report_path = output_dir / "FULL_VERIFICATION.json"
    progress: VerificationProgress | None = None
    created_report = False
    try:
        assert_verification_start_clean(output_dir)
        terminal_preview = strict_load(output_dir / "main_results.json")
        if not isinstance(terminal_preview, dict):
            raise VerificationError("Terminal preview is not a JSON object")
        terminal_mode = str(terminal_preview.get("mode"))
        terminal_config = terminal_preview.get("config")
        if terminal_mode not in {"smoke", "canonical"} or not isinstance(
            terminal_config, dict
        ):
            raise VerificationError("Terminal preview mode/config is invalid")
        runs = int(terminal_config.get("runs", -1))
        if runs < 1 or runs > EXPECTED_CATALOGS:
            raise VerificationError("Terminal preview run count is invalid")
        progress = VerificationProgress(
            output_dir=output_dir,
            mode=terminal_mode,
            catalogs_total=runs,
            cells_total=runs * len(PROFILE_ORDER) * len(ARMS),
        )
        progress.start()
        report = verify_campaign(output_dir, args.mode, progress=progress)
        atomic_write_json(report_path, report)
        created_report = True
        progress.complete(report_path)
    except Exception as exc:  # fail closed: no report is left behind
        if created_report and report_path.exists():
            report_path.unlink()
        if progress is not None:
            try:
                progress.fail(exc)
            except Exception as status_exc:
                print(
                    "BRIDGE_VERIFICATION_STATUS_FAIL "
                    f"{type(status_exc).__name__}: {status_exc}",
                    file=sys.stderr,
                    flush=True,
                )
        print(f"BRIDGE_VERIFICATION_FAIL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"BRIDGE_VERIFICATION_PASS status=completed_verified report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
