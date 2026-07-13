"""Independent full-stochastic verifier for the same-target drift bridge."""

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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy


CAMPAIGN_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = CAMPAIGN_ROOT / "inputs"
CORE_PATH = CAMPAIGN_ROOT / "src" / "original_hybrid_core.py"
RUNNER_PATH = CAMPAIGN_ROOT / "src" / "drift_main.py"
MANIFEST_PATH = INPUT_ROOT / "data" / "processed" / "manifest.json"
RUN_MANIFEST_PATH = CAMPAIGN_ROOT / "RUN_MANIFEST.json"
EXPECTED_ENVIRONMENT = {"numpy": "1.26.0", "pandas": "2.2.3", "scipy": "1.16.3"}
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
REWARD_MODELS = ("inclusive_range_fix", "component_continuous_fix", "historical_funnel_coefficients_on_may_h")
METHODS = ("rl_only", "hybrid")
SUDDEN_BOUNDARY = 15000
DRIFT_START = 10000
DRIFT_END = 25000
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
    "primary_corrected_reward": "component_continuous_fix",
    "targets": "evaluation_only_in_corrected_arms",
    "verifier": "independent_full_stochastic_replay",
}
LOCK_REQUIRED_STATIC_PATHS = frozenset(
    {
        "PROTOCOL_LOCK.md", "src/drift_main.py", "src/lock_campaign.py", "src/original_hybrid_core.py",
        "verify_drift.py", "tests/test_drift_contract.py", "tests/test_verify_drift.py",
        "inputs/design/SAME_TARGET_BRIDGE_PROTOCOL_2026-07-12.md", "inputs/data/processed/manifest.json",
        "inputs/results/amazon_primary.json", "inputs/results/amazon_drift.json", "inputs/results/validation_extensions.json",
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
RUNNER_TERMINAL_OUTPUT_SET = frozenset({"sealed_records.jsonl", "STATUS.json", "PROGRESS.json", "TERMINAL.json"})
VERIFIER_STALE_SET = frozenset({"FULL_VERIFICATION.json", "VERIFICATION_FAILURE.json", "VERIFICATION_STATUS.json", "VERIFICATION_PROGRESS.json"})


class VerificationError(RuntimeError):
    pass


class Audit:
    def __init__(self) -> None:
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise VerificationError(message)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_module("independent_drift_core", CORE_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii")); digest.update(str(arr.shape).encode("ascii")); digest.update(arr.tobytes())
    return digest.hexdigest()


def state_digest(q: np.ndarray, visits: np.ndarray) -> str:
    return hashlib.sha256((array_digest(q) + array_digest(visits)).encode("ascii")).hexdigest()


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise VerificationError(f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def reject_constant(value: str) -> None:
    raise VerificationError(f"Non-finite JSON constant: {value}")


def strict_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object, parse_constant=reject_constant)


def strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line, object_pairs_hook=strict_object, parse_constant=reject_constant)
        except Exception as exc:
            raise VerificationError(f"Invalid sealed JSONL line {number}") from exc
        if not isinstance(row, dict):
            raise VerificationError(f"Non-object sealed JSONL line {number}")
        rows.append(row)
    return rows


def ensure_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise VerificationError(f"Nonfinite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items(): ensure_finite(item, f"{path}.{key}")
    if isinstance(value, list):
        for idx, item in enumerate(value): ensure_finite(item, f"{path}[{idx}]")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def assert_verifier_start_clean(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError("Verifier requires an existing terminal output directory")
    names = {item.name for item in output_dir.iterdir() if item.is_file()}
    subdirs = [item.name for item in output_dir.iterdir() if not item.is_file()]
    if subdirs:
        raise FileExistsError(f"Verifier output contains unexpected subdirectories: {subdirs}")
    stale = sorted(names & set(VERIFIER_STALE_SET))
    if stale:
        raise FileExistsError(f"Verifier refuses stale PASS/FAIL/status artifacts: {stale}")
    if names != set(RUNNER_TERMINAL_OUTPUT_SET):
        missing = sorted(set(RUNNER_TERMINAL_OUTPUT_SET) - names)
        unexpected = sorted(names - set(RUNNER_TERMINAL_OUTPUT_SET))
        raise FileExistsError(f"Verifier terminal artifact set mismatch: missing={missing} unexpected={unexpected}")


class VerificationProgress:
    def __init__(self, output_dir: Path, total: int) -> None:
        self.output_dir = output_dir
        self.total = int(total)
        self.completed = 0
        self.started = time.time()
        self.status_path = output_dir / "VERIFICATION_STATUS.json"
        self.progress_path = output_dir / "VERIFICATION_PROGRESS.json"
        self._write("running")

    def _payload(self, status: str) -> dict[str, Any]:
        elapsed = time.time() - self.started
        eta = elapsed / self.completed * (self.total - self.completed) if self.completed else None
        return {
            "schema_version": "same_target_drift.verification_progress.v1",
            "campaign_id": CAMPAIGN_ROOT.name,
            "status": status,
            "completed_profile_arm_replays": self.completed,
            "total_profile_arm_replays": self.total,
            "percent": 100.0 * self.completed / self.total if self.total else 100.0,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "scientific_metrics_exposed": False,
            "console_contract": "progress_only; run with python -u; redirect logs outside outputs",
        }

    def _write(self, status: str) -> None:
        payload = self._payload(status)
        atomic_json(self.progress_path, payload)
        atomic_json(self.status_path, payload)

    def step(self) -> None:
        self.completed += 1
        if self.completed > self.total:
            raise VerificationError("Verifier progress exceeded locked replay total")
        self._write("running")
        if self.completed == 1 or self.completed == self.total or self.completed % 10 == 0:
            payload = self._payload("running")
            eta = payload["eta_seconds"]
            eta_text = "unknown" if eta is None else f"{eta:.0f}"
            print(
                f"VERIFY_PROGRESS {self.completed}/{self.total} "
                f"percent={payload['percent']:.1f} eta_seconds={eta_text}",
                flush=True,
            )

    def finish(self, status: str, extra: Mapping[str, Any] | None = None) -> None:
        payload = self._payload(status)
        if extra:
            payload.update(extra)
        atomic_json(self.progress_path, payload)
        atomic_json(self.status_path, payload)


def verify_environment(audit: Audit) -> None:
    actual = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
    audit.require(actual == EXPECTED_ENVIRONMENT, f"Verifier environment mismatch: {actual}")


def verify_frozen_and_manifest(audit: Audit) -> dict[str, Any]:
    for relative, expected in EXPECTED_HASHES.items():
        path = CAMPAIGN_ROOT / relative
        audit.require(path.is_file(), f"Missing frozen file: {relative}")
        audit.require(sha256_file(path) == expected, f"Frozen hash mismatch: {relative}")
    run_manifest = strict_load(RUN_MANIFEST_PATH)
    audit.require(set(run_manifest)=={"schema_version","campaign_id","status","created_at","tool","model","operation_id","hash_algorithm","contract","environment","producer_provenance","files"},"Run-manifest top-level schema mismatch")
    audit.require(run_manifest.get("schema_version") == "same_target_drift.run_manifest.v1", "Run-manifest schema mismatch")
    audit.require(run_manifest.get("campaign_id") == CAMPAIGN_ROOT.name, "Run-manifest campaign mismatch")
    audit.require(run_manifest.get("status") == "LOCKED_BEFORE_COMPUTE", "Run-manifest not locked")
    audit.require(run_manifest.get("hash_algorithm")=="SHA-256" and run_manifest.get("tool")=="Codex" and run_manifest.get("model")=="GPT-5 Codex","Run-manifest writer/hash metadata mismatch")
    audit.require(run_manifest.get("contract") == RUN_MANIFEST_CONTRACT, "Run-manifest scientific contract mismatch")
    environment=run_manifest.get("environment",{})
    audit.require(set(environment)=={"python","platform","packages"} and environment.get("python")==sys.version and environment.get("platform")==platform.platform() and environment.get("packages")==EXPECTED_ENVIRONMENT,"Run-manifest environment mismatch")
    audit.require(run_manifest.get("producer_provenance")=={"tag":"v1.0-submission","commit_sha1":"3b92f6485d20d1a45ac03b60077d20af08060885"},"Run-manifest producer provenance mismatch")
    entries = run_manifest.get("files")
    audit.require(isinstance(entries, list) and len(entries) > 0, "Run-manifest file list incomplete")
    paths = [str(entry.get("path")) for entry in entries]
    audit.require(len(paths) == len(set(paths)), "Duplicate run-manifest file path")
    input_manifest = strict_load(MANIFEST_PATH)
    catalog_paths = {"inputs/" + str(entry["path"]).replace("\\", "/") for entry in input_manifest.get("runs", [])}
    expected_paths = set(LOCK_REQUIRED_STATIC_PATHS) | catalog_paths
    audit.require(len(catalog_paths) == 50, "Run-manifest catalog path derivation mismatch")
    audit.require(set(paths) == expected_paths, "Run-manifest exact required path set mismatch")
    root = CAMPAIGN_ROOT.resolve()
    for entry in entries:
        audit.require(set(entry) == {"path", "sha256", "bytes"}, f"Locked file-entry schema mismatch: {entry.get('path')}")
        path = (CAMPAIGN_ROOT / str(entry["path"])).resolve()
        audit.require(root == path or root in path.parents, "Run-manifest path escape")
        audit.require(path.is_file(), f"Locked file missing: {entry['path']}")
        audit.require(path.stat().st_size == int(entry["bytes"]), f"Locked size mismatch: {entry['path']}")
        audit.require(sha256_file(path) == str(entry["sha256"]).lower(), f"Locked hash mismatch: {entry['path']}")
    manifest = input_manifest
    audit.require(len(manifest.get("runs", [])) == 50, "Catalog manifest run count mismatch")
    for idx, entry in enumerate(manifest["runs"]):
        audit.require(entry.get("run_index") == idx, "Catalog ordering mismatch")
        path = INPUT_ROOT / Path(str(entry["path"]).replace("\\", "/"))
        audit.require(path.is_file() and sha256_file(path) == str(entry["sha256"]).lower(), f"Catalog hash mismatch: {idx}")
    replay = strict_load(INPUT_ROOT / "exact_replay_provenance" / "EXACT_REPLAY_AUDIT.json")
    audit.require(replay.get("all_exact") is True and replay.get("cells_exact") == 250 and replay.get("cells_mismatched") == 0, "250-cell producer gate mismatch")
    public = strict_load(INPUT_ROOT / "producer_provenance" / "v1.0-submission" / "PUBLIC_TAG_EXACT_HASHES.json")
    audit.require(public.get("public_ls_remote_commit_sha1") == "3b92f6485d20d1a45ac03b60077d20af08060885", "Public tag commit mismatch")
    return manifest


def verify_runner_source(audit: Audit) -> None:
    runner = load_module("drift_source_contract", RUNNER_PATH)
    for function in (runner.corrected_sudden_train, runner.corrected_gradual_train):
        signature = inspect.signature(function)
        audit.require(not any("target" in name.lower() or name.lower().startswith("gt") for name in signature.parameters), f"Target parameter in {function.__name__}")
        tree = ast.parse(inspect.getsource(function))
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        audit.require(not ({"build_ground_truth", "sudden_targets", "gradual_target"} & calls), f"Target construction in {function.__name__}")
    compact = "".join(RUNNER_PATH.read_text(encoding="utf-8").split())
    audit.require("ifepisode>SUDDEN_BOUNDARY" in compact, "Sudden boundary contract missing")
    audit.require("int(round(gradual_fraction(episode)*40))" in compact, "Gradual round(key*40) contract missing")
    audit.require("+7000+key" in compact, "Gradual target seed contract missing")
    audit.require("pool=np.arange(len(df),dtype=int)" in compact, "Corrected full-catalog pool missing")


def shifted_profile(profile: Mapping[str, object]) -> dict[str, Any]:
    shifted = CORE.flip_brand_preferences(profile)
    ranked = sorted(shifted["cat_affinity"].items(), key=lambda item: item[1])
    shifted["cat_affinity"] = {ranked[0][0]: ranked[2][1], ranked[1][0]: ranked[1][1], ranked[2][0]: ranked[0][1]}
    lo, hi = profile["price_range"]; width = hi - lo; center = (lo + hi) / 2.0
    new_center = center + 250.0 if center < 350.0 else center - 250.0
    shifted["price_range"] = (max(10.0, new_center-width/2.0), min(1000.0, new_center+width/2.0))
    shifted["recency_weight"] = float(np.clip(1.0-float(profile["recency_weight"]), 0.10, 0.90))
    return shifted


def gradual_fraction(episode: int) -> float:
    if episode <= DRIFT_START: return 0.0
    if episode >= DRIFT_END: return 1.0
    return (episode-DRIFT_START)/(DRIFT_END-DRIFT_START)


def interpolate(pre: Mapping[str, object], post: Mapping[str, object], frac: float) -> dict[str, Any]:
    frac = float(np.clip(frac, 0.0, 1.0)); brands = sorted(set(pre["brand_pref"])|set(post["brand_pref"])); cats = sorted(set(pre["cat_affinity"])|set(post["cat_affinity"])); plo, phi = pre["price_range"]; qlo, qhi = post["price_range"]
    return {
        "brand_pref": {k:(1-frac)*float(pre["brand_pref"].get(k,0.0))+frac*float(post["brand_pref"].get(k,0.0)) for k in brands},
        "cat_affinity": {k:(1-frac)*float(pre["cat_affinity"].get(k,0.0))+frac*float(post["cat_affinity"].get(k,0.0)) for k in cats},
        "price_range": ((1-frac)*plo+frac*qlo,(1-frac)*phi+frac*qhi),
        "recency_weight": (1-frac)*float(pre["recency_weight"])+frac*float(post["recency_weight"]),
    }


def legacy_probabilities(df: pd.DataFrame, profile: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    brand=np.asarray([profile["brand_pref"].get(x,0.10) for x in df["brand"]],dtype=float); lo,hi=profile["price_range"]; center=(lo+hi)/2; half=(hi-lo)/2+1; fit=np.clip(1-np.abs(df["price"].to_numpy(float)-center)/half,0,1); price=(fit>0.999999).astype(float); cat=np.ones(len(df)); rw=float(profile["recency_weight"]); rec=df["recency_pct"].to_numpy(float)*rw+(1-rw)*0.5
    return np.clip(.40*brand+.35*price+.15*cat+.10*rec,0,1), np.clip(.50*brand+.30*price+.20*cat,0,1)


def corrected_probabilities(df: pd.DataFrame, profile: Mapping[str, object], model: str) -> tuple[np.ndarray, np.ndarray]:
    brand=np.asarray([profile["brand_pref"].get(x,0.10) for x in df["brand"]],dtype=float); rw=float(profile["recency_weight"]); rec=df["recency_pct"].to_numpy(float)*rw+(1-rw)*.5
    if model == "inclusive_range_fix":
        lo,hi=profile["price_range"]; price=((df["price"].to_numpy(float)>=lo)&(df["price"].to_numpy(float)<=hi)).astype(float); scale=max(float(v) for v in profile["cat_affinity"].values()); cat=np.asarray([float(profile["cat_affinity"].get(x,0.0))/scale for x in df["category"]])
        return np.clip(.40*brand+.35*price+.15*cat+.10*rec,0,1),np.clip(.50*brand+.30*price+.20*cat,0,1)
    if model == "component_continuous_fix":
        comp=CORE.hidden_components(df,profile); price=np.asarray(comp["price_fit"]); cat=np.asarray(comp["cat_score"])
        return np.clip(.40*brand+.35*price+.15*cat+.10*rec,0,1),np.clip(.50*brand+.30*price+.20*cat,0,1)
    if model == "historical_funnel_coefficients_on_may_h":
        hidden=CORE.hidden_utility(df,profile); return np.clip(.70*hidden+.10,.05,.95),np.clip(.50*hidden,.02,.80)
    raise KeyError(model)


def state_payload(q: np.ndarray, visits: np.ndarray, topsis: np.ndarray) -> dict[str, Any]:
    rl=[int(x) for x in CORE.top_k_ranking(q)]; hybrid=[int(x) for x in CORE.top_k_ranking(CORE.static_hybrid_score(q,topsis,lambda_q=.50))]
    return {"q_sha256":array_digest(q),"visits_sha256":array_digest(visits),"state_sha256":state_digest(q,visits),"rl_rank":rl,"hybrid_rank":hybrid}


def replay_profile(df: pd.DataFrame, profile_name: str, profile_idx: int, seed: int, topsis: np.ndarray, scenario: str, model: str | None) -> dict[str, Any]:
    legacy = model is None
    checkpoints = SUDDEN_CHECKPOINTS if scenario == "sudden" else GRADUAL_CHECKPOINTS
    pre=CORE.PROFILE_HIDDEN[profile_name]; post=CORE.flip_brand_preferences(pre) if scenario=="sudden" else shifted_profile(pre)
    q=np.zeros(len(df),float); visits=np.zeros(len(df),np.int32); eps=.30
    if scenario=="sudden": reward_rng=np.random.RandomState(seed+profile_idx*997); act_rng=np.random.RandomState(seed+profile_idx*13)
    else: reward_rng=np.random.RandomState(seed+profile_idx*2221); act_rng=np.random.RandomState(seed+profile_idx*4441)
    if legacy:
        if scenario=="sudden":
            base=CORE.profile_seed(seed,profile_name); pre_gt=CORE.top_k_set(CORE.build_ground_truth(df,pre,base,observable_alpha=CORE.MAIN_ALPHA)); post_gt=CORE.top_k_set(CORE.build_ground_truth(df,post,base+5000,observable_alpha=CORE.MAIN_ALPHA))
        else:
            base=CORE.profile_seed(seed,profile_name); pre_gt=CORE.top_k_set(CORE.build_ground_truth(df,pre,base,observable_alpha=CORE.MAIN_ALPHA)); post_gt=CORE.top_k_set(CORE.build_ground_truth(df,post,base+5000,observable_alpha=CORE.MAIN_ALPHA))
        pool=CORE.build_candidate_pool(df,[pre,post],[pre_gt,post_gt])
    else: pool=np.arange(len(df),dtype=int)
    cache: dict[int,tuple[np.ndarray,np.ndarray,set[int],np.ndarray|None]]={}
    out: dict[str,Any]={}
    topsis_rank=[int(x) for x in CORE.top_k_ranking(topsis)]
    for episode in range(1,checkpoints[-1]+1):
        if scenario=="sudden":
            key=1 if episode>SUDDEN_BOUNDARY else 0; phase=post if key else pre
            if key not in cache:
                if legacy:
                    gt=post_gt if key else pre_gt; mask=np.zeros(len(df),bool); mask[list(gt)]=True; pe,pc=legacy_probabilities(df,phase)
                else: gt=set(); mask=None; pe,pc=corrected_probabilities(df,phase,str(model))
                cache[key]=(pe,pc,gt,mask)
        else:
            key=int(round(gradual_fraction(episode)*40)); phase=interpolate(pre,post,key/40)
            if key not in cache:
                gt_seed=CORE.profile_seed(seed,profile_name)+7000+key; gt=CORE.top_k_set(CORE.build_ground_truth(df,phase,gt_seed,observable_alpha=CORE.MAIN_ALPHA))
                if legacy: mask=np.zeros(len(df),bool); mask[list(gt)]=True; pe,pc=legacy_probabilities(df,phase)
                else: mask=None; pe,pc=corrected_probabilities(df,phase,str(model))
                cache[key]=(pe,pc,gt,mask)
        pe,pc,gt,mask=cache[key]
        action=int(act_rng.choice(pool)) if act_rng.random()<eps else int(pool[np.argmax(q[pool])])
        reward=-.02
        if reward_rng.random()<pe[action]:
            reward+=.30
            if reward_rng.random()<pc[action]: reward+=1.0
        if legacy and mask is not None and mask[action]: reward+=.20
        visits[action]+=1; q[action]+=.05*(reward-q[action]); eps=max(.05,eps*.9997)
        if episode in checkpoints:
            row=state_payload(q.copy(),visits.copy(),topsis)
            if scenario=="sudden":
                if not legacy:
                    base=CORE.profile_seed(seed,profile_name); eval_gt=CORE.top_k_set(CORE.build_ground_truth(df,phase,base+(5000 if key else 0),observable_alpha=CORE.MAIN_ALPHA)); row["target_phase"]="post" if key else "pre"; row["target_sha256"]=canonical_sha256({"set":sorted(eval_gt)})
                else: eval_gt=gt
            else:
                eval_gt=gt; row["target_key"]=key
                if not legacy: row["target_sha256"]=canonical_sha256({"set":sorted(eval_gt)})
            row["f1"]={"rl_only":float(CORE.f1_score(set(row["rl_rank"]),eval_gt)),"hybrid":float(CORE.f1_score(set(row["hybrid_rank"]),eval_gt))}
            if legacy and scenario=="gradual": row["topsis_rank"]=topsis_rank; row["f1"]={"topsis_only":float(CORE.f1_score(set(topsis_rank),eval_gt)),**row["f1"]}
            if not legacy: row["ndcg"]={"rl_only":float(CORE.ndcg_at_k(row["rl_rank"],eval_gt)),"hybrid":float(CORE.ndcg_at_k(row["hybrid_rank"],eval_gt))}
            out[str(episode)]=row
    return {"profile_name":profile_name,"checkpoints":out,"epsilon_final":float(eps)}


def aggregate(profiles: Sequence[Mapping[str,Any]], methods: Sequence[str], checkpoints: Sequence[int]) -> dict[str,dict[str,float]]:
    return {m:{str(cp):float(np.mean([p["checkpoints"][str(cp)]["f1"][m] for p in profiles])) for cp in checkpoints} for m in methods}


def compare_exact(audit: Audit, actual: Any, expected: Any, label: str) -> None:
    audit.require(type(actual) is type(expected), f"Type mismatch: {label}")
    if isinstance(expected, dict):
        audit.require(set(actual)==set(expected), f"Key mismatch: {label}")
        for key in expected: compare_exact(audit,actual[key],expected[key],f"{label}.{key}")
    elif isinstance(expected, list):
        audit.require(len(actual)==len(expected), f"Length mismatch: {label}")
        for idx,(x,y) in enumerate(zip(actual,expected)): compare_exact(audit,x,y,f"{label}[{idx}]")
    elif isinstance(expected, float): audit.require(actual==expected, f"Float mismatch: {label}")
    else: audit.require(actual==expected, f"Value mismatch: {label}")


def normalized_auc(values: Sequence[float],grid: Sequence[int])->float:
    return float(np.trapz(np.asarray(values,float),np.asarray(grid,float))/(grid[-1]-grid[0]))


def analysis_seed(label: str) -> int:
    digest=hashlib.sha256(f"same_target_drift_analysis_v1|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8],"big",signed=False)


def summary(values: Sequence[float],label: str)->dict[str,Any]:
    arr=np.asarray(values,float)
    if arr.ndim!=1 or len(arr)==0 or not np.all(np.isfinite(arr)): raise VerificationError(f"Invalid analysis vector: {label}")
    seed=analysis_seed(label); rng=np.random.default_rng(seed); samples=arr[rng.integers(0,len(arr),size=(BOOTSTRAP_REPS,len(arr)))].mean(axis=1)
    return {"analysis_label":label,"bootstrap_seed":seed,"bootstrap_reps":BOOTSTRAP_REPS,"mean":float(arr.mean()),"sample_sd":float(arr.std(ddof=1)) if len(arr)>1 else 0.0,"bootstrap_ci95":[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],"n_catalog_resamples":int(len(arr))}


def paired_summary(values: Sequence[float],label: str)->dict[str,Any]:
    arr=np.asarray(values,float); result=summary(arr,label)
    result.update({"direction":"hybrid_minus_rl","raw_catalog_resample_vector":[float(x) for x in arr],"wins":int(np.sum(arr>PAIR_TOLERANCE)),"ties":int(np.sum(np.abs(arr)<=PAIR_TOLERANCE)),"losses":int(np.sum(arr < -PAIR_TOLERANCE)),"tie_tolerance":PAIR_TOLERANCE})
    return result


def independent_analysis(records: Sequence[Mapping[str,Any]],sudden_runs:int,gradual_runs:int)->dict[str,Any]:
    report={"unit":"paired catalog-resample/Monte Carlo run; five profiles averaged within run","primary_reward_model":"component_continuous_fix","sudden":{},"gradual":{}}
    for scenario,runs,grid in (("sudden",sudden_runs,SUDDEN_AUC_GRID),("gradual",gradual_runs,GRADUAL_AUC_GRID)):
        for model in REWARD_MODELS:
            finals={m:[] for m in METHODS}; aucs={m:[] for m in METHODS}
            for record in records[:runs]:
                profiles=record[scenario]["corrected_future_blind"][model]
                for method in METHODS:
                    vals=[float(np.mean([p["checkpoints"][str(cp)]["f1"][method] for p in profiles])) for cp in grid]; finals[method].append(vals[-1]); aucs[method].append(normalized_auc(vals,grid))
            final_diff=np.asarray(finals["hybrid"],float)-np.asarray(finals["rl_only"],float); auc_diff=np.asarray(aucs["hybrid"],float)-np.asarray(aucs["rl_only"],float)
            report[scenario][model]={m:{"final_f1":summary(finals[m],f"{scenario}|{model}|{m}|final_f1"),"checkpoint_normalized_post_change_auc":summary(aucs[m],f"{scenario}|{model}|{m}|auc")} for m in METHODS}
            report[scenario][model]["paired_hybrid_minus_rl"]={"final_f1":paired_summary(final_diff,f"{scenario}|{model}|hybrid_minus_rl|final_f1"),"checkpoint_normalized_post_change_auc":paired_summary(auc_diff,f"{scenario}|{model}|hybrid_minus_rl|auc")}
    return report


def verify_campaign(output_dir: Path, progress_holder: list[VerificationProgress] | None = None) -> dict[str,Any]:
    audit=Audit(); verify_environment(audit); manifest=verify_frozen_and_manifest(audit); verify_runner_source(audit)
    terminal_path=output_dir/"TERMINAL.json"; status_path=output_dir/"STATUS.json"; runner_progress_path=output_dir/"PROGRESS.json"; records_path=output_dir/"sealed_records.jsonl"
    terminal=strict_load(terminal_path); status=strict_load(status_path); runner_progress=strict_load(runner_progress_path); records=strict_jsonl(records_path)
    ensure_finite(terminal); ensure_finite(status); ensure_finite(runner_progress); ensure_finite(records)
    audit.require(terminal.get("status")=="completed_unverified" and status.get("status")=="completed_unverified","Campaign is not completed_unverified")
    audit.require(terminal.get("campaign_id")==CAMPAIGN_ROOT.name and status.get("campaign_id")==CAMPAIGN_ROOT.name,"Campaign ID mismatch")
    audit.require(terminal.get("environment")==EXPECTED_ENVIRONMENT,"Terminal environment mismatch")
    audit.require(terminal.get("run_manifest_sha256")==sha256_file(RUN_MANIFEST_PATH),"Terminal run-manifest hash mismatch")
    audit.require(terminal.get("sealed_records_sha256")==sha256_file(records_path),"Sealed-record hash mismatch")
    audit.require(status.get("terminal_sha256")==sha256_file(terminal_path),"Runner STATUS does not bind terminal hash")
    audit.require(status.get("sealed_records_sha256")==sha256_file(records_path),"Runner STATUS does not bind sealed-record hash")
    audit.require(status.get("run_manifest_sha256")==sha256_file(RUN_MANIFEST_PATH),"Runner STATUS does not bind run-manifest hash")
    audit.require(runner_progress.get("sealed_records_sha256")==sha256_file(records_path),"Runner PROGRESS does not bind sealed-record hash")
    audit.require(runner_progress.get("run_manifest_sha256")==sha256_file(RUN_MANIFEST_PATH),"Runner PROGRESS does not bind run-manifest hash")
    audit.require(runner_progress.get("scientific_metrics_exposed") is False and status.get("scientific_metrics_exposed") is False,"Runner checkpoint blindness flag mismatch")
    mode=terminal.get("mode"); audit.require(mode in ("smoke","canonical"),"Unknown mode")
    sudden_runs=50 if mode=="canonical" else len([r for r in records if r.get("sudden") is not None]); gradual_runs=30 if mode=="canonical" else len([r for r in records if r.get("gradual") is not None])
    audit.require(len(records)==max(sudden_runs,gradual_runs),"Record count mismatch")
    if mode=="canonical": audit.require(len(records)==50,"Canonical must contain 50 catalog records")
    audit.require(status.get("completed_catalogs")==len(records) and runner_progress.get("completed_catalogs")==len(records),"Runner terminal completion count mismatch")
    total_replays=5*4*(sudden_runs+gradual_runs)
    tracker=VerificationProgress(output_dir,total_replays)
    if progress_holder is not None: progress_holder.append(tracker)
    sudden_stored=strict_load(INPUT_ROOT/"results"/"amazon_drift.json"); gradual_stored=strict_load(INPUT_ROOT/"results"/"validation_extensions.json")["gradual_multidim_drift"]["summary"]
    replayed_profiles=0; sudden_cells=0; gradual_cells=0
    for idx,record in enumerate(records):
        unsigned=dict(record); digest=unsigned.pop("payload_sha256",None); audit.require(digest==canonical_sha256(unsigned),f"Record digest mismatch: {idx}")
        entry=manifest["runs"][idx]; audit.require(record.get("run_index")==idx and record.get("run_seed")==entry["seed"],f"Record identity mismatch: {idx}"); audit.require(record.get("dataset_sha256")==str(entry["sha256"]).lower(),f"Record dataset mismatch: {idx}")
        path=INPUT_ROOT/Path(str(entry["path"]).replace("\\","/")); df=pd.read_csv(path); topsis=CORE.topsis_artifacts(df)["scores"]; seed=int(entry["seed"])
        for scenario,run_count,checkpoints,legacy_methods in (("sudden",sudden_runs,SUDDEN_CHECKPOINTS,METHODS),("gradual",gradual_runs,GRADUAL_CHECKPOINTS,("topsis_only","rl_only","hybrid"))):
            block=record.get(scenario)
            audit.require((block is not None)==(idx<run_count),f"Scenario coverage mismatch: {idx}/{scenario}")
            if block is None: continue
            expected_legacy=[]
            for profile_idx,profile_name in enumerate(CORE.PROFILE_ORDER):
                expected=replay_profile(df,profile_name,profile_idx,seed,topsis,scenario,None); compare_exact(audit,block["legacy_exact"][profile_idx],expected,f"{idx}.{scenario}.legacy.{profile_name}"); expected_legacy.append(expected); replayed_profiles+=1; tracker.step()
            agg=aggregate(expected_legacy,legacy_methods,checkpoints)
            for method in legacy_methods:
                for cp in checkpoints:
                    stored=float(sudden_stored["summary"][method][str(cp)]["raw"][idx]) if scenario=="sudden" else float(gradual_stored[method][str(cp)]["raw"][idx]); audit.require(agg[method][str(cp)]==stored,f"Stored raw mismatch: {idx}/{scenario}/{method}/{cp}")
                    if scenario=="sudden": sudden_cells+=1
                    else: gradual_cells+=1
            audit.require(set(block["corrected_future_blind"])==set(REWARD_MODELS),f"Corrected arms mismatch: {idx}/{scenario}")
            for model in REWARD_MODELS:
                for profile_idx,profile_name in enumerate(CORE.PROFILE_ORDER):
                    expected=replay_profile(df,profile_name,profile_idx,seed,topsis,scenario,model); compare_exact(audit,block["corrected_future_blind"][model][profile_idx],expected,f"{idx}.{scenario}.{model}.{profile_name}"); replayed_profiles+=1; tracker.step()
    audit.require(sudden_cells==sudden_runs*18,"Sudden raw-cell count mismatch"); audit.require(gradual_cells==gradual_runs*18,"Gradual raw-cell count mismatch")
    audit.require(terminal.get("exact_legacy_raw_cells")=={"sudden":sudden_cells,"gradual":gradual_cells},"Terminal raw-cell count mismatch")
    compare_exact(audit,terminal.get("analysis"),independent_analysis(records,sudden_runs,gradual_runs),"terminal.analysis")
    source_contract=terminal.get("future_blind_source_contract",{}); audit.require(set(source_contract)=={"corrected_sudden_train","corrected_gradual_train"},"Future-blind source proof mismatch")
    preflight=terminal.get("prefix_invariance_metamorphic_preflight",{}); audit.require(set(preflight)==set(REWARD_MODELS),"Metamorphic reward coverage mismatch")
    for model,proof in preflight.items(): audit.require(all(proof.values()),f"Metamorphic proof failed: {model}")
    audit.require(replayed_profiles==total_replays and tracker.completed==total_replays,"Full stochastic replay total mismatch")
    return {
        "schema_version":"same_target_drift.full_verification.v1","campaign_id":CAMPAIGN_ROOT.name,"status":"PASS","mode":mode,
        "checks":audit.checks,"full_stochastic_profile_replays":replayed_profiles,"estimated_training_episodes_replayed":replayed_profiles*30000,
        "exact_legacy_raw_cells":{"sudden":sudden_cells,"gradual":gradual_cells},"environment":EXPECTED_ENVIRONMENT,
        "hash_chain":{
            "terminal_sha256":sha256_file(terminal_path),"runner_status_completed_unverified_sha256":sha256_file(status_path),
            "runner_progress_sha256":sha256_file(runner_progress_path),"sealed_records_sha256":sha256_file(records_path),
            "run_manifest_sha256":sha256_file(RUN_MANIFEST_PATH),"verifier_source_sha256":sha256_file(Path(__file__).resolve()),
            "frozen_input_sha256":dict(EXPECTED_HASHES),
            "run_manifest_locked_file_sha256":{
                str(entry["path"]):str(entry["sha256"]).lower()
                for entry in strict_load(RUN_MANIFEST_PATH)["files"]
            },
        },
        "independence_disclosure":(
            "The verifier independently reimplements drift loops, reward schedules, exact raw-cell gates, target evaluation, "
            "and analysis, but intentionally reuses the hash-locked public-tag core for primitive profile, TOPSIS, ground-truth, "
            "ranking, and metric semantics. This is deterministic cross-implementation replay, not a wholly independent software stack."
        ),
    }


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",required=True); return parser.parse_args()


def main()->int:
    args=parse_args(); output_dir=Path(args.output_dir).resolve()
    try:
        assert_verifier_start_clean(output_dir)
    except Exception as exc:
        # Never overwrite or coexist with an older PASS/FAIL/status artifact.
        print(f"DRIFT_VERIFIER_START_REJECT type={type(exc).__name__}",flush=True)
        return 1
    progress_holder: list[VerificationProgress]=[]
    try:
        report=verify_campaign(output_dir,progress_holder)
    except Exception as exc:
        failure={"schema_version":"same_target_drift.full_verification.v1","campaign_id":CAMPAIGN_ROOT.name,"status":"FAIL","error_type":type(exc).__name__,"error":str(exc),"full_verification_written":False}
        atomic_json(output_dir/"VERIFICATION_FAILURE.json",failure)
        extra={"verification_failure_sha256":sha256_file(output_dir/"VERIFICATION_FAILURE.json"),"full_verification_written":False}
        if progress_holder: progress_holder[0].finish("failed",extra)
        else:
            fallback={"schema_version":"same_target_drift.verification_progress.v1","campaign_id":CAMPAIGN_ROOT.name,"status":"failed","completed_profile_arm_replays":0,"total_profile_arm_replays":None,"scientific_metrics_exposed":False,**extra}
            atomic_json(output_dir/"VERIFICATION_PROGRESS.json",fallback); atomic_json(output_dir/"VERIFICATION_STATUS.json",fallback)
        if (output_dir/"FULL_VERIFICATION.json").exists():
            raise RuntimeError("Fail-closed invariant violated: stale FULL_VERIFICATION exists")
        print(f"DRIFT_VERIFICATION_FAIL type={type(exc).__name__}",flush=True); return 1
    status_path=output_dir/"STATUS.json"; previous_status_hash=report["hash_chain"]["runner_status_completed_unverified_sha256"]
    if sha256_file(status_path)!=previous_status_hash: raise RuntimeError("Runner STATUS changed during verification")
    status=strict_load(status_path)
    full_path=output_dir/"FULL_VERIFICATION.json"; atomic_json(full_path,report); full_hash=sha256_file(full_path)
    status.update({
        "status":"completed_verified","previous_status_sha256":previous_status_hash,"full_verification_sha256":full_hash,
        "terminal_sha256":report["hash_chain"]["terminal_sha256"],"sealed_records_sha256":report["hash_chain"]["sealed_records_sha256"],
        "run_manifest_sha256":report["hash_chain"]["run_manifest_sha256"],"verifier_source_sha256":report["hash_chain"]["verifier_source_sha256"],
    })
    atomic_json(status_path,status); final_status_hash=sha256_file(status_path)
    progress_holder[0].finish("completed_verified",{"full_verification_sha256":full_hash,"final_runner_status_sha256":final_status_hash,"terminal_sha256":report["hash_chain"]["terminal_sha256"]})
    print(f"DRIFT_VERIFICATION_PASS checks={report['checks']} profile_replays={report['full_stochastic_profile_replays']}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
