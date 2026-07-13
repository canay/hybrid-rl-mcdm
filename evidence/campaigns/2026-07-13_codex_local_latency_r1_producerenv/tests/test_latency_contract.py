from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("latency_prepare_test", CAMPAIGN / "src" / "prepare_verified_fixture.py")
BENCH = load("latency_bench_test", CAMPAIGN / "src" / "latency_benchmark.py")
VERIFY = load("latency_verify_lineage_test", CAMPAIGN / "verify_latency.py")


def namespace(mode: str, **overrides):
    values = {name: None for name in BENCH.CANONICAL_CONTRACT}
    values.update(overrides)
    return argparse.Namespace(mode=mode, **values)


def copied_inputs(tmp_path: Path) -> tuple[Path, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest_path = inputs / "fixture_manifest.json"
    fixture_path = inputs / "verified_vectors.npz"
    shutil.copy2(CAMPAIGN / "inputs" / "fixture_manifest.json", manifest_path)
    shutil.copy2(CAMPAIGN / "inputs" / "verified_vectors.npz", fixture_path)
    manifest_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    fixture_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    return manifest_path, fixture_path


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_npz(path: Path, mutate) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    return arrays


def test_canonical_contract_is_exact_and_nonconfigurable():
    assert BENCH.resolve_config(namespace("canonical")) == BENCH.CANONICAL_CONTRACT
    assert BENCH.CANONICAL_CONTRACT["calls_per_block"] // BENCH.PAIR_COUNT == 20
    assert (
        BENCH.CANONICAL_CONTRACT["passes"]
        * BENCH.CANONICAL_CONTRACT["blocks"]
        * BENCH.CANONICAL_CONTRACT["calls_per_block"]
        // BENCH.PAIR_COUNT
        == 1200
    )
    with pytest.raises(AssertionError, match="immutable"):
        BENCH.resolve_config(namespace("canonical", calls_per_block=4_999))


def test_thread_limits_are_one_after_pre_numpy_setup():
    assert all(__import__("os").environ[key] == "1" for key in BENCH.THREAD_ENV_KEYS)


def test_fixture_builder_accepts_only_distinctly_verified_canonical_data(tmp_path):
    manifest = PREPARE.build_fixture(tmp_path)
    assert manifest["schema_version"] == "hre.latency_fixture.v2"
    assert manifest["material_passport"]["verification_status"] == "VERIFIED_BY_DISTINCT_OVERLAY"
    assert manifest["data_source"]["run_indices"] == list(range(50))
    assert manifest["data_source"]["arm_id"] == BENCH.PRIMARY_ARM_ID
    assert manifest["verification_evidence"]["campaign"] == BENCH.EVIDENCE_CAMPAIGN_NAME
    assert manifest["arrays"]["topsis"]["shape"] == [50, 400]
    assert manifest["arrays"]["q_scores"]["shape"] == [50, 5, 400]
    assert manifest["arrays"]["expected_top7"]["shape"] == [50, 5, 7]
    assert manifest["pair_schedule"]["pair_count"] == 250


def test_primary_gate_fails_raw_tail_even_with_18_of_20_stable_blocks():
    raw = np.full((1, 20, 100), 50_000, dtype=np.uint64)
    raw[0, 18:, :] = 2_000_000
    analysis = BENCH.analyze_durations(raw, 100, 20)
    assert analysis["passes"][0]["stable_blocks"] == 18
    assert analysis["threshold_diagnostics"]["stability_sufficient"] is True
    assert analysis["threshold_diagnostics"]["primary_condition_met"] is False


def test_primary_gate_is_strict_and_exact_one_millisecond_fails():
    raw = np.full((1, 20, 100), 1_000_000, dtype=np.uint64)
    analysis = BENCH.analyze_durations(raw, 100, 20)
    assert analysis["raw_all_samples"]["p99_ns"] == 1_000_000
    assert analysis["threshold_diagnostics"]["primary_condition_met"] is False


def test_any_pass_all_sample_p99_failure_cannot_hide_in_pooled_p99():
    raw = np.full((3, 20, 100), 50_000, dtype=np.uint64)
    raw[0, :, -2:] = 2_000_000  # 2% in one pass, only 0.67% when pooled.
    analysis = BENCH.analyze_durations(raw, 100, 20)
    diagnostics = analysis["threshold_diagnostics"]
    assert diagnostics["pooled_raw_p99_strictly_below_threshold"] is True
    assert diagnostics["every_pass_all_sample_p99_strictly_below_threshold"] is False
    assert diagnostics["primary_condition_met"] is False


@pytest.mark.parametrize(
    "mutation",
    ["wrong_arm", "wrong_run", "profile_count", "q", "topsis", "ranking", "source_full_hash", "source_catalog_hash"],
)
def test_fixture_lineage_negative_matrix(tmp_path, mutation):
    manifest_path, fixture_path = copied_inputs(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "wrong_arm":
        manifest["data_source"]["arm_id"] = "wrong-arm"
    elif mutation == "wrong_run":
        manifest["data_source"]["run_indices"][0] = 1
    elif mutation == "profile_count":
        manifest["data_source"]["profile_order"] = manifest["data_source"]["profile_order"][:-1]
    elif mutation == "ranking":
        arrays = rewrite_npz(fixture_path, lambda values: values["expected_top7"].__setitem__((0, 0, 0), 399))
        manifest["fixture_sha256"] = VERIFY.sha256_file(fixture_path)
        manifest["arrays"]["expected_top7"] = VERIFY.typed_array_contract(arrays["expected_top7"])
    elif mutation == "source_full_hash":
        manifest["verification_evidence"]["full_sha256"] = "0" * 64
    elif mutation == "source_catalog_hash":
        manifest["data_source"]["main_catalogs_sha256"] = "0" * 64
    elif mutation == "q":
        arrays = rewrite_npz(fixture_path, lambda values: values["q_scores"].__setitem__((0, 0, 0), values["q_scores"][0, 0, 0] + 1.0))
        manifest["fixture_sha256"] = VERIFY.sha256_file(fixture_path)
        manifest["arrays"]["q_scores"] = VERIFY.typed_array_contract(arrays["q_scores"])
    elif mutation == "topsis":
        arrays = rewrite_npz(fixture_path, lambda values: values["topsis"].__setitem__((0, 0), values["topsis"][0, 0] + 0.01))
        manifest["fixture_sha256"] = VERIFY.sha256_file(fixture_path)
        manifest["arrays"]["topsis"] = VERIFY.typed_array_contract(arrays["topsis"])
    write_manifest(manifest_path, manifest)
    with pytest.raises(AssertionError):
        VERIFY.validate_fixture_lineage(VERIFY.Audit(), manifest_path, fixture_path)


def test_source_terminal_status_and_hash_are_fail_closed():
    good = {
        "status": "completed_verified",
        "verdict": "PASS",
        "output_hashes": {
            "main_catalogs.jsonl": VERIFY.CANONICAL_CATALOGS_SHA256,
            "main_results.json": VERIFY.CANONICAL_TERMINAL_SHA256,
            "status.json": VERIFY.CANONICAL_STATUS_SHA256,
        },
    }
    VERIFY.require_verified_source_terminal(VERIFY.Audit(), good)
    for change in (
        {"status": "completed_unverified"},
        {"verdict": "FAIL"},
        {"output_hashes": {"main_catalogs.jsonl": "0" * 64}},
    ):
        bad = {**good, **change}
        with pytest.raises(AssertionError):
            VERIFY.require_verified_source_terminal(VERIFY.Audit(), bad)


def test_preexisting_full_and_verification_status_are_quarantined(tmp_path):
    (tmp_path / "FULL_VERIFICATION.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    (tmp_path / "verification_status.json").write_text('{"status":"completed_verified"}', encoding="utf-8")
    with pytest.raises(AssertionError, match="Pre-existing"):
        VERIFY.reject_preexisting_verification_artifacts(tmp_path)
    assert not (tmp_path / "FULL_VERIFICATION.json").exists()
    assert not (tmp_path / "verification_status.json").exists()
    assert len(list(tmp_path.glob("REJECTED_PREEXISTING*"))) == 2


def test_preexisting_verifier_artifacts_end_to_end_leave_failed_closed_without_stale_full(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "FULL_VERIFICATION.json").write_text('{"verdict":"PASS"}', encoding="utf-8")
    (output / "verification_status.json").write_text('{"status":"completed_verified"}', encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(CAMPAIGN / "verify_latency.py"), "--mode", "smoke", "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not (output / "FULL_VERIFICATION.json").exists()
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    verification = json.loads((output / "verification_status.json").read_text(encoding="utf-8"))
    assert status["status"] == verification["status"] == "failed_closed"
    terminal = output / status["terminal_path"]
    assert status["terminal_sha256"] == verification["terminal_sha256"] == VERIFY.sha256_file(terminal)
    assert list(output.glob("REJECTED_PREEXISTING*FULL_VERIFICATION.json"))


def test_runtime_contract_detects_pre_post_priority_or_affinity_mutation():
    env = {
        "runtime_pre": {"logical_processors": 1, "priority_name": "Normal", "priority_class": 32, "affinity_mask": "0x1"},
        "runtime_post": {"logical_processors": 1, "priority_name": "Normal", "priority_class": 32, "affinity_mask": "0x1"},
        "threadpool_pre": {"available": False},
        "threadpool_post": {"available": False},
        "thread_environment": {
            "forced_before_numpy_import": {key: "1" for key in BENCH.THREAD_ENV_KEYS},
            "post_timing_values": {key: "1" for key in BENCH.THREAD_ENV_KEYS},
        },
    }
    assert VERIFY.runtime_contract_ok(env)
    env["runtime_post"]["affinity_mask"] = "0x2"
    assert not VERIFY.runtime_contract_ok(env)
    env["runtime_post"]["affinity_mask"] = "0x1"
    env["runtime_post"]["priority_class"] = 128
    assert not VERIFY.runtime_contract_ok(env)


def test_timer_and_positive_duration_regression_contract():
    assert BENCH.THRESHOLD_NS == 1_000_000
    assert BENCH.summary(np.asarray([1, 2, 3], dtype=np.uint64))["min_ns"] > 0
    zero = np.zeros((1, 1, 3), dtype=np.uint64)
    assert not bool(np.all(zero > 0))


def test_locked_execution_policy_authorizes_only_smoke_and_canonical():
    locked = {
        "allowed_modes": ["smoke", "canonical"],
        "canonical_launch_authorized": True,
    }
    BENCH.enforce_execution_policy("smoke", {"execution_policy": locked})
    BENCH.enforce_execution_policy("canonical", {"execution_policy": locked})
    with pytest.raises(AssertionError, match="allowlist"):
        BENCH.enforce_execution_policy("other", {"execution_policy": locked})


def test_runner_preexisting_output_failure_is_top_level_failed_closed(tmp_path):
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "junk.txt").write_text("preexisting", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN / "src" / "latency_benchmark.py"),
            "--mode",
            "smoke",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed_closed"
    terminal = output / status["terminal_path"]
    assert status["terminal_sha256"] == VERIFY.sha256_file(terminal)


def test_completed_fresh_smoke_status_binds_full_hash_if_present():
    output = CAMPAIGN / "outputs" / "smoke"
    full = output / "FULL_VERIFICATION.json"
    if not full.is_file():
        pytest.skip("No post-audit terminal smoke fixture yet")
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    verification = json.loads((output / "verification_status.json").read_text(encoding="utf-8"))
    expected = VERIFY.sha256_file(full)
    assert status["status"] == verification["status"] == "completed_verified"
    assert status["terminal_sha256"] == verification["terminal_sha256"] == expected
    assert json.loads(full.read_text(encoding="utf-8"))["verdict"] == "PASS"
