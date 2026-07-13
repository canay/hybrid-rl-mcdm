from __future__ import annotations

import importlib.util
import copy
import json
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


VERIFY = load("latency_verify_test", CAMPAIGN / "verify_latency.py")


def test_independent_scorer_random_constant_and_ties():
    rng = np.random.RandomState(123)
    q = rng.normal(size=400)
    topsis = rng.uniform(size=400)
    q_norm = (q - q.min()) / (q.max() - q.min() + 1e-10)
    t_norm = (topsis - topsis.min()) / (topsis.max() - topsis.min() + 1e-10)
    expected = [int(x) for x in np.argsort(0.5 * q_norm + 0.5 * t_norm)[::-1][:7]]
    assert VERIFY.independent_top7(q, topsis) == expected

    constant = np.ones(400)
    ramp = np.linspace(0.0, 1.0, 400)
    assert VERIFY.independent_top7(constant, ramp) == [399, 398, 397, 396, 395, 394, 393]

    ties = np.ones(400)
    producer_tie_expected = [
        int(x) for x in np.argsort(0.5 * ties + 0.5 * ties)[::-1][:7]
    ]
    assert VERIFY.independent_top7(ties, ties) == producer_tie_expected


def test_bootstrap_label_and_known_constant_statistic():
    blocks = []
    for _ in range(20):
        blocks.append(
            {
                "median_ns": 10.0,
                "p95_ns": 20.0,
                "p99_ns": 30.0,
            }
        )
    result = VERIFY.block_bootstrap_ci(blocks, 50, 123)
    assert result["method"] == "nonparametric bootstrap over blocks; estimator is the median of the selected block-level statistic"
    assert result["statistics"]["median_ns"] == {"estimate_ns": 10.0, "ci_lo_ns": 10.0, "ci_hi_ns": 10.0}
    assert result["statistics"]["p95_ns"] == {"estimate_ns": 20.0, "ci_lo_ns": 20.0, "ci_hi_ns": 20.0}
    assert result["statistics"]["p99_ns"] == {"estimate_ns": 30.0, "ci_lo_ns": 30.0, "ci_hi_ns": 30.0}


def test_verifier_primary_gate_matches_runner_on_clean_raw():
    raw = np.full((3, 20, 25), 60_000, dtype=np.uint64)
    result = VERIFY.analyze_durations(raw, 25, 20)
    assert len(result["blocks"]) == 60
    assert result["threshold_diagnostics"]["primary_condition_met"] is True


def valid_environment_record() -> dict:
    return {
        "python": {
            "version_info": [3, 12, 12],
            "implementation": "CPython",
            "executable": str(VERIFY.expected_producer_executable()),
        },
        "libraries": {"numpy": "1.26.0"},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("python.version_info", [3, 12, 11]),
        ("python.implementation", "PyPy"),
        ("python.executable", "intentional-nonpath-mismatch-sentinel"),
        ("libraries.numpy", "2.0.0"),
    ],
)
def test_exact_producer_environment_negative_matrix(field, value):
    environment = valid_environment_record()
    section, key = field.split(".")
    environment[section][key] = value
    assert not VERIFY.producer_environment_record_ok(environment)
    assert VERIFY.producer_environment_record_ok(valid_environment_record())


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "time.time_ns"),
        ("implementation", "wrong timer"),
        ("monotonic", False),
        ("resolution_seconds", 0.0),
    ],
)
def test_exact_timer_negative_matrix(field, value):
    timer = {
        "name": "perf_counter_ns",
        "implementation": "QueryPerformanceCounter()",
        "monotonic": True,
        "resolution_seconds": 1e-7,
    }
    assert VERIFY.timer_record_ok(timer)
    timer[field] = value
    assert not VERIFY.timer_record_ok(timer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "wrong"),
        ("campaign_id", "wrong"),
        ("claim_boundary", "end-to-end request"),
        ("batch_size", 2),
    ],
)
def test_exact_result_header_negative_matrix(field, value):
    result = {
        "schema_version": "hre.latency_result.v1",
        "campaign_id": CAMPAIGN.name,
        "claim_boundary": VERIFY.EXPECTED_CLAIM_BOUNDARY,
        "batch_size": 1,
    }
    result[field] = value
    with pytest.raises(AssertionError):
        VERIFY.validate_result_header(VERIFY.Audit(), result)


@pytest.mark.parametrize(
    "mutation",
    ["path_traversal", "dtype", "shape"],
)
def test_artifact_declaration_negative_matrix(tmp_path, mutation):
    array = np.ones((1, 2, 3), dtype=np.uint64)
    declaration = {"path": "raw_durations_ns.npy", "dtype": "uint64", "shape": [1, 2, 3]}
    if mutation == "path_traversal":
        declaration["path"] = "../raw_durations_ns.npy"
    elif mutation == "dtype":
        declaration["dtype"] = "float64"
    else:
        declaration["shape"] = [6]
    with pytest.raises(AssertionError):
        VERIFY.validate_artifact_declaration(
            VERIFY.Audit(), tmp_path, declaration, "raw_durations_ns.npy", array
        )


def test_recomputed_retention_gate_exact_dict_and_adversarial_mutations():
    expected = {
        "threshold_ns": 1_000_000,
        "status": "PASS_RETAIN_CACHED_PATH_CLAIM",
        "canonical_condition_met": True,
        "accuracy_ok": True,
        "runtime_ok": True,
        "verifier_pass_required": True,
    }
    assert VERIFY.recomputed_retention_gate("canonical", True, True, True) == expected
    for key in expected:
        tampered = copy.deepcopy(expected)
        if isinstance(tampered[key], bool):
            tampered[key] = not tampered[key]
        elif isinstance(tampered[key], int):
            tampered[key] += 1
        else:
            tampered[key] = "tampered"
        with pytest.raises(AssertionError):
            VERIFY.compare(VERIFY.Audit(), expected, tampered, "retention_gate")


@pytest.mark.parametrize("mutation", ["runner_path", "runner_hash", "manifest_path", "manifest_hash"])
def test_locked_manifest_runner_binding_negative_matrix(mutation):
    manifest_path = CAMPAIGN / "RUN_MANIFEST.json"
    if not manifest_path.is_file():
        pytest.skip("Locked manifest is generated after source finalization")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {
        "run_manifest": {
            "path": str(manifest_path.resolve().relative_to(VERIFY.PROJECT_ROOT)),
            "sha256": VERIFY.sha256_file(manifest_path),
            "status": manifest["status"],
        },
        "runner": copy.deepcopy(manifest["lock_files"]["runner"]),
    }
    if mutation == "runner_path":
        result["runner"]["path"] = result["runner"]["path"].replace("latency_benchmark.py", "wrong.py")
    elif mutation == "runner_hash":
        result["runner"]["sha256"] = "0" * 64
    elif mutation == "manifest_path":
        result["run_manifest"]["path"] = "wrong.json"
    else:
        result["run_manifest"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        VERIFY.validate_run_manifest(VERIFY.Audit(), result)
