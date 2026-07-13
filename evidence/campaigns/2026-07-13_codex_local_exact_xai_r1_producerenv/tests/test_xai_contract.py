from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
SOURCE = CAMPAIGN.parent / "2026-07-12_codex_local_same_target_bridge_r01_producerenv"
EVIDENCE = CAMPAIGN.parent / "2026-07-13_codex_static_verifier_topsis_gatefix"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load(CAMPAIGN / "src" / "xai_main.py", "exact_xai_runner_tests")


def smoke_rows():
    return RUNNER.read_jsonl(SOURCE / "outputs" / "smoke" / "main_catalogs.jsonl")


def test_verified_smoke_allowlist_has_no_evaluation_or_label_fields():
    rows = RUNNER.extract_allowlisted_inputs(smoke_rows(), SOURCE)
    assert len(rows) == 1
    RUNNER.assert_label_free_keys(rows)
    flattened_keys = []

    def walk(value):
        if isinstance(value, dict):
            flattened_keys.extend(str(key).lower() for key in value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(rows)
    for token in RUNNER.FORBIDDEN_INPUT_KEY_PARTS:
        assert all(token not in key for key in flattened_keys)


def test_allowlist_guard_fails_closed_on_gt_and_metric_keys():
    with pytest.raises(ValueError):
        RUNNER.assert_label_free_keys({"gt": [1]})
    with pytest.raises(ValueError):
        RUNNER.assert_label_free_keys({"gt_rank": [1]})
    with pytest.raises(ValueError):
        RUNNER.assert_label_free_keys({"final_metrics": {}})


def test_exact_shapley_efficiency_dummy_symmetry_and_additivity():
    # A two-active-player additive game: players 2 and 3 are exact dummies.
    v1 = {mask: float(2 * bool(mask & 1) + 3 * bool(mask & 2)) for mask in range(16)}
    p1 = RUNNER.exact_shapley_from_values(v1, 4)
    assert np.allclose(p1, [2.0, 3.0, 0.0, 0.0], atol=1e-15, rtol=0)
    assert abs(v1[0] + p1.sum() - v1[15]) <= 1e-15

    # Cardinality-only game is symmetric in all four players.
    vs = {mask: float(mask.bit_count() ** 2) for mask in range(16)}
    ps = RUNNER.exact_shapley_from_values(vs, 4)
    assert np.max(np.abs(ps - ps[0])) <= 1e-15
    assert abs(vs[0] + ps.sum() - vs[15]) <= 1e-12

    v2 = {mask: float(5 * bool(mask & 4) - bool(mask & 8)) for mask in range(16)}
    p2 = RUNNER.exact_shapley_from_values(v2, 4)
    combined = {mask: v1[mask] + v2[mask] for mask in range(16)}
    pc = RUNNER.exact_shapley_from_values(combined, 4)
    assert np.allclose(pc, p1 + p2, atol=1e-15, rtol=0)


def test_fixed_topsis_shapley_reconstructs_every_smoke_item_to_1e12():
    row = smoke_rows()[0]
    path = SOURCE / "inputs" / Path(row["dataset_path"])
    frame = pd.read_csv(path)
    bundle = RUNNER.fixed_topsis_bundle(frame)
    base, phi, error = RUNNER.exact_topsis_shapley(bundle)
    assert error <= 1e-12
    actual = np.asarray(bundle["normalized_scores"])
    assert np.max(np.abs(base + phi.sum(axis=1) - actual)) <= 1e-12


def test_full_smoke_budget_reward_replay_and_trace_hash_are_exact():
    selected = RUNNER.extract_allowlisted_inputs(smoke_rows(), SOURCE)[0]
    source_full = json.loads((SOURCE / "outputs" / "smoke" / "FULL_VERIFICATION.json").read_text())
    core_path = SOURCE / "src" / "original_hybrid_core.py"
    assert RUNNER.sha256_file(core_path) == source_full["input_hashes"]["src/original_hybrid_core.py"]
    core = RUNNER.load_module(core_path)
    frame = pd.read_csv(SOURCE / "inputs" / Path(selected["dataset_path"]))
    profile = selected["profiles"][0]
    replay1 = RUNNER.replay_reward_components(
        frame,
        "budget",
        0,
        selected["run_seed"],
        np.asarray(profile["q_scores"]),
        np.asarray(profile["visits"], dtype=np.int32),
        core,
    )
    replay2 = RUNNER.replay_reward_components(
        frame,
        "budget",
        0,
        selected["run_seed"],
        np.asarray(profile["q_scores"]),
        np.asarray(profile["visits"], dtype=np.int32),
        core,
    )
    assert replay1["source_error"] <= 1e-12
    assert replay1["component_error"] <= 1e-12
    assert replay1["action_trace_sha256"] == replay2["action_trace_sha256"]
    assert replay1["reward_event_trace_sha256"] == replay2["reward_event_trace_sha256"]
    assert len(replay1["action_trace_sha256"]) == 64
    cq = RUNNER.cq_affine_decomposition(replay1)
    reconstructed = cq["reference"] + cq["base"] + cq["engage"] + cq["convert"]
    assert np.max(np.abs(cq["c_q"] - reconstructed)) <= 1e-12


def test_constant_q_affine_decomposition_has_quarter_reference():
    zeros = np.zeros(4)
    value = RUNNER.cq_affine_decomposition(
        {"q_total": zeros, "q_base": zeros, "q_engage": zeros, "q_convert": zeros}
    )
    assert value["is_constant"] is True
    assert value["reference"] == 0.25
    assert np.array_equal(value["c_q"], np.full(4, 0.25))
    assert np.array_equal(value["base"], zeros)


def test_nonempty_output_directory_is_rejected(tmp_path: Path):
    (tmp_path / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        RUNNER.prepare_output(tmp_path)


def test_nonempty_cli_failure_leaves_existing_tree_byte_identical(tmp_path: Path):
    output = CAMPAIGN / "outputs" / f"_test_occupied_{os.getpid()}_{tmp_path.name}"
    if output.exists():
        shutil.rmtree(output)
    nested = output / "nested"
    nested.mkdir(parents=True)
    (output / "keep.bin").write_bytes(b"\x00do-not-touch\xff")
    (nested / "keep.txt").write_text("unchanged", encoding="utf-8")

    def snapshot(root: Path):
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
        }

    before = snapshot(output)
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-u",
                str(CAMPAIGN / "src" / "xai_main.py"),
                "--source-data-campaign", str(SOURCE),
                "--source-data-root", str(SOURCE / "outputs" / "smoke"),
                "--source-evidence-campaign", str(EVIDENCE),
                "--source-evidence-root", str(EVIDENCE / "outputs" / "smoke_overlay"),
                "--output-dir", str(output),
                "--mode", "smoke",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        assert process.returncode != 0
        assert snapshot(output) == before
        assert not (output / "status.json").exists()
    finally:
        if output.exists():
            shutil.rmtree(output)


def test_protocol_declares_locked_authorization_and_no_touch_boundary():
    text = (CAMPAIGN / "PROTOCOL_LOCK.md").read_text(encoding="utf-8")
    assert "Status: `LOCKED`" in text
    assert "HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19" in text
    assert "manuscript" in text
    assert "response" in text
    assert "python -u" in text


def test_smoke_source_cannot_be_mislabeled_as_canonical():
    with pytest.raises(ValueError, match="name mismatch"):
        RUNNER.validate_source(
            SOURCE,
            SOURCE / "outputs" / "smoke",
            EVIDENCE,
            EVIDENCE / "outputs" / "smoke_overlay",
            "canonical",
        )


def test_source_pins_environment_and_hash_chain_are_exact():
    rows, _core, provenance = RUNNER.validate_source(
        SOURCE,
        SOURCE / "outputs" / "smoke",
        EVIDENCE,
        EVIDENCE / "outputs" / "smoke_overlay",
        "smoke",
    )
    assert len(rows) == 1
    assert RUNNER.environment_versions() == RUNNER.EXPECTED_ENVIRONMENT
    assert provenance["data_run_manifest_sha256"] == RUNNER.SOURCE_RUN_MANIFEST_SHA256
    assert provenance["evidence_run_manifest_sha256"] == RUNNER.SOURCE_EVIDENCE_RUN_MANIFEST_SHA256
    assert provenance["core_sha256"] == RUNNER.SOURCE_CORE_SHA256


def test_canonical_binding_and_full_allowlist_extraction_are_50x5_complete():
    rows, _core, provenance = RUNNER.validate_source(
        SOURCE,
        SOURCE / "outputs" / "canonical_main",
        EVIDENCE,
        EVIDENCE / "outputs" / "canonical_overlay",
        "canonical",
    )
    selected = RUNNER.extract_allowlisted_inputs(rows, SOURCE)
    assert [row["run_index"] for row in selected] == list(range(50))
    assert sum(len(row["profiles"]) for row in selected) == 250
    assert all(
        [profile["profile_name"] for profile in row["profiles"]] == list(RUNNER.PROFILE_ORDER)
        for row in selected
    )
    RUNNER.assert_label_free_keys(selected)
    assert provenance["data_main_catalogs_sha256"] == RUNNER.CANONICAL_DATA_HASHES["main_catalogs.jsonl"]
    assert provenance["evidence_full_verification_sha256"] == RUNNER.CANONICAL_EVIDENCE_HASHES["FULL_VERIFICATION.json"]


def test_locked_policy_authorizes_canonical_without_launching_it():
    _manifest_sha, _hashes, policy = RUNNER.validate_run_manifest(CAMPAIGN)
    assert policy["allowed_modes"] == ["smoke", "canonical"]
    assert policy["canonical_authorized"] is True
    assert policy["authorization_operation_id"] == "HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonstandard_numeric_constants(tmp_path: Path, constant: str):
    path = tmp_path / "bad.json"
    path.write_text('{"value": ' + constant + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-standard JSON constant"):
        RUNNER.read_json(path)


def test_finite_tree_rejects_runtime_infinity():
    with pytest.raises(ValueError, match="Nonfinite"):
        RUNNER.assert_finite({"value": float("inf")})


def test_run_manifest_is_exact_and_authorizing():
    manifest_sha, hashes, policy = RUNNER.validate_run_manifest(CAMPAIGN)
    manifest = RUNNER.read_json(CAMPAIGN / "RUN_MANIFEST.json")
    assert len(manifest_sha) == 64
    assert manifest["status"] == "LOCKED"
    assert manifest["canonical_authorized"] is True
    assert set(hashes) == {
        "PROTOCOL_LOCK.md",
        "src/xai_main.py",
        "verify_xai.py",
        "build_run_manifest.py",
        "tests/test_xai_contract.py",
        "tests/test_verify_xai.py",
    }
    assert policy["allowed_modes"] == ["smoke", "canonical"]
    assert policy["canonical_authorized"] is True


def test_manifest_builder_matches_checked_in_lock():
    builder = load(CAMPAIGN / "build_run_manifest.py", "exact_xai_manifest_builder_tests")
    assert builder.build_manifest(CAMPAIGN) == RUNNER.read_json(CAMPAIGN / "RUN_MANIFEST.json")
