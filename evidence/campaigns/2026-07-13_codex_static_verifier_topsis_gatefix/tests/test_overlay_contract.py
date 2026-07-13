from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("static_verifier_overlay", CAMPAIGN / "verify_overlay.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)


def test_explicit_source_bindings_and_original_failure_are_current():
    audit = VERIFY.Audit()
    evidence = VERIFY.verify_source_and_overlay_contract(audit)
    assert evidence["source_run_manifest_sha256"] == VERIFY.SOURCE_RUN_MANIFEST_SHA256
    assert evidence["source_canonical_terminal_sha256"] == VERIFY.SOURCE_CANONICAL_TERMINAL_SHA256
    assert evidence["source_canonical_checkpoint_sha256"] == VERIFY.SOURCE_CANONICAL_CHECKPOINT_SHA256
    assert evidence["source_canonical_runner_status_sha256"] == VERIFY.SOURCE_CANONICAL_RUNNER_STATUS_SHA256
    assert evidence["python_version"] == "3.12.12"
    assert audit.gates[-1]["gate_id"] == "G00_overlay_source_and_failure_binding"


def test_hash_source_manifest_and_path_mismatches_fail(tmp_path: Path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"immutable")
    with pytest.raises(VERIFY.VerificationError, match="hash mismatch"):
        VERIFY.require_bound_file(sample, "0" * 64, "negative")
    with pytest.raises(VERIFY.VerificationError, match="hash mismatch"):
        VERIFY.require_bound_file(VERIFY.RUN_MANIFEST_PATH, "0" * 64, "source manifest negative")
    with pytest.raises(VERIFY.VerificationError, match="escapes"):
        VERIFY.safe_path(tmp_path, "../escape")


def test_original_failure_evidence_mismatch_fails():
    path = VERIFY.SOURCE_CAMPAIGN_ROOT / "outputs" / "canonical_main" / "verification_status.json"
    stderr_path = VERIFY.SOURCE_CAMPAIGN_ROOT / "runtime_logs" / "canonical_verify.stderr.log"
    status = VERIFY.strict_load(path)
    stderr = stderr_path.read_text(encoding="utf-8")
    VERIFY.validate_original_failure(status, stderr)
    tampered = copy.deepcopy(status)
    tampered["catalogs_completed"] = 30
    with pytest.raises(VERIFY.VerificationError, match="failure evidence mismatch"):
        VERIFY.validate_original_failure(tampered, stderr)
    with pytest.raises(VERIFY.VerificationError, match="stderr mismatch"):
        VERIFY.validate_original_failure(status, stderr + "tamper")


def test_overlay_manifest_source_binding_mismatch_fails():
    payload = VERIFY.strict_load(VERIFY.OVERLAY_MANIFEST_PATH)
    tampered = copy.deepcopy(payload)
    tampered["source_campaign"]["canonical_terminal_sha256"] = "0" * 64
    with pytest.raises(VERIFY.VerificationError, match="source binding mismatch"):
        VERIFY.validate_overlay_manifest(tampered)


def test_all_cli_arguments_are_required_without_creating_output():
    before = VERIFY.snapshot_tree_hashes(VERIFY.OVERLAY_ROOT / "outputs")
    with pytest.raises(SystemExit) as exc:
        VERIFY.parse_args([])
    assert exc.value.code == 2
    assert VERIFY.snapshot_tree_hashes(VERIFY.OVERLAY_ROOT / "outputs") == before


def test_locked_policy_allows_canonical_preflight_without_any_output_write(tmp_path: Path):
    output = VERIFY.OVERLAY_ROOT / "outputs" / "canonical_policy_probe_must_not_exist"
    assert not output.exists()
    evidence = VERIFY.assert_execution_policy(
        "canonical",
        VERIFY.SOURCE_CAMPAIGN_ROOT / "outputs" / "canonical_main",
        output,
    )
    assert evidence["execution_policy"]["canonical_authorized"] is True
    assert (
        evidence["execution_policy"]["authorization_operation_id"]
        == "HRE_R1_STATIC_GATEFIX_LOCK_20260713_CODEX_15"
    )
    assert not output.exists()


def test_locked_policy_allows_matching_smoke_without_writing():
    output = VERIFY.OVERLAY_ROOT / "outputs" / "smoke_policy_probe_must_not_exist"
    assert not output.exists()
    evidence = VERIFY.assert_execution_policy(
        "smoke", VERIFY.SOURCE_CAMPAIGN_ROOT / "outputs" / "smoke", output
    )
    assert evidence["execution_policy"] == VERIFY.EXPECTED_EXECUTION_POLICY
    assert not output.exists()


def test_policy_status_and_source_mode_mismatches_fail():
    payload = VERIFY.strict_load(VERIFY.OVERLAY_MANIFEST_PATH)
    tampered = copy.deepcopy(payload)
    tampered["execution_policy"]["authorization_operation_id"] = "unauthorized"
    with pytest.raises(VERIFY.VerificationError, match="execution-policy mismatch"):
        VERIFY.validate_overlay_manifest(tampered)
    with pytest.raises(VERIFY.VerificationError, match="mode/source-output status mismatch"):
        VERIFY.assert_execution_policy(
            "smoke",
            VERIFY.SOURCE_CAMPAIGN_ROOT / "outputs" / "canonical_main",
            VERIFY.OVERLAY_ROOT / "outputs" / "mismatch_probe_must_not_exist",
        )


def test_python_31212_is_bound_in_manifest_source_and_runtime(monkeypatch: pytest.MonkeyPatch):
    payload = VERIFY.strict_load(VERIFY.OVERLAY_MANIFEST_PATH)
    assert payload["environment"]["python"] == "3.12.12"
    tampered = copy.deepcopy(payload)
    tampered["environment"]["python"] = "3.12.11"
    with pytest.raises(VERIFY.VerificationError, match="Python version mismatch"):
        VERIFY.validate_overlay_manifest(tampered)
    monkeypatch.setattr(VERIFY.platform, "python_version", lambda: "3.12.11")
    with pytest.raises(VERIFY.VerificationError, match="runtime Python mismatch"):
        VERIFY.verify_source_and_overlay_contract(VERIFY.Audit())


def test_preexisting_verifier_outputs_fail_without_changing_any_file(tmp_path: Path):
    science = tmp_path / "main_results.json"
    science.write_text('{"status":"completed_unverified"}\n', encoding="utf-8")
    before = VERIFY.snapshot_tree_hashes(tmp_path)
    VERIFY.assert_verification_start_clean(tmp_path)
    assert VERIFY.snapshot_tree_hashes(tmp_path) == before
    status = tmp_path / "verification_status.json"
    status.write_text("{}\n", encoding="utf-8")
    before_blocked = VERIFY.snapshot_tree_hashes(tmp_path)
    with pytest.raises(FileExistsError, match="Stale verification artifacts"):
        VERIFY.assert_verification_start_clean(tmp_path)
    assert VERIFY.snapshot_tree_hashes(tmp_path) == before_blocked


def test_strict_json_nonfinite_and_duplicate_keys_fail():
    with pytest.raises(VERIFY.VerificationError, match="Non-finite"):
        VERIFY.strict_loads('{"x": NaN}')
    with pytest.raises(VERIFY.VerificationError, match="Duplicate"):
        VERIFY.strict_loads('{"x": 1, "x": 2}')
