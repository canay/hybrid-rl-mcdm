from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


verify = load("drift_verifier_test", ROOT / "verify_drift.py")


def test_verifier_frozen_hash_and_source_contract_without_outputs():
    audit = verify.Audit()
    verify.verify_environment(audit)
    # RUN_MANIFEST is intentionally absent until independent lock approval, so
    # frozen inputs and source are tested separately here.
    for relative, expected in verify.EXPECTED_HASHES.items():
        assert verify.sha256_file(ROOT / relative) == expected
    verify.verify_runner_source(audit)
    assert audit.checks >= 6


def test_strict_json_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="Duplicate JSON key"):
        verify.strict_load(path)


def test_strict_json_rejects_nonfinite(tmp_path: Path):
    path = tmp_path / "nan.json"
    path.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="Non-finite"):
        verify.strict_load(path)


def test_exact_comparator_detects_metric_tamper():
    audit = verify.Audit()
    with pytest.raises(verify.VerificationError, match="Float mismatch"):
        verify.compare_exact(audit, {"x": 0.2}, {"x": 0.1}, "payload")


def test_record_digest_is_order_invariant_but_value_sensitive():
    a = verify.canonical_sha256({"a": 1, "b": 2})
    b = verify.canonical_sha256({"b": 2, "a": 1})
    c = verify.canonical_sha256({"a": 1, "b": 3})
    assert a == b and a != c


def test_verifier_declares_full_stochastic_replay_in_source():
    source = (ROOT / "verify_drift.py").read_text(encoding="utf-8")
    assert "replay_profile(" in source
    assert "full_stochastic_profile_replays" in source
    assert "exact_legacy_raw_cells" in source


def test_verifier_start_clean_rejects_stale_pass_and_unexpected_log(tmp_path: Path):
    for name in verify.RUNNER_TERMINAL_OUTPUT_SET:
        (tmp_path / name).write_text("{}" if name.endswith(".json") else "", encoding="utf-8")
    (tmp_path / "FULL_VERIFICATION.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="stale"):
        verify.assert_verifier_start_clean(tmp_path)
    (tmp_path / "FULL_VERIFICATION.json").unlink()
    (tmp_path / "verifier.stdout.log").write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match="unexpected"):
        verify.assert_verifier_start_clean(tmp_path)


def test_verification_progress_is_atomic_progress_only(tmp_path: Path):
    progress = verify.VerificationProgress(tmp_path, 2)
    progress.step()
    payload = verify.strict_load(tmp_path / "VERIFICATION_PROGRESS.json")
    assert payload["completed_profile_arm_replays"] == 1
    assert payload["total_profile_arm_replays"] == 2
    assert payload["scientific_metrics_exposed"] is False
    assert "eta_seconds" in payload and not list(tmp_path.glob("*.tmp"))


def test_independent_paired_summary_degenerate_matches_seed_contract():
    result = verify.paired_summary([0.0, 0.0], "degenerate|verify")
    assert (result["wins"], result["ties"], result["losses"]) == (0, 2, 0)
    assert result["bootstrap_seed"] == verify.analysis_seed("degenerate|verify")
    assert result["bootstrap_reps"] == 20_000


def test_pass_report_source_declares_hash_chain_and_core_reuse_disclosure():
    source = (ROOT / "verify_drift.py").read_text(encoding="utf-8")
    for token in ("runner_status_completed_unverified_sha256", "verifier_source_sha256", "frozen_input_sha256", "independence_disclosure"):
        assert token in source
