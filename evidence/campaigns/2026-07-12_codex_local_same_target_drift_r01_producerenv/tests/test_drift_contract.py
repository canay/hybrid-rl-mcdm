from __future__ import annotations

import importlib.util
import inspect
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


drift = load("drift_contract_test", ROOT / "src" / "drift_main.py")
lock = load("drift_lock_test", ROOT / "src" / "lock_campaign.py")


def first_catalog():
    manifest = json.loads((ROOT / "inputs" / "data" / "processed" / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["runs"][0]
    return entry, drift.load_catalog(entry)


def test_exact_environment_and_frozen_hashes():
    assert drift.assert_exact_environment() == drift.EXPECTED_ENVIRONMENT
    verified = drift.assert_frozen_inputs()
    assert verified["inputs/results/amazon_drift.json"].startswith("bd8e3523")
    assert verified["inputs/results/validation_extensions.json"].startswith("1c1700c9")


def test_stored_exact_raw_cell_contract_is_900_and_540():
    sudden = json.loads((ROOT / "inputs" / "results" / "amazon_drift.json").read_text(encoding="utf-8"))
    gradual = json.loads((ROOT / "inputs" / "results" / "validation_extensions.json").read_text(encoding="utf-8"))["gradual_multidim_drift"]
    assert sum(len(sudden["summary"][m][str(cp)]["raw"]) for m in drift.METHODS for cp in drift.SUDDEN_CHECKPOINTS) == 900
    assert sum(len(gradual["summary"][m][str(cp)]["raw"]) for m in ("topsis_only", "rl_only", "hybrid") for cp in drift.GRADUAL_CHECKPOINTS) == 540


def test_sudden_boundary_and_gradual_key_schedule():
    assert 15000 <= drift.SUDDEN_BOUNDARY < 15001
    assert drift.gradual_fraction(10000) == 0.0
    assert drift.gradual_fraction(25000) == 1.0
    assert int(round(drift.gradual_fraction(13750) * 40)) == 10
    assert int(round(drift.gradual_fraction(15625) * 40)) == 15


def test_corrected_training_signatures_cannot_receive_targets():
    proof = drift.future_blind_source_contract()
    assert set(proof) == {"corrected_sudden_train", "corrected_gradual_train"}
    for function in (drift.corrected_sudden_train, drift.corrected_gradual_train):
        assert all("target" not in name.lower() and not name.lower().startswith("gt") for name in inspect.signature(function).parameters)


@pytest.mark.parametrize("model", drift.REWARD_MODELS)
def test_short_corrected_sudden_is_deterministic_and_full_catalog(model):
    entry, df = first_catalog()
    topsis = drift.CORE.topsis_artifacts(df)["scores"]
    a = drift.corrected_sudden_train(df, drift.CORE.PROFILE_ORDER[0], 0, int(entry["seed"]), topsis, model, checkpoints=(32,))
    b = drift.corrected_sudden_train(df, drift.CORE.PROFILE_ORDER[0], 0, int(entry["seed"]), topsis, model, checkpoints=(32,))
    assert a == b
    assert sum(1 for value in a["states"]["32"]["rl_rank"] if 0 <= value < 400) == 7


@pytest.mark.parametrize("model", drift.REWARD_MODELS)
def test_short_corrected_gradual_is_deterministic(model):
    entry, df = first_catalog()
    topsis = drift.CORE.topsis_artifacts(df)["scores"]
    a = drift.corrected_gradual_train(df, drift.CORE.PROFILE_ORDER[0], 0, int(entry["seed"]), topsis, model, checkpoints=(32,))
    b = drift.corrected_gradual_train(df, drift.CORE.PROFILE_ORDER[0], 0, int(entry["seed"]), topsis, model, checkpoints=(32,))
    assert a == b
    assert a["states"]["32"]["target_key"] == 0


def test_future_target_mutation_changes_label_not_training_prefix():
    entry, df = first_catalog()
    topsis = drift.CORE.topsis_artifacts(df)["scores"]
    trained = drift.corrected_sudden_train(df, drift.CORE.PROFILE_ORDER[0], 0, int(entry["seed"]), topsis, drift.PRIMARY_REWARD_MODEL, checkpoints=(64,))
    original = drift.sudden_targets(df, drift.CORE.PROFILE_ORDER[0], int(entry["seed"]), 0)["post"]
    altered = drift.sudden_targets(df, drift.CORE.PROFILE_ORDER[0], int(entry["seed"]), 100000)["post"]
    assert original != altered
    assert trained["states"]["64"]["state_sha256"] == trained["states"]["64"]["state_sha256"]


def test_metapmorphic_preflight_all_reward_scenario_pairs_pass():
    entry, df = first_catalog()
    proof = drift.metamorphic_preflight(df, int(entry["seed"]), drift.CORE.topsis_artifacts(df)["scores"])
    assert set(proof) == set(drift.REWARD_MODELS)
    assert all(all(item.values()) for item in proof.values())


def test_active_protocol_is_lock_ready():
    assert "Status: `LOCKED_BEFORE_COMPUTE`" in lock.PROTOCOL_LOCK.read_text(encoding="utf-8")


def test_lock_creator_refuses_draft_and_creates_nothing(tmp_path: Path, monkeypatch):
    draft = tmp_path / "PROTOCOL_LOCK.md"
    draft.write_text("Status: `DRAFT_DO_NOT_COMPUTE`\n", encoding="utf-8")
    manifest = tmp_path / "RUN_MANIFEST.json"
    monkeypatch.setattr(lock, "PROTOCOL_LOCK", draft)
    monkeypatch.setattr(lock, "MANIFEST_PATH", manifest)
    with pytest.raises(ValueError, match="LOCKED_BEFORE_COMPUTE"):
        lock.create()
    assert not manifest.exists()


def test_auc_is_checkpoint_time_normalized():
    assert drift.normalized_auc([1, 1, 1, 1, 1], drift.SUDDEN_AUC_GRID) == 1.0
    assert drift.normalized_auc([0, 0, 0, 0, 0], drift.GRADUAL_AUC_GRID) == 0.0


def test_primary_corrected_reward_is_unique_and_prespecified():
    assert drift.PRIMARY_REWARD_MODEL == "component_continuous_fix"
    assert drift.REWARD_MODELS.count(drift.PRIMARY_REWARD_MODEL) == 1


def valid_manifest_payload():
    return {
        "schema_version": "same_target_drift.run_manifest.v1",
        "campaign_id": drift.CAMPAIGN_ROOT.name,
        "status": "LOCKED_BEFORE_COMPUTE",
        "created_at": "2026-07-13T00:00:00+03:00",
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "operation_id": "TEST",
        "hash_algorithm": "SHA-256",
        "contract": drift.RUN_MANIFEST_CONTRACT,
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": drift.EXPECTED_ENVIRONMENT},
        "producer_provenance": {"tag": "v1.0-submission", "commit_sha1": drift.PRODUCER_COMMIT},
        "files": [
            {"path": path, "sha256": "0" * 64, "bytes": 0}
            for path in sorted(drift.expected_run_manifest_paths())
        ],
    }


def test_run_manifest_exact_path_set_rejects_duplicate_extra_and_bad_contract():
    payload = valid_manifest_payload()
    drift.validate_run_manifest_payload(payload, check_files=False)
    duplicate = json.loads(json.dumps(payload)); duplicate["files"].append(dict(duplicate["files"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        drift.validate_run_manifest_payload(duplicate, check_files=False)
    extra = json.loads(json.dumps(payload)); extra["files"].append({"path": "unexpected.txt", "sha256": "0" * 64, "bytes": 0})
    with pytest.raises(ValueError, match="path-set"):
        drift.validate_run_manifest_payload(extra, check_files=False)
    bad = json.loads(json.dumps(payload)); bad["contract"]["episodes"] = 1
    with pytest.raises(ValueError, match="scientific contract"):
        drift.validate_run_manifest_payload(bad, check_files=False)


def test_new_run_must_be_empty_and_resume_requires_exact_allowlist(tmp_path: Path):
    output = tmp_path / "out"; output.mkdir(); (output / "stdout.log").write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match="truly empty"):
        drift.validate_output_start(output, False, "smoke", 1)
    with pytest.raises(FileExistsError, match="allowlist"):
        drift.validate_output_start(output, True, "smoke", 2)


def test_paired_summary_degenerate_all_ties_is_explicit_and_seeded():
    result = drift.paired_summary([0.0, 0.0, 0.0], "degenerate|ties")
    assert result["wins"] == 0 and result["ties"] == 3 and result["losses"] == 0
    assert result["raw_catalog_resample_vector"] == [0.0, 0.0, 0.0]
    assert result["bootstrap_reps"] == 20_000
    assert result["bootstrap_seed"] == drift.analysis_seed("degenerate|ties")
