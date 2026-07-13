from __future__ import annotations

import importlib.util
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


RUNNER = load(CAMPAIGN / "src" / "xai_main.py", "xai_runner_for_crosscheck")
VERIFIER = load(CAMPAIGN / "verify_xai.py", "xai_independent_verifier_tests")


def test_verifier_does_not_import_runner():
    source = (CAMPAIGN / "verify_xai.py").read_text(encoding="utf-8")
    assert "import xai_main" not in source
    assert "from xai_main" not in source
    assert "src.xai_main" not in source


def test_independent_extractor_matches_runner_on_verified_smoke():
    records = RUNNER.read_jsonl(SOURCE / "outputs" / "smoke" / "main_catalogs.jsonl")
    a = RUNNER.extract_allowlisted_inputs(records, SOURCE)
    b = VERIFIER.independent_source_extract(records, SOURCE)
    assert a == b
    VERIFIER.reject_denied_keys(b)


def test_independent_bound_source_uses_distinct_smoke_data_and_evidence_roots():
    records, _core, provenance, full = VERIFIER.validate_bound_source(
        SOURCE,
        SOURCE / "outputs" / "smoke",
        EVIDENCE,
        EVIDENCE / "outputs" / "smoke_overlay",
        "smoke",
    )
    assert len(records) == 1
    assert full["campaign_id"] == VERIFIER.SOURCE_EVIDENCE_CAMPAIGN_ID
    assert provenance["data_campaign_id"] == VERIFIER.SOURCE_CAMPAIGN_ID
    assert provenance["evidence_campaign_id"] == VERIFIER.SOURCE_EVIDENCE_CAMPAIGN_ID


def test_independent_guard_rejects_exact_gt_key():
    with pytest.raises(ValueError):
        VERIFIER.reject_denied_keys({"gt": [1]})


def test_independent_topsis_and_shapley_match_runner_to_1e12():
    record = RUNNER.read_jsonl(SOURCE / "outputs" / "smoke" / "main_catalogs.jsonl")[0]
    frame = pd.read_csv(SOURCE / "inputs" / Path(record["dataset_path"]))
    a = RUNNER.fixed_topsis_bundle(frame)
    b = VERIFIER.topsis_independent(frame)
    assert np.max(np.abs(a["raw_scores"] - b["raw"])) <= 1e-12
    assert np.max(np.abs(a["normalized_scores"] - b["normalized"])) <= 1e-12
    for item in (0, 137, 399):
        values = {mask: RUNNER.fixed_coalition_value(a["matrix"][item], mask, a) for mask in range(16)}
        runner_phi = RUNNER.exact_shapley_from_values(values, 4)
        verifier_base, verifier_phi, error = VERIFIER.shapley4(b["matrix"][item], b)
        assert abs(verifier_base - values[0]) <= 1e-12
        assert np.max(np.abs(runner_phi - verifier_phi)) <= 1e-12
        assert error <= 1e-12


def test_independent_reward_replay_matches_verified_smoke_budget():
    record = RUNNER.read_jsonl(SOURCE / "outputs" / "smoke" / "main_catalogs.jsonl")[0]
    extracted = RUNNER.extract_allowlisted_inputs([record], SOURCE)[0]
    core = RUNNER.load_module(SOURCE / "src" / "original_hybrid_core.py")
    frame = pd.read_csv(SOURCE / "inputs" / Path(extracted["dataset_path"]))
    replay = VERIFIER.replay_independent(frame, core.PROFILE_HIDDEN["budget"], 0, extracted["run_seed"])
    source = extracted["profiles"][0]
    assert np.max(np.abs(replay["q"] - np.asarray(source["q_scores"]))) <= 1e-12
    assert np.array_equal(replay["visits"], np.asarray(source["visits"], dtype=np.int32))
    assert np.max(np.abs(replay["q"] - replay["qb"] - replay["qe"] - replay["qc"])) <= 1e-12
    parts = VERIFIER.cq_parts_independent(replay)
    assert np.max(
        np.abs(parts["c_q"] - parts["reference"] - parts["base"] - parts["engage"] - parts["convert"])
    ) <= 1e-12


@pytest.mark.parametrize(
    "actual,expected",
    [([float("nan")], np.array([0.0])), ([0.0], np.array([float("inf")]))],
)
def test_independent_close_array_rejects_nonfinite(actual, expected):
    with pytest.raises(AssertionError, match="nonfinite"):
        VERIFIER.close_array(actual, expected, "negative")
