from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
MODULE_PATH = CAMPAIGN / "src" / "bridge_main.py"
SPEC = importlib.util.spec_from_file_location("bridge_main_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def first_catalog():
    manifest = json.loads(BRIDGE.MANIFEST_PATH.read_text(encoding="utf-8"))
    meta = manifest["runs"][0]
    df = pd.read_csv(BRIDGE.INPUT_ROOT / meta["path"])
    original = json.loads(BRIDGE.ORIGINAL_RESULT_PATH.read_text(encoding="utf-8"))
    stored = original["artifacts"][0]["profile_results"][0]
    profile = BRIDGE.CORE.PROFILE_HIDDEN[stored["profile_name"]]
    gt = set(int(x) for x in stored["final"]["gt_set"])
    return df, profile, gt


def test_arm_contract_is_frozen():
    ids = [arm["arm_id"] for arm in BRIDGE.ARMS]
    assert len(BRIDGE.mandatory_arms()) == 18
    assert len(BRIDGE.sensitivity_arms()) == 2
    assert len(ids) == 20
    assert len(set(ids)) == len(ids)
    assert BRIDGE.EXACT_ARM_ID in ids
    assert BRIDGE.PRIMARY_CORRECTED_ARM_ID in ids


def test_frozen_input_hashes():
    assert BRIDGE.sha256_file(BRIDGE.CORE_PATH) == "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f"
    assert BRIDGE.sha256_file(BRIDGE.ORIGINAL_RESULT_PATH) == "cfeaff03084df0d3f0a07a5c8c40308027ca7980288a89cf3616c588d0791ce4"
    assert BRIDGE.HISTORICAL_V2_SHA256 == (
        "90AF7D4D3150099D840C510F5FF420B8773659C2A7A579D04E7B6E711DA65E4F"
    )
    assert BRIDGE.HISTORICAL_SUPPLEMENTARY_SHA256 == (
        "A18AF10D9D7C2C81E400910EC1D0DAE4071322DC1FC24AA0F3B6E022984D8BDC"
    )
    assert BRIDGE.assert_historical_reward_provenance() == {
        "hybrid_rl_mcdm_v2.py": BRIDGE.HISTORICAL_V2_SHA256,
        "supplementary_runs.py": BRIDGE.HISTORICAL_SUPPLEMENTARY_SHA256,
    }


def test_producer_environment_tag_and_exact_replay_provenance():
    assert BRIDGE.assert_exact_environment() == BRIDGE.EXPECTED_ENVIRONMENT
    producer = BRIDGE.assert_producer_provenance()
    assert producer["code/hybrid_core.py"] == BRIDGE.PRODUCER_FILE_SHA256["code/hybrid_core.py"]
    assert producer["code/run_amazon_experiments.py"] == BRIDGE.PRODUCER_FILE_SHA256["code/run_amazon_experiments.py"]
    assert producer["requirements.txt"] == BRIDGE.PRODUCER_FILE_SHA256["requirements.txt"]
    replay = BRIDGE.assert_exact_replay_provenance()
    assert replay == {
        "sha256": BRIDGE.EXACT_REPLAY_REPORT_SHA256,
        "cells_total": 250,
        "cells_exact": 250,
        "all_exact": True,
    }


def test_candidate_paths_withhold_gt_when_required():
    df, profile, gt = first_catalog()
    oracle = BRIDGE.candidate_pool(df, profile, "oracle_gt_hidden30", gt)
    hidden = BRIDGE.candidate_pool(df, profile, "hidden30_only", None)
    full = BRIDGE.candidate_pool(df, profile, "full_catalog", None)
    assert gt.issubset(set(int(x) for x in oracle))
    assert len(hidden) == 30
    assert np.array_equal(full, np.arange(len(df)))
    with pytest.raises(ValueError):
        BRIDGE.candidate_pool(df, profile, "hidden30_only", gt)
    with pytest.raises(ValueError):
        BRIDGE.candidate_pool(df, profile, "full_catalog", gt)


def test_reward_repairs_are_active_and_bounded():
    df, profile, _ = first_catalog()
    _, _, old = BRIDGE.reward_probabilities(df, profile, "implemented_r0")
    for model in ("inclusive_range_fix", "component_continuous_fix", "historical_funnel_coefficients_on_may_h"):
        engage, convert, diag = BRIDGE.reward_probabilities(df, profile, model)
        assert np.all(np.isfinite(engage)) and np.all(np.isfinite(convert))
        assert np.all((0 <= engage) & (engage <= 1))
        assert np.all((0 <= convert) & (convert <= 1))
        assert diag["category_distinct"] > 1
        assert diag["price_match_count"] > 0
    assert old["category_distinct"] == 1


def test_historical_h_funnel_matches_frozen_source_formula():
    df, profile, _ = first_catalog()
    hidden = BRIDGE.CORE.hidden_utility(df, profile)
    engage, convert, diag = BRIDGE.reward_probabilities(
        df, profile, "historical_funnel_coefficients_on_may_h"
    )
    assert np.array_equal(engage, np.clip(0.70 * hidden + 0.10, 0.05, 0.95))
    assert np.array_equal(convert, np.clip(0.50 * hidden, 0.02, 0.80))
    assert diag["historical_source_sha256"] == BRIDGE.HISTORICAL_REWARD_SOURCE_SHA256


def test_same_target_and_topsis_recompute_for_all_cells():
    manifest = json.loads(BRIDGE.MANIFEST_PATH.read_text(encoding="utf-8"))
    original = json.loads(BRIDGE.ORIGINAL_RESULT_PATH.read_text(encoding="utf-8"))
    profile_map = BRIDGE.original_profile_map(original)
    checked = 0
    for run_index, meta in enumerate(manifest["runs"]):
        path = BRIDGE.INPUT_ROOT / meta["path"]
        assert BRIDGE.sha256_file(path) == meta["sha256"]
        df = pd.read_csv(path)
        topsis = BRIDGE.CORE.top_k_ranking(BRIDGE.CORE.topsis_artifacts(df)["scores"])
        for profile_name in BRIDGE.CORE.PROFILE_ORDER:
            stored = profile_map[(run_index, profile_name)]["final"]
            gt = [int(x) for x in stored["gt_rank"]]
            assert len(gt) == 7 and len(set(gt)) == 7
            assert topsis == [int(x) for x in stored["topsis_rank"]]
            checked += 1
    assert checked == 250


def test_paired_summary_degenerate_policy_is_json_safe():
    zero = np.zeros(50, dtype=float)
    all_ties = BRIDGE.paired_summary(
        zero, zero, np.random.default_rng(BRIDGE.ANALYSIS_SEED)
    )
    assert all_ties["paired_t_defined"] is False
    assert all_ties["paired_t_stat"] is None
    assert all_ties["paired_t_p_two_sided"] is None
    assert all_ties["cohen_dz_defined"] is False
    assert all_ties["cohen_dz"] is None
    assert all_ties["wilcoxon_defined"] is False
    assert all_ties["wilcoxon_stat"] is None
    assert all_ties["wilcoxon_p_two_sided"] is None
    assert (all_ties["wins"], all_ties["ties"], all_ties["losses"]) == (0, 50, 0)

    constant = BRIDGE.paired_summary(
        np.full(50, 0.1),
        zero,
        np.random.default_rng(BRIDGE.ANALYSIS_SEED),
    )
    assert constant["paired_t_defined"] is False
    assert constant["cohen_dz_defined"] is False
    assert constant["wilcoxon_defined"] is True
    assert constant["wilcoxon_stat"] == pytest.approx(0.0)
    assert 0.0 <= constant["wilcoxon_p_two_sided"] <= 1.0
    assert (constant["wins"], constant["ties"], constant["losses"]) == (50, 0, 0)
    BRIDGE.assert_json_finite(all_ties)
    BRIDGE.assert_json_finite(constant)


def test_nonfinite_analysis_values_fail_before_write(tmp_path: Path):
    with pytest.raises(ValueError, match="50 finite catalog values"):
        BRIDGE.summarize_vector(
            np.r_[np.zeros(49), np.nan],
            np.random.default_rng(BRIDGE.ANALYSIS_SEED),
        )
    target = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="Non-finite JSON number"):
        BRIDGE.atomic_json(target, {"bad": float("nan")})
    assert not target.exists()
    jsonl = tmp_path / "nonfinite.jsonl"
    with pytest.raises(ValueError, match="Non-finite JSON number"):
        BRIDGE.append_jsonl(jsonl, {"bad": float("inf")})
    assert not jsonl.exists()

    stale = tmp_path / "stale-output"
    stale.mkdir()
    (stale / "arbitrary.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="truly empty"):
        BRIDGE.prepare_output_directory(stale, resume=False)
    full_verification = tmp_path / "full-verification-output"
    full_verification.mkdir()
    (full_verification / "FULL_VERIFICATION.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="truly empty"):
        BRIDGE.prepare_output_directory(full_verification, resume=False)

    empty = tmp_path / "empty-output"
    BRIDGE.prepare_output_directory(empty, resume=False)
    assert empty.is_dir() and not any(empty.iterdir())
    resume = tmp_path / "resume-output"
    resume.mkdir()
    (resume / "main_catalogs.jsonl").write_text("{}\n", encoding="utf-8")
    (resume / "status.json").write_text("{}", encoding="utf-8")
    BRIDGE.prepare_output_directory(resume, resume=True)
    (resume / "verification_status.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="stale/unexpected"):
        BRIDGE.prepare_output_directory(resume, resume=True)
