from __future__ import annotations

import importlib.util
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
MODULE_PATH = CAMPAIGN / "verify_bridge.py"
SPEC = importlib.util.spec_from_file_location("independent_bridge_verifier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "bridge_runner_test_only", CAMPAIGN / "src" / "bridge_main.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_arm_contract_is_independently_frozen():
    arms = VERIFY.expected_arms()
    assert len(arms) == 20
    assert sum(arm["role"] == "mandatory_factorial" for arm in arms) == 18
    assert len({arm["arm_id"] for arm in arms}) == 20
    assert VERIFY.EXACT_ARM_ID in {arm["arm_id"] for arm in arms}
    assert VERIFY.PRIMARY_CORRECTED_ARM_ID in {arm["arm_id"] for arm in arms}


def test_strict_json_rejects_nan_and_duplicate_keys():
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.strict_loads('{"x": NaN}')
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.strict_loads('{"x": 1, "x": 2}')


def test_raw_ranking_metrics_are_independent():
    truth = set(range(7))
    rank = [0, 1, 2, 7, 8, 9, 10]
    assert VERIFY.f1_at_7(rank, truth) == pytest.approx(3 / 7)
    expected_dcg = sum(1 / np.log2(index + 2) for index in range(3))
    expected_idcg = sum(1 / np.log2(index + 2) for index in range(7))
    assert VERIFY.ndcg_at_7(rank, truth) == pytest.approx(expected_dcg / expected_idcg)


def test_run_manifest_hash_and_size_gate(tmp_path: Path):
    (tmp_path / "a.txt").write_text("locked\n", encoding="utf-8")
    entry = {
        "path": "a.txt",
        "sha256": VERIFY.sha256_file(tmp_path / "a.txt"),
        "bytes": (tmp_path / "a.txt").stat().st_size,
    }
    payload = {
        "schema_version": "same_target_bridge.run_manifest.v1",
        "environment": {"packages": dict(VERIFY.EXPECTED_ENVIRONMENT)},
        "files": [entry],
    }
    (tmp_path / "RUN_MANIFEST.json").write_text(json.dumps(payload), encoding="utf-8")
    _, digest = VERIFY.verify_run_manifest(tmp_path, {"a.txt"})
    assert digest == VERIFY.sha256_file(tmp_path / "RUN_MANIFEST.json")
    entry["sha256"] = "0" * 64
    (tmp_path / "RUN_MANIFEST.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.verify_run_manifest(tmp_path, {"a.txt"})

    verification_dir = tmp_path / "verification-clean"
    verification_dir.mkdir()
    VERIFY.assert_verification_start_clean(verification_dir)
    (verification_dir / "verification_status.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="Stale verification artifacts"):
        VERIFY.assert_verification_start_clean(verification_dir)
    (verification_dir / "verification_status.json").unlink()
    (verification_dir / "FULL_VERIFICATION.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Stale verification artifacts"):
        VERIFY.assert_verification_start_clean(verification_dir)

    progress_dir = tmp_path / "verification-progress"
    progress_dir.mkdir()
    progress = VERIFY.VerificationProgress(
        output_dir=progress_dir,
        mode="smoke",
        catalogs_total=2,
        cells_total=200,
    )
    progress.start()
    progress.update(catalogs_completed=1, cells_completed=100)
    running = VERIFY.strict_load(progress.status_path)
    assert running["status"] == "running"
    assert running["progress_percent"] == pytest.approx(50.0)
    assert running["scientific_values_exposed"] is False
    report = progress_dir / "FULL_VERIFICATION.json"
    report.write_text('{"verdict":"PASS"}\n', encoding="utf-8")
    progress.complete(report)
    completed = VERIFY.strict_load(progress.status_path)
    assert completed["status"] == "completed_verified"
    assert completed["progress_percent"] == pytest.approx(100.0)
    assert completed["full_verification_sha256"] == VERIFY.sha256_file(report)


def test_candidate_support_is_derived_without_runner_import():
    n = 40
    data = {
        "n": n,
        "brand": np.asarray((["budget_brand", "mid_brand", "premium_brand", "mid_brand"] * 10), dtype=object),
        "category": np.asarray((["Electronics", "Computers", "HomeKitchen", "Electronics"] * 10), dtype=object),
        "price": np.linspace(10, 1000, n),
        "recency_pct": np.linspace(0, 1, n),
        "price_pct": np.linspace(0, 1, n),
        "quality_pct": np.linspace(1, 0, n),
        "popularity_pct": np.linspace(0.2, 0.8, n),
        "rating_pct": np.linspace(0.8, 0.2, n),
    }
    gt = set(range(7))
    hidden = VERIFY.expected_candidate(data, "budget", "hidden30_only", gt)
    oracle = VERIFY.expected_candidate(data, "budget", "oracle_gt_hidden30", gt)
    full = VERIFY.expected_candidate(data, "budget", "full_catalog", gt)
    assert len(hidden) == 30
    assert gt.issubset(set(oracle.tolist()))
    assert np.array_equal(full, np.arange(n))
    assert VERIFY.candidate_hash(hidden) == VERIFY.candidate_hash(hidden.copy())


def test_runner_source_contract_includes_historical_h_funnel():
    VERIFY.verify_runner_source_contract(CAMPAIGN / "src" / "bridge_main.py")
    verifier_source = (CAMPAIGN / "verify_bridge.py").read_text(encoding="utf-8")
    assert "import bridge_main" not in verifier_source
    assert "import original_hybrid_core" not in verifier_source


def test_independent_producer_environment_tag_and_replay_gate():
    audit = VERIFY.Audit()
    evidence = VERIFY.verify_producer_and_replay_provenance(audit)
    assert evidence["environment"] == VERIFY.EXPECTED_ENVIRONMENT
    assert evidence["producer_commit_sha1"] == VERIFY.PRODUCER_COMMIT_SHA1
    assert evidence["exact_replay_cells"] == 250
    assert audit.checks >= 8


def test_independent_statistics_recompute_and_reject_tampering():
    vector = np.linspace(-0.2, 0.3, VERIFY.EXPECTED_CATALOGS)
    expected = VERIFY.independent_paired_summary(
        vector, np.random.default_rng(VERIFY.ANALYSIS_SEED)
    )
    audit = VERIFY.Audit()
    VERIFY.verify_reported_vector(
        audit,
        expected,
        vector,
        "synthetic-paired",
        np.random.default_rng(VERIFY.ANALYSIS_SEED),
        paired=True,
    )
    assert audit.checks == 7
    for field in (
        "mean",
        "sample_sd",
        "bootstrap_ci95_lo",
        "bootstrap_ci95_hi",
        "paired_t_stat",
        "paired_t_p_two_sided",
        "cohen_dz",
        "wilcoxon_stat",
        "wilcoxon_p_two_sided",
        "wins",
        "ties",
        "losses",
    ):
        tampered = copy.deepcopy(expected)
        tampered[field] = tampered[field] + (
            1 if field in {"wins", "ties", "losses"} else 0.01
        )
        with pytest.raises(VERIFY.VerificationError):
            VERIFY.verify_reported_vector(
                VERIFY.Audit(),
                tampered,
                vector,
                f"tampered-{field}",
                np.random.default_rng(VERIFY.ANALYSIS_SEED),
                paired=True,
            )
    for field in (
        "paired_t_defined",
        "cohen_dz_defined",
        "wilcoxon_defined",
    ):
        tampered = copy.deepcopy(expected)
        tampered[field] = not tampered[field]
        with pytest.raises(VERIFY.VerificationError):
            VERIFY.verify_reported_vector(
                VERIFY.Audit(),
                tampered,
                vector,
                f"tampered-{field}",
                np.random.default_rng(VERIFY.ANALYSIS_SEED),
                paired=True,
            )


def test_independent_degenerate_statistics_match_null_policy():
    zero = np.zeros(VERIFY.EXPECTED_CATALOGS, dtype=float)
    reported = VERIFY.independent_paired_summary(
        zero, np.random.default_rng(VERIFY.ANALYSIS_SEED)
    )
    assert reported["paired_t_defined"] is False
    assert reported["paired_t_stat"] is None
    assert reported["cohen_dz_defined"] is False
    assert reported["cohen_dz"] is None
    assert reported["wilcoxon_defined"] is False
    assert reported["wilcoxon_stat"] is None
    assert reported["ties"] == VERIFY.EXPECTED_CATALOGS


def test_independent_corrected_stochastic_replay_exact_and_negative():
    manifest = json.loads(VERIFY.MANIFEST_PATH.read_text(encoding="utf-8"))
    original = json.loads(VERIFY.ORIGINAL_RESULT_PATH.read_text(encoding="utf-8"))
    meta = manifest["runs"][0]
    catalog_path = VERIFY.INPUT_ROOT / meta["path"]
    data = VERIFY.read_catalog(catalog_path)
    frame = pd.read_csv(catalog_path)
    stored = original["artifacts"][0]["profile_results"][0]
    profile_name = stored["profile_name"]
    profile_idx = VERIFY.PROFILE_ORDER.index(profile_name)
    gt_set = set(int(item) for item in stored["final"]["gt_set"])
    gt_rank = [int(item) for item in stored["final"]["gt_rank"]]
    independent_topsis, _ = VERIFY.independent_topsis(data)
    runner_topsis = RUNNER.CORE.topsis_artifacts(frame)["scores"]
    assert np.max(np.abs(independent_topsis - runner_topsis)) <= 1e-15
    assert VERIFY.top_rank(independent_topsis) == RUNNER.CORE.top_k_ranking(runner_topsis)
    arm = VERIFY.ARM_MAP[VERIFY.PRIMARY_CORRECTED_ARM_ID]
    produced = RUNNER.train_arm_profile(
        frame,
        profile_name,
        profile_idx,
        int(meta["seed"]),
        gt_set,
        gt_rank,
        runner_topsis,
        arm,
        VERIFY.EPISODES,
    )
    replayed = VERIFY.independent_train_profile(
        data,
        profile_name,
        profile_idx,
        int(meta["seed"]),
        gt_set,
        gt_rank,
        independent_topsis,
        arm,
        VERIFY.EPISODES,
    )
    assert produced == replayed
    topsis_rank = VERIFY.top_rank(independent_topsis)
    audit = VERIFY.Audit()
    VERIFY.verify_profile(
        audit,
        produced,
        arm,
        data,
        profile_name,
        int(meta["seed"]),
        independent_topsis,
        topsis_rank,
        stored,
        False,
    )
    tampered = copy.deepcopy(produced)
    tamper_index = int(np.argmin(np.asarray(tampered["q_scores"], dtype=float)))
    tampered["q_scores"][tamper_index] -= 1e-9
    with pytest.raises(VERIFY.VerificationError, match="Full stochastic replay Q mismatch"):
        VERIFY.verify_profile(
            VERIFY.Audit(),
            tampered,
            arm,
            data,
            profile_name,
            int(meta["seed"]),
            independent_topsis,
            topsis_rank,
            stored,
            False,
        )


def test_completed_smoke_if_present():
    candidates = (CAMPAIGN / "outputs" / "smoke", CAMPAIGN / "outputs" / "smoke_r01")
    output = next((path for path in candidates if (path / "main_results.json").exists()), None)
    if output is None or not (CAMPAIGN / "RUN_MANIFEST.json").exists() or not (CAMPAIGN / "PROTOCOL_LOCK.md").exists():
        pytest.skip("No terminal smoke fixture is available yet")
    report = VERIFY.verify_campaign(output, "smoke")
    assert report["verdict"] == "PASS"
    assert report["status"] == "completed_verified"
