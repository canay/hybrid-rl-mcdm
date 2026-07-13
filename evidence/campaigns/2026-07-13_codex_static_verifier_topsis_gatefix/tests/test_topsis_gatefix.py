from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("static_verifier_overlay_topsis", CAMPAIGN / "verify_overlay.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)


def base_vectors() -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(0.0, 1.0, 400), np.asarray([0.25, 0.25, 0.25, 0.25])


def test_exact_full_order_top7_and_2e15_gate_passes():
    scores, weights = base_vectors()
    stored = scores.copy()
    stored[100] += 1.5e-15
    score_abs, weight_abs = VERIFY.verify_topsis_gate(
        VERIFY.Audit(), stored, scores, weights, weights, "positive"
    )
    assert score_abs <= VERIFY.SCORE_ABS_TOLERANCE
    assert weight_abs == 0.0


def test_plus_1e12_score_mutation_fails():
    scores, weights = base_vectors()
    mutated = scores.copy()
    mutated[100] += 1e-12
    with pytest.raises(VERIFY.VerificationError, match="absolute difference"):
        VERIFY.verify_topsis_gate(VERIFY.Audit(), mutated, scores, weights, weights, "+1e-12")


def test_top7_swap_fails_even_when_values_are_finite():
    scores, weights = base_vectors()
    mutated = scores.copy()
    mutated[-1], mutated[-2] = mutated[-2], mutated[-1]
    with pytest.raises(VERIFY.VerificationError, match="top-7 rank mismatch"):
        VERIFY.verify_topsis_gate(VERIFY.Audit(), mutated, scores, weights, weights, "top7-swap")


def test_full_order_only_swap_fails_even_when_top7_is_unchanged():
    scores, weights = base_vectors()
    mutated = scores.copy()
    mutated[10], mutated[11] = mutated[11], mutated[10]
    with pytest.raises(VERIFY.VerificationError, match="full-order rank mismatch"):
        VERIFY.verify_topsis_gate(VERIFY.Audit(), mutated, scores, weights, weights, "full-order-swap")


def test_near_value_top7_boundary_swap_fails_within_2e15():
    scores = np.linspace(0.0, 0.1, 400)
    for index, value in zip(range(399, 392, -1), (1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94)):
        scores[index] = value
    scores[392] = 0.94 - 1e-15
    mutated = scores.copy()
    mutated[393], mutated[392] = mutated[392], mutated[393]
    assert np.max(np.abs(mutated - scores)) <= VERIFY.SCORE_ABS_TOLERANCE
    with pytest.raises(VERIFY.VerificationError, match="top-7 rank mismatch"):
        VERIFY.verify_topsis_gate(
            VERIFY.Audit(), mutated, scores, np.full(4, 0.25), np.full(4, 0.25), "near-top7"
        )


def test_near_value_full_order_only_swap_fails_within_2e15():
    scores = np.linspace(0.0, 0.1, 400)
    for index, value in zip(range(399, 392, -1), (1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94)):
        scores[index] = value
    scores[10], scores[11] = 0.20, 0.20 - 1e-15
    mutated = scores.copy()
    mutated[10], mutated[11] = mutated[11], mutated[10]
    assert np.max(np.abs(mutated - scores)) <= VERIFY.SCORE_ABS_TOLERANCE
    assert VERIFY.top_rank(mutated) == VERIFY.top_rank(scores)
    with pytest.raises(VERIFY.VerificationError, match="full-order rank mismatch"):
        VERIFY.verify_topsis_gate(
            VERIFY.Audit(), mutated, scores, np.full(4, 0.25), np.full(4, 0.25), "near-full-order"
        )


@pytest.mark.parametrize("target", ["stored_score", "manual_score", "stored_weight", "manual_weight"])
def test_nonfinite_score_or_weight_fails(target: str):
    scores, weights = base_vectors()
    stored_s, manual_s = scores.copy(), scores.copy()
    stored_w, manual_w = weights.copy(), weights.copy()
    if target == "stored_score":
        stored_s[0] = np.nan
    elif target == "manual_score":
        manual_s[0] = np.inf
    elif target == "stored_weight":
        stored_w[0] = np.nan
    else:
        manual_w[0] = -np.inf
    with pytest.raises(VERIFY.VerificationError, match="Non-finite TOPSIS"):
        VERIFY.verify_topsis_gate(VERIFY.Audit(), stored_s, manual_s, stored_w, manual_w, target)


def test_weight_plus_1e12_fails():
    scores, weights = base_vectors()
    mutated = weights.copy()
    mutated[0] += 1e-12
    with pytest.raises(VERIFY.VerificationError, match="weight absolute difference"):
        VERIFY.verify_topsis_gate(VERIFY.Audit(), scores, scores, mutated, weights, "weight+1e-12")


def test_regression_evidence_is_exactly_the_locked_observation():
    manifest = VERIFY.strict_load(VERIFY.MANIFEST_PATH)
    original = VERIFY.strict_load(VERIFY.ORIGINAL_RESULT_PATH)
    evidence = VERIFY.recompute_topsis_regression_evidence(VERIFY.Audit(), manifest, original)
    assert evidence["producer_pandas_score_vectors_bit_exact"] == 50
    assert evidence["producer_pandas_weight_vectors_bit_exact"] == 50
    assert evidence["producer_profile_score_vectors_bit_exact"] == 250
    assert evidence["producer_profile_weight_vectors_bit_exact"] == 250
    assert evidence["manual_parser_global_score_max_abs"] == 1.1102230246251565e-15
    assert evidence["manual_parser_runs_score_abs_gt_1e_15"] == [29, 35]
    assert evidence["manual_parser_full_400_order_exact"] == 50
    assert evidence["manual_parser_top7_exact"] == 50


def test_source_smoke_tree_is_unchanged_by_direct_read_only_verification_contract():
    source = VERIFY.SOURCE_CAMPAIGN_ROOT / "outputs" / "smoke"
    before = VERIFY.snapshot_tree_hashes(source)
    VERIFY.verify_source_and_overlay_contract(VERIFY.Audit())
    after = VERIFY.snapshot_tree_hashes(source)
    assert after == before
