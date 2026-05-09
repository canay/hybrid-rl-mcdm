"""
home_real_experiment.py
=======================

External validation branch on the public Amazon Home and Kitchen dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hybrid_core import (
    CONFIDENCE_LAMBDA_MAX,
    QAgent,
    TOP_K,
    confidence_gated_score,
    compute_significance,
    f1_score,
    ild_score,
    ndcg_at_k,
    norm,
    random_ranking,
    static_hybrid_score,
    summarize_nested,
    top_k_ranking,
    top_k_set,
    topsis_artifacts,
)
from mccauley_home_data import ITEMS_PATH, MANIFEST_PATH, PROC_DIR, USERS_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "mccauley_home_real_results.json"

EPOCH_CHECKPOINTS = [1, 2, 5, 10, 20, 30]
BOOTSTRAP_RUNS = 30
LAMBDA_GRID = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
N_EVAL_NEGATIVES = 200
TAU_REAL = 8.0
EPS_INIT_REAL = 0.30
EPS_DECAY_REAL = 0.998
EPS_MIN_REAL = 0.05


def load_processed() -> tuple[pd.DataFrame, list[dict], dict]:
    items = pd.read_csv(ITEMS_PATH)
    with USERS_PATH.open("r", encoding="utf-8") as handle:
        users = json.load(handle)
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return items, users, manifest


def hidden_utility_real(items: pd.DataFrame, profile: dict) -> np.ndarray:
    lvl2_aff = profile["cat_lvl2_affinity"]
    leaf_aff = profile["cat_leaf_affinity"]
    recency_pref = float(profile["recency_pref"])
    center = float(profile["log_price_center"])
    spread = float(profile["log_price_spread"])

    lvl2_score = np.array([lvl2_aff.get(str(x), 0.0) for x in items["cat_lvl2"]], dtype=float)
    leaf_score = np.array([leaf_aff.get(str(x), 0.0) for x in items["cat_leaf"]], dtype=float)
    if lvl2_score.max() > 0:
        lvl2_score = lvl2_score / lvl2_score.max()
    if leaf_score.max() > 0:
        leaf_score = leaf_score / leaf_score.max()

    price_log = np.log1p(items["price"].to_numpy(dtype=float))
    price_fit = np.clip(1.0 - np.abs(price_log - center) / spread, 0.0, 1.0)
    recency_score = items["recency_pct"].to_numpy(dtype=float) * recency_pref + (1.0 - recency_pref) * 0.5

    hidden = 0.45 * lvl2_score + 0.25 * leaf_score + 0.20 * price_fit + 0.10 * recency_score
    return norm(hidden)


def build_candidate_pool(items: pd.DataFrame, hidden_scores: np.ndarray, topsis_scores: np.ndarray, train_asins: list[str], asin_to_idx: dict[str, int]) -> np.ndarray:
    pool: set[int] = set(top_k_ranking(hidden_scores, k=250))
    pool.update(top_k_ranking(topsis_scores, k=120))
    pool.update(asin_to_idx[asin] for asin in train_asins if asin in asin_to_idx)
    return np.array(sorted(pool), dtype=int)


def bootstrap_summary(user_metrics: dict, checkpoints: list[int], methods: list[str]) -> dict:
    user_ids = list(user_metrics.keys())
    rng = np.random.RandomState(42)
    raw = {method: {cp: [] for cp in checkpoints} for method in methods}

    for _ in range(BOOTSTRAP_RUNS):
        sample_ids = [user_ids[idx] for idx in rng.randint(0, len(user_ids), size=len(user_ids))]
        for method in methods:
            for cp in checkpoints:
                raw[method][cp].append(float(np.mean([user_metrics[user_id]["f1"][method][cp] for user_id in sample_ids])))

    return summarize_nested(raw)


def bootstrap_final_metric(user_metrics: dict, metric_key: str, methods: list[str]) -> dict:
    user_ids = list(user_metrics.keys())
    rng = np.random.RandomState(123)
    raw = {method: [] for method in methods}
    for _ in range(BOOTSTRAP_RUNS):
        sample_ids = [user_ids[idx] for idx in rng.randint(0, len(user_ids), size=len(user_ids))]
        for method in methods:
            raw[method].append(float(np.mean([user_metrics[user_id][metric_key][method] for user_id in sample_ids])))
    return summarize_nested(raw)


def compute_significance_at_checkpoint(summary: dict, baselines: list[str], checkpoint: int) -> dict:
    last_checkpoint = str(checkpoint)
    hybrid = np.asarray(summary["hybrid"][last_checkpoint]["raw"], dtype=float)
    adjusted_alpha = 0.05 / len(baselines)
    results: dict = {}

    for baseline in baselines:
        baseline_values = np.asarray(summary[baseline][last_checkpoint]["raw"], dtype=float)
        diffs = hybrid - baseline_values
        diff_std = np.std(diffs, ddof=1)
        if diff_std <= 1e-12:
            t_stat = 0.0
            p_value = 1.0
            cohen_d = 0.0
        else:
            test = np.asarray(diffs, dtype=float)
            mean_diff = float(np.mean(test))
            se = float(np.std(test, ddof=1) / np.sqrt(len(test)))
            t_stat = mean_diff / se if se > 1e-12 else 0.0
            from scipy import stats
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df=len(test) - 1))
            cohen_d = float(mean_diff / diff_std)

        results[f"hybrid_vs_{baseline}"] = {
            "delta_f1": float(np.mean(diffs)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "bonferroni_alpha": float(adjusted_alpha),
            "significant": bool(p_value < adjusted_alpha and np.mean(diffs) > 0),
            "cohen_d": float(cohen_d),
        }

    return results


def sampled_eval_candidates(
    rng: np.random.RandomState,
    n_items: int,
    seen_idx: set[int],
    gt_set: set[int],
    n_negatives: int = N_EVAL_NEGATIVES,
) -> np.ndarray:
    all_idx = np.arange(n_items, dtype=int)
    banned = seen_idx | gt_set
    unseen = np.array([idx for idx in all_idx if idx not in banned], dtype=int)
    if len(unseen) <= n_negatives:
        negatives = unseen
    else:
        negatives = rng.choice(unseen, size=n_negatives, replace=False)
    return np.array(sorted(set(int(x) for x in negatives) | gt_set), dtype=int)


def rank_with_candidates(scores: np.ndarray, candidates: np.ndarray, k: int = TOP_K) -> list[int]:
    ranked_local = np.argsort(scores[candidates])[::-1][:k]
    return [int(candidates[idx]) for idx in ranked_local]


def run_external_validation() -> dict:
    run_started = time.perf_counter()
    items, users, manifest = load_processed()
    asin_to_idx = {asin: idx for idx, asin in enumerate(items["asin"])}
    n_items = len(items)
    methods = ["random", "popularity", "topsis_only", "rl_only", "hybrid"]

    topsis_scores = topsis_artifacts(items[["price_pct", "quality_pct", "popularity_pct", "rating_pct"]].assign(
        price_pct=items["price_pct"],
        quality_pct=items["quality_pct"],
        popularity_pct=items["popularity_pct"],
        rating_pct=items["rating_pct"],
    ))["scores"]
    popularity_rank = top_k_ranking(items["popularity_pct"].to_numpy(dtype=float))
    topsis_rank = top_k_ranking(topsis_scores)

    user_metrics: dict = {}

    print("=" * 72)
    print("Amazon Home and Kitchen external validation")
    print("=" * 72)
    print(f"Users: {len(users)}")
    print(f"Items: {n_items}")
    print(f"Epoch checkpoints: {EPOCH_CHECKPOINTS}")
    print("-" * 72)

    for user_index, user_payload in enumerate(users):
        user_id = str(user_payload["user_id"])
        train_asins = [row["asin"] for row in user_payload["train"]]
        gt_asins = [row["asin"] for row in user_payload["test"]]
        train_idx = {asin_to_idx[asin] for asin in train_asins if asin in asin_to_idx}
        gt_set = {asin_to_idx[asin] for asin in gt_asins if asin in asin_to_idx}
        hidden_scores = hidden_utility_real(items, user_payload["profile"])
        pool = build_candidate_pool(items, hidden_scores, topsis_scores, train_asins, asin_to_idx)

        agent = QAgent(
            n_products=n_items,
            n_profiles=1,
            seed=1000 + user_index,
            eps_init=EPS_INIT_REAL,
            eps_decay=EPS_DECAY_REAL,
            eps_min=EPS_MIN_REAL,
        )

        results = {method: {} for method in methods}
        eval_rng = np.random.RandomState(7000 + user_index)
        eval_candidates = sampled_eval_candidates(eval_rng, n_items, train_idx, gt_set)

        for epoch in range(1, max(EPOCH_CHECKPOINTS) + 1):
            for _ in range(len(train_asins)):
                action = agent.act(0, pool)
                reward = float(hidden_scores[action])
                agent.update(0, action, reward)

            if epoch in EPOCH_CHECKPOINTS:
                q_scores = agent.q_scores(0)
                visits = agent.visit_counts(0)
                rl_rank = rank_with_candidates(q_scores, eval_candidates)
                hybrid_rank = rank_with_candidates(static_hybrid_score(q_scores, topsis_scores, lambda_q=0.50), eval_candidates)
                random_rank = random_ranking(np.random.RandomState(5000 + user_index + epoch), n_products=len(eval_candidates))
                random_rank = [int(eval_candidates[idx]) for idx in random_rank]
                popularity_rank_eval = rank_with_candidates(items["popularity_pct"].to_numpy(dtype=float), eval_candidates)
                topsis_rank_eval = rank_with_candidates(topsis_scores, eval_candidates)

                results["random"][epoch] = f1_score(set(random_rank[:TOP_K]), gt_set)
                results["popularity"][epoch] = f1_score(set(popularity_rank_eval[:TOP_K]), gt_set)
                results["topsis_only"][epoch] = f1_score(set(topsis_rank_eval[:TOP_K]), gt_set)
                results["rl_only"][epoch] = f1_score(set(rl_rank[:TOP_K]), gt_set)
                results["hybrid"][epoch] = f1_score(set(hybrid_rank[:TOP_K]), gt_set)

        final_q = agent.q_scores(0)
        final_visits = agent.visit_counts(0)
        final_rl_rank = rank_with_candidates(final_q, eval_candidates)
        final_hybrid_rank = rank_with_candidates(static_hybrid_score(final_q, topsis_scores, lambda_q=0.50), eval_candidates)
        final_gated_rank = rank_with_candidates(
            confidence_gated_score(final_q, topsis_scores, final_visits, tau=TAU_REAL, lambda_max=CONFIDENCE_LAMBDA_MAX)[0],
            eval_candidates,
        )

        lambda_scores = {}
        for lam in LAMBDA_GRID:
            rank = rank_with_candidates(static_hybrid_score(final_q, topsis_scores, lambda_q=lam), eval_candidates)
            lambda_scores[str(lam)] = f1_score(set(rank[:TOP_K]), gt_set)

        ild_map = {
            "random": ild_score(items, set(random_rank[:TOP_K])),
            "popularity": ild_score(items, set(popularity_rank_eval[:TOP_K])),
            "topsis_only": ild_score(items, set(topsis_rank_eval[:TOP_K])),
            "rl_only": ild_score(items, set(final_rl_rank[:TOP_K])),
            "hybrid": ild_score(items, set(final_hybrid_rank[:TOP_K])),
        }

        ndcg_map = {
            "random": ndcg_at_k(random_rank[:TOP_K], gt_set),
            "popularity": ndcg_at_k(popularity_rank_eval[:TOP_K], gt_set),
            "topsis_only": ndcg_at_k(topsis_rank_eval[:TOP_K], gt_set),
            "rl_only": ndcg_at_k(final_rl_rank[:TOP_K], gt_set),
            "hybrid": ndcg_at_k(final_hybrid_rank[:TOP_K], gt_set),
        }

        gated_score, lambda_q_vec = confidence_gated_score(final_q, topsis_scores, final_visits, tau=TAU_REAL, lambda_max=CONFIDENCE_LAMBDA_MAX)
        user_metrics[user_id] = {
            "f1": results,
            "ndcg": ndcg_map,
            "ild": ild_map,
            "lambda_sensitivity": lambda_scores,
            "confidence_gated": {
                "static_f1": f1_score(set(final_hybrid_rank[:TOP_K]), gt_set),
                "gated_f1": f1_score(set(final_gated_rank[:TOP_K]), gt_set),
                "mean_lambda_q": float(np.mean(lambda_q_vec)),
            },
        }

        if (user_index + 1) % 100 == 0 or user_index == 0:
            final_h = user_metrics[user_id]["f1"]["hybrid"][max(EPOCH_CHECKPOINTS)]
            final_r = user_metrics[user_id]["f1"]["rl_only"][max(EPOCH_CHECKPOINTS)]
            print(f"user {user_index + 1}/{len(users)}: hybrid={final_h:.3f}, rl={final_r:.3f}")

    summary = bootstrap_summary(user_metrics, EPOCH_CHECKPOINTS, methods)
    significance = compute_significance_at_checkpoint(summary, baselines=["random", "popularity", "topsis_only", "rl_only"], checkpoint=max(EPOCH_CHECKPOINTS))
    ndcg_summary = bootstrap_final_metric(user_metrics, "ndcg", methods)
    ild_summary = bootstrap_final_metric(user_metrics, "ild", methods)

    lambda_raw = {str(lam): [] for lam in LAMBDA_GRID}
    cgf_raw = {"static": [], "gated": [], "mean_lambda_q": []}
    for user_id, payload in user_metrics.items():
        for lam in LAMBDA_GRID:
            lambda_raw[str(lam)].append(float(payload["lambda_sensitivity"][str(lam)]))
        cgf_raw["static"].append(float(payload["confidence_gated"]["static_f1"]))
        cgf_raw["gated"].append(float(payload["confidence_gated"]["gated_f1"]))
        cgf_raw["mean_lambda_q"].append(float(payload["confidence_gated"]["mean_lambda_q"]))

    report = {
        "config": {
            "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "n_users": len(users),
            "n_items": n_items,
            "epoch_checkpoints": EPOCH_CHECKPOINTS,
            "bootstrap_runs": BOOTSTRAP_RUNS,
            "n_eval_negatives": N_EVAL_NEGATIVES,
            "tau_real": TAU_REAL,
        },
        "summary": summary,
        "significance": significance,
        "secondary_metrics": {
            "ndcg_at_7": ndcg_summary,
            "ild": ild_summary,
        },
        "lambda_sensitivity": {
            "lambda_q_grid": LAMBDA_GRID,
            "summary": summarize_nested(lambda_raw),
        },
        "confidence_gated": {
            "tau": TAU_REAL,
            "lambda_max": CONFIDENCE_LAMBDA_MAX,
            "summary": summarize_nested(cgf_raw),
        },
        "notes": {
            "evaluation_design": (
                "External validation on a real public catalog with temporal positive-only holdout, "
                "seen-item masking, and sampled-candidate ranking. Behavioral reward is inferred "
                "from train-history preferences rather than from the synthetic GT construction."
            )
        },
        "runtime_seconds": {
            "total": float(time.perf_counter() - run_started),
        },
    }
    return report


def save_report(report: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def print_summary(report: dict) -> None:
    last_cp = str(max(EPOCH_CHECKPOINTS))
    print("=" * 72)
    print("External validation summary")
    print("=" * 72)
    for method in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]:
        stats_dict = report["summary"][method][last_cp]
        print(
            f"{method:12s}: F1={stats_dict['mean']:.4f} +/- {stats_dict['std']:.4f} "
            f"95%CI=[{stats_dict['ci_lo']:.4f}, {stats_dict['ci_hi']:.4f}]"
        )
    print(f"\nSaved external validation report to {OUTPUT_PATH}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-data external validation on Amazon Home and Kitchen.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)
    report = run_external_validation()
    save_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()

