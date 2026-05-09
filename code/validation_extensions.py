"""Additional Hybrid RL-TOPSIS validation experiments.

The script adds three robustness checks:

1. LinUCB as a lightweight CF-free contextual-bandit baseline.
2. Catalog-size sensitivity for the tabular policy.
3. Gradual multi-dimensional drift rather than only a sudden brand flip.

The experiments reuse the Hybrid RL-TOPSIS Amazon India preprocessing and the same explicit
profile-level feedback model as the primary paper experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from hybrid_core import (
    DRIFT_EPISODE,
    MAIN_ALPHA,
    MAIN_CHECKPOINTS,
    MAIN_LAMBDA_Q,
    MAIN_LAMBDA_T,
    N_PROFILES,
    PROFILE_HIDDEN,
    PROFILE_ORDER,
    REWARD_SHAPING_BONUS,
    TOP_K,
    build_candidate_pool,
    build_ground_truth,
    f1_score,
    flip_brand_preferences,
    norm,
    popularity_recs,
    profile_seed,
    random_recs,
    static_hybrid_score,
    summarize_nested,
    top_k_set,
    topsis_artifacts,
)
from run_amazon_experiments import (
    PROJECT_ROOT,
    load_enriched_catalog,
    load_products_for_run,
    reward_arrays,
    run_seed,
    stratified_sample,
)


RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
OUT_JSON = RESULTS_DIR / "validation_extensions.json"
OUT_CSV = RESULTS_DIR / "validation_extensions_summary.csv"

DRIFT_CHECKPOINTS_EXT = [5000, 10000, 15000, 20000, 25000, 30000]
DRIFT_START = 10000
DRIFT_END = 25000


def item_context_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df[["price_pct", "quality_pct", "popularity_pct", "rating_pct", "recency_pct"]].to_numpy(dtype=float)
    categories = pd.get_dummies(df["category"], prefix="cat", dtype=float)
    brands = pd.get_dummies(df["brand"], prefix="brand", dtype=float)
    x = np.hstack([numeric, categories.to_numpy(dtype=float), brands.to_numpy(dtype=float)])
    return np.hstack([np.ones((len(df), 1), dtype=float), x])


class LinUCB:
    def __init__(self, n_features: int, alpha: float = 0.60, ridge: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.a_inv = (1.0 / ridge) * np.eye(n_features, dtype=float)
        self.b = np.zeros(n_features, dtype=float)

    def theta(self) -> np.ndarray:
        return self.a_inv @ self.b

    def select(self, x: np.ndarray, pool: np.ndarray) -> int:
        theta = self.theta()
        x_pool = x[pool]
        mean = x_pool @ theta
        unc = np.sqrt(np.einsum("ij,jk,ik->i", x_pool, self.a_inv, x_pool))
        return int(pool[int(np.argmax(mean + self.alpha * unc))])

    def update(self, x_i: np.ndarray, reward: float) -> None:
        ax = self.a_inv @ x_i
        denom = 1.0 + float(x_i @ ax)
        self.a_inv -= np.outer(ax, ax) / denom
        self.b += reward * x_i

    def score(self, x: np.ndarray) -> np.ndarray:
        return x @ self.theta()


def sampled_reward(
    action: int,
    p_engage: np.ndarray,
    p_convert: np.ndarray,
    gt_mask: np.ndarray,
    rng: np.random.RandomState,
    shaping_bonus: float = REWARD_SHAPING_BONUS,
) -> float:
    reward = -0.02
    if rng.random() < p_engage[action]:
        reward += 0.30
        if rng.random() < p_convert[action]:
            reward += 1.00
    if gt_mask[action]:
        reward += shaping_bonus
    return float(reward)


def train_profile_with_linucb(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    seed: int,
    topsis_scores: np.ndarray,
    checkpoints: Sequence[int],
) -> dict:
    profile = PROFILE_HIDDEN[profile_name]
    gt_scores = build_ground_truth(df, profile, profile_seed(seed, profile_name), observable_alpha=MAIN_ALPHA)
    gt_set = top_k_set(gt_scores)
    pool = build_candidate_pool(df, [profile], [gt_set])

    n_products = len(df)
    q_scores = np.zeros(n_products, dtype=float)
    eps = 0.30
    eps_decay = 0.9997
    eps_min = 0.05
    lr = 0.05

    rng_q = np.random.RandomState(seed + profile_idx * 997)
    rng_lin = np.random.RandomState(seed + profile_idx * 1777)
    act_rng = np.random.RandomState(seed + profile_idx * 13)

    p_engage, p_convert = reward_arrays(df, profile)
    gt_mask = np.zeros(n_products, dtype=bool)
    gt_mask[list(gt_set)] = True

    x = item_context_matrix(df)
    linucb = LinUCB(x.shape[1])
    random_set = random_recs(np.random.RandomState(seed + 5555), n_products=n_products)
    popularity_set = popularity_recs(df)
    topsis_set = top_k_set(topsis_scores)
    topsis_norm = norm(topsis_scores)

    raw = {m: {} for m in ["random", "popularity", "topsis_only", "rl_only", "linucb", "hybrid"]}
    checkpoint_set = set(int(cp) for cp in checkpoints)

    for episode in range(1, max(checkpoints) + 1):
        if act_rng.random() < eps:
            q_action = int(act_rng.choice(pool))
        else:
            q_action = int(pool[int(np.argmax(q_scores[pool]))])
        q_reward = sampled_reward(q_action, p_engage, p_convert, gt_mask, rng_q)
        q_scores[q_action] += lr * (q_reward - q_scores[q_action])
        eps = max(eps_min, eps * eps_decay)

        lin_action = linucb.select(x, pool)
        lin_reward = sampled_reward(lin_action, p_engage, p_convert, gt_mask, rng_lin)
        linucb.update(x[lin_action], lin_reward)

        if episode in checkpoint_set:
            hybrid_scores = MAIN_LAMBDA_Q * norm(q_scores) + MAIN_LAMBDA_T * topsis_norm
            lin_scores = linucb.score(x)
            raw["random"][episode] = f1_score(random_set, gt_set)
            raw["popularity"][episode] = f1_score(popularity_set, gt_set)
            raw["topsis_only"][episode] = f1_score(topsis_set, gt_set)
            raw["rl_only"][episode] = f1_score(top_k_set(q_scores), gt_set)
            raw["linucb"][episode] = f1_score(top_k_set(lin_scores), gt_set)
            raw["hybrid"][episode] = f1_score(top_k_set(hybrid_scores), gt_set)

    return raw


def train_profile_q_only(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    seed: int,
    topsis_scores: np.ndarray,
    checkpoints: Sequence[int],
) -> dict:
    profile = PROFILE_HIDDEN[profile_name]
    gt_scores = build_ground_truth(df, profile, profile_seed(seed, profile_name), observable_alpha=MAIN_ALPHA)
    gt_set = top_k_set(gt_scores)
    pool = build_candidate_pool(df, [profile], [gt_set])

    n_products = len(df)
    q_scores = np.zeros(n_products, dtype=float)
    eps = 0.30
    eps_decay = 0.9997
    eps_min = 0.05
    lr = 0.05

    rng = np.random.RandomState(seed + profile_idx * 997)
    act_rng = np.random.RandomState(seed + profile_idx * 13)
    p_engage, p_convert = reward_arrays(df, profile)
    gt_mask = np.zeros(n_products, dtype=bool)
    gt_mask[list(gt_set)] = True

    random_set = random_recs(np.random.RandomState(seed + 5555), n_products=n_products)
    popularity_set = popularity_recs(df)
    topsis_set = top_k_set(topsis_scores)
    topsis_norm = norm(topsis_scores)
    raw = {m: {} for m in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]}
    checkpoint_set = set(int(cp) for cp in checkpoints)

    for episode in range(1, max(checkpoints) + 1):
        if act_rng.random() < eps:
            action = int(act_rng.choice(pool))
        else:
            action = int(pool[int(np.argmax(q_scores[pool]))])
        reward = sampled_reward(action, p_engage, p_convert, gt_mask, rng)
        q_scores[action] += lr * (reward - q_scores[action])
        eps = max(eps_min, eps * eps_decay)

        if episode in checkpoint_set:
            hybrid_scores = MAIN_LAMBDA_Q * norm(q_scores) + MAIN_LAMBDA_T * topsis_norm
            raw["random"][episode] = f1_score(random_set, gt_set)
            raw["popularity"][episode] = f1_score(popularity_set, gt_set)
            raw["topsis_only"][episode] = f1_score(topsis_set, gt_set)
            raw["rl_only"][episode] = f1_score(top_k_set(q_scores), gt_set)
            raw["hybrid"][episode] = f1_score(top_k_set(hybrid_scores), gt_set)

    return raw


def shifted_profile(profile: Mapping[str, object]) -> dict:
    shifted = flip_brand_preferences(profile)

    cat = shifted["cat_affinity"]
    ranked = sorted(cat.items(), key=lambda item: item[1])
    shifted["cat_affinity"] = {ranked[0][0]: ranked[2][1], ranked[1][0]: ranked[1][1], ranked[2][0]: ranked[0][1]}

    lo, hi = profile["price_range"]
    width = hi - lo
    center = (lo + hi) / 2.0
    new_center = center + 250.0 if center < 350.0 else center - 250.0
    new_lo = max(10.0, new_center - width / 2.0)
    new_hi = min(1000.0, new_center + width / 2.0)
    shifted["price_range"] = (new_lo, new_hi)
    shifted["recency_weight"] = float(np.clip(1.0 - float(profile["recency_weight"]), 0.10, 0.90))
    return shifted


def interpolate_profile(pre: Mapping[str, object], post: Mapping[str, object], frac: float) -> dict:
    frac = float(np.clip(frac, 0.0, 1.0))
    brand_keys = sorted(set(pre["brand_pref"]) | set(post["brand_pref"]))
    cat_keys = sorted(set(pre["cat_affinity"]) | set(post["cat_affinity"]))
    pre_lo, pre_hi = pre["price_range"]
    post_lo, post_hi = post["price_range"]
    return {
        "brand_pref": {
            key: (1.0 - frac) * float(pre["brand_pref"].get(key, 0.0)) + frac * float(post["brand_pref"].get(key, 0.0))
            for key in brand_keys
        },
        "cat_affinity": {
            key: (1.0 - frac) * float(pre["cat_affinity"].get(key, 0.0)) + frac * float(post["cat_affinity"].get(key, 0.0))
            for key in cat_keys
        },
        "price_range": ((1.0 - frac) * pre_lo + frac * post_lo, (1.0 - frac) * pre_hi + frac * post_hi),
        "recency_weight": (1.0 - frac) * float(pre["recency_weight"]) + frac * float(post["recency_weight"]),
    }


def drift_fraction(episode: int) -> float:
    if episode <= DRIFT_START:
        return 0.0
    if episode >= DRIFT_END:
        return 1.0
    return (episode - DRIFT_START) / (DRIFT_END - DRIFT_START)


def train_gradual_drift_profile(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    seed: int,
    topsis_scores: np.ndarray,
) -> dict:
    pre = PROFILE_HIDDEN[profile_name]
    post = shifted_profile(pre)
    n_products = len(df)
    topsis_norm = norm(topsis_scores)
    topsis_set = top_k_set(topsis_scores)

    q_scores = np.zeros(n_products, dtype=float)
    eps = 0.30
    eps_decay = 0.9997
    eps_min = 0.05
    lr = 0.05
    rng_q = np.random.RandomState(seed + profile_idx * 2221)
    rng_lin = np.random.RandomState(seed + profile_idx * 3331)
    act_rng = np.random.RandomState(seed + profile_idx * 4441)

    pre_gt = top_k_set(build_ground_truth(df, pre, profile_seed(seed, profile_name), observable_alpha=MAIN_ALPHA))
    post_gt = top_k_set(build_ground_truth(df, post, profile_seed(seed, profile_name) + 5000, observable_alpha=MAIN_ALPHA))
    pool = build_candidate_pool(df, [pre, post], [pre_gt, post_gt])

    raw = {m: {} for m in ["topsis_only", "rl_only", "hybrid"]}
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]] = {}

    def active_payload(episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
        # Cache phase values at 2.5 percentage-point drift increments.
        key = int(round(drift_fraction(episode) * 40))
        if key not in cache:
            frac = key / 40.0
            profile = interpolate_profile(pre, post, frac)
            gt_seed = profile_seed(seed, profile_name) + 7000 + key
            gt_set = top_k_set(build_ground_truth(df, profile, gt_seed, observable_alpha=MAIN_ALPHA))
            p_engage, p_convert = reward_arrays(df, profile)
            gt_mask = np.zeros(n_products, dtype=bool)
            gt_mask[list(gt_set)] = True
            cache[key] = (p_engage, p_convert, gt_mask, gt_set)
        return cache[key]

    for episode in range(1, max(DRIFT_CHECKPOINTS_EXT) + 1):
        p_engage, p_convert, gt_mask, gt_set = active_payload(episode)

        if act_rng.random() < eps:
            q_action = int(act_rng.choice(pool))
        else:
            q_action = int(pool[int(np.argmax(q_scores[pool]))])
        q_reward = sampled_reward(q_action, p_engage, p_convert, gt_mask, rng_q)
        q_scores[q_action] += lr * (q_reward - q_scores[q_action])
        eps = max(eps_min, eps * eps_decay)

        if episode in DRIFT_CHECKPOINTS_EXT:
            _, _, _, eval_gt = active_payload(episode)
            hybrid_scores = MAIN_LAMBDA_Q * norm(q_scores) + MAIN_LAMBDA_T * topsis_norm
            raw["topsis_only"][episode] = f1_score(topsis_set, eval_gt)
            raw["rl_only"][episode] = f1_score(top_k_set(q_scores), eval_gt)
            raw["hybrid"][episode] = f1_score(top_k_set(hybrid_scores), eval_gt)

    return raw


def run_linucb_baseline(runs: int) -> dict:
    raw = {m: {cp: [] for cp in MAIN_CHECKPOINTS} for m in ["random", "popularity", "topsis_only", "rl_only", "linucb", "hybrid"]}
    for run_index in range(runs):
        seed = run_seed(run_index)
        df = load_products_for_run(run_index)
        topsis_scores = topsis_artifacts(df)["scores"]
        per_run = {m: {cp: [] for cp in MAIN_CHECKPOINTS} for m in raw}
        for profile_idx, profile_name in enumerate(PROFILE_ORDER):
            outcome = train_profile_with_linucb(df, profile_name, profile_idx, seed, topsis_scores, MAIN_CHECKPOINTS)
            for method in raw:
                for cp in MAIN_CHECKPOINTS:
                    per_run[method][cp].append(outcome[method][cp])
        for method in raw:
            for cp in MAIN_CHECKPOINTS:
                raw[method][cp].append(float(np.mean(per_run[method][cp])))
        if (run_index + 1) % 5 == 0 or run_index == 0:
            print(f"LinUCB run {run_index + 1}/{runs}: hybrid={raw['hybrid'][30000][-1]:.3f}, linucb={raw['linucb'][30000][-1]:.3f}")
    return {"runs": runs, "checkpoints": MAIN_CHECKPOINTS, "summary": summarize_nested(raw)}


def run_catalog_size_sensitivity(runs: int, sizes: Sequence[int]) -> dict:
    enriched = load_enriched_catalog()
    raw = {
        int(size): {m: [] for m in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]}
        for size in sizes
    }
    for size in sizes:
        for run_index in range(runs):
            seed = run_seed(run_index) + int(size) * 19
            df = stratified_sample(enriched, seed, n_products=int(size))
            topsis_scores = topsis_artifacts(df)["scores"]
            per_profile = {m: [] for m in raw[int(size)]}
            for profile_idx, profile_name in enumerate(PROFILE_ORDER):
                outcome = train_profile_q_only(df, profile_name, profile_idx, seed, topsis_scores, [MAIN_CHECKPOINTS[-1]])
                for method in per_profile:
                    per_profile[method].append(outcome[method][MAIN_CHECKPOINTS[-1]])
            for method in raw[int(size)]:
                raw[int(size)][method].append(float(np.mean(per_profile[method])))
        print(f"Catalog size {size}: hybrid={np.mean(raw[int(size)]['hybrid']):.3f}, rl={np.mean(raw[int(size)]['rl_only']):.3f}")
    return {"runs_per_size": runs, "sizes": [int(s) for s in sizes], "summary": summarize_nested(raw)}


def run_gradual_drift(runs: int) -> dict:
    raw = {m: {cp: [] for cp in DRIFT_CHECKPOINTS_EXT} for m in ["topsis_only", "rl_only", "hybrid"]}
    for run_index in range(runs):
        seed = run_seed(run_index)
        df = load_products_for_run(run_index)
        topsis_scores = topsis_artifacts(df)["scores"]
        per_run = {m: {cp: [] for cp in DRIFT_CHECKPOINTS_EXT} for m in raw}
        for profile_idx, profile_name in enumerate(PROFILE_ORDER):
            outcome = train_gradual_drift_profile(df, profile_name, profile_idx, seed, topsis_scores)
            for method in raw:
                for cp in DRIFT_CHECKPOINTS_EXT:
                    per_run[method][cp].append(outcome[method][cp])
        for method in raw:
            for cp in DRIFT_CHECKPOINTS_EXT:
                raw[method][cp].append(float(np.mean(per_run[method][cp])))
        if (run_index + 1) % 5 == 0 or run_index == 0:
            print(f"Gradual drift run {run_index + 1}/{runs}: hybrid={raw['hybrid'][30000][-1]:.3f}, rl={raw['rl_only'][30000][-1]:.3f}")
    return {
        "runs": runs,
        "drift_start": DRIFT_START,
        "drift_end": DRIFT_END,
        "reference_sudden_drift_episode": DRIFT_EPISODE,
        "checkpoints": DRIFT_CHECKPOINTS_EXT,
        "summary": summarize_nested(raw),
    }


def save_csv(report: dict) -> None:
    rows = []
    lin = report["linucb_baseline"]["summary"]
    for method, by_cp in lin.items():
        row = {
            "experiment": "linucb_baseline",
            "group": "main_400",
            "method": method,
            "checkpoint": 30000,
            "mean_f1": by_cp["30000"]["mean"],
            "std_f1": by_cp["30000"]["std"],
            "ci_lo": by_cp["30000"]["ci_lo"],
            "ci_hi": by_cp["30000"]["ci_hi"],
        }
        rows.append(row)

    for size, methods in report["catalog_size_sensitivity"]["summary"].items():
        for method, stats in methods.items():
            rows.append(
                {
                    "experiment": "catalog_size",
                    "group": f"n_items_{size}",
                    "method": method,
                    "checkpoint": 30000,
                    "mean_f1": stats["mean"],
                    "std_f1": stats["std"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                }
            )

    for method, by_cp in report["gradual_multidim_drift"]["summary"].items():
        for cp, stats in by_cp.items():
            rows.append(
                {
                    "experiment": "gradual_multidim_drift",
                    "group": "main_400",
                    "method": method,
                    "checkpoint": int(cp),
                    "mean_f1": stats["mean"],
                    "std_f1": stats["std"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                }
            )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_test_from_stats(a_stats: Mapping[str, object], b_stats: Mapping[str, object]) -> dict:
    a = np.asarray(a_stats["raw"], dtype=float)
    b = np.asarray(b_stats["raw"], dtype=float)
    diff = a - b
    t_test = stats.ttest_rel(a, b)
    try:
        wilcoxon = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
        wilcoxon_p = float(wilcoxon.pvalue)
        wilcoxon_stat = float(wilcoxon.statistic)
    except ValueError:
        wilcoxon_p = 1.0
        wilcoxon_stat = 0.0
    return {
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "paired_t_p_two_sided": float(t_test.pvalue),
        "wilcoxon_p_one_sided_greater": wilcoxon_p,
        "wilcoxon_stat": wilcoxon_stat,
        "wins": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "losses": int(np.sum(diff < 0)),
        "n": int(diff.size),
    }


def add_paired_tests(report: dict) -> None:
    lin = report["linucb_baseline"]["summary"]
    report["linucb_baseline"]["paired_tests_at_30000"] = {
        baseline: paired_test_from_stats(lin["hybrid"]["30000"], lin[baseline]["30000"])
        for baseline in ["linucb", "rl_only", "topsis_only", "popularity", "random"]
    }

    size_tests = {}
    for size, methods in report["catalog_size_sensitivity"]["summary"].items():
        size_tests[size] = {
            baseline: paired_test_from_stats(methods["hybrid"], methods[baseline])
            for baseline in ["rl_only", "topsis_only", "popularity", "random"]
        }
    report["catalog_size_sensitivity"]["paired_tests_at_30000"] = size_tests

    drift = report["gradual_multidim_drift"]["summary"]
    report["gradual_multidim_drift"]["paired_tests_at_30000"] = {
        baseline: paired_test_from_stats(drift["hybrid"]["30000"], drift[baseline]["30000"])
        for baseline in ["rl_only", "topsis_only"]
    }


def run_all(runs: int, size_runs: int, sizes: Sequence[int]) -> dict:
    started = time.perf_counter()
    print("=" * 72)
    print("Hybrid RL-TOPSIS additional validation extensions")
    print("=" * 72)
    print(f"Main runs: {runs}; catalog-size runs: {size_runs}; sizes: {list(sizes)}")
    print("-" * 72)
    report = {
        "config": {
            "main_runs": int(runs),
            "catalog_size_runs": int(size_runs),
            "catalog_sizes": [int(x) for x in sizes],
            "linucb_alpha": 0.60,
            "top_k": TOP_K,
            "note": "Additional validation experiments; primary claims remain based on amazon_primary.json.",
        },
        "linucb_baseline": run_linucb_baseline(runs),
        "catalog_size_sensitivity": run_catalog_size_sensitivity(size_runs, sizes),
        "gradual_multidim_drift": run_gradual_drift(runs),
    }
    report["runtime_seconds"] = float(time.perf_counter() - started)
    add_paired_tests(report)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    save_csv(report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run additional Hybrid RL-TOPSIS validation extensions.")
    parser.add_argument("--runs", type=int, default=30, help="Runs for LinUCB and gradual drift.")
    parser.add_argument("--size-runs", type=int, default=30, help="Runs for each catalog size.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800], help="Catalog sizes to evaluate.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_all(args.runs, args.size_runs, args.sizes)
    print("=" * 72)
    print("Validation extension summary")
    print("=" * 72)
    last = "30000"
    lin = report["linucb_baseline"]["summary"]
    print(f"LinUCB baseline at 30K: hybrid={lin['hybrid'][last]['mean']:.4f}, linucb={lin['linucb'][last]['mean']:.4f}, rl={lin['rl_only'][last]['mean']:.4f}")
    drift = report["gradual_multidim_drift"]["summary"]
    print(f"Gradual drift final: hybrid={drift['hybrid'][last]['mean']:.4f}, rl={drift['rl_only'][last]['mean']:.4f}")
    for size, methods in report["catalog_size_sensitivity"]["summary"].items():
        print(f"Size {size}: hybrid={methods['hybrid']['mean']:.4f}, rl={methods['rl_only']['mean']:.4f}")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()


