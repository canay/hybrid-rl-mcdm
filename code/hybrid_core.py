"""
hybrid_core.py
==============

Shared experiment logic for the Hybrid RL-TOPSIS manuscript.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


MAIN_CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 20000, 30000]
DRIFT_CHECKPOINTS = [2000, 5000, 10000, 14000, 15000, 16000, 20000, 25000, 30000]

TOP_K = 7
N_PROFILES = 5
N_EPISODES = 30000
DRIFT_EPISODE = 15000
N_BOOTSTRAP_RUNS = 30
N_BOOTSTRAP_DRIFT = 30

PROFILE_ORDER = ["budget", "quality_seeker", "explorer", "loyal", "balanced"]
PROFILE_SEED_OFFSETS = {name: 101 + idx * 97 for idx, name in enumerate(PROFILE_ORDER)}

MAIN_ALPHA = 0.50
MAIN_LAMBDA_Q = 0.50
MAIN_LAMBDA_T = 0.50
GT_NOISE_SD = 0.015

LAMBDA_GRID = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
GT_SPLIT_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

CONFIDENCE_TAU = 300.0
CONFIDENCE_LAMBDA_MAX = 0.50
REWARD_SHAPING_BONUS = 0.20

LEARNING_RATE = 0.05
EPS_INIT = 0.30
EPS_DECAY = 0.9997
EPS_MIN = 0.05

FEATURE_COLUMNS = ["price_pct", "quality_pct", "popularity_pct", "rating_pct", "recency_pct"]
TOPSIS_COLUMNS = ["price_pct", "quality_pct", "popularity_pct", "rating_pct"]
CRITERION_LABELS = {
    "price_pct": "Price",
    "quality_pct": "Quality",
    "popularity_pct": "Popularity",
    "rating_pct": "Rating",
}

PROFILE_HIDDEN: Dict[str, Dict[str, object]] = {
    "budget": {
        "brand_pref": {"budget_brand": 0.80, "mid_brand": 0.15, "premium_brand": 0.05},
        "price_range": (10, 160),
        "cat_affinity": {"Electronics": 0.20, "Computers": 0.20, "HomeKitchen": 0.60},
        "recency_weight": 0.10,
    },
    "quality_seeker": {
        "brand_pref": {"budget_brand": 0.05, "mid_brand": 0.20, "premium_brand": 0.75},
        "price_range": (300, 1000),
        "cat_affinity": {"Electronics": 0.55, "Computers": 0.35, "HomeKitchen": 0.10},
        "recency_weight": 0.75,
    },
    "explorer": {
        "brand_pref": {"budget_brand": 0.25, "mid_brand": 0.50, "premium_brand": 0.25},
        "price_range": (50, 700),
        "cat_affinity": {"Electronics": 0.35, "Computers": 0.30, "HomeKitchen": 0.35},
        "recency_weight": 0.65,
    },
    "loyal": {
        "brand_pref": {"budget_brand": 0.05, "mid_brand": 0.30, "premium_brand": 0.65},
        "price_range": (100, 800),
        "cat_affinity": {"Electronics": 0.70, "Computers": 0.20, "HomeKitchen": 0.10},
        "recency_weight": 0.25,
    },
    "balanced": {
        "brand_pref": {"budget_brand": 0.20, "mid_brand": 0.50, "premium_brand": 0.30},
        "price_range": (80, 500),
        "cat_affinity": {"Electronics": 0.35, "Computers": 0.30, "HomeKitchen": 0.35},
        "recency_weight": 0.45,
    },
}


def top_k_set(scores: np.ndarray, k: int = TOP_K) -> set[int]:
    return set(int(i) for i in np.argsort(scores)[::-1][:k])


def top_k_ranking(scores: np.ndarray, k: int = TOP_K) -> list[int]:
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo + 1e-10)


def summarize_list(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n": int(arr.size),
        "raw": arr.tolist(),
    }


def summarize_nested(raw: Mapping) -> dict:
    summary: dict = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            summary[str(key)] = summarize_nested(value)
        else:
            summary[str(key)] = summarize_list(value)
    return summary


def profile_seed(run_seed: int, profile_name: str) -> int:
    return run_seed + PROFILE_SEED_OFFSETS[profile_name]


def topsis_artifacts(df: pd.DataFrame) -> dict:
    matrix = df[TOPSIS_COLUMNS].to_numpy(dtype=float).copy()
    matrix = np.clip(matrix, 1e-10, None)

    proportions = matrix / matrix.sum(axis=0, keepdims=True)
    proportions = np.clip(proportions, 1e-10, 1.0)
    entropy = -np.sum(proportions * np.log(proportions), axis=0) / np.log(len(matrix))
    diversification = 1.0 - entropy

    floor = 0.10
    remaining = 1.0 - floor * len(TOPSIS_COLUMNS)
    weights = floor + (diversification / diversification.sum()) * remaining
    weights = np.clip(weights, floor, None)
    weights = weights / weights.sum()

    norms = np.sqrt((matrix**2).sum(axis=0))
    norms[norms == 0] = 1.0
    weighted_matrix = (matrix / norms) * weights

    ideal_plus = weighted_matrix.max(axis=0)
    ideal_minus = weighted_matrix.min(axis=0)
    d_plus = np.sqrt(((weighted_matrix - ideal_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted_matrix - ideal_minus) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    denom[denom == 0] = 1e-10
    scores = d_minus / denom

    return {
        "scores": scores,
        "weights": weights,
        "raw_matrix": matrix,
    }


def hidden_components(df: pd.DataFrame, profile: Mapping[str, object]) -> dict:
    brand_pref = profile["brand_pref"]
    cat_affinity = profile["cat_affinity"]
    price_lo, price_hi = profile["price_range"]
    recency_weight = float(profile["recency_weight"])

    brand_score = np.array([brand_pref.get(b, 0.10) for b in df["brand"]], dtype=float)
    center = (price_lo + price_hi) / 2.0
    half_range = (price_hi - price_lo) / 2.0 + 1.0
    price_fit = np.clip(1.0 - np.abs(df["price"].to_numpy(dtype=float) - center) / half_range, 0, 1)
    cat_score = np.array([cat_affinity.get(c, 0.05) for c in df["category"]], dtype=float)
    cat_score = cat_score / cat_score.max()
    recency_score = df["recency_pct"].to_numpy(dtype=float) * recency_weight + (1.0 - recency_weight) * 0.5

    return {
        "brand_score": brand_score,
        "price_fit": price_fit,
        "cat_score": cat_score,
        "recency_score": recency_score,
    }


def hidden_utility(df: pd.DataFrame, profile: Mapping[str, object]) -> np.ndarray:
    comp = hidden_components(df, profile)
    hidden = (
        0.45 * comp["brand_score"]
        + 0.30 * comp["price_fit"]
        + 0.15 * comp["cat_score"]
        + 0.10 * comp["recency_score"]
    )
    return norm(hidden)


def observable_utility(df: pd.DataFrame) -> np.ndarray:
    obs = (
        df["price_pct"].to_numpy(dtype=float)
        + df["quality_pct"].to_numpy(dtype=float)
        + df["popularity_pct"].to_numpy(dtype=float)
        + df["rating_pct"].to_numpy(dtype=float)
    ) / 4.0
    return norm(obs)


def build_ground_truth(
    df: pd.DataFrame,
    profile: Mapping[str, object],
    seed: int,
    observable_alpha: float = MAIN_ALPHA,
) -> np.ndarray:
    rng = np.random.RandomState(seed + 7777)
    observable = observable_utility(df)
    hidden = hidden_utility(df, profile)
    gt = observable_alpha * observable + (1.0 - observable_alpha) * hidden + rng.normal(0.0, GT_NOISE_SD, len(df))
    return norm(gt)


def build_candidate_pool(
    df: pd.DataFrame,
    profiles: Iterable[Mapping[str, object]],
    gt_sets: Iterable[set[int]],
    top_n_hidden: int = 30,
) -> np.ndarray:
    items: set[int] = set()
    for gt_set in gt_sets:
        items.update(gt_set)
    for profile in profiles:
        hid_scores = hidden_utility(df, profile)
        items.update(top_k_set(hid_scores, k=top_n_hidden))
    return np.array(sorted(items), dtype=int)


def popularity_recs(df: pd.DataFrame, k: int = TOP_K) -> set[int]:
    return set(int(idx) for idx in df["popularity_pct"].nlargest(k).index)


def popularity_ranking(df: pd.DataFrame, k: int = TOP_K) -> list[int]:
    return [int(idx) for idx in df["popularity_pct"].nlargest(k).index]


def random_recs(rng: np.random.RandomState, n_products: int, k: int = TOP_K) -> set[int]:
    return set(int(x) for x in rng.choice(n_products, size=k, replace=False))


def random_ranking(rng: np.random.RandomState, n_products: int, k: int = TOP_K) -> list[int]:
    return [int(x) for x in rng.choice(n_products, size=k, replace=False)]


def f1_score(predicted: set[int], truth: set[int]) -> float:
    if not predicted or not truth:
        return 0.0
    tp = len(predicted & truth)
    return 2.0 * tp / (len(predicted) + len(truth))


def ild_score(df: pd.DataFrame, rec_set: set[int]) -> float:
    items = sorted(rec_set)
    features = df.loc[items, FEATURE_COLUMNS].to_numpy(dtype=float)
    if len(features) < 2:
        return 0.0

    distances: List[float] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            distances.append(float(np.linalg.norm(features[i] - features[j])))
    return float(np.mean(distances))


def ndcg_at_k(ranked_items: Sequence[int], truth: set[int], k: int = TOP_K) -> float:
    ranked = list(ranked_items)[:k]
    if not ranked or not truth:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(ranked, start=1):
        if int(item) in truth:
            dcg += 1.0 / np.log2(rank + 1.0)

    ideal_hits = min(k, len(truth))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg)


def flip_brand_preferences(profile: Mapping[str, object]) -> dict:
    flipped = copy.deepcopy(profile)
    brand_pref = flipped["brand_pref"]
    ranked = sorted(brand_pref.items(), key=lambda item: item[1])
    low_key, low_val = ranked[0]
    mid_key, mid_val = ranked[1]
    high_key, high_val = ranked[2]
    flipped["brand_pref"] = {
        low_key: high_val,
        mid_key: mid_val,
        high_key: low_val,
    }
    return flipped


class QAgent:
    def __init__(
        self,
        n_products: int,
        n_profiles: int,
        seed: int,
        alpha: float = LEARNING_RATE,
        eps_init: float = EPS_INIT,
        eps_decay: float = EPS_DECAY,
        eps_min: float = EPS_MIN,
    ) -> None:
        self.q_table = np.zeros((n_profiles, n_products), dtype=float)
        self.visit_table = np.zeros((n_profiles, n_products), dtype=np.int32)
        self.alpha = alpha
        self.eps = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.rng = np.random.RandomState(seed)

    def act(self, state: int, pool: np.ndarray) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.choice(pool))
        q_values = self.q_table[state, pool]
        return int(pool[np.argmax(q_values)])

    def update(self, state: int, action: int, reward: float) -> None:
        self.visit_table[state, action] += 1
        self.q_table[state, action] += self.alpha * (reward - self.q_table[state, action])
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def q_scores(self, state: int) -> np.ndarray:
        return self.q_table[state].copy()

    def visit_counts(self, state: int) -> np.ndarray:
        return self.visit_table[state].copy()


def compute_reward(
    idx: int,
    profile: Mapping[str, object],
    df: pd.DataFrame,
    gt_set: set[int],
    rng: np.random.RandomState,
    shaping_bonus: float = REWARD_SHAPING_BONUS,
) -> float:
    components = hidden_components(df.iloc[[idx]], profile)

    brand_match = float(components["brand_score"][0])
    cat_match = float(components["cat_score"][0])
    in_range = float(components["price_fit"][0] > 0.999999)
    recency_signal = float(components["recency_score"][0])

    p_engage = np.clip(0.40 * brand_match + 0.35 * in_range + 0.15 * cat_match + 0.10 * recency_signal, 0, 1)
    p_convert = np.clip(0.50 * brand_match + 0.30 * in_range + 0.20 * cat_match, 0, 1)

    reward = -0.02
    if rng.random() < p_engage:
        reward += 0.30
        if rng.random() < p_convert:
            reward += 1.00

    if idx in gt_set:
        reward += shaping_bonus
    return float(reward)


def static_hybrid_score(q_scores: np.ndarray, topsis_scores: np.ndarray, lambda_q: float = MAIN_LAMBDA_Q) -> np.ndarray:
    lambda_t = 1.0 - lambda_q
    return lambda_q * norm(q_scores) + lambda_t * norm(topsis_scores)


def confidence_gated_score(
    q_scores: np.ndarray,
    topsis_scores: np.ndarray,
    visits: np.ndarray,
    tau: float = CONFIDENCE_TAU,
    lambda_max: float = CONFIDENCE_LAMBDA_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    lambda_q = lambda_max * (1.0 - np.exp(-visits / tau))
    score = lambda_q * norm(q_scores) + (1.0 - lambda_q) * norm(topsis_scores)
    return score, lambda_q


def criterion_contribution_label(df: pd.DataFrame, topsis_weights: np.ndarray, idx: int) -> str:
    values = df.loc[idx, TOPSIS_COLUMNS].to_numpy(dtype=float)
    weighted = values * topsis_weights
    return CRITERION_LABELS[TOPSIS_COLUMNS[int(np.argmax(weighted))]]


def hidden_contribution_label(df: pd.DataFrame, profile: Mapping[str, object], idx: int) -> str:
    row = df.iloc[[idx]]
    comp = hidden_components(row, profile)
    values = {
        str(row.iloc[0]["brand"]): float(comp["brand_score"][0]),
        "PriceRange": float(comp["price_fit"][0]),
        str(row.iloc[0]["category"]): float(comp["cat_score"][0]),
        "Recency": float(comp["recency_score"][0]),
    }
    return max(values.items(), key=lambda item: item[1])[0]


def build_xai_rows(
    df: pd.DataFrame,
    profile_name: str,
    profile: Mapping[str, object],
    topsis_scores: np.ndarray,
    topsis_weights: np.ndarray,
    q_scores: np.ndarray,
    gt_set: set[int],
) -> dict:
    topsis_norm = norm(topsis_scores)
    q_norm = norm(q_scores)
    hybrid_scores = MAIN_LAMBDA_T * topsis_norm + MAIN_LAMBDA_Q * q_norm
    ranked_items = list(np.argsort(hybrid_scores)[::-1][:TOP_K])

    rows = []
    gt_hits = 0
    for idx in ranked_items:
        c_t = float(MAIN_LAMBDA_T * topsis_norm[idx])
        c_q = float(MAIN_LAMBDA_Q * q_norm[idx])
        if c_t >= c_q:
            dominant_signal = "Observable"
            dominant_label = criterion_contribution_label(df, topsis_weights, idx)
        else:
            dominant_signal = "Behavioral"
            dominant_label = hidden_contribution_label(df, profile, idx)

        in_gt = int(idx) in gt_set
        gt_hits += int(in_gt)
        rows.append(
            {
                "item": int(idx),
                "c_t": round(c_t, 3),
                "c_q": round(c_q, 3),
                "dominant_signal": dominant_signal,
                "dominant_label": dominant_label,
                "in_gt": bool(in_gt),
            }
        )

    return {
        "profile": profile_name,
        "gt_hits": gt_hits,
        "rows": rows,
    }


def train_main_profile(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    run_seed: int,
    topsis_bundle: Mapping[str, np.ndarray],
    checkpoints: Sequence[int],
    shaping_bonus: float = REWARD_SHAPING_BONUS,
) -> dict:
    profile = PROFILE_HIDDEN[profile_name]
    gt_seed = profile_seed(run_seed, profile_name)
    gt_scores = build_ground_truth(df, profile, gt_seed, observable_alpha=MAIN_ALPHA)
    gt_set = top_k_set(gt_scores)
    gt_ranking = top_k_ranking(gt_scores)

    pool = build_candidate_pool(df, [profile], [gt_set])
    n_products = len(df)
    random_rank = random_ranking(np.random.RandomState(run_seed + 5555), n_products=n_products)
    random_set = set(random_rank)
    popularity_rank = popularity_ranking(df)
    popularity_set = popularity_recs(df)
    topsis_rank = top_k_ranking(topsis_bundle["scores"])
    topsis_set = top_k_set(topsis_bundle["scores"])

    rng = np.random.RandomState(run_seed + profile_idx * 997)
    agent = QAgent(n_products, N_PROFILES, seed=run_seed + profile_idx * 13)

    results = {m: {} for m in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]}
    final_payload: dict | None = None

    for episode in range(1, max(checkpoints) + 1):
        action = agent.act(profile_idx, pool)
        reward = compute_reward(action, profile, df, gt_set, rng, shaping_bonus=shaping_bonus)
        agent.update(profile_idx, action, reward)

        if episode in checkpoints:
            q_scores = agent.q_scores(profile_idx)
            visits = agent.visit_counts(profile_idx)

            rl_rank = top_k_ranking(q_scores)
            rl_set = top_k_set(q_scores)
            hybrid_scores = static_hybrid_score(q_scores, topsis_bundle["scores"], lambda_q=MAIN_LAMBDA_Q)
            hybrid_rank = top_k_ranking(hybrid_scores)
            hybrid_set = set(hybrid_rank)

            results["random"][episode] = f1_score(random_set, gt_set)
            results["popularity"][episode] = f1_score(popularity_set, gt_set)
            results["topsis_only"][episode] = f1_score(topsis_set, gt_set)
            results["rl_only"][episode] = f1_score(rl_set, gt_set)
            results["hybrid"][episode] = f1_score(hybrid_set, gt_set)

            if episode == max(checkpoints):
                gated_scores, lambda_q_vec = confidence_gated_score(q_scores, topsis_bundle["scores"], visits)
                gated_rank = top_k_ranking(gated_scores)
                final_payload = {
                    "profile_name": profile_name,
                    "profile_idx": profile_idx,
                    "gt_seed": gt_seed,
                    "random_set": random_set,
                    "random_rank": random_rank,
                    "popularity_set": popularity_set,
                    "popularity_rank": popularity_rank,
                    "topsis_set": topsis_set,
                    "topsis_rank": topsis_rank,
                    "rl_set": rl_set,
                    "rl_rank": rl_rank,
                    "hybrid_set": hybrid_set,
                    "hybrid_rank": hybrid_rank,
                    "gated_set": set(gated_rank),
                    "gated_rank": gated_rank,
                    "gt_set": gt_set,
                    "gt_rank": gt_ranking,
                    "q_scores": q_scores,
                    "visits": visits,
                    "topsis_scores": topsis_bundle["scores"],
                    "topsis_weights": topsis_bundle["weights"],
                    "lambda_q_mean": float(np.mean(lambda_q_vec)),
                    "shaping_bonus": float(shaping_bonus),
                }

    assert final_payload is not None
    return {
        "f1": results,
        "final": final_payload,
    }


def train_drift_profile(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    run_seed: int,
    topsis_scores: np.ndarray,
    checkpoints: Sequence[int],
) -> dict:
    pre_profile = PROFILE_HIDDEN[profile_name]
    post_profile = flip_brand_preferences(pre_profile)

    pre_seed = profile_seed(run_seed, profile_name)
    post_seed = pre_seed + 5000
    pre_gt_set = top_k_set(build_ground_truth(df, pre_profile, pre_seed, observable_alpha=MAIN_ALPHA))
    post_gt_set = top_k_set(build_ground_truth(df, post_profile, post_seed, observable_alpha=MAIN_ALPHA))

    pool = build_candidate_pool(df, [pre_profile, post_profile], [pre_gt_set, post_gt_set])
    rng = np.random.RandomState(run_seed + profile_idx * 997)
    agent = QAgent(len(df), N_PROFILES, seed=run_seed + profile_idx * 13)

    results = {m: {} for m in ["rl_only", "hybrid"]}
    topsis_norm = norm(topsis_scores)

    for episode in range(1, max(checkpoints) + 1):
        if episode > DRIFT_EPISODE:
            active_profile = post_profile
            active_gt_set = post_gt_set
        else:
            active_profile = pre_profile
            active_gt_set = pre_gt_set

        action = agent.act(profile_idx, pool)
        reward = compute_reward(action, active_profile, df, active_gt_set, rng)
        agent.update(profile_idx, action, reward)

        if episode in checkpoints:
            q_scores = agent.q_scores(profile_idx)
            rl_set = top_k_set(q_scores)
            hybrid_scores = MAIN_LAMBDA_Q * norm(q_scores) + MAIN_LAMBDA_T * topsis_norm
            hybrid_set = top_k_set(hybrid_scores)

            results["rl_only"][episode] = f1_score(rl_set, active_gt_set)
            results["hybrid"][episode] = f1_score(hybrid_set, active_gt_set)

    return results


def main_diagnostics(df: pd.DataFrame, run_seed: int, topsis_scores: np.ndarray) -> dict:
    diagnostics: dict = {}
    topsis_set = top_k_set(topsis_scores)

    for profile_name in PROFILE_ORDER:
        profile = PROFILE_HIDDEN[profile_name]
        gt_seed = profile_seed(run_seed, profile_name)
        gt_scores = build_ground_truth(df, profile, gt_seed, observable_alpha=MAIN_ALPHA)
        gt_set = top_k_set(gt_scores)
        hid_set = top_k_set(hidden_utility(df, profile))
        obs_set = top_k_set(observable_utility(df))
        diagnostics[profile_name] = {
            "topsis_f1": f1_score(topsis_set, gt_set),
            "hid_oracle_f1": f1_score(hid_set, gt_set),
            "obs_oracle_f1": f1_score(obs_set, gt_set),
            "topsis_gt_corr": float(np.corrcoef(topsis_scores, gt_scores)[0, 1]),
        }
    return diagnostics


def compute_significance(summary: Mapping[str, Mapping[str, dict]], baselines: Sequence[str]) -> dict:
    last_checkpoint = str(MAIN_CHECKPOINTS[-1])
    hybrid = np.asarray(summary["hybrid"][last_checkpoint]["raw"], dtype=float)
    adjusted_alpha = 0.05 / len(baselines)
    results: dict = {}

    for baseline in baselines:
        baseline_values = np.asarray(summary[baseline][last_checkpoint]["raw"], dtype=float)
        diffs = hybrid - baseline_values
        test = stats.ttest_rel(hybrid, baseline_values)
        diff_std = np.std(diffs, ddof=1)
        cohen_d = float(np.mean(diffs) / diff_std) if diff_std > 1e-12 else float("inf")
        results[f"hybrid_vs_{baseline}"] = {
            "delta_f1": float(np.mean(diffs)),
            "t_stat": float(test.statistic),
            "p_value": float(test.pvalue),
            "bonferroni_alpha": float(adjusted_alpha),
            "significant": bool(test.pvalue < adjusted_alpha and np.mean(diffs) > 0),
            "cohen_d": cohen_d,
        }

    return results

