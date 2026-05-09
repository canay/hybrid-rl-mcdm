"""
Hybrid RL-TOPSIS Amazon India experiment runner.

This script rebuilds the main evidence chain on a public Amazon India
product/review catalog. It writes deterministic enriched bootstrap catalogs
plus primary, extended, robustness, and drift reports under the package-level
data, results, and logs folders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from hybrid_core import (
    CONFIDENCE_LAMBDA_MAX,
    CONFIDENCE_TAU,
    DRIFT_CHECKPOINTS,
    DRIFT_EPISODE,
    FEATURE_COLUMNS,
    GT_SPLIT_GRID,
    LAMBDA_GRID,
    MAIN_ALPHA,
    MAIN_CHECKPOINTS,
    MAIN_LAMBDA_Q,
    MAIN_LAMBDA_T,
    N_BOOTSTRAP_DRIFT,
    N_BOOTSTRAP_RUNS,
    PROFILE_HIDDEN,
    PROFILE_ORDER,
    REWARD_SHAPING_BONUS,
    build_candidate_pool,
    build_ground_truth,
    build_xai_rows,
    confidence_gated_score,
    compute_significance,
    f1_score,
    ild_score,
    main_diagnostics,
    ndcg_at_k,
    norm,
    popularity_ranking,
    popularity_recs,
    profile_seed,
    random_ranking,
    static_hybrid_score,
    summarize_nested,
    top_k_ranking,
    top_k_set,
    topsis_artifacts,
)


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_india.csv"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
BOOTSTRAP_DIR = DATA_DIR / "bootstrap_catalogs"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULTS_DIR = PROJECT_ROOT / "results"
PRIMARY_PATH = RESULTS_DIR / "amazon_primary.json"
EXTENDED_PATH = RESULTS_DIR / "amazon_extended.json"
ROBUSTNESS_PATH = RESULTS_DIR / "amazon_robustness.json"
DRIFT_PATH = RESULTS_DIR / "amazon_drift.json"
RUN_SUMMARY_PATH = RESULTS_DIR / "run_summary.json"

DATASET_VERSION = "hybrid_rl_topsis_amazon_india_enriched"
N_PRODUCTS = 400
SEED_BASE = 17042
RUN_SEED_STEP = 17
SHAPING_GRID = [0.00, 0.10, 0.20]
REQUIRED_COLUMNS = [
    "price",
    "quality",
    "popularity",
    "rating",
    "category",
    "brand",
    "recency",
    "price_pct",
    "quality_pct",
    "popularity_pct",
    "rating_pct",
    "recency_pct",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_number(value: object) -> float:
    text = str(value)
    match = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not match:
        return float("nan")
    return float(match[0])


def parse_rating(value: object) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return float("nan")


def reviewer_count(value: object) -> int:
    if pd.isna(value):
        return 0
    return len([part for part in str(value).split(",") if part.strip()])


def text_richness(*values: object) -> float:
    joined = " ".join("" if pd.isna(value) else str(value) for value in values)
    return float(min(len(joined), 2000))


def coarse_category(raw_category: object) -> str:
    head = str(raw_category).split("|")[0]
    if head == "Computers&Accessories" or head == "OfficeProducts":
        return "Computers"
    if head in {"Home&Kitchen", "HomeImprovement"}:
        return "HomeKitchen"
    return "Electronics"


def extract_brand_label(product_name: object) -> str:
    text = str(product_name).strip()
    if not text:
        return "Unknown"
    token = re.split(r"\s+|,", text)[0]
    token = re.sub(r"[^A-Za-z0-9&+-]", "", token)
    return token[:40] or "Unknown"


def percentile_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    ranked = series.rank(pct=True, method="average")
    return 1.0 - ranked if invert else ranked


def load_enriched_catalog() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Amazon India raw CSV not found: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH)
    df = raw.copy()
    df["price"] = df["discounted_price"].map(parse_number)
    df["actual_price"] = df["actual_price"].map(parse_number)
    df["discount_pct"] = df["discount_percentage"].map(parse_number).clip(0, 100)
    df["rating"] = df["rating"].map(parse_rating)
    df["rating_count"] = df["rating_count"].map(parse_number)
    df["inferred_reviewer_count"] = df["user_id"].map(reviewer_count)
    df["review_text_richness"] = [
        text_richness(title, content, about)
        for title, content, about in zip(df["review_title"], df["review_content"], df["about_product"])
    ]
    df["category_raw"] = df["category"].astype(str)
    df["category"] = df["category_raw"].map(coarse_category)
    df["brand_label"] = df["product_name"].map(extract_brand_label)

    keep = df.dropna(subset=["product_id", "price", "rating", "rating_count"]).copy()
    keep = keep[(keep["price"] > 0) & keep["rating"].between(1.0, 5.0)]
    keep = keep.sort_values(["rating_count", "rating"], ascending=[False, False])
    keep = keep.drop_duplicates(subset=["product_id"], keep="first").reset_index(drop=True)

    rating_norm = (keep["rating"] - 1.0) / 4.0
    review_count_norm = percentile_rank(np.log1p(keep["rating_count"]))
    richness_norm = percentile_rank(keep["review_text_richness"])
    reviewer_breadth_norm = percentile_rank(keep["inferred_reviewer_count"])

    keep["quality"] = 100.0 * (0.70 * rating_norm + 0.20 * review_count_norm + 0.10 * richness_norm)
    keep["popularity"] = 100.0 * review_count_norm
    keep["recency"] = 100.0 * (0.65 * reviewer_breadth_norm + 0.35 * percentile_rank(keep["discount_pct"]))

    price_rank = keep["price"].rank(pct=True)
    keep["brand"] = np.select(
        [price_rank <= 1 / 3, price_rank >= 2 / 3],
        ["budget_brand", "premium_brand"],
        default="mid_brand",
    )

    keep["price_pct"] = percentile_rank(keep["price"], invert=True)
    keep["quality_pct"] = percentile_rank(keep["quality"])
    keep["popularity_pct"] = percentile_rank(keep["popularity"])
    keep["rating_pct"] = percentile_rank(keep["rating"])
    keep["recency_pct"] = percentile_rank(keep["recency"])

    columns = [
        "product_id",
        "product_name",
        "category_raw",
        "category",
        "brand_label",
        "brand",
        "price",
        "actual_price",
        "discount_pct",
        "rating",
        "rating_count",
        "inferred_reviewer_count",
        "review_text_richness",
        "quality",
        "popularity",
        "recency",
        "price_pct",
        "quality_pct",
        "popularity_pct",
        "rating_pct",
        "recency_pct",
    ]
    return keep[columns].reset_index(drop=True)


def run_seed(run_index: int) -> int:
    return SEED_BASE + run_index * RUN_SEED_STEP


def dataset_path(run_index: int) -> Path:
    seed = run_seed(run_index)
    return BOOTSTRAP_DIR / f"amazon_india_enriched_run{run_index:02d}_seed{seed}.csv"


def stratified_sample(enriched: pd.DataFrame, seed: int, n_products: int = N_PRODUCTS) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    work = enriched.copy()
    work["_price_hi"] = work["price_pct"] >= work["price_pct"].median()
    work["_quality_hi"] = work["quality_pct"] >= work["quality_pct"].median()
    work["_popularity_hi"] = work["popularity_pct"] >= work["popularity_pct"].median()

    picked: list[int] = []
    per_octant = n_products // 8
    for p_hi in [False, True]:
        for q_hi in [False, True]:
            for pop_hi in [False, True]:
                mask = (
                    (work["_price_hi"] == p_hi)
                    & (work["_quality_hi"] == q_hi)
                    & (work["_popularity_hi"] == pop_hi)
                )
                candidates = work.index[mask].to_numpy(dtype=int)
                take = min(per_octant, len(candidates))
                if take:
                    picked.extend(int(x) for x in rng.choice(candidates, size=take, replace=False))

    remaining = [int(idx) for idx in work.index if int(idx) not in set(picked)]
    if len(picked) < n_products:
        fill = rng.choice(np.array(remaining, dtype=int), size=n_products - len(picked), replace=False)
        picked.extend(int(x) for x in fill)
    elif len(picked) > n_products:
        picked = [int(x) for x in rng.choice(np.array(picked, dtype=int), size=n_products, replace=False)]

    sample = work.loc[picked].drop(columns=["_price_hi", "_quality_hi", "_popularity_hi"]).reset_index(drop=True)
    for col in ["price", "quality", "popularity", "rating", "recency"]:
        sample[f"{col}_pct"] = percentile_rank(sample[col], invert=(col == "price"))
    return sample


def build_manifest(enriched: pd.DataFrame, n_runs: int) -> dict:
    run_payloads = []
    for run_index in range(n_runs):
        path = dataset_path(run_index)
        run_payloads.append(
            {
                "run_index": run_index,
                "seed": run_seed(run_index),
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )

    return {
        "dataset_version": DATASET_VERSION,
        "source": "Amazon India sales/reviews product catalog CSV",
        "raw_path": str(RAW_PATH.relative_to(PROJECT_ROOT)),
        "raw_sha256": sha256_file(RAW_PATH),
        "raw_rows": int(pd.read_csv(RAW_PATH, usecols=["product_id"]).shape[0]),
        "unique_products_after_cleaning": int(len(enriched)),
        "n_products_per_run": N_PRODUCTS,
        "seed_base": SEED_BASE,
        "run_seed_step": RUN_SEED_STEP,
        "n_runs": int(n_runs),
        "privacy_publication_note": (
            "Generated Hybrid RL-TOPSIS bootstrap catalogs exclude raw user_id, user_name, review_id, "
            "review_title, review_content, image links, and product links."
        ),
        "feature_derivation": {
            "price": "numeric discounted_price parsed from INR text",
            "actual_price": "numeric actual_price parsed from INR text",
            "discount_pct": "numeric discount_percentage",
            "rating": "Amazon product rating",
            "rating_count": "Amazon rating_count parsed as integer-like numeric",
            "quality": "0.70 normalized rating + 0.20 review-count percentile + 0.10 review-text-richness percentile",
            "popularity": "log1p(rating_count) percentile scaled to 0-100",
            "recency": "engagement proxy: 0.65 inferred reviewer breadth percentile + 0.35 discount percentile; source CSV has no timestamp",
            "category": "coarse category mapped to Electronics, Computers, or HomeKitchen from Amazon category path",
            "brand": "price-tier proxy required by the behavioral layer: budget/mid/premium derived from price tertiles",
        },
        "runs": run_payloads,
    }


def generate_datasets(n_runs: int, overwrite: bool = False) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    enriched = load_enriched_catalog()
    written = []
    for run_index in range(n_runs):
        path = dataset_path(run_index)
        if overwrite or not path.exists():
            sample = stratified_sample(enriched, run_seed(run_index))
            sample.to_csv(path, index=False)
        written.append(path)

    manifest = build_manifest(enriched, n_runs)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return written


def ensure_datasets(n_runs: int) -> list[Path]:
    missing = [dataset_path(run_index) for run_index in range(n_runs) if not dataset_path(run_index).exists()]
    if missing or not MANIFEST_PATH.exists():
        return generate_datasets(n_runs=n_runs, overwrite=False)
    return [dataset_path(run_index) for run_index in range(n_runs)]


def load_products_for_run(run_index: int) -> pd.DataFrame:
    path = dataset_path(run_index)
    if not path.exists():
        ensure_datasets(run_index + 1)
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Hybrid RL-TOPSIS dataset missing required columns: {missing}")
    return df


def serialize_profile_final(final: Mapping[str, object]) -> dict:
    return {
        "profile_name": final["profile_name"],
        "profile_idx": int(final["profile_idx"]),
        "gt_seed": int(final["gt_seed"]),
        "random_set": sorted(int(x) for x in final["random_set"]),
        "popularity_set": sorted(int(x) for x in final["popularity_set"]),
        "topsis_set": sorted(int(x) for x in final["topsis_set"]),
        "rl_set": sorted(int(x) for x in final["rl_set"]),
        "hybrid_set": sorted(int(x) for x in final["hybrid_set"]),
        "gated_set": sorted(int(x) for x in final["gated_set"]),
        "gt_set": sorted(int(x) for x in final["gt_set"]),
        "random_rank": [int(x) for x in final["random_rank"]],
        "popularity_rank": [int(x) for x in final["popularity_rank"]],
        "topsis_rank": [int(x) for x in final["topsis_rank"]],
        "rl_rank": [int(x) for x in final["rl_rank"]],
        "hybrid_rank": [int(x) for x in final["hybrid_rank"]],
        "gated_rank": [int(x) for x in final["gated_rank"]],
        "gt_rank": [int(x) for x in final["gt_rank"]],
        "q_scores": [float(x) for x in final["q_scores"]],
        "visits": [int(x) for x in final["visits"]],
        "topsis_scores": [float(x) for x in final["topsis_scores"]],
        "topsis_weights": [float(x) for x in final["topsis_weights"]],
        "lambda_q_mean": float(final["lambda_q_mean"]),
        "shaping_bonus": float(final["shaping_bonus"]),
    }


def reward_arrays(df: pd.DataFrame, profile: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    brand_pref = profile["brand_pref"]
    price_lo, price_hi = profile["price_range"]
    recency_weight = float(profile["recency_weight"])

    brand_match = np.array([brand_pref.get(b, 0.10) for b in df["brand"]], dtype=float)
    center = (price_lo + price_hi) / 2.0
    half_range = (price_hi - price_lo) / 2.0 + 1.0
    price_fit = np.clip(1.0 - np.abs(df["price"].to_numpy(dtype=float) - center) / half_range, 0, 1)
    in_range = (price_fit > 0.999999).astype(float)

    # Mirrors hybrid_core.compute_reward: hidden_components is called on a
    # one-row frame there, so category affinity normalizes to 1.0 for the item.
    cat_match = np.ones(len(df), dtype=float)
    recency_signal = df["recency_pct"].to_numpy(dtype=float) * recency_weight + (1.0 - recency_weight) * 0.5

    p_engage = np.clip(0.40 * brand_match + 0.35 * in_range + 0.15 * cat_match + 0.10 * recency_signal, 0, 1)
    p_convert = np.clip(0.50 * brand_match + 0.30 * in_range + 0.20 * cat_match, 0, 1)
    return p_engage, p_convert


def fast_train_main_profile(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    seed: int,
    topsis_bundle: Mapping[str, np.ndarray],
    checkpoints: Sequence[int],
    shaping_bonus: float = REWARD_SHAPING_BONUS,
) -> dict:
    profile = PROFILE_HIDDEN[profile_name]
    gt_seed = profile_seed(seed, profile_name)
    gt_scores = build_ground_truth(df, profile, gt_seed, observable_alpha=MAIN_ALPHA)
    gt_set = top_k_set(gt_scores)
    gt_ranking = top_k_ranking(gt_scores)
    pool = build_candidate_pool(df, [profile], [gt_set])

    n_products = len(df)
    q_scores = np.zeros(n_products, dtype=float)
    visits = np.zeros(n_products, dtype=np.int32)
    eps = 0.30
    eps_decay = 0.9997
    eps_min = 0.05
    lr = 0.05

    random_rank = random_ranking(np.random.RandomState(seed + 5555), n_products=n_products)
    random_set = set(random_rank)
    popularity_rank = popularity_ranking(df)
    popularity_set = popularity_recs(df)
    topsis_rank = top_k_ranking(topsis_bundle["scores"])
    topsis_set = top_k_set(topsis_bundle["scores"])

    rng = np.random.RandomState(seed + profile_idx * 997)
    act_rng = np.random.RandomState(seed + profile_idx * 13)
    p_engage, p_convert = reward_arrays(df, profile)
    gt_mask = np.zeros(n_products, dtype=bool)
    gt_mask[list(gt_set)] = True

    results = {m: {} for m in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]}
    final_payload: dict | None = None

    max_checkpoint = max(checkpoints)
    checkpoint_set = set(int(cp) for cp in checkpoints)
    for episode in range(1, max_checkpoint + 1):
        if act_rng.random() < eps:
            action = int(act_rng.choice(pool))
        else:
            action = int(pool[np.argmax(q_scores[pool])])

        reward = -0.02
        if rng.random() < p_engage[action]:
            reward += 0.30
            if rng.random() < p_convert[action]:
                reward += 1.00
        if gt_mask[action]:
            reward += shaping_bonus

        visits[action] += 1
        q_scores[action] += lr * (reward - q_scores[action])
        eps = max(eps_min, eps * eps_decay)

        if episode in checkpoint_set:
            q_snapshot = q_scores.copy()
            visits_snapshot = visits.copy()
            rl_rank = top_k_ranking(q_snapshot)
            rl_set = top_k_set(q_snapshot)
            hybrid_scores = static_hybrid_score(q_snapshot, topsis_bundle["scores"], lambda_q=MAIN_LAMBDA_Q)
            hybrid_rank = top_k_ranking(hybrid_scores)
            hybrid_set = set(hybrid_rank)

            results["random"][episode] = f1_score(random_set, gt_set)
            results["popularity"][episode] = f1_score(popularity_set, gt_set)
            results["topsis_only"][episode] = f1_score(topsis_set, gt_set)
            results["rl_only"][episode] = f1_score(rl_set, gt_set)
            results["hybrid"][episode] = f1_score(hybrid_set, gt_set)

            if episode == max_checkpoint:
                gated_scores, lambda_q_vec = confidence_gated_score(q_snapshot, topsis_bundle["scores"], visits_snapshot)
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
                    "q_scores": q_snapshot,
                    "visits": visits_snapshot,
                    "topsis_scores": topsis_bundle["scores"],
                    "topsis_weights": topsis_bundle["weights"],
                    "lambda_q_mean": float(np.mean(lambda_q_vec)),
                    "shaping_bonus": float(shaping_bonus),
                }

    assert final_payload is not None
    return {"f1": results, "final": final_payload}


def flip_brand_preferences(profile: Mapping[str, object]) -> dict:
    flipped = json.loads(json.dumps(profile))
    brand_pref = flipped["brand_pref"]
    ranked = sorted(brand_pref.items(), key=lambda item: item[1])
    low_key, low_val = ranked[0]
    mid_key, mid_val = ranked[1]
    high_key, high_val = ranked[2]
    flipped["brand_pref"] = {low_key: high_val, mid_key: mid_val, high_key: low_val}
    return flipped


def fast_train_drift_profile(
    df: pd.DataFrame,
    profile_name: str,
    profile_idx: int,
    seed: int,
    topsis_scores: np.ndarray,
    checkpoints: Sequence[int],
) -> dict:
    pre_profile = PROFILE_HIDDEN[profile_name]
    post_profile = flip_brand_preferences(pre_profile)
    pre_seed = profile_seed(seed, profile_name)
    post_seed = pre_seed + 5000
    pre_gt_set = top_k_set(build_ground_truth(df, pre_profile, pre_seed, observable_alpha=MAIN_ALPHA))
    post_gt_set = top_k_set(build_ground_truth(df, post_profile, post_seed, observable_alpha=MAIN_ALPHA))
    pool = build_candidate_pool(df, [pre_profile, post_profile], [pre_gt_set, post_gt_set])

    n_products = len(df)
    q_scores = np.zeros(n_products, dtype=float)
    eps = 0.30
    eps_decay = 0.9997
    eps_min = 0.05
    lr = 0.05
    rng = np.random.RandomState(seed + profile_idx * 997)
    act_rng = np.random.RandomState(seed + profile_idx * 13)

    pre_engage, pre_convert = reward_arrays(df, pre_profile)
    post_engage, post_convert = reward_arrays(df, post_profile)
    pre_mask = np.zeros(n_products, dtype=bool)
    post_mask = np.zeros(n_products, dtype=bool)
    pre_mask[list(pre_gt_set)] = True
    post_mask[list(post_gt_set)] = True

    topsis_norm = norm(topsis_scores)
    results = {m: {} for m in ["rl_only", "hybrid"]}
    checkpoint_set = set(int(cp) for cp in checkpoints)
    for episode in range(1, max(checkpoints) + 1):
        if act_rng.random() < eps:
            action = int(act_rng.choice(pool))
        else:
            action = int(pool[np.argmax(q_scores[pool])])

        if episode > DRIFT_EPISODE:
            p_engage, p_convert, gt_mask, active_gt_set = post_engage, post_convert, post_mask, post_gt_set
        else:
            p_engage, p_convert, gt_mask, active_gt_set = pre_engage, pre_convert, pre_mask, pre_gt_set

        reward = -0.02
        if rng.random() < p_engage[action]:
            reward += 0.30
            if rng.random() < p_convert[action]:
                reward += 1.00
        if gt_mask[action]:
            reward += REWARD_SHAPING_BONUS
        q_scores[action] += lr * (reward - q_scores[action])
        eps = max(eps_min, eps * eps_decay)

        if episode in checkpoint_set:
            rl_set = top_k_set(q_scores)
            hybrid_scores = MAIN_LAMBDA_Q * norm(q_scores) + MAIN_LAMBDA_T * topsis_norm
            hybrid_set = top_k_set(hybrid_scores)
            results["rl_only"][episode] = f1_score(rl_set, active_gt_set)
            results["hybrid"][episode] = f1_score(hybrid_set, active_gt_set)

    return results


def run_primary_suite(n_runs: int) -> dict:
    raw_f1 = {m: {cp: [] for cp in MAIN_CHECKPOINTS} for m in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]}
    raw_ndcg = {m: [] for m in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]}
    profile_summary_raw = {
        profile_name: {m: [] for m in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]}
        for profile_name in PROFILE_ORDER
    }
    diagnostics_log: list[dict] = []
    run_artifacts: list[dict] = []
    xai_report: dict | None = None

    print("=" * 72)
    print("Hybrid RL-TOPSIS Amazon India primary suite")
    print("=" * 72)
    print(f"Bootstrap runs: {n_runs}")
    print(f"Source manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print("-" * 72)

    ensure_datasets(n_runs)

    for run_index in range(n_runs):
        seed = run_seed(run_index)
        df = load_products_for_run(run_index)
        topsis_bundle = topsis_artifacts(df)

        if run_index == 0:
            diagnostics = main_diagnostics(df, seed, topsis_bundle["scores"])
            diagnostics_log.append(diagnostics)
            print("Run 0 diagnostics (TOPSIS | hidden oracle | observable oracle | corr):")
            for profile_name in PROFILE_ORDER:
                diag = diagnostics[profile_name]
                print(
                    f"  {profile_name:16s}: "
                    f"{diag['topsis_f1']:.3f} | {diag['hid_oracle_f1']:.3f} | "
                    f"{diag['obs_oracle_f1']:.3f} | rho={diag['topsis_gt_corr']:.3f}"
                )
            print()

        run_f1 = {m: {cp: [] for cp in MAIN_CHECKPOINTS} for m in raw_f1}
        run_ndcg = {m: [] for m in raw_ndcg}
        profile_artifacts = []

        for profile_idx, profile_name in enumerate(PROFILE_ORDER):
            outcome = fast_train_main_profile(df, profile_name, profile_idx, seed, topsis_bundle, MAIN_CHECKPOINTS)
            for method in raw_f1:
                for checkpoint in MAIN_CHECKPOINTS:
                    run_f1[method][checkpoint].append(outcome["f1"][method][checkpoint])

            final = outcome["final"]
            gt_set = set(int(x) for x in final["gt_set"])
            ranking_map = {
                "random": final["random_rank"],
                "popularity": final["popularity_rank"],
                "topsis_only": final["topsis_rank"],
                "rl_only": final["rl_rank"],
                "hybrid": final["hybrid_rank"],
            }
            for method, ranking in ranking_map.items():
                run_ndcg[method].append(ndcg_at_k(ranking, gt_set))

            final_checkpoint = MAIN_CHECKPOINTS[-1]
            for method in profile_summary_raw[profile_name]:
                profile_summary_raw[profile_name][method].append(outcome["f1"][method][final_checkpoint])

            if xai_report is None and profile_name == "budget":
                xai_report = build_xai_rows(
                    df,
                    profile_name,
                    PROFILE_HIDDEN[profile_name],
                    final["topsis_scores"],
                    final["topsis_weights"],
                    final["q_scores"],
                    final["gt_set"],
                )

            profile_artifacts.append(
                {
                    "profile_name": profile_name,
                    "f1": {
                        method: {str(cp): float(value) for cp, value in outcome["f1"][method].items()}
                        for method in outcome["f1"]
                    },
                    "final": serialize_profile_final(final),
                }
            )

        for method in raw_f1:
            for checkpoint in MAIN_CHECKPOINTS:
                raw_f1[method][checkpoint].append(float(np.mean(run_f1[method][checkpoint])))
        for method in raw_ndcg:
            raw_ndcg[method].append(float(np.mean(run_ndcg[method])))

        run_artifacts.append(
            {
                "run_index": run_index,
                "run_seed": seed,
                "dataset_path": str(dataset_path(run_index).relative_to(PROJECT_ROOT)),
                "dataset_sha256": sha256_file(dataset_path(run_index)),
                "profile_results": profile_artifacts,
            }
        )

        if (run_index + 1) % 5 == 0 or run_index == 0:
            last_cp = MAIN_CHECKPOINTS[-1]
            print(f"Run {run_index + 1}/{n_runs} summary at {last_cp} episodes:")
            for method in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]:
                values = raw_f1[method][last_cp]
                print(f"  {method:12s}: F1={np.mean(values):.3f} +/- {np.std(values):.3f}")
            print()

    assert xai_report is not None
    main_summary = summarize_nested(raw_f1)
    return {
        "config": {
            "dataset_version": DATASET_VERSION,
            "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "bootstrap_runs": n_runs,
            "observable_alpha": MAIN_ALPHA,
            "lambda_q": MAIN_LAMBDA_Q,
            "lambda_t": MAIN_LAMBDA_T,
            "reward_shaping_bonus": REWARD_SHAPING_BONUS,
            "checkpoints": MAIN_CHECKPOINTS,
        },
        "summary": main_summary,
        "significance": compute_significance(main_summary, baselines=["random", "popularity", "topsis_only", "rl_only"]),
        "secondary_metrics": {"ndcg_at_7": summarize_nested(raw_ndcg)},
        "profile_summary": summarize_nested(profile_summary_raw),
        "diagnostics": diagnostics_log,
        "xai": xai_report,
        "artifacts": run_artifacts,
        "notes": {
            "dataset_design": (
                "The experiment uses deterministic 400-item bootstrap samples from a real Amazon India "
                "product/review catalog. Observable MCDM features are derived from catalog fields; "
                "the behavioral layer remains an explicit preference-simulation layer because the "
                "source CSV does not contain temporal clickstream or purchase labels."
            )
        },
    }


def compute_lambda_sensitivity(primary_report: dict) -> dict:
    raw_lambda = {lam: [] for lam in LAMBDA_GRID}
    for run_artifact in primary_report["artifacts"]:
        run_lambda = {lam: [] for lam in LAMBDA_GRID}
        for profile_artifact in run_artifact["profile_results"]:
            final = profile_artifact["final"]
            q_scores = np.asarray(final["q_scores"], dtype=float)
            topsis_scores = np.asarray(final["topsis_scores"], dtype=float)
            gt_set = set(int(x) for x in final["gt_set"])
            for lambda_q in LAMBDA_GRID:
                score = static_hybrid_score(q_scores, topsis_scores, lambda_q=lambda_q)
                run_lambda[lambda_q].append(f1_score(top_k_set(score), gt_set))
        for lambda_q in LAMBDA_GRID:
            raw_lambda[lambda_q].append(float(np.mean(run_lambda[lambda_q])))
    return {"lambda_q_grid": LAMBDA_GRID, "summary": summarize_nested(raw_lambda)}


def compute_split_sensitivity(primary_report: dict) -> dict:
    raw_split = {m: {alpha: [] for alpha in GT_SPLIT_GRID} for m in ["topsis_only", "rl_only", "hybrid"]}
    for run_artifact in primary_report["artifacts"]:
        df = pd.read_csv(PROJECT_ROOT / run_artifact["dataset_path"])
        run_split = {m: {alpha: [] for alpha in GT_SPLIT_GRID} for m in raw_split}
        for profile_artifact in run_artifact["profile_results"]:
            final = profile_artifact["final"]
            profile_name = profile_artifact["profile_name"]
            gt_seed = int(final["gt_seed"])
            topsis_set = set(int(x) for x in final["topsis_set"])
            rl_set = set(int(x) for x in final["rl_set"])
            hybrid_set = set(int(x) for x in final["hybrid_set"])
            for alpha in GT_SPLIT_GRID:
                alt_gt_scores = build_ground_truth(df, PROFILE_HIDDEN[profile_name], gt_seed, observable_alpha=alpha)
                alt_gt_set = top_k_set(alt_gt_scores)
                run_split["topsis_only"][alpha].append(f1_score(topsis_set, alt_gt_set))
                run_split["rl_only"][alpha].append(f1_score(rl_set, alt_gt_set))
                run_split["hybrid"][alpha].append(f1_score(hybrid_set, alt_gt_set))
        for method in raw_split:
            for alpha in GT_SPLIT_GRID:
                raw_split[method][alpha].append(float(np.mean(run_split[method][alpha])))
    return {
        "observable_alpha_grid": GT_SPLIT_GRID,
        "analysis_note": f"Evaluation-only target sensitivity on final policies from alpha={MAIN_ALPHA:.2f}.",
        "summary": summarize_nested(raw_split),
    }


def compute_ild(primary_report: dict) -> dict:
    raw_ild = {m: [] for m in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]}
    for run_artifact in primary_report["artifacts"]:
        df = pd.read_csv(PROJECT_ROOT / run_artifact["dataset_path"])
        run_ild = {m: [] for m in raw_ild}
        for profile_artifact in run_artifact["profile_results"]:
            final = profile_artifact["final"]
            final_sets = {
                "random": set(int(x) for x in final["random_set"]),
                "popularity": set(int(x) for x in final["popularity_set"]),
                "topsis_only": set(int(x) for x in final["topsis_set"]),
                "rl_only": set(int(x) for x in final["rl_set"]),
                "hybrid": set(int(x) for x in final["hybrid_set"]),
            }
            for method, rec_set in final_sets.items():
                run_ild[method].append(ild_score(df, rec_set))
        for method in raw_ild:
            raw_ild[method].append(float(np.mean(run_ild[method])))
    return {"feature_space": FEATURE_COLUMNS, "summary": summarize_nested(raw_ild)}


def compute_confidence_gated(primary_report: dict) -> dict:
    raw = {"static": [], "gated": [], "mean_lambda_q": []}
    for run_artifact in primary_report["artifacts"]:
        run_static = []
        run_gated = []
        run_lambda = []
        for profile_artifact in run_artifact["profile_results"]:
            final = profile_artifact["final"]
            gt_set = set(int(x) for x in final["gt_set"])
            hybrid_set = set(int(x) for x in final["hybrid_set"])
            gated_set = set(int(x) for x in final["gated_set"])
            run_static.append(f1_score(hybrid_set, gt_set))
            run_gated.append(f1_score(gated_set, gt_set))
            run_lambda.append(float(final["lambda_q_mean"]))
        raw["static"].append(float(np.mean(run_static)))
        raw["gated"].append(float(np.mean(run_gated)))
        raw["mean_lambda_q"].append(float(np.mean(run_lambda)))
    return {"tau": CONFIDENCE_TAU, "lambda_max": CONFIDENCE_LAMBDA_MAX, "summary": summarize_nested(raw)}


def build_extended_report(primary_report: dict) -> dict:
    return {
        "config": {
            "dataset_version": DATASET_VERSION,
            "source_primary_report": str(PRIMARY_PATH.relative_to(PROJECT_ROOT)),
            "observable_alpha_design_point": MAIN_ALPHA,
        },
        "lambda_sensitivity": compute_lambda_sensitivity(primary_report),
        "gt_split_sensitivity": compute_split_sensitivity(primary_report),
        "ild": compute_ild(primary_report),
        "confidence_gated": compute_confidence_gated(primary_report),
        "xai": primary_report["xai"],
    }


def profile_gap_summary(primary_report: dict) -> dict:
    summary = primary_report["profile_summary"]
    output: dict = {}
    for profile_name in PROFILE_ORDER:
        hybrid_mean = float(summary[profile_name]["hybrid"]["mean"])
        rl_mean = float(summary[profile_name]["rl_only"]["mean"])
        topsis_mean = float(summary[profile_name]["topsis_only"]["mean"])
        output[profile_name] = {
            "hybrid_mean_f1": hybrid_mean,
            "rl_mean_f1": rl_mean,
            "topsis_mean_f1": topsis_mean,
            "hybrid_minus_rl": hybrid_mean - rl_mean,
            "hybrid_minus_topsis": hybrid_mean - topsis_mean,
        }
    return output


def run_reward_shaping_sensitivity(n_runs: int) -> dict:
    raw = {
        "hybrid": {bonus: [] for bonus in SHAPING_GRID},
        "rl_only": {bonus: [] for bonus in SHAPING_GRID},
        "delta_hybrid_vs_rl": {bonus: [] for bonus in SHAPING_GRID},
    }
    print("-" * 72)
    print("Hybrid RL-TOPSIS reward-shaping robustness suite")
    print("-" * 72)
    for run_index in range(n_runs):
        seed = run_seed(run_index)
        df = load_products_for_run(run_index)
        topsis_bundle = topsis_artifacts(df)
        for shaping_bonus in SHAPING_GRID:
            run_hybrid = []
            run_rl = []
            for profile_idx, profile_name in enumerate(PROFILE_ORDER):
                outcome = fast_train_main_profile(
                    df,
                    profile_name,
                    profile_idx,
                    seed,
                    topsis_bundle,
                    [MAIN_CHECKPOINTS[-1]],
                    shaping_bonus=shaping_bonus,
                )
                run_hybrid.append(outcome["f1"]["hybrid"][MAIN_CHECKPOINTS[-1]])
                run_rl.append(outcome["f1"]["rl_only"][MAIN_CHECKPOINTS[-1]])
            raw["hybrid"][shaping_bonus].append(float(np.mean(run_hybrid)))
            raw["rl_only"][shaping_bonus].append(float(np.mean(run_rl)))
            raw["delta_hybrid_vs_rl"][shaping_bonus].append(float(np.mean(run_hybrid) - np.mean(run_rl)))
        if (run_index + 1) % 5 == 0 or run_index == 0:
            print(f"Robustness run {run_index + 1}/{n_runs}")
    return {"shaping_bonus_grid": SHAPING_GRID, "summary": summarize_nested(raw)}


def build_robustness_report(primary_report: dict, n_runs: int) -> dict:
    return {
        "config": {
            "dataset_version": DATASET_VERSION,
            "source_primary_report": str(PRIMARY_PATH.relative_to(PROJECT_ROOT)),
            "bootstrap_runs": n_runs,
        },
        "profile_gap_summary": profile_gap_summary(primary_report),
        "reward_shaping_sensitivity": run_reward_shaping_sensitivity(n_runs=n_runs),
    }


def run_drift_suite(n_runs: int) -> dict:
    raw = {m: {cp: [] for cp in DRIFT_CHECKPOINTS} for m in ["rl_only", "hybrid"]}
    print("-" * 72)
    print("Hybrid RL-TOPSIS Amazon India concept drift suite")
    print("-" * 72)
    for run_index in range(n_runs):
        seed = run_seed(run_index)
        df = load_products_for_run(run_index)
        topsis_scores = topsis_artifacts(df)["scores"]
        run_raw = {m: {cp: [] for cp in DRIFT_CHECKPOINTS} for m in raw}
        for profile_idx, profile_name in enumerate(PROFILE_ORDER):
            outcome = fast_train_drift_profile(df, profile_name, profile_idx, seed, topsis_scores, DRIFT_CHECKPOINTS)
            for method in raw:
                for checkpoint in DRIFT_CHECKPOINTS:
                    run_raw[method][checkpoint].append(outcome[method][checkpoint])
        for method in raw:
            for checkpoint in DRIFT_CHECKPOINTS:
                raw[method][checkpoint].append(float(np.mean(run_raw[method][checkpoint])))
        if (run_index + 1) % 5 == 0 or run_index == 0:
            print(f"Drift run {run_index + 1}/{n_runs}")
    return {
        "config": {
            "dataset_version": DATASET_VERSION,
            "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "bootstrap_runs": n_runs,
            "drift_episode": DRIFT_EPISODE,
            "checkpoints": DRIFT_CHECKPOINTS,
        },
        "summary": summarize_nested(raw),
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def final_metrics(primary: dict, extended: dict, robustness: dict, drift: dict) -> dict:
    last_main = str(MAIN_CHECKPOINTS[-1])
    last_drift = str(DRIFT_CHECKPOINTS[-1])
    primary_final = {
        method: primary["summary"][method][last_main]["mean"]
        for method in ["random", "popularity", "topsis_only", "rl_only", "hybrid"]
    }
    lambda_summary = extended["lambda_sensitivity"]["summary"]
    best_lambda = max(LAMBDA_GRID, key=lambda lam: lambda_summary[str(lam)]["mean"])
    return {
        "dataset_version": DATASET_VERSION,
        "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "primary_f1_at_30000": primary_final,
        "hybrid_minus_topsis": primary_final["hybrid"] - primary_final["topsis_only"],
        "hybrid_minus_rl": primary_final["hybrid"] - primary_final["rl_only"],
        "best_lambda_q": best_lambda,
        "best_lambda_q_mean_f1": lambda_summary[str(best_lambda)]["mean"],
        "robustness_bonus_0_20": {
            "hybrid": robustness["reward_shaping_sensitivity"]["summary"]["hybrid"]["0.2"]["mean"],
            "rl_only": robustness["reward_shaping_sensitivity"]["summary"]["rl_only"]["0.2"]["mean"],
        },
        "drift_f1_at_30000": {
            "rl_only": drift["summary"]["rl_only"][last_drift]["mean"],
            "hybrid": drift["summary"]["hybrid"][last_drift]["mean"],
        },
        "outputs": {
            "primary": str(PRIMARY_PATH.relative_to(PROJECT_ROOT)),
            "extended": str(EXTENDED_PATH.relative_to(PROJECT_ROOT)),
            "robustness": str(ROBUSTNESS_PATH.relative_to(PROJECT_ROOT)),
            "drift": str(DRIFT_PATH.relative_to(PROJECT_ROOT)),
        },
    }


def run_all(n_runs: int, drift_runs: int, overwrite_data: bool) -> dict:
    generate_datasets(n_runs=max(n_runs, drift_runs), overwrite=overwrite_data)
    primary = run_primary_suite(n_runs=n_runs)
    save_json(primary, PRIMARY_PATH)

    extended = build_extended_report(primary)
    save_json(extended, EXTENDED_PATH)

    robustness = build_robustness_report(primary, n_runs=n_runs)
    save_json(robustness, ROBUSTNESS_PATH)

    drift = run_drift_suite(n_runs=drift_runs)
    save_json(drift, DRIFT_PATH)

    summary = final_metrics(primary, extended, robustness, drift)
    save_json(summary, RUN_SUMMARY_PATH)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hybrid RL-TOPSIS Amazon India experiments.")
    parser.add_argument("--runs", type=int, default=N_BOOTSTRAP_RUNS, help="Primary/robustness bootstrap runs.")
    parser.add_argument("--drift-runs", type=int, default=N_BOOTSTRAP_DRIFT, help="Drift bootstrap runs.")
    parser.add_argument("--overwrite-data", action="store_true", help="Regenerate bootstrap catalogs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_all(n_runs=args.runs, drift_runs=args.drift_runs, overwrite_data=args.overwrite_data)
    print("=" * 72)
    print("Hybrid RL-TOPSIS Amazon India run summary")
    print("=" * 72)
    for method, value in summary["primary_f1_at_30000"].items():
        print(f"{method:12s}: F1={value:.4f}")
    print(f"Hybrid - TOPSIS: {summary['hybrid_minus_topsis']:+.4f}")
    print(f"Hybrid - RL:     {summary['hybrid_minus_rl']:+.4f}")
    print(f"Saved summary: {RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

