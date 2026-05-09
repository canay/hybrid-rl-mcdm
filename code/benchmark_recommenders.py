"""
Classical recommender benchmarks for the Hybrid RL-TOPSIS Amazon India package.

The main Hybrid RL-TOPSIS experiment is a catalog-level decision simulation.
This benchmark uses the reviewer-product associations embedded in the raw
Amazon India CSV as a sparse implicit-feedback graph and evaluates common
recommender baselines with repeated leave-one-out splits.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from run_amazon_experiments import PROJECT_ROOT, load_enriched_catalog
from hybrid_core import norm, topsis_artifacts


OUT_JSON = PROJECT_ROOT / "results" / "recommender_benchmarks.json"
OUT_CSV = PROJECT_ROOT / "results" / "recommender_benchmarks_summary.csv"
K = 10


def summarize(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n": int(arr.size),
        "raw": [float(x) for x in arr],
    }


def build_interactions(min_items: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "amazon_india.csv")
    enriched = load_enriched_catalog()
    valid_items = set(enriched["product_id"].astype(str))

    user_items: dict[str, set[str]] = defaultdict(set)
    for _, row in raw.iterrows():
        product_id = str(row["product_id"])
        if product_id not in valid_items:
            continue
        for user_id in str(row["user_id"]).split(","):
            user_id = user_id.strip()
            if user_id:
                user_items[user_id].add(product_id)

    eligible = {
        user_id: sorted(items)
        for user_id, items in user_items.items()
        if len(items) >= min_items
    }
    return enriched, eligible


def holdout_split(user_items: dict[str, list[str]], seed: int) -> tuple[dict[str, list[str]], dict[str, str]]:
    rng = np.random.RandomState(seed)
    train: dict[str, list[str]] = {}
    test: dict[str, str] = {}
    for user_id, items in user_items.items():
        items = list(items)
        test_item = items[int(rng.randint(0, len(items)))]
        train[user_id] = [item for item in items if item != test_item]
        test[user_id] = test_item
    return train, test


def build_matrix(train: dict[str, list[str]], item_ids: list[str]) -> tuple[list[str], dict[str, int], np.ndarray]:
    users = sorted(train)
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    matrix = np.zeros((len(users), len(item_ids)), dtype=np.float32)
    for u_idx, user_id in enumerate(users):
        for item in train[user_id]:
            if item in item_to_idx:
                matrix[u_idx, item_to_idx[item]] = 1.0
    return users, item_to_idx, matrix


def top_k(scores: np.ndarray, seen: set[int], k: int = K) -> list[int]:
    scores = scores.copy()
    if seen:
        scores[list(seen)] = -np.inf
    return [int(x) for x in np.argsort(scores)[::-1][:k]]


def rank_metrics(ranked: list[int], truth_idx: int) -> dict:
    if truth_idx not in ranked:
        return {"hit": 0.0, "ndcg": 0.0, "mrr": 0.0}
    rank = ranked.index(truth_idx) + 1
    return {
        "hit": 1.0,
        "ndcg": float(1.0 / np.log2(rank + 1.0)),
        "mrr": float(1.0 / rank),
    }


def item_feature_matrix(enriched: pd.DataFrame, item_ids: list[str]) -> np.ndarray:
    frame = enriched.set_index("product_id").loc[item_ids]
    cols = ["price_pct", "quality_pct", "popularity_pct", "rating_pct", "recency_pct"]
    features = frame[cols].to_numpy(dtype=float)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def bpr_scores(
    train_matrix: np.ndarray,
    seed: int,
    factors: int,
    epochs: int,
    lr: float,
    reg: float,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n_users, n_items = train_matrix.shape
    user_factors = 0.05 * rng.normal(size=(n_users, factors))
    item_factors = 0.05 * rng.normal(size=(n_items, factors))
    positives = [np.flatnonzero(train_matrix[u] > 0) for u in range(n_users)]
    all_items = np.arange(n_items)

    for _ in range(epochs):
        for u in rng.permutation(n_users):
            pos_items = positives[u]
            if pos_items.size == 0:
                continue
            i = int(rng.choice(pos_items))
            neg_pool = all_items[train_matrix[u] == 0]
            if neg_pool.size == 0:
                continue
            j = int(rng.choice(neg_pool))
            x = float(user_factors[u] @ (item_factors[i] - item_factors[j]))
            grad = 1.0 / (1.0 + np.exp(x))
            u_old = user_factors[u].copy()
            i_old = item_factors[i].copy()
            j_old = item_factors[j].copy()
            user_factors[u] += lr * (grad * (i_old - j_old) - reg * u_old)
            item_factors[i] += lr * (grad * u_old - reg * i_old)
            item_factors[j] += lr * (-grad * u_old - reg * j_old)
    return user_factors @ item_factors.T


def evaluate_once(
    enriched: pd.DataFrame,
    user_items: dict[str, list[str]],
    seed: int,
    factors: int,
    epochs: int,
) -> tuple[dict, dict]:
    train, test = holdout_split(user_items, seed)
    item_ids = sorted(enriched["product_id"].astype(str).unique())
    users, item_to_idx, train_matrix = build_matrix(train, item_ids)
    feature_matrix = item_feature_matrix(enriched, item_ids)
    enriched_ordered = enriched.set_index("product_id").loc[item_ids].reset_index()

    popularity_scores = train_matrix.sum(axis=0)
    topsis_scores = topsis_artifacts(enriched_ordered)["scores"]
    random_rng = np.random.RandomState(seed + 9000)

    item_norm = train_matrix.T
    item_den = np.linalg.norm(item_norm, axis=1, keepdims=True)
    item_den[item_den == 0] = 1.0
    item_sim = (item_norm / item_den) @ (item_norm / item_den).T
    np.fill_diagonal(item_sim, 0.0)

    user_den = np.linalg.norm(train_matrix, axis=1, keepdims=True)
    user_den[user_den == 0] = 1.0
    user_sim = (train_matrix / user_den) @ (train_matrix / user_den).T
    np.fill_diagonal(user_sim, 0.0)

    bpr_start = time.perf_counter()
    bpr_score_matrix = bpr_scores(train_matrix, seed + 12000, factors=factors, epochs=epochs, lr=0.05, reg=0.002)
    bpr_runtime = time.perf_counter() - bpr_start

    metrics = {
        name: {"hit": [], "ndcg": [], "mrr": [], "coverage_items": set()}
        for name in ["random", "popularity", "topsis", "content_cbf", "item_knn", "user_knn", "bpr_mf"]
    }
    timings = defaultdict(float)
    timings["bpr_mf_train_seconds"] = bpr_runtime

    for u_idx, user_id in enumerate(users):
        truth = item_to_idx[test[user_id]]
        seen = {item_to_idx[item] for item in train[user_id] if item in item_to_idx}

        model_scores: dict[str, np.ndarray] = {}
        t0 = time.perf_counter()
        model_scores["random"] = random_rng.random(len(item_ids))
        timings["random_score_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        model_scores["popularity"] = popularity_scores.astype(float)
        timings["popularity_score_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        model_scores["topsis"] = topsis_scores
        timings["topsis_score_seconds"] += time.perf_counter() - t0

        train_idx = list(seen)
        t0 = time.perf_counter()
        if train_idx:
            centroid = feature_matrix[train_idx].mean(axis=0)
            den = np.linalg.norm(centroid)
            model_scores["content_cbf"] = feature_matrix @ (centroid / den if den > 0 else centroid)
        else:
            model_scores["content_cbf"] = np.zeros(len(item_ids))
        timings["content_cbf_score_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        model_scores["item_knn"] = item_sim[:, train_idx].sum(axis=1) if train_idx else np.zeros(len(item_ids))
        timings["item_knn_score_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        model_scores["user_knn"] = user_sim[u_idx] @ train_matrix
        timings["user_knn_score_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        model_scores["bpr_mf"] = bpr_score_matrix[u_idx]
        timings["bpr_mf_score_seconds"] += time.perf_counter() - t0

        for model_name, scores in model_scores.items():
            ranked = top_k(scores, seen)
            row = rank_metrics(ranked, truth)
            for metric_name, value in row.items():
                metrics[model_name][metric_name].append(value)
            metrics[model_name]["coverage_items"].update(ranked)

    summary = {}
    for model_name, payload in metrics.items():
        summary[model_name] = {
            "hit_at_10": float(np.mean(payload["hit"])),
            "ndcg_at_10": float(np.mean(payload["ndcg"])),
            "mrr_at_10": float(np.mean(payload["mrr"])),
            "coverage_at_10": float(len(payload["coverage_items"]) / len(item_ids)),
        }
    return summary, dict(timings)


def run_benchmarks(runs: int, min_items: int, factors: int, epochs: int) -> dict:
    enriched, user_items = build_interactions(min_items=min_items)
    raw_by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    raw_timing: dict[str, list[float]] = defaultdict(list)

    print("=" * 72)
    print("Hybrid RL-TOPSIS sparse implicit recommender benchmark")
    print("=" * 72)
    print(f"Eligible users: {len(user_items)}")
    print(f"Runs: {runs}; min_items_per_user: {min_items}; BPR epochs: {epochs}")

    for run in range(runs):
        start = time.perf_counter()
        summary, timings = evaluate_once(enriched, user_items, seed=2718 + run * 37, factors=factors, epochs=epochs)
        elapsed = time.perf_counter() - start
        for model_name, metrics in summary.items():
            for metric_name, value in metrics.items():
                raw_by_model[model_name][metric_name].append(value)
        for timing_key, value in timings.items():
            raw_timing[timing_key].append(value)
        raw_timing["total_run_seconds"].append(elapsed)
        if (run + 1) % 5 == 0 or run == 0:
            best = max(summary, key=lambda name: summary[name]["ndcg_at_10"])
            print(f"Run {run + 1}/{runs}: best={best}, ndcg@10={summary[best]['ndcg_at_10']:.4f}, seconds={elapsed:.2f}")

    report = {
        "config": {
            "dataset": "Amazon India reviewer-product implicit graph derived from data/raw/amazon_india.csv",
            "runs": runs,
            "min_items_per_user": min_items,
            "k": K,
            "bpr_factors": factors,
            "bpr_epochs": epochs,
            "evaluation": "Repeated random leave-one-out over users with at least min_items interactions.",
            "limitation": "The raw CSV has no timestamps, so this is not a temporal next-item benchmark.",
        },
        "n_eligible_users": len(user_items),
        "summary": {
            model_name: {metric_name: summarize(values) for metric_name, values in metrics.items()}
            for model_name, metrics in raw_by_model.items()
        },
        "runtime_seconds": {key: summarize(values) for key, values in raw_timing.items()},
    }
    return report


def save_report(report: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    rows = []
    for model_name, metrics in report["summary"].items():
        row = {"model": model_name}
        for metric_name, stats in metrics.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
        rows.append(row)
    pd.DataFrame(rows).sort_values("ndcg_at_10_mean", ascending=False).to_csv(OUT_CSV, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hybrid RL-TOPSIS classical recommender benchmarks.")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--min-items", type=int, default=3)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_benchmarks(args.runs, args.min_items, args.factors, args.epochs)
    save_report(report)
    print("=" * 72)
    print("Benchmark summary")
    print("=" * 72)
    for model_name, metrics in sorted(
        report["summary"].items(),
        key=lambda item: item[1]["ndcg_at_10"]["mean"],
        reverse=True,
    ):
        print(
            f"{model_name:12s}: "
            f"Hit@10={metrics['hit_at_10']['mean']:.4f}, "
            f"NDCG@10={metrics['ndcg_at_10']['mean']:.4f}, "
            f"MRR@10={metrics['mrr_at_10']['mean']:.4f}"
        )
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()

