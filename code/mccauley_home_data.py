"""
home_real_data.py
=================

Preprocess the Amazon Home and Kitchen public dataset into a clean external
validation artifact for the hybrid recommendation study.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "amazon_mccauley_home"
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "amazon_mccauley_home"

RAW_META_PATH = RAW_DIR / "meta_Home_and_Kitchen.json.gz"
RAW_REVIEW_PATH = RAW_DIR / "reviews_Home_and_Kitchen_5.json.gz"

ITEMS_PATH = PROC_DIR / "items.csv"
USERS_PATH = PROC_DIR / "users.json"
MANIFEST_PATH = PROC_DIR / "manifest.json"

MIN_UNIQUE_ITEMS = 25
MAX_USERS = 1000
TEST_K = 7
MIN_POSITIVE_RATING = 4.0


def parse_eval_gzip(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield ast.literal_eval(line)


def extract_sales_rank(payload: dict) -> float | None:
    sales_rank = payload.get("salesRank")
    if isinstance(sales_rank, dict) and sales_rank:
        values = [float(v) for v in sales_rank.values() if isinstance(v, (int, float))]
        if values:
            return min(values)
    return None


def extract_categories(payload: dict) -> tuple[str, str]:
    categories = payload.get("categories") or []
    if not categories:
        return "Unknown", "Unknown"

    path = categories[0]
    if not path:
        return "Unknown", "Unknown"

    level2 = str(path[1]) if len(path) >= 2 else str(path[0])
    leaf = str(path[-1])
    return level2, leaf


def load_metadata() -> pd.DataFrame:
    rows = []
    for payload in parse_eval_gzip(RAW_META_PATH):
        price = payload.get("price")
        sales_rank = extract_sales_rank(payload)
        level2, leaf = extract_categories(payload)
        if not isinstance(price, (int, float)):
            continue
        if sales_rank is None:
            continue
        rows.append(
            {
                "asin": str(payload["asin"]),
                "price": float(price),
                "sales_rank": float(sales_rank),
                "brand": str(payload.get("brand") or "Unknown"),
                "cat_lvl2": level2,
                "cat_leaf": leaf,
            }
        )

    meta = pd.DataFrame(rows).drop_duplicates(subset=["asin"]).reset_index(drop=True)
    return meta


def load_reviews(valid_asins: set[str], min_rating: float = MIN_POSITIVE_RATING) -> pd.DataFrame:
    rows = []
    for payload in parse_eval_gzip(RAW_REVIEW_PATH):
        asin = str(payload["asin"])
        if asin not in valid_asins:
            continue
        rating = float(payload["overall"])
        if rating < min_rating:
            continue
        rows.append(
            {
                "reviewerID": str(payload["reviewerID"]),
                "asin": asin,
                "overall": rating,
                "unixReviewTime": int(payload["unixReviewTime"]),
            }
        )
    return pd.DataFrame(rows)


def build_unique_sequences(reviews: pd.DataFrame) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for reviewer_id, frame in reviews.groupby("reviewerID"):
        ordered = frame.sort_values(["unixReviewTime", "asin"]).to_dict(orient="records")
        unique_rows = []
        seen: set[str] = set()
        for row in ordered:
            asin = str(row["asin"])
            if asin in seen:
                continue
            seen.add(asin)
            unique_rows.append(row)
        grouped[str(reviewer_id)] = unique_rows
    return grouped


def select_users(unique_sequences: dict[str, list[dict]], max_users: int, min_unique_items: int) -> dict[str, list[dict]]:
    eligible = [(user_id, rows) for user_id, rows in unique_sequences.items() if len(rows) >= min_unique_items]
    eligible.sort(key=lambda item: len(item[1]), reverse=True)
    return {user_id: rows for user_id, rows in eligible[:max_users]}


def global_train_item_stats(selected_sequences: dict[str, list[dict]]) -> pd.DataFrame:
    train_rows = []
    for user_id, rows in selected_sequences.items():
        train_rows.extend(rows[:-TEST_K])

    train = pd.DataFrame(train_rows)
    grouped = train.groupby("asin")
    stats = grouped.agg(
        popularity=("asin", "size"),
        rating_mean=("overall", "mean"),
        last_train_time=("unixReviewTime", "max"),
    ).reset_index()
    return stats


def percentile_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    ranked = series.rank(pct=True)
    return 1.0 - ranked if invert else ranked


def build_item_table(meta: pd.DataFrame, selected_sequences: dict[str, list[dict]]) -> pd.DataFrame:
    selected_items = {
        row["asin"]
        for rows in selected_sequences.values()
        for row in rows
    }

    item_stats = global_train_item_stats(selected_sequences)
    item_table = meta[meta["asin"].isin(selected_items)].merge(item_stats, on="asin", how="left")

    global_rating = float(item_table["rating_mean"].dropna().median()) if item_table["rating_mean"].notna().any() else 3.5
    global_time = float(item_table["last_train_time"].dropna().min()) if item_table["last_train_time"].notna().any() else 0.0

    item_table["popularity"] = item_table["popularity"].fillna(0.0)
    item_table["rating_mean"] = item_table["rating_mean"].fillna(global_rating)
    item_table["last_train_time"] = item_table["last_train_time"].fillna(global_time)

    item_table["price_pct"] = percentile_rank(item_table["price"], invert=True)
    item_table["quality_pct"] = percentile_rank(item_table["sales_rank"], invert=True)
    item_table["popularity_pct"] = percentile_rank(item_table["popularity"], invert=False)
    item_table["rating_pct"] = percentile_rank(item_table["rating_mean"], invert=False)
    item_table["recency_pct"] = percentile_rank(item_table["last_train_time"], invert=False)

    return item_table.sort_values("asin").reset_index(drop=True)


def infer_user_profile(rows: list[dict], item_lookup: pd.DataFrame) -> dict:
    train_rows = rows[:-TEST_K]
    train_asins = [row["asin"] for row in train_rows]
    train_items = item_lookup.loc[train_asins]

    lvl2_counts = train_items["cat_lvl2"].value_counts(normalize=True).to_dict()
    leaf_counts = train_items["cat_leaf"].value_counts(normalize=True).to_dict()

    prices = train_items["price"].to_numpy(dtype=float)
    log_prices = np.log1p(prices)
    center = float(np.median(log_prices))
    spread = float(np.percentile(log_prices, 75) - np.percentile(log_prices, 25))
    if spread < 0.15:
        spread = 0.15

    recency_pref = float(train_items["recency_pct"].mean())

    return {
        "cat_lvl2_affinity": {str(k): float(v) for k, v in lvl2_counts.items()},
        "cat_leaf_affinity": {str(k): float(v) for k, v in leaf_counts.items()},
        "log_price_center": center,
        "log_price_spread": spread,
        "recency_pref": recency_pref,
    }


def build_user_payloads(selected_sequences: dict[str, list[dict]], item_table: pd.DataFrame) -> list[dict]:
    item_lookup = item_table.set_index("asin", drop=False)
    payloads = []

    for idx, (_, rows) in enumerate(selected_sequences.items()):
        payloads.append(
            {
                "user_id": f"user_{idx:04d}",
                "n_unique_items": len(rows),
                "train": [
                    {
                        "asin": str(row["asin"]),
                        "rating": float(row["overall"]),
                        "time": int(row["unixReviewTime"]),
                    }
                    for row in rows[:-TEST_K]
                ],
                "test": [
                    {
                        "asin": str(row["asin"]),
                        "rating": float(row["overall"]),
                        "time": int(row["unixReviewTime"]),
                    }
                    for row in rows[-TEST_K:]
                ],
                "profile": infer_user_profile(rows, item_lookup),
            }
        )

    return payloads


def save_outputs(item_table: pd.DataFrame, user_payloads: list[dict], max_users: int, min_unique_items: int) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    item_table.to_csv(ITEMS_PATH, index=False)
    with USERS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(user_payloads, handle, indent=2)

    manifest = {
        "source": "Amazon productGraph 2014 / Home and Kitchen 5-core",
        "raw_meta": str(RAW_META_PATH.relative_to(PROJECT_ROOT)),
        "raw_reviews": str(RAW_REVIEW_PATH.relative_to(PROJECT_ROOT)),
        "n_items": int(len(item_table)),
        "n_users": int(len(user_payloads)),
        "test_k": TEST_K,
        "min_unique_items": int(min_unique_items),
        "max_users": int(max_users),
        "min_positive_rating": float(MIN_POSITIVE_RATING),
        "privacy_publication_note": "Public user identifiers are anonymized; raw reviewerID values are not written to the processed user file.",
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def build_processed_dataset(max_users: int = MAX_USERS, min_unique_items: int = MIN_UNIQUE_ITEMS) -> None:
    meta = load_metadata()
    reviews = load_reviews(set(meta["asin"]))
    unique_sequences = build_unique_sequences(reviews)
    selected = select_users(unique_sequences, max_users=max_users, min_unique_items=min_unique_items)
    item_table = build_item_table(meta, selected)
    selected = {
        user_id: [row for row in rows if row["asin"] in set(item_table["asin"])]
        for user_id, rows in selected.items()
    }
    selected = {user_id: rows for user_id, rows in selected.items() if len(rows) >= min_unique_items}
    item_table = build_item_table(meta, selected)
    user_payloads = build_user_payloads(selected, item_table)
    save_outputs(item_table, user_payloads, max_users=max_users, min_unique_items=min_unique_items)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the processed Amazon Home and Kitchen external-validation dataset.")
    parser.add_argument("--max-users", type=int, default=MAX_USERS, help="Maximum number of active users to retain.")
    parser.add_argument("--min-unique-items", type=int, default=MIN_UNIQUE_ITEMS, help="Minimum unique interactions per user.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_processed_dataset(max_users=args.max_users, min_unique_items=args.min_unique_items)
    print(f"Wrote {ITEMS_PATH}")
    print(f"Wrote {USERS_PATH}")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

