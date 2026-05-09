"""
Real SHAP explainability analysis for the Hybrid RL-TOPSIS Amazon India
experiment.

The report combines intrinsic RL-vs-TOPSIS branch contributions with genuine
TreeExplainer SHAP values from a random-forest surrogate trained to approximate
the final hybrid ranking score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The Hybrid RL-TOPSIS XAI analysis requires the real 'shap' package. Install it with: "
        "python -m pip install shap"
    ) from exc

from hybrid_core import MAIN_LAMBDA_Q, MAIN_LAMBDA_T, norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_PATH = PROJECT_ROOT / "results" / "amazon_primary.json"
OUT_DIR = PROJECT_ROOT / "results" / "xai"
OUT_JSON = OUT_DIR / "xai_report.json"
GLOBAL_CSV = OUT_DIR / "global_importance.csv"
LOCAL_CSV = OUT_DIR / "local_top7_explanations.csv"
LOCAL_SHAP_CSV = OUT_DIR / "local_top7_shap_values.csv"
SHAP_VALUES_CSV = OUT_DIR / "sample_shap_values.csv"
COUNTERFACTUAL_CSV = OUT_DIR / "counterfactual_rank_shifts.csv"

FEATURES = [
    "price_pct",
    "quality_pct",
    "popularity_pct",
    "rating_pct",
    "recency_pct",
    "discount_pct",
    "rating_count",
    "inferred_reviewer_count",
    "review_text_richness",
]
MCDM_FEATURES = ["price_pct", "quality_pct", "popularity_pct", "rating_pct"]


def load_primary() -> dict:
    with PRIMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_frame(run_artifact: dict) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / run_artifact["dataset_path"])


def branch_frame(primary: dict, max_runs: int | None = None) -> pd.DataFrame:
    rows = []
    artifacts = primary["artifacts"][:max_runs] if max_runs else primary["artifacts"]
    for run in artifacts:
        df = load_run_frame(run)
        for profile in run["profile_results"]:
            final = profile["final"]
            q_norm = norm(np.asarray(final["q_scores"], dtype=float))
            t_norm = norm(np.asarray(final["topsis_scores"], dtype=float))
            hybrid = MAIN_LAMBDA_Q * q_norm + MAIN_LAMBDA_T * t_norm
            ranks = np.empty(len(hybrid), dtype=int)
            ranks[np.argsort(hybrid)[::-1]] = np.arange(1, len(hybrid) + 1)
            top_items = set(int(x) for x in final["hybrid_set"])
            gt_items = set(int(x) for x in final["gt_set"])

            for idx, item in df.iterrows():
                rows.append(
                    {
                        "run_index": int(run["run_index"]),
                        "profile_name": profile["profile_name"],
                        "item_index": int(idx),
                        "product_id": str(item.get("product_id", idx)),
                        "product_name": str(item.get("product_name", ""))[:160],
                        "category": str(item.get("category", "")),
                        "brand": str(item.get("brand", "")),
                        "hybrid_score": float(hybrid[idx]),
                        "q_component": float(MAIN_LAMBDA_Q * q_norm[idx]),
                        "topsis_component": float(MAIN_LAMBDA_T * t_norm[idx]),
                        "hybrid_rank": int(ranks[idx]),
                        "in_hybrid_top7": bool(idx in top_items),
                        "in_ground_truth_top7": bool(idx in gt_items),
                        **{feature: float(item[feature]) for feature in FEATURES if feature in item},
                    }
                )
    return pd.DataFrame(rows)


def compute_shap_values(
    forest: RandomForestRegressor,
    x: pd.DataFrame,
    data: pd.DataFrame,
    seed: int,
    sample_size: int,
) -> pd.DataFrame:
    sample_n = min(sample_size, len(x))
    sample = x.sample(n=sample_n, random_state=seed)
    meta_cols = ["run_index", "profile_name", "item_index", "product_id", "hybrid_score", "hybrid_rank", "in_hybrid_top7"]
    meta = data.loc[sample.index, meta_cols].copy().reset_index(drop=True)
    explainer = shap.TreeExplainer(forest)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[0]

    out = meta
    for idx, feature in enumerate(FEATURES):
        out[f"value_{feature}"] = sample[feature].to_numpy(dtype=float)
        out[f"shap_{feature}"] = values[:, idx].astype(float)
    out["shap_base_value"] = float(np.ravel(explainer.expected_value)[0])
    out["shap_sum_plus_base"] = out["shap_base_value"] + out[[f"shap_{feature}" for feature in FEATURES]].sum(axis=1)
    return out


def fit_surrogates(data: pd.DataFrame, seed: int, shap_sample: int) -> tuple[dict, pd.DataFrame, object, pd.DataFrame]:
    x = data[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = data["hybrid_score"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(x_train, y_train)
    ridge_r2 = float(r2_score(y_test, ridge.predict(x_test)))

    forest = RandomForestRegressor(n_estimators=250, min_samples_leaf=10, random_state=seed, n_jobs=-1)
    forest.fit(x_train, y_train)
    forest_r2 = float(r2_score(y_test, forest.predict(x_test)))
    perm = permutation_importance(forest, x_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1)
    shap_frame = compute_shap_values(forest, x, data, seed=seed, sample_size=shap_sample)
    mean_abs_shap = shap_frame[[f"shap_{feature}" for feature in FEATURES]].abs().mean()

    ridge_model = ridge.named_steps["ridge"]
    scaler = ridge.named_steps["standardscaler"]
    rows = []
    for idx, feature in enumerate(FEATURES):
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(mean_abs_shap[f"shap_{feature}"]),
                "ridge_coefficient": float(ridge_model.coef_[idx]),
                "ridge_abs_coefficient": float(abs(ridge_model.coef_[idx])),
                "rf_impurity_importance": float(forest.feature_importances_[idx]),
                "rf_permutation_importance_mean": float(perm.importances_mean[idx]),
                "rf_permutation_importance_std": float(perm.importances_std[idx]),
                "feature_mean": float(x[feature].mean()),
                "feature_std": float(x[feature].std()),
                "scaler_mean": float(scaler.mean_[idx]),
                "scaler_scale": float(scaler.scale_[idx]),
            }
        )

    diagnostics = {
        "n_rows": int(len(data)),
        "n_features": int(len(FEATURES)),
        "ridge_r2_test": ridge_r2,
        "random_forest_r2_test": forest_r2,
        "shap_version": str(shap.__version__),
        "shap_sample_rows": int(len(shap_frame)),
        "target": "final hybrid score from the trained RL-TOPSIS policy",
        "surrogate_note": (
            "Random forest surrogate is used for TreeExplainer SHAP and permutation importance. "
            "Standardized ridge is retained as a linear sanity-check surrogate."
        ),
    }
    return diagnostics, pd.DataFrame(rows), ridge, shap_frame


def local_explanations(data: pd.DataFrame, ridge, shap_frame: pd.DataFrame, run_index: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = data[(data["run_index"] == run_index) & (data["in_hybrid_top7"])].copy()
    scaler = ridge.named_steps["standardscaler"]
    model = ridge.named_steps["ridge"]
    shap_lookup = shap_frame.set_index(["run_index", "profile_name", "item_index"], drop=False)
    rows = []
    shap_rows = []

    for _, row in subset.sort_values(["profile_name", "hybrid_rank"]).iterrows():
        values = row[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        z = (values - scaler.mean_) / scaler.scale_
        linear_contrib = z * model.coef_
        key = (int(row["run_index"]), row["profile_name"], int(row["item_index"]))

        if key in shap_lookup.index:
            srow = shap_lookup.loc[key]
            if isinstance(srow, pd.DataFrame):
                srow = srow.iloc[0]
            shap_vals = np.array([float(srow[f"shap_{feature}"]) for feature in FEATURES])
            top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
            top_shap_features = "; ".join(f"{FEATURES[i]}={shap_vals[i]:+.4f}" for i in top_idx)
            for i, feature in enumerate(FEATURES):
                shap_rows.append(
                    {
                        "run_index": int(row["run_index"]),
                        "profile_name": row["profile_name"],
                        "hybrid_rank": int(row["hybrid_rank"]),
                        "product_id": row["product_id"],
                        "feature": feature,
                        "feature_value": float(row[feature]),
                        "shap_value": float(shap_vals[i]),
                    }
                )
        else:
            top_shap_features = "not_sampled"

        rows.append(
            {
                "run_index": int(row["run_index"]),
                "profile_name": row["profile_name"],
                "hybrid_rank": int(row["hybrid_rank"]),
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "hybrid_score": float(row["hybrid_score"]),
                "q_component": float(row["q_component"]),
                "topsis_component": float(row["topsis_component"]),
                "dominant_branch": "RL" if row["q_component"] >= row["topsis_component"] else "TOPSIS",
                "in_ground_truth_top7": bool(row["in_ground_truth_top7"]),
                "top_shap_features": top_shap_features,
                "linear_sanity_check_features": "; ".join(
                    f"{FEATURES[i]}={linear_contrib[i]:+.4f}" for i in np.argsort(np.abs(linear_contrib))[::-1][:5]
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(shap_rows)


def counterfactual_rank_shifts(primary: dict, max_runs: int = 10) -> pd.DataFrame:
    rows = []
    for run in primary["artifacts"][:max_runs]:
        df = load_run_frame(run)
        feature_matrix = df[MCDM_FEATURES].to_numpy(dtype=float)
        for profile in run["profile_results"]:
            final = profile["final"]
            q_norm = norm(np.asarray(final["q_scores"], dtype=float))
            topsis_weights = np.asarray(final["topsis_weights"], dtype=float)
            base_topsis = np.asarray(final["topsis_scores"], dtype=float)
            base_hybrid = MAIN_LAMBDA_Q * q_norm + MAIN_LAMBDA_T * norm(base_topsis)
            base_ranks = np.empty(len(base_hybrid), dtype=int)
            base_ranks[np.argsort(base_hybrid)[::-1]] = np.arange(1, len(base_hybrid) + 1)
            tracked = sorted(set(int(x) for x in final["hybrid_set"]) | set(int(x) for x in final["gt_set"]))

            for feature in MCDM_FEATURES:
                col = MCDM_FEATURES.index(feature)
                perturbed = feature_matrix.copy()
                perturbed[:, col] = np.clip(perturbed[:, col] + 0.10, 0.0, 1.0)
                weighted = perturbed * topsis_weights
                ideal_plus = weighted.max(axis=0)
                ideal_minus = weighted.min(axis=0)
                d_plus = np.sqrt(((weighted - ideal_plus) ** 2).sum(axis=1))
                d_minus = np.sqrt(((weighted - ideal_minus) ** 2).sum(axis=1))
                perturbed_topsis = d_minus / np.clip(d_plus + d_minus, 1e-10, None)
                perturbed_hybrid = MAIN_LAMBDA_Q * q_norm + MAIN_LAMBDA_T * norm(perturbed_topsis)
                perturbed_ranks = np.empty(len(perturbed_hybrid), dtype=int)
                perturbed_ranks[np.argsort(perturbed_hybrid)[::-1]] = np.arange(1, len(perturbed_hybrid) + 1)

                for idx in tracked:
                    rows.append(
                        {
                            "run_index": int(run["run_index"]),
                            "profile_name": profile["profile_name"],
                            "item_index": int(idx),
                            "product_id": str(df.loc[idx, "product_id"]) if "product_id" in df else str(idx),
                            "feature_perturbed": feature,
                            "delta_feature": 0.10,
                            "base_rank": int(base_ranks[idx]),
                            "perturbed_rank": int(perturbed_ranks[idx]),
                            "rank_shift_positive_is_improvement": int(base_ranks[idx] - perturbed_ranks[idx]),
                            "base_score": float(base_hybrid[idx]),
                            "perturbed_score": float(perturbed_hybrid[idx]),
                            "score_delta": float(perturbed_hybrid[idx] - base_hybrid[idx]),
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_counterfactual(cf: pd.DataFrame) -> dict:
    out = {}
    for feature, frame in cf.groupby("feature_perturbed"):
        shifts = frame["rank_shift_positive_is_improvement"].to_numpy(dtype=float)
        deltas = frame["score_delta"].to_numpy(dtype=float)
        out[str(feature)] = {
            "mean_rank_shift": float(np.mean(shifts)),
            "median_rank_shift": float(np.median(shifts)),
            "mean_score_delta": float(np.mean(deltas)),
            "share_improved": float(np.mean(shifts > 0)),
            "share_worsened": float(np.mean(shifts < 0)),
        }
    return out


def branch_summary(data: pd.DataFrame) -> dict:
    top = data[data["in_hybrid_top7"]]
    return {
        "all_items": {
            "mean_q_component": float(data["q_component"].mean()),
            "mean_topsis_component": float(data["topsis_component"].mean()),
            "q_dominant_share": float(np.mean(data["q_component"] >= data["topsis_component"])),
        },
        "hybrid_top7_items": {
            "mean_q_component": float(top["q_component"].mean()),
            "mean_topsis_component": float(top["topsis_component"].mean()),
            "q_dominant_share": float(np.mean(top["q_component"] >= top["topsis_component"])),
            "gt_hit_share": float(top["in_ground_truth_top7"].mean()),
        },
    }


def run(max_runs: int | None, seed: int, shap_sample: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = load_primary()
    data = branch_frame(primary, max_runs=max_runs)
    diagnostics, global_imp, ridge, shap_frame = fit_surrogates(data, seed, shap_sample=shap_sample)
    local, local_shap = local_explanations(data, ridge, shap_frame)
    cf = counterfactual_rank_shifts(primary, max_runs=min(max_runs or 10, 10))

    global_imp = global_imp.sort_values("mean_abs_shap", ascending=False)
    global_imp.to_csv(GLOBAL_CSV, index=False)
    local.to_csv(LOCAL_CSV, index=False)
    local_shap.to_csv(LOCAL_SHAP_CSV, index=False)
    shap_frame.to_csv(SHAP_VALUES_CSV, index=False)
    cf.to_csv(COUNTERFACTUAL_CSV, index=False)

    report = {
        "diagnostics": diagnostics,
        "branch_summary": branch_summary(data),
        "global_importance_top": global_imp.head(10).to_dict(orient="records"),
        "local_explanation_rows": int(len(local)),
        "local_shap_rows": int(len(local_shap)),
        "counterfactual_summary": aggregate_counterfactual(cf),
        "outputs": {
            "global_importance_csv": str(GLOBAL_CSV.relative_to(PROJECT_ROOT)),
            "local_top7_explanations_csv": str(LOCAL_CSV.relative_to(PROJECT_ROOT)),
            "local_top7_shap_values_csv": str(LOCAL_SHAP_CSV.relative_to(PROJECT_ROOT)),
            "sample_shap_values_csv": str(SHAP_VALUES_CSV.relative_to(PROJECT_ROOT)),
            "counterfactual_rank_shifts_csv": str(COUNTERFACTUAL_CSV.relative_to(PROJECT_ROOT)),
        },
        "interpretation_guardrail": (
            "These are TreeExplainer SHAP explanations of a random-forest surrogate for final hybrid scores, "
            "combined with intrinsic RL-vs-TOPSIS branch contributions. They are not presented as causal "
            "explanations of user behavior."
        ),
    }
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hybrid RL-TOPSIS real SHAP XAI analysis.")
    parser.add_argument("--max-runs", type=int, default=50, help="Number of primary bootstrap runs to include.")
    parser.add_argument("--shap-sample", type=int, default=5000, help="Rows sampled for TreeExplainer SHAP values.")
    parser.add_argument("--seed", type=int, default=20260509)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(max_runs=args.max_runs, seed=args.seed, shap_sample=args.shap_sample)
    print("=" * 72)
    print("Hybrid RL-TOPSIS real SHAP XAI report")
    print("=" * 72)
    print(f"Rows explained: {report['diagnostics']['n_rows']}")
    print(f"Ridge surrogate R2: {report['diagnostics']['ridge_r2_test']:.4f}")
    print(f"RF surrogate R2: {report['diagnostics']['random_forest_r2_test']:.4f}")
    print(f"SHAP version: {report['diagnostics']['shap_version']}")
    print(f"SHAP sample rows: {report['diagnostics']['shap_sample_rows']}")
    print("Top global features:")
    for row in report["global_importance_top"][:6]:
        print(f"  {row['feature']:24s} mean_abs_shap={row['mean_abs_shap']:.5f}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()

