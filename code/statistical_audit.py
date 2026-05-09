"""Paired statistical audit for the Hybrid RL-TOPSIS Amazon India experiment.

This script does not rerun any recommendation experiment. It reads the
per-run/per-profile F1 values stored in amazon_primary.json and writes an
additional distribution-free audit of the headline Hybrid-vs-baseline claims.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PRIMARY_PATH = RESULTS_DIR / "amazon_primary.json"
JSON_OUT = RESULTS_DIR / "statistical_audit.json"
CSV_OUT = RESULTS_DIR / "statistical_audit_summary.csv"

BASELINES = ["random", "popularity", "topsis_only", "rl_only"]
CHECKPOINT = "30000"


def load_run_level_means() -> dict[str, np.ndarray]:
    with PRIMARY_PATH.open("r", encoding="utf-8") as f:
        report = json.load(f)

    values: dict[str, list[float]] = {"hybrid": []}
    values.update({method: [] for method in BASELINES})

    for artifact in report["artifacts"]:
        profile_results = artifact["profile_results"]
        for method in values:
            per_profile = [
                float(profile["f1"][method][CHECKPOINT])
                for profile in profile_results
            ]
            values[method].append(float(np.mean(per_profile)))

    return {method: np.asarray(scores, dtype=float) for method, scores in values.items()}


def audit_pair(hybrid: np.ndarray, baseline: np.ndarray) -> dict:
    diff = hybrid - baseline
    t_test = stats.ttest_rel(hybrid, baseline)
    wilcoxon = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
    sd = float(np.std(diff, ddof=1))
    cohen_dz = float(np.mean(diff) / sd) if sd > 0 else float("inf")

    return {
        "mean_hybrid": float(np.mean(hybrid)),
        "mean_baseline": float(np.mean(baseline)),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "std_diff": sd,
        "cohen_dz": cohen_dz,
        "paired_t_stat": float(t_test.statistic),
        "paired_t_p_two_sided": float(t_test.pvalue),
        "wilcoxon_stat": float(wilcoxon.statistic),
        "wilcoxon_p_one_sided_greater": float(wilcoxon.pvalue),
        "wins": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "losses": int(np.sum(diff < 0)),
        "n": int(diff.size),
    }


def main() -> None:
    scores = load_run_level_means()
    hybrid = scores["hybrid"]
    pairs = {baseline: audit_pair(hybrid, scores[baseline]) for baseline in BASELINES}

    report = {
        "source": str(PRIMARY_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint": CHECKPOINT,
        "unit_of_analysis": "run-level mean F1@7 averaged across five profiles",
        "baselines": BASELINES,
        "bonferroni_alpha": 0.05 / len(BASELINES),
        "pairs": pairs,
    }

    with JSON_OUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fieldnames = [
        "baseline",
        "mean_hybrid",
        "mean_baseline",
        "mean_diff",
        "median_diff",
        "cohen_dz",
        "paired_t_p_two_sided",
        "wilcoxon_p_one_sided_greater",
        "wins",
        "ties",
        "losses",
        "n",
    ]
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for baseline, payload in pairs.items():
            row = {"baseline": baseline}
            row.update({key: payload[key] for key in fieldnames if key != "baseline"})
            writer.writerow(row)

    print(json.dumps(report, indent=2))
    print(f"Saved: {JSON_OUT}")
    print(f"Saved: {CSV_OUT}")


if __name__ == "__main__":
    main()

