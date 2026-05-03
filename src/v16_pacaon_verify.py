"""
Verify that v16 TOPSIS-only baseline (entropy-weighted TOPSIS) is operationally
identical to the canonical Pacaon-Ballera 2024 entropy-EWM-TOPSIS recommender.

Method:
1. Use the same synthetic product catalogs that v16 generates (30 bootstrap seeds).
2. For each catalog, compute rankings via:
   (a) v16's compute_topsis (current TOPSIS-only baseline).
   (b) A fresh "Pacaon-Ballera 2024-style" canonical entropy-EWM-TOPSIS.
3. Report Kendall tau, Spearman rho, and identity rate per seed.
4. Aggregate.
"""

import sys, os
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import kendalltau, spearmanr
import json

from hybrid_rl_mcdm_v2 import (
    generate_products, compute_topsis,
    BOOTSTRAP_RUNS, N_PRODUCTS,
)

def pacaon_ballera_2024(products):
    """
    Canonical entropy-EWM-TOPSIS as per Pacaon & Ballera (2024) IEEE ICSINTESA.
    Algorithm: entropy weighting -> normalize -> TOPSIS closeness coefficient.
    Following Hwang & Yoon (1981) for TOPSIS and Shannon (1948) for entropy.
    """
    n = len(products)
    # 1) Decision matrix: rows=alternatives, cols=criteria
    X = np.array([[p["price_pct"], p["quality_pct"], p["pop_pct"], p["rating_pct"]]
                  for p in products])
    # 2) Cost-benefit normalization: price is cost (lower better) -> invert
    X_proc = X.copy()
    X_proc[:, 0] = 1.0 - X_proc[:, 0]
    # 3) Vector (L2) normalization per column
    norms = np.linalg.norm(X_proc, axis=0)
    norms[norms == 0] = 1.0
    R = X_proc / norms
    # 4) Shannon entropy per criterion
    eps = 1e-12
    k = 1.0 / np.log(n + eps)
    weights = np.zeros(R.shape[1])
    for j in range(R.shape[1]):
        col = R[:, j]
        s = col.sum()
        if s == 0:
            weights[j] = 0.25
            continue
        p_ij = np.clip(col / (s + eps), eps, 1.0)
        ej = -k * np.sum(p_ij * np.log(p_ij))
        weights[j] = 1.0 - ej
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.full(R.shape[1], 0.25)
    # 5) Weighted matrix
    V = R * weights
    # 6) Ideal / anti-ideal (all benefit after price inversion)
    A_pos = V.max(axis=0)
    A_neg = V.min(axis=0)
    # 7) Distances
    d_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    # 8) Closeness
    C_star = d_neg / (d_pos + d_neg + eps)
    return C_star, weights


# Run verification across all 30 bootstrap seeds
results = []
for seed in range(30):
    rng = np.random.default_rng(seed)
    products = generate_products(rng)
    C_v16, w_v16 = compute_topsis(products)
    C_pb, w_pb = pacaon_ballera_2024(products)
    # Compare rankings
    rank_v16 = np.argsort(-C_v16)
    rank_pb = np.argsort(-C_pb)
    tau, p_tau = kendalltau(C_v16, C_pb)
    rho, p_rho = spearmanr(C_v16, C_pb)
    # Identity rate
    score_diff = np.max(np.abs(C_v16 - C_pb))
    weight_diff = np.max(np.abs(w_v16 - w_pb))
    rank_top7_overlap = len(set(rank_v16[:7]) & set(rank_pb[:7])) / 7.0
    results.append({
        "seed": seed,
        "kendall_tau": float(tau),
        "spearman_rho": float(rho),
        "max_score_diff": float(score_diff),
        "max_weight_diff": float(weight_diff),
        "top7_overlap": float(rank_top7_overlap),
    })

# Aggregate
taus = [r["kendall_tau"] for r in results]
rhos = [r["spearman_rho"] for r in results]
score_diffs = [r["max_score_diff"] for r in results]
weight_diffs = [r["max_weight_diff"] for r in results]
top7_overlaps = [r["top7_overlap"] for r in results]

print("=== Pacaon-Ballera 2024 vs v16 TOPSIS-only verification ===")
print(f"Bootstrap seeds: 30")
print(f"Products per seed: {N_PRODUCTS}")
print()
print(f"Kendall tau:    mean={np.mean(taus):.6f}  min={np.min(taus):.6f}  max={np.max(taus):.6f}")
print(f"Spearman rho:   mean={np.mean(rhos):.6f}  min={np.min(rhos):.6f}  max={np.max(rhos):.6f}")
print(f"Max score diff: mean={np.mean(score_diffs):.2e}  max={np.max(score_diffs):.2e}")
print(f"Max weight diff:mean={np.mean(weight_diffs):.2e}  max={np.max(weight_diffs):.2e}")
print(f"Top-7 overlap:  mean={np.mean(top7_overlaps):.6f}  min={np.min(top7_overlaps):.6f}")

# Save
out = {
    "summary": {
        "kendall_tau_mean": float(np.mean(taus)),
        "kendall_tau_min":  float(np.min(taus)),
        "spearman_rho_mean":float(np.mean(rhos)),
        "max_score_diff_max":float(np.max(score_diffs)),
        "max_weight_diff_max":float(np.max(weight_diffs)),
        "top7_overlap_mean":float(np.mean(top7_overlaps)),
        "n_seeds": 30,
        "n_products": N_PRODUCTS,
    },
    "per_seed": results,
    "verdict": ("operationally identical" if np.min(taus) > 0.999 else "differ"),
}
with open('results/v16_pacaon_verify.json','w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: results/v16_pacaon_verify.json")
print(f"Verdict: {out['verdict']}")
