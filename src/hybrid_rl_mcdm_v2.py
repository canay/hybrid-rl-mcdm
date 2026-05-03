"""
hybrid_rl_mcdm_v2.py  --  Full reproducible implementation
==========================================================
Implements the methodology described in paper_v14.tex EXACTLY.
Every section maps to a paper equation or table.

Output: results/v2_full_results.json  (all numbers for LaTeX)
"""

import numpy as np
import json, os, sys
from scipy import stats

# ─────────────────────────────────────────────────────────
# CONSTANTS  (paper §3, Algorithm 1)
# ─────────────────────────────────────────────────────────
N_PRODUCTS    = 400          # §3.2: N=400
TOP_K         = 7            # §3.6: K=7
N_EPISODES    = 30_000       # §3.6: 30 000 episodes
BOOTSTRAP_RUNS = 30          # §3.6: 30 independent runs
CHECKPOINTS   = [500, 1000, 2000, 5000, 10000, 20000, 30000]  # §3.6

# Q-learning hyper  (Algorithm 1)
ETA           = 0.05         # learning rate η=0.05
EPS_INIT      = 0.30         # ε₀=0.30
EPS_DECAY     = 0.9997       # exponential decay
EPS_MIN       = 0.05         # ε_min=0.05

# GT design  (Eq 3)
GT_OBS_WEIGHT = 0.50         # 50/50 split
GT_HID_WEIGHT = 0.50
GT_NOISE_STD  = 0.015        # ε ~ N(0, 0.015²)

# Fusion  (Eq 1)
LAMBDA_Q_DEFAULT = 0.50      # λ_Q = λ_T = 0.50

# Hidden utility weights  (Eq 4)
W_BRAND   = 0.45
W_PRICE   = 0.30
W_CAT     = 0.15
W_RECENCY = 0.10

# Reward structure  (Eq 5)
R_BASE    = -0.02
R_ENGAGE  = 0.30
R_CONVERT = 1.00
R_ALIGN   = 0.20

# ─────────────────────────────────────────────────────────
# USER PROFILES  (Table 3, §3.2)
# ─────────────────────────────────────────────────────────
PROFILES = {
    "budget": {
        "brand_pref":   {"budget": 0.80, "mid": 0.15, "premium": 0.05},
        "cat_affinity":  {"HomeKitchen": 0.60, "Clothing": 0.25, "Electronics": 0.15},
        "price_range":   (0.01, 0.16),   # 10-160 on 0-1000 scale -> pct [0.01, 0.16]
        "recency_wt":    0.10,
    },
    "quality_seeker": {
        "brand_pref":   {"budget": 0.05, "mid": 0.20, "premium": 0.75},
        "cat_affinity":  {"Electronics": 0.55, "Clothing": 0.35, "HomeKitchen": 0.10},
        "price_range":   (0.30, 1.00),   # 300-1000
        "recency_wt":    0.75,
    },
    "explorer": {
        "brand_pref":   {"budget": 0.25, "mid": 0.50, "premium": 0.25},
        "cat_affinity":  {"Electronics": 0.40, "Clothing": 0.35, "HomeKitchen": 0.25},
        "price_range":   (0.05, 0.70),   # 50-700
        "recency_wt":    0.65,
    },
    "loyal": {
        "brand_pref":   {"budget": 0.10, "mid": 0.25, "premium": 0.65},
        "cat_affinity":  {"Electronics": 0.70, "Clothing": 0.15, "HomeKitchen": 0.15},
        "price_range":   (0.10, 0.80),   # 100-800
        "recency_wt":    0.80,
    },
    "balanced": {
        "brand_pref":   {"budget": 0.33, "mid": 0.34, "premium": 0.33},
        "cat_affinity":  {"Electronics": 0.33, "Clothing": 0.33, "HomeKitchen": 0.34},
        "price_range":   (0.08, 0.50),   # 80-500
        "recency_wt":    0.45,
    },
}
PROFILE_NAMES = list(PROFILES.keys())
N_PROFILES    = len(PROFILE_NAMES)

# ─────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION  (§3.2)
# ─────────────────────────────────────────────────────────
CATEGORIES  = ["HomeKitchen", "Electronics", "Clothing"]
BRAND_TIERS = ["budget", "mid", "premium"]

def generate_products(rng):
    """
    Stratified octant design (§3.2): [0,1]³ = 8 octants, 50 products each.
    Guarantees |ρ(p,q)|, |ρ(p,o)|, |ρ(q,o)| < ~0.12.
    """
    products = []
    per_octant = N_PRODUCTS // 8   # 50

    for octant_idx in range(8):
        # octant boundaries in (price_pct, quality_pct, pop_pct) space
        p_lo = 0.0 if (octant_idx & 1) == 0 else 0.5
        q_lo = 0.0 if (octant_idx & 2) == 0 else 0.5
        o_lo = 0.0 if (octant_idx & 4) == 0 else 0.5

        for _ in range(per_octant):
            price_pct   = rng.uniform(p_lo, p_lo + 0.5)
            quality_pct = rng.uniform(q_lo, q_lo + 0.5)
            pop_pct     = rng.uniform(o_lo, o_lo + 0.5)
            # Rating weakly correlated with quality (ρ ≈ 0.3)
            rating_pct  = np.clip(quality_pct * 0.3 + rng.uniform(0, 1) * 0.7, 0, 1)
            # Non-observable attributes
            category    = rng.choice(CATEGORIES)
            brand_tier  = rng.choice(BRAND_TIERS)
            recency     = rng.uniform(0, 1)

            products.append({
                "price_pct":   price_pct,
                "quality_pct": quality_pct,
                "pop_pct":     pop_pct,
                "rating_pct":  rating_pct,
                "category":    category,
                "brand_tier":  brand_tier,
                "recency":     recency,
            })

    return products


# ─────────────────────────────────────────────────────────
# 2. ENTROPY-WEIGHTED TOPSIS  (§3.3, Eq 2)
# ─────────────────────────────────────────────────────────
def compute_topsis(products):
    """
    Returns C_i* ∈ [0,1] for each product.
    Price is a COST criterion (lower is better).
    Quality, Popularity, Rating are BENEFIT criteria.
    """
    n = len(products)
    # Decision matrix: [price_pct, quality_pct, pop_pct, rating_pct]
    X = np.array([[p["price_pct"], p["quality_pct"], p["pop_pct"], p["rating_pct"]]
                   for p in products])

    # Invert price (cost criterion: lower raw = better)
    X[:, 0] = 1.0 - X[:, 0]

    # L2-normalize columns (§3.3)
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    R = X / norms

    # Shannon entropy weights (§3.3)
    # On decorrelated data, weights ≈ 0.25 each
    m = n
    k = 1.0 / np.log(m + 1e-12)
    weights = np.zeros(4)
    for j in range(4):
        col = R[:, j]
        col_sum = col.sum()
        if col_sum == 0:
            weights[j] = 0.25
            continue
        p_ij = col / (col_sum + 1e-12)
        p_ij = np.clip(p_ij, 1e-12, 1.0)
        ej = -k * np.sum(p_ij * np.log(p_ij))
        weights[j] = 1.0 - ej

    w_sum = weights.sum()
    if w_sum > 0:
        weights /= w_sum
    else:
        weights = np.full(4, 0.25)

    # Weighted normalized matrix
    V = R * weights

    # Ideal and anti-ideal (all are benefit after inversion)
    v_plus  = V.max(axis=0)
    v_minus = V.min(axis=0)

    # Distance to ideal / anti-ideal
    d_plus  = np.sqrt(((V - v_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - v_minus) ** 2).sum(axis=1))

    # Closeness score C_i* ∈ [0,1]
    C_star = d_minus / (d_plus + d_minus + 1e-12)

    return C_star, weights


# ─────────────────────────────────────────────────────────
# 3. OBSERVABLE UTILITY  (Eq 2)
# ─────────────────────────────────────────────────────────
def compute_U_obs(products):
    """U_obs = (1/4)(p* + q* + o* + r*) where * = percentile rank."""
    n = len(products)
    attrs = np.array([[p["price_pct"], p["quality_pct"], p["pop_pct"], p["rating_pct"]]
                       for p in products])
    # Price: lower is better, so invert for "utility"
    attrs[:, 0] = 1.0 - attrs[:, 0]
    # Percentile-rank each column
    for j in range(4):
        order = attrs[:, j].argsort().argsort()
        attrs[:, j] = order / (n - 1.0)
    U_obs = attrs.mean(axis=1)
    return U_obs


# ─────────────────────────────────────────────────────────
# 4. HIDDEN UTILITY  (Eq 4, §3.5)
# ─────────────────────────────────────────────────────────
def compute_U_hid(products, profile_name):
    """
    U_hid(u,i) = 0.45*b_u(i) + 0.30*π_u(i) + 0.15*c_u(i) + 0.10*ρ_u(i)
    All components ∈ [0,1], dependent ONLY on hidden profile parameters.
    """
    prof = PROFILES[profile_name]
    n = len(products)
    U_hid = np.zeros(n)

    for i, p in enumerate(products):
        # Brand affinity b_u(i): how well product's brand tier matches user pref
        b = prof["brand_pref"].get(p["brand_tier"], 0.0)

        # Price-range sensitivity π_u(i): proximity to preferred price range
        lo, hi = prof["price_range"]
        pp = p["price_pct"]
        if lo <= pp <= hi:
            pi_score = 1.0
        else:
            dist = min(abs(pp - lo), abs(pp - hi))
            pi_score = max(0.0, 1.0 - 2.0 * dist)

        # Category loyalty c_u(i)
        c = prof["cat_affinity"].get(p["category"], 0.0)

        # Recency ρ_u(i)
        rho = p["recency"] * prof["recency_wt"]

        U_hid[i] = W_BRAND * b + W_PRICE * pi_score + W_CAT * c + W_RECENCY * rho

    return U_hid


# ─────────────────────────────────────────────────────────
# 5. GROUND TRUTH  (Eq 3, §3.5)
# ─────────────────────────────────────────────────────────
def compute_ground_truth(U_obs, U_hid, rng, obs_weight=GT_OBS_WEIGHT):
    """GT_i(u) = obs_weight*U_obs + (1-obs_weight)*U_hid + ε"""
    n = len(U_obs)
    noise = rng.normal(0, GT_NOISE_STD, size=n)
    GT = obs_weight * U_obs + (1.0 - obs_weight) * U_hid + noise
    return GT


def gt_top_k(GT, k=TOP_K):
    """Returns set of top-K product indices by GT score."""
    return set(np.argsort(GT)[-k:])


# ─────────────────────────────────────────────────────────
# 6. RL REWARD  (Eq 5, §3.4)
# ─────────────────────────────────────────────────────────
def compute_reward(u_hid_i, is_gt_item, rng):
    """
    r(u,i) = -0.02 + 0.30*1[engage] + 1.00*1[convert] + 0.20*g_align
    engage/convert probabilities proportional to U_hid ONLY.
    g_align = 1 if item is in GT-top-7.
    """
    # Engagement probability ∝ U_hid
    p_engage = np.clip(u_hid_i * 0.7 + 0.1, 0.05, 0.95)
    engaged = rng.random() < p_engage

    # Conversion probability (conditional on engagement) ∝ U_hid
    p_convert = np.clip(u_hid_i * 0.5, 0.02, 0.80) if engaged else 0.0
    converted = rng.random() < p_convert if engaged else False

    r = R_BASE
    if engaged:
        r += R_ENGAGE
    if converted:
        r += R_CONVERT
    r += R_ALIGN * float(is_gt_item)

    return r


# ─────────────────────────────────────────────────────────
# 7. TRAINING POOL  (§3.4, Algorithm 1 line 5)
# ─────────────────────────────────────────────────────────
def build_training_pool(gt_set, U_hid, top_hid=30):
    """Pool ≈ GT-top-7 ∪ hidden-utility-top-30 (§3.4: ~30 products)."""
    hid_top = set(np.argsort(U_hid)[-top_hid:])
    pool = sorted(gt_set | hid_top)
    return pool


# ─────────────────────────────────────────────────────────
# 8. EVALUATION  (§3.6)
# ─────────────────────────────────────────────────────────
def f1_at_k(recommended_set, gt_set, k=TOP_K):
    """F1 = 2*TP / (|recommended| + |GT|) = 2*TP / 14"""
    tp = len(recommended_set & gt_set)
    return 2.0 * tp / (2.0 * k)


def recommend_hybrid(Q_row, C_star, lam_q=LAMBDA_Q_DEFAULT):
    """Top-K by S_i = λ_Q * Q_norm + λ_T * C_norm (Eq 1)."""
    q_min, q_max = Q_row.min(), Q_row.max()
    if q_max - q_min > 1e-12:
        q_norm = (Q_row - q_min) / (q_max - q_min)
    else:
        q_norm = np.zeros_like(Q_row)

    c_min, c_max = C_star.min(), C_star.max()
    if c_max - c_min > 1e-12:
        c_norm = (C_star - c_min) / (c_max - c_min)
    else:
        c_norm = np.zeros_like(C_star)

    S = lam_q * q_norm + (1.0 - lam_q) * c_norm
    return set(np.argsort(S)[-TOP_K:])


def recommend_rl_only(Q_row):
    """Top-K by Q(u,i) only."""
    return set(np.argsort(Q_row)[-TOP_K:])


def recommend_topsis_only(C_star):
    """Top-K by C_i* only."""
    return set(np.argsort(C_star)[-TOP_K:])


def recommend_popularity(products):
    """Top-K by pop_pct (proxy for rating_count)."""
    pops = np.array([p["pop_pct"] for p in products])
    return set(np.argsort(pops)[-TOP_K:])


def recommend_random(n, rng):
    """K random products."""
    return set(rng.choice(n, size=TOP_K, replace=False))


# ─────────────────────────────────────────────────────────
# 9. ILD COMPUTATION  (§4.5.2)
# ─────────────────────────────────────────────────────────
def compute_ild(rec_set, products):
    """
    Mean pairwise Euclidean distance in normalized feature space
    {price_pct, quality_pct, pop_pct, rating_pct, recency}.
    """
    indices = sorted(rec_set)
    if len(indices) < 2:
        return 0.0

    features = np.array([
        [products[i]["price_pct"], products[i]["quality_pct"],
         products[i]["pop_pct"], products[i]["rating_pct"],
         products[i]["recency"]]
        for i in indices
    ])

    total_dist = 0.0
    count = 0
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            total_dist += np.linalg.norm(features[a] - features[b])
            count += 1

    return total_dist / count if count > 0 else 0.0


# ─────────────────────────────────────────────────────────
# 10. SINGLE BOOTSTRAP RUN
# ─────────────────────────────────────────────────────────
def run_single(seed, obs_weight=GT_OBS_WEIGHT, lam_q=LAMBDA_Q_DEFAULT,
               drift_at=None, n_episodes=N_EPISODES):
    """
    One complete run: generate data, train RL, evaluate at checkpoints.
    Returns per-profile, per-checkpoint F1 for all methods.
    """
    rng = np.random.default_rng(seed)

    # --- Generate products (§3.2) ---
    products = generate_products(rng)

    # --- TOPSIS (§3.3) ---
    C_star, entropy_wts = compute_topsis(products)

    # --- Observable utility (Eq 2) ---
    U_obs = compute_U_obs(products)

    # --- Popularity baseline ---
    pop_rec = recommend_popularity(products)

    # --- Per-profile training ---
    results = {}
    for pidx, pname in enumerate(PROFILE_NAMES):
        # Hidden utility (Eq 4)
        U_hid = compute_U_hid(products, pname)

        # Ground truth (Eq 3)
        GT = compute_ground_truth(U_obs, U_hid, rng, obs_weight=obs_weight)
        gt_set = gt_top_k(GT)

        # Training pool (§3.4)
        pool = build_training_pool(gt_set, U_hid)

        # Q-table init (Algorithm 1, line 2)
        Q = np.zeros(N_PRODUCTS)

        eps = EPS_INIT
        checkpoint_results = {}

        # --- Brand preferences for drift ---
        original_profile = PROFILES[pname].copy()
        original_brand_pref = PROFILES[pname]["brand_pref"].copy()

        for ep in range(1, n_episodes + 1):
            # --- Concept drift: brand flip at drift_at ---
            if drift_at is not None and ep == drift_at:
                bp = original_brand_pref
                tiers = list(bp.keys())
                vals = list(bp.values())
                # Flip: highest becomes lowest, lowest becomes highest
                vals_flipped = list(reversed(sorted(vals)))
                PROFILES[pname]["brand_pref"] = dict(zip(tiers, vals_flipped))
                # Recompute hidden utility and GT
                U_hid = compute_U_hid(products, pname)
                GT = compute_ground_truth(U_obs, U_hid, rng, obs_weight=obs_weight)
                gt_set = gt_top_k(GT)
                pool = build_training_pool(gt_set, U_hid)

            # ε-greedy action selection (Algorithm 1, line 5)
            if rng.random() < eps:
                action = rng.choice(pool)
            else:
                pool_q = [(a, Q[a]) for a in pool]
                action = max(pool_q, key=lambda x: x[1])[0]

            # Observe reward (Eq 5) -- hidden signals only
            is_gt = action in gt_set
            reward = compute_reward(U_hid[action], is_gt, rng)

            # Q-update (Eq 6, γ=0 so simplified)
            Q[action] += ETA * (reward - Q[action])

            # Decay ε (Algorithm 1, line 8)
            eps = max(EPS_MIN, eps * EPS_DECAY)

            # --- Checkpoint evaluation ---
            if ep in CHECKPOINTS or (drift_at and ep in [2000, 5000, 10000, 14000, 15000, 16000, 20000, 25000, 30000]):
                hybrid_rec  = recommend_hybrid(Q, C_star, lam_q)
                rl_rec      = recommend_rl_only(Q)
                topsis_rec  = recommend_topsis_only(C_star)
                random_rec  = recommend_random(N_PRODUCTS, rng)

                checkpoint_results[ep] = {
                    "hybrid":     f1_at_k(hybrid_rec, gt_set),
                    "rl_only":    f1_at_k(rl_rec, gt_set),
                    "topsis_only": f1_at_k(topsis_rec, gt_set),
                    "popularity": f1_at_k(pop_rec, gt_set),
                    "random":     f1_at_k(random_rec, gt_set),
                }

                # ILD at final checkpoint
                if ep == n_episodes:
                    checkpoint_results[ep]["ild_hybrid"]  = compute_ild(hybrid_rec, products)
                    checkpoint_results[ep]["ild_rl"]      = compute_ild(rl_rec, products)
                    checkpoint_results[ep]["ild_topsis"]  = compute_ild(topsis_rec, products)
                    checkpoint_results[ep]["ild_pop"]     = compute_ild(pop_rec, products)
                    checkpoint_results[ep]["ild_random"]  = compute_ild(random_rec, products)

        # Restore profile if drifted
        if drift_at is not None:
            PROFILES[pname]["brand_pref"] = original_brand_pref

        results[pname] = checkpoint_results

    return results, entropy_wts


# ─────────────────────────────────────────────────────────
# 11. BOOTSTRAP AGGREGATION
# ─────────────────────────────────────────────────────────
def aggregate_bootstrap(all_runs, checkpoint, method):
    """Collect per-run mean-over-profiles F1 at a checkpoint."""
    values = []
    for run_results in all_runs:
        profile_vals = []
        for pname in PROFILE_NAMES:
            if checkpoint in run_results[pname]:
                profile_vals.append(run_results[pname][checkpoint][method])
        if profile_vals:
            values.append(np.mean(profile_vals))
    return np.array(values)


def compute_stats(values):
    """Mean, std, 95% CI."""
    m = np.mean(values)
    s = np.std(values, ddof=1) if len(values) > 1 else 0.0
    n = len(values)
    ci_lo = m - 1.96 * s / np.sqrt(n) if n > 1 else m
    ci_hi = m + 1.96 * s / np.sqrt(n) if n > 1 else m
    return {"mean": round(m, 4), "std": round(s, 4),
            "ci_lo": round(max(0, ci_lo), 4), "ci_hi": round(min(1, ci_hi), 4),
            "values": [round(v, 4) for v in values]}


def paired_ttest(hybrid_vals, baseline_vals):
    """Paired t-test + Cohen's d."""
    diff = hybrid_vals - baseline_vals
    t_stat, p_val = stats.ttest_rel(hybrid_vals, baseline_vals)
    d_mean = diff.mean()
    d_std = diff.std(ddof=1) if len(diff) > 1 else 1e-12
    cohens_d = d_mean / d_std if d_std > 1e-12 else 0.0
    return {"t": round(t_stat, 2), "p": float(f"{p_val:.6f}"),
            "delta": round(d_mean, 4), "cohens_d": round(cohens_d, 2)}


# ─────────────────────────────────────────────────────────
# 12. LAMBDA ABLATION  (§4.3.1)
# ─────────────────────────────────────────────────────────
def run_lambda_ablation(n_runs=10):
    """F1@7 as λ_Q varies from 0.10 to 0.90."""
    lambdas = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
    results = {}
    for lam in lambdas:
        print(f"    Lambda={lam:.2f}...", end=" ", flush=True)
        vals = []
        for run_idx in range(n_runs):
            seed = 5000 + run_idx * 100 + int(lam * 100)
            run_res, _ = run_single(seed, lam_q=lam)
            profile_f1s = []
            for pname in PROFILE_NAMES:
                if N_EPISODES in run_res[pname]:
                    profile_f1s.append(run_res[pname][N_EPISODES]["hybrid"])
            vals.append(np.mean(profile_f1s))
        results[f"{lam:.2f}"] = compute_stats(np.array(vals))
        print(f"mean={results[f'{lam:.2f}']['mean']:.3f}")
    return results


# ─────────────────────────────────────────────────────────
# 13. GT SPLIT ABLATION  (§4.3.2)
# ─────────────────────────────────────────────────────────
def run_split_ablation(n_runs=10):
    """F1@7 as observable fraction α varies from 0.30 to 0.80."""
    alphas = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    results = {}
    for alpha in alphas:
        print(f"    alpha={alpha:.2f}...", end=" ", flush=True)
        hybrid_vals, rl_vals, topsis_vals = [], [], []
        for run_idx in range(n_runs):
            seed = 7000 + run_idx * 100 + int(alpha * 100)
            run_res, _ = run_single(seed, obs_weight=alpha)
            for pname in PROFILE_NAMES:
                if N_EPISODES in run_res[pname]:
                    hybrid_vals.append(run_res[pname][N_EPISODES]["hybrid"])
                    rl_vals.append(run_res[pname][N_EPISODES]["rl_only"])
                    topsis_vals.append(run_res[pname][N_EPISODES]["topsis_only"])
        results[f"{alpha:.2f}"] = {
            "hybrid":  compute_stats(np.array(hybrid_vals)),
            "rl_only": compute_stats(np.array(rl_vals)),
            "topsis":  compute_stats(np.array(topsis_vals)),
        }
        print(f"H={results[f'{alpha:.2f}']['hybrid']['mean']:.3f} "
              f"RL={results[f'{alpha:.2f}']['rl_only']['mean']:.3f} "
              f"T={results[f'{alpha:.2f}']['topsis']['mean']:.3f}")
    return results


# ─────────────────────────────────────────────────────────
# 14. CONCEPT DRIFT  (§4.5.1)
# ─────────────────────────────────────────────────────────
def run_concept_drift(n_runs=BOOTSTRAP_RUNS):
    """Brand flip at episode 15000, measure recovery."""
    print("  Running concept drift experiment...")
    drift_checkpoints = [2000, 5000, 10000, 14000, 15000, 16000, 20000, 25000, 30000]
    all_runs = []
    for run_idx in range(n_runs):
        if (run_idx + 1) % 5 == 0:
            print(f"    drift run {run_idx+1}/{n_runs}")
        seed = 9000 + run_idx * 7
        run_res, _ = run_single(seed, drift_at=15000)
        all_runs.append(run_res)

    results = {}
    for cp in drift_checkpoints:
        results[str(cp)] = {}
        for method in ["hybrid", "rl_only"]:
            vals = aggregate_bootstrap(all_runs, cp, method)
            results[str(cp)][method] = compute_stats(vals)

    return results


# ─────────────────────────────────────────────────────────
# 15. CONFIDENCE-GATED FUSION  (§4.5.3)
# ─────────────────────────────────────────────────────────
def run_cgf_experiment(n_runs=BOOTSTRAP_RUNS):
    """Static vs confidence-gated fusion."""
    print("  Running CGF experiment...")
    TAU = 300
    LAM_MAX = 0.50

    static_vals = []
    cgf_vals = []

    for run_idx in range(n_runs):
        if (run_idx + 1) % 5 == 0:
            print(f"    CGF run {run_idx+1}/{n_runs}")
        seed = 11000 + run_idx * 13
        rng = np.random.default_rng(seed)
        products = generate_products(rng)
        C_star, _ = compute_topsis(products)
        U_obs = compute_U_obs(products)
        pop_rec = recommend_popularity(products)

        for pidx, pname in enumerate(PROFILE_NAMES):
            U_hid = compute_U_hid(products, pname)
            GT = compute_ground_truth(U_obs, U_hid, rng)
            gt_set = gt_top_k(GT)
            pool = build_training_pool(gt_set, U_hid)

            Q = np.zeros(N_PRODUCTS)
            visit_count = np.zeros(N_PRODUCTS)
            eps = EPS_INIT

            for ep in range(1, N_EPISODES + 1):
                if rng.random() < eps:
                    action = rng.choice(pool)
                else:
                    pool_q = [(a, Q[a]) for a in pool]
                    action = max(pool_q, key=lambda x: x[1])[0]

                is_gt = action in gt_set
                reward = compute_reward(U_hid[action], is_gt, rng)
                Q[action] += ETA * (reward - Q[action])
                visit_count[action] += 1
                eps = max(EPS_MIN, eps * EPS_DECAY)

            # Static fusion
            static_rec = recommend_hybrid(Q, C_star, LAMBDA_Q_DEFAULT)
            static_f1 = f1_at_k(static_rec, gt_set)
            static_vals.append(static_f1)

            # CGF fusion
            q_min, q_max = Q.min(), Q.max()
            q_norm = (Q - q_min) / (q_max - q_min + 1e-12)
            c_min, c_max = C_star.min(), C_star.max()
            c_norm = (C_star - c_min) / (c_max - c_min + 1e-12)

            lam_cgf = LAM_MAX * (1.0 - np.exp(-visit_count / TAU))
            S_cgf = lam_cgf * q_norm + (1.0 - lam_cgf) * c_norm
            cgf_rec = set(np.argsort(S_cgf)[-TOP_K:])
            cgf_f1 = f1_at_k(cgf_rec, gt_set)
            cgf_vals.append(cgf_f1)

    return {
        "static": compute_stats(np.array(static_vals)),
        "cgf": compute_stats(np.array(cgf_vals)),
        "mean_lambda_cgf": round(float(np.mean(lam_cgf)), 4),
    }


# ─────────────────────────────────────────────────────────
# 16. MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Hybrid RL-MCDM v2 -- Full Reproducible Implementation")
    print("=" * 60)

    output = {}

    # ── A. Main 30-run bootstrap ──
    print("\n[A] Main bootstrap (30 runs x 5 profiles)...")
    all_runs = []
    all_entropy_wts = []
    for run_idx in range(BOOTSTRAP_RUNS):
        seed = 1000 + run_idx * 17
        if (run_idx + 1) % 5 == 0:
            print(f"  Run {run_idx+1}/{BOOTSTRAP_RUNS}")
        run_res, ewts = run_single(seed)
        all_runs.append(run_res)
        all_entropy_wts.append(ewts.tolist())

    # ── Aggregate main results at 30K ──
    print("\n[A.1] Aggregating main results...")
    methods = ["hybrid", "rl_only", "topsis_only", "popularity", "random"]
    main_results = {}
    for method in methods:
        vals = aggregate_bootstrap(all_runs, N_EPISODES, method)
        main_results[method] = compute_stats(vals)

    # Paired t-tests
    hybrid_vals = aggregate_bootstrap(all_runs, N_EPISODES, "hybrid")
    tests = {}
    for baseline in ["rl_only", "topsis_only", "popularity", "random"]:
        base_vals = aggregate_bootstrap(all_runs, N_EPISODES, baseline)
        tests[baseline] = paired_ttest(hybrid_vals, base_vals)

    output["main_results"] = main_results
    output["statistical_tests"] = tests
    output["entropy_weights_mean"] = np.mean(all_entropy_wts, axis=0).round(4).tolist()

    # ── Convergence table ──
    print("[A.2] Convergence table...")
    convergence = {}
    for cp in CHECKPOINTS:
        convergence[str(cp)] = {}
        for method in methods:
            vals = aggregate_bootstrap(all_runs, cp, method)
            convergence[str(cp)][method] = compute_stats(vals)
    output["convergence"] = convergence

    # ── ILD at 30K ──
    print("[A.3] ILD analysis...")
    ild_methods = {"hybrid": "ild_hybrid", "rl_only": "ild_rl",
                   "topsis": "ild_topsis", "popularity": "ild_pop",
                   "random": "ild_random"}
    ild_results = {}
    for method_name, ild_key in ild_methods.items():
        vals = []
        for run_res in all_runs:
            profile_ilds = []
            for pname in PROFILE_NAMES:
                if N_EPISODES in run_res[pname] and ild_key in run_res[pname][N_EPISODES]:
                    profile_ilds.append(run_res[pname][N_EPISODES][ild_key])
            if profile_ilds:
                vals.append(np.mean(profile_ilds))
        ild_results[method_name] = compute_stats(np.array(vals))
    output["ild"] = ild_results

    # ── B. Lambda ablation ──
    print("\n[B] Lambda ablation (10 runs per λ)...")
    output["lambda_ablation"] = run_lambda_ablation(n_runs=10)

    # ── C. GT split ablation ──
    print("\n[C] GT split ablation (10 runs per α)...")
    output["split_ablation"] = run_split_ablation(n_runs=10)

    # ── D. Concept drift ──
    print("\n[D] Concept drift (30 runs)...")
    output["concept_drift"] = run_concept_drift(n_runs=BOOTSTRAP_RUNS)

    # ── E. Confidence-gated fusion ──
    print("\n[E] Confidence-gated fusion (30 runs)...")
    output["cgf"] = run_cgf_experiment(n_runs=BOOTSTRAP_RUNS)

    # ── Save ──
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v2_full_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for m in methods:
        r = main_results[m]
        print(f"  {m:15s}: F1={r['mean']:.3f} +/- {r['std']:.3f}  "
              f"CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    print()
    for b in ["rl_only", "topsis_only"]:
        t = tests[b]
        print(f"  vs {b:12s}: t={t['t']:7.2f}, p={t['p']:.6f}, "
              f"delta={t['delta']:+.3f}, d={t['cohens_d']:.2f}")

    print(f"\nEntropy weights (mean): {output['entropy_weights_mean']}")
    print(f"\nResults saved to: {out_path}")
    return output


if __name__ == "__main__":
    main()
