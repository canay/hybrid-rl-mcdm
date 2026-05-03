"""
supplementary_runs.py -- Profile-level F1, NDCG@7, reward-shaping sensitivity
Reads from v2 code, outputs to results/v2_supplementary.json
"""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from hybrid_rl_mcdm_v2 import (
    generate_products, compute_topsis, compute_U_obs, compute_U_hid,
    compute_ground_truth, gt_top_k, build_training_pool, compute_reward,
    recommend_hybrid, recommend_rl_only, recommend_topsis_only,
    recommend_popularity, recommend_random, f1_at_k,
    PROFILES, PROFILE_NAMES, N_PRODUCTS, TOP_K, N_EPISODES, ETA,
    EPS_INIT, EPS_DECAY, EPS_MIN, BOOTSTRAP_RUNS, R_ALIGN
)


def ndcg_at_k(recommended_list, gt_set, k=TOP_K):
    """NDCG@K: ordered recommended_list, unordered gt_set."""
    dcg = 0.0
    for rank, item in enumerate(recommended_list[:k]):
        if item in gt_set:
            dcg += 1.0 / np.log2(rank + 2)
    # Ideal: all K items relevant
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(gt_set))))
    return dcg / idcg if idcg > 0 else 0.0


def recommend_hybrid_ranked(Q_row, C_star, lam_q=0.50):
    """Returns ordered list (not set) for NDCG."""
    q_min, q_max = Q_row.min(), Q_row.max()
    q_norm = (Q_row - q_min) / (q_max - q_min + 1e-12) if q_max - q_min > 1e-12 else np.zeros_like(Q_row)
    c_min, c_max = C_star.min(), C_star.max()
    c_norm = (C_star - c_min) / (c_max - c_min + 1e-12) if c_max - c_min > 1e-12 else np.zeros_like(C_star)
    S = lam_q * q_norm + (1.0 - lam_q) * c_norm
    return list(np.argsort(S)[::-1][:TOP_K])


def run_with_profile_detail(seed, shaping_bonus=0.20):
    """Single run returning per-profile F1 at 30K + NDCG@7."""
    rng = np.random.default_rng(seed)
    products = generate_products(rng)
    C_star, _ = compute_topsis(products)
    U_obs = compute_U_obs(products)
    pop_rec = recommend_popularity(products)

    profile_results = {}
    for pname in PROFILE_NAMES:
        U_hid = compute_U_hid(products, pname)
        GT = compute_ground_truth(U_obs, U_hid, rng)
        gt_set = gt_top_k(GT)
        pool = build_training_pool(gt_set, U_hid)

        Q = np.zeros(N_PRODUCTS)
        eps = EPS_INIT
        for ep in range(1, N_EPISODES + 1):
            if rng.random() < eps:
                action = rng.choice(pool)
            else:
                pool_q = [(a, Q[a]) for a in pool]
                action = max(pool_q, key=lambda x: x[1])[0]
            is_gt = action in gt_set
            # Use shaping_bonus instead of fixed R_ALIGN
            reward = -0.02
            p_engage = np.clip(U_hid[action] * 0.7 + 0.1, 0.05, 0.95)
            if rng.random() < p_engage:
                reward += 0.30
                p_convert = np.clip(U_hid[action] * 0.5, 0.02, 0.80)
                if rng.random() < p_convert:
                    reward += 1.00
            reward += shaping_bonus * float(is_gt)
            Q[action] += ETA * (reward - Q[action])
            eps = max(EPS_MIN, eps * EPS_DECAY)

        # Evaluate
        hybrid_rec = recommend_hybrid(Q, C_star)
        rl_rec = recommend_rl_only(Q)
        topsis_rec = recommend_topsis_only(C_star)
        random_rec = recommend_random(N_PRODUCTS, rng)

        hybrid_ranked = recommend_hybrid_ranked(Q, C_star)
        rl_ranked = list(np.argsort(Q)[::-1][:TOP_K])
        topsis_ranked = list(np.argsort(C_star)[::-1][:TOP_K])
        pop_ranked = list(np.argsort([p["pop_pct"] for p in products])[::-1][:TOP_K])
        random_ranked = list(rng.choice(N_PRODUCTS, size=TOP_K, replace=False))

        profile_results[pname] = {
            "f1_hybrid":  f1_at_k(hybrid_rec, gt_set),
            "f1_rl":      f1_at_k(rl_rec, gt_set),
            "f1_topsis":  f1_at_k(topsis_rec, gt_set),
            "f1_pop":     f1_at_k(pop_rec, gt_set),
            "f1_random":  f1_at_k(random_rec, gt_set),
            "ndcg_hybrid": ndcg_at_k(hybrid_ranked, gt_set),
            "ndcg_rl":     ndcg_at_k(rl_ranked, gt_set),
            "ndcg_topsis": ndcg_at_k(topsis_ranked, gt_set),
            "ndcg_pop":    ndcg_at_k(pop_ranked, gt_set),
            "ndcg_random": ndcg_at_k(random_ranked, gt_set),
        }
    return profile_results


def main():
    output = {}

    # 1. Profile-level F1 + NDCG (30 runs)
    print("[1] Profile-level F1 and NDCG@7 (30 runs)...")
    all_profile_data = {pname: {k: [] for k in ["f1_hybrid","f1_rl","f1_topsis","f1_pop","f1_random",
                                                  "ndcg_hybrid","ndcg_rl","ndcg_topsis","ndcg_pop","ndcg_random"]}
                        for pname in PROFILE_NAMES}

    for run_idx in range(BOOTSTRAP_RUNS):
        if (run_idx + 1) % 10 == 0:
            print(f"  Run {run_idx+1}/{BOOTSTRAP_RUNS}")
        seed = 1000 + run_idx * 17  # Same seeds as main experiment
        res = run_with_profile_detail(seed, shaping_bonus=0.20)
        for pname in PROFILE_NAMES:
            for k in all_profile_data[pname]:
                all_profile_data[pname][k].append(res[pname][k])

    # Aggregate profile-level
    profile_summary = {}
    for pname in PROFILE_NAMES:
        profile_summary[pname] = {}
        for metric in ["f1_hybrid","f1_rl","f1_topsis","ndcg_hybrid","ndcg_rl","ndcg_topsis"]:
            vals = np.array(all_profile_data[pname][metric])
            profile_summary[pname][metric] = {
                "mean": round(float(vals.mean()), 3),
                "std": round(float(vals.std(ddof=1)), 3),
            }

    # Aggregate NDCG across profiles
    ndcg_agg = {}
    for method in ["hybrid","rl","topsis","pop","random"]:
        vals = []
        for run_idx in range(BOOTSTRAP_RUNS):
            profile_means = [all_profile_data[pn][f"ndcg_{method}"][run_idx] for pn in PROFILE_NAMES]
            vals.append(np.mean(profile_means))
        arr = np.array(vals)
        ndcg_agg[method] = {
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std(ddof=1)), 3),
            "ci_lo": round(float(arr.mean() - 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 3),
            "ci_hi": round(float(arr.mean() + 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))), 3),
        }

    output["profile_level"] = profile_summary
    output["ndcg"] = ndcg_agg

    # 2. Reward-shaping sensitivity (bonus = 0.00, 0.10, 0.20)
    print("\n[2] Reward-shaping sensitivity...")
    shaping_results = {}
    for bonus in [0.00, 0.10, 0.20]:
        print(f"  bonus={bonus:.2f}...")
        hybrid_vals, rl_vals = [], []
        for run_idx in range(BOOTSTRAP_RUNS):
            seed = 1000 + run_idx * 17
            res = run_with_profile_detail(seed, shaping_bonus=bonus)
            pf1_h = np.mean([res[pn]["f1_hybrid"] for pn in PROFILE_NAMES])
            pf1_r = np.mean([res[pn]["f1_rl"] for pn in PROFILE_NAMES])
            hybrid_vals.append(pf1_h)
            rl_vals.append(pf1_r)
        shaping_results[f"{bonus:.2f}"] = {
            "hybrid": round(float(np.mean(hybrid_vals)), 3),
            "rl_only": round(float(np.mean(rl_vals)), 3),
            "delta": round(float(np.mean(hybrid_vals) - np.mean(rl_vals)), 3),
        }
        print(f"    H={shaping_results[f'{bonus:.2f}']['hybrid']:.3f} "
              f"RL={shaping_results[f'{bonus:.2f}']['rl_only']:.3f}")

    output["reward_shaping"] = shaping_results

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "results", "v2_supplementary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary
    print("\n=== PROFILE-LEVEL F1 ===")
    for pn in PROFILE_NAMES:
        ps = profile_summary[pn]
        print(f"  {pn:18s}  T={ps['f1_topsis']['mean']:.3f}  "
              f"RL={ps['f1_rl']['mean']:.3f}  H={ps['f1_hybrid']['mean']:.3f}  "
              f"delta={ps['f1_hybrid']['mean']-ps['f1_rl']['mean']:+.3f}")

    print("\n=== NDCG@7 ===")
    for m in ["hybrid","rl","topsis","pop","random"]:
        print(f"  {m:10s}: {ndcg_agg[m]['mean']:.3f} [{ndcg_agg[m]['ci_lo']:.3f}, {ndcg_agg[m]['ci_hi']:.3f}]")

    print("\n=== REWARD SHAPING ===")
    for b in ["0.00","0.10","0.20"]:
        r = shaping_results[b]
        print(f"  bonus={b}  H={r['hybrid']:.3f}  RL={r['rl_only']:.3f}  delta={r['delta']:+.3f}")


if __name__ == "__main__":
    main()
