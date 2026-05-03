"""
Learned-vs-static lambda analysis (no new training; use v2_full_results.json
lambda_ablation as the upper bound on what any validation-tuned scheme could
achieve).

Logic:
- The lambda_ablation in v2_full_results.json tested lambda_Q over 10 grid points
  for each of 10 bootstrap runs (n_runs=10 per run_lambda_ablation, BOOTSTRAP_RUNS=30
  for primary). Each entry reports mean F1 at the design point.
- The "best lambda" (validation-tuned upper bound) per run is the lambda that
  maximizes mean F1 over those runs in aggregate. Static lambda is 0.50.
- We do NOT have a separate validation set, so per-run optimal lambda would
  overfit. We therefore frame this as: even if a learned-lambda scheme could
  achieve the population-best lambda perfectly, the gain over static 0.50 is
  bounded by max(F1) - F1(0.50).

This is the most honest framing without running a fresh nested-bootstrap
experiment.
"""
import json
import numpy as np

with open('results/v2_full_results.json') as f:
    d = json.load(f)

la = d['lambda_ablation']
rows = []
for lam_str, st in la.items():
    if isinstance(st, dict) and 'mean' in st:
        rows.append((float(lam_str), st['mean'], st['std']))

rows.sort()
print("Lambda ablation (from v2_full_results.json):")
print(f"{'lambda_Q':>10}  {'mean F1':>10}  {'std':>8}")
for lam, m, s in rows:
    print(f"{lam:>10.2f}  {m:>10.4f}  {s:>8.4f}")
print()

static_lam = 0.50
static_idx = next(i for i,(l,_,_) in enumerate(rows) if l == static_lam)
static_f1 = rows[static_idx][1]

best_idx = int(np.argmax([m for _,m,_ in rows]))
best_lam, best_f1, best_std = rows[best_idx]

print(f"Static design point:  lambda={static_lam:.2f}, F1={static_f1:.4f}")
print(f"Best in grid (oracle): lambda={best_lam:.2f}, F1={best_f1:.4f}")
print(f"Upper-bound delta:     +{best_f1 - static_f1:.4f}")
print(f"Failure threshold (F2 |delta| > 0.05):  {'PASS (within threshold)' if abs(best_f1-static_f1) <= 0.05 else 'FAIL'}")

# Range
f1s = [m for _,m,_ in rows]
print(f"\nGrid F1 range: [{min(f1s):.4f}, {max(f1s):.4f}], width = {max(f1s)-min(f1s):.4f}")

# Save analysis
out = {
    "method": "Lambda grid analysis as upper bound on learned-lambda gain",
    "rationale": ("No separate validation set was constructed for v16; the "
                  "lambda_ablation grid already represents an oracle upper "
                  "bound on what any validation-tuned learned-lambda scheme "
                  "could achieve, since picking the best lambda from the grid "
                  "is at least as good as any unbiased validation estimate."),
    "static_lambda": static_lam,
    "static_f1": static_f1,
    "best_grid_lambda": best_lam,
    "best_grid_f1": best_f1,
    "upper_bound_delta": float(best_f1 - static_f1),
    "failure_threshold": 0.05,
    "verdict": ("Static design within 0.05 of oracle upper bound; "
                "interpretability cost is negligible"
                if abs(best_f1-static_f1) <= 0.05 else
                "Static design pays meaningful cost; reconsider learned lambda"),
    "grid": [{"lambda": lam, "f1_mean": m, "f1_std": s} for lam,m,s in rows],
}
with open('results/v16_learned_lambda.json','w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: results/v16_learned_lambda.json")
print(f"Verdict: {out['verdict']}")
