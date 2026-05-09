# Hybrid RL-TOPSIS Manuscript and Experiment Audit

Date: 2026-05-09

## Manuscript Consistency Checks

- Replaced earlier controlled-dataset wording with the Hybrid RL-TOPSIS package framing: a real Amazon
  India criterion-rich product catalog plus an explicit profile-level feedback
  model.
- Updated the manuscript to the 50-run Amazon India outputs:
  - Hybrid F1@7 = 0.9006 at 30,000 episodes.
  - RL-only F1@7 = 0.5743.
  - TOPSIS-only F1@7 = 0.2594.
  - Drift final F1@7 = 0.8274 for Hybrid and 0.5663 for RL-only.
  - Oracle-grid best lambda_Q = 0.45 with F1@7 = 0.9074; static lambda_Q = 0.50
    remains within +0.006 F1 of this upper-bound grid point.
- Corrected stale ground-truth split values:
  - At alpha = 0.80, TOPSIS is best; Hybrid = 0.497 and RL-only = 0.294.
  - The correct interpretation is that Hybrid is strongest when observable and
    hidden behavioral utility both matter.
- Added the real SHAP audit as interpretability evidence, not as a causal user
  behavior explanation.
- Reframed collaborative, popularity, and deep recommenders as boundary checks:
  they delimit interaction-rich regimes where CF/graph/sequential models should
  be preferred.
- Updated README/runbook language so the public Hybrid RL-TOPSIS package matches the
  manuscript and no longer points readers to the obsolete 30-run snapshot.

## Additional Validation Added

- Added `code/statistical_audit.py`.
- Added `scripts/run_statistical_audit.ps1`.
- Generated:
  - `results/statistical_audit.json`
  - `results/statistical_audit_summary.csv`
  - `logs/statistical_audit.log`
- Result: Hybrid beats every baseline in all 50 paired runs. Against RL-only,
  median paired delta F1@7 is +0.314 and the one-sided Wilcoxon signed-rank
  p-value is 3.55e-10.

## Experiment Coverage Status

Already covered:

- 50-run primary Amazon India experiment.
- Reward-shaping sensitivity including zero shaping.
- Fusion-weight/lambda sensitivity.
- Observable-hidden ground-truth split sensitivity.
- Concept drift via brand-preference flip.
- Diversity/ILD analysis.
- Static versus confidence-gated fusion.
- External McAuley Amazon Home & Kitchen validation.
- Sparse reviewer-product benchmark with UserKNN, ItemKNN, BPR-MF, CBF,
  TOPSIS, popularity, and random.
- CUDA deep recommender boundary check with LightGCN, BPR-MF, NeuMF, and SASRec.
- Real SHAP XAI audit with RF and ridge surrogates.
- Paired t-tests, Bonferroni correction, effect sizes, and Wilcoxon signed-rank
  checks.
- Additional 30-run reviewer-facing validation:
  - LinUCB contextual-bandit baseline.
  - Catalog-size sensitivity at 200, 400, and 800 items.
  - Gradual multi-dimensional drift over brand, category, price, and recency.

Useful but not essential next experiments:

- A DQN-style deep-RL baseline if the paper is repositioned toward RL rather
  than lightweight decision support.

Current judgment: the Hybrid RL-TOPSIS experimental package is enough for a coherent Q1
submission if the paper is framed as CF-free, criterion-rich decision support
rather than a universal recommender-system SOTA claim.



