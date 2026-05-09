# Reviewer Question Audit for Hybrid RL-TOPSIS

Date: 2026-05-09

This file lists likely reviewer questions and where the Hybrid RL-TOPSIS manuscript now
answers them.

| # | Likely reviewer question | Status |
|---|---|---|
| 1 | What is the actual novelty beyond putting TOPSIS and Q-learning side by side? | Addressed in Introduction, Research Gap, Related Work, and Discussion: the contribution is separable structural/behavioral fusion plus ablations, XAI, and boundary validation. |
| 2 | Why is the main method CF-free when e-commerce recommendation is usually collaborative? | Addressed as scope: CF-free criterion-rich decision support, not universal recommender SOTA. External CF/deep checks are reported as boundaries. |
| 3 | Is the dataset real, synthetic, or partly simulated? | Addressed: Amazon India is a real product/review catalog; the behavioral feedback layer and GT profiles are explicit experimental constructs because the CSV has no temporal purchase/click labels. |
| 4 | Why use Amazon India rather than a standard recommender benchmark? | Addressed: it has price, discount, rating, rating count, category, description, and review fields needed for MCDM; McAuley Home is added as external interaction validation. |
| 5 | Why 400 products rather than 800 or 1,200? | Addressed in Dataset Specification: 400 is a bootstrap-design choice preserving variability across 50 samples from 1,351 unique products; 200/400/800 sensitivity checks verify it is not a narrow artifact. |
| 6 | Are the user profiles arbitrary? | Mostly addressed: profile parameters and heterogeneity are shown. Residual risk: real user segmentation remains future/live validation. |
| 7 | Does the ground truth favor the hybrid by construction? | Addressed by split sensitivity: TOPSIS wins when target becomes strongly observable; Hybrid is best when both signals matter. |
| 8 | Does reward shaping leak the evaluation target into training? | Addressed by explaining behavior-dominant reward plus GT-alignment shaping and by zero/low shaping sensitivity where Hybrid remains above RL-only. |
| 9 | Why call this Q-learning if there is no long sequential horizon? | Newly addressed: it is explicitly framed as a one-step profile-specific action-value estimator; LinUCB is added as a contextual-bandit comparison. |
| 10 | Why not use LinUCB or another contextual bandit? | Addressed by 30-run LinUCB validation: LinUCB F1@7 = 0.097 versus Hybrid = 0.902; Hybrid wins all paired runs. |
| 11 | Is the result statistically reliable? | Addressed: 50-run bootstrap, paired t-tests, Bonferroni correction, effect sizes, and Wilcoxon signed-rank checks. |
| 12 | Is F1@7 enough? | Addressed: NDCG@7, profile-level results, ILD/diversity, and external recommender metrics are included. |
| 13 | Does the method work only after many training episodes? | Addressed: convergence shows advantage already at 500 episodes, supporting the cold-start interpretation. |
| 14 | Is lambda_Q = 0.50 tuned to the test set? | Addressed: lambda grid shows oracle best 0.45 with only +0.006 F1 over static 0.50; static choice preserves interpretability. |
| 15 | Why did confidence-gated dynamic fusion fail, and does that weaken the method? | Addressed mechanistically: bounded-pool visits leave many items with zero Q confidence, allowing TOPSIS to dominate unvisited items. |
| 16 | Is drift evaluation too artificial? | Addressed: original brand-flip drift plus new gradual multi-dimensional drift over brand/category/price/recency. |
| 17 | Does the method scale beyond 400 items? | Addressed up to 800 items; limitation remains for very large catalogs requiring function approximation. |
| 18 | Does SHAP really explain the model? | Addressed carefully: SHAP is a surrogate audit of final hybrid scores, not a causal explanation of user behavior; branch decomposition gives intrinsic explanation. |
| 19 | Why do McAuley/popularity/CF baselines sometimes outperform the proposed method? | Addressed as scope-boundary evidence: interaction-rich regimes should use popularity/CF/graph/sequential models. |
| 20 | Can the work be reproduced and shared without privacy problems? | Addressed: public artifacts exclude raw reviewer IDs/names/review text/links; scripts, manifests, logs, and generated outputs are organized in the Hybrid RL-TOPSIS package. |

## Remaining Honest Limitations

- No live A/B test or business metric such as CTR/revenue.
- No very-large-catalog experiment beyond 800 products.
- No exhaustive contextual-bandit or nonlinear bandit benchmark.
- No DQN/deep-RL policy baseline, intentionally left out to preserve the
  lightweight decision-support framing.
- User profiles are explicit experimental constructs rather than learned from
  observed temporal purchase histories.



