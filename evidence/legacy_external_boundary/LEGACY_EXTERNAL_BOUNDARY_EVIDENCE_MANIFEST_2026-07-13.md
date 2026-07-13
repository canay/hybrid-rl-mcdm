# Legacy external-boundary evidence manifest

Date/time: 2026-07-13 09:18 +03:00  
Tool: Codex  
Model, if known: GPT-5 Codex  
Operation ID: HRE_R1_LEGACY_EXTERNAL_BOUNDARY_BINDING_20260713_CODEX_29

## Scope

These three R1 boundary checks predate the corrected primary experiment and are
not outputs of the 2026-07-13 verified-science extractor. They are retained as
secondary, hash-bound scope evidence only. They must not be pooled with the
leakage-free primary, described as full-catalog McAuley performance where the
protocol used sampled candidates, or used to claim universal superiority.

## Bound files

| Check | Source | SHA-256 | Result | SHA-256 |
|---|---|---|---|---|
| McAuley Home & Kitchen | `code/mccauley_home_experiment.py` | `5af4c7080b054b0e27cd3671efa920e9064eae468a0142beaffbf9a921fb2804` | `results/mccauley_home_real_results.json` | `9f04ebedb34be8b6811f6d52a48af3c30d6eaec13f0d659180c211b737efef6c` |
| Sparse reviewer--product | `code/benchmark_recommenders.py` | `2afa343c4722e7c2cb23908061f3e69a31355844be1138887301b0c2fc0eab03` | `results/recommender_benchmarks.json` | `ca3e016f810dcff121d6089f11566c6a3560c9c10c42d852bdf892b32f31b249` |
| CUDA deep recommenders | `code/deep_recommender_benchmarks.py` | `2ca8d7ffac38af3cf8d90d999e09db1d32679c90032e7ef8e2c280e40d5e83ab` | `results/deep_recommender_benchmarks.json` | `6ce9663fb51c0a8657e97bce65929bb57b813b12cbcfa2081d73cd699ff4ff4d` |

The shared processed McAuley manifest is
`data/processed/amazon_mccauley_home/manifest.json`, SHA-256
`a347d341e46b984db8b7dad11b228bfb1bbec641a2e9b7874d0006dfdfdc316e`.

## Protocol boundaries

- **McAuley.** The public Home & Kitchen 5-core branch contains 808 users and
  11,287 items. It uses a temporal positive holdout. Each user is evaluated by
  ranking the held-out positives together with 200 sampled unseen negatives
  after seen-item masking. Q training actions come from a history-derived pool:
  the top 250 hidden-utility items, top 120 TOPSIS items, and observed training
  items. The reported endpoint is a 30 user-bootstrap summary. Mean F1@7 is
  0.0658 for Hybrid, 0.0642 for TOPSIS-only, 0.0559 for RL-only, 0.1301 for
  popularity, and 0.0391 for random. The Hybrid--TOPSIS margin is numerical and
  does not establish full-catalog or production superiority.
- **Sparse reviewer--product graph.** The Amazon India reviewer graph has 357
  eligible reviewer identifiers and uses 30 repeated random leave-one-out
  splits at K=10. The raw source has no timestamps, so this is not a temporal
  next-item benchmark. Mean NDCG@10 is 0.8908 for UserKNN, 0.8836 for ItemKNN,
  0.7191 for BPR-MF, 0.6877 for content-based filtering, and 0.0121 for TOPSIS.
- **Deep recommenders.** The CUDA check uses 10 runs, 30 epochs, the temporal
  McAuley holdout, and full-catalog scoring after masking training items. Mean
  NDCG@10 is 0.0465 for LightGCN, 0.0455 for BPR-MF, 0.0196 for NeuMF, and
  0.0052 for SASRec. These values delimit an interaction-rich regime; they are
  not directly comparable point-for-point with the constructed K=7 primary.

## Packaging boundary

Privacy-screened copies of the six bound source/result files and an internal
machine-readable manifest are included in the anonymous reviewer supplement
under `legacy_external_boundary/`. Raw data and runtime logs are excluded.
