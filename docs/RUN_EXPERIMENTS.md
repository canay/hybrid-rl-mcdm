# Hybrid RL-TOPSIS Experiment Runbook

Run all commands from the repository root:

```powershell
cd C:\DOCS\AKADEMIK\GIT_clones\hybrid-rl-claude_code
```

## 1. Main Hybrid RL-TOPSIS Experiments

Recommended full run:

```powershell
python code\run_amazon_experiments.py --runs 50 --drift-runs 50 *> logs\main_50run.log
```

Fast sanity run only:

```powershell
python code\run_amazon_experiments.py --runs 30 --drift-runs 30 *> logs\main_30run.log
```

The manuscript results are based on the 50-run command, not the 30-run sanity
run.

Generated outputs:

- `results\run_summary.json`
- `results\amazon_primary.json`
- `results\amazon_extended.json`
- `results\amazon_robustness.json`
- `results\amazon_drift.json`
- `data\processed\manifest.json`
- `logs\main_50run.log`
- `logs\main_50run_runtime.json`

## 2. Sparse Reviewer-Product Benchmark

This is a supplementary recommender benchmark over the reviewer-product graph
embedded in the Amazon India CSV. It is not temporal because the raw CSV has no
review timestamps.

Recommended full run:

```powershell
python code\benchmark_recommenders.py --runs 30 --min-items 3 --factors 32 --epochs 80 *> logs\benchmark_30run.log
```

Generated outputs:

- `results\recommender_benchmarks.json`
- `results\recommender_benchmarks_summary.csv`
- `logs\benchmark_30run.log`

## 3. Optional Stress Run

Use this only if there is enough time:

```powershell
python code\run_amazon_experiments.py --runs 100 --drift-runs 100 *> logs\main_100run.log
python code\benchmark_recommenders.py --runs 50 --min-items 3 --factors 32 --epochs 100 *> logs\benchmark_50run.log
```

## 4. CUDA-Aware Deep Recommender Benchmarks

Run this after the McAuley Home processed dataset exists. It trains BPR-MF,
NeuMF, LightGCN, and SASRec-style baselines. If CUDA is available, PyTorch uses
the GPU automatically.

```powershell
.\scripts\run_deep_benchmark.ps1
```

Generated outputs:

- `results\deep_recommender_benchmarks.json`
- `results\deep_recommender_benchmarks_summary.csv`
- `logs\deep_benchmark_10run.log`
- `logs\deep_benchmark_10run_runtime.json`

For a longer stress run:

```powershell
python -u code\deep_recommender_benchmarks.py --runs 30 --epochs 50 --factors 64 --batch-size 1024 2>&1 | Tee-Object -FilePath logs\deep_benchmark_30run.log
```

## 5. Real SHAP Explainability Analysis

Run after the main Amazon India experiment has produced
`results/amazon_primary.json`.

Install SHAP first if needed:

```powershell
python -m pip install shap
```

```powershell
.\scripts\run_xai.ps1
```

Generated outputs:

- `results\xai\xai_report.json`
- `results\xai\global_importance.csv`
- `results\xai\local_top7_explanations.csv`
- `results\xai\local_top7_shap_values.csv`
- `results\xai\sample_shap_values.csv`
- `results\xai\counterfactual_rank_shifts.csv`
- `logs\xai_analysis_runtime.json`
- `logs\xai_analysis.log`

## 6. Paired Statistical Audit

Run after `results/amazon_primary.json` exists. This does not rerun the
experiment; it computes paired t-test and one-sided Wilcoxon signed-rank
checks from the stored 50-run primary output.

```powershell
.\scripts\run_statistical_audit.ps1
```

Generated outputs:

- `results\statistical_audit.json`
- `results\statistical_audit_summary.csv`
- `logs\statistical_audit.log`

## 7. Additional Reviewer-Facing Validation

This runs three supplementary checks: LinUCB contextual-bandit baseline,
catalog-size sensitivity at 200/400/800 products, and gradual
multi-dimensional drift.

```powershell
.\scripts\run_validation_extensions.ps1
```

Generated outputs:

- `results\validation_extensions.json`
- `results\validation_extensions_summary.csv`
- `logs\validation_extensions.log`
- `logs\validation_extensions_runtime.json`

## Notes

The public package excludes raw reviewer-level files. Commands that rebuild raw-derived artifacts or reviewer-product associations require placing the separately obtained raw datasets under `data/raw/`. When reporting a reproduction run, also note:

- machine CPU/RAM if easy to check;
- wall-clock time for the main run;
- wall-clock time for the benchmark run;
- whether any warnings or errors appeared in the logs.





