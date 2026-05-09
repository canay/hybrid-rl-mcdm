# Hybrid RL-TOPSIS Research Package

Public reproducibility artifact for the Hybrid RL-TOPSIS e-commerce
recommendation study.

This repository contains the code, processed datasets, logs, and result
artifacts needed to inspect the reported experiments. Raw review files are not
included in the public package because they may contain reviewer-level text,
identifiers, names, and external links. To fully regenerate raw-derived
artifacts, obtain the source datasets separately and place them under
`data/raw/` using the paths documented in `docs/RUN_EXPERIMENTS.md`.

## Repository Layout

- `code/`: experiment, benchmark, validation, and XAI scripts
- `scripts/`: PowerShell wrappers for the main experiment suite
- `data/processed/`: shareable enriched catalogs and external-validation files
- `results/`: JSON/CSV outputs used by the manuscript
- `logs/`: run logs and runtime summaries
- `docs/`: dataset decision, runbook, manuscript audit, and reviewer audit

## Main Evidence Snapshot

- Hybrid F1@7: 0.9006
- RL-only F1@7: 0.5743
- TOPSIS-only F1@7: 0.2594
- Popularity F1@7: 0.1531
- Random F1@7: 0.0200
- Drift final F1@7: Hybrid 0.8274 versus RL-only 0.5663
- LinUCB additional validation: 0.0971 versus Hybrid 0.9019
- Catalog-size sensitivity: Hybrid remains strong at 200, 400, and 800 items
- Real SHAP surrogate audit: RF surrogate R2 = 0.6404; top drivers are
  `quality_pct`, `price_pct`, and `review_text_richness`

## Reproducing From Included Processed Artifacts

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the primary experiment from the included processed bootstrap catalogs:

```powershell
python code\run_amazon_experiments.py --runs 50 --drift-runs 50
```

Run the statistical audit from the stored primary output:

```powershell
python code\statistical_audit.py
```

Run the SHAP explainability audit:

```powershell
python code\xai_analysis.py --max-runs 50 --shap-sample 5000
```

The sparse reviewer-product benchmark requires the raw Amazon India CSV because
it reconstructs reviewer-product associations. The McAuley and deep benchmark
scripts can be run from the included processed McAuley artifacts.

## Notes

The work is positioned as CF-free, criterion-rich decision support rather than
a universal replacement for collaborative filtering. The included CF, graph,
deep, and external-validation checks are used to delimit the method's operating
regime honestly.
