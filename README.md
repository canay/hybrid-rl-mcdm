# Hybrid RL-TOPSIS Research Package

Public reproducibility artifact for the Hybrid RL-TOPSIS e-commerce
recommendation study.

The repository is intentionally limited to files needed by researchers who want
to inspect or rerun the experiments:

- `code/`: experiment, benchmark, validation, and XAI scripts
- `scripts/`: PowerShell wrappers for common runs
- `data/processed/`: shareable processed datasets used by the experiments
- `results/`: JSON/CSV outputs reported by the study
- `requirements.txt`: Python dependencies used in the experiments

Raw review files, manuscript drafts, internal audit notes, and local run logs
are not included. The processed Amazon India catalogs exclude raw reviewer IDs,
reviewer names, review IDs, review text, image links, and product links. The
processed McAuley user file uses anonymized user identifiers.

## Main Evidence Snapshot

- Hybrid F1@7: 0.9006
- RL-only F1@7: 0.5743
- TOPSIS-only F1@7: 0.2594
- Popularity F1@7: 0.1531
- Random F1@7: 0.0200
- Drift final F1@7: Hybrid 0.8274 versus RL-only 0.5663
- LinUCB validation: 0.0971 versus Hybrid 0.9019
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

Run additional validation checks:

```powershell
python code\validation_extensions.py --runs 30 --size-runs 30 --sizes 200 400 800
```

The sparse reviewer-product benchmark requires the original Amazon India CSV
because it reconstructs reviewer-product associations from the raw file. Place
that file at `data/raw/amazon_india.csv` before running:

```powershell
python code\benchmark_recommenders.py --runs 30 --min-items 3 --factors 32 --epochs 80
```

To rebuild the McAuley processed files from scratch, place
`meta_Home_and_Kitchen.json.gz` and `reviews_Home_and_Kitchen_5.json.gz` under
`data/raw/amazon_mccauley_home/`, then run:

```powershell
python code\mccauley_home_data.py --max-users 1000 --min-unique-items 25
python code\mccauley_home_experiment.py
```

## Scope

The method is positioned as CF-free, criterion-rich decision support rather
than a universal replacement for collaborative filtering. The included
collaborative, graph, deep, and external-validation checks delimit where the
method is appropriate and where interaction-rich recommenders should be
preferred.
