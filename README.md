# Hybrid RL-MCDM

Reproducibility artifact for the manuscript
*"Observable and Behavioral Signal Complementarity in E-Commerce
Recommendation: A Hybrid Entropy-Weighted TOPSIS and Tabular Q-Learning
Framework"* (under review at Expert Systems with Applications).

This repository contains the code, synthetic data, and result artifacts
needed to reproduce every empirical claim in the paper. The manuscript
itself is published through the journal of record and is not mirrored
here.

## Authors and contact

- **Cem Özkurt** (first author)
  Department of Data Science and Analytics
  Faculty of Computer and Information Sciences, Sakarya University,
  Sakarya, Türkiye
  Email: <cemozkurt@sakarya.edu.tr>
  ORCID: [0000-0002-1251-7715](https://orcid.org/0000-0002-1251-7715)

- **Özkan Canay** (corresponding author)
  Department of Information Systems and Technologies
  Faculty of Computer and Information Sciences, Sakarya University,
  Sakarya, Türkiye
  Email: <canay@sakarya.edu.tr>
  ORCID: [0000-0001-7539-6001](https://orcid.org/0000-0001-7539-6001)

For questions about the code, the synthetic data design, or the
reproduction protocol, please contact the corresponding author by email.

## Repository layout

```
.
├── src/                            # Active source code
│   ├── hybrid_pipeline.py          # Main pipeline (data, primary,
│   │                               # ablations, drift, CGF, XAI)
│   ├── pacaon_verify.py            # Operational equivalence with the
│   │                               # entropy-EWM-TOPSIS recommender of
│   │                               # Pacaon and Ballera (2024)
│   └── learned_lambda.py           # Static-vs-oracle lambda analysis
├── data/
│   ├── amazon_stratified_400.csv   # Stratified 400-product catalog seed
│   └── bootstrap/                  # 30 deterministic synthetic runs
│       ├── manifest.json
│       └── synthetic_products_run00..29_seed*.csv
├── results/
│   ├── full_results.json           # Primary results (main, lambda,
│   │                               # split, drift, CGF, ILD)
│   ├── supplementary.json          # Profile-level NDCG, reward shaping
│   ├── pacaon_verify.json          # Kendall tau across 30 seeds
│   └── learned_lambda.json         # Oracle-bound lambda summary
├── requirements.txt                # Pinned package versions
├── CITATION.cff                    # Citation metadata
└── README.md                       # This file
```

## Active experiment files

- `src/hybrid_pipeline.py` — full reproducible pipeline (data
  generation, 30-run primary bootstrap, lambda and GT-split ablations,
  concept drift, confidence-gated fusion, XAI signal decomposition).
- `src/pacaon_verify.py` — empirical verification that the
  TOPSIS-only baseline reproduces the entropy-EWM-TOPSIS recommender of
  Pacaon and Ballera (2024) (Kendall tau = 1.000 across 30 seeds).
- `src/learned_lambda.py` — oracle-bound static-versus-learned
  lambda analysis on the existing lambda ablation grid.

## Reproducing the headline numbers

All scripts assume Python 3.12+ with the dependencies pinned in
`requirements.txt`. From the repository root:

```bash
pip install -r requirements.txt
python src/hybrid_pipeline.py            # primary + ablations + drift + CGF
python src/pacaon_verify.py              # Kendall tau verification
python src/learned_lambda.py             # static vs oracle lambda
```

The four published figures in the manuscript are produced from the JSON
artifacts in `results/`. The figure-generation script is part of the
private working tree and is not redistributed here, since the figures
themselves are part of the manuscript record.

The pipeline is deterministic at the process level. Random seeds are
recorded in every JSON artifact in `results/`. Wall-clock time on a
single CPU-only workstation (12th-generation Intel Core i5-12500,
15.7 GB RAM): approximately 48 minutes for the full pipeline.

## Citing this artifact

Please cite the published manuscript when using this code or data. A
citable software release will be tagged on acceptance; until then,
reference the repository commit hash. Citation metadata is available in
`CITATION.cff`.

## License

Code: MIT. Synthetic data: CC BY 4.0.

## Acknowledgements

This research did not receive any specific grant from funding agencies
in the public, commercial, or not-for-profit sectors.
