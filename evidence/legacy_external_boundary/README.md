# Legacy External-Boundary Diagnostics

Date/time: 2026-07-13 09:14 +03:00  
Tool: Codex  
Model: GPT-5 Codex  
Operation ID: HRE_R1_ANON_SUPPLEMENT_PRIVACY_CLOSURE_20260713_CODEX_31

## Interpretation boundary

This directory contains three legacy R1 code/result pairs used as secondary
external-boundary diagnostics:

1. `mccauley_home_experiment.py` with `mccauley_home_real_results.json`;
2. `benchmark_recommenders.py` with `recommender_benchmarks.json`; and
3. `deep_recommender_benchmarks.py` with
   `deep_recommender_benchmarks.json`.

These artifacts are preserved as observed legacy diagnostics. They are not the
corrected target-blind primary evaluation and are not outputs from the current
hash-bound verified payload extractor. Their metrics must therefore not be
substituted for, pooled with, or presented as replications of the corrected
primary results.

No raw data, execution logs, local paths, or identity-bearing metadata are
included. Some scripts require data or dependencies that are deliberately
outside this anonymous package, so direct rerun completeness is not claimed.
The exact code/result hashes and package-relative paths are bound in
`LEGACY_EXTERNAL_BOUNDARY_MANIFEST.json`. The adjacent
`LEGACY_EXTERNAL_BOUNDARY_EVIDENCE_MANIFEST_2026-07-13.md` is a privacy-safe
copy of the project-side human-readable evidence note.
