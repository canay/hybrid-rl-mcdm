# Verified R1 Scientific Payload Extraction

Date/time: 2026-07-13 05:14 +03:00  
Tool: Codex  
Model, if known: GPT-5 Codex  
Operation ID: `HRE_R1_VERIFIED_SCIENCE_EXTRACTION_20260713_CODEX_25`

This isolated, derivation-only campaign extracts the complete verified
scientific payload from the canonical same-target static bridge, future-blind
drift bridge, cached latency benchmark, and exact-XAI campaign. It does not
modify the manuscript, response-to-reviewers, bibliography, figures, old diff,
submission package, submitted r0, or any upstream experiment artifact.

## Fail-closed contract

- Only the four explicit canonical roots are accepted. Smoke, archive,
  superseded, partial, and arbitrary caller-supplied paths are not supported.
- Static data are authorized only by the separate TOPSIS gatefix overlay. The
  preserved failed original verifier status is evidence, not the terminal gate.
- Every source campaign ID, schema, terminal state, verification verdict,
  gate, count, manifest hash, and output hash is checked before scientific
  fields are opened.
- JSON parsing rejects duplicate keys, NaN, and infinity.
- The extraction is complete by construction: all 20 static arms, all locked
  factorial and sensitivity contrasts, all future-blind drift conditions, all
  legacy drift reproduction checkpoints, the unfiltered latency gate, and all
  predeclared exact-XAI summaries are emitted.
- Five fixed profile archetypes are averaged within a catalog-resample before
  any across-run XAI summary. Profile breakdowns are descriptive and are not
  promoted to independent samples.
- Exact XAI is described only as a fixed-reference TOPSIS decomposition and a
  realized-policy reward decomposition. It is not causal, counterfactual, or a
  population-preference estimate.

## Execution

Use the locked producer environment and unbuffered Python:

```powershell
python -B build_run_manifest.py
python -B -u extractor.py
python -B -u independent_verify.py
```

`extractor.py` refuses a nonempty `outputs/canonical` directory.
`independent_verify.py` does not import extractor code and writes
`FULL_EXTRACTION_VERIFICATION.json` only after all independent checks pass.

## Canonical outputs

- `VERIFIED_SCIENTIFIC_PAYLOAD.json`
- `static_all_arms.csv`
- `static_locked_anchors.csv`
- `static_all_contrasts.csv`
- `static_anchor_diagnostics.csv`
- `lambda_posthoc_diagnostic.csv`
- `drift_all_conditions.csv`
- `latency_summary.json`
- `xai_run_level.jsonl`
- `xai_summary.json`
- `SOURCE_HASH_MANIFEST.json`
- `FULL_EXTRACTION_VERIFICATION.json` after independent PASS

The lambda grid is the complete fixed grid `lambda_Q=0.1,...,0.9` with
`lambda_T=1-lambda_Q`. It is an evaluation-label, post-hoc oracle diagnostic
reported at every grid point. It is not a tuning exercise, and no grid point
may replace the prespecified `0.50/0.50` primary result.
