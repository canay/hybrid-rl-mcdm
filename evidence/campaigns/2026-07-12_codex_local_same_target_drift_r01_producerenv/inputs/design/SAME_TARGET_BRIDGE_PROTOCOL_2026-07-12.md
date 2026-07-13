# Same-Target Bridge Protocol

Date/time: 2026-07-13 00:16 +03:00  
Tool: Codex  
Model, if known: GPT-5 Codex  
Operation ID: HRE_R1_PRODUCERENV_BRIDGE_LOCK_20260713_CODEX_04

Status: `DRAFT_PRODUCER_ENV_RELOCK_AFTER_250_EXACT_PASS`  
Campaign ID: `2026-07-12_codex_local_same_target_bridge_r01_producerenv`

## Purpose

Determine which reviewed r1 findings survive minimal protocol corrections
without changing the research question, evaluation target, catalog membership,
method definitions, seeds, checkpoints, metric, or fusion weight. This bridge
does not replace r1 with r02. It isolates why the original result changes.

## Producer Environment and Provenance Gate

The canonical producer evidence is public Git tag `v1.0-submission`, commit
`3b92f6485d20d1a45ac03b60077d20af08060885`. Its original-result SHA-256
`cfeaff03084df0d3f0a07a5c8c40308027ca7980288a89cf3616c588d0791ce4`
and catalog-manifest SHA-256
`81b01f5580109552fc6086c67441159ddad40c1d1447f1061a092a88c6c89652`
are byte-identical to the project artifacts. Its requirements file SHA-256 is
`5241d0abaccd86ffad73f36592acbabb1bf9331be83dd4678b4ef5d6be71f391`
and pins Python 3.12-compatible NumPy 1.26.0, pandas 2.2.3, and SciPy 1.16.3.

Before any corrected-arm canonical result, all 50 catalogs by five profiles were
replayed under this isolated producer environment. All 250/250 cells matched
stored final rankings, seven checkpoints, Q arrays, and visits exactly; report
SHA-256 `ba76de6b8043ac4baf34f4f5763c96b2c778d458ec497febc97b6f9135aa655d`,
stderr 0. The fresh runner and independent verifier must refuse any different
NumPy/pandas/SciPy version and must hash-lock the tag sources, requirements, and
250-cell report.

The earlier NumPy-2.x campaign is failed-closed, archived, and noncanonical.
Its partial corrected outputs are never interpreted or resumed.

## Frozen Invariants

- The 50 original 400-item catalogs and their original order are reused.
- Each catalog hash must match `results/amazon_primary.json` and
  `data/processed/manifest.json`.
- The original five profiles and original target sets/rankings stored in
  `results/amazon_primary.json` are immutable evaluation labels.
- The frozen May code values are used exactly, including Loyal recency weight
  `0.25`. The restored manuscript's `0.80` Loyal entry is a disclosure mismatch
  to resolve only after results are verified; it is not silently changed inside
  this same-target bridge because that would redefine the target.
- TOPSIS scores/weights and TOPSIS top-7 must match the original artifact.
- Original run/profile seeds, 30,000 interactions, epsilon schedule, learning
  rate 0.05, static `lambda_Q=lambda_T=0.50`, F1@7, NDCG@7, and checkpoints
  remain fixed.
- RL-only and Hybrid share the same Q snapshot within every arm.
- The paired inferential unit is a catalog-resample/Monte Carlo run; the five
  profiles are averaged within each run. No profile or arm is promoted as an
  independent sample, and the 50 resamples are not described as independent
  source-population catalogs.
- No arm result may tune a coefficient, threshold, candidate rule, reward
  model, seed, or interpretation gate.

## Main Factorial

All 18 combinations of three candidate rules, two GT bonuses, and three core
reward implementations are run. Two equation-locked H-funnel endpoints are
added as secondary specification sensitivity, for 20 arms total. This prevents
a favorable single repair from being selected after seeing results.

### Candidate factor

1. `oracle_gt_hidden30`: exact original `GT-top-7 union hidden-top-30` pool.
2. `hidden30_only`: original hidden-utility top-30, with exact GT membership
   removed from pool construction.
3. `full_catalog`: all 400 items; no evaluation information in support.

### GT-bonus factor

1. `0.20`: exact original `+0.20` when the sampled action is in GT-top-7.
2. `0.00`: no evaluation-label term in reward.

The bonus factor is retained under every candidate rule to separate reward-label
exposure from candidate-label exposure.

### Reward-implementation factor

1. `implemented_r0`: exact original fast path, including category fixed at 1
   and midpoint-only `price_fit > 0.999999`. This is a reproduction/control
   condition, not a defensible corrected mechanism.
2. `inclusive_range_fix`: retain the original engagement/conversion coefficient
   formulas, but compute category affinity on the full profile scale and define
   `in_range` as the stated inclusive price interval rather than its exact
   midpoint. This is the narrow literal repair of the old variable semantics.
3. `component_continuous_fix`: retain the original engagement/conversion
   coefficient formulas, but use the full-vector category score and the same
   continuous triangular `price_fit` component used in the frozen hidden-utility
   equation. This is the equation-component fidelity repair.

Secondary `historical_funnel_coefficients_on_may_h` sensitivity is run only at the original
oracle/bonus endpoint and the clean full-catalog/no-bonus endpoint. It preserves
the exact original normalized H used by the target and applies the historically
sourced funnel equations
`P(engage)=clip(0.7H+0.1, 0.05, 0.95)` and
`P(convert|engage)=clip(0.5H, 0.02, 0.80)`. These equations are frozen from
`archive/2026-05-09_v16_yedek/src/hybrid_rl_mcdm_v2.py` (SHA-256
`90AF7D4D3150099D840C510F5FF420B8773659C2A7A579D04E7B6E711DA65E4F`)
and `archive/2026-05-09_v16_yedek/src/supplementary_runs.py` (SHA-256
`A18AF10D9D7C2C81E400910EC1D0DAE4071322DC1FC24AA0F3B6E022984D8BDC`).
Byte-identical copies are locked under
`inputs/historical_reward_provenance/` and verified before compute. This
sensitivity applies historical funnel coefficients to the frozen **May H**;
it is a cross-version reward sensitivity, not a claim that the historical v2
implementation and May utility were identical. It is not a core factorial
level and does not redefine the target.

## Primary Interpretation Anchors

- `oracle_gt_hidden30 / 0.20 / implemented_r0`: exact r0 reproduction gate.
- `oracle_gt_hidden30 / 0.20 / component_continuous_fix`: primary internal
  implementation-repair anchor while preserving the disclosed constructed
  training protocol. Its inclusive-range counterpart is literal-fix
  sensitivity only.
- `full_catalog / 0.00 / component_continuous_fix`: **single primary corrected
  robustness anchor** because it removes both target exposures and uses the
  reward components that generate the frozen May hidden utility.
- `full_catalog / 0.00 / inclusive_range_fix`: narrow literal-fix sensitivity;
  it cannot replace or override the primary corrected anchor after results.
- `full_catalog / 0.00 / historical_funnel_coefficients_on_may_h`:
  cross-version coefficient sensitivity on May H; it cannot replace or
  override the primary corrected anchor after results.

The first anchor establishes numerical provenance. The second determines
whether the internal-by-construction complementarity result survives the actual
code repair. The third is the sole primary leakage-free robustness anchor; the
fourth and fifth are prespecified sensitivities only. A failure in the primary
robustness anchor cannot erase the disclosed internal result; conversely,
success in the internal anchor cannot be presented as leakage-free generalization.

## Exact Reproduction Gate

Before any bridge result is interpreted, the exact-r0 arm must reproduce, for
all 50 catalogs and five profiles, the stored final RL, Hybrid, and TOPSIS sets
and F1 values. The seven original checkpoint F1 values must match as well.
Any mismatch stops the campaign and is diagnosed before other arms are used.

## Analysis

- Report per-arm F1@7 and NDCG@7 means, sample SDs, and fixed-seed 20,000-resample
  percentile bootstrap 95% CIs for each method.
- Report paired catalog-level Hybrid-minus-RL and Hybrid-minus-TOPSIS means,
  bootstrap CIs, win/tie/loss counts, paired t-test, and Wilcoxon sensitivity.
- Report locked paired between-arm contrasts for bonus removal, GT-candidate
  removal, full-catalog exposure, and reward implementation while keeping the
  other factors fixed, for both F1@7 and NDCG@7. Historical-funnel contrasts
  are reported separately as sensitivity, not as a core factor.
- The paired unit is a catalog-resample/Monte Carlo run; five profiles are
  averaged within each run. The 50 resamples are not described as 50
  independent source-population catalogs. Bootstrap intervals and p-values are
  fixed-seed diagnostic sensitivities, not population-generalization claims.
- All bridge inference is diagnostic/exploratory. No familywise claim is made
  across all 18 arms; manuscript claim decisions use the five locked anchors
  and must disclose their exact operating condition.
- Preserve full raw catalog/profile outputs and hashes; do not print partial
  performance during execution. Live console output is progress only and runs
  under `python -u`.

## Drift Bridge

After the main reproduction gate passes, run a separate same-target drift
stage using the original targets, schedules, seeds, and checkpoints. Both
legacy scenarios require an exact replay gate; neither is treated as valid
future-blind evidence:

1. exact original sudden-drift replay (`future pre/post union` candidate pool,
   GT bonus `0.20`, `implemented_r0` reward);
2. exact original 41-step gradual-drift replay with the same legacy endpoint
   union pool, GT bonus, target interpolation keys, and per-key target noise;
3. full-catalog/no-GT-bonus sudden drift with `inclusive_range_fix` reward;
4. full-catalog/no-GT-bonus sudden drift with
   `component_continuous_fix` reward;
5. full-catalog/no-GT-bonus sudden drift with
   `historical_funnel_coefficients_on_may_h` reward;
6. the same three full-catalog/no-GT-bonus reward specifications under the
   original 41-step gradual target schedule, with target membership used only
   at evaluation.

For sudden drift, episode `15000` is the final pre-drift step and the post target
starts only at `15001`; the frozen pre/post target seeds are the original
profile seed and profile seed `+5000`. For gradual drift, the frozen target key
is `round(drift_fraction*40)`, the profile is interpolated at `key/40`, and the
target seed is profile seed `+7000+key` before the ground-truth helper's own
offset. This is reported as **41-step gradual drift**, not as mathematically
continuous drift. The corrected full-catalog arms must be prefix-invariant to
any future target change and may not construct a GT mask during training.

Report final F1 and checkpoint-normalized post-change AUC for Hybrid and RL.
The locked AUC grids are `[15000,16000,20000,25000,30000]` for sudden
drift and `[10000,15000,20000,25000,30000]` for 41-step gradual drift.
Legacy raw-cell replay covers 900 sudden cells (two methods, nine checkpoints,
50 catalogs) and 540 gradual cells (three methods, six checkpoints, 30
catalogs); stored artifacts do not support claiming exact Q/visit replay for
drift. New runs preserve Q/visit hashes prospectively.
Do not use r02's changed target or its continuous-drift redefinition in this
bridge. Those remain separate follow-up evidence.

## Claim Decision Rules

- `0.901/0.574/0.259` remains an authentic result of the exact disclosed
  original protocol if the reproduction gate passes; it is never relabeled as
  leakage-free.
- The statement that the implemented reward matches the manuscript survives
  only if text is changed to the actual implementation or a corrected anchor is
  adopted with newly reported values.
- Strong Hybrid-over-both wording after repair is decided **only** from the
  locked primary corrected anchor
  `full_catalog / 0.00 / component_continuous_fix`: it requires positive paired
  mean differences with bootstrap lower bounds above zero against both RL-only
  and TOPSIS-only. Otherwise wording becomes conditional/mixed without
  suppressing the negative result. Inclusive-range and historical-funnel
  sensitivities characterize specification dependence and cannot rescue a
  failed primary anchor.
- Drift recovery language survives only in a future-blind same-target drift
  arm; the exact old arm is provenance evidence only.
- Manuscript and response-to-reviewers must be revised together after human
  approval of the verified bridge interpretation; a fresh r0-to-r1 diff is last.

## No-Touch Boundary

Until terminal bridge verification, do not alter active manuscript/BibTeX,
response letter, old diff, figures, tables, highlights, or submission package.
Submitted r0 and public GitHub remain untouched.

## Pre-compute Independent Review Resolution

The lock was taken before any multi-arm result was produced or inspected.
Independent design, reward-provenance, drift-forensic, restored-artifact, and
adversarial-runner reviews were resolved as follows:

- active pre-r02 manuscript, Bib, PDF, response TeX/PDF/DOCX, and diff TeX/PDF
  are byte-identical to the revision-end archive;
- the main matrix remains the complete 18-arm factorial, with two historical-
  coefficient sensitivities; no arm was selected after seeing performance;
- the full-catalog/no-bonus/continuous-component arm is the single primary
  corrected anchor; literal-range and historical-coefficient arms are
  sensitivities only;
- the lock creator rejects draft protocols, existing locks, and existing
  outputs; the runner rejects nonempty output directories and validates every
  resumed catalog against the current manifest, seed, dataset, 20-arm/five-
  profile structure, and exact-r0 replay;
- terminal analysis contains F1@7, NDCG@7, paired within-arm comparisons, locked
  between-arm factorial contrasts, and separate sensitivity contrasts under a
  catalog-resample/Monte Carlo interpretation;
- the historical coefficients are explicitly applied to May H as a cross-
  version sensitivity; and
- source compilation plus contract tests passed in the exact producer runtime
  (`19 passed, 1 terminal-smoke test skipped` because compute had not started).
  One exact-r0 30K
  catalog-profile replay also matched stored rankings, seven checkpoints, Q,
  and visits before lock.
