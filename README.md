# Hybrid RL and entropy-weighted TOPSIS: Revision 1 evidence repository

This repository is the audit-oriented evidence snapshot for the Revision 1
evaluation of:

> *How can criterion-rich e-marketplaces recommend without CF? An explainable
> hybrid RL and entropy-weighted TOPSIS framework*

It preserves locked protocols, executable analysis and verification code,
hash-bound campaign records, independent verification reports, and the
scientific payload used to update the revised manuscript. It is not a claim of
journal acceptance, production deployment, or universal method superiority.

## What the revised evidence shows

The corrected primary evaluation is target-blind: hidden relevance is used for
evaluation, not for candidate construction, reward construction, or policy
input. Across 50 catalog resamples, mean F1@7 is:

| Method | Mean F1@7 |
|---|---:|
| Hybrid | 0.330857 |
| RL-only | 0.129143 |
| TOPSIS-only | 0.259429 |

The paired Hybrid-minus-RL difference is +0.201714 (catalog-bootstrap 95% CI
[0.181714, 0.221143]; 50 wins, 0 losses, 0 ties). The paired
Hybrid-minus-TOPSIS difference is +0.071429 (95% CI [0.044571, 0.098286]; 34
wins, 12 losses, 4 ties).

These aggregate results do not imply dominance in every operating profile.
Hybrid exceeds RL-only in all five fixed profiles, while TOPSIS-only is better
than Hybrid in the Explorer and Balanced profiles. The method is therefore
presented as a complementary, auditable fusion design rather than a universally
best recommender.

## Original-protocol and corrected-primary boundary

The exact replay of the original, target-assisted protocol remains authentic:
F1@7 is 0.900571 for Hybrid, 0.574286 for RL-only, and 0.259429 for
TOPSIS-only. Those values are retained as historical protocol evidence only.
They are not used as the revised primary performance claim.

The stricter target-blind evaluation was triggered by peer-review scrutiny. It
does not erase the original experiment; it changes which protocol supports the
main empirical claim.

## Why a `failed` verifier record is retained

The corrected primary computation completed all 50 catalogs and wrote an
immutable terminal output. Its first independent verifier then rejected run 29
because a manually reconstructed TOPSIS score differed by
`1.1102230246251565e-15`, just above an exact `1e-15` numeric threshold. The
top-7 order and the full 400-item order were identical. Thus the failure was a
verifier-tolerance classification error, not an unrun experiment or a failed
scientific result.

The original failure record is deliberately preserved. A separate immutable
overlay verifier introduced a binary64 margin of `2e-15` while still requiring
exact top-7 and exact full-order decision equivalence. It replayed all 50
catalogs and 5,000 cells and ended `completed_verified/PASS`. The source
scientific output was neither rerun nor edited for this gate fix. See
[`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) for the exact paths and hashes.

## Repository map

- `evidence/campaigns/`: locked campaigns, source code, tests, canonical
  outputs, and independent verification reports. Internal provenance strings
  retain their original `experiments/...` staging paths so the immutable
  records remain byte-identical.
- `evidence/legacy_external_boundary/`: secondary McAuley, sparse-graph, and
  deep-model boundary checks. These are not pooled with the corrected primary.
- `evidence/privacy/`: the staging-source privacy report, redaction manifest,
  and the current public-tree scan report.
- `data/README.md`: the public data and redistribution boundary.
- `EVIDENCE_INDEX.md`: claim-to-artifact map and interpretation limits.
- `EVIDENCE_MANIFEST.json` and `SHA256SUMS.txt`: byte-level inventory.
- `verify_evidence.py`: dependency-free integrity and semantic-gate verifier.

## Verify the evidence snapshot

Python 3.12 is recommended. The top-level verifier uses only the standard
library:

```bash
python verify_evidence.py
python -m pytest -q
```

A successful run prints a JSON report with `"verdict": "PASS"`. It checks the
complete file inventory, SHA-256 values, the preserved source-verifier failure,
the independent overlay PASS, all canonical FULL-verification reports, the
canonical-to-privacy-safe payload hash binding, corrected-primary and exact-r0 values, drift effects, exact-XAI
reconstruction bounds, latency quantiles, and forbidden-path/privacy rules.

The default pytest configuration runs only the public-snapshot integrity,
privacy, and semantic-gate tests. Campaign-level tests remain beside their
source code for audit and can be invoked by explicit path with the exact
versions in [`ENVIRONMENT.md`](ENVIRONMENT.md). Their full contract suites also
require the locked NumPy 1.26.0 producer environment and source-derived
catalogs or smoke trees that are intentionally not distributed here; running
all campaign tests in an arbitrary environment is therefore not a valid
verification procedure. This limitation is explicit in
[`DATA_AND_THIRD_PARTY_NOTICE.md`](DATA_AND_THIRD_PARTY_NOTICE.md).

Four JSON artifacts contain privacy-safe replacements for local interpreter or
host-path strings. `evidence/privacy/REDACTION_MANIFEST.json` binds each original and
public SHA-256 value and records a deep comparison with zero scientific numeric
changes.

## Additional verified boundaries

- Future-blind drift: Hybrid-minus-RL final-F1 gaps are +0.252000 under sudden
  drift and +0.229524 under gradual drift for the revised primary reward model.
- Exact XAI: the fixed-reference decomposition is numerically exact within a
  maximum recorded reconstruction error of `6.66133814775094e-16`; it is
  descriptive, not causal.
- Latency: median/P95/P99 are 53.5/117.6/150.2 microseconds for a warm, cached,
  400-item fusion-plus-full-sort kernel on the tested laptop. This is not an
  end-to-end, network, production, or generic edge-device benchmark.
- McAuley secondary boundary: sampled-candidate mean F1@7 is 0.0658 for Hybrid,
  0.0642 for TOPSIS-only, 0.0559 for RL-only, and 0.1301 for popularity. The
  result explicitly rules out a universal-superiority interpretation.

## Citation and license

Citation metadata are in [`CITATION.cff`](CITATION.cff). Original repository
code is released under the MIT License. Dataset rights, frozen third-party
provenance, and evidence-output reuse are governed by
[`DATA_AND_THIRD_PARTY_NOTICE.md`](DATA_AND_THIRD_PARTY_NOTICE.md); the MIT
license does not transfer rights in third-party source data.
