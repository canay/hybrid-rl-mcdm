# Evidence index and claim boundaries

This index maps each submission-facing empirical statement to its canonical
evidence. SHA-256 values for every included file are recorded in
`EVIDENCE_MANIFEST.json` and `SHA256SUMS.txt`.

## Corrected target-blind primary

Primary arm:
`candidate=full_catalog__bonus=0.00__reward=component_continuous_fix`.

| Statement | Canonical evidence |
|---|---|
| Hybrid/RL/TOPSIS F1@7 = 0.330857/0.129143/0.259429 | `evidence/campaigns/2026-07-13_codex_verified_payload_extraction_r1/outputs/canonical/VERIFIED_SCIENTIFIC_PAYLOAD.json` |
| Hybrid-RL = +0.201714, 95% CI [0.181714, 0.221143], 50/0/0 | Same payload; `static.analysis.summaries` for the primary arm |
| Hybrid-TOPSIS = +0.071429, 95% CI [0.044571, 0.098286], 34/12/4 | Same payload; `static.analysis.summaries` for the primary arm |
| All corrected-primary source cells independently verified | `evidence/campaigns/2026-07-13_codex_static_verifier_topsis_gatefix/outputs/canonical_overlay/FULL_VERIFICATION.json` |
| Extracted payload independently verified | `evidence/campaigns/2026-07-13_codex_verified_payload_extraction_r1/outputs/canonical/FULL_EXTRACTION_VERIFICATION.json` |

The FULL extraction report binds the canonical payload hash. The public payload
replaces only local runtime-path strings; `evidence/privacy/REDACTION_MANIFEST.json`
binds the canonical and privacy-safe hashes and records zero scientific numeric
redactions.

The unit of uncertainty is the catalog resample. The fixed profile archetypes
are not treated as independent population draws.

## Preserved verifier history

| Artifact | Meaning |
|---|---|
| `evidence/campaigns/2026-07-12_codex_local_same_target_bridge_r01_producerenv/outputs/canonical_main/status.json` | Scientific computation completed all 50 runs and 5,000 trajectories; terminal output written. |
| `evidence/campaigns/2026-07-12_codex_local_same_target_bridge_r01_producerenv/outputs/canonical_main/verification_status.json` | Original verifier failed closed at catalog 29 because its `1e-15` numeric threshold was too strict. |
| `evidence/campaigns/2026-07-12_codex_local_same_target_bridge_r01_producerenv/runtime_logs/canonical_verify.stderr.txt` | One-line original failure message, preserved verbatim; only the public filename extension differs from the locked staging path. |
| `evidence/campaigns/2026-07-13_codex_static_verifier_topsis_gatefix/PROTOCOL_LOCK.md` | Immutable rationale and exact-decision-equivalence gate. |
| `evidence/campaigns/2026-07-13_codex_static_verifier_topsis_gatefix/outputs/canonical_overlay/FULL_VERIFICATION.json` | Independent overlay result: `completed_verified/PASS`; exact top-7 and full order for all 50 catalogs. |

The overlay does not reinterpret a scientific failure. It corrects the checker
classification while preserving the original source output and failure record.

## Exact replay of the original protocol

The same verified payload records the exact-r0 arm
`candidate=oracle_gt_hidden30__bonus=0.20__reward=implemented_r0`:

- Hybrid F1@7: 0.900571
- RL-only F1@7: 0.574286
- TOPSIS-only F1@7: 0.259429

These numbers are authentic but target-assisted. They document continuity with
the reviewer-seen experiment; they do not support the revised primary claim.

## Future-blind drift

Canonical evidence:

- `evidence/campaigns/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/FULL_VERIFICATION.json`
- `evidence/campaigns/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/TERMINAL.json`
- verified payload `drift.analysis`

For the `component_continuous_fix` primary reward model, paired final-F1 gaps
are +0.252000 under sudden drift (95% CI [0.238857, 0.265143], 50/0/0) and
+0.229524 under gradual drift (95% CI [0.205714, 0.253333], 30/0/0).

## Exact descriptive XAI

Canonical evidence:

- `evidence/campaigns/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/FULL_VERIFICATION.json`
- `evidence/campaigns/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/xai_results.json`
- verified payload `exact_xai`

The maximum recorded reconstruction error is
`6.66133814775094e-16`. The decomposition is exact for the fixed scoring and
realized-policy definitions. It is descriptive and must not be described as
causal, counterfactual, or evidence of population preferences.

## Warm-kernel latency

Canonical evidence:

- `evidence/campaigns/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/FULL_VERIFICATION.json`
- `evidence/campaigns/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/latency_results.json`
- `evidence/campaigns/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/raw_durations_ns.npy`

Across 300,000 timed samples, raw median/P95/P99 are
53,500/117,600/150,200 ns. Scope is a warm cached 400-item
fusion-plus-full-sort kernel on the recorded laptop environment, not an
end-to-end serving path.

## Legacy external-boundary diagnostics

The three code/result pairs and their exact hashes are listed in
`evidence/legacy_external_boundary/LEGACY_EXTERNAL_BOUNDARY_MANIFEST.json` and
`evidence/legacy_external_boundary/LEGACY_EXTERNAL_BOUNDARY_EVIDENCE_MANIFEST_2026-07-13.md`.

They are secondary scope checks, not corrected-primary replications. In the
McAuley sampled-candidate check, popularity (0.1301 F1@7) exceeds Hybrid
(0.0658); Hybrid only has a small numerical margin over TOPSIS-only (0.0642).
This negative boundary is intentionally retained.
