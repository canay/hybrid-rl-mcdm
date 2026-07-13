# Static verifier TOPSIS gate-fix overlay protocol (LOCKED)

Date/time: 2026-07-13 03:02 +03:00  
Tool: Codex  
Model: GPT-5  
Operation ID: HRE_R1_STATIC_GATEFIX_LOCK_20260713_CODEX_15

Status: **LOCKED — CANONICAL EXECUTION AUTHORIZED BUT NOT STARTED**

## Purpose and immutability boundary

This is a standalone verifier overlay for the immutable source campaign
`../2026-07-12_codex_local_same_target_bridge_r01_producerenv`. It does not
modify or replace the source campaign's runner, inputs, outputs, verifier,
failure status, or runtime logs. It reads source artifacts and writes only to
this overlay's own `outputs/` and `runtime_logs/` directories.

The source campaign is bound by:

- `RUN_MANIFEST.json`: `0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222`
- canonical terminal: `48677825f4446e2df427a0940dc8c0947b99aef1373ca5dfb6933f35728ad861`
- canonical checkpoint: `803eebfe09be8d62b5f446955f2106fe7ef8b220a979b68c9f9d71acb4827ecd`
- canonical runner status: `6470b05e83827637e34983511359a9eda24d26d0977d7976eb517dfe156ec2f3`
- original verifier: `d1caa594a08b5f5b1142e37cedede3a56bf50f9c40bf3640dc0267a5aa1ed1ed`
- original canonical failure status: `43ed99ccd388949ab2364affa53f86c8eb1c7d3eb201f3f42f47c095eb5f58cd`
- original canonical failure stderr: `992b2e6038f1209ee2b04218ce2c9bb4e195ca2447dfee439d25b0323ee369e5`
- source verified smoke report: `d0579097e7fd1965292a32b10fdb4b2dd3dbd52e5eaa8794fc8eccdd60094cad`

The source manifest, source terminal, and verifier runtime are additionally
bound to Python `3.12.12`; the package lock remains NumPy `1.26.0`, pandas
`2.2.3`, and SciPy `1.16.3`.

## Corrected TOPSIS decision-equivalence gate

For every stored/independently reconstructed TOPSIS vector, the overlay requires:

1. identical vector shapes;
2. finite stored and reconstructed scores and weights;
3. exact top-7 order;
4. exact full 400-item order;
5. maximum absolute score difference no greater than `2e-15`; and
6. maximum absolute weight difference no greater than `2e-15`.

The tolerance is not a result-acceptance relaxation. It is a bounded binary64
representation margin subordinate to exact decision equivalence. Any top-7 or
full-order change fails regardless of numeric proximity.

## Locked regression facts that motivate the gate

Across all 50 source catalogs:

- the frozen producer pandas pathway is bit-exact with all 50 stored score and
  weight vectors and with all 250 profile copies;
- the independent manual CSV parser has global score max-absolute difference
  `1.1102230246251565e-15`;
- only runs 29 and 35 exceed `1e-15` (both equal
  `1.1102230246251565e-15`);
- all 50 full 400-item orders and all 50 top-7 orders are exact.

## Fail-closed behavior

- The independent verifier reconstructs every source-smoke/canonical
  trajectory from zero; it does not import the source runner or frozen core.
- A pre-existing overlay `verification_status.json` or
  `FULL_VERIFICATION.json` blocks execution.
- `--source-output-dir`, `--output-dir`, and `--mode` are mandatory CLI
  arguments. The only syntactically valid modes are `smoke` and `canonical`.
- The manifest execution policy is lifecycle `LOCKED`,
  `canonical_authorized=true`, and `allowed_modes=[smoke, canonical]`, bound to
  authorization operation `HRE_R1_STATIC_GATEFIX_LOCK_20260713_CODEX_15`. This policy is
  asserted before progress initialization, before any status/report write, and
  before scientific replay. Thus canonical fails without creating output.
- Source manifest, terminal, path, environment, original failure status, and
  original failure stderr mismatches fail.
- Non-finite values fail.
- A score mutation of `+1e-12`, any top-7 swap, and any full-order-only swap fail.
- Near-value top-7-boundary and full-order-only swaps fail even when every
  element-wise score difference remains within `2e-15`.
- The source output tree is hashed before and after verification and must remain
  unchanged.
- A report is written only after every gate passes; failure leaves no report.

## Execution gate

The already verified source smoke remains the required smoke input and must
write to a separate, initially clean overlay directory. The independent audit
conditions have been incorporated into this immutable lock. Canonical overlay
execution is authorized by the operation ID above, but this lock promotion does
not itself start canonical execution. A later launcher must still provide all
three required CLI arguments and use a fresh overlay output directory.
