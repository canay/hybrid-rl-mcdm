# Cached static Hybrid-RL ranking latency protocol

Status: **LOCKED / CANONICAL RUN AUTHORIZED BUT NOT YET EXECUTED**  
Date/time: 2026-07-13 04:15 +03:00  
Tool: Codex  
Model: GPT-5 Codex  
Operation ID: `HRE_R1_LATENCY_CANONICAL_LOCK_20260713_CODEX_20`

## Purpose and claim boundary

This isolated campaign measures only the warm, cached inference path used after
the 400-item Q and TOPSIS score vectors already exist:

1. `static_hybrid_score(q_scores, topsis_scores, lambda_q=0.50)` from the
   hash-locked public-tag producer core; then
2. the producer's original full-array top-7 operation,
   `np.argsort(scores)[::-1][:7]`.

It does **not** time data loading, feature construction, TOPSIS construction,
RL training, explanation generation, serialization, network transport, or an
end-to-end request. Any later manuscript statement must preserve this boundary.

## Evidence source and allowlist

- Source campaign:
  `experiments/2026-07-12_codex_local_same_target_bridge_r01_producerenv`
- Scientific-data source: `outputs/canonical_main/main_catalogs.jsonl`, exactly
  50 ordered runs, SHA-256
  `803eebfe09be8d62b5f446955f2106fe7ef8b220a979b68c9f9d71acb4827ecd`.
- Canonical source terminal/status/source-manifest SHA-256 values:
  `48677825...`, `6470b05e...`, and `0428ecd9...` respectively.
- Distinct verification evidence campaign:
  `experiments/2026-07-13_codex_static_verifier_topsis_gatefix`.
- Required overlay FULL/status/RUN-manifest SHA-256 values:
  `a3112da7...`, `2513c892...`, and `ab29bb47...`; overlay state must be
  `completed_verified/PASS` and bind all three canonical source outputs.
- Required locked scoring core SHA-256:
  `46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f`.
- Arm allowlist:
  `candidate=full_catalog__bonus=0.00__reward=component_continuous_fix`.
- Vector allowlist: exactly 50 runs x 5 ordered profiles = 250 cached pairs.
  Fixture schema v2 contains only TOPSIS `(50,400) <f8`, Q
  `(50,5,400) <f8`, and expected top-7 `(50,5,7) <i8`, with typed
  dtype/shape/nbytes hashes. Ground truth, metrics, visits and reward diagnostics
  are explicitly excluded.

No drift payload may be read by this campaign.

## Canonical benchmark contract

- batch size: 1 profile/vector pair per timed call;
- vector length: exactly 400;
- warm-up: 10,000 untimed calls;
- passes: 3;
- blocks per pass: 20;
- calls per block: 5,000;
- total timed calls: 300,000;
- deterministic balanced schedule: 250 pairs, 20 calls per pair per block and
  1,200 timed calls per pair across the complete canonical run;
- timer: `time.perf_counter_ns`;
- exact timer backend: Windows `QueryPerformanceCounter()`, monotonic with a
  positive recorded resolution;
- timed region: locked score fusion plus full `argsort` top-7 only;
- garbage collection: disabled for every timed block and restored immediately;
- process: Normal priority, exactly one logical processor;
- exact runtime: CPython 3.12.12 from
  `experiments/_runtime/hre_submission_py312_numpy1260_pandas223/Scripts/python.exe`
  with NumPy 1.26.0;
- BLAS/OpenMP variables: forced to one before NumPy import;
- raw durations: retained as an unsigned 64-bit NumPy array;
- statistics: raw median/P95/P99, mean, population SD and CV; per-block and
  per-pass equivalents; timer overhead measured separately;
- uncertainty: deterministic-seed block bootstrap CIs over the medians of
  block-level median/P95/P99 estimates;
- accuracy: top-7 must exactly match all 250 allowlisted canonical pairs before
  and after timing.

Smoke mode may reduce warm-up, passes, blocks, calls, and bootstrap replicates
only to validate plumbing. Smoke output cannot support a paper latency claim.

## Stability diagnostics fixed before canonical execution

Within each pass, let the robust center be the median of the block medians and
let MAD be the median absolute deviation. A block is stable exactly when its
sample count is complete and its median differs from the robust center by no
more than
`max(20% of center, 6 * 1.4826 * MAD, 500 ns)`.

The rule is independent of the 1 ms claim threshold. Stable blocks are not
deleted from the raw artifact; their flags are recorded and independently
recomputed. These diagnostics describe timing stability only and are never used
to exclude slow calls from the primary retention gate.

## Fail-closed retention gate

The `<1 ms P99` statement may be retained only after a canonical run and an
independent verifier PASS, and only if all of these hold:

1. the P99 over **all 300,000 raw timed calls** is strictly below 1,000,000 ns;
2. every pass's P99 over **all raw calls in that pass** is strictly below
   1,000,000 ns;
3. equality to 1,000,000 ns fails both comparisons;
4. all pre/post top-7 accuracy checks exactly match the allowlist;
5. pre/post priority and one-core affinity, available actual thread-pool state,
   timer monotonicity/resolution, source hashes, fixture lineage, raw hash,
   dimensions, statistics, bootstrap results, and gate are independently
   verified.

No timer-overhead subtraction, stability-based filtering, outlier trimming, or
block exclusion is used for the retention gate. If any condition fails, the
sub-millisecond claim must be removed or narrowed; a failed gate is not
reclassified as a scientific success. Stable-block P99 values remain secondary
diagnostics.

## Runtime and output rules

The runner requires an empty output directory, writes status atomically, emits
unbuffered progress/ETA, and terminates as `completed_unverified`. The verifier
does not import the runner; it recomputes hashes, statistics, stability,
bootstrap intervals, the exact fixture lineage from the canonical scientific
data plus distinct overlay verification evidence,
accuracy, and the gate independently. Pre-existing verifier artifacts are a
fail-closed error and are quarantined so a stale PASS cannot remain at the
canonical terminal path. Terminal status files bind the terminal artifact by
SHA-256. Only a verifier PASS may produce `completed_verified`.

The exact runner path and SHA-256, fixture builder, lock script, verifier,
protocol, verified fixture and both adversarial/regression test files are bound
by `RUN_MANIFEST.json`. This manifest explicitly records `LOCKED`,
`canonical_launch_authorized=true`, the allowed smoke/canonical modes, exact
producer environment and all canonical data/overlay evidence hashes. The
verifier reconstructs the complete retention-gate object
from verified raw data, accuracy and runtime facts, and writes that recomputed
object into `FULL_VERIFICATION.json`; it never copies the runner-supplied gate.

All manifest-listed lock files are read-only after the lock operation. This
lock authorizes a later canonical launch but does not claim canonical timing has
already run. The lock operation itself finishes with producer-environment tests
and a fresh verified smoke; canonical timing remains a separate execution.
