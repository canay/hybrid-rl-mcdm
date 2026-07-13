# Same-Target Drift Campaign Lock

Date/time: 2026-07-13 00:54 +03:00  
Tool: Codex  
Model, if known: GPT-5 Codex  
Operation ID: HRE_R1_PRODUCERENV_DRIFT_LOCK_20260713_CODEX_08

Status: `LOCKED_BEFORE_COMPUTE`

This campaign implements only the Drift Bridge section of
`MD/02_design/SAME_TARGET_BRIDGE_PROTOCOL_2026-07-12.md`. Independent
adversarial review confirmed the exact legacy raw-cell gates, future-blind
training separation, prefix-invariance tests, producer-environment lock,
paired analysis, resumability, progress/blinding, and terminal verifier
contract before this status was changed. This copy is now immutable input to
`RUN_MANIFEST.json`.

Frozen design: public tag `v1.0-submission` at commit
`3b92f6485d20d1a45ac03b60077d20af08060885`; Python-compatible NumPy 1.26.0,
pandas 2.2.3, SciPy 1.16.3; 50 exact sudden catalogs and 30 exact 41-step
gradual catalogs; sudden boundary 15000/15001; gradual key
`round(drift_fraction*40)` and target seed `profile_seed+7000+key`; exact
legacy replay gates of 900 sudden and 540 gradual stored raw cells; corrected
full-catalog/no-GT-bonus inclusive, continuous-component, and historical-May-H
reward arms; continuous-component is the sole primary corrected drift arm;
targets are evaluation-only in corrected arms; scientific console output is
forbidden before terminal independent verification.

## Adversarial execution and verification contract

Date/time: 2026-07-13 00:54 +03:00  
Tool: Codex  
Model, if known: GPT-5 Codex  
Operation ID: HRE_R1_PRODUCERENV_DRIFT_LOCK_20260713_CODEX_08

- A new runner output directory must be truly empty. Resume accepts exactly
  `sealed_records.jsonl`, `STATUS.json`, and `PROGRESS.json`; all three must
  agree on campaign, mode, count, sealed-payload hash, and blindness state.
  Resumed scientific records are then fully replayed before continuation.
- Runner and verifier stdout/stderr logs must be redirected to the campaign
  `runtime_logs/` tree, outside `outputs/`. Both commands run with `python -u`; console
  content is progress, percentage, ETA, health, and terminal gate status only.
- The verifier starts only from the exact four-file runner terminal set. It
  rejects stale `FULL_VERIFICATION`, failure, verification-status, verification-
  progress, log, or unexpected artifacts. During the 1,600 canonical profile-
  arm replays (48,000,000 training episodes), it atomically publishes progress-
  only verification status/ETA. `FULL_VERIFICATION.json` is created only after
  terminal PASS; a failure creates a fail-closed failure/status pair and cannot
  coexist with an earlier PASS.
- Every scenario/reward model reports the paired catalog-resample vector for
  Hybrid-minus-RL final F1 and checkpoint-normalized post-change AUC, its mean,
  sample SD, fixed label-derived deterministic 20,000-resample bootstrap CI,
  and win/tie/loss counts. The bootstrap seed and exact raw 50/30-run vector are
  retained; the verifier independently recomputes them exactly.
- The run manifest must contain exactly the prespecified source, test,
  provenance, design, result, manifest, and 50 catalog paths--no duplicates,
  omissions, or extras--and must match campaign, locked status, complete
  scientific contract, exact producer packages, byte sizes, and SHA-256 hashes.
- The terminal binds the sealed payload and run manifest. The completed-
  unverified runner status binds terminal, sealed payload, and manifest. The
  PASS report binds that status, runner progress, terminal, sealed payload,
  manifest, verifier source, and all frozen input hashes. The final runner
  status binds the PASS report and its predecessor status.

The verifier independently reimplements drift loops, reward schedules, target
evaluation, exact raw-cell gates, and analysis. It intentionally reuses the
hash-locked public-tag core for primitive profile, TOPSIS, ground-truth,
ranking, and metric semantics. This preserves exact producer semantics but is
not a wholly independent software stack; that residual dependence is disclosed
in the terminal PASS report.

Pre-lock verification: producer-environment py_compile PASS; 28/28 tests PASS;
independent final adversarial verdict `LOCK-READY` with no open P0/P1. The
public-tag core reuse described above remains a disclosed P2 independence
limitation and cannot be hidden or overstated as a wholly independent stack.
