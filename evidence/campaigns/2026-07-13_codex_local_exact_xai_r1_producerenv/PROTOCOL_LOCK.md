# Exact XAI Validation Protocol

Date/time: 2026-07-13 02:05 +03:00  
Tool: Codex  
Model, if known: GPT-5 Codex  
Operation ID: HRE_R1_EXACT_XAI_DESIGN_20260713_CODEX_12

Status: `LOCKED`  
Campaign ID: `2026-07-13_codex_local_exact_xai_r1_producerenv`
Authorization operation ID: `HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19`

## Purpose

Replace the earlier row-split surrogate interpretation with an exact,
mechanism-aligned audit of the single prespecified corrected bridge arm:
`candidate=full_catalog__bonus=0.00__reward=component_continuous_fix`.
This campaign does not compare recommendation accuracy, select a favorable
arm, change the restored manuscript, or read an unverified bridge payload.

## Source Gate and No-Touch Boundary

The runner uses two distinct, required roots. Scientific data come from the
same-target bridge data campaign (`outputs/canonical_main` for canonical),
while terminal verification evidence comes from the independently locked
TOPSIS-gatefix overlay (`outputs/canonical_overlay`). The overlay
`FULL_VERIFICATION.json` must be `completed_verified/PASS`, bind all three data
output hashes, and be paired with its terminal overlay verification status.
The extractor reads the verified data record only to select:

- run index and seed;
- catalog relative path and SHA-256;
- primary-arm profile name, Q vector, and visit vector.

It rejects any extracted key containing `gt`, `ground_truth`, `target`,
`label`, `metric`, `f1`, `ndcg`, or `relevance`. No such field is written to
the XAI input. Active manuscript, BibTeX, response, figures, package, r0,
locked static/drift campaigns, and public GitHub are read-only and out of scope.

The bridge dependency is pinned to RUN_MANIFEST SHA-256
`0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222`,
producer core SHA-256
`46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f`,
and NumPy 1.26.0 / pandas 2.2.3 / SciPy 1.16.3. Source FULL, status,
terminal, manifest, catalog, arm identity, run sequence, path containment, and
file hashes are all fail-closed gates. JSON parsing rejects NaN and infinities.

The canonical data binding is fixed to `canonical_main` hashes
`803eebfe...` (catalogs), `48677825...` (terminal), and `6470b05e...`
(runner status). Verification evidence is fixed to overlay FULL
`a3112da7...`, overlay verification status `2513c892...`, and overlay
RUN_MANIFEST `ab29bb47...`. The data RUN_MANIFEST remains `0428ecd9...` and
the frozen core remains `46022b73...`. Canonical completeness is exactly
50 ordered runs x five ordered profiles.

Both runner and verifier require four explicit source arguments: data campaign,
data root, evidence campaign, and evidence root. The independently audited
execution policy is now `LOCKED`, allows `smoke` and `canonical`, and records
`canonical_authorized=true` under authorization operation
`HRE_R1_XAI_CANONICAL_LOCK_20260713_CODEX_19`. This lock authorizes a later
canonical launch; it does not itself launch one.

## Frozen Primary Mechanism

- 50 catalog-resample runs x five frozen profile archetypes for canonical;
  one run x five profiles for smoke.
- Full 400-item candidate set; evaluation-label bonus exactly zero.
- 30,000 interactions; alpha 0.05; epsilon 0.30 decayed by 0.9997 to 0.05.
- Action RNG seed `run_seed + profile_index*13` and reward RNG seed
  `run_seed + profile_index*997`.
- Corrected continuous-component engagement/conversion probabilities are
  reconstructed from the frozen producer profile and catalog columns.
- Static fusion uses `cT=0.5*norm(TOPSIS)` and `cQ=0.5*norm(Q)`.

## Exact Q Reward Decomposition

The same actions and same random draws update four parallel arrays:

`Q_total`, `Q_base`, `Q_engage`, and `Q_convert`.

Per interaction the components are `-0.02`, `0.30 I(engage)`, and
`1.00 I(convert)`. Each component receives the same linear Q update. Therefore
`Q_total = Q_base + Q_engage + Q_convert` must hold itemwise within `1e-12`.
The replayed total Q and visits must also match the verified bridge payload
within `1e-12` and exactly, respectively. Action and reward-event traces are
stream-hashed without exposing performance labels.

For score-space reporting, the producer min-max normalization is decomposed
affinely. For nonconstant Q, with
`d=max(Q_total)-min(Q_total)+1e-10`, the terms are
`cQ_reference=-0.5*min(Q_total)/d` and `cQ_j=0.5*Q_j/d` for base,
engage, and convert. For constant Q the reference is 0.25 and all three driver
terms are zero. Their sum must reconstruct cQ within `1e-12`. These are exact
contributions along the single realized policy path whose actions were
selected by total Q. They are not counterfactual or causal effects of
independently changing reward drivers.

## Exact Four-Criterion TOPSIS Shapley

For every catalog, the complete data fix:

- the entropy-floor weights;
- the criterion vector-normalization denominators;
- the positive and negative weighted ideals;
- the full-catalog TOPSIS-score min and max used by the producer min-max norm;
- the per-criterion catalog medians as the reference vector.

For each item and each of all 16 criterion coalitions, present criteria take the
item value and absent criteria take the fixed catalog median. The coalition
value is the resulting fixed-parameter TOPSIS closeness transformed with the
fixed full-score min/max. Exact four-player Shapley values are enumerated, not
estimated. Efficiency must hold within `1e-12`; dummy, symmetry, and additivity
are unit-tested on controlled games.

## Full Reconstruction Gates

For all 400 items in every catalog/profile cell:

- `Q_total = Q_base + Q_engage + Q_convert` within `1e-12`;
- Shapley baseline plus four criterion values equals normalized TOPSIS within
  `1e-12`;
- `hybrid_score = cT + cQ` within `1e-12`;
- recomputed top-7 rank is the descending producer ordering
  `numpy.argsort(score)[::-1][:7]`, including its tie semantics, and equals the
  verified bridge rank and the ordering from the reconstructed score;
- runner and an independent verifier agree on every array, hash, rank, and
  diagnostic within `1e-12`.

The independent verifier does not import runner code. It recomputes reward
probabilities, stochastic replay, TOPSIS, all 16 coalition values, Shapley
values, fusion scores, hashes, and ordering through its own implementation.

## Execution and Outputs

Both programs must be launched with `python -u`. Output directories must be
new/empty; refusal of a nonempty directory must leave its entire existing tree
byte-identical. Failure status may be written only to a new directory owned by
that launch. Status and progress JSON are written atomically. Console output is
operational progress/ETA only; no partial scientific value is printed. Runner
terminal state is `completed_unverified`. Only the independent verifier may
write `FULL_VERIFICATION.json` and terminal `completed_verified/PASS`.

Expected files:

- `xai_inputs.jsonl`: label-free allowlisted bridge extraction;
- `xai_attributions.jsonl`: exact per-cell/per-item arrays and gates;
- `xai_results.json`: operational terminal summary and hashes;
- `status.json` and `verification_status.json`;
- `FULL_VERIFICATION.json` after independent PASS only.

All JSON/JSONL artifacts use exact versioned schemas, finite-number checks,
fixed catalog-column order, contained paths, explicit TOPSIS criterion order,
score-min/max diagnostics, and evaluation/label-key rejection. The draft run
RUN manifest pins protocol, runner, verifier, builder, tests, source evidence,
and producer environment. All lock files are ReadOnly after the manifest is
generated and checked. Canonical evidence may be produced only by these exact
locked files and the exact bound roots.

## Interpretation Boundary

The output supports exact explanations of the adopted corrected mechanism and
its fixed reference game. It does not establish causal user preference,
population generalization, or accuracy superiority, and it does not revive the
earlier surrogate's broader claims. Manuscript/response integration is a later
paired, archived, approval-controlled step after canonical verification.
