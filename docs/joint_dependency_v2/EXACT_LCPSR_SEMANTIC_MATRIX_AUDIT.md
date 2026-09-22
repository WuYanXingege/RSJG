# Exact LCPSR Semantic Matrix Audit

Date: 2026-09-22

Task: `EXACT_LCPSR_SEMANTIC_MATRIX_AUDIT`

Architecture: `strict_no_z`

Final status: **PHASE 1 SEMANTIC GATE FAILED; STOPPED BEFORE PHASE 2**

Production sampler changed: **no**

Training / Stage B executed: **no / no**

## 1. Executive decision

The proposed three-level LCPSR semantics

```text
degree == 0: identity
degree > 0: maximize exactly (J, C_stay, C_geom)
```

is not deployment-complete on the frozen ETH matrices. Of 2,910 real
interacting-agent/round matrices, 614 (21.10%) retain more than one exact
optimum after all three objectives. The required selected-edge exclusion
re-solves provide constructive alternative assignments with exactly identical
`(J, C_stay, C_geom)` tuples.

The failure is concentrated where the earlier tie audit predicted it:
degree-one agents in size-two components have a 46.9% full-tuple ambiguity
rate. There is no evidence that SciPy D0 sacrificed the exact primary
objective: its exact-primary-optimal rate is 100%, with zero regret in all
2,910 matrices.

For matrices whose full tuple is unique, exact LCPSR is tensor-exact
equivariant to slot, candidate, slot-plus-candidate, and
slot-plus-agent-plus-candidate permutations. The all-scene rates fail only
because unresolved scenes are correctly counted as unresolved rather than
being assigned an arbitrary representative.

Accordingly:

- Phase 2 paired trajectory evaluation was not run;
- no LCPSR prediction metrics are reported;
- the current LCPSR design is **not eligible for production-adoption
  validation**;
- the unique next recommendation is to return to a separate semantic design
  review for the residual exact equivalence classes. This audit does not
  authorize a fourth tie breaker, index fallback, epsilon, random choice, or
  production change.

## 2. Scope and immutable protocol

| Item | Value |
|---|---:|
| Source HEAD at start | `1cd175c9ac2ca96ad310f7f0f2f26c8e2fa57439` |
| Checkpoint | strict no-z Stage-A epoch 13 |
| Checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Dataset / split | ETH validation |
| Windows | 139 |
| Seeds | 2035, 2036, 2037, 2038, 2039 |
| K / P / M / rank | 21 / 20 / 4 / 8 |
| Candidate mask | agent-level, slot-invariant; valid K >= P |
| Refinement | two synchronous rounds |
| Device | NVIDIA GeForce RTX 5070 Ti |
| Interacting matrices | 2,910 |
| Degree-zero structural identity records | 300 |

The audit added only:

- `tools/lexicographic_continuity_audit.py`;
- `tests/test_lexicographic_continuity_audit.py`;
- this report and the compact machine result.

No file under `src/` was modified. The tool uses an audit-only controller and
restores every monkey patch on exit.

## 3. Exact reference implementation

### 3.1 Canonical snapshots

Each refinement round uses one detached FP32 conditional-score snapshot.
Permutation tests transform that snapshot; they never recompute relation,
energy, `index_add_`, or conditional scores.

Every finite IEEE-754 FP32 value is decoded into a signed integer mantissa and
a power-of-two exponent. A common exact unit converts a complete score matrix
to Python arbitrary-precision integers without rounding. Candidate world
coordinates receive the same treatment, and squared geometric distances are
computed with exact integer arithmetic.

No `isclose`, ULP tolerance, floating epsilon, or scalar perturbation defines
an optimum.

### 3.2 Tuple-valued assignment

For each interacting agent, the rectangular injective assignment maximizes:

```text
J       = sum selected exact conditional scores
C_stay  = count of slot-wise previous IDs retained
C_geom  = negative exact squared world-metre displacement
```

The implementation uses exact radix composition whose bases are derived from
strict upper bounds on all lower-priority total ranges. This is an integer
encoding of lexicographic tuple comparison, not epsilon scalarization.
The underlying rectangular Hungarian augmenting-path solver performs all
arithmetic and comparisons with Python arbitrary-precision integers.

### 3.3 Uniqueness certificate

After obtaining an optimum, the audit forbids each selected edge in turn and
re-solves the exact assignment. An alternative with the same exact objective
certifies non-uniqueness. A certificate records:

- the forbidden selected edge;
- an alternative assignment;
- the original and alternative exact tuples.

No arbitrary representative from an ambiguous optimum is passed to trajectory
evaluation.

## 4. Phase 0 reproduction gate

Phase 0 reproduced the authoritative Policy D0 result exactly:

- original categorical metrics and decomposition: maximum absolute error 0;
- D0 metrics and decomposition: maximum absolute error 0;
- D0 slot-permutation diagnostics: maximum absolute error 0;
- final unique candidates: 20.0 / 20;
- 300 degree-zero identity checks;
- 695 shared Round-0 checks;
- 695 pre-diffusion, sampler-global-RNG, and window-end checks;
- 920 unchanged refinement-generator checks.

One separately recomputed floating Policy-D pass differed from its historical
trajectory result for seed 2036 by at most 0.001019. Its permutation audit was
still reproduced exactly. The source is an invalid cross-forward bitwise
assumption: CUDA `index_add_` reductions can perturb a near-tied floating
representative even when the canonical per-forward computation is valid.
Policy D prediction reproduction is therefore retained as a diagnostic, not
as the Policy-D0 hard gate. This does not relax exact equality inside any
canonical matrix. The required D0 reproduction itself was exact.

## 5. Four-level exact tie statistics

### 5.1 Overall

| Exact resolution level | Count | Rate |
|---|---:|---:|
| `PRIMARY_UNIQUE` | 2,023 | 69.519% |
| `PRIMARY_TIED_RESOLVED_BY_STAY` | 128 | 4.399% |
| `STAY_TIED_RESOLVED_BY_GEOM` | 145 | 4.983% |
| `FULL_TUPLE_AMBIGUOUS` | 614 | 21.100% |
| Total | 2,910 | 100% |

Round-specific results are stable:

| Round | Primary unique | Stay resolves | Geometry resolves | Full ambiguous |
|---|---:|---:|---:|---:|
| Round 1, n=1,455 | 69.622% | 4.055% | 4.948% | 21.375% |
| Round 2, n=1,455 | 69.416% | 4.742% | 5.017% | 20.825% |

Thus 887/2,910 matrices (30.48%) have a genuine exact primary tie. The other
2,023 (69.52%) have a nonzero exact primary gap and require no continuity
semantics to identify the primary assignment.

### 5.2 Degree and component localization

| Stratum | n | Primary unique | Stay | Geometry | Full ambiguous |
|---|---:|---:|---:|---:|---:|
| degree=1 | 1,000 | 30.3% | 9.1% | 13.7% | **46.9%** |
| degree=2-3 | 1,020 | 81.37% | 3.63% | 0.78% | 14.22% |
| degree>=4 | 890 | 100% | 0% | 0% | 0% |
| component size=2 | 1,000 | 30.3% | 9.1% | 13.7% | **46.9%** |
| component size=3-4 | 900 | 89.78% | 2.44% | 0.22% | 7.56% |
| component size>=5 | 1,010 | 90.30% | 1.49% | 0.59% | 7.62% |

Agent-count localization:

| Agent count | n | Primary unique | Full ambiguous |
|---|---:|---:|---:|
| N=2 | 400 | 75.25% | 14.25% |
| N=3-4 | 1,500 | 54.0% | 32.0% |
| N=5-8 | 830 | 88.19% | 9.28% |
| N>=9 | 180 | 100% | 0% |

The remaining blocker is therefore not a rare global numerical corner case.
It is a material exact symmetry of low-degree/small-component conditional
assignment problems.

## 6. D0 exact-primary audit

| Audit | Round 1 | Round 2 | Overall |
|---|---:|---:|---:|
| Matrices | 1,455 | 1,455 | 2,910 |
| D0 exact-primary-optimal | 100% | 100% | 100% |
| Non-optimal count | 0 | 0 | 0 |
| Mean/median/p90/max exact regret | 0 | 0 | 0 | 0 |

The D0-to-LCPSR comparison is:

| Category | Count | Rate |
|---|---:|---:|
| D0 non-primary-optimal / primary correction | 0 | 0% |
| Changed by stay semantics | 107 | 3.677% |
| Changed by geometry semantics | 65 | 2.234% |
| Full tuple unresolved | 614 | 21.100% |
| Unchanged | 2,124 | 72.990% |

This rules out “SciPy missed the exact primary optimum” as the mechanism on
these canonical D0 matrices. The prior pathology is set-valued primary/tie
semantics and enumeration dependence, not primary regret.

The audit does not claim that every old permutation-changed scene maps
one-to-one to one tie level, because the old audit aggregated scene-world
changes while this audit classifies agent-round matrices. What it establishes
exactly is the population decomposition: 30.48% are true primary ties and
69.52% have an exact nonzero primary gap; none of the original D0 assignments
has nonzero exact primary regret.

## 7. Permutation semantics

The same canonical snapshot was transformed under:

- slot permutation: score rows, mask rows, previous IDs, and world slots;
- candidate permutation: score/mask columns, coordinates, previous labels,
  and output labels;
- slot plus candidate;
- slot plus agent plus candidate.

For scenes in which every interacting assignment was uniquely resolved:

- slot pathwise equivariance: 100%;
- candidate-enumeration equivariance: 100%;
- combined equivariance: 100%;
- scene world-set equality: 100%.

Across all 460 E>0 seed-window observations per round, unresolved scenes are
correctly treated as failures rather than assigned arbitrary outputs:

| Round | Fully resolved scenes | All-scene exact world-set rate |
|---|---:|---:|
| Round 1 | 259 / 460 | 56.304% |
| Round 2 | 265 / 460 | 57.609% |

Candidate, slot, and both combined transforms have the same rates. They do
not reach the required >=95%. This is not an upstream recomputation artifact:
all transforms use the same frozen score snapshot. It is directly explained
by exact full-tuple ambiguity.

## 8. Example ambiguity certificate

For seed 2035, window 80, round 1, agent 0 (degree 1, component size 2),
forbidding selected edge `(slot=1, candidate=20)` produces a different
injective assignment with the exact same tuple:

```text
J       = 24525196 exact score units
C_stay  = 0
C_geom  = -67151837897894 exact geometry units
```

The compact machine result contains all 614 certificates. It does not store
dense score tensors.

## 9. Runtime and production feasibility

| Quantity | Value |
|---|---:|
| Exact interacting matrices | 2,910 |
| Exact assignment solves, including exclusions/transforms | 66,909 |
| Exact certification mean per matrix | 22.65 ms |
| Median / p90 | 20.45 / 33.17 ms |
| Exact certification total | 65.93 s |
| Complete five-seed Phase-1 wall time | 646.54 s |

The Python arbitrary-precision implementation is suitable as a correctness
reference. It is not accepted as production-feasible: the audit includes
tens of thousands of exact re-solves, CPU/GPU synchronization, and no
production-oriented optimization. Runtime is secondary here because semantic
non-uniqueness already blocks adoption.

## 10. Phase 1 gate and Phase 2 stop

Passed:

- all returned interacting assignments are exact-primary-optimal;
- every returned assignment is valid and injective, preserving 20/20
  coverage;
- uniquely resolved outputs are exactly equivariant under all required
  transformations.

Failed:

- zero full-tuple ambiguity: 614, not 0;
- all-scene slot/candidate/combined pathwise invariance: 56-58%, not 100%;
- whole-scene unordered world-set invariance: 56-58%, not >=95%.

The hard stop was therefore activated. There are no LCPSR five-seed prediction
metrics, E=0/E>0 trajectory results, runtime comparison against D0, or metric
preservation decision. Reporting such numbers would require choosing an
unauthorized representative for 614 exact equivalence classes.

## 11. Answers to the required questions

1. **How much of the prior pathology is an exact primary tie versus a tiny
   nonzero primary gap?** At the real-matrix population level, 887/2,910
   (30.48%) are exact primary ties and 2,023/2,910 (69.52%) are exact-primary
   unique. D0 has zero exact primary regret everywhere; there are no primary
   correction cases.
2. **Primary unique rate?** 69.519% overall.
3. **Needs stay to become unique?** 128 matrices, 4.399%.
4. **Needs geometry to become unique?** 145 matrices, 4.983%.
5. **Any full-tuple ambiguity?** Yes: 614 matrices, 21.100%.
6. **D0 exact-primary-optimal rate?** 100%, with zero regret.
7. **Why does D0 differ from LCPSR?** Never because of primary correction;
   107 stay changes, 65 geometry changes, and 614 unresolved cases dominate.
8. **Is unique LCPSR 100% slot-equivariant?** Yes, conditional on the full
   tuple being unique. The unconditional scene rate fails because unresolved
   scenes are not assigned outputs.
9. **Candidate-enumeration invariance?** 100% on the uniquely resolved subset;
   the all-scene rate is 56.30%/57.61% because of the hard-stop cases.
10. **Whole-scene invariance >=95%?** No: 56.30% in round 1 and 57.61% in
    round 2 under fail-closed accounting.
11. **Did LCPSR preserve D0 metrics?** Not evaluated. Phase 2 was prohibited.
12. **Solver feasibility?** Appropriate as an exact reference, not accepted
    for production.
13. **Is the blocker solved?** No. Enumeration dependence is removed only for
    the unique subset; material exact equivalence classes remain.
14. **Unique next recommendation?** Do **not** enter LCPSR
    production-adoption validation. Return the residual exact-equivalence
    semantics to a separate design review; make no production or training
    change from this audit.

## 12. Artifacts

- Machine result:
  `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/exact_lcpsr_semantic_matrix/results.json`
- Machine result SHA256 at report generation:
  `f25e5c33f9d3c0e60bdef0369b3028ed7d27471ba11bc01abe417224249552d8`
- Audit implementation: `tools/lexicographic_continuity_audit.py`
- Tests: `tests/test_lexicographic_continuity_audit.py`

No training, Stage-A freeze, production sampler adoption, or Stage B was
performed.
