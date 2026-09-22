# Residual Exact-Equivalence Stochastic Audit

Date: 2026-09-22

Task: `RESIDUAL_EXACT_EQUIVALENCE_STOCHASTIC_AUDIT`

Architecture: `strict_no_z`

Final status: **PHASE 1 PASS; PHASE 2 ACCEPTANCE GATE PASS; NO PRODUCTION CHANGE**

Training / Stage B executed: **no / no**

Production sampler or `src/` changed: **no / no**

## 1. Executive decision

The audit-only **persistent exchangeable stochastic symmetry resolution
inside an exact deterministic equivalence class** passed both the semantic
kernel gate and the controlled prediction-sensitivity gate.

- It preserved the exact `(J, C_stay, C_geom)` optimum in 100% of the 614
  historical full-tuple ambiguous matrices.
- It left all 2,296 deterministic-unique matrices unchanged, preserved
  20/20 injective coverage, and retained degree-zero structural identity.
- Conditional pathwise slot, candidate, slot-plus-candidate, and
  slot-plus-agent-plus-candidate equivariance was 100% in both rounds when
  the frozen `R` payload was permuted with the problem.
- Different tie replicas did choose different representatives: 93.65% of
  the 614 historical ambiguous matrices showed more than one representative,
  with 2.042 representatives on average across five replicas.
- Those choices changed the exact final E>0 joint-world set in 42.17% of
  `(eval seed, window)` records when requiring equality across every replica
  pair, but the matched normalized Hamming distance was only 1.254% and
  96.12% of matched worlds were exact.
- The resulting metric sensitivity was very small. The maximum aggregate
  spread was 0.003% for minFDE, 0.021% for JFDE, 0.027% for JADE, 0.015% for
  endpoint error, and 0.003% for relative-motion error. All were far below
  the frozen 2% limit.
- The five-replica mean retained D0's prediction quality and remained better
  than exact-protocol GDTS: minFDE -2.98%, JADE -11.69%, JFDE -13.89%, goal
  endpoint -13.73%, and relative motion -20.68% (lower is better).

The data therefore support treating the remaining exact equivalence as
approximately harmless unresolved uncertainty for this ETH validation
protocol. The unique next recommendation is **production-adoption
validation**, not component/global objective redesign. This is authorization
for a separate validation/design step only: this audit did not add a
production policy, freeze Stage A, or authorize Stage B.

## 2. Scope and provenance

| Item | Value |
|---|---|
| Source HEAD at start | `bd26c9dc027c3cee4c38961818dd2d66ef481a8f` |
| Branch | `research/joint-dependency-v2-clean` |
| Checkpoint | strict no-z Stage-A epoch 13 |
| Checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Dataset / split | ETH validation |
| Windows | 139 |
| Evaluation seeds | 2035, 2036, 2037, 2038, 2039 |
| Tie replicas | 0, 1, 2, 3, 4 |
| K / P / M / rank | 21 / 20 / 4 / 8 |
| Device | NVIDIA GeForce RTX 5070 Ti |
| Precision path | frozen BF16 evaluation with existing FP32 islands |
| Machine result | `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/residual_exact_equivalence_stochastic/results.json` |

The audit reused the existing exact FP32 lifting, exact geometry,
arbitrary-precision integer Hungarian solver, lexicographic radix encoding,
and ambiguity-certificate implementation. It did not save dense score
matrices.

## 3. Resolver semantics

For `degree == 0`, refinement is tensor-exact identity. For an interacting
agent, the existing solver first maximizes the exact deterministic tuple

```text
(J, C_stay, C_geom).
```

If that tuple has a unique assignment, the assignment is returned and no
random state is created. Only a `FULL_TUPLE_AMBIGUOUS` matrix receives the
fourth objective `R`.

For all valid edges `e=(slot,candidate)`, a dedicated local CPU RNG constructs
a random permutation `rho`; the exact edge priority is

```text
R[e] = 2 ** rho(e).
```

Different assignments select different edge subsets and therefore have
different exact `R` sums. With `E_valid` valid edges, the implementation uses

```text
extended_weight = deterministic_full_weight * 2**E_valid + R[e].
```

Every feasible assignment has an `R` total strictly below `2**E_valid`, so
the fourth objective cannot alter any of the first three objectives. This is
not score noise, Gumbel refinement, a posterior sample, Gibbs/MCMC, or a
uniform sample over optimal assignments.

The stable state key is

```text
(evaluation_seed, window_index, tie_replica, agent_index, fixed_namespace)
```

and intentionally excludes the refinement round. The same `R_i[P,K]` is
reused in Round 1 and Round 2. Seed mixing does not use Python `hash()`.

## 4. RNG control

The tie stream uses only a local `random.Random` instance. It consumes none of
the global Python, NumPy, CPU Torch, CUDA, production Round-0, or refinement
Gumbel streams. For every `(eval seed, window)` and every tie replica:

1. the categorical control's `pre_initial` state is restored;
2. Round-0 candidate IDs are therefore shared;
3. exact refinement verifies that the round generator is unchanged;
4. the categorical control's `pre_diffusion` state is restored;
5. the complete `window_end` RNG state is compared with the control.

Each replica passed 695 sampler-global, pre-diffusion, and window-end checks,
and 920 round-generator checks. The five replicas therefore differ only in
the persistent residual tie state.

## 5. Phase 0 reproduction

All baseline gates passed exactly:

- unchanged categorical prediction/coverage reproduction: pass;
- D0 prediction reproduction: pass;
- exact real-matrix census reproduction: pass;
- D0 exact-primary-optimal: 100%;
- D0 primary regret: zero in all 2,910 interacting matrices.

The frozen matrix census was reproduced without drift:

| Exact tie level | Count | Rate |
|---|---:|---:|
| `PRIMARY_UNIQUE` | 2,023 | 69.519% |
| `PRIMARY_TIED_RESOLVED_BY_STAY` | 128 | 4.399% |
| `STAY_TIED_RESOLVED_BY_GEOM` | 145 | 4.983% |
| `FULL_TUPLE_AMBIGUOUS` | 614 | 21.100% |
| Total interacting matrices | 2,910 | 100% |

There were also 300 degree-zero structural-identity records.

## 6. Phase 1 semantic kernel gate

Every hard requirement passed:

| Check | Result |
|---|---:|
| Unique cases unchanged | 100% |
| Ambiguous cases preserve `(J,C_stay,C_geom)` | 100% |
| Extended exact assignment unique | 100% |
| Valid/injective 20-of-20 coverage | 100% |
| Same state reproducibility | 100% |
| Persistent state across rounds | 100% |
| Degree-zero identity | 100% |
| D0 exact-primary-optimal | 100% |

Sixteen real ambiguous matrices additionally received selected-edge exclusion
re-solves as sampled exact uniqueness certificates. The general uniqueness
claim follows from the distinct powers-of-two subset-sum construction and is
also covered by exhaustive tiny tests.

### 6.1 Conditional pathwise permutation validation

The score, mask, previous IDs, candidate coordinates, agent tensors, and
frozen `R` payload were consistently relabeled. Both rounds obtained the
following rates:

| Transform | Assignment pathwise | Scene world-set | Matched Hamming |
|---|---:|---:|---:|
| Slot | 100% | 100% | 0 |
| Candidate | 100% | 100% | 0 |
| Slot + candidate | 100% | 100% | 0 |
| Slot + agent + candidate | 100% | 100% | 0 |

This is conditional pathwise equivariance for a jointly transformed random
state. The audit does not impose the incorrect requirement that the same
integer seed regenerated after relabeling must yield the same path.

### 6.2 Historical representative diversity

Across the 614 historical ambiguous matrices and five tie replicas:

| Stratum | Ambiguous matrices | Mean distinct representatives | Mean pairwise assignment Hamming |
|---|---:|---:|---:|
| Overall | 614 | 2.042 | 5.459% |
| Round 1 | 311 | 2.048 | 5.453% |
| Round 2 | 303 | 2.036 | 5.465% |
| Degree 1 / component size 2 | 469 | 2.053 | 5.546% |
| Degree 2-3 | 145 | 2.007 | 5.179% |

Overall, 93.65% of ambiguous matrices produced more than one representative.
Thus the sensitivity experiment exercised real stochastic symmetry breaking;
it did not silently collapse to one deterministic representative.

## 7. Phase 2 prediction metrics

All values are five-evaluation-seed means; parentheses contain the standard
deviation across evaluation seeds. Lower is better.

| Policy / tie replica | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 reference | 0.275674 (.002374) | 0.382212 (.004668) | 0.413081 (.005172) | 0.701150 (.006312) | 0.688916 (.006378) | 0.487555 (.016353) | 0.302062 (.003986) |
| R replica 0 | 0.275516 (.002376) | 0.382648 (.004529) | 0.413199 (.005708) | 0.702125 (.007549) | 0.689827 (.007851) | 0.487309 (.015375) | 0.301813 (.004563) |
| R replica 1 | 0.275606 (.002358) | 0.382660 (.004539) | 0.413309 (.005596) | 0.702275 (.007450) | 0.689933 (.007654) | 0.487513 (.015452) | 0.301803 (.004557) |
| R replica 2 | 0.275606 (.002358) | 0.382660 (.004539) | 0.413309 (.005596) | 0.702275 (.007450) | 0.689933 (.007654) | 0.487513 (.015452) | 0.301803 (.004557) |
| R replica 3 | 0.275606 (.002358) | 0.382660 (.004539) | 0.413309 (.005596) | 0.702275 (.007450) | 0.689933 (.007654) | 0.487513 (.015452) | 0.301803 (.004557) |
| R replica 4 | 0.275516 (.002376) | 0.382648 (.004529) | 0.413199 (.005708) | 0.702125 (.007549) | 0.689827 (.007851) | 0.487309 (.015375) | 0.301813 (.004563) |

The repeated aggregate rows do not mean that the tie stream was inactive.
Phase 1 observed representative diversity directly. Most representative
changes amount to small slot/world substitutions that leave the evaluated
set metrics invariant; the only nonzero fixed-seed metric spread occurred at
evaluation seed 2036.

### 7.1 Mean R policy versus D0 and GDTS

| Metric | Mean across R replicas | Relative vs D0 | Relative vs GDTS |
|---|---:|---:|---:|
| minADE | 0.275570 | -0.04% | -3.76% |
| minFDE | 0.382655 | +0.12% | -2.98% |
| JADE | 0.413265 | +0.04% | -11.69% |
| JFDE | 0.702215 | +0.15% | -13.89% |
| Joint Goal Endpoint | 0.689891 | +0.14% | -13.73% |
| Compatibility | 0.487431 | -0.03% | -27.74% |
| Relative Motion | 0.301807 | -0.08% | -20.68% |

Compatibility is reported but was not a hard gate. It remained essentially
equal to D0 and substantially better than GDTS.

### 7.2 E=0 / E>0 control

E=0 contains no interacting resolver and was tensor-identical across D0 and
all R replicas: minFDE 0.529271, JADE 0.358294, and JFDE 0.529271. For R
replica 0, E>0 values were minADE 0.263395, minFDE 0.361180, JADE 0.441249,
JFDE 0.790431, endpoint 0.789232, compatibility 0.736260, and relative motion
0.456001. D0 E>0 minFDE/JFDE were 0.360680/0.788957, consistent with the small
overall differences above.

### 7.3 Degree, component, and agent-count strata

The tie mechanism localized as expected:

| Degree / component | R-invoked rate | Mean representatives | Pairwise assignment Hamming |
|---|---:|---:|---:|
| degree 0 / size 1 | 0% | 1.000 | 0% |
| degree 1 / size 2 | 45.9% | 1.501 | 2.632% |
| degree 2-3 | 14.2% | 1.157 | 0.809% |
| degree >=4 | 0% on historical matrices | 1.009 | 0.104% |

The small degree>=4 cross-replica difference arises downstream after other
agents choose different representatives; those matrices did not themselves
invoke `R` in the historical census.

Replica-0 per-agent trajectory decomposition gave minFDE 0.385846,
0.329472, 0.407173, and 0.411522 for degree 0, 1, 2-3, and >=4. These
degree rows use direct per-agent trajectory decomposition and should not be
algebraically recombined with scene-reduced joint metrics.

By scene agent count, replica-0 `(minFDE, JFDE)` was: N=1
`(0.529271, 0.529271)`, N=2 `(0.450264, 0.917759)`, N=3-4
`(0.314473, 0.673190)`, N=5-8 `(0.393412, 1.000992)`, and N>=9
`(0.481656, 1.162069)`. Complete seven-metric strata are retained in the
machine result.

## 8. Coverage, oracle, churn, and two-cycle behavior

Every replica maintained exactly 20 unique candidates at Round 0, Round 1,
Round 2, and final selection. The five replicas had identical aggregate goal
oracle values:

| Stage | Unique candidates | Goal oracle |
|---|---:|---:|
| Round 0 | 20.000 | 0.366712 |
| Round 1, E>0 population | 20.000 | 0.350775 |
| Round 2, E>0 population | 20.000 | 0.350778 |
| Final, all-agent population | 20.000 | 0.368162 |

The final row uses all agents, including E=0, whereas round rows include the
interacting-refinement population; the values are therefore not a direct
round-to-final delta.

Persistent R reduced fresh-Gumbel slot churn but did not eliminate the
existing synchronous two-cycle behavior:

| Statistic | Fresh-Gumbel CPSR | Persistent R (range across replicas) | D0 |
|---|---:|---:|---:|
| Round0 -> Round1 churn | 94.62% | 85.18%-85.19% | 85.41% |
| Round1 -> Round2 churn | 93.45% | 78.81%-78.82% | 79.12% |
| Round0 -> Round2 churn | 91.82% | 46.36%-46.46% | 46.57% |
| Two-cycle rate | 7.47% | 40.56%-40.66% | 40.70% |

Thus persistence clearly removes the fresh per-round noise component:
Round0-to-Round2 churn falls by about 49.5% relative, and Round1-to-Round2 by
about 15.7%. The high two-cycle rate is inherited from deterministic
synchronous refinement semantics, not caused by fresh random state. It is a
remaining engineering/method behavior to monitor during production-adoption
validation, but it did not create coverage loss or material metric
sensitivity here.

## 9. Direct tie-only sensitivity

### 9.1 Assignment sensitivity

Across all actual Phase-2 matrices, Round 1 had 1.204 representatives on
average and 1.060% mean pairwise assignment Hamming; Round 2 had 1.213 and
1.151%. Degree-one/size-two cases remained the main sensitivity locus.

### 9.2 Joint-world sensitivity

| Stratum | Exact set across all replica pairs | Matched exact-world fraction | Matched normalized Hamming |
|---|---:|---:|---:|
| E=0 | 100.00% | 100.00% | 0% |
| E>0 | 57.83% | 96.12% | 1.254% |
| Overall | 72.09% | 97.43% | 0.830% |

The exact set-equality rate shows that representative choice is not merely a
no-op. However, the matching statistics show that changes are sparse and
localized rather than a wholesale change of joint hypotheses.

### 9.3 Prediction sensitivity versus normal eval-seed variability

| Metric | Max aggregate spread / D0 | Mean fixed-seed tie-only std | D0 eval-seed std |
|---|---:|---:|---:|
| minADE | 0.0329% | 0.0000444 | 0.002374 |
| minFDE | 0.0030% | 0.0000057 | 0.004668 |
| JADE | 0.0267% | 0.0000539 | 0.005172 |
| JFDE | 0.0214% | 0.0000733 | 0.006312 |
| Endpoint | 0.0154% | 0.0000520 | 0.006378 |
| Relative Motion | 0.0034% | 0.0000051 | 0.003986 |
| Compatibility | 0.0418% | 0.0000998 | 0.016353 |

Tie-only standard deviations are roughly 0.1%-1.9% of the corresponding D0
eval-seed standard deviations. The measured representative uncertainty is
therefore negligible compared with ordinary evaluation-seed variability.

## 10. Runtime and memory

This was a correctness-first Python arbitrary-precision audit implementation,
not a production-optimized solver.

- Per R replica: 2,910 exact matrices, about 72.4-72.8 seconds of exact-solver
  time and 79.2-79.8 seconds of total sampler time.
- D0 sampler time: 7.32 seconds for the same five-seed evaluation.
- R-replica full-forward time: 489.8-503.2 seconds; D0: 428.8 seconds.
- Peak CUDA allocated/reserved: 107,064,832 / 132,120,576 bytes.

The semantic result is positive, but this audit implementation itself should
not be copied into production without a separate performance and parity
review.

## 11. Quantitative acceptance gate

| Frozen criterion | Result |
|---|---:|
| Every replica, every six gated metrics within ±2% of D0 | Pass |
| Maximum aggregate replica spread <=2% | Pass |
| minFDE relative to exact-protocol GDTS <=+2% | Pass (-2.98%) |
| JADE/JFDE/Endpoint/Relative Motion remain better than GDTS | Pass |
| Semantic exactness/equivariance/coverage/identity requirements | Pass |

Final gate interpretation:

```text
eligible_for_separate_production_adoption_validation
```

This phrase does not mean adopted, productionized, or Stage-A frozen.

## 12. Tests and engineering integrity

Added:

- `tools/residual_equivalence_stochastic_audit.py`;
- `tests/test_residual_equivalence_stochastic_audit.py`;
- this report and the compact machine-readable result.

Validation completed:

- targeted CPU tests: 19 passed, 1 CUDA skip;
- targeted real-GPU tests: 20 passed;
- full suite before the final report: 344 passed, 10 skipped;
- `python -m compileall -q .`: pass;
- `git diff --check`: pass.

The tests cover unique-case immunity, ambiguous exact resolution, exact tuple
preservation, powers-of-two uniqueness, radix equivalence, deterministic
replay, cross-seed diversity, persistent state, all required permutations,
degree-zero identity, global/round/diffusion RNG isolation, synchronous
updates, baseline reproduction, CUDA/BF16 execution, and unchanged production
source. The one audit-only metric collector defect encountered during the
first run was a false assumption that all joint metrics were per-agent. The
run stopped before the new resolver was evaluated; the collector was fixed to
preserve native scene/agent reductions, a regression test was added, and all
baseline gates were rerun from the start.

## 13. Required questions

1. **Can persistent exchangeable R resolve the residual ambiguity without
   changing the first three optima?** Yes: 100% triple preservation and 100%
   extended uniqueness.
2. **Are original unique matrices unaffected?** Yes, 100%; `R` is not created
   or invoked for them.
3. **Is conditional pathwise permutation equivariance 100%?** Yes for slot,
   candidate, combined, and agent-combined transformations in both rounds.
4. **How much representative diversity occurs in the 614 historical cases?**
   Mean 2.042 representatives across five replicas; 93.65% changed at least
   once; mean assignment Hamming 5.459%.
5. **Does persistent R reduce fresh-Gumbel churn?** Yes: Round0-to-Round2
   churn drops from 91.82% to about 46.4%, and Round1-to-Round2 from 93.45%
   to about 78.8%. A deterministic synchronous two-cycle remains visible.
6. **Do exact-equivalent representatives change final joint-world sets?**
   Often in exact-combinatorial terms (42.17% of E>0 records fail all-pair set
   equality), but only locally: 1.254% matched Hamming and 96.12% exact matched
   worlds.
7. **How large is tie-only joint-metric sensitivity?** Aggregate spreads are
   0.021% JFDE, 0.027% JADE, 0.015% endpoint, and 0.003% relative motion.
8. **Is it small relative to D0 and eval-seed variability?** Yes. Every
   replica stays within 0.16% of D0 on gated metrics, and tie-only std is only
   about 0.1%-1.9% of normal eval-seed std.
9. **Are D0's strong metrics retained?** Yes; all six aggregate metrics pass
   the ±2% band and the GDTS comparison gates pass.
10. **Is residual equivalence reasonably harmless unresolved uncertainty?**
    Yes for this ETH validation checkpoint/protocol. The conclusion should not
    be generalized to another dataset or checkpoint without validation.
11. **Does the evidence require component/global objective redesign now?**
    No. Representative changes are measurable but prediction-insensitive; the
    failure condition for that redesign was not met.
12. **Unique next recommendation?** Conduct a separate
    **production-adoption validation** of this exact stochastic symmetry
    semantics, including an optimized solver/parity implementation and the
    observed two-cycle behavior. Do not start Stage B yet.

## 14. Stop condition

The audit stops here. No production sampler change, retraining, Stage-A
freeze, Stage-B run, temperature change, fifth deterministic objective, or
fresh-Gumbel modification was performed.
