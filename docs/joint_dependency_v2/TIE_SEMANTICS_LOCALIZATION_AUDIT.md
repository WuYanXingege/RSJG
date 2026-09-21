# Tie Semantics Localization Audit

## 1. Scope and decision

This is the inference-only, read-only `TIE_SEMANTICS_LOCALIZATION_AUDIT`.
It uses the protected strict-no-z epoch-13 checkpoint and does not train,
change `src/`, alter the production sampler, or start Stage B.

The audit first reproduced Policy D from the accepted stochasticity-placement
review. It then evaluated one counterfactual only:

```text
Policy D0
  Round 0: unchanged production weighted Gumbel-Top-P
  refinement, degree(i) == 0: C_i^(l) = C_i^(l-1)
  refinement, degree(i) > 0: unchanged deterministic injective assignment
```

The source hypothesis is correct: every observed degree-zero agent had exactly
slot-invariant conditional-score rows, yet Policy D sent those agents through
Hungarian assignment whenever another agent in the window had an edge. This
caused an arbitrary near-complete slot reorder despite there being no incident
interaction evidence.

D0 removes that defect for isolated agents: their pathwise invariance rises
from 0% to 100%, and their permutation-induced slot-change rate falls from
98.54% to 0%. Prediction quality and 20/20 coverage are preserved.

However, D0 does **not** clear the required scene-level gate. Exact unordered
joint-world-set invariance is only 62.68% in Round 1 and 62.17% in Round 2,
well below 95%. Non-singleton components remain unstable, most strongly for
size-two components (44.53% and 42.27% exact component-world-set invariance).
Fully connected E>0 scenes also remain below the gate (82.37% and 83.66%).

**Decision: Case B2 — within-component multiple-optimum assignment
degeneracy remains the blocker.** The unique next recommendation is a
separate design review of explicit lexicographic continuity semantics for
multiple-optimum assignment. No resolver is implemented here, and D0 is not
productionized.

## 2. Provenance and fixed protocol

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Source HEAD at audit start | `a092f910e3f170cacb16aaa5402858ffaf172558` |
| Architecture | `strict_no_z` |
| Checkpoint epoch | 13 |
| Checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Cache manifest SHA256 | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| Legacy GDTS source SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Stochasticity-review result SHA256 | `9e0a1a08d6bb0d8be6b3433ccbe5aa51a341da203753f3a4d2f2d100a1e19075` |
| CPSR reference SHA256 | `d3346dfa9c457a2c5672a48a152bf94c1a29c63b2b0007990efa394cee701de6` |
| Split / windows | ETH validation / 139 |
| E=0 / E>0 windows per seed | 47 / 92 |
| Seeds | 2035, 2036, 2037, 2038, 2039 |
| K / P / M / energy rank | 21 / 20 / 4 / 8 |
| Refinement | Two synchronous rounds, temperature 1.0 |
| Device | NVIDIA GeForce RTX 5070 Ti |
| Machine result SHA256 | `7200c0bfc3cc4df42e6c022062188a0ad9c60df5b4e452962bc5242e27fe955e` |

Exact command:

```bash
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python \
  tools/tie_semantics_localization_audit.py \
  --config outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/config.yaml \
  --checkpoint outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt \
  --stochasticity-reference outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stochasticity_placement_review/results.json \
  --cpsr-reference outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/cpsr_inference_validation/results.json \
  --output outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/tie_semantics_localization/results.json \
  --device cuda:0
```

## 3. Verified source semantics

The current sampler constructs each refinement score as:

```text
S_i[s,k] = unary_i[k] - accumulated_incident_pair_energy_i[s,k]
```

The accumulator is initialized to zero and receives values only through
`index_add_` at edge source and destination indices. Therefore, if
`degree_i == 0`, no term is added for agent `i` and all P rows equal the same
unary vector exactly.

The refinement loop itself is guarded by window-level `edge_count > 0`, not
agent-level degree. Consequently, an isolated agent in an E>0 window still
enters the assignment solver. This is exactly the source behavior assumed by
the audit.

Observed FP32 diagnostics confirm the derivation:

| Degree | R1 exact / near-invariant rows | R1 mean max row difference | R2 exact / near-invariant rows | R2 mean max row difference |
|---|---:|---:|---:|---:|
| 0 | **100% / 100%** | **0.0000** | **100% / 100%** | **0.0000** |
| 1 | 0% / 0% | 1.8652 | 0% / 0% | 1.9105 |
| 2–3 | 0% / 0% | 6.9764 | 0% / 0% | 8.9147 |
| >=4 | 0% / 0% | 9.8341 | 0% / 0% | 13.8324 |

The near-invariant tolerance is `1e-6`; exact equality was also recorded.

## 4. Audit-only implementation and controls

The intervention is confined to
`tools/tie_semantics_localization_audit.py`. It monkey-patches the existing
structured selection primitive only inside a context manager. The default
sampler, checkpoint, training path, relation, energy, conditional score,
temperature, diffusion, and masks are unchanged.

For D0, every refinement round first computes the existing score exactly as
before. It then copies previous IDs for degree-zero agents and calls the
unchanged exact injective assignment for degree-positive agents. All agents
are synchronously committed by the existing sampler after selection.

For every `(seed, window)`:

1. the full Python, NumPy, CPU Torch, and all-device CUDA pre-initial state was
   restored from the categorical control;
2. D and D0 regenerated the production Round-0 IDs;
3. D0 asserted tensor-exact Round-0 equality with D;
4. deterministic refinement asserted that neither round generator was
   consumed;
5. D0 asserted degree-zero identity and degree-positive equality with D at
   every executed round;
6. the paired categorical pre-diffusion state was restored;
7. the window-end RNG state was required to match the control.

| Assertion | Observed / expected |
|---|---:|
| D0 exact shared Round 0 | 695 / 695 windows |
| D0 degree-zero identity | 300 / 300 agent-rounds |
| D0 degree-positive assignment equal to D | 2910 / 2910 agent-rounds |
| D/D0 deterministic round generators unchanged | 920 / 920 calls each |
| D/D0 global sampler RNG unchanged | 695 / 695 windows each |
| D/D0 paired pre-diffusion state | 695 / 695 windows each |
| D/D0 paired window-end state | 695 / 695 windows each |

The common-slot permutation test consistently permuted score rows, mask rows,
and previous candidate IDs. Results were unpermuted before pathwise and
unordered-set comparison. Component and scene world sets were compared using
a P-by-P Hungarian cost of normalized candidate-ID Hamming distance.

One exploratory execution stopped on a degree-positive equality assertion;
the instrumented clean rerun completed all 2910 exact comparisons. The clean
run is the reported result. The transient is retained as an engineering
caution because CUDA `index_add_` and a multiple-optimum assignment boundary
can be numerically sensitive; it does not relax any acceptance condition.

## 5. Baseline reproduction gate

Both mandatory gates passed before D0 was evaluated:

| Gate | Maximum absolute error | Result |
|---|---:|---:|
| Original categorical metrics/coverage/oracle | 0.0 at `1e-10` | PASS |
| Policy D prediction metrics/coverage/oracle | 0.0 at `1e-10` | PASS |
| Policy D real-score permutation audit | 0.0 at `1e-10` | PASS |

Policy D reproduced 20.000 unique candidates in every round, Round-0 oracle
0.366712, final goal oracle 0.367658, trajectory minFDE 0.382742, and the
previous 55.51% / 56.38% scene-world-set invariance.

## 6. Degree-localized permutation result

All support-set invariance values below are 100%; the failure is candidate-to-
slot path semantics, not loss of injective candidate support.

| Policy / degree | R1 pathwise invariant | R1 slot change | R2 pathwise invariant | R2 slot change |
|---|---:|---:|---:|---:|
| D / degree=0 | **0.00%** | **98.54%** | **0.00%** | **98.54%** |
| D0 / degree=0 | **100.00%** | **0.00%** | **100.00%** | **0.00%** |
| D and D0 / degree=1 | 60.00% | 4.74% | 57.00% | 4.99% |
| D and D0 / degree=2–3 | 90.92% | 0.93% | 91.96% | 0.84% |
| D and D0 / degree>=4 | 100.00% | 0.00% | 100.00% | 0.00% |

The D0 branch completely fixes isolated-agent path semantics. It does not
change any degree-positive result. Degree-positive agents are nevertheless
not generally pathwise safe: degree-one agents fail in 40–43% of tested
agent/permutation cases, and degree-2–3 agents fail in 8–9%.

Mean assignment-objective deltas are zero or numerical-noise scale
(`<=3.7e-8`). Combined with 100% support equality, this is evidence for
multiple equivalent or numerically tied row-to-candidate assignments rather
than a materially different optimum.

## 7. Scene- and component-level localization

### 7.1 Whole-scene world sets

| Policy / round | Exact unordered scene-world-set | Matched exact-world fraction | Matched mean normalized Hamming |
|---|---:|---:|---:|
| D / R1 | 55.51% | 65.95% | 0.12680 |
| D0 / R1 | **62.68%** | **93.99%** | **0.02076** |
| D / R2 | 56.38% | 66.05% | 0.12719 |
| D0 / R2 | **62.17%** | **93.87%** | **0.02153** |

D0 strongly reduces the *severity* of mismatches, but it recovers only 7.17
and 5.80 percentage points of exact scene-set invariance. It therefore fails
the >=95% gate in both rounds.

### 7.2 Graph-class localization

The ETH validation graph population in E>0 windows contains two observed
classes: 62 fully connected windows per seed and 30 windows per seed with one
singleton plus one non-singleton component. No window contains multiple
disconnected components all of size at least two, so that requested stratum
cannot be estimated from this split.

| Policy / graph class | R1 exact set | R1 Hamming | R2 exact set | R2 Hamming |
|---|---:|---:|---:|---:|
| D / singleton-containing | 0.00% | 0.36885 | 0.00% | 0.37059 |
| D0 / singleton-containing | **22.00%** | **0.04367** | **17.78%** | **0.04656** |
| D / fully connected | 82.37% | 0.00967 | 83.66% | 0.00942 |
| D0 / fully connected | 82.37% | 0.00967 | 83.66% | 0.00942 |

In Policy D Round 1, 450 of 614 failed permutation trials (73.29%) occur in
singleton-containing windows. But the causal D0 recovery is only 99 trials,
or 16.12% of all D failures: removing isolated-agent reordering reveals that
the interacting component itself is still often ambiguous. Round 2 gives the
same qualitative result.

### 7.3 Component-local world sets

Component-local results are identical between D and D0 because both retain
the same unordered candidate support per component; D0 changes relative slot
semantics, not the component support set.

| Component size | R1 exact set | R1 Hamming | R2 exact set | R2 Hamming |
|---|---:|---:|---:|---:|
| 1 | **100.00%** | **0.00000** | **100.00%** | **0.00000** |
| 2 | **44.53%** | 0.04593 | **42.27%** | 0.04783 |
| 3–4 | 87.47% | 0.00411 | 88.00% | 0.00378 |
| >=5 | 79.61% | 0.00659 | 82.75% | 0.00596 |

Component-local sets are generally more stable than whole-scene sets, and
singleton component sets are perfectly stable. They are not close to
universally invariant, however. Size-two component failure and fully
connected-scene failure directly establish a within-component assignment
degeneracy. The remaining blocker cannot be reduced to cross-component
relative pairing alone.

## 8. Prediction metrics

All values are lower-is-better five-seed mean ± population standard
deviation.

| Policy | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| D | 0.275460 ± 0.002460 | 0.382742 ± 0.004980 | 0.413168 ± 0.006368 | 0.701790 ± 0.006836 | 0.690064 ± 0.009856 | 0.484819 ± 0.013807 | 0.302062 ± 0.003986 |
| D0 | 0.275674 ± 0.002374 | **0.382212 ± 0.004668** | **0.413081 ± 0.005172** | **0.701150 ± 0.006312** | **0.688916 ± 0.006378** | 0.487555 ± 0.016353 | 0.302062 ± 0.003986 |
| Relative D0 vs D | +0.08% | **-0.14%** | -0.02% | -0.09% | -0.17% | +0.56% | 0.00% |

The changes are small and within the predeclared 2% preservation band. D0
therefore preserves Policy D's strong marginal/joint behavior; it neither
causes a prediction regression nor supplies a metric-based reason to overlook
the failed permutation gate.

### 8.1 E=0 and E>0

E=0 is tensor-equivalent because no refinement round runs.

| Stratum / policy | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| E=0 / D = D0 | 0.358294 | 0.529271 | 0.358294 | 0.529271 | 0.495247 | 0 | 0 |
| E>0 / D | 0.263332 | 0.361287 | 0.441201 | 0.789924 | 0.789591 | 0.732498 | 0.456377 |
| E>0 / D0 | 0.263577 | **0.360680** | **0.441071** | **0.788957** | **0.787856** | 0.736632 | 0.456377 |

### 8.2 D0 per-seed metrics

| Seed | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.279470 | 0.386023 | 0.421386 | 0.708426 | 0.691645 | 0.475572 | 0.304481 |
| 2036 | 0.274752 | 0.385926 | 0.407850 | 0.697990 | 0.676619 | 0.494740 | 0.303541 |
| 2037 | 0.277141 | 0.382573 | 0.416700 | 0.706201 | 0.693972 | 0.509899 | 0.307399 |
| 2038 | 0.272686 | 0.373299 | 0.410717 | 0.690704 | 0.689053 | 0.494413 | 0.297072 |
| 2039 | 0.274320 | 0.383241 | 0.408755 | 0.702426 | 0.693292 | 0.463152 | 0.297817 |

### 8.3 Agent-count and degree strata

| Agent count | minADE | minFDE | JADE | JFDE | Endpoint |
|---|---:|---:|---:|---:|---:|
| N=1 | 0.358294 | 0.529271 | 0.358294 | 0.529271 | 0.495247 |
| N=2 | 0.301169 | 0.450264 | 0.476064 | 0.916671 | 0.939681 |
| N=3–4 | 0.232553 | 0.313559 | 0.382682 | 0.670742 | 0.664217 |
| N=5–8 | 0.295401 | 0.393412 | 0.579798 | 1.000992 | 0.986566 |
| N>=9 | 0.343533 | 0.481894 | 0.656381 | 1.172474 | 1.179343 |

The agent-count table uses the established prediction metric implementation.
Degree is an agent-local label, so the degree table reports the two
well-defined marginal metrics only:

| Degree | D0 minADE | D0 minFDE | D minADE | D minFDE |
|---|---:|---:|---:|---:|
| 0 | 0.161985 | **0.385846** | 0.161373 | 0.388377 |
| 1 | 0.137198 | 0.327826 | 0.137198 | 0.327826 |
| 2–3 | 0.176944 | 0.407173 | 0.176944 | 0.407173 |
| >=4 | 0.186831 | 0.411570 | 0.186831 | 0.411570 |

As required by the intervention, degree-positive marginal results are exact;
all prediction differences are confined to degree-zero agents in E>0
windows and their joint world pairing.

## 9. Coverage, oracle, churn, and two-cycle behavior

Both D and D0 retain exactly 20 unique candidates at Round 0, Round 1, Round
2, and final output. D0 is not a diversity heuristic and does not alter
candidate support size.

| Policy | R0 oracle | R1 oracle | R2 oracle | Final goal oracle | Trajectory minFDE |
|---|---:|---:|---:|---:|---:|
| D | 0.366712 | 0.350198 | 0.350200 | 0.367658 | 0.382742 |
| D0 | 0.366712 | 0.350775 | 0.350778 | 0.368162 | 0.382212 |

| Policy | R0→R1 churn | R1→R2 churn | R0→R2 churn | Two-cycle rate |
|---|---:|---:|---:|---:|
| D | 94.23% | 79.12% | 55.39% | 40.70% |
| D0 | **85.41%** | 79.12% | **46.57%** | 40.70% |

For degree-zero agents specifically, D has 94.40% R0→R1 and R0→R2 churn,
whereas D0 is exactly zero for every transition and has zero two-cycles.
Degree-positive churn and two-cycle statistics are unchanged. Thus D0 removes
only the unsupported isolated-agent permutation and leaves the interacting
refinement dynamics intact.

## 10. Runtime and engineering validation

D0 used 6.793 seconds of sampler time across all 695 forwards (9.77 ms per
window mean) and peaked at 107,065,344 CUDA-allocated and 132,120,576
CUDA-reserved bytes. It is an audit-time measurement, not a production
runtime claim.

Dedicated tests cover graph components and isolates, degree-zero tensor-exact
identity, degree-positive equality, no refinement-generator consumption,
synchronous agent-local update, common permutation of previous IDs, mixed and
multi-component graphs, component matching, exact injective optimality, and a
CUDA/BF16 path. Policy-D exact reproduction and diffusion/global RNG pairing
were additionally enforced by the full-data runtime gates.

Final repository checks:

```text
python -m compileall -q .  -> PASS
pytest                    -> 307 passed, 12 warnings
git diff --check          -> PASS
```

The CUDA/BF16 test ran on the real GPU; it was not skipped. No production
source file changed.

## 11. Answers to the required questions

1. **Do degree-zero agents have slot-invariant conditional rows?** Yes. Exact
   equality and `1e-6` near-equality are both 100% in both rounds; maximum row
   difference is exactly zero.
2. **Does current Hungarian perform an information-free reorder for them?**
   Yes. Policy D changes 98.54% of their slot positions under the common
   permutation test despite identical score rows and unchanged support.
3. **How much of the approximately 44% pathology is attributable to windows
   containing degree-zero agents?** They contain 73.29% of Policy-D Round-1
   failed trials by association. The D0 causal intervention recovers 99 of
   614 failures (16.12%), raising overall exact invariance by 7.17 percentage
   points. The remaining failures show that isolation is important but not
   sufficient.
4. **Does D0 restore whole-scene invariance?** It materially reduces matched
   Hamming, but no: exact invariance is only 62.68% / 62.17%, below 95%.
5. **Does D0 retain Policy-D metrics?** Yes. Every principal metric changes
   by less than 0.6%, 20/20 coverage remains exact, and most errors improve
   slightly; compatibility worsens by 0.56%.
6. **Are degree-positive agents basically pathwise safe after removing
   isolates?** No. Degree-one pathwise invariance is only 60% / 57%, and
   degree-2–3 is 90.92% / 91.96%.
7. **Do disconnected non-singleton components retain relative ambiguity?**
   This ETH split has no E>0 window in that stratum, so it cannot answer that
   counterfactual. Singleton-plus-nonsingleton windows remain unstable, but
   their non-singleton component is already locally unstable.
8. **Are component-local sets more stable than whole-scene sets?** Generally
   yes, especially singletons (100% exact), but size-two components remain
   highly unstable and prevent a cross-component-only explanation.
9. **What is the remaining blocker?** **Within-component assignment
   degeneracy**, with degree-one/size-two components providing the clearest
   evidence. Isolated-agent semantics is a real secondary defect; unsupported
   disconnected-nonsingleton semantics is not the present primary finding.
10. **What is the unique next recommendation?** Conduct a separate design
    review for an explicit lexicographic continuity semantic that resolves
    multiple optimal assignments without changing the conditional score.
    Do not productionize D0 or modify the sampler until that review is
    approved.

## 12. Stop condition

The audit stops here. No production sampler modification, retraining,
temperature/Gumbel change, Sinkhorn, continuity resolver, Stage-A freeze, or
Stage-B execution is authorized or performed.
