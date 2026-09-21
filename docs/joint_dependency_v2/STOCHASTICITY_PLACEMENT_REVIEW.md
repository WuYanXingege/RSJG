# Paired Stochasticity Placement Review

## 1. Scope and decision

This is the Phase-A inference-only audit requested by
`PAIRED_STOCHASTICITY_PLACEMENT_REVIEW`. It uses the protected strict-no-z
epoch-13 checkpoint and changes no learned model, score, loss, or checkpoint.
No training or Stage B was started.

The experiment generated one production-style weighted Gumbel-Top-P Round-0
candidate set for every `(seed, validation window)` and compared:

- **Policy G:** the shared Round 0 followed by fresh unit-Gumbel perturbed
  injective assignment in rounds 1 and 2;
- **Policy D:** the identical Round 0 followed by deterministic
  maximum-weight injective assignment in rounds 1 and 2.

The paired evidence confirms that fresh refinement-stage Gumbel perturbations
cause the observed CPSR V1 joint-metric degradation. Policy D is materially
better than Policy G on all E>0 joint-coherence metrics while retaining 20/20
coverage and strong marginal accuracy.

However, deterministic Hungarian refinement fails the required real-data
slot/tie audit. A common permutation of semantically unlabeled slots changes
the final unordered joint-world set in about 44% of tested real ETH matrices,
and degree-zero agents are never pathwise invariant in the tested tied cases.

**Phase-A gate result: FAILED. Phase B was not entered.** The proposed
`structured_deterministic_refinement` production policy was not added. The
default categorical path and CPSR V1 production-reproduction path remain
unchanged.

## 2. Repository and experiment provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Source HEAD at audit start | `0a6e1df7da66bb521da9cc733f09e58d7237abe5` |
| Architecture | `strict_no_z` |
| Checkpoint epoch | 13 |
| Checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Cache manifest SHA256 | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| Legacy GDTS source SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| CPSR reference SHA256 | `d3346dfa9c457a2c5672a48a152bf94c1a29c63b2b0007990efa394cee701de6` |
| Exact-protocol GDTS reference SHA256 | `5f5e4207a920dfd4c34c91c1fb00531f08850487aa781be13f9d3ff8d0df8da1` |
| Split / windows | ETH validation / 139 |
| Seeds | 2035, 2036, 2037, 2038, 2039 |
| K / P / M / rank | 21 / 20 / 4 / 8 |
| Refinement rounds | 2 synchronous rounds |
| Existing temperature | 1.0 |
| Device | NVIDIA GeForce RTX 5070 Ti |
| Result SHA256 | `9e0a1a08d6bb0d8be6b3433ccbe5aa51a341da203753f3a4d2f2d100a1e19075` |

Exact command:

```bash
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python \
  tools/stochasticity_placement_review.py \
  --config outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/config.yaml \
  --checkpoint outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt \
  --cpsr-reference outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/cpsr_inference_validation/results.json \
  --gdts-reference outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_failure_decomposition/results.json \
  --output outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stochasticity_placement_review/results.json \
  --device cuda:0
```

## 3. Source-fact review

### 3.1 Current production CPSR V1

The current `structured_gumbel_assignment` source implements:

```text
Round 0: weighted_gumbel_top_p(unary / T + unit Gumbel)
Round 1/2: maximum-weight injective assignment(
               conditional_score / T + fresh unit Gumbel)
```

It derives independent initial/round streams through
`make_sampling_generators`. Stream zero uses the stable constant `0x5A17`.

### 3.2 Historical ASSIGN audit

The historical audit implemented:

```text
Round 0: weighted_without_replacement with _initial_generator
Round 1/2: deterministic coverage_assignment on unchanged scores
```

Its initial generator used `+17`, not `0x5A17`. Consequently, the earlier
CPSR-versus-ASSIGN table compared different deterministic seed derivations at
Round 0 and was not a strictly paired Round-0 intervention.

### 3.3 Historical ASSIGN was never a fully deterministic deployment path

Its inference path was always stochastic:

```text
C0 ~ q_init(. | X, evaluation seed)
C1 = R_det(C0, X)
C2 = R_det(C1, X)
Y  ~ frozen GDTS diffusion(. | C2, X, evaluation seed)
```

Only refinement was deterministic. Randomness remained in the sampled
Round-0 finite hypothesis set and in trajectory diffusion.

## 4. Exact pairing protocol

For each of 695 `(seed, window)` evaluations, both policies:

1. restored the same complete pre-initial Python, NumPy, CPU Torch, and
   all-device CUDA RNG state;
2. constructed the same production initial generator;
3. independently regenerated Round-0 IDs and asserted tensor-exact equality;
4. used the same candidate bank, unary, relation, energy, mask, conditional
   score, temperature, and synchronous two-round update;
5. restored the same categorical-control pre-diffusion RNG state;
6. required the same final window RNG state.

Policy D additionally asserted that the provided round-1 and round-2
generators were not consumed.

| Assertion | Passed |
|---|---:|
| Exact shared Round-0 IDs | 695 / 695 |
| Policy D round generators unchanged | 920 / 920 E>0 round calls |
| Policy G global RNG unchanged | 695 / 695 |
| Policy D global RNG unchanged | 695 / 695 |
| Paired pre-diffusion RNG, each policy | 695 / 695 |
| Paired window-end RNG, each policy | 695 / 695 |

The original categorical baseline reproduced the accepted result with maximum
absolute error `0.0`. Policy G reproduced CPSR V1 with maximum absolute error
`0.0`. The causal comparison therefore changes only refinement Gumbel
placement.

## 5. Strictly paired prediction metrics

All values are lower-is-better five-seed mean ± population standard deviation.

| Policy | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original categorical reference | 0.285284 ± 0.002814 | 0.436064 ± 0.006308 | 0.417186 ± 0.010041 | 0.729035 ± 0.016001 | 0.739033 ± 0.013453 | 0.495512 ± 0.018406 | 0.301660 ± 0.006638 |
| Policy G: shared R0 + fresh Gumbel | 0.278724 ± 0.002867 | 0.384472 ± 0.006717 | 0.429704 ± 0.005661 | 0.741496 ± 0.014299 | 0.728432 ± 0.011750 | 0.557870 ± 0.012553 | 0.326660 ± 0.007576 |
| Policy D: shared R0 + deterministic refinement | **0.275460 ± 0.002460** | **0.382742 ± 0.004980** | **0.413168 ± 0.006368** | **0.701790 ± 0.006836** | **0.690064 ± 0.009856** | **0.484819 ± 0.013807** | **0.302062 ± 0.003986** |

Policy D relative to the strictly paired Policy G:

| minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|
| -1.17% | -0.45% | **-3.85%** | **-5.35%** | **-5.27%** | **-13.09%** | **-7.53%** |

### 5.1 E=0 control

Both policies are exactly equal because E=0 executes Round 0 and diffusion
but no refinement:

| minADE / JADE | minFDE / JFDE | Endpoint | Compatibility / relative motion |
|---:|---:|---:|---:|
| 0.358294 | 0.529271 | 0.495247 | 0 / 0 |

This equality is a direct control for shared initialization and diffusion.

### 5.2 E>0 causal result

| Policy | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Policy G | 0.267074 | 0.363271 | 0.466185 | 0.849914 | 0.847560 | 0.842869 | 0.493540 |
| Policy D | **0.263332** | **0.361287** | **0.441201** | **0.789924** | **0.789591** | **0.732498** | **0.456377** |
| Relative D vs G | -1.40% | -0.55% | **-5.36%** | **-7.06%** | **-6.84%** | **-13.09%** | **-7.53%** |

Because Round 0 and diffusion noise are paired, this consistently lower E>0
joint error supports attributing the CPSR V1 gap to fresh refinement-stage
perturbation/reassignment. It is an empirical causal intervention on this
checkpoint and split, not a general theoretical proof.

### 5.3 Policy D per-seed results

| Seed | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.279571 | 0.386445 | 0.424049 | 0.708448 | 0.692040 | 0.468026 | 0.304481 |
| 2036 | 0.273827 | 0.387305 | 0.406678 | 0.693925 | 0.672465 | 0.496804 | 0.303541 |
| 2037 | 0.276077 | 0.383969 | 0.416233 | 0.710683 | 0.702857 | 0.491124 | 0.307399 |
| 2038 | 0.272251 | 0.373364 | 0.411020 | 0.694878 | 0.692965 | 0.499670 | 0.297072 |
| 2039 | 0.275576 | 0.382626 | 0.407857 | 0.701014 | 0.689995 | 0.468470 | 0.297817 |

## 6. Coverage, oracle, score, and churn

Both policies retain exactly 20 unique candidates in Round 0, Round 1,
Round 2, and the final set.

| Policy | R0 oracle | R1 oracle E>0 | R2 oracle E>0 | Final oracle overall |
|---|---:|---:|---:|---:|
| Policy G | 0.366712 | **0.348986** | **0.349891** | **0.367388** |
| Policy D | 0.366712 | 0.350198 | 0.350200 | 0.367658 |

The tiny geometric-oracle differences cannot explain the large joint-metric
gap.

| Policy / round | Assigned conditional score | Independent-MAP score | Assignment regret |
|---|---:|---:|---:|
| Policy G / R1 | 3.78698 | 4.66379 | 0.87681 |
| Policy G / R2 | 4.06793 | 4.94757 | 0.87964 |
| Policy D / R1 | **4.16348** | 4.66379 | **0.50031** |
| Policy D / R2 | **4.61469** | 5.08777 | **0.47308** |

| Policy | R0→R1 churn | R1→R2 churn | R0→R2 churn | Two-cycle rate |
|---|---:|---:|---:|---:|
| Policy G | 94.62% | 93.45% | 91.82% | 7.47% |
| Policy D | 94.23% | **79.12%** | **55.39%** | **40.70%** |

Removing fresh round noise reduces unperturbed score regret and prevents the
second round from replacing almost the entire finite support ordering again.
This is consistent with, but does not alone prove, the joint-metric mechanism.

## 7. Cross-seed goal-support audit

Policy D remains a stochastic finite joint-hypothesis support policy because
its production Round 0 remains stochastic.

### 7.1 Per-agent final support

| Statistic | Policy D | Policy G |
|---|---:|---:|
| Mean distinct excluded IDs across five seeds | 1.590 | 3.522 |
| Mean excluded-ID entropy | 0.298 nats | 1.127 nats |
| All five seeds have same excluded ID | 60.87% | 3.80% |
| Mean unordered support Jaccard | 0.9781 | 0.9262 |

Policy D reduces candidate-set variability relative to Policy G but does not
eliminate it: 39.13% of agents exclude more than one candidate across the five
seeds.

### 7.2 Permutation-invariant joint-world-set matching

World sets are matched with a P-by-P Hungarian cost whose entries are
normalized agent-wise Hamming distances. Raw slot-wise agreement is not used.

| Policy/round | Matched mean Hamming | Matched exact-world fraction | Exact whole-set equality |
|---|---:|---:|---:|
| Policy D / Round 0 | 0.4228 | 0.3349 | 11.94% |
| Policy D / final Round 2 | 0.3931 | 0.3395 | 11.94% |
| Policy G / Round 0 | 0.4228 | 0.3349 | 11.94% |
| Policy G / final Round 2 | 0.4189 | 0.3350 | 11.94% |

Deterministic refinement modestly compresses Round-0 cross-seed differences
but leaves substantial final joint-world variation. Goal-level stochasticity
therefore exists before diffusion and is expressed mainly through joint-world
pairing, with smaller but nonzero per-agent support-set variation. It is not
an artifact of diffusion alone.

The outputs are appropriately described as a **stochastic finite joint-
hypothesis support set**, not calibrated iid samples from an exact posterior.

## 8. Deterministic Hungarian permutation and tie audit

### 8.1 Synthetic matrices

| Case | Objective invariant | Pathwise slot-equivariant | Unordered support invariant |
|---|---|---|---|
| Unique optimum | yes | yes | yes |
| Identical rows | yes | **no** | yes |
| Repeated/tied scores | yes | yes for the tested tie | yes |

This matches the mathematical limitation: a unique optimum is pathwise
equivariant under a common row permutation, while identical rows admit
multiple equally optimal assignments and expose solver row-order tie behavior.

### 8.2 Real ETH conditional-score matrices

Three common slot permutations were tested for every E>0 window, seed, and
round: 1,380 real matrices per round.

| Statistic | Round 1 | Round 2 |
|---|---:|---:|
| Mean maximum objective delta | 5.25e-7 | 5.16e-7 |
| Objective delta p90 | 3.81e-6 | 3.81e-6 |
| Per-agent unordered support invariant | 100% | 100% |
| All-agent pathwise assignment invariant | **55.51%** | **56.38%** |
| Unordered joint-world set invariant | **55.51%** | **56.38%** |
| Matched exact-world fraction | 65.95% | 66.05% |
| Matched mean normalized Hamming | 0.1268 | 0.1272 |
| Degree-zero agent pathwise invariant | **0%** | **0%** |

Objective differences are within the declared `1e-5` FP32 accumulation
tolerance and every per-agent support set is invariant, so there is no solver
optimality or feasibility error. The problem is the pairing of candidate IDs
to world slots under multiple optima. Approximately 44% of real score matrices
produce a different unordered set of complete joint worlds after a common
slot permutation, with nonzero matched Hamming. This is a material dependence
on arbitrary slot labels, not merely an internal reordering of the same joint
world set.

Degree-zero agents are the clearest source of tied rows: none of their tested
pathwise assignments is invariant. Since their arbitrarily paired candidate
can be combined with other agents' slot-specific candidates, per-agent support
invariance does not imply joint-world-set invariance.

No slot-index epsilon or heuristic tie breaker was introduced.

## 9. Phase-A gate

| Required condition | Result | Evidence |
|---|---|---|
| Shared-R0 deterministic refinement improves joint coherence | PASS | all five E>0 joint metrics improve by 5.36–13.09% |
| Preserve 20/20 coverage and marginal performance | PASS | 20 unique every round; minFDE 2.96% better than GDTS |
| Seed-dependent final goal support remains | PASS | excluded-ID and permutation-invariant world-set variation remain |
| No material arbitrary slot/tie pathology | **FAIL** | only 55.5–56.4% of real unordered joint-world sets invariant |
| No solver/RNG/feasibility error | PASS | exact support, objective tolerance, and all RNG assertions pass |

Overall: **`PHASE_A_GATE_FAILED_STOPPED_BEFORE_PHASE_B`**.

The explicit materiality rule was fixed before interpretation: fail when a
common slot permutation changes more than 5% of real unordered joint-world
sets and the mean matched normalized Hamming is nonzero. Observed change rates
are about 44%, far beyond this boundary; the conclusion is not threshold-
borderline.

## 10. Counterfactual adoption metrics and why they are insufficient

Policy D would pass the metric-only portion of the prior adoption gate:

- versus exact-protocol GDTS: minADE -3.80%, minFDE -2.96%, JADE -11.71%,
  JFDE -13.94%, endpoint -13.71%, compatibility -28.13%, relative motion
  -20.61%;
- versus original categorical: minFDE -12.23%, JADE -0.96%, JFDE -3.74%,
  endpoint -6.63%, compatibility -2.16%, relative motion +0.13%.

Those metrics do not override the formal tie/permutation requirement. A method
whose complete joint-world set materially depends on arbitrary slot ordering
cannot yet be declared a trustworthy production refinement policy.

## 11. Runtime

| Policy | Sampler total | Sampler mean/window | Full forward total | Full mean/window |
|---|---:|---:|---:|---:|
| Policy G | 6.935 s | 9.978 ms | 440.170 s | 633.34 ms |
| Policy D | 6.354 s | 9.143 ms | 428.528 s | 616.59 ms |

Policy D measured 8.37% less sampler time and 2.64% less end-to-end time than
Policy G in this sequential run. These are audit-run measurements, not a
general speed claim. Both peaked at approximately 107 MB allocated and 132 MB
reserved CUDA memory.

## 12. Direct answers

1. **Was historical ASSIGN's complete policy stochastic?** Yes.
2. **Where did its randomness come from?** Weighted Round-0 hypothesis-set
   sampling and frozen GDTS trajectory diffusion; its two refinement rounds
   were deterministic.
3. **Was the old CPSR-vs-ASSIGN Round-0 comparison strictly paired?** No.
   Production used the `0x5A17` seed tag and the historical audit used `+17`.
4. **With shared Round 0, does fresh refinement Gumbel still degrade joint
   metrics?** Yes. Policy D improves every E>0 joint metric by 5.36–13.09%
   relative to Policy G.
5. **Does deterministic refinement preserve seed-dependent final goal
   support?** Yes. Both excluded candidates and permutation-invariant matched
   joint-world sets vary across seeds.
6. **Where is final stochasticity visible?** Primarily in joint-world pairing,
   with smaller per-agent candidate-support variation, and additionally in
   downstream diffusion. It is not diffusion-only.
7. **Does deterministic Hungarian have an impactful slot/tie pathology?**
   Yes. Objective and per-agent support remain valid, but about 44% of real
   unordered joint-world sets change under a common slot permutation;
   degree-zero tied rows are never pathwise invariant in the tested cases.
8. **Did `structured_deterministic_refinement` pass adoption?** No formal
   policy was implemented because Phase A failed before Phase B. Its metric
   profile is strong, but it fails the prerequisite tie/permutation gate.
9. **Can Stage A now be frozen?** No. The proposed refinement semantics remain
   unresolved at the joint-world pairing level.
10. **Should a separate request to enter Stage B be submitted?** No. First
    resolve and separately review the deterministic tie semantics without
    adding an arbitrary slot-index heuristic.

## 13. Engineering artifacts and status

Added for Phase A only:

- `tools/stochasticity_placement_review.py`;
- `tests/test_stochasticity_placement_review.py`;
- this report;
- the lightweight aggregated `results.json`.

The audit did not persist dense `[scene,N,P,K]` score tensors. Because the
Phase-A gate failed, the following Phase-B artifacts intentionally do not
exist:

- no `structured_deterministic_refinement` production policy;
- no `STRUCTURED_DETERMINISTIC_REFINEMENT_VALIDATION_REPORT.md`;
- no `structured_deterministic_refinement/results.json`.

Final status: **audit complete; no training; no Stage B; production sampler
unchanged**.
