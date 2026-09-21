# CPSR Inference-Only Validation Report

## 1. Scope and decision

This report evaluates Coverage-Preserving Structured Refinement (CPSR) under
the frozen strict-no-z Stage-A checkpoint. The experiment is inference-only:
no model was trained, no learned score was changed, and Stage B was not
started.

The implemented policy is `structured_gumbel_assignment`:

- round 0 uses weighted Gumbel-Top-P sampling without replacement;
- rounds 1 and 2 use unit-Gumbel-perturbed maximum-weight injective
  assignment;
- the existing conditional score and its existing temperature are used
  exactly once;
- the candidate mask is agent-level and slot-invariant, and every agent must
  have at least `P` valid candidates;
- the operation is a **sample-set-level structured allocation of existing
  neighbor-conditioned conditional compatibility scores**. It does not
  globally maximize the full multi-agent joint compatibility objective.

**Decision: CPSR V1 does not pass the production adoption gate.** It fixes
finite-slot coverage and marginal minFDE, but it systematically degrades some
principal joint metrics relative to the original strict-no-z sampler. It is
therefore not adopted, deterministic ASSIGN remains diagnostic-only, and no
temperature tuning, retraining, or Stage-B execution follows this report.

## 2. Provenance and fixed protocol

| Item | Value |
|---|---|
| Repository branch | `research/joint-dependency-v2-clean` |
| Implementation base commit | `cb088dba9d051b27715863a5b7b24033bc2ddf2a` plus the uncommitted CPSR audit diff described below |
| Architecture | `strict_no_z` |
| Checkpoint epoch | 13 |
| Checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Legacy GDTS source checkpoint SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Cache manifest SHA256 | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| Historical refinement audit SHA256 | `08bff37b941113823797377319feb814642c63d0d701b746dfd9cf9a4b2ead3e` |
| Exact-protocol GDTS reference SHA256 | `5f5e4207a920dfd4c34c91c1fb00531f08850487aa781be13f9d3ff8d0df8da1` |
| Dataset/split | ETH validation |
| Windows | 139 |
| Seeds | 2035, 2036, 2037, 2038, 2039 |
| K / P / M / energy rank | 21 / 20 / 4 / 8 |
| Refinement rounds | 2 |
| Existing sampling temperature | 1.0 |
| Device | NVIDIA GeForce RTX 5070 Ti, CUDA |
| Machine-readable result SHA256 | `d3346dfa9c457a2c5672a48a152bf94c1a29c63b2b0007990efa394cee701de6` |

The cache, checkpoint, candidate logits, relation model, pair energy,
conditional-score calculation, refinement round count, frozen GDTS diffusion,
coordinate conversion, and metric implementation were identical across all
three policies.

The exact validation command was:

```bash
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python \
  tools/cpsr_inference_validation.py \
  --config outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/config.yaml \
  --checkpoint outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt \
  --refinement-reference outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/refinement_coalescence/results.json \
  --gdts-reference outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_failure_decomposition/results.json \
  --output outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/cpsr_inference_validation/results.json \
  --device cuda:0
```

## 3. Implementation and invariants

Implementation changes are limited to:

- `src/models/joint_dependency_v2/joint_sampler.py`: low-level weighted
  Gumbel-Top-P, exact injective assignment, CPSR policy dispatch, explicit RNG
  streams, and read-only diagnostic callbacks;
- `src/parser.py`, `src/models/model.py`, and `src/trainer.py`: a default-off
  inference policy option, deterministic evaluation context, and checkpoint /
  evaluation protocol metadata;
- `tools/cpsr_inference_validation.py`: the inference-only comparison runner;
- `tests/test_cpsr_sampler.py`: CPSR feasibility, determinism, invariance, RNG,
  and regression tests.

The default remains `categorical`. That branch still invokes the original
categorical operation and does not create or consume CPSR generators. The
strict-no-z state, Stage-A loss, relation and energy networks, Goal U-Net,
candidate bank, diffusion model, K/P/M/rank, and checkpoint parameters are
unchanged. The new inference policy has no learned parameters.

CPSR V1 rejects all of the following explicitly:

- `P > K`;
- fewer than `P` valid candidates for any agent;
- slot-varying candidate masks;
- non-finite scores;
- use outside `strict_no_z` through the parser/model contract.

## 4. RNG control and reproducibility

The audit isolates four random streams per `(evaluation seed, window)`:

1. initial allocation;
2. round-1 refinement;
3. round-2 refinement;
4. diffusion.

The original categorical policy captures the complete Python, NumPy, CPU
Torch, and all-device CUDA RNG states at the initial, refinement, diffusion,
and window-end boundaries. Deterministic ASSIGN restores the corresponding
historical states. CPSR uses explicit derived `torch.Generator` instances for
initial allocation and each refinement round, consumes no global RNG, and
restores the same categorical pre-diffusion state before trajectory sampling.

Automated checks passed for every CPSR window:

| Check | Passed / expected |
|---|---:|
| Pre-diffusion RNG paired | 695 / 695 |
| CPSR left global RNG unchanged | 695 / 695 |
| Window-end RNG paired | 695 / 695 |

Thus policy differences below are not explained by a shifted diffusion random
stream.

## 5. Reproduction gates

The original categorical policy was evaluated first. All five-seed metrics,
round-0 unique count and oracle, final unique count and oracle, and trajectory
minFDE matched the accepted refinement audit with maximum absolute error
`0.0` at tolerance `1e-10`. The baseline reproduction gate therefore passed.

Deterministic ASSIGN was independently reproduced before interpreting CPSR.
Its corresponding 40 comparisons also had maximum absolute error `0.0`.

## 6. Five-seed prediction results

All metrics are lower-is-better. Values are population mean ± population
standard deviation over five evaluation seeds.

| Policy | minADE | minFDE | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original categorical | 0.285284 ± 0.002814 | 0.436064 ± 0.006308 | 0.417186 ± 0.010041 | 0.729035 ± 0.016001 | 0.739033 ± 0.013453 | 0.495512 ± 0.018406 | 0.301660 ± 0.006638 |
| Deterministic ASSIGN diagnostic | **0.277473 ± 0.002062** | **0.383394 ± 0.004226** | **0.415280 ± 0.007885** | **0.699226 ± 0.009338** | **0.682405 ± 0.011575** | **0.481323 ± 0.014784** | **0.298115 ± 0.003290** |
| Stochastic CPSR | 0.278724 ± 0.002867 | 0.384472 ± 0.006717 | 0.429704 ± 0.005661 | 0.741496 ± 0.014299 | 0.728432 ± 0.011750 | 0.557870 ± 0.012553 | 0.326660 ± 0.007576 |

### 6.1 CPSR per-seed values

| Seed | minADE | minFDE | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.281279 | 0.387610 | 0.425023 | 0.734465 | 0.722531 | 0.541326 | 0.313901 |
| 2036 | 0.275440 | 0.394867 | 0.434033 | 0.756934 | 0.726732 | 0.561214 | 0.325654 |
| 2037 | 0.276989 | 0.382400 | 0.436966 | 0.758111 | 0.750990 | 0.563202 | 0.335949 |
| 2038 | 0.276979 | 0.374400 | 0.421613 | 0.720459 | 0.716959 | 0.576698 | 0.332563 |
| 2039 | 0.282934 | 0.383083 | 0.430883 | 0.737508 | 0.724949 | 0.546910 | 0.325232 |

The full original-categorical and ASSIGN per-seed values are retained in
`results.json`.

### 6.2 Relative comparisons

| Metric | CPSR vs exact-protocol GDTS | CPSR vs original categorical | CPSR vs deterministic ASSIGN |
|---|---:|---:|---:|
| minADE | -2.66% | -2.30% | +0.45% |
| minFDE | -2.52% | -11.83% | +0.28% |
| JADE | -8.18% | **+3.00%** | +3.47% |
| JFDE | -9.07% | **+1.71%** | +6.05% |
| Goal endpoint | -8.91% | -1.43% | +6.74% |
| Compatibility | -17.30% | **+12.58%** | +15.90% |
| Relative motion | -14.14% | **+8.29%** | +9.58% |

CPSR completely closes the former +10.56% minFDE degradation and is 2.52%
better than GDTS. However, its joint behavior is materially worse than both
original categorical and deterministic ASSIGN. Compatibility is reported but
is not itself a hard gate; JADE and relative-motion error already fail the
2% principal-joint-metric condition.

## 7. E=0 and E>0 results

### 7.1 E=0 control

| Policy | minADE / JADE | minFDE / JFDE | Goal endpoint |
|---|---:|---:|---:|
| Original categorical | 0.386192 | 0.638617 | 0.662258 |
| Deterministic ASSIGN | 0.364403 | 0.532534 | 0.500359 |
| Stochastic CPSR | **0.358294** | **0.529271** | **0.495247** |

There is no relation edge or refinement in E=0 windows. This control confirms
that round-0 without-replacement coverage is beneficial and that the paired
diffusion path is functioning.

### 7.2 E>0 interacting scenes

| Policy | minADE | minFDE | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original categorical | 0.270509 | 0.406407 | **0.433020** | **0.775226** | 0.778254 | 0.748653 | 0.455769 |
| Deterministic ASSIGN | **0.264745** | **0.361557** | 0.441272 | 0.784384 | **0.775407** | **0.727217** | **0.450412** |
| Stochastic CPSR | 0.267074 | 0.363271 | 0.466185 | 0.849914 | 0.847560 | 0.842869 | 0.493540 |

The adoption failure is localized to interacting scenes. Relative to original
categorical, CPSR improves E>0 minFDE by 10.61%, but worsens JADE by 7.66%,
JFDE by 9.63%, endpoint by 8.91%, compatibility by 12.59%, and relative motion
by 8.29%. Aggregate metrics hide part of this effect because E=0 improves
strongly.

## 8. Round-wise coverage and geometric error

### 8.1 Unique candidate count

| Policy | Round 0 | Round 1 (E>0) | Round 2 (E>0) | Final overall |
|---|---:|---:|---:|---:|
| Original categorical | 12.090 | 11.508 | 11.271 | 11.325 |
| Deterministic ASSIGN | 20.000 | 20.000 | 20.000 | 20.000 |
| Stochastic CPSR | 20.000 | 20.000 | 20.000 | 20.000 |

CPSR satisfies injectivity exactly. Every agent keeps 20 unique candidates at
every applicable round; no refinement coalescence remains.

### 8.2 Overall error decomposition

| Policy | Bank oracle | Round-0 oracle | Final goal oracle | Trajectory minFDE | Diffusion gap |
|---|---:|---:|---:|---:|---:|
| Original categorical | 0.361863 | 0.498033 | 0.474993 | 0.436064 | -0.038929 |
| Deterministic ASSIGN | 0.361863 | 0.366551 | 0.368323 | 0.383394 | +0.015071 |
| Stochastic CPSR | 0.361863 | 0.366712 | **0.367388** | 0.384472 | +0.017084 |

For E>0 CPSR evolves from round-0 oracle `0.349116`, to round-1
`0.348986`, to round-2/final `0.349891`; round-1 delta is `-0.000130` and
round-2 delta is `+0.000905`. It therefore preserves marginal geometric
coverage through refinement. In E=0 its bank, round-0/final, trajectory values
are `0.486705`, `0.486890`, and `0.529271` respectively. In E>0 they are
`0.343584`, `0.349116`, `0.349891`, and `0.363271`.

Coverage restoration clearly propagates to selected-goal oracle and minFDE.
It does not, by itself, preserve social joint coherence.

## 9. Slot churn, excluded candidates, and assignment behavior

### 9.1 Slot churn and two-cycles on E>0 agents

| Policy | R0→R1 churn | R1→R2 churn | R0→R2 churn | Two-cycle rate |
|---|---:|---:|---:|---:|
| Original categorical | 94.26% | 92.31% | 93.01% | 6.45% |
| Deterministic ASSIGN | 93.92% | 78.88% | 55.95% | 39.99% |
| Stochastic CPSR | **94.62%** | **93.45%** | **91.82%** | 7.47% |

CPSR avoids collisions but does not stabilize world-slot identity. Almost all
slots are reassigned in both rounds, and only 7.47% return to their round-0
candidate through a two-cycle. Deterministic ASSIGN has the same injective
constraint but much lower R0→R2 churn and substantially better joint metrics.
This is direct evidence that coverage alone is not the source of the CPSR /
ASSIGN gap.

### 9.2 Excluded-candidate diagnostics

Since `P=20` and `K=21`, CPSR excludes exactly one candidate per agent per
round.

| Round | Frozen-prior rank | Unary rank | Conditional probability | GT-oracle exclusion rate |
|---|---:|---:|---:|---:|
| 0 | 15.769 | 17.627 | 0.020811 | 2.77% |
| 1 | 15.196 | 15.069 | 0.022961 | 2.93% |
| 2 | 15.044 | 14.633 | 0.023231 | 2.55% |

At round 2, the excluded candidate has mean frozen-prior probability
`0.030097`, unary probability `0.033761`, conditional score `1.606569`, and
maximum conditional probability across slots `0.058905`. The exclusion is
usually low-ranked, and the low GT-oracle exclusion rate is consistent with
the preserved geometric oracle.

### 9.3 Conditional-score allocation

| Statistic | Round 1 | Round 2 |
|---|---:|---:|
| Conditional entropy | 2.6172 | 2.5687 |
| Mean max probability | 0.1621 | 0.1757 |
| Unique unperturbed argmax candidates | 6.900 | 7.118 |
| Unique assigned candidates | 20.000 | 20.000 |
| CPSR assigned unperturbed score | 3.7870 | 4.0679 |
| Independent MAP unperturbed score | 4.6638 | 4.9476 |
| Mean assignment score regret | 0.8768 | 0.8796 |
| Pairwise JS divergence | 0.1151 | 0.1346 |

The unperturbed score matrices remain concentrated around only about seven
argmax candidates, while CPSR deliberately spreads assignments over all 20.
The unit-Gumbel perturbation makes the stochastic assignment pay an average
unperturbed score regret of about `0.88` per slot. Deterministic ASSIGN obtains
mean selected unperturbed scores `4.1672` and `4.6105` in rounds 1 and 2,
respectively, and its prediction metrics are better. This comparison supports
the following inference: CPSR's stochastic perturbation and high round-to-round
reassignment weaken the learned conditional compatibility organization even
though the geometric candidate support is excellent. It does not show a
solver, RNG, or feasibility error.

## 10. Joint-world diversity

CPSR and deterministic ASSIGN both maintain exactly 20 unique joint-world
signatures at rounds 0, 1, and 2, with zero exact duplicate joint worlds.
Mean pairwise joint-world Hamming distance is `2.6475` at round 0 and `3.4891`
at rounds 1 and 2. Therefore the CPSR failure is not hidden exact joint-world
duplication. It is a coherence/continuity issue within a fully diverse finite
set.

## 11. Degree and agent-count strata

The following CPSR table reports agent-level counts pooled across five seeds,
final trajectory minFDE, final selected-goal oracle, and round-0 oracle.

| Degree | Count | minFDE | Final goal oracle | Round-0 oracle |
|---|---:|---:|---:|---:|
| degree=0 | 385 | 0.386139 | 0.352530 | 0.353812 |
| degree=1 | 500 | 0.310334 | 0.295617 | 0.294263 |
| degree=2–3 | 510 | 0.423674 | 0.394724 | 0.396320 |
| degree≥4 | 445 | 0.421403 | 0.429556 | 0.425343 |

| Agent count | Count | minFDE | Final goal oracle | Round-0 oracle |
|---|---:|---:|---:|---:|
| N=1 | 235 | 0.529271 | 0.486890 | 0.486890 |
| N=2 | 200 | 0.432792 | 0.444544 | 0.437500 |
| N=3–4 | 900 | 0.316432 | 0.285552 | 0.287622 |
| N=5–8 | 415 | 0.402175 | 0.407667 | 0.403577 |
| N≥9 | 90 | 0.497776 | 0.516527 | 0.516527 |

CPSR remains injective in all bins. The final-oracle changes are small; the
strong E>0 joint-metric degradation cannot be explained by a specific bin
losing candidate coverage.

## 12. Runtime and memory

| Policy | Sampler total | Sampler mean/window | Full forward total | Full mean/window | Sampler ratio vs categorical | Full ratio vs categorical |
|---|---:|---:|---:|---:|---:|---:|
| Original categorical | 6.163 s | 8.868 ms | 430.962 s | 620.09 ms | 1.000 | 1.000 |
| Deterministic ASSIGN | 7.253 s | 10.436 ms | 430.294 s | 619.13 ms | 1.177 | 0.998 |
| Stochastic CPSR | 6.473 s | 9.314 ms | 409.138 s | 588.69 ms | 1.050 | 0.949 |

CPSR's measured sampler overhead is 5.03%. Peak CUDA allocated/reserved memory
was `107,063,808 / 132,120,576` bytes. The lower end-to-end wall time in this
single sequential benchmark should be treated as run-order/system variance,
not as evidence that CPSR accelerates diffusion.

## 13. Test and engineering validation

The test suite covers:

- exact preservation of the default categorical operation;
- uniqueness for `P<=K` and exact full coverage for `P=K`;
- fixed-generator reproducibility and different-seed variability;
- explicit rejection of `P>K`, invalid masks, slot-varying masks, insufficient
  valid candidates, and non-finite scores;
- tiny brute-force verification of exact maximum-weight assignment;
- temperature application exactly once;
- slot/agent permutation equivariance with coupled noise;
- no cross-agent/scene leakage and valid E=0/single-agent behavior;
- separate initial/round RNG streams and paired diffusion/global RNG;
- diagnostic round callbacks and churn/two-cycle calculation;
- real-CUDA BF16 score/Gumbel/assignment execution;
- unchanged all-off/legacy GDTS and production-default regressions through the
  full repository suite.

Final validation commands:

```bash
python -m compileall -q .
pytest
git diff --check
```

Final result: `292 passed`; compileall and `git diff --check` passed. The CUDA
CPSR test executed on the real GPU.

## 14. Adoption gate

| Gate condition | Result | Evidence |
|---|---|---|
| minFDE degradation vs GDTS ≤ +2% | PASS | CPSR is 2.52% better than GDTS |
| JADE/JFDE/endpoint/relative motion better than GDTS | PASS | improvements of 8.18%, 9.07%, 8.91%, and 14.14% |
| No principal joint metric degrades >2% vs original strict no-z | **FAIL** | JADE +3.00%; relative motion +8.29% |
| Feasibility/RNG/regression tests | PASS | all tests and all 695 paired windows passed |
| Runtime measured and reported | PASS for measurement only | sampler overhead +5.03% |

Overall: **FAIL — CPSR V1 is not proposed for formal adoption.**

## 15. Scientific interpretation and next action

The experiment establishes three points:

1. Injective stochastic allocation solves the slot-coverage failure. Coverage
   remains 20/20 through both refinement rounds, final goal oracle falls from
   `0.474993` to `0.367388`, and minFDE falls from `0.436064` to `0.384472`.
2. This coverage gain propagates to marginal trajectory accuracy. The former
   strict-no-z marginal erosion is removed under the fixed five-seed protocol.
3. The canonical stochastic CPSR policy does not preserve interacting-scene
   joint coherence as well as the original categorical sampler or deterministic
   ASSIGN. The strongest evidence is the E>0 breakdown, 93–95% per-round slot
   churn, about `0.88` unperturbed conditional-score regret per assigned slot,
   and the large CPSR-versus-ASSIGN gap despite identical 20/20 coverage.

The authoritative design anticipated perturb-and-MAP bias and perturbation
sensitivity as residual risks; this validation observes exactly that class of
risk. It would be scientifically invalid to conceal the failure by changing
the temperature, adding a noise scale, modifying the score/loss, retraining,
or deploying deterministic ASSIGN without a separately approved design.

**Unique next recommendation:** stop at the failed adoption gate and request a
separate design review of CPSR stochastic assignment dynamics—specifically the
high cross-round churn and stochastic conditional-score regret—before any
further sampler change. No production sampler change, training, or Stage B is
authorized by these results.

## 16. Artifacts

- Machine-readable results:
  `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/cpsr_inference_validation/results.json`
- This report:
  `docs/joint_dependency_v2/CPSR_INFERENCE_VALIDATION_REPORT.md`

No dense per-scene `[agent,P,K]` score tensors were retained.
