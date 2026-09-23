# Exact Persistent Refinement Production Adoption Validation

## Decision

**Final state: `ADOPTION_VALIDATION_PASSED`.**

The default-off inference policy `exact_lexicographic_persistent_tie` passed
the semantic, prediction, RNG-isolation, regression, CUDA/BF16, and performance
gates. It is an **accepted Stage-A deployment candidate** and is only
`eligible_for_stage_a_freeze_review`. This result does not freeze Stage A and
does not authorize Stage B.

No training was run. No model weight, loss, relation module, energy module,
conditional score, diffusion process, or checkpoint architecture was changed.

## Scope and provenance

- Branch: `research/joint-dependency-v2-clean`
- Implementation base commit: `e7b1460836247b46b2e89cbf9f00ab025bb33e57`
- Dataset/split: ETH validation, 139 windows
- Evaluation seeds: 2035, 2036, 2037, 2038, 2039
- Architecture: `strict_no_z`
- Checkpoint epoch: 13
- Checkpoint SHA256: `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`
- Cache manifest SHA256: `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949`
- Frozen legacy source checkpoint SHA256: `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950`
- Configuration: K=21, P=20, M=4, energy rank=8
- Device: NVIDIA GeForce RTX 5070 Ti, CUDA 12.8
- Machine-readable result: `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/exact_persistent_refinement_production_validation/results.json`

Reference artifacts were hash-checked by the runner. Their hashes are stored
in the machine-readable result.

## Production policy

The parser and sampler now accept the explicit, default-off policy:

```text
exact_lexicographic_persistent_tie
```

The existing default remains `categorical`. The new policy is restricted to
Joint Dependency V2 `strict_no_z` inference.

Round 0 is the unchanged weighted Gumbel-Top-P allocation without replacement.
At each of the two synchronous refinement rounds:

- degree-zero agents return their previous IDs tensor-exactly;
- interacting agents run one exact injective assignment solve maximizing the
  strict lexicographic tuple `(J, C_stay, C_geom, R)`;
- `J` is the existing conditional-score objective;
- `C_stay` preserves previous slot/candidate identity when `J` is tied;
- `C_geom` minimizes exact world-metre endpoint displacement when the first
  two objectives are tied;
- `R` is a persistent exchangeable fourth-level priority generated from the
  stable `(evaluation seed, window, agent, namespace)` key.

This is sample-set-level structured allocation of neighbor-conditioned
conditional compatibility scores. It is not claimed to globally maximize
joint compatibility.

### Exactness and one-solve fast path

Canonical finite FP32 scores and world goals are lifted losslessly from their
IEEE-754 bit representation to arbitrary-precision integers. Squared geometry
is then computed exactly. Strict radix bases are derived from upper bounds on
the complete lower-level spans, so no lower objective can override a nonzero
higher-level gap. The rectangular Hungarian solver operates only on Python
integers.

Production performs exactly one assignment solve per interacting agent per
round. It does not run the audit-only uniqueness certification or selected-edge
exclusion re-solves.

`R` uses distinct powers of two on valid edges, so different assignments have
different `R` totals. It is regenerated locally from the same stable key in
both rounds. It does not consume global Python, NumPy, Torch CPU, CUDA, Round-0,
refinement, or diffusion RNG streams.

## Source/reference semantic parity

The approved audit resolver remained an independent reference. The runner
first captured the approved D0 FP32 matrices and then evaluated production and
reference solvers on the exact same immutable matrices. No tolerance was used
for candidate IDs or exact objectives.

| Check | Equal / total | Rate |
|---|---:|---:|
| Round-0 IDs | 695 / 695 | 100% |
| Refinement rounds | 920 / 920 | 100% |
| Interacting assignments | 2910 / 2910 | 100% |
| Exact `(J,C_stay,C_geom)` | 2910 / 2910 | 100% |
| Exact `R` objective | 2910 / 2910 | 100% |
| Degree-zero identity | 300 / 300 | 100% |
| Final joint-world IDs | 695 / 695 | 100% |
| 20/20 candidate coverage | 5050 / 5050 agent-round sets | 100% |
| Historical ambiguous replica-0 cases | 614 / 614 | 100% |

The unique deterministic-triple test varied the `R` payload and retained one
assignment in every trial. The radix proof and one-ULP tests additionally show
that `R` cannot cross a nonzero `J`, `C_stay`, or `C_geom` gap. In the 614 real
exact-equivalence cases, production selected the approved audit replica-0
representative every time.

### Permutation contracts

The source helper passed all five frozen-payload pathwise contract classes:

1. slot permutation;
2. candidate permutation;
3. agent permutation;
4. slot plus candidate permutation;
5. slot plus agent plus candidate permutation.

Inputs and the frozen `R` payload were permuted consistently, and restored
assignments were exact. Thus conditional pathwise equivariance was 100% in the
test matrix. A raw integer seed is not required to generate the same pathwise
payload after relabeling; the stochastic kernel is distributionally
exchangeable, while pathwise equivalence is conditioned on the permuted
payload.

## RNG protocol

Round-0 allocation, persistent tie priority, and diffusion use isolated
streams. For each `(seed, window)`, the evaluator restored the paired
pre-initial state, verified that the source sampler consumed no global RNG,
verified that the unused Round-1/2 Gumbel generators were unchanged, and
restored the paired pre-diffusion state.

All assertions passed:

- global sampler RNG unchanged: 695/695 windows;
- paired pre-diffusion state: 695/695 windows;
- paired window-end state: 695/695 windows;
- unused refinement generators unchanged: 920/920 rounds;
- degree-zero identity events: 300/300;
- exact interacting solver invocations: 2910/2910 expected.

## Baseline reproduction

The original categorical policy reproduced the prior artifact exactly: maximum
absolute error was 0 for all five-seed prediction metrics and the recorded
coverage/oracle diagnostics. In particular, initial unique count was
12.089674, final unique count 11.325000, initial goal oracle 0.498033, final
goal oracle 0.474993, and trajectory minFDE 0.436064.

D0 reproduced within the predeclared 0.002 historical GPU profile tolerance;
the maximum discrepancy was 0.001019 and occurred only in seed 2036. This is
an upstream CUDA FP32 recomputation difference. It was not used to tolerate a
sampler mismatch: all frozen-matrix assignments and objectives above were
checked with exact equality.

The production prediction profile was paired exactly with the approved audit
replica-0 artifact: maximum absolute metric error was 0.

## Five-seed prediction results

All metrics are lower-is-better. Values below are mean ± population standard
deviation across the five evaluation seeds.

| Metric | Production | D0 | Exact-protocol GDTS | vs D0 | vs GDTS |
|---|---:|---:|---:|---:|---:|
| minADE | 0.275516 ± 0.002376 | 0.275674 | 0.286332 | −0.057% | −3.778% |
| minFDE | 0.382648 ± 0.004529 | 0.382212 | 0.394404 | +0.114% | −2.981% |
| JADE | 0.413199 ± 0.005708 | 0.413081 | 0.467982 | +0.029% | −11.706% |
| JFDE | 0.702125 ± 0.007549 | 0.701150 | 0.815469 | +0.139% | −13.899% |
| Joint Goal Endpoint Error | 0.689827 ± 0.007851 | 0.688916 | 0.799670 | +0.132% | −13.736% |
| Compatibility | 0.487309 ± 0.015375 | 0.487555 | 0.674531 | −0.050% | −27.756% |
| Relative Motion Error | 0.301813 ± 0.004563 | 0.302062 | 0.380473 | −0.082% | −20.674% |

The D0 column is from this final rerun; its small difference from older D0
tables is the documented seed-2036 CUDA recomputation drift. Compatibility is
reported but was not part of the ±2% protected-metric band.

### Per-seed production values

| Seed | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.279360 | 0.385788 | 0.421649 | 0.709756 | 0.692777 | 0.478072 | 0.303519 |
| 2036 | 0.274859 | 0.386226 | 0.407272 | 0.696862 | 0.675271 | 0.492697 | 0.304368 |
| 2037 | 0.276781 | 0.383142 | 0.418275 | 0.709532 | 0.697011 | 0.509598 | 0.308078 |
| 2038 | 0.272420 | 0.373862 | 0.410333 | 0.690296 | 0.688420 | 0.492245 | 0.296404 |
| 2039 | 0.274158 | 0.384221 | 0.408468 | 0.704181 | 0.695656 | 0.463932 | 0.296698 |

### E=0 / E>0

| Stratum | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| E=0 | 0.358294 | 0.529271 | 0.358294 | 0.529271 | 0.495247 | 0 | 0 |
| E>0 | 0.263395 | 0.361180 | 0.441249 | 0.790431 | 0.789232 | 0.736260 | 0.456001 |

E=0 agents never invoke social refinement. Their unchanged values provide the
trajectory/diffusion control.

### Agent-count strata

| Agents | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| N=1 | 0.358294 | 0.529271 | 0.358294 | 0.529271 | 0.495247 | 0 | 0 |
| N=2 | 0.301169 | 0.450264 | 0.476081 | 0.917759 | 0.940269 | 0.575085 | 0.395923 |
| N=3–4 | 0.232415 | 0.314473 | 0.383252 | 0.673190 | 0.666574 | 0.668219 | 0.397376 |
| N=5–8 | 0.295401 | 0.393412 | 0.579798 | 1.000992 | 0.986566 | 1.130824 | 0.710867 |
| N≥9 | 0.341679 | 0.481656 | 0.648728 | 1.162069 | 1.171966 | 1.259906 | 0.757443 |

### Degree and component-size strata

These are agent-level marginal reductions. Scene-level joint metrics are not
attributed to individual degree/component bins because doing so would duplicate
one scene score across agents.

| Degree | Agent records | agent minADE | agent minFDE |
|---|---:|---:|---:|
| 0 | 385 | 0.161985 | 0.385846 |
| 1 | 500 | 0.137048 | 0.329472 |
| 2–3 | 510 | 0.176944 | 0.407173 |
| ≥4 | 445 | 0.186606 | 0.411522 |

| Connected-component size | Agent records | agent minADE | agent minFDE |
|---|---:|---:|---:|
| 1 | 385 | 0.161985 | 0.385846 |
| 2 | 500 | 0.137048 | 0.329472 |
| 3–4 | 450 | 0.180611 | 0.409268 |
| ≥5 | 505 | 0.182189 | 0.409138 |

## Coverage, oracle, churn, and two-cycle

| Stage | Agent records | Mean unique candidates | Mean goal oracle |
|---|---:|---:|---:|
| Round 0 | 1840 | 20.000 | 0.366712 |
| Round 1 (E>0) | 1605 | 20.000 | 0.350775 |
| Round 2 (E>0) | 1605 | 20.000 | 0.350778 |
| Final | 1840 | 20.000 | 0.368162 |

The overall frozen candidate-bank oracle was 0.361863 and final trajectory
minFDE was 0.382648, giving a diffusion gap of +0.014486. For E>0, final goal
oracle was 0.350778 and trajectory minFDE 0.361180.

Among interacting-agent records:

- Round0→Round1 slot churn: 85.19%;
- Round1→Round2 slot churn: 78.82%;
- Round0→Round2 slot churn: 46.36%;
- two-cycle rate: 40.66%.

Degree-zero churn and two-cycle rates were exactly zero. The 40.66% synchronous
two-cycle remains at the previously observed level; production integration did
not amplify it. Per the task boundary, no attempt was made to change it.

## Performance

The benchmark contains 695 complete window forwards (139 windows × 5 seeds).

| Backend | Exact solver total | Exact solve mean | Sampler total | Sampler ms/window | Full total | Full ms/window |
|---|---:|---:|---:|---:|---:|---:|
| D0 | N/A | N/A | 7.550 s | 10.863 | 438.349 s | 630.718 |
| Audit reference | 72.776 s | 25.009 ms | 79.676 s | 114.642 | 502.629 s | 723.208 |
| Production one-solve | 13.046 s | 4.483 ms | 19.762 s | 28.434 | 433.487 s | 623.723 |

Relative runtime:

- production sampler vs D0: +161.74%;
- production full forward vs D0: −1.11%;
- production sampler vs audit reference: −75.20%;
- production full forward vs audit reference: −13.76%.

The sampler ratio is higher than D0 as expected, but the registered hard gate
is end-to-end full-forward overhead. The observed −1.11% is timing noise around
parity and comfortably passes the ≤+20% gate. Peak CUDA allocated/reserved was
unchanged across all three paths: 107,064,832 / 132,120,576 bytes.

## Regression and test status

The final formal preflight reported `381 passed`:

```text
python -m compileall -q .   PASS
pytest -q                  381 passed
git diff --check           PASS
```

The dedicated suite covers exhaustive tiny assignments, four-level radix
equivalence, one-ULP priority protection, persistence, R uniqueness, unique-
triple immunity, the 614-case reference semantics, degree-zero identity, all
permutation contracts, invalid masks/capacity, synchronous update, RNG
isolation, one-solve invocation, categorical/CPSR regression, parser/config
compatibility, random FP32 parity, full source-sampler execution, and real
CUDA/BF16 execution.

Consequently:

- default categorical remains tensor-exact and remains the default;
- CPSR V1 is unchanged and reproducible;
- strict-no-z epoch-13 checkpoint loading is unchanged;
- all-off GDTS passthrough remains unchanged;
- training behavior and Stage-A losses are unchanged;
- relation, energy, K/P/M/rank, diffusion, and two-round schedule are unchanged.

## Adoption-gate answers

1. **Production/reference parity:** yes, 100% for all real matrices, IDs, and
   exact four-level objectives.
2. **Unique deterministic triple:** yes; `R` cannot affect the assignment when
   `(J,C_stay,C_geom)` has a unique optimum.
3. **Historical ambiguous cases:** all 614/614 match approved replica 0.
4. **Degree-zero identity:** yes, 300/300 exact events; no solver invocation.
5. **Coverage:** yes, 20/20 in all 5050 observed agent-round sets.
6. **Conditional equivariance:** yes, all slot/candidate/agent and combined
   frozen-payload tests passed exactly.
7. **Prediction profile:** yes, production equals audit replica 0 for every
   reported per-seed metric (paired maximum absolute error 0).
8. **Prediction gates:** yes. All protected metrics are within ±2% of D0;
   minFDE is 2.98% better than GDTS; JADE, JFDE, Endpoint, and Relative Motion
   remain better than GDTS.
9. **Two-cycle:** 40.66%, consistent with the known approximately 40% level and
   not enlarged by integration.
10. **Runtime:** sampler/full-forward changes are +161.74%/−1.11% vs D0 and
    −75.20%/−13.76% vs the audit reference.
11. **Performance gate:** yes; end-to-end overhead is below +20%.
12. **Existing paths:** categorical, CPSR, GDTS, training, checkpoint, and loss
    behavior remain unaffected.
13. **Final state:** `ADOPTION_VALIDATION_PASSED`.
14. **Next step:** only **Stage-A freeze review** is eligible. Stage A is not
    automatically frozen, and Stage B remains prohibited pending review.
