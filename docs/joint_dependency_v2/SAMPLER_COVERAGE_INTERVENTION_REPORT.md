# RSJG Joint Dependency V2 — Sampler Coverage Intervention Report

Date: 2026-09-20  
Dataset/split: ETH validation, 139 synchronized windows  
Scope: inference-only, read-only initial-slot allocation intervention  
Training and Stage B status: **not started**

## 1. Executive conclusion

The experiment isolates a precise two-part failure.

1. Sampling with replacement is the dominant direct cause of the **initial**
   finite-slot coverage loss. Weighted sampling without replacement raises
   mean unique candidates from 12.090 to 20.000, raises frozen-prior rank-1
   coverage from 71.25% to 98.59%, and reduces the initial goal oracle from
   0.498033 to 0.366551, nearly the K=21 bank oracle of 0.361863. It removes
   96.56% of the measured initial finite-slot loss.
2. This gain does not propagate through the complete E>0 pipeline. After the
   unchanged two-round refinement, unique coverage falls back to 12.803 and
   the goal oracle worsens from 0.366551 to 0.461294. The final minFDE is
   0.432770, only 0.76% lower than the iid baseline's 0.436064 and still 9.73%
   worse than same-protocol GDTS.

The E=0 control is decisive: without relation edges or refinement, weighted
without replacement improves minFDE from 0.638617 to 0.532534 (-16.61%). In
E>0 windows, however, minFDE worsens from 0.406407 to 0.418163 (+2.89%). Thus
the initial duplication hypothesis is correct, but eliminating initial
duplication alone is not a sufficient end-to-end repair. The recovered
coverage is primarily lost during refinement in interacting scenes.

This is neither full success nor the predeclared promising partial success:
only 7.91% of the iid-versus-GDTS minFDE gap is recovered, below the 50%
criterion. The unique next recommendation is **D: inspect the two-round
refinement's slot-collapse behavior**. Weighted without replacement should
not yet replace the formal sampler, and Stage B should not start on the basis
of this intervention.

## 2. Experiment scope and invariants

All three policies use the same strict no-z epoch-13 checkpoint and differ
only in the first candidate-allocation call of each validation window.

Unchanged components include the cached K=21 goal bank, candidate logits and
temperature, strict no-z relation model, pair energy, two synchronous
refinement rounds, P=20 world slots, frozen GDTS diffusion, coordinate
conversion, metric code, validation split, and five evaluation seeds. No
optimizer was constructed and no parameter was updated.

The intervention is implemented only in
`tools/sampler_coverage_intervention.py`. It temporarily intercepts the
low-level categorical primitive inside a context manager and restores it on
exit. No `src/` file, training path, checkpoint, default inference behavior,
relation/energy module, loss, or diffusion implementation was modified.

## 3. Provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Strict no-z training source commit | `84bab030baed3044826b9b0d89a783a86e324978` |
| Audit implementation/execution commit | `383e7193645e6c536cc35b4298f63fbf4ca1a31c` |
| Architecture | `strict_no_z` |
| Checkpoint epoch | 13 |
| Checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Legacy GDTS source SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Cache manifest SHA256 | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| K / P / M / energy rank | 21 / 20 / 4 / 8 |
| Split / windows | ETH validation / 139 |
| Evaluation seeds | 2035, 2036, 2037, 2038, 2039 |

Machine-readable artifacts:

- `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/sampler_coverage_intervention/results.json`
  (SHA256 `d64670ac83e62fc17b658819c4505c423732c4a33dab308e0bb04becb0190abd`);
- `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/sampler_coverage_intervention/per_agent_results.jsonl`
  (SHA256 `91c70350642d241e4997884c7dabe9c6151479c12f1b60c55893325b3f53f9fe`).

The latter contains 5,520 rows: 368 validation-agent observations x five
seeds x three policies. No NaN or Inf was found.

## 4. The three initial allocation policies

### 4.1 `iid_with_replacement`

This calls the unchanged production categorical primitive and is the exact
deployed strict no-z policy. Repeated candidate IDs are allowed.

### 4.2 `weighted_without_replacement`

For each agent, the audit uses the same masked, temperature-scaled initial
logits and adds independent Gumbel noise from an explicit per-window
generator. Top-P over the perturbed K scores implements a Plackett-Luce style
weighted draw without replacement. It asserts P<=valid K and P unique valid
IDs; there is no uniform or silent fallback.

### 4.3 `deterministic_top_p`

This takes Top-P from the same masked, temperature-scaled logits without
noise. It is a coverage diagnostic, not a proposed deployed policy. Since
P=20 and K=21, it excludes only the lowest trained-unary candidate.

Only the first categorical call is changed. Both refinement calls remain the
unchanged production categorical-with-replacement implementation.

## 5. Exact downstream RNG control

For every `(seed, window)` Policy A captures a complete RNG snapshot at four
boundaries: before initial allocation, before refinement, before diffusion,
and at window end. A snapshot includes Python, NumPy, CPU Torch, all CUDA
device states, and cuDNN deterministic/benchmark flags.

Policies B/C start from A's pre-initial state. B draws Gumbel noise from an
independent explicit CUDA generator; C consumes no allocation randomness.
Immediately afterward, both restore A's exact pre-refinement state. After the
unchanged refinement calls, they restore A's exact pre-diffusion state. The
window-end state must then equal A's state exactly.

For each intervention policy, all three boundary checks passed for 695
windows (`5 seeds x 139 windows`):

| Policy | pre-refinement | pre-diffusion | window-end |
|---|---:|---:|---:|
| weighted without replacement | 695/695 | 695/695 | 695/695 |
| deterministic Top-P | 695/695 | 695/695 | 695/695 |

Consequently, metric differences are not caused by shifted downstream random
streams.

## 6. Baseline reproduction gate

Policy A was required to pass before B/C were evaluated. It reproduced all
35 per-seed prediction metrics and eight candidate diagnostics from the
protected strict no-z audit. All 43 absolute errors were exactly `0.0` under
a tolerance of `1e-10`.

| Quantity | Protected reference | Policy A |
|---|---:|---:|
| minADE | 0.285284 | 0.285284 |
| minFDE | 0.436064 | 0.436064 |
| JADE | 0.417186 | 0.417186 |
| JFDE | 0.729035 | 0.729035 |
| Goal endpoint | 0.739033 | 0.739033 |
| Compatibility | 0.495512 | 0.495512 |
| Relative motion | 0.301660 | 0.301660 |
| Initial unique candidates | 12.089674 | 12.089674 |
| Initial goal oracle | 0.498033 | 0.498033 |

This gate excludes a protocol, metric, checkpoint, or RNG mismatch as an
explanation for the intervention results.

## 7. Main comparison

All metrics are lower-is-better. Prediction metrics are five-seed means.

| Policy | unique initial | frozen rank-1 | initial oracle | final goal oracle | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iid with replacement | 12.090 | 71.25% | 0.498033 | 0.474993 | 0.436064 | 0.417186 | 0.729035 | 0.739033 | 0.495512 | 0.301660 |
| weighted without replacement | **20.000** | 98.59% | **0.366551** | **0.461294** | **0.432770** | 0.412915 | 0.704535 | 0.693684 | 0.498550 | 0.304229 |
| deterministic Top-P | **20.000** | **100.00%** | 0.367717 | 0.463272 | 0.436136 | **0.410185** | **0.696065** | **0.684369** | **0.493075** | **0.301091** |

Weighted without replacement almost eliminates the initial coverage error,
but its final minFDE improvement is only 0.003294. Deterministic Top-P has
equally complete initial coverage yet essentially the same minFDE as iid.

## 8. Frozen-prior and trained-unary coverage

The two rank systems are reported separately.

| Policy | Rank family | top-1 | top-3 | top-5 | top-10 |
|---|---|---:|---:|---:|---:|
| iid | frozen prior, initial | 71.25% | 98.15% | 99.84% | 100.00% |
| iid | trained unary, initial | 80.92% | 98.86% | 99.89% | 100.00% |
| weighted no-replacement | frozen prior, initial | 98.59% | 100.00% | 100.00% | 100.00% |
| weighted no-replacement | trained unary, initial | 98.97% | 100.00% | 100.00% | 100.00% |
| deterministic Top-P | frozen prior, initial | 100.00% | 100.00% | 100.00% | 100.00% |
| deterministic Top-P | trained unary, initial | 100.00% | 100.00% | 100.00% | 100.00% |

After refinement, B's frozen-prior top-1 coverage is only 77.07% and its
trained-unary top-1 coverage is 67.77%. C is similar at 77.28% and 68.21%.
Thus the almost complete initial top-rank coverage does not survive the
existing refinement.

The GT-oracle candidate has mean frozen-prior rank 8.133 and mean
trained-unary rank 9.715. This shows imperfect unary ordering, but it is not
the immediate cause of the B/C failure: with P=20, both coverage policies
already retain almost every candidate and bring the initial oracle within
0.0047--0.0059 of the full-bank oracle.

## 9. Refinement behavior

| Policy | initial unique | final unique | changed slots | refinement frozen-rank delta | refinement unary-rank delta |
|---|---:|---:|---:|---:|---:|
| iid | 12.090 | 11.325 | 81.13% | -0.810 | +0.510 |
| weighted no-replacement | 20.000 | 12.803 | 80.96% | -1.270 | -0.769 |
| deterministic Top-P | 20.000 | 12.840 | 80.36% | -1.086 | -0.595 |

Negative rank delta means movement toward a higher-probability rank. For B/C,
refinement improves the *mean* rank while sharply reducing the number of
unique candidates. This is consistent with multiple slots coalescing onto
high-probability candidates: average rank improves while geometric oracle
coverage worsens. It does not by itself prove that relation energy is wrong;
it localizes the next audit to refinement's categorical-with-replacement
slot update and world-slot mapping.

## 10. Complete error decomposition

### 10.1 Overall

| Policy | bank oracle | initial oracle | final goal oracle | trajectory minFDE | finite-slot loss | refinement delta | diffusion gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid | 0.361863 | 0.498033 | 0.474993 | 0.436064 | +0.136171 | -0.023040 | -0.038929 |
| weighted no-replacement | 0.361863 | 0.366551 | 0.461294 | 0.432770 | **+0.004688** | **+0.094743** | -0.028524 |
| deterministic Top-P | 0.361863 | 0.367717 | 0.463272 | 0.436136 | +0.005854 | +0.095555 | -0.027136 |

The intervention moves the dominant positive error term from initial
allocation to refinement. The diffusion gap remains negative overall and is
not the dominant overall residual.

### 10.2 E=0 control

| Policy | bank | initial | final goal | trajectory | finite-slot | refinement | diffusion |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid | 0.486705 | 0.662007 | 0.662007 | 0.638617 | +0.175302 | 0 | -0.023390 |
| weighted no-replacement | 0.486705 | 0.492113 | 0.492113 | 0.532534 | +0.005408 | 0 | +0.040421 |
| deterministic Top-P | 0.486705 | 0.487627 | 0.487627 | 0.536401 | +0.000921 | 0 | +0.048774 |

E=0 is also N=1 and has no refinement. Here coverage recovery reaches the
trajectory and lowers minFDE by 16.61%. The remaining bank-to-trajectory gap
is now primarily downstream diffusion/trajectory realization, not slot
duplication.

### 10.3 E>0

| Policy | bank | initial | final goal | trajectory | finite-slot | refinement | diffusion |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid | 0.343584 | 0.474025 | 0.447611 | 0.406407 | +0.130441 | -0.026414 | -0.041204 |
| weighted no-replacement | 0.343584 | 0.348167 | 0.456782 | 0.418163 | +0.004583 | **+0.108615** | -0.038619 |
| deterministic Top-P | 0.343584 | 0.350160 | 0.459706 | 0.421456 | +0.006576 | +0.109546 | -0.038250 |

In E>0 windows, refinement erases the recovered coverage and produces a
slightly worse final goal oracle and minFDE than iid. The paired diffusion
gap is nearly unchanged, so different diffusion noise cannot explain this.

## 11. E=0 and E>0 prediction metrics

Five-seed means:

| Policy / stratum | minADE | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid / E=0 | 0.386192 | 0.638617 | 0.386192 | 0.638617 | 0.662258 | 0 | 0 |
| weighted / E=0 | **0.364403** | **0.532534** | **0.364403** | **0.532534** | **0.500359** | 0 | 0 |
| deterministic / E=0 | 0.361154 | 0.536401 | 0.361154 | 0.536401 | 0.495247 | 0 | 0 |
| iid / E>0 | **0.270509** | **0.406407** | **0.433020** | **0.775226** | **0.778254** | **0.748653** | **0.455769** |
| weighted / E>0 | 0.275393 | 0.418163 | 0.437699 | 0.792404 | 0.792448 | 0.753244 | 0.459651 |
| deterministic / E>0 | 0.274736 | 0.421456 | 0.435233 | 0.777633 | 0.780985 | 0.744972 | 0.454909 |

Relative to iid, weighted E>0 minFDE worsens 2.89%, and E>0 JFDE worsens
2.22%. This stratified result prevents the strong E=0 gain from being
mistaken for a general interacting-scene improvement.

## 12. Degree and agent-count strata

The table reports `initial oracle -> final goal oracle -> trajectory minFDE`.

| Stratum | iid | weighted no-replacement | deterministic Top-P |
|---|---|---|---|
| degree=0 | 0.484675 -> 0.476906 -> 0.464549 | **0.357552 -> 0.373205 -> 0.399798** | 0.351854 -> 0.370466 -> 0.402158 |
| degree=1 | 0.421069 -> 0.404537 -> 0.379383 | 0.293864 -> 0.410032 -> 0.383412 | 0.305420 -> 0.411421 -> 0.389230 |
| degree=2-3 | 0.555712 -> 0.505804 -> 0.465100 | 0.394726 -> 0.510400 -> 0.469220 | 0.394147 -> 0.513998 -> 0.476155 |
| degree>=4 | 0.529964 -> 0.517191 -> 0.441830 | 0.423717 -> 0.538827 -> 0.474983 | 0.421146 -> 0.543687 -> 0.472373 |

| Agent count | iid trajectory minFDE | weighted trajectory minFDE | relative change |
|---|---:|---:|---:|
| N=1 | 0.638617 | **0.532534** | -16.61% |
| N=2 | 0.502742 | 0.513143 | +2.07% |
| N=3-4 | 0.367320 | 0.369523 | +0.60% |
| N=5-8 | 0.435250 | 0.464755 | +6.78% |
| N>=9 | 0.450208 | 0.478665 | +6.32% |

Every degree and agent-count bin receives a much better initial oracle under
weighted allocation. The benefit survives mainly at degree zero/N=1 and is
reversed as interaction/refinement complexity rises. This strengthens the
refinement-localization result.

## 13. Five-seed prediction metrics

### 13.1 Mean +/- population standard deviation

| Policy | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| iid | 0.285284 +/- 0.002814 | 0.436064 +/- 0.006308 | 0.417186 +/- 0.010041 | 0.729035 +/- 0.016001 |
| weighted no-replacement | 0.286761 +/- 0.004787 | 0.432770 +/- 0.011473 | 0.412915 +/- 0.002887 | 0.704535 +/- 0.008531 |
| deterministic Top-P | 0.285773 +/- 0.002526 | 0.436136 +/- 0.008817 | 0.410185 +/- 0.004818 | 0.696065 +/- 0.009420 |

| Policy | endpoint | compatibility | relative motion |
|---|---:|---:|---:|
| iid | 0.739033 +/- 0.013453 | 0.495512 +/- 0.018406 | 0.301660 +/- 0.006638 |
| weighted no-replacement | 0.693684 +/- 0.011990 | 0.498550 +/- 0.012808 | 0.304229 +/- 0.003892 |
| deterministic Top-P | 0.684369 +/- 0.003662 | 0.493075 +/- 0.008388 | 0.301091 +/- 0.005497 |

### 13.2 Per-seed values

| Policy | Seed | minADE | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iid | 2035 | 0.283513 | 0.434358 | 0.429169 | 0.743422 | 0.759753 | 0.507630 | 0.309355 |
| iid | 2036 | 0.282575 | 0.443970 | 0.410379 | 0.724685 | 0.730207 | 0.494686 | 0.296464 |
| iid | 2037 | 0.284507 | 0.427095 | 0.401209 | 0.700372 | 0.719804 | 0.470872 | 0.291520 |
| iid | 2038 | 0.285201 | 0.432488 | 0.423102 | 0.743790 | 0.741985 | 0.481567 | 0.304465 |
| iid | 2039 | 0.290623 | 0.442411 | 0.422071 | 0.732904 | 0.743414 | 0.522804 | 0.306498 |
| weighted | 2035 | 0.277644 | 0.413022 | 0.410207 | 0.711320 | 0.703014 | 0.502733 | 0.303111 |
| weighted | 2036 | 0.287931 | 0.445093 | 0.417578 | 0.692262 | 0.684657 | 0.488958 | 0.302966 |
| weighted | 2037 | 0.291128 | 0.437114 | 0.414978 | 0.714596 | 0.709328 | 0.481415 | 0.300462 |
| weighted | 2038 | 0.287035 | 0.427505 | 0.411267 | 0.696979 | 0.676268 | 0.500795 | 0.302844 |
| weighted | 2039 | 0.290070 | 0.441119 | 0.410546 | 0.707515 | 0.695153 | 0.518851 | 0.311764 |
| deterministic | 2035 | 0.282410 | 0.423617 | 0.414073 | 0.691158 | 0.682120 | 0.495487 | 0.305862 |
| deterministic | 2036 | 0.283194 | 0.450659 | 0.406444 | 0.692633 | 0.678942 | 0.498968 | 0.301360 |
| deterministic | 2037 | 0.286767 | 0.432743 | 0.403959 | 0.690288 | 0.689708 | 0.478365 | 0.295479 |
| deterministic | 2038 | 0.287652 | 0.438896 | 0.417079 | 0.714845 | 0.684731 | 0.490180 | 0.294427 |
| deterministic | 2039 | 0.288842 | 0.434764 | 0.409370 | 0.691402 | 0.686342 | 0.502376 | 0.308327 |

These seeds measure deployed-policy sampling variability under one trained
checkpoint; they are not independent training seeds and are not used for a
formal significance claim.

## 14. Relative comparison with GDTS and iid

Negative values mean lower/better.

| Policy vs GDTS | minADE | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid | -0.37% | +10.56% | -10.85% | -10.60% | -7.58% | -26.54% | -20.71% |
| weighted no-replacement | +0.15% | **+9.73%** | -11.77% | -13.60% | -13.25% | -26.09% | -20.04% |
| deterministic Top-P | -0.20% | +10.58% | -12.35% | -14.64% | -14.42% | -26.90% | -20.86% |

| Policy vs iid | minADE | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| weighted no-replacement | +0.52% | **-0.76%** | -1.02% | -3.36% | -6.14% | +0.61% | +0.85% |
| deterministic Top-P | +0.17% | +0.02% | -1.68% | -4.52% | -7.40% | -0.49% | -0.19% |

Weighted allocation preserves the aggregate joint gains and improves JADE,
JFDE, and endpoint. Compatibility and relative motion change by less than 1%.
However, E>0-only JFDE worsens by 2.22%; the aggregate improvement is partly
driven by the large E=0 endpoint recovery. The result therefore does not
justify claiming a uniformly better joint sampler.

The iid minFDE gap above GDTS is 0.041660. Weighted allocation leaves a gap
of 0.038366, recovering only 7.91% of it. This fails both the <=+2% full
success criterion and the >=50% gap-reduction partial-success criterion.

## 15. Marginal-versus-joint trade-off and case classification

The observed pattern is principally **Case C**:

- initial unique coverage and the initial oracle recover almost completely;
- final selected-goal coverage/oracle deteriorates again during refinement;
- final minFDE changes little overall and worsens in E>0;
- aggregate joint metrics remain strong, but the E>0 stratification exposes
  a small coherence/coverage trade-off.

It is not Case A because minFDE remains far from GDTS. It is not Case B:
deterministic Top-P also fully repairs initial coverage but does not restore
final minFDE, so probability calibration/stochastic allocation is not the
next primary bottleneck. It is not Case E because the policies differ greatly
at the initial and E=0 stages even though refinement compresses their final
differences.

## 16. Decision table

| Candidate decision | Evidence | Decision |
|---|---|---|
| A. Formally adopt weighted without replacement | Initial coverage is fixed, but minFDE remains +9.73% vs GDTS and E>0 minFDE worsens | Reject for now |
| B. Design a hybrid allocator | Could become relevant later, but the present full-coverage allocation is already destroyed downstream | Not next |
| C. Inspect unary ranking | GT oracle rank is imperfect, but P=20 already makes initial oracle near-bank | Secondary |
| **D. Inspect refinement** | Reintroduces about 7.2 duplicate slots and adds +0.1086 E>0 oracle error after coverage is fixed | **Unique next recommendation** |
| E. Enter Stage B | Stage-A marginal mechanism remains unresolved | Reject |

The next work should be a read-only refinement decomposition before any new
training or formal sampler modification. It should determine whether slot
coalescence comes from per-round categorical replacement, synchronous
world-slot pairing, or the conditional score passed to refinement. This
report does not authorize such a modification.

## 17. Tests and validation

Added tests cover:

1. exact delegation of iid to the current primitive;
2. uniqueness for P<=K;
3. exact full coverage for P=K;
4. deterministic Top-P repeatability;
5. fixed-generator weighted repeatability;
6. legal differences across seeds;
7. finite/valid/masked IDs and no cross-agent leakage;
8. E=0/single-agent behavior;
9. explicit P>K rejection;
10. full paired downstream RNG restoration;
11. restoration of the default sampler primitive;
12. CUDA RNG round-trip on all devices.

Validation results:

- `python -m compileall -q .`: passed;
- `pytest`: **260 passed**, 12 warnings, CUDA test executed;
- `git diff --check`: passed;
- Policy-A reproduction gate: **43/43 exact**;
- B/C paired RNG boundary checks: **4,170/4,170 exact**;
- audit status: `AUDIT_COMPLETE`.

## 18. Direct answers

**Q1. Is iid-with-replacement proven to be the main direct mechanism of the
strict no-z minFDE erosion?**  It is proven to be the main direct mechanism
of the initial finite-slot erosion: removing replacement eliminates 96.56%
of that term. It is not, by itself, the main sufficient explanation of final
end-to-end minFDE, because unchanged refinement removes most of the gain.

**Q2. Does weighted without replacement improve unique coverage, rank-1
coverage, and initial oracle?**  Yes: 12.090 -> 20 unique, 71.25% -> 98.59%
frozen rank-1 coverage, and 0.498033 -> 0.366551 initial oracle.

**Q3. Do those gains propagate?**  Only weakly overall and not in E>0. The
final goal oracle improves 2.88% and minFDE 0.76% overall, while E>0 final
goal oracle and minFDE both worsen. E=0 minFDE improves 16.61%.

**Q4. Are joint gains preserved?**  At aggregate level, yes: JADE, JFDE, and
endpoint improve, while compatibility and relative motion worsen by less
than 1%. But E>0 JFDE worsens 2.22%, so preservation is not uniform across
interacting scenes.

**Q5. What does deterministic Top-P add?**  It confirms that nearly perfect
initial coverage is insufficient. Its initial oracle is near the bank oracle,
yet final minFDE is essentially unchanged from iid. Since weighted and
deterministic policies behave similarly after refinement, unary calibration
and initial stochasticity are not the primary next target.

**Q6. Where is the remaining marginal error?**  Overall and especially for
E>0, the newly exposed dominant term is refinement-induced slot coalescence.
For E=0 after coverage is repaired, the remaining gap is chiefly trajectory
realization/diffusion. The next global Stage-A investigation should target
refinement because it reverses gains across all interacting degree bins.

**Q7. What is the unique next step?**  **D: inspect refinement.** Do not yet
formally adopt weighted without replacement, redesign a hybrid allocator,
retrain Stage A, or enter Stage B.

