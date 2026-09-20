# RSJG Joint Dependency V2 — Strict No-z Stage-A Ablation Report

Date: 2026-09-20  
Dataset/run: ETH, `jdv2_stage_a_no_z_full_seed2035`  
Decision scope: whether the global categorical scene latent `z` is necessary in Stage A.  
Stage B status: **not started**.

## 1. Executive conclusion

The strict no-z ablation answers the scoped question positively: the useful
Stage-A joint-prediction behavior does not require the global categorical
scene latent used by V1/V2.

- Against exact same-protocol GDTS, strict no-z improves the five-seed means
  for JADE by 10.85%, JFDE by 10.60%, Joint Goal Endpoint Error by 7.58%,
  Joint Goal Compatibility by 26.54%, and Relative Motion Error by 20.71%.
  minADE also improves by 0.37%.
- Against protected V1 epoch 11, strict no-z improves JADE by 1.63%, JFDE by
  2.92%, Joint Goal Endpoint Error by 2.66%, minADE by 1.27%, and minFDE by
  2.24%. Compatibility and Relative Motion Error differ by only +0.54% and
  +0.25%, respectively. With five stochastic evaluation seeds rather than
  independent training seeds, these small differences should be treated as
  practical parity, not a significance claim.
- Against rejected V2 epoch 4, strict no-z is better in all seven reported
  metrics.
- All four relation modes remain used. The full candidate-pair dynamic
  relation entropy is 0.9839 nats (maximum `log(4)=1.3863`), so deleting `z`
  did not force relation-mode collapse.
- The sampler-induced marginal erosion remains. minFDE is still 10.56% worse
  than GDTS. The frozen candidate-bank oracle is 0.361863, but iid allocation
  of 20 slots produces an initial selected-goal oracle of 0.498033. Pair
  refinement and diffusion recover part of this loss; neither is its primary
  source.

Within the current ETH evidence, the global categorical `z` can be removed
from the main architecture. The next component meriting a controlled
experiment is the sampler/sample-slot allocator. No such experiment was run
in this work.

## 2. Frozen implementation and provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Source commit | `84bab030baed3044826b9b0d89a783a86e324978` |
| Architecture variant | `strict_no_z` |
| Legacy GDTS initialization SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Best no-z checkpoint epoch | 13 |
| Best checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Last checkpoint epoch | 25 |
| Last checkpoint SHA256 | `d3546e9d91cac48b21b9b4e4f758b65145fe2b0f65fb36085bb1f6d43eb6b731` |
| Candidate bank / deployed samples | K=21 / P=20 |
| Relation modes / energy rank | M=4 / rank=8 |
| Training seed | 2035 |
| Validation selection seed | 2035 |

The checkpoint architecture metadata records `scene_modes=0`,
`latent_objective=strict_no_z`, and `architecture_variant=strict_no_z`.
Read-only inspection found no state-dict key containing a scene prior,
posterior head, or scene embedding. The implemented inference graph is

```text
p(r_ij | X, g_i, g_j)
```

with no `p(z|X)`, `q(z|X,Y*)`, gamma, scene-mode sampler, scene embedding,
or z-indexed relation/energy branch.

The frozen Goal U-Net, K=21 bank, SocialMotionEncoder, M=4 dynamic relation,
rank-eight joint energy, P=20 stochastic sampler, two synchronous refinement
rounds, GDTS denoiser, cache, split, optimizer, scheduler, precision, and
evaluation implementation were retained.

## 3. Pre-training verification

Before the formal run:

- full test suite: **246 passed**, with no skipped CUDA tests in the
  non-sandbox GPU run;
- `python -m compileall .`: passed;
- `git diff --check`: passed;
- all-off GDTS tensor-exact regression: passed;
- strict no-z permutation, E=0, single-agent, shape, and incompatible-resume
  tests: passed;
- BF16/CUDA and RNG restoration tests: passed;
- a real ETH BF16 forward/backward preflight had 63 trainable gradient
  tensors, all finite, with every frozen baseline tensor unchanged.

No model source changed after the formal run began. The only post-training
executable added is a read-only audit artifact under the output directory; it
does not construct an optimizer or update parameters.

## 4. Training outcome and checkpoint selection

The run started from the same frozen legacy GDTS checkpoint as V1/V2. All
strict no-z modules were freshly initialized; no V1/V2 weight or optimizer
state was resumed.

Training stopped normally at epoch 25 because deterministic validation JFDE
did not improve for 12 consecutive validations. Runtime was 6,913.57 seconds
(1 h 55 min 13.57 s measured across epoch diagnostics; the program log reports
1 h 55 min 56 s wall time). Peak CUDA allocation/reservation was
414,651,904/574,619,648 bytes (395.44/548.00 MiB).

| Epoch | LR | L_PL_post | L_PL_prior | relation KL | L_no_z | minADE | minFDE | JADE | JFDE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000e-4 | 3.02690 | 3.03819 | 0.23415 | 3.03281 | 0.30436 | 0.46438 | 0.45661 | 0.81475 |
| 7 | 9.704e-5 | 2.93718 | 2.97289 | 0.13743 | 2.95652 | 0.28602 | 0.43729 | 0.42281 | 0.74791 |
| **13** | 9.416e-5 | 2.91143 | 2.94114 | 0.05995 | 2.92753 | 0.28351 | 0.43436 | 0.42917 | **0.74342** |
| 16 | 9.276e-5 | 2.90279 | 2.92859 | 0.04260 | 2.91679 | **0.28281** | **0.43284** | 0.42341 | 0.74394 |
| 25 | 8.867e-5 | 2.88457 | 2.90068 | 0.01429 | 2.89321 | 0.30001 | 0.45770 | 0.45399 | 0.79684 |

The selected checkpoint is epoch 13 because JFDE is the frozen primary
criterion. Epoch 16 has slightly better marginal metrics but a worse JFDE and
therefore does not replace the selected checkpoint. Continued train-loss
decrease alongside worsening validation after the plateau supports the early
stopping decision.

## 5. Five-seed evaluation protocol

- split: ETH validation, 139 windows;
- stochastic deployed-policy evaluation, not MAP;
- seeds: `{2035,2036,2037,2038,2039}`;
- P=20 for GDTS, V1, V2, and strict no-z;
- identical metric implementation and world-coordinate conversion;
- Python, NumPy, CPU Torch, and all CUDA RNG streams isolated and restored;
- 47 E=0 and 92 E>0 windows per seed;
- V1/GDTS/V2 references come from the already established exact
  same-protocol audit, not from old legacy test artifacts.

Machine-readable outputs:

- `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_no_z/five_seed_results.json`
  (`cb5bb4f33404047bd1eddb561f79030d52a6c81c54e990ae0c4cfe1b4b67fe60`);
- `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_no_z/detailed_results.json`
  (`16745c154f89711ecb24dd885ce8eca12f4e5f4369d6f6f248ecbeb19f91136e`);
- reference audit `stage_a_failure_decomposition/results.json`
  (`5f5e4207a920dfd4c34c91c1fb00531f08850487aa781be13f9d3ff8d0df8da1`).

## 6. Same-protocol model comparison

Values are five-seed mean ± population standard deviation; lower is better.

| Model | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| GDTS | 0.286332 ± 0.005157 | 0.394404 ± 0.008789 | 0.467982 ± 0.002848 | 0.815469 ± 0.015009 |
| V1 epoch 11 | 0.288946 ± 0.004816 | 0.446046 ± 0.009029 | 0.424094 ± 0.010590 | 0.750989 ± 0.015997 |
| V2 epoch 4 | 0.291849 ± 0.003377 | 0.442809 ± 0.008770 | 0.433161 ± 0.003463 | 0.769828 ± 0.015022 |
| **Strict no-z epoch 13** | **0.285284 ± 0.002814** | **0.436064 ± 0.006308** | **0.417186 ± 0.010041** | **0.729035 ± 0.016001** |

| Model | Joint goal endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|
| GDTS | 0.799670 ± 0.010164 | 0.674531 ± 0.005805 | 0.380473 ± 0.005892 |
| V1 epoch 11 | 0.759258 ± 0.020128 | **0.492833 ± 0.027062** | **0.300921 ± 0.007298** |
| V2 epoch 4 | 0.778261 ± 0.009799 | 0.555609 ± 0.018883 | 0.322814 ± 0.009157 |
| **Strict no-z epoch 13** | **0.739033 ± 0.013453** | 0.495512 ± 0.018406 | 0.301660 ± 0.006638 |

Relative change of strict no-z (negative means lower/better):

| Reference | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| vs GDTS | −0.37% | **+10.56%** | −10.85% | −10.60% | −7.58% | −26.54% | −20.71% |
| vs V1 | −1.27% | −2.24% | −1.63% | −2.92% | −2.66% | +0.54% | +0.25% |
| vs V2 | −2.25% | −1.52% | −3.69% | −5.30% | −5.04% | −10.82% | −6.55% |

The no-z versus GDTS improvement occurs for every evaluation seed in all five
joint metrics. Its minFDE is better than V1 for every seed. The five evaluation
seeds quantify deployed sampling variability only; they do not substitute for
multiple independent training seeds.

## 7. E=0 and E>0 results

| Stratum | minADE | minFDE | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| E=0 | 0.386192 ± 0.016553 | 0.638617 ± 0.035167 | 0.386192 ± 0.016553 | 0.638617 ± 0.035167 | 0.662258 ± 0.029757 | 0 | 0 |
| E>0 | 0.270509 ± 0.004079 | 0.406407 ± 0.006669 | 0.433020 ± 0.008704 | 0.775226 ± 0.015011 | 0.778254 ± 0.011114 | 0.748653 ± 0.027809 | 0.455769 ± 0.010029 |

Every E=0 window is single-agent, so JADE/JFDE equal marginal ADE/FDE and the
pairwise compatibility/relative-motion metrics are structurally zero. The
matched same-protocol GDTS minFDE values from the prior audit are 0.516320 for
E=0 and 0.376554 for E>0. Strict no-z therefore remains 23.69% worse in E=0
and 7.93% worse in E>0. The largest marginal failure persists where relation
and pair energy do not exist.

## 8. Relation health

The following values aggregate all E>0 windows, five seeds, all edges, and
candidate pairs where applicable:

| Distribution | Usage over M=4 | Mean entropy (nats) |
|---|---|---:|
| History-only base | `[0.0564, 0.1547, 0.2311, 0.5578]` | not separately retained |
| Full candidate-pair dynamic | `[0.0294, 0.1521, 0.2971, 0.5214]` | 0.9839 |
| Final selected-pair relation | `[0.0358, 0.1509, 0.2757, 0.5376]` | 0.9792 |

All modes have nonzero usage; no mode exceeds 54% in the dynamic or selected
distribution. Relative to the four-mode maximum entropy of 1.3863 nats, this
is a specialized but non-collapsed relation representation. Removing the
global scene mode did not remove hypothesis-conditioned candidate-pair
variation or reduce relation inference to a one-mode selector.

## 9. Candidate coverage and marginal-error decomposition

Five-seed population: 1,840 agent observations, representing 368 validation
agents under five stochastic repeats.

| Diagnostic | Initial iid slots | After two refinement rounds |
|---|---:|---:|
| Mean unique candidate IDs in P=20 | 12.090 | 11.325 |
| Frozen-prior rank-1 coverage | 71.25% | 75.92% |
| Top-3 coverage | 98.15% | 96.63% |
| Top-5 coverage | 99.84% | 99.84% |
| Mean frozen-prior rank | 9.644 | 8.834 |
| Mean trained-unary rank | 8.342 | 8.852 |
| Fraction of slots changed by refinement | — | 81.13% |

Endpoint-error decomposition:

| Quantity | GDTS | V1 | V2 | Strict no-z |
|---|---:|---:|---:|---:|
| Frozen K=21 bank oracle | 0.361863 | 0.361863 | 0.361863 | 0.361863 |
| Initial P=20 goal oracle | — | 0.499591 | 0.506813 | **0.498033** |
| Final selected-goal oracle | 0.384980 | 0.480680 | 0.478121 | **0.474993** |
| Final trajectory minFDE | 0.394404 | 0.446046 | 0.442809 | **0.436064** |
| Bank → initial finite-slot loss | — | +0.137729 | +0.144950 | **+0.136171** |
| Initial → selected refinement delta | — | −0.018912 | −0.028692 | **−0.023040** |
| Selected goal → trajectory gap | +0.009424 | −0.034634 | −0.035312 | **−0.038929** |

The strict no-z pipeline follows the same mechanism identified for V1/V2:

1. iid sampling with replacement covers only about 12 of 21 candidates and
   introduces +0.136171 m relative to the frozen bank oracle;
2. pair refinement recovers 0.023040 m on average;
3. goal-conditioned diffusion recovers another 0.038929 m;
4. the remaining trajectory-versus-bank gap is +0.074202 m.

For E=0, initial and final IDs are identical, the refinement delta is exactly
zero, and the finite-slot loss is +0.175302 m. Diffusion recovers 0.023390 m,
yet the final trajectory-versus-bank gap remains +0.151912 m. This is direct
evidence that joint energy, dynamic relation, and scene `z` are not the source
of the dominant marginal erosion.

For E>0, initial sampling loses +0.130441 m, pair refinement recovers
0.026414 m, and diffusion recovers 0.041204 m. The residual
trajectory-versus-bank gap is +0.062824 m.

## 10. Answers to the five acceptance questions

### 10.1 Does strict no-z retain the joint-metric gain over GDTS?

**Yes.** All five joint metrics improve in the five-seed mean, and each of
those improvements is present for every fixed evaluation seed. JFDE improves
10.60%, JADE 10.85%, endpoint error 7.58%, compatibility 26.54%, and relative
motion 20.71%.

### 10.2 Is strict no-z statistically/practically comparable to or better than V1?

**Practically, yes; statistically, this experiment is not a multi-training-seed
test.** Strict no-z improves the primary JFDE, JADE, endpoint error, minADE,
and minFDE means, including a minFDE improvement in all five stochastic runs.
Compatibility and relative motion are within 0.54% and 0.25% of V1. The five
fixed seeds measure sampler variance around one trained checkpoint; formal
training-seed significance would require separately authorized retraining.

### 10.3 Can the global categorical `z` be removed from the main architecture?

**Yes, for the architecture decision supported by this ETH experiment.** A
true construction-time deletion learns equal or better core joint metrics
than the collapsed single-branch V1/V2 models, while retaining all four
relation modes. This agrees with the intervention audit: V1/V2 derived their
useful behavior from one specialized branch, not scene-dependent routing.
The evidence should not be overgeneralized to every dataset before later
cross-dataset or training-seed confirmation.

### 10.4 Does the same sampler-induced marginal erosion remain?

**Yes.** minFDE remains 10.56% worse than GDTS; it is 23.69% worse in E=0,
where no relation or energy operation can cause the degradation. The initial
slot allocator creates +0.136171 m of bank-oracle loss, while refinement and
diffusion are net recovery mechanisms.

### 10.5 Should the next experiment target the sampler?

**Yes.** The minimum next scientific experiment should isolate initial slot
coverage using this frozen strict no-z checkpoint: compare the unchanged iid
with-replacement allocation against one deterministic/reproducible
coverage-preserving allocation under the same P=20 budget, while keeping
unary scores, two-round energy refinement, relation model, diffusion,
checkpoint, seeds, and metrics fixed. A read-only inference intervention
should precede any retraining. This report does not authorize or implement
that intervention.

## 11. Risks and decision boundary

- The conclusion about removing `z` is strong for the controlled ETH
  experiment but is based on one training seed. The five evaluation seeds are
  not independent model fits.
- minFDE remains outside the desired 1–2% marginal-degradation line despite
  being better than V1/V2. Joint-metric gains do not waive this risk.
- Relation mode 3 carries about 52% of mass, but all modes are active and
  entropy is far from zero; this does not meet a collapse definition.
- Candidate top-3 coverage decreases slightly after refinement even as the
  geometric goal oracle improves. Rank coverage and geometric coverage must
  continue to be reported separately.
- No claim is made that strict no-z is an exact joint likelihood; Stage A
  remains a sparse pseudo/composite-likelihood model with stochastic
  structured sampling.

Final engineering status: **strict no-z Stage A is healthy and answers the
scene-latent necessity question; Stage B remains blocked pending review of
this report and the unresolved sampler-driven marginal erosion.**
