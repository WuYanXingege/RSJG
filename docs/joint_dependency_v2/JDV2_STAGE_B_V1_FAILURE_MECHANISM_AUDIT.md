# JDV2 Stage-B V1 failure-mechanism audit

## Status

**`STAGE_B_V1_NO_CLEAR_GAIN_MECHANISM_IDENTIFIED`**

Mechanism state: **`MIXED_FAILURE_MECHANISM`**.

Stage-B V2 status: **`eligible_for_stage_b_v2_design_review`**.

The mandatory paired identity gate was restored before any result below was
used. This is a read-only audit of the unchanged Stage-A epoch-13 and Stage-B
V1 epoch-10 checkpoints. No optimizer step, training, finetuning, sampler
change, loss change, or V2 implementation occurred.

## 1. Provenance and protocol

- Source base: `b9ad89110c75dfaea3fdbcb1552b7c10563ba952`.
- Stage-A SHA256:
  `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`.
- Stage-B epoch-10 SHA256:
  `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a`.
- ETH validation: 139 windows; seeds 2035–2039.
- Each variant uses identical Stage-A worlds, IDs, `x_T`, branch noise,
  window order, coordinate conversion, and metric implementation.
- Coverage is 20/20 for all 1,840 agent-seed records.
- Endpoint and compatibility are unchanged across residual interventions, as
  required because goals are frozen.

The seed-2035 production gate is exact over all 139 windows: audit FULL versus
production FULL max difference 0; audit NONE versus production Stage A max
difference 0; candidate IDs and goals equal. The identity-fix audit separately
shows exact E=0, mixed degree-zero, active-path and RNG parity.

Machine result:
`outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v1/failure_mechanism_audit/results.json`.

## 2. Corrected paired V1 result

Five-seed mean ± population standard deviation; lower is better.

| Variant | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage A / NONE | 0.278201±0.001376 | 0.386215±0.002358 | 0.413185±0.004708 | 0.701908±0.008592 | 0.689827±0.007851 | 0.487309±0.015375 | 0.297907±0.004499 |
| FULL V1 | 0.282617±0.001970 | 0.391074±0.004110 | 0.416070±0.004487 | 0.703679±0.009700 | 0.689827±0.007851 | 0.487309±0.015375 | 0.297877±0.003840 |
| FULL relative to Stage A | +1.587% | +1.258% | +0.698% | +0.252% | 0 | 0 | -0.010% |

The fixed execution removes E=0 contamination, but V1 still has no clear
trajectory-level gain. All E=0 metrics are now exactly equal between FULL and
Stage A. On E>0, FULL remains worse than Stage A in minADE, minFDE, JADE and
JFDE.

## 3. H1 — component-common residual drift

For every corrector call, each interacting component was decomposed as
`delta = component_mean + centered_delta`. Numerical checks passed:

- reconstruction maximum error: `1.19209e-7`;
- centered component-sum maximum: `3.57628e-7`;
- edge-relative residual difference after centering: `5.96046e-8`.

### Residual energy

| Stratum | mean common fraction | median | p90 |
|---|---:|---:|---:|
| overall | **31.70%** | 29.15% | 58.81% |
| component size 2 | 36.96% | 35.26% | 63.12% |
| component size 3–4 | 24.12% | 20.72% | 49.46% |
| component size ≥5 | 27.38% | 25.02% | 47.44% |
| timestep 30 | 47.62% | 55.60% | 76.13% |
| timestep 25 | 28.83% | 28.11% | 51.73% |
| timestep 20 | 23.80% | 22.35% | 42.35% |
| timestep 15 | 25.81% | 24.77% | 44.00% |
| timestep 10 | 29.97% | 29.25% | 49.76% |
| timestep 5 | 34.18% | 33.65% | 57.24% |

### Causal counterfactuals

| Variant | minADE | minFDE | JADE | JFDE | Relative motion |
|---|---:|---:|---:|---:|---:|
| Stage A | 0.278201 | 0.386215 | 0.413185 | 0.701908 | 0.297907 |
| FULL V1 | 0.282617 | 0.391074 | 0.416070 | 0.703679 | 0.297877 |
| COMMON_ONLY | 0.280666 | 0.388158 | 0.414875 | 0.706200 | 0.297843 |
| COMPONENT_CENTERED | **0.280106** | **0.387595** | 0.414895 | **0.699614** | 0.298333 |
| SCENE_CENTERED | 0.280106 | 0.387595 | 0.414895 | 0.699614 | 0.298333 |

Component centering recovers 56.85% of FULL minADE harm, 71.59% of minFDE
harm, and 40.71% of JADE harm. It moves JFDE from 0.252% worse than Stage A to
0.327% better. Relative motion becomes only 0.143% worse than Stage A.
COMMON_ONLY retains substantial marginal damage and produces even worse JFDE
than FULL, despite containing only 31.7% of residual energy.

This satisfies the predeclared multi-evidence rule. **H1 is supported.**

## 4. H2 — oracle-branch train/deploy mismatch

The training goal-oracle branch is also the frozen Stage-A trajectory-ADE
oracle in only **55.11%** of scene-seed cases. The trajectory oracle's mean
rank under the goal criterion is 1.85 (median 1, p90 3.6, maximum 14).

Branch-level relationships are weak:

- Spearman goal error vs `delta_ADE`: `0.0104`;
- goal error vs `delta_FDE`: `-0.0625`;
- goal rank vs `delta_ADE`: `0.0681`;
- goal rank vs `delta_FDE`: `0.0611`.

Mean `delta_ADE` is 0.00353 at goal rank 1 and 0.00542 at ranks 11–20, but
`delta_FDE` does not follow the same monotone pattern (0.00702 at rank 1,
0.00463 at ranks 11–20).

ORACLE_ONLY is non-deployable and uses GT solely as a mechanism probe:

| Variant | minADE | minFDE | JADE | JFDE | Relative motion |
|---|---:|---:|---:|---:|---:|
| FULL V1 | 0.282617 | 0.391074 | 0.416070 | 0.703679 | 0.297877 |
| ORACLE_ONLY | 0.279509 | 0.387997 | 0.414772 | 0.705538 | 0.297596 |

ORACLE_ONLY is markedly less harmful on marginal metrics but remains worse
than Stage A and is more harmful than FULL on JFDE. This is evidence of a
train/deploy support limitation, but it is not consistent enough across joint
and branch-rank diagnostics to localize the failure primarily to H2.

**H2 is secondary, not the primary supported mechanism.**

## 5. H3 — objective-gradient interaction

The exact approved Stage-B training path was audited on 64 deterministic E>0
training windows at INIT and BEST. No optimizer was constructed or stepped;
all corrector tensors were byte-equal before/after.

| State | `||g_diff||` mean | `||g_rel||` mean | `||0.05g_rel||` mean | weighted/diff mean | cosine mean | conflict rate |
|---|---:|---:|---:|---:|---:|---:|
| INIT | 0.53966 | 0.03792 | 0.001896 | **0.552%** | 0.241 | 1.56% |
| BEST | 2.85829 | 0.57015 | 0.028507 | **1.303%** | 0.608 | 6.25% |

At BEST, the weighted/diff ratio has median 1.015%, p10 0.053%, p90 2.762%.
The mean cosine is positive in every real module group: state encoder 0.615,
time network 0.601, gate 0.584, value 0.607, output 0.589. At INIT, only the
output path has nonzero gradients because the corrector output layer is
zero-initialized; this is expected rather than missing instrumentation.

The relative objective is generally aligned, not adversarial, but after its
0.05 weight its gradient is roughly two orders of magnitude smaller than the
diffusion gradient. Optimization is overwhelmingly diffusion-driven.

**H3 is supported as objective-gradient magnitude imbalance, not gradient
conflict.**

## 6. H4 — relation-embedding sensitivity

| Variant | minADE | minFDE | JADE | JFDE | Relative motion |
|---|---:|---:|---:|---:|---:|
| FULL relation | 0.282617 | 0.391074 | 0.416070 | 0.703679 | 0.297877 |
| ZERO relation | 0.285775 | 0.401940 | 0.419800 | 0.712417 | 0.301696 |
| edge branch-mean | 0.282583 | 0.391535 | 0.416308 | 0.704005 | 0.298069 |
| fixed branch shuffle | 0.282518 | 0.391355 | 0.416335 | 0.704448 | 0.298117 |

ZERO relation materially worsens every trajectory metric, so the corrector
does use relation information. Tensor sensitivity is nontrivial: ZERO has
mean residual RMS difference 0.1054 and mean trajectory absolute difference
0.0446; mean relation has 0.0381/0.00887; branch shuffle has 0.0463/0.01174.

However, branch mean and fixed branch shuffle change every aggregate
trajectory metric by no more than about 0.12% relative to FULL. Thus the
network is relation-conditioned, while the same-slot branch-specific
association has weak aggregate predictive utility.

**`RELATION_CONDITIONING_WEAK` is supported specifically for branch
specificity, not for relation conditioning as a whole.**

## 7. Combined attribution

The required combined state is **`MIXED_FAILURE_MECHANISM`**:

1. `COMMON_MODE_DRIFT_SUPPORTED` is the strongest causal mechanism: a modest
   energy fraction causes disproportionate marginal/JFDE harm, and centering
   directly recovers it.
2. `OBJECTIVE_GRADIENT_MISMATCH_SUPPORTED` is supported as severe magnitude
   imbalance: denoising dominates the weighted relative objective.
3. `RELATION_CONDITIONING_WEAK` applies to branch-specific identity, although
   removing all relation information is clearly harmful.
4. Oracle/all-branch mismatch is plausible but secondary and not independently
   sufficient.

The narrowest scientifically justified Stage-B V2 design review should ask
whether the residual should be component-relative/zero-mean by construction,
thereby removing unidentifiable group translation while preserving the edge
relative residual. This audit does not implement that change and does not
authorize lambda tuning, retraining, or V2 code.

## 8. Required answers

1. Component-common energy: mean 31.70%; centered 68.30%.
2. Removing common residual recovers 56.85% minADE and 71.59% minFDE harm.
3. COMPONENT_CENTERED JADE is 0.414895; JFDE is 0.699614, the latter better
   than Stage A.
4. COMMON_ONLY explains material damage and worsens JFDE beyond FULL; yes,
   common drift is causally important, though not the sole mechanism.
5. Near-GT branch behavior is somewhat less harmful in ADE, but evidence is
   not monotone/consistent enough to make branch mismatch primary.
6. Goal oracle equals trajectory-ADE oracle 55.11% of the time.
7. ORACLE_ONLY is less marginally harmful but worse than FULL in JFDE; it does
   not outperform paired Stage A overall.
8. Goal/rank Spearman values are small: `0.010/-0.063` for error and
   `0.068/0.061` for rank versus ADE/FDE change.
9. Weighted/diff gradient ratio is 0.552% at INIT and 1.303% at BEST; cosine
   is 0.241 and 0.608.
10. `0.05*L_relative` is negligible in magnitude and mostly aligned, not
    broadly conflicting.
11. Relation information is materially used; branch-specific relation identity
    has weak aggregate metric utility.
12. Localization: **mixed**, dominated by common/component drift and objective
    gradient magnitude imbalance.
13. Narrowest V2 review direction: component-relative zero-mean residual
    semantics. No V2 implementation is included.

Final eligibility: **`eligible_for_stage_b_v2_design_review`**.
