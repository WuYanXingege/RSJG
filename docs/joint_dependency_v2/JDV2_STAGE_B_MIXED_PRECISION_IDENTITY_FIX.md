# JDV2 Stage-B mixed-precision identity-contract fix

## Status

**`STAGE_B_V1_NO_CLEAR_GAIN_MECHANISM_IDENTIFIED`**

The CUDA/BF16 numerical identity bug is fixed and the mandatory paired parity
gate passes. The same epoch-10 checkpoint was reevaluated without retraining.
It still has no clear trajectory-level gain, and the resumed audit identifies
a mixed failure mechanism led by component-common residual drift and an
objective-gradient magnitude imbalance.

No checkpoint, model architecture, loss, sampler, Stage-A method, or
DependencyCorrector implementation was changed. No training or finetuning was
run.

## 1. Provenance and immutable artifacts

| Item | Value |
|---|---|
| Starting HEAD | `b9ad89110c75dfaea3fdbcb1552b7c10563ba952` |
| Branch | `research/joint-dependency-v2-clean` |
| Stage-A checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Stage-B V1 epoch-10 SHA256 | `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a` |
| DependencyCorrector parameters | 30,851, unchanged |
| DependencyCorrector source SHA256 | `0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522` |

Machine result:
`outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v1/identity_contract_fix/results.json`.

## 2. First-divergence diagnosis

The trace used ETH validation window 14, seed 2035, the same Stage-A worlds,
candidate IDs, relation state, `x_T`, and six branch-noise tensors.

At the first active branch step, timestep 30:

| Tensor/operator | Stage A dtype | zero-residual Stage B dtype | max abs diff | differing elements |
|---|---|---|---:|---:|
| `x_t` before denoiser | FP32 | FP32 | 0 | 0/24 |
| `epsilon_base` | BF16 | BF16 | 0 | 0/24 |
| `delta_epsilon` | — | FP32 | 0 | 0/24 nonzero |
| `epsilon_base + delta` | BF16 baseline | **FP32** | 0 | 0/24 values |
| `first_term` | FP32 | FP32 | 0 | 0/24 |
| `second_term` | BF16 | **FP32** | `8.72575e-5` | **24/24** |
| `third_term` | FP32 | FP32 | 0 | 0/24 |
| `x_next` | FP32 | FP32 | `8.72612e-5` | **24/24** |
| final branch velocity | FP32 | FP32 | `0.00350177` | 24/24 |

The first semantic divergence is therefore the dtype promotion at
`epsilon_base + delta_epsilon`; the first value divergence is the immediately
following `second_term`. This confirms the mixed-precision zero-addition
hypothesis rather than stochastic or corrector-output error.

Compact trace:
`outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v1/identity_contract_fix/pre_fix_first_divergence.json`.

## 3. Minimal production change

Only the Stage-B branch section of `GDTS.ts_sample()` changed.

- Structural activity is derived once from the frozen `edge_index`.
- If `E=0`, the corrector and correction addition are not executed; the
  original Stage-A DDIM expression is used literally.
- For mixed windows, `epsilon_base`, `delta_epsilon`, and diffusion noise are
  each computed once. The existing corrected transition is retained for
  degree-positive agents, while degree-zero agents receive values from the
  literal baseline transition.
- If every agent has positive degree, the pre-fix V1 corrected expression is
  used directly, avoiding even a routing operation.
- The same single noise tensor feeds both arithmetic variants. There is no
  extra network forward and no extra RNG draw.

This is numerical routing only. It does not cast active residuals to BF16,
scale, normalize, center, clip, or otherwise redefine the learned corrector.

## 4. Mandatory 139-window parity

Protocol: ETH validation, seed 2035, CUDA/BF16, explicit paired NoiseTape.

| Contract | Result |
|---|---:|
| windows | 139/139 |
| E=0 windows | 47 |
| E>0 windows | 92 |
| mixed active/inactive windows | 30 |
| audit FULL vs fixed production max abs | **0.0** |
| audit NONE vs Stage-A production max abs | **0.0** |
| E=0 FULL vs Stage A max abs | **0.0** |
| mixed degree-zero vs Stage A max abs | **0.0** |
| degree-positive fixed vs pre-fix V1 max abs | **0.0** |
| post-RNG state equality | **PASS** |
| candidate IDs / goals unchanged | **PASS** |
| candidate coverage | **20/20** |

Thus both the whole-window and per-agent identity contracts are tensor-exact,
while the trained active path is tensor-exact to V1.

## 5. Corrected five-seed V1 evaluation

Same checkpoint, ETH validation, seeds 2035–2039, paired Stage-A standard
evaluator. Values are mean ± population standard deviation; lower is better.

| Method | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| paired Stage A | 0.278201±0.001376 | 0.386215±0.002358 | 0.413185±0.004708 | 0.701908±0.008592 | 0.689827±0.007851 | 0.487309±0.015375 | 0.297907±0.004499 |
| fixed Stage-B V1 | 0.282617±0.001970 | 0.391074±0.004110 | 0.416070±0.004487 | 0.703679±0.009700 | 0.689827±0.007851 | 0.487309±0.015375 | 0.297877±0.003840 |
| relative change | **+1.587%** | **+1.258%** | **+0.698%** | **+0.252%** | 0 | 0 | -0.010% |

All E=0 trajectories and therefore all E=0 metrics are exactly equal:
minADE/JADE `0.359907`, minFDE/JFDE `0.520719`, endpoint `0.495247`, and
zero compatibility/relative-motion metrics. In the 30 mixed windows,
degree-zero trajectories are also tensor-exact, which is stronger than an
aggregate subset-metric comparison.

The correction removes the previous E=0 contamination but does not change the
scientific interpretation: Stage-B V1 still worsens all four trajectory
metrics and has effectively unchanged relative motion. It remains
`STAGE_B_V1_NO_CLEAR_GAIN`.

## 6. Resumed failure-mechanism audit

The restored H1–H4 audit is documented in
`JDV2_STAGE_B_V1_FAILURE_MECHANISM_AUDIT.md`. Its combined interpretation is:

- **H1 supported:** 31.70% mean residual energy is component-common.
  Component centering recovers 56.85% of minADE harm and 71.59% of minFDE
  harm, while JFDE becomes 0.327% better than Stage A.
- **H2 secondary/incomplete:** ORACLE_ONLY reduces marginal harm, but does not
  beat Stage A and worsens JFDE more than FULL. Goal-oracle equals
  trajectory-oracle only 55.11%, yet branch-rank correlations with harm are
  weak.
- **H3 supported as magnitude imbalance, not directional conflict:** weighted
  relative-gradient norm is only 0.552% of diffusion-gradient norm at INIT
  and 1.303% at BEST. Mean cosine is positive (0.241/0.608), with low conflict
  rates (1.56%/6.25%).
- **H4 weak branch specificity:** zeroing relation is materially harmful, so
  relation conditioning is used. Replacing it with branch mean or a fixed
  branch permutation changes tensors but alters aggregate metrics by at most
  about 0.12%, so same-slot branch identity has weak metric utility.

Primary mechanism class: **`MIXED_FAILURE_MECHANISM`**, led by
`COMMON_MODE_DRIFT_SUPPORTED` and
`OBJECTIVE_GRADIENT_MISMATCH_SUPPORTED`, with weak branch-specific relation
utility as secondary evidence.

## 7. Answers to the required questions

1. First divergence: dtype promotion at `epsilon_base + delta`; first value
   divergence at `second_term` at timestep 30.
2. Dtypes: BF16 base epsilon plus FP32 zero became FP32; baseline second term
   stayed BF16 while Stage B became FP32.
3. Root cause confirmed: **yes**.
4. Fix: structurally bypass correction for E=0 and route literal baseline
   transitions to degree-zero agents using shared tensors/noise.
5. Whole-window E=0 exact: **yes, 47/47 windows**.
6. Mixed-window degree-zero exact: **yes, 30/30 mixed windows**.
7. Additional RNG consumed: **none**.
8. Degree-positive V1 changed: **no, maximum absolute difference 0**.
9. Stage-A candidates/worlds changed: **no; 100% equal, 20/20 coverage**.
10. Corrected metrics: reported in Section 5 and the machine artifact.
11. Previous no-clear-gain result changed: **no**.
12. Mandatory parity restored: **yes, all gates exact**.
13. H1/H3 supported; H2 secondary; H4 shows relation use but weak branch
    specificity.
14. Stage-B V2 design is now scientifically justified: **yes**, but no V2 was
    implemented.

## 8. Narrowest eligible V2 design direction

The counterfactual evidence most directly supports reviewing a
component-relative, zero-mean residual semantics that removes unidentifiable
component translation while retaining relative edge corrections. This is a
design-review recommendation only. It is not implemented here, and this task
does not authorize changing the objective or production residual semantics
beyond the identity bug fix.
