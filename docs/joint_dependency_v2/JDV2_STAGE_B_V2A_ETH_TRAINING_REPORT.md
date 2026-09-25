# JDV2 Stage-B V2-A ETH formal training report

## Status

**`STAGE_B_V2A_SUCCESS`**

Next state: **`eligible_for_stage_b_v2a_adoption_review`**.

One and only one formal ETH Stage-B V2-A training experiment was run, with
training seed 2035. The selected checkpoint was evaluated on the predeclared
five inference seeds 2035--2039. No hyperparameter sweep, second training
seed, Stage-A change, V2-B implementation, or non-ETH experiment occurred.

V2-A passes every hard implementation contract and every predeclared
scientific adoption condition. It recovers 59.92% of V1's paired Stage-A
minADE gap and 79.38% of its minFDE gap, while improving JFDE relative to
Stage A. The constrained training result is, however, only marginally
different from the V1 post-hoc component-centering counterfactual. The main
scientific effect is therefore attributable to the component-relative
zero-mean constraint; this experiment does not establish a large extra
benefit from retraining within that subspace.

Machine-readable result:
`outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v2a/final_results.json`.

## 1. Scope and provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Training source commit | `8645f96853e986abffe0a5aad48e3147c7f9168d` |
| Dataset / split | ETH validation, 139 windows |
| Training seed | 2035 |
| Inference seeds | 2035, 2036, 2037, 2038, 2039 |
| Stage-A parent | strict-no-z epoch 13 |
| Stage-A SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Stage-B V1 SHA256 | `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a` |
| Cache manifest SHA256 | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| DependencyCorrector source SHA256 | `0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522` |
| Architecture | `jdv2-stage-b-v2a` |
| Residual projection | `component_zero_mean` |
| Trainable parameters | 30,851, all `jdv2_corrector.*` |
| Projection parameters | 0 |

The protected Stage-A checkpoint, V1 checkpoint, and DependencyCorrector
source hashes were identical before and after training/audit. The selected
checkpoint differs from Stage A in 22 corrector tensors and zero
non-corrector tensors.

## 2. Resolved training protocol

The canonical configuration was used without changing a scientific variable:

- Adam, learning rate `1e-4`, ExponentialLR;
- `L = L_diff + 0.05 * L_relative`;
- BF16 with the existing FP32 islands;
- gradient clipping at 1;
- active timesteps `[30,25,20,15,10,5]`;
- maximum 100 epochs;
- validation from epoch 5 every 5 epochs;
- early-stopping patience 6 validation checks;
- checkpoint selection by minimum JADE, with JFDE tie-break;
- frozen Stage-A worlds, GDTS generator, relation, sampler, and diffusion;
- P=20, K=21 and the frozen exact persistent-tie Stage-A sampler.

The exact launch command is preserved in
`outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_b_v2a_eth_seed2035/training_command.txt`.

### Execution interruption and resume

The execution session terminated without a Python traceback during an
incomplete epoch 13. The incomplete epoch was discarded. Training was resumed
under `nohup+setsid` from `last_model.pt` at completed epoch 12, restoring the
model, optimizer, and scheduler and restarting at epoch 13. No partially
completed epoch-13 checkpoint was used.

The early-stopping patience counter was process-local and therefore restarted
on resume. Training consequently continued through epoch 40 before six
post-resume validation checks without a JADE improvement triggered stopping.
This changed compute duration, not checkpoint selection: epoch 5 remained the
global JADE/JFDE optimum throughout. The event is recorded in
`resume_provenance.txt`.

## 3. Training result and selected checkpoint

Training ended at epoch 40. The selected checkpoint is epoch 5:

```text
outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/
jdv2_stage_b_v2a_eth_seed2035/saved_models/best_model.pt
```

SHA256:
`e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73`.

The last checkpoint is epoch 40, SHA256
`c7c4019c7aa5ed485b54910aee9bc830d60c07ed2443beaef62aeaa35d6440e1`.

Epoch-5 deterministic selection validation:

| minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|
| 0.282328 | 0.389425 | **0.422988** | **0.691250** | 0.692777 | 0.478072 | 0.298416 |

Validation JADE was 0.424681 at epoch 10 and never returned below the epoch-5
value. The total training loss decreased from 0.819037 at epoch 1 to 0.752868
at epoch 40; this continued training-loss decrease did not translate into a
better validation checkpoint.

Selected compact training diagnostics:

| Epoch | L_diff | L_relative | Total | raw RMS | projected RMS | removed-common RMS | common energy fraction | zero-mean max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.812559 | 0.129577 | 0.819037 | 0.105070 | 0.088846 | 0.047254 | 0.260030 | 7.21e-8 |
| 5 | 0.773585 | 0.124107 | 0.779791 | 0.166793 | 0.147259 | 0.069896 | 0.231871 | 1.37e-7 |
| 10 | 0.779120 | 0.120314 | 0.785136 | 0.180176 | 0.162292 | 0.070697 | 0.170228 | 1.52e-7 |
| 20 | 0.766297 | 0.116910 | 0.772143 | 0.205509 | 0.181851 | 0.082173 | 0.214314 | 1.63e-7 |
| 30 | 0.749165 | 0.116318 | 0.754980 | 0.220185 | 0.190941 | 0.092948 | 0.243404 | 1.71e-7 |
| 40 | 0.747019 | 0.116978 | 0.752868 | 0.224505 | 0.196684 | 0.090982 | 0.235120 | 1.78e-7 |

Peak training CUDA memory was approximately 351.9 MB allocated and 585.1 MB
reserved. Epochs took approximately 8--11 minutes including scheduled
validation.

## 4. Five-seed prediction results

All metrics are lower-is-better. Values are population mean +/- standard
deviation across the five predeclared inference seeds.

| Variant | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: frozen Stage A | 0.278201+/-0.001376 | 0.386215+/-0.002358 | 0.413185+/-0.004708 | 0.701908+/-0.008592 | 0.689827+/-0.007851 | 0.487309+/-0.015375 | 0.297907+/-0.004499 |
| B: V1 epoch 10 | 0.282617+/-0.001970 | 0.391074+/-0.004110 | 0.416070+/-0.004487 | 0.703679+/-0.009700 | 0.689827+/-0.007851 | 0.487309+/-0.015375 | 0.297877+/-0.003840 |
| C: V1 + post-hoc centering | 0.280106+/-0.001641 | 0.387595+/-0.002411 | 0.414895+/-0.004490 | 0.699614+/-0.008765 | 0.689827+/-0.007851 | 0.487309+/-0.015375 | 0.298333+/-0.003600 |
| D: trained V2-A | **0.279971+/-0.001605** | **0.387217+/-0.003534** | 0.414958+/-0.004441 | **0.698169+/-0.008011** | 0.689933+/-0.007654 | 0.487513+/-0.015452 | **0.297441+/-0.004300** |

A/B/C reuse the approved paired V1 failure-mechanism artifact. D uses the
same ETH split, seed set, checkpointed Stage-A worlds, metric implementation,
coordinate conversion, and 20/20 coverage protocol. Independently
materialized five-seed artifacts show tiny endpoint/compatibility replay
differences (at most 0.042% in the means); these cannot be caused by the
trajectory corrector. The current-run paired seed-2035 audit verifies
candidate IDs and goals unchanged for all 139 windows, and endpoint and
compatibility are exactly equal between the paired residual paths. The tiny
cross-artifact replay difference is retained transparently and is not treated
as a method effect.

### D per seed

| Seed | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.282328 | 0.389425 | 0.422988 | 0.691250 | 0.692777 | 0.478072 | 0.298416 |
| 2036 | 0.280427 | 0.386951 | 0.409468 | 0.689806 | 0.675802 | 0.493716 | 0.292028 |
| 2037 | 0.279399 | 0.390181 | 0.413452 | 0.711465 | 0.697011 | 0.509598 | 0.303023 |
| 2038 | 0.280312 | 0.389048 | 0.415213 | 0.702648 | 0.688420 | 0.492245 | 0.300766 |
| 2039 | 0.277388 | 0.380482 | 0.413670 | 0.695677 | 0.695656 | 0.463932 | 0.292972 |

### D by interaction stratum

| Stratum | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| E=0 | 0.359907+/-0.007082 | 0.520719+/-0.011059 | 0.359907+/-0.007082 | 0.520719+/-0.011059 | 0.495247+/-0.000000 | 0 | 0 |
| E>0 | 0.268267+/-0.001806 | 0.367670+/-0.003576 | 0.443082+/-0.005135 | 0.788823+/-0.008775 | 0.789393+/-0.011564 | 0.736568+/-0.023346 | 0.449395+/-0.006496 |

Every one of 1,840 evaluated agent-seed records retained exactly 20 unique
candidates. Coverage was 20/20 at minimum, mean, and maximum.

## 5. Relative comparisons

Negative percentages are improvements because all metrics are
lower-is-better.

| Comparison | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| D vs A | +0.636% | +0.259% | +0.429% | **-0.533%** | +0.015% | +0.042% | **-0.156%** |
| D vs B | **-0.936%** | **-0.986%** | **-0.267%** | **-0.783%** | +0.015% | +0.042% | **-0.146%** |
| D vs C | -0.048% | -0.098% | +0.015% | -0.206% | +0.015% | +0.042% | -0.299% |

D recovers 59.92% of B's minADE harm and 79.38% of B's minFDE harm relative
to A. Recovery is not driven by a single seed: the D marginal gap is smaller
than the V1 gap on every seed for both minADE and minFDE. The seed-level
recovery magnitudes vary, as expected for a five-seed stochastic evaluation.

Marginal quality is substantially recovered but not exactly equal to Stage A:
the remaining mean gaps are +0.636% minADE and +0.259% minFDE. JADE is 0.429%
worse than Stage A but 0.267% better than V1. JFDE is 0.533% better than Stage
A and 0.783% better than V1.

## 6. Projection, identity, runtime, and memory audit

The best checkpoint audit covered 139 windows: 47 E=0, 92 E>0, including 30
mixed degree-zero windows and 62 all-active windows.

| Contract | Result |
|---|---:|
| E=0 V2-A versus Stage-A max difference | **0** |
| Mixed-scene degree-zero agent max difference | **0** |
| Current-corrector projection-none path versus same-weight reference | **0** |
| Projection versus audit counterfactual residual | 1.19e-7 |
| V2-A production versus audit counterfactual trajectory | 3.58e-7 |
| Raw reconstruction max difference | 7.45e-9 |
| Component-zero-mean max absolute sum | 2.98e-7 |
| Candidate-ID checks | 7,360 unchanged |
| Joint-goal value checks | 14,720 unchanged |
| RNG post-state equality | PASS |

The underlying reused audit helper retains the historical JSON key
`v1_production_vs_reference`. Because this post-training audit loads the V2-A
checkpoint, that key denotes the V2-A corrector with projection disabled
versus its same-weight reference arithmetic; it is not a fresh V1 epoch-10
evaluation. V1 comparison metrics come from the approved V1 failure audit.

Across 11,040 deployed corrector calls, raw residual RMS averaged 0.199580,
projected RMS 0.167590, and removed-common RMS 0.104032. The removed common
component accounted for a mean **33.33%** of raw residual energy (median
32.24%, p90 52.15%). Thus the projection is active and removes a substantial,
measured common mode rather than behaving as an identity transform.

Mean end-to-end inference time was 0.5452 s/window for the raw path and 0.5660
s/window for V2-A, an overhead of **3.82%**, below the 5% hard gate. Peak CUDA
allocation was 85,522,944 bytes and peak reserved memory 130,023,424 bytes for
the paired audit. The independent five-seed evaluator averaged 90.44 seconds
per seed and used 104,762,880 bytes peak allocated.

## 7. Gradient audit and V2-B decision

The read-only best-checkpoint gradient audit used 64 E>0 training windows and
performed no optimizer step. All audited parameter names belonged to the
corrector.

- mean diffusion-gradient norm: 2.4399;
- mean raw relative-gradient norm: 0.6380;
- mean weighted relative-gradient norm: 0.03190;
- mean weighted-relative/diffusion ratio: 1.775% (median 1.271%, p90 3.457%);
- mean gradient cosine: 0.8242;
- conflict rate: 1.56%.

The weighted relative gradient remains small, but this alone does not justify
V2-B. The approved V2 design permits V2-B review only when a correctly
enforced V2-A still has no clear trajectory gain. Here V2-A passes the full
adoption signature and improves JFDE while recovering most V1 marginal harm.
Therefore **V2-B objective balancing is not scientifically justified by this
experiment**.

## 8. Adoption gates

All hard gates passed:

- zero-mean projection and counterfactual parity;
- tensor-exact E=0 and mixed degree-zero Stage-A identity;
- unchanged Stage-A worlds, protected hashes, and 20/20 coverage;
- exactly 30,851 trainable corrector parameters and zero projection
  parameters;
- no non-corrector tensor change;
- paired candidate/goal invariance and RNG equality;
- no NaN/Inf or checkpoint mismatch;
- 3.82% runtime overhead, below 5%.

All predeclared scientific adoption conditions passed:

1. minADE V1-gap recovery: 59.92%, at least 50%;
2. minFDE V1-gap recovery: 79.38%, at least 50%;
3. JADE: +0.429% versus Stage A and better than V1;
4. JFDE: 0.533% better than Stage A;
5. Relative Motion: 0.156% better than Stage A;
6. raw common output removed to FP32 tolerance at every audited step;
7. improvements are not attributable to one evaluation seed.

## 9. Required scientific answers

**Best epoch and validation metrics.** Epoch 5, selected by JADE 0.422988
with JFDE 0.691250 tie-break; minADE 0.282328 and minFDE 0.389425.

**V2-A versus Stage A.** V2-A retains small marginal gaps (+0.636% minADE,
+0.259% minFDE) and a +0.429% JADE gap, while improving JFDE by 0.533% and
Relative Motion by 0.156%.

**V2-A versus V1.** V2-A improves minADE by 0.936%, minFDE by 0.986%, JADE by
0.267%, JFDE by 0.783%, and Relative Motion by 0.146%.

**V2-A versus V1 plus post-hoc centering.** Differences are all below 0.30%.
D slightly improves minADE, minFDE, JFDE, and Relative Motion, while JADE is
0.015% worse. This is practical parity, not evidence of a large C-to-D gain.

**Does C to D provide a real gain?** No material additional gain is
established at five seeds. The directions are mildly favorable overall, but
the effect is much smaller than seed-to-seed variation. The dominant causal
benefit remains the projection. D remains important because it is the actual
freshly trained, deployable constrained model; C is a read-only,
non-deployable intervention.

**How much common residual is removed?** Mean removed-common RMS is 0.104032;
the mean removed common-energy fraction is 33.33%.

**Is marginal quality recovered?** Substantially, but not completely. V2-A
recovers 59.92%/79.38% of V1's minADE/minFDE harm and finishes within
0.636%/0.259% of Stage A.

**Do JADE/JFDE improve?** Relative to V1, both improve. Relative to Stage A,
JADE is 0.429% worse while JFDE is 0.533% better; both satisfy the registered
gate.

**Is constrained-subspace training scientifically useful?** Yes as a stable,
identity-safe, deployable realization of the supported component-relative
mechanism. The data do not support claiming that retraining adds a large
benefit beyond the constraint itself.

**Is V2-B objective balancing justified?** No. Its eligibility premise is not
met because V2-A passes the adoption signature and shows clear recovery over
V1.

## 10. Final decision

Final validation completed successfully with the project interpreter:

```text
python -m compileall -q .
# PASS

pytest -q
# 432 passed, 12 warnings

pytest -q tests/test_jdv2_stage_b_v2a.py \
  tests/test_jdv2_stage_b_mixed_precision_identity.py \
  tests/test_jdv2_stage_a_freeze_contract.py
# 20 passed on the real RTX 5070 Ti CUDA/BF16 path

git diff --check
# clean
```

The warnings are existing dependency deprecations and a PyTorch Transformer
layout warning; no test failed. Post-test SHA256 verification reproduced the
protected Stage-A, V1, DependencyCorrector, and selected V2-A hashes recorded
above.

Primary final state: **`STAGE_B_V2A_SUCCESS`**.

Next state: **`eligible_for_stage_b_v2a_adoption_review`**.

No V2-B, additional training seed, other dataset, or hyperparameter change
was started.
