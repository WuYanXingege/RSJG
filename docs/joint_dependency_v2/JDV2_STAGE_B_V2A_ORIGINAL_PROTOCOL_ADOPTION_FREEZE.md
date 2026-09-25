# JDV2 Stage-B V2-A original-protocol adoption freeze

## Status

**`STAGE_B_V2A_ORIGINAL_GDTS_PROTOCOL_FROZEN`**

Next state only:
**`eligible_for_jdv2_cross_dataset_benchmark_preflight`**.

This is a protocol-scoped freeze. It adopts Stage-B V2-A for comparisons that
follow the original GDTS ETH/UCY mirrored validation/test protocol. It does
not claim an independent held-out test result and does not authorize V2-B or
additional training in this task.

Machine-readable contract:

```text
outputs/joint_dependency_v2/eth/joint_dependency_v2/
stage_b_v2a_original_protocol_freeze/manifest.json
```

## 1. Recovery from the superseded clean reproduction

The repository was recovered at adoption base commit
`fb767ee87f01342a073198a8c8c1959460d496a7` with no tracked modification from
the interrupted run. The clean Stage-A process was terminated, while its
runtime directory was preserved. It completed epoch 1 and stopped during
epoch 2 at the last logged progress 1938/3790.

That run is **`ABORTED_NON_AUTHORITATIVE`**. Its epoch-1 checkpoints, logs,
diagnostics, and metric values are excluded from every conclusion here, may
not initialize Stage B, and are not committed. No data, cache, historical
checkpoint, or clean-split infrastructure was deleted.

## 2. Evaluation-protocol scope

The original GDTS repository establishes the mirrored protocol through three
independent source facts:

1. `_eval_source_files_are_identical()` compares ordered validation and test
   files by scene, downsampling, basename, and byte content.
2. `_link_identical_eval_batches()` states that ETH/UCY validation and test
   files are identical and reuses validation cache batches as test batches.
3. the original `train.sh` uses `phase=train_test` for all five ETH/UCY
   leave-one-scene-out targets.

The later content audit independently confirmed that the 139 ETH validation
and 139 test windows are ordered-identical. Consequently, validation=test is
a feature of the original GDTS evaluation protocol, not JDV2-specific data
leakage.

All paired GDTS -> Stage A -> V1 -> V2-A development comparisons used the
same protocol. They are fair relative comparisons because the compared
methods share the same windows, five seeds, sampled worlds, coordinate
conversion, and metrics. They are not evidence of strict held-out
generalization after model selection.

The historical `ADOPTION_BLOCKED_BY_SPLIT_PROTOCOL` result remains correct
and immutable under the separate strict clean-heldout criterion. The clean
protocol is maintained as an optional robustness protocol; it is not a
precondition for this original-protocol freeze.

## 3. Frozen artifacts and implementation

| Item | Frozen value |
|---|---|
| Scientific evidence commit | `817cc7551bd7f3422422230821acf8b3dd5cdc6c` |
| Adoption base commit | `fb767ee87f01342a073198a8c8c1959460d496a7` |
| Stage-A parent | strict-no-z epoch 13 |
| Stage-A SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Stage-B V1 SHA256 | `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a` |
| Stage-B V2-A | epoch 5 |
| V2-A SHA256 | `e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73` |
| DependencyCorrector | unchanged, 30,851 parameters |
| Corrector source SHA256 | `0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522` |
| Projection | `component_zero_mean`, zero parameters |
| Projection source SHA256 | `355de2a9ace3265efe1d5fc1ed4b5a6c054a621fc452a7ad95d70b6a2c3b5630` |

Stage A remains strict-no-z with K=21, P=20, M=4, energy rank 8, the
parameter-free radius+TTC graph (6 m, TTC 8 s, dt 0.4 s), weighted
Gumbel-Top-P Round 0, and two synchronous
`exact_lexicographic_persistent_tie` refinement rounds.

Stage B retains the unchanged DependencyCorrector, same-slot relation state,
one scene-level oracle branch during training, active timesteps
`[30,25,20,15,10,5]`, and the parameter-free component-zero-mean projection
after the corrector and before epsilon addition. The optimizer is Adam at
1e-4, precision is BF16 with the existing FP32 islands, and the frozen loss
is `L_diff + 0.05 L_relative`. Only DependencyCorrector was trained. The
projection has no learned parameters.

## 4. Original-protocol scientific result

Five-seed means use inference seeds 2035--2039 on the 139 ETH evaluation
windows. All metrics are lower-is-better.

| Variant | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage A | 0.278201 | 0.386215 | 0.413185 | 0.701908 | 0.689827 | 0.487309 | 0.297907 |
| V1 | 0.282617 | 0.391074 | 0.416070 | 0.703679 | 0.689827 | 0.487309 | 0.297877 |
| V2-A | 0.279971 | 0.387217 | 0.414958 | 0.698169 | 0.689933 | 0.487513 | 0.297441 |

V2-A relative to V1:

- minADE -0.936%; minFDE -0.986%;
- JADE -0.267%; JFDE -0.783%;
- Relative Motion Error -0.146%.

V2-A relative to Stage A:

- minADE +0.636%; minFDE +0.259%;
- JADE +0.429%; JFDE -0.533%;
- Relative Motion Error -0.156%.

V2-A recovers 59.92% of V1's minADE gap and 79.38% of its minFDE gap. The
endpoint and compatibility differences across independently materialized
artifacts are tiny replay variation and are not attributed to the trajectory
corrector.

## 5. Supported mechanism

The V1 audit localized a harmful component-common residual mode. V2-A
constrains each interacting component's residual to the component-relative
zero-mean subspace. The post-training audit measured a mean 33.33% removed
component-common residual-energy fraction.

The supported scientific conclusion is therefore:

> The component-zero-mean residual constraint removes a harmful common-mode
> degree of freedom and recovers most V1 marginal degradation while improving
> JFDE under the original GDTS protocol.

It is not claimed that retraining itself supplies a large gain: the trained
V2-A checkpoint and V1 with post-hoc centering are in practical parity. Nor
is universal superiority, an exact probabilistic decomposition, or clean
independent-test generalization claimed.

## 6. Identity, coverage, and runtime contracts

- E=0 execution is tensor-exact Stage-A execution.
- Degree-zero agents inside mixed scenes are tensor-exact Stage A.
- Candidate IDs and joint goals remain unchanged on paired audits.
- Every evaluated agent retains 20/20 candidate coverage.
- Projection parameters: 0.
- Mean end-to-end projection overhead: 3.82%.
- Mean common-residual energy removed: 33.33%.
- Stage-A checkpoint bytes and all Stage-A model semantics remain unchanged.

These contracts are preserved by numerical routing: correction arithmetic is
only applied to active agents, while inactive agents take the exact Stage-A
transition.

## 7. Frozen boundary and limitations

Frozen scientific method:

- strict-no-z Stage A and its checkpoint;
- Goal U-Net candidate bank and order;
- K/P/M/rank and graph constants;
- exact persistent sampler and two synchronous rounds;
- DependencyCorrector architecture and 30,851 parameters;
- component-zero-mean projection location and zero-parameter semantics;
- active timesteps and same-slot relation conditioning;
- loss coefficients, Adam/1e-4, and BF16 precision;
- Stage-A identity routing for E=0 and degree-zero agents.

Known limitation: the primary ETH protocol does not provide a validation/test
separation, so strict held-out generalization remains unestablished. The
optional clean-split infrastructure is retained for a future robustness
study. The interrupted clean reproduction is not evidence.

V2-B is not authorized. The weighted relative-gradient norm (about 1.775% of
the diffusion gradient, cosine about 0.824, conflict about 1.56%) is not a
reason to reopen objective tuning after V2-A passed the registered
original-protocol gates. `lambda_diff=1.0` and `lambda_relative=0.05` remain
frozen.

## 8. Adoption decision

V2-A is frozen under **`ORIGINAL_GDTS_PROTOCOL`**. Future benchmark reports
must preserve the fairness/held-out distinction, use the unchanged method,
and perform a dataset-specific protocol preflight before any run.

Final state: **`STAGE_B_V2A_ORIGINAL_GDTS_PROTOCOL_FROZEN`**.

Next state only:
**`eligible_for_jdv2_cross_dataset_benchmark_preflight`**.
