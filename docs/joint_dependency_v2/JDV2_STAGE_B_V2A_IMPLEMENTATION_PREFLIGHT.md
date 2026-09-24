# JDV2 Stage-B V2-A implementation preflight

## Status

**`STAGE_B_V2A_READY_FOR_TRAINING`**

Next state only: **`eligible_for_stage_b_v2a_training_review`**.

This preflight implemented and validated the approved component-relative
zero-mean denoising residual. It did not train or finetune a model, change
Stage A, tune a hyperparameter, or implement V2-B. Formal V2-A training is not
automatically authorized by this result; it is eligible for a separate
training review.

Machine-readable evidence is stored in
`outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v2a/preflight/results.json`.

## 1. Scope and provenance

The implementation was made on branch
`research/joint-dependency-v2-clean`, starting from
`978fb63093f923787391aaa9174363d9f35227c2`.

| Artifact | Value |
|---|---|
| Frozen Stage-A checkpoint | epoch 13 |
| Stage-A SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Audit-only Stage-B V1 checkpoint | epoch 10 |
| Stage-B V1 SHA256 | `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a` |
| `DependencyCorrector` source SHA256 | `0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522` |
| ETH validation windows | 139 |
| Paired validation seed | 2035 |
| Device / precision | RTX 5070 Ti / CUDA BF16 |

Both protected checkpoint hashes and the `DependencyCorrector` source hash
remain unchanged.

## 2. Exact files changed

Production/configuration:

- `src/models/joint_dependency_v2/component_residual_projection.py`
- `src/models/joint_dependency_v2/__init__.py`
- `src/models/model.py`
- `src/parser.py`
- `src/trainer.py`
- `configs/joint_dependency_v2/jdv2_stage_b_v2a_eth.yaml`

Validation and evidence:

- `tests/test_jdv2_stage_b_v2a.py`
- `tools/jdv2_stage_b_v2a_preflight.py`
- `docs/joint_dependency_v2/JDV2_STAGE_B_V2A_IMPLEMENTATION_PREFLIGHT.md`
- `outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v2a/preflight/results.json`

`dependency_corrector.py`, `diffusion.py`, Stage-A modules, relation, energy,
sampler, losses, and protected checkpoints were not modified.

## 3. Production insertion and semantics

The only scientific operation added is:

```text
raw_delta = unchanged DependencyCorrector(...)
projected_delta = component_zero_mean_projection(raw_delta, metadata)
epsilon_joint = epsilon_base + projected_delta
```

It is inserted after the corrector output and before residual addition in both
`_jdv2_dependency_losses()` and `ts_sample()`. The V1 switch value `none`
executes the existing raw-residual path. The V2-A value
`component_zero_mean` executes the projection in an autocast-disabled FP32
island.

The projection is stateless and parameter-free. `DependencyCorrector` remains
30,851 parameters. No denoiser or corrector forward is duplicated.

### Shapes and dtypes

| Object | Shape | dtype |
|---|---:|---|
| `edge_index` | `[2,E]` | int64 |
| `degree`, `component_id` | `[N]` | int64 |
| `active_mask` | `[N]` | bool |
| `active_index` | `[A]` | int64 |
| `component_count` | `[Q]` | int64 |
| raw/projected residual | `[N,12,2]` | FP32 |
| component sums/means | `[Q,12,2]` | FP32 |
| `epsilon_base` under CUDA AMP | `[N,12,2]` | BF16 |

CUDA finite assertions are asynchronous, avoiding a host synchronization at
every branch/timestep while still rejecting NaN or Inf.

## 4. Component metadata and reuse

Connectivity uses only the frozen sparse graph and treats canonical edges as
undirected. Component IDs are compact and deterministic in ascending minimum
agent-index order. Edge indices, self edges, scene crossing, shapes, dtype,
and device are validated. Degree-zero agents use `component_id=-1` and are
excluded from reductions.

Inference constructs metadata once in `_jdv2_encode()` and stores it in
`dependency_state`; all 20 branches and six active DDIM steps reuse it.
Training constructs it once per training batch. Direct instrumentation over
all validation data observed exactly 139 metadata builds for 139 windows.
There was no build inside the 20 x 6 loop.

## 5. Mathematical contracts

The deterministic topology test was:

```text
0 -- 1 -- 2
3 -- 4
5 isolated
```

All mathematical tests passed:

- both active components are centered independently;
- the maximum real-data absolute component sum was `2.3841858e-7`;
- edge-relative residual differences are invariant within FP32 tolerance;
- `raw = removed_common + projected` had maximum error `2.7939677e-9`;
- degree-zero and E=0 projections are exact zero;
- gradients propagate through the projection;
- the projection has no parameters or optimizer state;
- cross-scene edges and invalid graph indices are rejected.

On the V1 checkpoint, the mean raw residual RMS was `0.172914`, projected RMS
was `0.151119`, and removed-common RMS was `0.077466`. The mean removed common
energy fraction was `0.317526`, reproducing the prior mechanism audit.

## 6. V1 backward compatibility

With `jdv2_residual_projection=none`, production V1 was replayed over all 139
ETH validation windows with paired noise. The maximum difference from the
frozen V1 reference arithmetic was exactly `0.0`.

Candidate IDs (7,360 values) and joint goal coordinates (14,720 values) were
unchanged. Python, NumPy, Torch CPU, and all CUDA RNG post-states matched.
This covers 47 E=0, 30 mixed, and 62 all-active windows.

## 7. Stage-A and mixed-precision identity

A fresh V2-A model was loaded from the protected Stage-A epoch-13 parent; no
optimizer step preceded the identity gate. Across all 139 windows:

| Subset | windows | max absolute trajectory difference vs Stage A |
|---|---:|---:|
| overall | 139 | 0.0 |
| E=0 | 47 | 0.0 |
| mixed degree-zero agents | 30 | 0.0 |
| all-active E>0 | 62 | 0.0 |

Thus the zero-initialized V2-A parent is tensor-exact Stage A, including CUDA
BF16. Whole-window E=0 bypasses the corrector arithmetic, mixed degree-zero
agents retain the literal baseline DDIM transition, and the all-active
zero-initialized case bypasses correction after verifying the unchanged zero
output head. No additional RNG was consumed.

## 8. Counterfactual parity

The V1 epoch-10 checkpoint was loaded read-only and projection was enabled
only as an inference counterfactual. With the same NoiseTape:

- production projection versus the old audit transform on the same raw
  residual: maximum difference `1.1920929e-7`;
- production V2-A trajectory versus old `COMPONENT_CENTERED` replay: maximum
  difference `4.7683716e-7`;
- predeclared acceptance tolerance: `2e-6`.

This is tensor-level parity, not metric-only similarity. The nonzero rounding
comes from mathematically equivalent FP32 reduction order. Seed-2035 overall
metrics further agree to floating-point noise:

| Path | minADE | minFDE | JADE | JFDE | Relative Motion |
|---|---:|---:|---:|---:|---:|
| old audit `COMPONENT_CENTERED` | 0.282156 | 0.389601 | 0.422901 | 0.691438 | 0.300478 |
| production V2-A projection | 0.282156 | 0.389601 | 0.422901 | 0.691438 | 0.300478 |

Endpoint (`0.692777`) and compatibility (`0.478072`) are unchanged because
candidate/world construction is unchanged. These are one-seed implementation
diagnostics, not V2-A training results.

## 9. Training-path preflight

One disposable real E>0 training batch was run in CUDA/BF16 through forward,
backward, and one optimizer-step smoke test. Nothing was saved.

- `L_diff = 1.366701`;
- `L_relative = 0.023786`;
- total with unchanged coefficients = `1.367890`;
- trainable parameters = 30,851, all under `jdv2_corrector.*`;
- gradients were finite and at least one corrector gradient was nonzero;
- every Stage-A/GDTS gradient remained `None`;
- only corrector parameters changed after the disposable step.

The projection is used identically in training and inference and is not
detached.

## 10. Checkpoint and provenance contracts

The explicit configuration switch is:

```yaml
jdv2_residual_projection: none | component_zero_mean
```

Its default is `none`. V1 requires `none`; `jdv2-stage-b-v2a` requires
`component_zero_mean`. Future V2-A checkpoints record:

```yaml
training_stage: joint_trajectory
stage_b_architecture_version: jdv2-stage-b-v2a
stage_b_residual_projection: component_zero_mean
```

They also retain the Stage-A parent hash, freeze-manifest hash, and freeze
source commit. The field was not added to the immutable Stage-A
`architecture_config`.

The canonical V2-A config points to the protected Stage-A epoch-13 checkpoint.
That parent loaded successfully. Attempting to resume V2-A training from the
V1 epoch-10 checkpoint was rejected with
`Stage-B checkpoint provenance mismatch: stage_b_architecture_version`.
The V1 checkpoint remains permitted only for explicit read-only audit use.

## 11. Runtime and CUDA memory

The paired benchmark alternated policy order by window and used the same
checkpoint, contexts, noise, and 139 ETH validation windows.

| Measurement | V1 | V2-A |
|---|---:|---:|
| mean `ts_sample` runtime/window | 0.559598 s | 0.582104 s |
| peak CUDA allocated (maximum) | 85,526,016 B | 85,522,944 B |
| peak CUDA reserved (maximum) | 130,023,424 B | 130,023,424 B |

Mean end-to-end overhead was **4.0217%**, below the 5% hard target. Mean
one-time metadata construction cost was `0.421 ms/window`. Mean projection
call cost, measured over 11,040 calls, was `0.266 ms`. Peak CUDA allocation
did not increase; reserved memory was identical.

## 12. Validation results

Commands completed successfully:

```text
python -m compileall -q .
pytest -q
# 432 passed, 12 warnings

git diff --check
# clean
```

The focused suite includes CPU FP32 and real CUDA/BF16 identity, mathematical
projection, disconnected components, scene isolation, autograd, RNG,
configuration discrimination, protected hashes, and existing Stage-A/Stage-B
regressions. The full suite includes the existing Stage-A freeze and Stage-B
V1 tests.

## 13. Hard-gate decision

All 15 required gates passed:

1. projection mathematics;
2. unchanged 30,851 corrector parameters;
3. V1 tensor-exact backward compatibility;
4. fresh V2-A tensor-exact Stage-A identity;
5. E=0 identity;
6. mixed degree-zero identity;
7. once-per-window metadata reuse;
8. RNG isolation;
9. absent Stage-A gradients;
10. V1-to-V2-A resume rejection;
11. frozen Stage-A epoch-13 parent;
12. CUDA/BF16 forward/backward;
13. old `COMPONENT_CENTERED` counterfactual parity;
14. runtime overhead below 5%;
15. full regression suite.

Primary final state: **`STAGE_B_V2A_READY_FOR_TRAINING`**.

Next state only: **`eligible_for_stage_b_v2a_training_review`**. No formal
training has been run, and no Stage-B V2-B or other dataset work is authorized.
