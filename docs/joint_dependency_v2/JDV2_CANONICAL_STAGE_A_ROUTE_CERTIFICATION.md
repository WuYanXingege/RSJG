# JDV2 Canonical Stage-A Route Certification

## 1. Certification result

**Primary state: `CANONICAL_STAGE_A_ZERO_ADD_DRIFT_CONFIRMED`**

The frozen canonical Stage-A route is not numerically identical to the
literal Stage-A diffusion route used as the reference by Stage-B.  Under
paired CUDA/BF16 execution, the first dtype divergence occurs at the exact
zero-residual addition:

```text
epsilon_base                 torch.bfloat16
delta_epsilon (exact zero)   torch.float32
epsilon_base + delta         torch.float32
```

The first value divergence occurs immediately afterward in the DDIM
`second_term`.  This confirms the mixed-precision zero-addition mechanism by
direct operator tracing; it is not inferred from final metrics.

This audit is read-only.  It did not change the Stage-A or Stage-B checkpoint,
the frozen configurations, the production sampler, training, or any reported
historical metric.

## 2. Scope and provenance

The independent audit identified the canonical-route question as the highest
priority certification boundary.  This audit therefore compares exactly three
paths:

| Path | Checkpoint | Corrector routing |
|---|---|---|
| A — canonical Stage-A | Stage-A epoch 13 | canonical `use_dependency_corrector=true` |
| B — literal Stage-A | same Stage-A epoch 13 | correction arithmetic disabled |
| C — Stage-B reference | Stage-B V2-A epoch 5 | correction arithmetic disabled |

Artifacts:

- Source commit used for execution: `60a92e29e6b474b40e2d9f2b1d61d0d15fd9e78b`.
- Stage-A checkpoint SHA256: `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`.
- Stage-B V2-A checkpoint SHA256: `e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73`.
- Stage-A config SHA256: `bf5d4a692b0a9e841de8523f2780013147ec3e9bcb78877cc3c352571bc63e62`.
- Stage-B config SHA256: `20ae4ba7fee93c6c61e9969c5828b880d78e600439ba0e20fd3b5dc2c78f9395`.
- Dataset/split: ETH validation, 139 windows.
- Evaluation seed: 2035.
- Precision/device: CUDA/BF16 on NVIDIA GeForce RTX 5070 Ti.
- Machine result: `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/canonical_stage_a_route_certification/results.json`.
- Machine-result SHA256: `0bac229d05672aafdebc0adfc997fcba4a1798220495061f9c504c428ab1a514`.

The frozen Stage-A config explicitly enables the dependency corrector at
`configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml:21`.  In the
production diffusion route, an active corrector is evaluated for E>0 at
`src/models/model.py:3783-3806`, and its residual is added at
`src/models/model.py:3841-3871`.

## 3. Exact paired protocol

For every validation window, A, B, and C used:

- identical input tensors and batch identity;
- identical Stage-A candidate IDs and joint goals;
- one literal shared Stage-A context tensor and dependency/relation state;
- identical Python, NumPy, CPU Torch, and all-device CUDA RNG state before
  diffusion;
- identical `x_T` and DDIM branch noise consumption;
- the same 20 branches and candidate/world allocation.

C deliberately receives the same Stage-A context and relation-state tensors,
instead of independently recomputed copies.  This makes the test a strict
numerical-routing experiment.  The Stage-B checkpoint is independently
verified to differ from Stage-A only in 22 `jdv2_corrector.*` tensors; no
non-corrector checkpoint tensor differs.  Because correction is disabled in C,
B and C should consequently be identical if the Stage-B reference path is
valid.

The audit runner implements the pairing and status gate at
`tools/jdv2_canonical_route_certification.py:289-297` and
`tools/jdv2_canonical_route_certification.py:355-482`.

## 4. Pairing and checkpoint contracts

All mandatory pairing invariants passed:

| Contract | Result |
|---|---:|
| Batch identity | PASS |
| Input tensors | PASS |
| Candidate IDs | PASS |
| Joint goals | PASS |
| Contexts | PASS |
| Literal shared relation state | PASS |
| Post-encode RNG state | PASS |
| Post-diffusion RNG state | PASS |
| 20/20 candidate coverage | PASS |

Checkpoint-state verification:

| Check | Result |
|---|---:|
| State-dict key sets equal | PASS |
| Changed tensors | 22 |
| Changed corrector tensors | 22 |
| Changed non-corrector tensors | 0 |
| Stage-A final corrector weight nonzeros | 0 |
| Stage-A final corrector bias nonzeros | 0 |

Thus the Stage-A corrector returns an exact mathematical zero, while its
internal layers need not be zero.

## 5. First-divergence localization

The trace replay was itself checked tensor-exactly against both production A
and production B.  Therefore the reported operator boundary reproduces the
actual route rather than an approximate reimplementation.

### First dtype divergence

| Field | Value |
|---|---|
| Branch | 0 |
| Diffusion timestep | 30 |
| Operator | `epsilon_base_plus_zero` |
| A dtype | `torch.float32` |
| B dtype | `torch.bfloat16` |
| Max absolute value difference at this operator | 0.0 |
| Numerically differing elements | 0 / 48 |

The values are initially equal, but the dtype is already no longer identical.

### First value divergence

| Field | Value |
|---|---|
| Branch | 0 |
| Diffusion timestep | 30 |
| Operator | `second_term` |
| A dtype | `torch.float32` |
| B dtype | `torch.bfloat16` |
| Max absolute difference | 0.0004484504461288452 |
| Differing elements | 48 / 48 |

This sequence directly confirms:

```text
BF16 epsilon + FP32 exact-zero
  -> dtype promotion to FP32
  -> different DDIM multiplication/rounding
  -> different x_next
  -> persistent trajectory divergence
```

## 6. Full trajectory parity

The validation split contains 47 E=0 windows and 92 E>0 windows.  Of the E>0
windows, 30 mix degree-positive and degree-zero agents; 62 have all agents
active.

| Comparison | Scope | Different elements | Compared elements | Nonidentical windows | Max abs diff |
|---|---|---:|---:|---:|---:|
| A vs B | all agents | 139,675 | 176,640 | 92 | 0.0108813047 |
| A vs B | degree-positive agents | 139,675 | 139,680 | 92 | 0.0108813047 |
| A vs B | inactive/degree-zero agents | 0 | 36,960 | 0 | 0.0 |
| B vs C | all agents | 0 | 176,640 | 0 | 0.0 |
| B vs C | degree-positive agents | 0 | 139,680 | 0 | 0.0 |
| B vs C | inactive/degree-zero agents | 0 | 36,960 | 0 | 0.0 |

The localization is exact:

- E=0 and degree-zero paths already preserve Stage-A identity;
- every E>0 window diverges between canonical A and literal B;
- the Stage-B reference C is tensor-exactly the same as literal B.

## 7. Single-seed validation metrics

These metrics are diagnostic consequences of the paired seed-2035 route
comparison.  They are not a replacement for the existing five-seed final
results.

### Overall

| Path | minADE@K | minFDE@K | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| A canonical | 0.279323 | 0.388427 | 0.421200 | 0.693689 | 0.692777 | 0.478072 | 0.296992 |
| B literal-off | 0.279302 | 0.388470 | 0.421198 | 0.693709 | 0.692777 | 0.478072 | 0.296951 |
| C Stage-B reference | 0.279302 | 0.388470 | 0.421198 | 0.693709 | 0.692777 | 0.478072 | 0.296951 |

A minus B relative changes are small but nonzero: minADE `+0.007703%`,
minFDE `-0.011194%`, JADE `+0.000561%`, JFDE `-0.002894%`, and relative
motion `+0.013691%`.  Goal-only metrics are identical because candidate IDs
and goals are unchanged.

### E=0

All seven metrics are exactly identical between A, B, and C.  In particular,
minADE/minFDE are `0.369579/0.508754` and JADE/JFDE are
`0.369579/0.508754`.

### E>0

| Path | minADE@K | minFDE@K | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| A canonical | 0.266108 | 0.370809 | 0.447572 | 0.788166 | 0.793689 | 0.722304 | 0.448716 |
| B/C literal reference | 0.266083 | 0.370858 | 0.447568 | 0.788197 | 0.793689 | 0.722304 | 0.448654 |

The metric effect is much smaller than the tensor-level identity failure.  A
small aggregate metric delta cannot certify a numerical identity contract.

## 8. Independent relation recomputation diagnostic

Before enforcing one literal common relation-state input, A and C also
independently recomputed the same relation embedding.  Candidate IDs, goals,
contexts, and RNG states remained equal, but 5 windows showed very small
floating differences: 19,862 / 137,920 values differed, with maximum absolute
difference `7.450580596923828e-08`.

This is reported transparently but is not the certified route divergence:

- the common-input certification supplies the exact same Stage-A relation
  tensor to A/B/C;
- relation state is not read by B or C when correction is disabled;
- B and C trajectories are nevertheless tensor-exact.

The small independent CUDA recomputation difference should not be cited as a
Stage-B effect or as evidence about the relation model.

## 9. Interpretation and certification boundary

The audit establishes four facts:

1. The canonical Stage-A config enables a structurally active corrector in
   E>0 windows even though the frozen final output layer makes its residual
   exactly zero.
2. Exact zero is not a numerical no-op under the real CUDA/BF16 arithmetic.
3. The Stage-B literal reference path is a valid Stage-A diffusion identity:
   B and C are tensor-exact under paired inputs and noise.
4. Existing canonical Stage-A and Stage-B-reference metrics were therefore
   not generated through exactly the same numerical route, although the
   observed single-seed metric delta is very small.

This does **not** invalidate the Stage-A architecture, sampler, checkpoint, or
the scientific joint-goal gains.  It does mean the path identity claim must
not be made for the current canonical Stage-A config.

## 10. Required next review decision

No automatic fix is made here.  The minimum next change, if approved, should
be restricted to establishing this semantic rule:

> A Stage-A checkpoint whose corrector output layer is tensor-exact zero must
> bypass correction arithmetic and execute the literal Stage-A DDIM path,
> independent of residual-projection mode.

After that change, the mandatory acceptance sequence should be:

1. repeat the A/B/C tensor-exact certification;
2. require A == B == C for all 176,640 trajectory elements;
3. rerun Stage-A and Stage-B paired final evaluation under the certified
   common reference;
4. preserve all historical JSON/report artifacts as historical records rather
   than silently overwriting them.

Until that review is approved, keep the frozen configs, checkpoints, existing
metrics, SDD run, and Stage-B status unchanged.

## 11. Validation executed

Audit command:

```bash
PYTHONPATH=. /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python \
  tools/jdv2_canonical_route_certification.py \
  --stage-a-config configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml \
  --stage-b-config configs/joint_dependency_v2/jdv2_stage_b_v2a_eth.yaml \
  --stage-a-checkpoint outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt \
  --stage-b-checkpoint outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_b_v2a_eth_seed2035/saved_models/best_model.pt \
  --output outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/canonical_stage_a_route_certification/results.json \
  --device cuda:0 --seed 2035
```

The run completed with exit code 0 in 271.19 seconds and peak audit-process
CUDA allocation of 2,637,812,736 bytes.  Focused audit tests passed `5/5`.
Repository validation also passed: `python -m compileall -q .`, `497 passed`
under `pytest -q`, and `git diff --check`.
