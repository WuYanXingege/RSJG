# RSJG Joint Dependency V2 — Stage-A Freeze

## Status

**`STAGE_A_FROZEN`**

Stage A is frozen at source commit
`ad35a440695a4b4eb23ad3e900eb27af9b3b8909` on branch
`research/joint-dependency-v2-clean`. This freeze authorizes no training and no
Stage-B implementation. The only eligible next activity is
`STAGE_B_DESIGN_REVIEW`.

Canonical configuration:
`configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml`.

Machine-readable contract:
`outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_a_freeze/manifest.json`.

## 1. Final Stage-A architecture

The frozen method is `joint_dependency_v2`, variant `strict_no_z`:

```text
history X
  -> frozen Goal U-Net candidate bank (K=21)
  -> SocialMotionEncoder
  -> p(r_ij | X, g_i, g_j), M=4
  -> sparse relation-specific low-rank goal energy, rank=8
  -> P=20 joint goal worlds
  -> frozen GDTS goal-relative history encoder and diffusion tree
```

There is no global categorical scene latent: no `p(z|X)`, `q(z|X,Y*)`,
`gamma`, scene embedding, scene-mode sampling, or z-indexed relation/energy
branch is part of the frozen architecture.

The frozen Stage-A checkpoint is epoch 13:

```text
outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/
jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt
```

SHA256:
`699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`.
The checkpoint is referenced, not copied.

## 2. Frozen inference algorithm

The canonical policy is explicitly
`exact_lexicographic_persistent_tie`. The global parser default deliberately
remains `categorical` for old configurations and tensor-exact backward
compatibility; canonical evaluation must load the frozen config and must not
rely on that default.

Round 0 uses the existing weighted Gumbel-Top-P allocation without
replacement. For each of two synchronous refinement rounds:

- `degree == 0`: return the previous candidate IDs tensor-exactly;
- `degree > 0`: solve one injective assignment with strict exact objective
  `(J, C_stay, C_geom, R)`;
- `J` is the existing neighbor-conditioned conditional-score sum;
- `C_stay` preserves previous slot/candidate identity only within a primary
  tie;
- `C_geom` minimizes exact squared endpoint displacement in world metres only
  within a `(J,C_stay)` tie;
- `R` is an exchangeable fourth-level exact-equivalence resolver derived from
  `(evaluation seed, window, agent, namespace)` and is persistent across both
  rounds.

Finite FP32 scores and goal coordinates are lifted exactly to arbitrary-
precision integer objectives. No epsilon, temperature adjustment, candidate
index, slot index, or learned tie breaker defines the lexicographic order.
This is sample-set-level structured allocation of neighbor-conditioned
conditional compatibility scores; it is not a claim of globally maximizing a
multi-agent joint likelihood or compatibility objective.

The frozen conditional score is:

```text
S_i[s,k] = unary_i[k] - accumulated_pair_energy_i[s,k].
```

## 3. Exact Stage-A dataflow and interfaces

Let `N` be agents, `E` canonical sparse edges, `K=21`, `P=20`, observed
length 8, future length 12, and relation embedding width 16.

### Inputs

- `obs_traj_world`: FP32 `[N,8,2]`, world metres.
- `world_coord`: FP32 `[20,N,2]`; future values are training/audit targets and
  are not part of deployed inference.
- `scene_index`: int64 `[N]`; edges may never cross scene IDs.
- `scene_ptr`: int64 scene boundaries.
- `frame_ids`: int64 `[20,N]` for synchronized windows.
- `x_augmented`: FP32 `[20,N,*]`; map-coordinate positions are columns 6:8.
- frozen cache candidates `goal_candidates_map/world`: FP32 `[N,21,2]`.
- `candidate_log_prior`: FP32 `[N,21]`, with frozen candidate order.
- `edge_index`: int64 `[2,E]`, canonical `src < dst` pairs.
- `edge_feat`: FP32 `[E,14]`; `edge_weight`: FP32 `[E]`.

The sparse graph is the parameter-free `radius_ttc` graph with radius 6 m,
TTC threshold 8 s, and `dt=0.4 s`. The Goal U-Net candidate cache has schema
`jdv2-cache-v1`, K=21, `world_m` / `world_m_per_s` units, and fixed ordering.

### Internal frozen Stage-A contract

- social agent feature: FP32 `[N,128]`;
- base relation logits/probability: FP32 `[E,4]`;
- unary score: FP32 `[N,21]`;
- strict-no-z full dynamic relation: FP32 `[E,21,21,4]` when materialized;
- energy factors: `[E,4,21,8]` per side;
- refinement score: FP32 `[N,20,21]`;
- sampled candidate IDs: int64 `[N,20]`;
- sampled joint goals: FP32 `[N,20,2]`, world metres;
- selected relation probability: FP32 `[E,20,4]`;
- selected relation embedding: FP32 `[E,20,16]`.

### Frozen outputs consumed downstream

`_jdv2_encode` appends one internal heatmap-argmax trunk endpoint to the 20
deployed branches:

- `joint_candidate_index`: int64 `[N,21]`; columns 0:20 are the joint worlds
  and column 20 is the internal GDTS trunk only;
- `joint_goal_points_world`: FP32 `[N,21,2]`, world metres;
- `joint_goal_points_map`: FP32 `[N,21,2]`;
- goal-conditioned contexts: `[21,N,1,D]` (20 branches plus trunk);
- `dependency_state.edge_index`: int64 `[2,E]`;
- `dependency_state.edge_weight`: FP32 `[E]`;
- `dependency_state.relation_embedding`: FP32 `[E,20,16]`;
- `dependency_state.last_position_world/map`: FP32 `[N,2]`;
- final prediction: FP32 `[20,20,N,2]` in map coordinates, containing exactly
  20 evaluation samples over observation+prediction time. The trunk is never
  exposed as a 21st prediction sample.

Stage B may consume this detached interface. It may not reinterpret candidate
IDs, reorder world slots, or feed gradients through Stage A.

## 4. Frozen hyperparameters and provenance

| Item | Frozen value |
|---|---:|
| K candidates | 21 |
| P joint worlds | 20 |
| M relation modes | 4 |
| energy rank | 8 |
| refinement rounds | 2 |
| relation embedding | 16 |
| social feature width | 128 |
| trajectory dt | 0.4 s |
| precision | BF16 with existing FP32 islands |
| validation seeds | 2035–2039 |

Cache manifest identity:
`2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949`.
Frozen legacy GDTS source checkpoint SHA256:
`126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950`.

## 5. Frozen ETH reference metrics

These are five-seed means on the exact-protocol ETH validation split (139
windows), seeds 2035–2039. All metrics are lower-is-better.

| Metric | Frozen production reference | Relative vs exact-protocol GDTS |
|---|---:|---:|
| minADE | 0.275516 | -3.778% |
| minFDE | 0.382648 | -2.981% |
| JADE | 0.413199 | -11.706% |
| JFDE | 0.702125 | -13.899% |
| Joint Goal Endpoint Error | 0.689827 | -13.736% |
| Compatibility | 0.487309 | -27.756% |
| Relative Motion Error | 0.301813 | -20.674% |

Every observed Round-0, Round-1, Round-2, and final agent candidate set must
retain 20/20 unique candidates. No seed was selected after the fact.

## 6. Production validation evidence

The authoritative artifact is:

```text
outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/
exact_persistent_refinement_production_validation/results.json
```

SHA256:
`89856d15ce8829aa4dac1d97f695e7482b9c9bfd55af00347b5db05508ba3574`;
status: `ADOPTION_VALIDATION_PASSED`.

Production/reference parity was exact for 695/695 Round-0 windows, 920/920
rounds, 2910/2910 interacting assignments and all exact objectives, 300/300
degree-zero identity events, 695/695 final world sets, and 5050/5050 20-of-20
coverage checks. The prediction profile equaled approved audit replica 0 with
maximum absolute error zero. Global and paired diffusion RNG checks passed.
The full-forward runtime change relative to D0 was -1.11%, within the +20%
gate. The full validation suite at adoption time reported 381 passed.

The freeze review hash-checks this evidence rather than rerunning the costly
five-seed validation without cause.

The final freeze-review gates passed with `python -m compileall -q .`,
`375 passed, 12 skipped` in the full test suite, `6 passed` in the dedicated
freeze-contract suite, `2 passed` in the real CUDA/BF16 canonical-policy
smoke, and a clean `git diff --check`.

## 7. Known non-blocking behaviors

1. **Synchronous two-cycle: approximately 40.66%.** This is
   `KNOWN_NON_BLOCKING_BEHAVIOR`: it preserves 20/20 coverage, did not amplify
   during production integration, and passed the semantic and prediction
   gates. It must be monitored, not “fixed” during this freeze.
2. **Historical D0 CUDA FP32 recomputation drift: max approximately
   0.001019.** This is an upstream CUDA reduction/recomputation difference,
   not a tolerated assignment or sampler semantic mismatch. Exact parity is
   defined on the same canonical score snapshot.
3. **Sampler-only runtime exceeds D0.** The exact production sampler is more
   expensive than D0, but the relevant end-to-end runtime and memory gates
   passed. It is not a freeze blocker.

## 8. Regression contract

`tests/test_jdv2_stage_a_freeze_contract.py` pins config loading, the explicit
canonical policy, the categorical global default, strict-no-z dimensions,
checkpoint/adoption hashes, degree-zero identity, solver availability,
categorical/CPSR availability, and all-off parser semantics. Existing
integration tests continue to enforce the bitwise all-off GDTS passthrough,
exact solver semantics, RNG isolation, synchronous updates, checkpoint
compatibility, and CUDA/BF16 execution.

Required gates for any future change are:

```text
python -m compileall -q .
pytest
git diff --check
real CUDA/BF16 canonical-policy targeted test
```

A changed checkpoint, config, adoption artifact, interface, or method constant
must fail closed and requires an explicit unfreeze review.

## 9. Frozen versus mutable boundary

Frozen items are: strict-no-z architecture and epoch-13 weights; candidate
bank/order; SocialMotionEncoder; RelationInference; dynamic relation; low-rank
energy; conditional score; graph semantics; K/P/M/rank; Round 0; degree-zero
identity; exact four-level objectives and persistent `R`; two synchronous
rounds; Stage-A outputs; frozen Goal U-Net, GDTS history encoder, and base
diffusion denoiser.

Stage B may introduce and train only explicitly approved trajectory-level
dependency modules (including a Stage-B-local dependency corrector), with its
own optimizer, scheduler, progress, checkpoint, and diagnostics. It must load
Stage A read-only, detach its outputs, and preserve the original Stage-A
checkpoint bytes. It may not modify/backpropagate through Stage-A modules,
sampler, relation, energy, graph, candidates, Goal U-Net, history encoder, or
base diffusion denoiser unless a later review explicitly authorizes it.

The dependency-corrector namespace present in the Stage-A checkpoint is part
of checkpoint compatibility and is zero-safe/frozen during Stage A. Any
future trained corrector belongs to a separate Stage-B checkpoint; it must not
overwrite this Stage-A reference.

## 10. Future benchmark rules

HOTEL, UNIV, ZARA1, and ZARA2 must use the same architecture, candidate and
graph semantics, exact sampler, K/P/M/rank, and two-round policy. Dataset-
specific training/checkpoints are permitted only where the original protocol
requires them; such runs validate generalization and do not redefine the
method. ETH is the method-development/freeze dataset. Results on later
datasets must not be used to return to ETH and tune the Stage-A sampler.

Benchmark reports must state the config and checkpoint hashes, use a fixed
predeclared seed set, preserve deployed stochastic evaluation (not replace it
with MAP), use the same coordinate conversion and metric implementation, and
report both marginal and joint metrics.

## 11. Conditions for unfreezing Stage A

Unfreezing requires a separate explicit authorization and a documented hard
failure, such as a reproducible semantic/interface bug, checkpoint or cache
integrity failure, violation of permutation/RNG contracts, invalid numerical
output, or failure to run the frozen method on a required dataset for reasons
intrinsic to the method contract. A desire for better metrics, a new dataset,
the known two-cycle, D0 CUDA drift, sampler-only runtime, or Stage-B research
convenience is insufficient.

Any authorized unfreeze must create a new version/config/checkpoint, rerun all
affected regression and same-protocol validation gates, and preserve this
manifest as the immutable historical Stage-A reference.

## 12. Freeze decision

All authoritative semantic, prediction, regression, RNG, CUDA/BF16, and
runtime evidence supports the freeze. The categorical global default remains
solely for backward compatibility; the canonical config removes ambiguity by
explicitly selecting `exact_lexicographic_persistent_tie`.

Final state: **`STAGE_A_FROZEN`**.

Next and only eligible step: **`eligible_for_stage_b_design_review`**.
