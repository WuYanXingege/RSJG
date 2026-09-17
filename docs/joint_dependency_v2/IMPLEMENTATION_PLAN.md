# RSJG Joint Dependency V2 — Final Implementation Plan

## 0. Scope, authority, and status

This is the resolved implementation plan for RSJG Joint Dependency V2. It is
based on:

1. `RSJG_Joint_Dependency_V2_THEORY.md`;
2. `RSJG_Joint_Dependency_V2_Final_Implementation_Specification.md`;
3. `RSJG_Joint_Dependency_V2_Codex_OneShot_Prompt.txt`;
4. `IMPLEMENTATION_PLAN_REVIEW.md`;
5. the accepted frozen decisions in `FINAL_IMPLEMENTATION_DECISIONS.md`.

The frozen decisions take precedence wherever the earlier plan or review left
an implementation detail unresolved.

This phase changes documentation only. No source or test code is implemented.

The target pipeline remains:

```text
History X
  -> unchanged SocialMotionEncoder and sparse graph
  -> scene latent z
  -> hypothesis-conditioned relation r_ij
  -> sparse low-rank joint goal energy
  -> structured joint sampling
  -> unchanged frozen GDTS denoiser
  -> branch-specific dependency residual
```

Dependency correction is part of each branch denoising update, not a post-hoc
trajectory edit:

```text
epsilon_joint(t) = epsilon_base(t) + Delta_epsilon(t)
```

## 1. Frozen symbols and dimensions

Use these symbols consistently in code, docstrings, assertions, and tests:

| Symbol | Meaning | Canonical value |
|---|---|---:|
| `C` | scenes in a tensor/pack | variable |
| `N` | agents | variable |
| `E` | canonical sparse edges | variable |
| `Z` | scene modes | 4 |
| `M` | relation modes | 4 |
| `K` | goal candidates per agent | 21 |
| `P` | returned joint samples | 20 |
| `T` | predicted timesteps | 12 |
| `R` | energy rank | 8 |
| `D_h` | agent feature dimension | 128 |
| `D_c` | dependency hidden dimension | 64 |

Do not use `S` ambiguously for both scenes and samples.

The canonical V2 configuration must validate the fixed values rather than
silently constructing a different architecture.

## 2. Current repository structure

### 2.1 Main execution path

The entry point is `main.py`:

```text
main.py
  -> src.parser.main_parser()
  -> src.data_pre_process.Trajectory_Data_Pre_Process(args)
  -> src.trainer.trainer(args)
  -> src.models.model.GDTS
  -> train / test / train_test
```

An explicit `build-jdv2-cache` phase will be added for active V2. It runs only
after the normal synchronized scene cache exists and uses a frozen goal
checkpoint to build checkpoint-hashed candidate artifacts.

### 2.2 Existing GDTS trajectory generator

`src/models/model.py::GDTS` contains:

- `goal_module`: `src/models/model_utils/U_net_CNN.py::UNet`;
- `encoder`: goal-relative history encoder in
  `src/models/model_utils/hist_traj_rnn_encoder.py`;
- `diffnet`: `src/models/diffusion.py::TransformerConcatLinear`;
- `var_sched`: `src/models/diffusion.py::VarianceSchedule`;
- the active tree sampler in `GDTS.ts_sample()`.

Original inference is:

```text
goal heatmap -> TTST goals -> goal-relative history context
  -> shared GDTS diffusion trunk -> per-goal branches
  -> integrate future velocities into positions
```

The public trajectory layout is `[P, obs_length+T, N, 2]`.

### 2.3 Existing social and goal infrastructure

Reuse these implementations:

- `src/models/interaction_graph.py::SparseInteractionGraph` for canonical
  same-scene radius/TTC edges and 14-D edge features;
- `src/models/social_encoder.py::SocialMotionEncoder` unchanged, producing
  `h_i in R^128`;
- `src/models/relation_inference.py::RelationInference` for history-only base
  relation logits;
- `src/models/model_utils/sampling_2D_map.py::generate_goal_candidates` for
  the fixed 21 candidates and heatmap priors;
- finite-mask and scene-index utilities from `src/models/joint_goal.py`;
- relation-energy numerical helpers from `src/models/goal_energy.py` only when
  their exact tensor/sign contract matches V2.

Do not reuse `LowRankJointGoal` or `LowRankGoalEnergy` as the V2 model: their
architectures and relation assumptions differ from the frozen V2 design.

### 2.4 Dataset and cache pipeline

`src/data_pre_process.py` builds either:

- legacy independently grouped batches for original GDTS; or
- exact synchronized variable-`N` scene windows for social models.

The synchronized format already contains `abs_pixel_coord`, `seq_list`,
`scene_index`, `scene_ptr`, synchronized `frame_ids`, semantic/RGB tensors, and
trajectory maps. `GDTS.prepare_inputs()` converts coordinates to world metres
and exposes `obs_traj_world [N,T_obs,2]`.

`src/data_grouping.py`, `src/batch_cache_io.py`, and `src/data_loader.py` own
cache naming, manifests, serialization, and loading.

### 2.5 Training, losses, and evaluation

`src/trainer.py` owns data loading, model construction, checkpoint loading,
optimizers, schedulers, gradient clipping, validation, and evaluation.

Current baseline losses are goal-map BCE and diffusion-noise MSE. Current
structured helpers live in `src/joint_goal_loss.py`.

`src/metrics.py` already provides marginal ADE/FDE, minADE/minFDE, joint
JADE/JFDE, collision rate, and joint-goal metrics. Official JMM export/scoring
is implemented by `tools/evaluate_jmm_official.py` and `src/jmm_protocol.py`.

## 3. Exact all-off GDTS passthrough

Define once during argument normalization:

```text
jdv2_active = (
    use_scene_latent
    or use_dynamic_relation
    or use_joint_energy
    or use_dependency_corrector
)
```

If `goal_model_type=joint_dependency_v2` and `jdv2_active=False`, use an exact
legacy passthrough:

- instantiate no V2 modules;
- consume no V2 initialization or sampling RNG;
- resolve to the legacy batch cache;
- use legacy augmentation behavior;
- do not require synchronized batch-format-v2 fields;
- use `_legacy_encode()`;
- use original TTST;
- use original GDTS diffusion, loss coefficients, metrics, best metric, and
  official-export behavior;
- strict-load the legacy checkpoint;
- use the original model parameter namespace.

V2-specific stages `joint_goal`, `joint_trajectory`, and `joint_finetune` must
raise a parser/configuration error when `jdv2_active=False`. Baseline training
is permitted only through the original baseline semantics.

The output directory may remain namespaced for audit purposes, but computation,
random draws, predictions, losses, metric names, and metric values must equal
original GDTS for the same input, checkpoint, seed, device, and deterministic
environment.

All later sections apply only when `jdv2_active=True`.

## 4. Active V2 information flow

```text
observed history X in world coordinates
  -> parameter-free sparse graph
  -> unchanged SocialMotionEncoder -> h [N,128]

goal U-Net from frozen goal checkpoint
  -> heatmap p0
  -> cached/generated candidates [N,K,2]
  -> candidate log priors [N,K]

h, scene_index
  -> scene prior p_z [C,Z]

future teacher descriptors (training only)
  -> scene posterior q_z [C,Z]
  -> relation teacher q_relation [E,M]

h, candidates, log p0
  -> unary u [N,K]

h, edges, z, candidate pairs
  -> p_relation [E,Z,K,K,M]
  -> factors left/right [E,Z,M,K,R]
  -> Eeff [E,Z,K,K]

p_z, unary, selected-neighbor relation/energy
  -> sampled modes [C,P]
  -> joint candidate indices [N,P]
  -> joint goals [N,P,2]
  -> selected relation embedding [E,P,16]

joint goals + observed history
  -> unchanged goal-relative GDTS contexts
  -> unchanged common trunk, no dependency correction
  -> branch-specific frozen epsilon_base + Delta_epsilon
  -> trajectories [P,obs_length+T,N,2]
```

Teacher tensors are never inputs to validation/test inference.

## 5. New model modules

Create:

```text
src/models/joint_dependency_v2/
  __init__.py
  scene_latent.py
  future_teacher.py
  unary_goal.py
  dynamic_relation.py
  joint_energy.py
  joint_sampler.py
  dependency_corrector.py
```

No additional temporal Transformer, dense global attention block, or new
trajectory generator is allowed.

### 5.1 `scene_latent.py`

Input:

- agent features `h [N,128]`;
- `scene_index [N]`.

Prior:

```text
attention score:  Linear(128,64) -> SiLU -> Linear(64,1)
alpha_i:           softmax within each scene
history_scene:     sum_i alpha_i h_i                 [C,128]
count feature:     log(1+N_scene)                    [C,1]
scene network:     Linear(129,128) -> SiLU
                   -> Linear(128,128) -> LayerNorm
prior head:        Linear(128,4)
p_z:               softmax in FP32                  [C,Z]
```

Return prior logits/log probabilities/probabilities, `history_scene`, compact
scene indices, and attention weights. Pooling must use permutation-invariant
segment operations.

If `use_scene_latent=False` in a partial ablation, do not train/use the prior or
posterior. Use no mode sampling, zero scene embedding in energy, and a
z-independent dynamic-relation residual obtained by averaging `Q` and `B`
across the four stored modes. This removes scene-mode information without
changing the candidate/relation architecture.

### 5.2 `future_teacher.py`

This module is training-only.

#### Scene posterior

The future per-timestep input has four channels:

```text
[position relative to last observation (2), future velocity (2)]
```

Network:

```text
Linear(4,64) -> SiLU -> GRU(64,128)
```

Pool final future-agent features per scene with a permutation-invariant mean.
Fuse with the history scene feature:

```text
concat(history_scene [128], future_scene [128]) -> [C,256]
Linear(256,128) -> SiLU -> LayerNorm(128) -> Linear(128,4)
q_z = softmax(logits in FP32)                      [C,Z]
```

#### Relation teacher

For each canonical sparse edge, compute the deterministic future descriptor:

```text
[
  min_distance,
  normalized_time_of_min_distance,
  final_relative_x,
  final_relative_y,
  formation_change_x,
  formation_change_y,
]
```

All spatial values use world metres. Normalized time lies in `[0,1]` over the
12 prediction steps. The relative convention is destination minus source;
reversal negates signed relative/formation channels and retains distance/time.
Evaluate both orientations with the shared teacher MLP and average logits so
the final physical-edge posterior is reversal invariant:

```text
Linear(6,64) -> SiLU -> Linear(64,64) -> SiLU -> Linear(64,4)
q_relation = softmax(averaged logits in FP32)       [E,M]
```

For `E=0`, return correctly shaped empty tensors and differentiable zero
relation losses.

### 5.3 `unary_goal.py`

Candidate feature:

```text
[g_x-x_last, g_y-y_last, ||g-x_last||, log_prior]   [N,K,4]
```

Network:

```text
goal encoder:   Linear(4,64) -> SiLU -> Linear(64,64)
residual input: concat(h_i [128], goal_hidden [64]) -> [N,K,192]
residual head:  Linear(192,64) -> SiLU -> Linear(64,1)
u_i(k):         log p0_i(k) + delta_u_i(k)          [N,K]
```

Zero-initialize the last residual layer. The first forward pass must satisfy
`u == log_p0` exactly in the active dtype.

This unary embedding contains `log_prior` and must **not** be reused by joint
energy. The energy module owns a prior-free candidate encoder.

The score is z-independent. Where the equations use `u_i^z(k)`, define it as a
broadcast view of `u_i(k)` across `Z`.

### 5.4 `dynamic_relation.py`

Reuse `RelationInference` for base history logits `[E,M]`.

For candidate goals `g_i,g_j` and last positions `x_i,x_j`, define:

```text
d_i = g_i - x_i
d_j = g_j - x_j

phi1 = ||g_j-g_i|| / graph_radius

phi2 = cos(d_i,d_j),
       or 0 when either displacement norm < eps

phi3 = ||(g_j-g_i) - (x_j-x_i)|| / graph_radius

phi4 = abs(||d_i|| - ||d_j||) / graph_radius
```

All calculations run in FP32 world metres. All four scalars are invariant to
endpoint reversal.

Geometry network:

```text
Linear(4,32) -> SiLU -> Linear(32,16)
```

Learn:

```text
Q [Z,M,16]
B [Z,M]
```

Relation residual and final logits:

```text
delta_logit(z,m,k,l)
  = dot(Q[z,m], geometry_embedding(k,l)) / sqrt(16) + B[z,m]

relation_logit(e,z,k,l,m)
  = base_relation_logit(e,m) + delta_logit(z,m,k,l)

p_relation = softmax_m(relation_logit in FP32)
```

APIs:

- full training path: `[E_chunk,Z,K,K,M]`;
- selected-neighbor inference path: `[E,P,K,M]`;
- selected joint hypotheses for corrector: `[E,P,M]`.

If `use_dynamic_relation=False`, broadcast the history-only base relation
probability across modes/candidates and disable relation KL.

### 5.5 `joint_energy.py`

Use a prior-free candidate representation.

For canonical edge `(i,j)`:

```text
source candidate input:
  [g_i-x_i, g_i-x_j]                             [E,K,4]

destination candidate input:
  [g_j-x_j, g_j-x_i]                             [E,K,4]

shared candidate encoder:
  Linear(4,64) -> SiLU -> Linear(64,64)
```

Pair context is orientation-aware:

```text
forward: [h_i, h_j, edge_feature_ij]              [E,270]
reverse: [h_j, h_i, reversed_edge_feature_ji]     [E,270]

shared pair encoder:
  Linear(270,128) -> SiLU -> Linear(128,64)
```

Learn scene-mode and relation embeddings in `R^64`. Fuse by addition, which
adds no new large module:

```text
left_hidden
  = forward_pair_context
    + source_candidate_hidden
    + scene_mode_embedding
    + relation_embedding

right_hidden
  = reverse_pair_context
    + destination_candidate_hidden
    + scene_mode_embedding
    + relation_embedding
```

Use the shared factor head:

```text
SiLU -> Linear(64,8)
left/right factors: [E,Z,M,K,R]
```

Relation-specific energy:

```text
E_ij^{z,m}(k,l)
  = -dot(F_i^{z,m}(k), F_j^{z,m}(l)) / sqrt(8)
```

Effective energy:

```text
Eeff_ij^z(k,l)
  = -logsumexp_m(
        log p_relation(m|X,z,k,l) - E_ij^{z,m}(k,l)
    )
```

Perform `logsumexp` and energy accumulation in FP32.

Joint score:

```text
Score_z(G)
  = sum_i u_i(k_i)
    - sum_(i,j in canonical edges) Eeff_ij^z(k_i,k_j)
```

Each canonical edge is counted once in the joint score. There is no degree
mean normalization and the default energy coefficient is exactly 1.

If `use_joint_energy=False`, return neutral zero pair energies and skip energy
loss/refinement without changing public sample layouts.

### 5.6 `joint_sampler.py`

This module has no trainable neural network.

Inference:

1. Sample `z [C,P]` from `p_z` once per scene and joint sample.
2. Gather scene modes to agents and edges.
3. Initialize candidate indices from unary score `[N,K]`.
4. Run exactly two synchronous conditional-refinement rounds.
5. In a round, score all local candidates against neighbor indices from the
   previous round:

```text
score_i^z(k)
  = u_i(k)
    - sum_(j in neighbors(i)) Eeff_ij^z(k,k_j)
```

6. Return candidate indices `[N,P]`, goals `[N,P,2]`, modes `[C,P]`, and
   relation probabilities/embeddings for selected pairs `[E,P,M/16]`.

Selected-neighbor tensors use `[E,P,K,M]`. Vectorize over `P=20` and `K=21`.
Only the fixed two-round loop may be a Python loop. If required by memory,
fixed tensor chunks over edges or samples are allowed; semantic one-agent or
joint-assignment loops are forbidden.

Never construct a Cartesian product over agents or any `K^N` object.

### 5.7 `dependency_corrector.py`

The existing GDTS diffusion state retains its velocity semantics. The base
denoiser continues to receive its original tensor/units. A separate FP32
geometry adapter exposes current noisy velocity in world metres/second.

For agent `i` and future time `tau`:

```text
p_i,tau = x_i,last + dt * cumulative_sum(v_i,1:tau)
```

For each directed view of canonical edge `(i,j)`:

```text
relative position = p_j - p_i                       [2]
relative velocity = v_j - v_i                       [2]
distance          = ||p_j-p_i||                     [1]
relation embedding                                  [16]
```

The corrector input is exactly 21 dimensions. Do not use an `x0` estimate in
V1. When the underlying GDTS state is in map displacement units, the adapter
must obtain world positions through the existing scene transform and finite
difference them by `dt`; this conversion affects corrector features only, not
the frozen base denoiser state.

Network:

```text
input:       Linear(21,64) -> SiLU -> Linear(64,64)
timestep:    FiLM on the 64-D hidden state
gate:        Linear(64,32) -> SiLU -> Linear(32,1)
value:       Linear(64,64) -> SiLU -> Linear(64,64)
aggregation: sum(alpha*v) / (sum(alpha)+eps)
output:      Linear(64,64) -> SiLU -> Linear(64,2)
```

Build both directed messages with shared weights. Accumulate numerator and
denominator with FP32 `index_add_`. Zero-initialize the final output layer.
For `E=0`, return exact zeros `[N,T,2]`.

## 6. Mathematical training contract

### 6.1 Soft ground-truth candidate targets

Build finite normalized soft targets `q_i*(k) [N,K]` from the true endpoint in
world metres, reusing the current soft-target helper where its behavior matches
V2. Invalid candidates receive zero mass.

The GT-near candidate used by the trajectory teacher condition is
`argmax_k q_i*(k)`.

### 6.2 Relation distillation

For edge `(i,j)`:

```text
L_relation_KL
  = sum_e sum_z q_z(z)
      sum_(k,l) q_i*(k) q_j*(l)
        KL(
          q_relation(e)
          || p_relation(e,z,k,l)
        )
```

Normalize by the number of valid edges; `E=0` yields differentiable zero.
Compute probabilities, logs, KL, and reductions in FP32.

### 6.3 Two pseudo-likelihood paths

Define sparse conditional pseudo-likelihood using the exact unary and
effective-energy score from Section 5.5.

- `L_PL_post` weights scene modes with `q_z` and uses teacher
  `q_relation [E,M]` broadcast across candidate pairs in the relation
  marginalization.
- `L_PL_prior` weights scene modes with deployable `p_z` and uses deployable
  `p_relation(e,z,k,l)`.

Both paths use soft target `q_i*`, sparse canonical edges, sum neighbor energy
without degree normalization, and accumulate edge chunks incrementally.

The Stage-A joint-goal objective is:

```text
L_JG
  = 0.5 * L_PL_post
    + 0.5 * L_PL_prior
    + beta_z * KL(q_z || p_z)
    + beta_r * L_relation_KL
```

This makes the posterior participate in the structured future objective and
prevents the KL-only future-agnostic solution identified in the review.

### 6.4 KL warm-up

In `joint_goal`, compute progress from completed optimizer steps after gradient
accumulation:

```text
warmup_fraction = min(stage_progress / 0.20, 1)
beta_z = 0.1 * warmup_fraction
beta_r = 0.1 * warmup_fraction
```

Thus both weights rise linearly from 0 to 0.1 over the first 20% of Stage-A
training and remain 0.1 afterward. `joint_finetune` uses the terminal value
0.1 unless an approved continuation checkpoint records a later explicit
schedule.

### 6.5 Dependency diffusion loss

For the current noised future velocity state:

```text
epsilon_base  = frozen_diffnet(x_t, beta, context)
Delta_epsilon = dependency_corrector(...)
epsilon_joint = epsilon_base + Delta_epsilon
L_diff        = MSE(epsilon_joint, sampled_noise)
```

The base denoiser is frozen and may run under `no_grad`; the corrector input
must remain available for its own gradients. Final MSE reduction is FP32.

## 7. Training stages and freeze policy

### 7.1 Stage A: `joint_goal`

Train:

- `SocialMotionEncoder`;
- scene prior and future posterior;
- unary residual;
- base/dynamic relation and relation teacher;
- joint energy.

Freeze and put in `eval()`:

- GDTS goal U-Net from the source goal checkpoint;
- GDTS goal-relative history encoder;
- GDTS denoiser;
- dependency corrector.

Objective: `L_JG`.

Primary validation metric: `JFDE`.

### 7.2 Stage B: `joint_trajectory`

Train only the dependency corrector. Freeze and put every upstream module,
including the full joint-goal stack, in `eval()`.

Teacher curriculum based on completed Stage-B optimizer steps:

```text
0%   <= progress < 20%: teacher probability = 1
20%  <= progress < 60%: linearly decay 1 -> 0
60%  <= progress <=100%: teacher probability = 0
```

For each batch draw one curriculum decision with the stage RNG:

- teacher condition: GT-near candidates plus `q_relation`;
- deployable condition: sampled joint goals plus corresponding
  `p_relation`.

Objective: dependency-corrected `L_diff`.

Primary validation metric: `JADE`; tie-break with `JFDE`.

### 7.3 Stage C: `joint_finetune`

Train the V2 goal-dependency modules and dependency corrector jointly. Keep the
GDTS goal U-Net, history encoder, and base denoiser frozen and in `eval()`.

Use deployable conditioning (`teacher probability=0`) for the corrector. The
goal stack receives gradients from `L_JG`; discrete sampled indices are not
expected to carry pathwise diffusion gradients.

Canonical objective:

```text
L_total = lambda_JG * L_JG + lambda_diff * L_diff
lambda_JG = 1
lambda_diff = 1
```

Both stage-level coefficients are explicit config values and are recorded in
the checkpoint; the canonical defaults are one. The internal coefficients of
`L_JG` are frozen as above.

Primary validation metric: `JADE`.

### 7.4 Parameter families

Expose disjoint optimizer families:

- `gdts_frozen_backbone`;
- `jdv2_goal_dependency`;
- `jdv2_corrector`.

Assert unique parameter ids. The base denoiser must have no trainable parameter
in any V2 stage. Stage changes must not be able to unfreeze it through the
legacy `finetune` policy.

## 8. Tree diffusion integration

### 8.1 Goal contexts

For active V2, create `P=20` branch goal contexts and the one internal trunk
context expected by the existing tree sampler. The trunk context is not a 21st
evaluation sample.

### 8.2 Shared trunk

Run the shared trunk exactly as original GDTS:

- no dependency corrector;
- no V2 relation embedding;
- unchanged denoiser, schedule, random draws, and update equation.

### 8.3 Branches

After the existing split point, branch `p` uses its own:

- sampled scene mode `z^(p)`;
- complete joint goal assignment `G^(p)`;
- selected relation distribution/embedding `R^(p)`.

Only branch denoising uses `epsilon_base + Delta_epsilon`. Execute branches in
the existing sequential order and release corrector intermediates per branch.
Do not duplicate 20 complete diffusion chains and do not retain all 20 branch
hidden states.

## 9. Cache design

### 9.1 Explicit phase

Add:

```text
--phase build-jdv2-cache
```

The phase reads synchronized scene windows, loads the frozen goal checkpoint,
and writes a V2 cache/sidecar atomically. Training and evaluation require a
matching complete manifest; they do not silently rebuild it.

### 9.2 Cached artifacts

Cache only deterministic or frozen artifacts:

- synchronized scene metadata;
- parameter-free sparse graph candidates/features;
- 21 goal candidates;
- candidate log priors;
- deterministic GT scene-teacher inputs;
- deterministic six-channel future pair descriptors.

Do not cache:

- `SocialMotionEncoder` output;
- scene prior/posterior output;
- relation logits/probabilities;
- energy factors;
- dependency-corrector state;
- any other trainable-network output.

If adaptive graph gating is enabled in a noncanonical experiment, cache only
the parameter-free candidate graph and recompute learned gates online.

### 9.3 Manifest

The manifest must contain:

- schema version;
- dataset identity and per-split source hashes;
- frozen goal checkpoint SHA256 computed from file bytes;
- graph type, radius, TTC threshold, and other graph configuration;
- coordinate units (`world_m`, `world_m_per_s` where applicable);
- `dt`;
- `K=21`;
- observation/prediction lengths and down factor;
- candidate extraction settings and deterministic seed/version;
- completion state for every split.

Candidate order is part of the schema. Cache mismatch, missing split,
non-finite tensor, wrong shape, or wrong hash is a hard error with an explicit
rebuild instruction.

### 9.4 Baseline separation

Active V2 uses its separately versioned synchronized raw/cache directories.
All-off V2 resolves to the untouched legacy GDTS cache. Existing V1 joint and
V4 cache paths remain unchanged.

## 10. Memory and vectorization plan

### 10.1 Training full `K^2` path

Default:

```text
edge_chunk_size = 256
```

For each sparse-edge chunk:

1. build `[E_chunk,Z,K,K,M]` relation logits/probabilities in FP32 where
   required;
2. build/gather relation-specific factors;
3. compute effective energy and the chunk contribution to PL/relation KL;
4. accumulate FP32 numerator/count reductions;
5. release chunk intermediates before the next chunk.

Do not retain the full-scene relation, geometry, and energy matrices
simultaneously.

### 10.2 Inference sampler

Use selected-neighbor tensors `[E,P,K,M]`, vectorized across all 20 samples and
21 candidates. Only the two refinement rounds may be a Python loop.

### 10.3 Corrector

Follow existing sequential branch execution. Corrector memory is per branch,
approximately `[2E,T,64]`, never `[P,2E,T,64]`.

### 10.4 Complexity guard

Allowed complexity:

```text
unary:                 O(N*K)
scene prior/posterior: O(N + C*Z)
chunked full loss:     O(E*Z*M*K^2)
selected refinement:  O(rounds*E*P*M*K*R)
corrector:             O(P*E*T*64), executed branch-sequentially
```

Forbidden:

- `K^N` enumeration;
- one candidate axis per agent;
- dense `[N,N,K,K]` tensors;
- loops over agents or joint assignments;
- dense global attention.

Add a crowded-scene stress test that records peak memory and confirms scaling
with `E`, not `K^N`.

## 11. AMP, FP16, and BF16

Add explicit options for AMP enablement and dtype.

- BF16: use autocast, no `GradScaler`.
- FP16: use autocast and `GradScaler`.
- CPU or AMP disabled: retain current FP32 behavior.
- Validate device support and fail clearly for unsupported requested dtype.

Always compute these in FP32:

- softmax and log-softmax;
- logsumexp;
- KL terms;
- distance, norm, and cosine-sensitive calculations;
- candidate log priors;
- low-rank dot products and energy accumulation;
- `index_add_` message numerators and denominators;
- multinomial probabilities;
- final loss/metric reductions.

For FP16, unscale gradients before clipping. Check every loss component,
weighted total, and gradients for NaN/Inf before optimizer step. Zero-residual
equality must be tested with AMP disabled, FP16, and BF16 where supported.

## 12. Checkpoints and selection

### 12.1 Payload

V2 checkpoints may additionally store:

- optimizer state;
- scheduler state;
- `GradScaler` state when FP16 is active;
- frozen architecture configuration;
- ablation configuration;
- V2 cache manifest hash;
- source checkpoint SHA256;
- stage and completed optimizer-step progress;
- teacher curriculum and KL warm-up progress.

Legacy GDTS checkpoint loading remains unchanged. Extra training-state fields
are optional on read; model-state compatibility is not optional.

### 12.2 Loading matrix

- legacy -> legacy: current strict behavior;
- legacy -> active V2 initialization: allow only an explicit allowlist of
  missing V2 keys; reject unexpected or shape-mismatched legacy keys;
- V2 resume/test: strict architecture, stage, ablation, and cache-hash checks;
- `joint_goal` -> `joint_trajectory`: strict shared architecture and validated
  source stage;
- full V2 -> a different ablation: reject unless a future explicit
  initialization mode is approved;
- all-off V2 -> legacy: strict legacy state dict, no V2 keys.

### 12.3 Stage checkpoint comparator

- `joint_goal`: minimize `JFDE`.
- `joint_trajectory`: minimize `JADE`; if equal within the configured numeric
  comparison tolerance, minimize `JFDE`.
- `joint_finetune`: minimize `JADE`.

Persist the comparator, primary value, tie-break value, epoch, and seed in the
evaluation protocol and checkpoint metadata.

## 13. Evaluation

Reuse existing metric definitions:

- ADE/FDE and minADE@20/minFDE@20;
- JADE/JFDE with one shared sample index per scene;
- collision rate when configured;
- goal minFDE/recall and joint-goal endpoint/compatibility.

Do not modify `src/metrics.py` unless implementation reveals a missing generic
helper; no new V2 metric is currently required.

Evaluation assertions:

- exactly 20 returned samples;
- sample `p` denotes one complete scene future;
- one sampled scene mode is shared by agents in that scene/sample;
- no per-agent sample reordering after generation;
- the internal trunk is excluded from metrics;
- no teacher descriptors enter inference;
- official JMM export retains its existing file/layout contract.

Ablation artifacts record all four switches, stage, cache hash, source
checkpoint hash, seed, AMP dtype, and fixed architecture.

## 14. Permutation invariance and zero-edge behavior

Permutation invariance/equivariance is preserved through:

- shared temporal encoders;
- within-scene attention and invariant pooling;
- canonical sparse edges;
- reversal-invariant dynamic-relation geometry;
- forward/reverse shared pair encoders;
- synchronous conditional refinement;
- shared directed corrector messages and normalized sum aggregation;
- no raw agent id/index as a feature.

Tests must permute agents, rebuild/reindex edges and cached metadata, invert the
permutation, and compare prior/posterior, unary, relation, energy, selected
goals, corrected epsilon, and final trajectories.

For `E=0`:

- relation tensors are correctly shaped empty tensors;
- relation/energy losses are differentiable zeros;
- sampler reduces to unary selection;
- corrector returns exact zero;
- all outputs remain finite.

## 15. Configuration and ablations

Add `goal_model_type=joint_dependency_v2` plus aliases. Add the three V2
training stages and explicit cache phase.

Add four switches, default true for the canonical full model:

- `use_scene_latent`;
- `use_dynamic_relation`;
- `use_joint_energy`;
- `use_dependency_corrector`.

Partial neutral behaviors:

- scene latent off: no p/q loss or sampling, zero scene embedding, z-independent
  averaged relation query;
- dynamic relation off: history-only base relation, no relation KL;
- joint energy off: unary-only sampling and no energy PL contribution;
- corrector off: exact zero residual.

The all-four-off case uses the special exact legacy passthrough in Section 3,
not the partial-neutral V2 path.

Add validated configuration for:

- fixed dimensions;
- `edge_chunk_size=256`;
- `dt`;
- AMP enable/dtype;
- stage loss coefficients;
- checkpoint state saving;
- frozen goal/source checkpoint;
- cache path/hash;
- deterministic cache seed/schema.

## 16. Source-file modification plan

### 16.1 New files

| File | Purpose |
|---|---|
| `src/models/joint_dependency_v2/__init__.py` | Stable exports for the seven V2 components. |
| `src/models/joint_dependency_v2/scene_latent.py` | Scene prior and invariant pooling. |
| `src/models/joint_dependency_v2/future_teacher.py` | Exact future scene posterior and relation teacher. |
| `src/models/joint_dependency_v2/unary_goal.py` | Zero-initialized unary residual. |
| `src/models/joint_dependency_v2/dynamic_relation.py` | Exact four-scalar hypothesis-conditioned relation. |
| `src/models/joint_dependency_v2/joint_energy.py` | Prior-free relation-specific rank-8 energy. |
| `src/models/joint_dependency_v2/joint_sampler.py` | Vectorized two-round structured sampler. |
| `src/models/joint_dependency_v2/dependency_corrector.py` | Branch-only zero-initialized epsilon residual. |
| `src/joint_dependency_v2_cache.py` | Explicit frozen/deterministic V2 cache builder and manifest validation. |

### 16.2 Existing files to modify

| File | Exact scope |
|---|---|
| `main.py` | Dispatch `build-jdv2-cache`; keep all existing phases unchanged. |
| `src/parser.py` | V2 model type/stages/cache phase, fixed dimensions, four switches, derived `jdv2_active`, all-off guards, chunk/AMP/cache/checkpoint options, validation and isolated output path. |
| `src/models/model.py` | Active-V2 construction/routing, exact all-off passthrough, teacher/goal state, branch-only corrector integration, freeze/optimizer families, diagnostics, strict compatibility. |
| `src/joint_goal_loss.py` | Add V2 soft-target PL-post/PL-prior, scene KL, relation KL, edge-chunk FP32 reductions; do not change existing call behavior. |
| `src/data_grouping.py` | Active-V2 synchronized cache namespace/manifest; all-off resolves to legacy cache. |
| `src/data_pre_process.py` | Produce deterministic synchronized metadata and teacher inputs needed by the explicit cache phase; no trainable outputs. |
| `src/data_loader.py` | Load/validate active-V2 cached artifacts; preserve legacy all-off behavior. |
| `src/batch_cache_io.py` | Reuse atomic pickle/zstd helpers; add only generic manifest/hash validation helpers if necessary. |
| `src/trainer.py` | Three V2 stages, freeze/eval policy, schedules, stage RNG, AMP dtype/scaler behavior, comparator/tie-break, extended V2 checkpoint payload and diagnostics. |
| `tools/evaluate_jmm_official.py` | Expected no algorithm change; only compatibility handling if the frozen V2 config fields require it. |

`src/models/diffusion.py`, `src/models/social_encoder.py`,
`src/models/interaction_graph.py`, `src/models/relation_inference.py`, and
`src/metrics.py` should remain behaviorally unchanged.

### 16.3 Tests to add or extend

- parser/all-off passthrough tests;
- module shape, finite, permutation, reversal, and `E=0` tests;
- exact geometry formula tests;
- relation teacher and posterior tests;
- energy sign/logsumexp/factor-shape tests;
- vectorized sampler and two-round update tests;
- corrector world-geometry, zero-init, trunk-bypass, and branch conditioning
  tests;
- PL-post/PL-prior, KL warm-up, and teacher-curriculum tests;
- cache SHA/schema/stale-checkpoint/atomic-write/leakage tests;
- legacy and V2 checkpoint matrix tests;
- FP16/BF16 precision-island and NaN tests;
- crowded-scene peak-memory stress test;
- same-process and end-to-end all-off baseline equivalence tests;
- active-V2 CPU integration and official JMM export smoke tests.

## 17. Implementation order and release gates

1. Parser, fixed config, `jdv2_active`, and exact all-off tests.
2. Explicit cache builder, manifest, and cache tests.
3. Scene prior and future teacher with unit tests.
4. Unary, dynamic relation, and exact geometry tests.
5. Relation-specific energy and chunked loss tests.
6. Vectorized sampler and complexity tests.
7. Corrector geometry, zero initialization, and branch-only tests.
8. Active V2 integration into `GDTS` without touching legacy methods.
9. Stage losses, schedules, freeze/eval policy, AMP, and checkpoints.
10. Evaluation comparators, diagnostics, and official export verification.
11. Full existing pytest suite.
12. CPU fast-debug runs for all three stages.
13. GPU FP16 and BF16 smoke tests where supported.
14. Crowded-scene memory stress test.
15. Fixed-seed baseline equivalence and official JMM release gates.

Implementation is accepted only if:

- original GDTS tests and fixed-seed outputs remain unchanged;
- all-off V2 matches original GDTS end to end;
- no `K^N` or dense `[N,N,K,K]` object exists;
- permutation/reversal and zero-edge tests pass;
- every stage keeps the base denoiser frozen;
- cache hashes and checkpoint compatibility are enforced;
- all FP32-sensitive operations remain finite under configured AMP;
- output contains exactly 20 coherent joint samples.

## 18. Documentation-phase completion state

All previous design-review blockers are resolved by the frozen decisions and
incorporated above. No source or test code has been modified. Implementation
must wait for a separate explicit authorization.
