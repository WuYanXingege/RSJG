# RSJG Joint Dependency V2 — Frozen Final Implementation Decisions

## Status

The design review is accepted. The decisions in this document are frozen for
the first source implementation of Joint Dependency V2. They resolve the
blockers raised by `IMPLEMENTATION_PLAN_REVIEW.md` and take precedence over
earlier unresolved alternatives.

This document authorizes planning only. Source implementation still requires a
separate explicit instruction.

## 1. Exact all-off GDTS passthrough

Define:

```text
jdv2_active =
    use_scene_latent
    or use_dynamic_relation
    or use_joint_energy
    or use_dependency_corrector
```

If `goal_model_type=joint_dependency_v2` and `jdv2_active=False`:

- instantiate no V2 modules;
- consume no V2 RNG;
- use the legacy cache and legacy augmentation;
- use `_legacy_encode()` and original TTST;
- use original GDTS diffusion, losses, metrics, and checkpoint selection;
- strict-load the legacy checkpoint;
- reject all V2-specific training stages.

The same checkpoint/input/seed/environment must produce the same output as
original GDTS.

## 2. Frozen symbols

```text
C = scenes
N = agents
E = sparse canonical edges
Z = 4 scene modes
M = 4 relation modes
K = 21 goal candidates
P = 20 joint samples
T = 12 prediction steps
R = 8 energy rank
```

## 3. Scene posterior

Future sequence input:

```text
[position relative to last observation, future velocity]  # 4 channels
```

Future encoder:

```text
Linear(4,64) -> SiLU -> GRU(64,128)
```

Pool future-agent features invariantly per scene. Fuse:

```text
concat(history_scene [128], future_scene [128]) -> [C,256]
Linear(256,128) -> SiLU -> LayerNorm(128) -> Linear(128,4)
q(z|X,Y*) = softmax(output)
```

## 4. Joint-goal training objective

Use two structured pseudo-likelihood paths:

```text
L_JG =
    0.5 * L_PL_post
    + 0.5 * L_PL_prior
    + beta_z * KL(q_z || p_z)
    + beta_r * L_relation_KL
```

- `L_PL_post` uses `q_z` and `q_relation`.
- `L_PL_prior` uses `p_z` and deployable `p_relation`.
- `beta_z` and `beta_r` warm linearly from 0 to 0.1 over the first
  20% of Stage-A (`joint_goal`) training, then remain 0.1.

## 5. Relation teacher

Per canonical edge, the future pair descriptor is:

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

Teacher network:

```text
Linear(6,64) -> SiLU -> Linear(64,64) -> SiLU -> Linear(64,4)
q_relation = softmax(output)
```

Relation distillation:

```text
L_relation_KL =
  sum_e sum_z q_z(z)
    sum_(k,l) q_i*(k) q_j*(l)
      KL(
        q_relation(e)
        || p_relation(e,z,k,l)
      )
```

## 6. Dynamic relation geometry

Let:

```text
d_i = g_i - x_i_last
d_j = g_j - x_j_last
```

Use exactly four symmetric scalars:

```text
phi1 = ||g_j-g_i|| / graph_radius

phi2 = cos(d_i,d_j),
       or 0 if either norm < eps

phi3 = ||(g_j-g_i) - (x_j_last-x_i_last)|| / graph_radius

phi4 = abs(||d_i|| - ||d_j||) / graph_radius
```

All are invariant to endpoint reversal.

Geometry network:

```text
Linear(4,32) -> SiLU -> Linear(32,16)
```

Learn `Q [Z,M,16]` and `B [Z,M]`:

```text
delta_logit(z,m,k,l)
  = dot(Q[z,m], geometry_embedding(k,l)) / sqrt(16) + B[z,m]

relation_logit(e,z,k,l,m)
  = base_relation_logit(e,m) + delta_logit(z,m,k,l)

p_relation = softmax_m(relation_logit)
```

## 7. Relation-specific joint energy

Do not reuse the unary embedding containing `log_prior`.

Prior-free candidate inputs:

```text
source:      [g_i-x_i_last, g_i-x_j_last] -> R^4
destination: [g_j-x_j_last, g_j-x_i_last] -> R^4
```

Shared candidate encoder:

```text
Linear(4,64) -> SiLU -> Linear(64,64)
```

Oriented pair context:

```text
[h_receiver, h_neighbor, oriented_edge_feature]
Linear(270,128) -> SiLU -> Linear(128,64)
```

Use the same pair encoder after edge reversal. Add learned scene-mode and
relation embeddings in `R^64`. Apply the shared factor head:

```text
SiLU -> Linear(64,8)
left/right factors [E,Z,M,K,8]
```

Energy:

```text
E_ij^{z,m}(k,l)
  = -dot(F_i,F_j) / sqrt(8)
```

Effective energy:

```text
Eeff_ij^z(k,l)
  = -logsumexp_m(
      log p_relation(m|X,z,k,l) - E_ij^{z,m}(k,l)
    )
```

Joint and conditional scores:

```text
Score_z(G)
  = sum_i u_i(k_i)
    - sum_(i,j in canonical edges) Eeff_ij^z(k_i,k_j)

score_i^z(k)
  = u_i(k)
    - sum_(j in neighbors(i)) Eeff_ij^z(k,k_j)
```

Each canonical edge is counted once in the joint score. Use no degree-mean
normalization. The energy coefficient is 1.

## 8. Dependency-corrector geometry

Keep the existing GDTS velocity semantics. For current noisy world velocity:

```text
p_i,tau = x_i_last + dt * cumulative_sum(v_i,1:tau)
relative position = p_j-p_i
relative velocity = v_j-v_i
```

Use world metres and metres/second. Do not use an `x0` estimate in V1.

## 9. Shared tree trunk

Do not apply dependency correction on the common GDTS trunk. Run the trunk
unchanged. After the existing branch split, branch `p` uses its own:

```text
z^(p), G^(p), R^(p)
```

and applies its branch-specific dependency corrector. Do not duplicate full
diffusion chains.

## 10. Trajectory-stage teacher curriculum

For `joint_trajectory`:

```text
0-20%:   teacher probability = 1
20-60%:  linearly decay teacher probability 1 -> 0
60-100%: teacher probability = 0
```

- teacher condition: GT-near goals plus `q_relation`;
- deployable condition: sampled joint goals plus `p_relation`.

Canonical `joint_finetune` uses deployable conditioning.

## 11. Memory execution

Training full `K^2` path:

- default `edge_chunk_size=256`;
- chunk over sparse edges;
- accumulate losses incrementally in FP32;
- release chunk intermediates.

Inference sampler:

- selected-neighbor representation `[E,P,K,M]`;
- vectorize over `P=20` and `K=21`;
- only two refinement rounds may use a Python loop.

Corrector:

- follow existing branch execution;
- never retain all 20 branch hidden states.

Add a crowded-scene stress test.

## 12. Cache policy

Use an explicit `build-jdv2-cache` phase.

Cache only deterministic/frozen artifacts:

- synchronized scene metadata;
- parameter-free sparse graph;
- goal candidates;
- candidate log priors;
- deterministic GT scene and pair teacher descriptors.

Do not cache trainable-network outputs.

Manifest fields:

- dataset and split hashes;
- goal-checkpoint file SHA256;
- graph configuration;
- coordinate units;
- `dt`;
- `K`;
- schema version.

## 13. AMP

- BF16: no `GradScaler`.
- FP16: use `GradScaler`.

Always use FP32 for:

- softmax/log-softmax;
- logsumexp;
- KL;
- distance/cosine-sensitive calculations;
- energy accumulation;
- `index_add_` sums;
- multinomial probabilities;
- final reductions.

## 14. Stage checkpoint selection

- `joint_goal`: primary metric `JFDE`.
- `joint_trajectory`: primary metric `JADE`, tie-break `JFDE`.
- `joint_finetune`: primary metric `JADE`.

The base GDTS denoiser remains frozen in every V2 stage.

## 15. Checkpoint payload

V2 checkpoints may additionally store:

- optimizer;
- scheduler;
- `GradScaler` when applicable;
- architecture config;
- ablation config;
- cache hash;
- source checkpoint hash;
- stage/schedule progress.

Legacy GDTS checkpoint loading remains unchanged.

## 16. Implementation authorization gate

These decisions resolve the accepted review. No source implementation begins
until the user explicitly authorizes it.
