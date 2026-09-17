# RSJG Joint Dependency V2 — Final Design Re-review

## 0. Final verdict

**Design status: approved and implementation-ready, pending a separate explicit
instruction to modify source code.**

The updated `IMPLEMENTATION_PLAN.md` was re-reviewed against:

- the theory document;
- the final implementation specification;
- the one-shot engineering requirements;
- every blocker in the first design review;
- `FINAL_IMPLEMENTATION_DECISIONS.md`.

Every previous blocking issue is resolved. No remaining design blocker
genuinely prevents implementation.

This re-review changes documentation only.

## 1. Architecture consistency

### 1.1 End-to-end order

The updated plan implements the required dependency chain correctly:

```text
History X
  -> unchanged sparse SocialMotionEncoder -> h_i
  -> p(z|X), with q(z|X,Y*) during training
  -> p(r_ij|X,z,g_i,g_j), with q_relation during training
  -> relation-specific sparse rank-8 energy
  -> two-round structured joint sampling
  -> unchanged GDTS common trunk
  -> frozen GDTS branch denoiser + branch-specific dependency residual
```

The endpoint heatmap/candidate path runs in parallel from the observed input and
provides the required goal hypotheses. This is consistent with the intended
architecture.

### 1.2 Information supplied to each module

| Module | Required information | Updated-plan result |
|---|---|---|
| Social encoder | history, sparse graph | Correct: unchanged implementation produces `[N,128]`. |
| Scene prior | agent features, scene grouping/count | Correct: invariant attention pool yields `p_z [C,4]`. |
| Scene posterior | history scene feature, future relative positions/velocities | Correct: exact `256 -> 128 -> 4` fusion yields `q_z [C,4]`. |
| Relation teacher | six deterministic future pair descriptors | Correct: exact `6 -> 64 -> 64 -> 4` teacher on sparse edges. |
| Unary | agent feature, candidate displacement/distance, `log_prior` | Correct: `[N,K]`, zero-initialized residual. |
| Dynamic relation | history relation, scene mode, exact symmetric goal geometry | Correct: `Q [Z,M,16]`, `B [Z,M]`, full and selected APIs. |
| Joint energy | oriented pair context, prior-free candidate representation, z/m embeddings | Correct: factors `[E,Z,M,K,8]` and relation marginalization. |
| Sampler | `p_z`, unary, selected-neighbor effective energy | Correct: modes `[C,P]`, indices `[N,P]`, no `K^N`. |
| GDTS diffusion | branch goal context and original noisy state | Correct: base denoiser and shared trunk unchanged. |
| Corrector | noisy velocity state, world anchor/geometry, timestep, branch relation embedding | Correct: exact 21-D input and `[N,T,2]` residual. |

### 1.3 Diffusion placement

The corrected interpretation is explicit:

- no correction on the common trunk;
- branch correction begins only after the existing split;
- each branch uses its own `z`, joint goals, and selected relation state;
- correction changes epsilon inside denoising, not completed trajectories;
- full diffusion chains are not duplicated.

This preserves GDTS tree sampling while making branch correction hypothesis
conditioned.

**Architecture consistency: approved.**

## 2. Baseline preservation

### 2.1 Original GDTS

The updated plan keeps `goal_model_type=independent` behavior unchanged:

- no V2 construction;
- no V2 RNG consumption;
- legacy cache and augmentation;
- original `_legacy_encode()`, TTST, losses, diffusion, metrics, and checkpoint
  selection;
- legacy state-dict namespace and loading behavior;
- AMP disabled by default for historical configurations.

### 2.2 Exact all-off behavior

The former blocker is resolved with:

```text
jdv2_active = OR(the four V2 switches)
```

When false, the plan now bypasses V2 at configuration, cache, construction,
training/evaluation routing, checkpoint, metric, and RNG levels. It does not
merely return a zero corrector residual.

V2 training stages fail early when no V2 module is active. The plan includes
same-process tensor equality and end-to-end cache/prediction/metric equality
tests.

### 2.3 Partial ablations

Each partial ablation has a stable neutral contract:

- scene latent off: no scene sampling/loss, zero scene energy embedding,
  z-independent averaged relation query;
- dynamic relation off: history-only relation and no relation KL;
- joint energy off: unary-only sampling;
- corrector off: exact zero residual.

Public output layouts remain unchanged.

**Baseline preservation: approved.**

## 3. Mathematical consistency

### 3.1 Scene latent

The prior and posterior are now fully specified:

```text
p(z|X)        [C,Z]
q(z|X,Y*)     [C,Z]
```

The future encoder receives exactly relative position plus future velocity.
Posterior collapse is addressed by making `q_z` participate in
`L_PL_post`, while KL transfers future information to the prior.

### 3.2 Relation

The deployable relation is exactly:

```text
p(r_ij | X,z,g_i,g_j)
```

The four geometry features now have explicit symmetric formulas, units,
normalization, zero-norm behavior, and mode/relation query parameters. The
future teacher and relation KL are also fully specified.

### 3.3 Joint energy

The energy contract is closed:

```text
left/right factors [E,Z,M,K,8]
E^{z,m}(k,l) = -dot(F_i,F_j)/sqrt(8)
Eeff^z(k,l) = -logsumexp_m(log p_relation - E^{z,m})
```

The joint and conditional scores use the same sign. Canonical edges are counted
once in the joint score. There is no degree normalization and the coefficient
is 1.

The energy candidate representation is prior-free, eliminating the earlier
risk of double-counting the heatmap prior.

### 3.4 Structured objective

The final goal loss is explicit:

```text
L_JG = 0.5 L_PL_post + 0.5 L_PL_prior
       + beta_z KL(q_z||p_z)
       + beta_r L_relation_KL
```

Both KL weights have a precise Stage-A optimizer-step warm-up. The posterior
path and deployable prior path train the same structured model under their
respective relation distributions.

### 3.5 Dependency correction

The equation remains:

```text
epsilon_joint = epsilon_base + Delta_epsilon
```

The base denoiser is frozen in every V2 stage. Corrector geometry is derived
from the current noisy velocity state in world metres and metres/second without
an `x0` estimate. The zero-initialized last layer guarantees equality to the
base epsilon for identical context/state at initialization.

**Mathematical consistency: approved.**

## 4. Implementation-risk re-review

### 4.1 Tensor shapes

The plan now freezes symbols and public shapes, including:

- `p_z/q_z [C,Z]`;
- `q_relation [E,M]`;
- full relation `[E_chunk,Z,K,K,M]`;
- selected relation `[E,P,K,M]`;
- factors `[E,Z,M,K,R]`;
- modes `[C,P]`;
- candidate indices/goals `[N,P]` / `[N,P,2]`;
- corrector relation embedding `[E,P,16]`;
- public trajectories `[P,obs_length+T,N,2]`.

Axis naming, scene-to-agent/edge gathering, canonical reversal, `P+1` internal
contexts, `E=0`, and packed-scene validation are all included in the test plan.

Residual risk is ordinary implementation error, not missing design.

### 4.2 Dataset and cache

The explicit `build-jdv2-cache` phase resolves cache lifecycle ambiguity.

Approved safeguards:

- only deterministic/frozen artifacts are cached;
- no trainable-network outputs are cached;
- goal checkpoint is hashed by file SHA256;
- dataset/split hashes, graph config, units, `dt`, `K`, and schema are in the
  manifest;
- candidate order is schema-controlled;
- writes are atomic and mismatches are hard failures;
- future teacher data is prohibited from inference inputs;
- all-off V2 resolves to the legacy cache.

### 4.3 Training stages

The three stages now have explicit train/freeze/eval sets, objectives,
curriculum, and checkpoint-selection metrics. The denoiser cannot be unfrozen
by the legacy finetune policy.

The teacher/deployable train-test gap is addressed by the Stage-B curriculum
and deployable-only Stage C. Discrete sampler indices are not incorrectly
treated as pathwise differentiable; the goal stack is trained by `L_JG`.

### 4.4 Checkpoints

The load matrix is explicit. Legacy behavior remains unchanged; V2 transitions
validate architecture, ablation, source stage, cache hash, and source
checkpoint. Optional optimizer/scheduler/scaler and schedule progress support
true V2 training resume.

### 4.5 AMP and BF16

The former AMP blocker is resolved:

- FP16 uses `GradScaler`;
- BF16 does not;
- precision-sensitive probability, geometry, energy, scatter accumulation,
  sampling, and reduction operations are forced to FP32;
- device support and finite gradients are validated;
- equality/finite tests cover supported AMP modes.

### 4.6 Memory complexity

The full `K^2` training path is edge-chunked with default 256 and incremental
loss accumulation. Inference uses `[E,P,K,M]` selected-neighbor tensors,
vectorized over `P` and `K`, with only two Python refinement iterations.

The corrector follows branch-sequential execution and never holds 20 branch
hidden states. A crowded-scene peak-memory test is a release gate.

The design remains polynomial in sparse `E` and fixed `K/Z/M/P`; no `K^N` or
dense `[N,N,K,K]` object is permitted.

**Implementation risks: bounded and adequately mitigated.**

## 5. Previous blocker resolution matrix

| Previous blocker | Resolution | Status |
|---|---|---|
| All-off V2 did not equal original GDTS | Exact configuration/cache/model/RNG/loss/metric/checkpoint passthrough | Resolved |
| Missing shape/probabilistic contract | Frozen symbols, shapes, effective energy, joint and conditional scores | Resolved |
| Posterior/teacher training path unclear | Exact posterior, dual PL paths, relation teacher/KL, warm-up | Resolved |
| Relation geometry undefined | Four exact symmetric normalized scalars and `Q/B` query | Resolved |
| Relation-specific factors/fusion undefined | Prior-free candidate encoder, oriented pair context, z/m embeddings, `[E,Z,M,K,8]` | Resolved |
| Corrector coordinate/trunk policy undefined | World velocity geometry, no `x0`, unchanged trunk, branch-only correction | Resolved |
| Full-pair memory/vectorization risk | Edge chunks of 256, incremental FP32 loss, `[E,P,K,M]` inference, branch-sequential corrector | Resolved |
| Cache lifecycle unclear | Explicit cache phase and frozen/deterministic manifest policy | Resolved |
| FP16/BF16 behavior unclear | Exact scaler policy and FP32 islands | Resolved |
| Stage selection/resume incomplete | Stage metrics/tie-breaks and extended V2 checkpoint state | Resolved |

## 6. Approved source-file plan

### New production files

- `src/models/joint_dependency_v2/__init__.py`
- `src/models/joint_dependency_v2/scene_latent.py`
- `src/models/joint_dependency_v2/future_teacher.py`
- `src/models/joint_dependency_v2/unary_goal.py`
- `src/models/joint_dependency_v2/dynamic_relation.py`
- `src/models/joint_dependency_v2/joint_energy.py`
- `src/models/joint_dependency_v2/joint_sampler.py`
- `src/models/joint_dependency_v2/dependency_corrector.py`
- `src/joint_dependency_v2_cache.py`

### Existing production files to modify

- `main.py`
- `src/parser.py`
- `src/models/model.py`
- `src/joint_goal_loss.py`
- `src/data_grouping.py`
- `src/data_pre_process.py`
- `src/data_loader.py`
- `src/batch_cache_io.py` only if generic manifest/hash helpers are needed
- `src/trainer.py`
- `tools/evaluate_jmm_official.py` only if config compatibility requires it

The following should remain behaviorally unchanged:

- `src/models/diffusion.py`
- `src/models/social_encoder.py`
- `src/models/interaction_graph.py`
- `src/models/relation_inference.py`
- `src/metrics.py`

Tests will be added/extended for parser/passthrough, all seven modules,
mathematics/losses, cache, stages/curriculum, checkpoint matrix, AMP,
permutation/zero-edge/NaN, memory stress, integrated forward/loss, baseline
equivalence, and official JMM export.

## 7. Remaining blockers

There are no unresolved design blockers that genuinely prevent implementation.

Implementation must nevertheless wait for the user's next explicit source-code
authorization.
