# JDV2 Stage-B V2-A design review

## Status

**`STAGE_B_V2_DESIGN_APPROVED`**

Next state only: **`eligible_for_stage_b_v2_implementation_preflight`**.

This document approves one narrowly scoped design: a parameter-free,
component-relative zero-mean projection of the Stage-B denoising residual.
It does not authorize implementation, training, finetuning, objective
reweighting, or any Stage-A change.

Design-review source HEAD:
`9251974c99f102fbfed11f9ad989d6c51ccb70fb`.

## 1. Motivation and evidence boundary

After the mixed-precision identity fix, the paired five-seed Stage-B V1
result remains worse than frozen Stage A by 1.587% minADE, 1.258% minFDE,
0.698% JADE, and 0.252% JFDE; Relative Motion changes by only -0.010%.
The failure audit provides direct counterfactual evidence for one mechanism:

- the component-common part contains 31.70% of mean residual energy;
- removing it post hoc recovers 56.85% of V1 minADE harm, 71.59% of minFDE
  harm, and 40.71% of JADE harm;
- the same intervention moves JFDE from 0.252% worse than Stage A to 0.327%
  better.

This supports testing a residual constrained to the component-relative
subspace. It does **not** establish that a common residual is physically
invalid, nor that projection will improve a newly trained model. The claim
supported by the audit is narrower: a component-common residual does not
change within-component pairwise residual differences, and V1's learned
common mode causes disproportionate trajectory harm.

H3 remains secondary. The weighted relative-loss gradient is only 0.552% of
the diffusion-gradient norm at initialization and 1.303% at the selected V1
checkpoint, but the gradients are mostly aligned. Changing
`lambda_relative` together with projection would confound attribution. H2
does not justify changing the oracle branch, and H4 does not justify changing
the relation pathway.

## 2. Reviewed current source path

The current source supports the proposed insertion without changing the
corrector architecture:

- `src/models/model.py:3212-3321` implements Stage-B training. The raw
  corrector result is produced at lines 3263-3266 and immediately added to
  `epsilon_base` at line 3267.
- `src/models/model.py:3716-3823` implements diffusion inference. At each of
  the six branch timesteps, the corrector result is produced at lines
  3785-3790 and immediately added at line 3791.
- `src/models/model.py:1091-1099` creates the per-window
  `dependency_state`, which is already reused by all 20 branches and all six
  active branch timesteps.
- `src/models/joint_dependency_v2/dependency_corrector.py:52-124` returns the
  raw residual `[N,T,2]`. Its network and 30,851 parameters need no change.
- `src/models/diffusion.py` defines the frozen diffusion schedule and
  denoiser. It has no suitable graph-aware insertion point and must remain
  unchanged.

The current JDV2 production state does not contain connected-component IDs.
Audit-only component traversal exists under `tools/`, but production code
must not import it.

## 3. Canonical architecture and dataflow

V2-A changes exactly one operation after the unchanged corrector:

```text
frozen Stage-A world and relation state
  -> frozen GDTS denoiser: epsilon_base [N,T,2]
  -> unchanged DependencyCorrector: delta_raw [N,T,2]
  -> component_zero_mean(delta_raw, metadata): delta_rel [N,T,2]
  -> active-agent epsilon_base + delta_rel
  -> unchanged DDIM transition
```

Inactive agents bypass both the correction addition and the corrected DDIM
arithmetic. Stage-A candidates, 20 worlds, relation embeddings, context,
diffusion noise, timestep schedule, and trajectory integration are unchanged.

### Projection-location decision

The canonical location is **C: after `DependencyCorrector` and before
`epsilon_base + delta`**, in both training and inference.

This is preferred because:

1. it maps exactly to the successful V1 `COMPONENT_CENTERED`
   counterfactual;
2. it leaves message construction, gates, values, aggregation, output head,
   and parameter count unchanged;
3. the unprojected V1 residual remains observable for diagnostics;
4. the projection is parameter-free and can be identical in training and
   inference;
5. the BF16 inactive-agent routing remains independent and explicit.

Location A, inside message aggregation, would redefine the corrector and
would not be equivalent to the audited intervention. Location B, inside the
output head, would hide the raw residual and couple graph-component semantics
to the 30,851-parameter module. Location D, after epsilon addition or inside
DDIM, would project a mixed base/correction quantity, make dtype behavior less
transparent, and risk regressing the numerical identity fix.

## 4. Mathematical definition

Let the frozen sparse graph be `G=(V,E)`. Edges are treated as undirected for
connectivity even though `edge_index` stores each canonical physical pair
once. Let `degree_i` be the number of incident edges and let
`A={i: degree_i>0}`. The subgraph induced by `A` has connected components
`C_1,...,C_Q`; every such component has at least two agents.

For raw residual `delta_raw in R^(N x T x 2)`, define for each active
component:

```text
mu_q[tau,d] = (1 / |C_q|) * sum_{j in C_q} delta_raw[j,tau,d]

delta_rel[i,tau,d] = delta_raw[i,tau,d] - mu_q[tau,d],  i in C_q
delta_rel[i,tau,d] = 0,                                  degree_i = 0
```

The operator is an orthogonal centering projection in the agent dimension of
each component. It is not presented as an exact probabilistic score
decomposition.

### Required contracts

For every active component, future timestep, and coordinate:

```text
sum_{i in C_q} delta_rel[i,tau,:] = 0
```

For every edge `(i,j)`:

```text
delta_rel[j] - delta_rel[i]
  = delta_raw[j] - delta_raw[i]
```

Define `delta_common[i]=mu_q` for active agents and zero for inactive agents.
Then, on active agents:

```text
delta_raw = delta_common + delta_rel
```

All equalities are algebraic; implementation tests use FP32 tolerances for
reduction/subtraction rounding.

## 5. Graph and component metadata

### Source graph

Components use only the existing frozen `edge_index [2,E]` from the
parameter-free radius/TTC graph. No learned, dynamic, timestep-dependent,
kNN, or fully connected graph is introduced. Before component construction,
the implementation must validate:

- indices lie in `[0,N)`;
- `edge_index` has shape `[2,E]`;
- endpoints do not cross `scene_index`;
- connectivity is interpreted as undirected.

An isolated agent is conceptually a singleton component but is structurally
inactive. It receives `component_id=-1`, is excluded from all component
means, and has projected residual zero.

### Metadata contract

The proposed immutable per-window structure is:

| Field | Shape | dtype | Meaning |
|---|---:|---|---|
| `component_id` | `[N]` | int64 | compact `0..Q-1` for active agents; `-1` for isolates |
| `active_mask` | `[N]` | bool | `degree > 0` |
| `active_index` | `[A]` | int64 | indices of active agents |
| `component_count` | `[Q]` | int64 | active agents per component |
| `num_components` | scalar | Python int | `Q` |

Component IDs must be deterministic, compact, device-aligned, and independent
of edge order. Canonical IDs should follow the smallest agent index in each
component. The implementation may use a small deterministic union-find once
per window/batch, followed by tensor conversion; it must not perform CPU
traversal inside a DDIM step. A device-native equivalent is also acceptable
if it produces the same canonical IDs.

### Computation and caching points

- **Inference:** compute metadata once while constructing
  `dependency_state` in `_jdv2_encode()` (`model.py:1091-1099`), then reuse
  it for 20 branches x 6 correction timesteps in `ts_sample()`.
- **Training:** compute it once from `structured['edge_index']` and
  `inputs['scene_index']` in `_jdv2_dependency_losses()`, immediately after
  line 3247. Training has one corrector call, but it must use the identical
  metadata builder and projection operator.
- **E=0:** create the all-inactive metadata without graph traversal and keep
  the literal Stage-A/no-corrector path.

Component data is derived runtime metadata, not a cache-schema field. The
frozen JDV2 candidate cache and its hash remain unchanged.

## 6. Vectorized projection

The projection must run in an autocast-disabled FP32 island. In the canonical
path, noisy world velocity and therefore `delta_raw` are already FP32; the
implementation should nevertheless make the contract explicit.

Conceptual pseudocode:

```python
def component_zero_mean(raw_delta, metadata):
    # raw_delta: [N,T,2]
    raw = raw_delta.float()
    if metadata.num_components == 0:
        return torch.zeros_like(raw)

    idx = metadata.active_index                 # [A]
    cid = metadata.component_id[idx]            # [A]
    sums = raw.new_zeros((metadata.num_components,
                          raw.shape[1], raw.shape[2]))
    sums.index_add_(0, cid, raw.index_select(0, idx))
    counts = metadata.component_count.to(raw.dtype)[:, None, None]
    means = sums / counts
    active_rel = raw.index_select(0, idx) - means.index_select(0, cid)

    projected = torch.zeros_like(raw)
    projected.index_copy_(0, idx, active_rel)
    return projected
```

No `torch-scatter` dependency is needed. Once metadata exists, time and
memory complexity are `O(N*T)` with `O(Q*T)` temporary storage. The code must
retain autograd from `delta_rel` through `index_add_`, mean subtraction, and
the raw corrector output.

The implementation should return FP32 in the canonical path. It must reject
shape/device mismatches and non-finite input/output rather than silently
broadcasting.

## 7. Tensor and precision contracts

Canonical ETH shapes and types are:

| Tensor | Shape | canonical dtype |
|---|---:|---|
| `edge_index` | `[2,E]` | int64 |
| `edge_weight` | `[E]` | FP32 |
| selected training relation | `[E,16]` | FP32 |
| deployed relation bank | `[E,20,16]` | FP32 |
| noisy world velocity | `[N,12,2]` | FP32 |
| `epsilon_base` | `[N,12,2]` | BF16 under CUDA autocast |
| `delta_raw` | `[N,12,2]` | FP32 |
| `delta_rel` | `[N,12,2]` | FP32 |
| component sums/means | `[Q,12,2]` | FP32 |
| active-agent corrected epsilon | `[A,12,2]` | FP32 after addition |

For active agents, V2 intentionally uses
`epsilon_base + delta_rel`. For inactive agents, it must not materialize a
BF16-plus-FP32-zero result and must not route the value through corrected
DDIM arithmetic.

## 8. Numerical identity routing

The mixed-precision identity fix at HEAD is authoritative.

### Whole-window `E=0`

`ts_sample()` must not call the corrector or projection and must execute the
literal Stage-A DDIM expression. Training returns the existing differentiable
corrector-connected zero losses and skips the optimizer step; it does not add
a zero residual to `epsilon_base`.

### Mixed active/inactive windows

The corrector and projector may operate on the full `[N,T,2]` tensors, but
only active agents enter corrected epsilon/DDIM arithmetic. The one shared
diffusion-noise tensor is generated exactly once. The next state is assembled
from:

- literal Stage-A transition values for `active_mask=False`;
- V2 projected-correction transition values for `active_mask=True`.

This preserves degree-zero Stage-A trajectories tensor-exactly. It must not
add RNG draws, rerun the denoiser/corrector, or use a zero-addition as a
substitute for bypass.

## 9. Training contract

V2-A is initialized fresh from the same protected Stage-A epoch-13
checkpoint, not from the V1 epoch-10 checkpoint. The corrector namespace in
the Stage-A checkpoint is zero-initialized, so all 30,851 Stage-B parameters
start from the same state as V1.

Everything except projection remains V1-identical:

- one common scene-level oracle world;
- actual 20 frozen Stage-A production worlds;
- same selected-world relation embedding;
- active timesteps `[30,25,20,15,10,5]`;
- scene-consistent timestep sampling;
- Adam, learning rate `1e-4`, ExponentialLR, BF16;
- `lambda_diff=1.0`, `lambda_relative=0.05`;
- only `jdv2_corrector.*` trainable;
- E=0 items skip optimizer steps;
- JADE primary selection with JFDE tie-break.

The only changed equations are:

```text
delta_raw = DependencyCorrector(...)
delta_rel = component_zero_mean(delta_raw, component_metadata)
epsilon_joint(active) = epsilon_base(active) + delta_rel(active)

L = L_diff + 0.05 * L_relative
```

Both `L_diff` and clean-trajectory recovery for `L_relative` must consume the
projected residual. No common-mode penalty, projection penalty, new teacher,
new relative loss, gradient balancing, GradNorm, PCGrad, or loss rescaling is
permitted.

The null component-common direction receives no task gradient after the
projection. This is expected: it cannot affect deployed V2 predictions. Raw
common magnitude remains diagnostic only and is not regularized.

## 10. Inference contract

The shared trunk is unchanged and never corrected. At each of the six active
branch timesteps and for each of the same 20 world slots:

1. compute `epsilon_base` once;
2. compute `delta_raw` once with the unchanged selected-slot relation;
3. apply the same training-time projection using cached metadata;
4. compute one corrected transition for active agents;
5. route literal Stage-A transition values to inactive agents;
6. consume exactly the original single diffusion-noise draw.

No projection is applied after epsilon addition, after DDIM transition, or
after trajectory integration. Candidate IDs, goals, relations, world order,
and 20/20 support must remain unchanged.

## 11. Versioning and implementation scope

The implementation preflight should use the explicit version
`jdv2-stage-b-v2-a`. It must fail closed against V1 checkpoints.

Expected future implementation files are limited to:

- add `src/models/joint_dependency_v2/component_projection.py` containing
  the metadata builder and parameter-free projection;
- update `src/models/joint_dependency_v2/__init__.py` only to export those
  utilities;
- update the two call sites and dependency-state construction in
  `src/models/model.py`;
- update `src/parser.py` to accept the explicit V2-A version while preserving
  all frozen Stage-A requirements;
- update `src/trainer.py` so Stage-A -> V2-A is a weights-only transition,
  V2-A resume requires exact V2-A provenance, and V1 -> V2-A resume is
  rejected;
- add a V2-A config copied from the V1 config with only run name and
  architecture version changed;
- add focused V2-A tests.

`dependency_corrector.py`, `diffusion.py`, Stage-A sampler/relation/energy,
loss definitions, cache files, and checkpoint bytes must remain unchanged.

V2-A checkpoint metadata must store the architecture version and projection
semantics. Because the state dict gains no parameter/buffer, the architecture
version—not missing keys—is what prevents accidental V1/V2-A resume.

## 12. Lightweight diagnostics

Only compact aggregates are required:

- `raw_delta_rms`;
- `projected_delta_rms`;
- `removed_common_rms`;
- `common_energy_fraction`;
- maximum absolute per-component projected sum;
- projected-delta/base-epsilon RMS ratio;
- component-size histogram.

Diagnostics must use detached FP32 tensors, must not save dense residuals,
and must not enter the loss. `common_energy_fraction` must document its exact
denominator and zero-residual convention.

## 13. Performance plan

The graph traversal is performed once per window/batch, never inside the
20-by-6 correction loop. Projection uses one `index_add_`, one component
division, one gather/subtraction, and one indexed write per corrector call.
It performs no network inference and introduces no `O(N^2)` tensor.

The implementation preflight benchmark must compare V1 and V2-A arithmetic
on identical windows, worlds, and NoiseTape after warm-up. Report component
metadata time separately from per-call projection and end-to-end inference.
The proposed acceptance gate is less than 5% V1 end-to-end inference overhead
with no material peak-memory increase. This is an engineering gate, not a
scientific claim.

## 14. Required tests before training

### Projection mathematics

1. Single active component has zero mean for every timestep/coordinate.
2. Multiple components are centered independently.
3. Edge-relative raw differences are preserved.
4. `raw = common + projected` on active agents.
5. Isolates are exactly zero and excluded from component counts.
6. `E=0` returns exact zeros with valid shapes.
7. Component IDs are deterministic under edge permutation and edge-direction
   reversal.
8. Cross-scene edges and invalid indices fail closed.
9. FP32 CUDA/BF16-autocast execution is finite and gradients reach all
   corrector parameters reachable under the V1 architecture.
10. Synthetic backward agrees with an explicit loop reference.

### Integration and identity

11. DependencyCorrector source and parameter count remain unchanged at
    30,851.
12. Training and inference invoke the same projection utility.
13. E=0 CUDA/BF16 Stage B equals Stage A tensor-exactly and does not call the
    corrector/projector.
14. Degree-zero agents in mixed windows equal Stage A tensor-exactly.
15. Active-agent inference matches a reference projected transition.
16. No extra Python, NumPy, CPU Torch, or CUDA RNG is consumed.
17. Denoiser/corrector call counts remain one per authorized step.
18. Candidate IDs, goals, relations, and 20/20 coverage are unchanged.
19. Stage-A parameters remain frozen with `grad is None`; an optimizer step
    changes only corrector parameters.
20. Stage-A -> V2-A resets epoch/progress/optimizer/scheduler; V1 resume into
    V2-A is rejected; same-version V2-A resume is strict.
21. Legacy/all-off, Stage-A freeze, V1 identity, and V1 checkpoint-loading
    regressions remain green.
22. Real CUDA/BF16 full-window parity and performance gates pass.

No training is eligible until all tests, `compileall`, full `pytest`,
`git diff --check`, checkpoint hashes, and frozen-source checks pass.

## 15. Future experimental protocol (not executed here)

Use ETH only, training seed 2035, validation seeds 2035-2039, the same cache,
frozen Stage-A checkpoint, optimizer, scheduler, precision, stopping rule,
and deterministic checkpoint-selection validation as V1. V2-A must start
fresh from Stage A.

The paired final comparison contains exactly:

| Label | Model/path | Purpose |
|---|---|---|
| A | frozen Stage A | immutable baseline |
| B | Stage-B V1 epoch 10 | historical learned V1 |
| C | V1 epoch 10 + inference-only component centering | post-hoc causal reference |
| D | V2-A trained and inferred with projection | actual proposed method |

All four use paired Stage-A worlds and diffusion randomness. Report minADE,
minFDE, JADE, JFDE, Joint Goal Endpoint, Compatibility, Relative Motion,
E=0/E>0, mixed degree-zero parity, per-seed values, mean/std, 20/20 coverage,
projection diagnostics, runtime, and peak memory. C is not a deployable model
and cannot substitute for D.

Model selection remains minimum JADE with JFDE tie-break. Changing selection
would confound V1/V2 attribution and is not supported by the source or audit.

## 16. Adoption and rejection criteria

Hard implementation gates are non-negotiable:

- zero-mean and pairwise-invariance tests pass;
- E=0 and degree-zero identity are tensor-exact;
- Stage-A worlds/coverage and checkpoint hashes are unchanged;
- only 30,851 corrector parameters are trainable;
- endpoint and compatibility are identical under paired evaluation;
- no NaN/Inf, RNG drift, checkpoint mismatch, or >5% runtime overhead.

The predeclared scientific adoption signature requires all of the following,
not one favorable metric:

1. D reduces at least 50% of V1's paired Stage-A minADE and minFDE gaps;
2. D does not worsen JADE by more than 1% relative to paired Stage A and is
   no worse than V1;
3. D preserves the counterfactual JFDE direction by being no worse than
   paired Stage A;
4. Relative Motion is reported and is not more than 1% worse than Stage A;
5. raw common output is removed to FP32 tolerance at every deployed step;
6. results are not driven solely by one evaluation seed.

These percentages are engineering adoption gates, not theoretical
guarantees. If hard contracts pass but the joint metric signature fails,
V2-A is rejected rather than tuned inside the same experiment.

A later V2-B objective-balancing design review becomes scientifically
eligible only if V2-A is correctly enforced, trains stably with a nonzero
projected residual, removes common drift, still shows no clear trajectory
gain, and a repeated gradient audit continues to show a negligible weighted
relative gradient. It does not become eligible merely because one V2-A metric
misses a threshold.

## 17. Paper-level interpretation

Audit-supported motivation:

> V1 contains a component-common residual mode that leaves pairwise residual
> differences unchanged and causes disproportionate trajectory harm in the
> paired counterfactual.

Hypothesis to test with V2-A:

> When the absolute/marginal realization is supplied by frozen GDTS,
> restricting the learned social correction to the component-relative
> zero-mean residual subspace may improve dependency correction without
> eroding marginal trajectories.

Only a successful freshly trained D comparison can support the stronger
paper-level statement that frozen marginal generation should be paired with
component-relative social residual correction. The V1 post-hoc result alone
is insufficient.

## 18. Answers to the design questions

1. Insert projection in `model.py` immediately after the corrector calls at
   current lines 3263-3266 and 3785-3790, before epsilon addition.
2. This preserves the corrector architecture, exposes raw V1 output, matches
   the audited intervention, and keeps BF16 routing explicit.
3. Derive components from the frozen scene-local `edge_index`, interpreted as
   an undirected graph; isolates are inactive.
4. Compute once in `_jdv2_encode()` for inference and once in
   `_jdv2_dependency_losses()` for training; cache in `dependency_state` for
   all branch steps.
5. Shapes/dtypes are specified in Sections 5 and 7; projection is FP32.
6. Use `index_add_` component sums, count division, indexed gather/subtraction,
   and indexed output assembly.
7. E=0 bypasses correction entirely; mixed degree-zero agents receive the
   literal Stage-A DDIM transition.
8. Yes. Subtracting the same component mean from both endpoints preserves
   every edge-relative residual difference up to FP32 rounding.
9. No. The count remains exactly 30,851.
10. No relation dimension, gathering, network, or branch semantics change.
11. No Stage-A candidate, world, order, sampler, or coverage semantics change.
12. No. Loss remains `L_diff + 0.05*L_relative`.
13. Only raw-to-projected residual conversion is new; every other V1
    train/inference choice remains fixed.
14. The required pre-training tests are enumerated in Section 14.
15. Adoption requires all identity/geometry gates plus the multi-metric
    signature in Section 16.
16. V2-B review requires a correct but insufficient V2-A together with
    repeated evidence that objective-gradient magnitude remains the blocker.

## 19. Implementation checklist

- [ ] Add parameter-free component metadata/projection utility.
- [ ] Add deterministic scene-boundary and index validation.
- [ ] Compute metadata once per train batch and inference window.
- [ ] Apply the identical projection at both corrector call sites.
- [ ] Preserve literal inactive-agent Stage-A routing and shared noise.
- [ ] Add explicit `jdv2-stage-b-v2-a` config/checkpoint contract.
- [ ] Reject V1 -> V2-A resume; initialize from protected Stage A.
- [ ] Add compact raw/projected/common diagnostics only.
- [ ] Add mathematical, gradient, identity, RNG, checkpoint, and CUDA tests.
- [ ] Benchmark metadata, projection, end-to-end time, and peak memory.
- [ ] Verify protected hashes and unchanged `DependencyCorrector`/diffusion.
- [ ] Run `python -m compileall -q .`, full `pytest`, and
      `git diff --check`.
- [ ] Stop after implementation preflight; obtain separate authorization
      before any V2-A training.

## 20. Decision

The design is theoretically coherent, directly tied to the strongest causal
V1 evidence, and implementable without changing the corrector architecture,
Stage A, relation semantics, loss, or parameter count. Its remaining risk is
scientific rather than architectural: training in the projected subspace may
not reproduce the post-hoc V1 counterfactual benefit, and projection alone
does not address the secondary objective-gradient magnitude imbalance.

Final state: **`STAGE_B_V2_DESIGN_APPROVED`**.

Next state only: **`eligible_for_stage_b_v2_implementation_preflight`**.
