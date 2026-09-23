# JDV2 Stage-B V1 implementation and preflight

## Status

**`STAGE_B_V1_READY_FOR_TRAINING`**

This status means that the implementation is eligible for a separate
Stage-B V1 training review. No Stage-B training was started by this task.
Stage A remains frozen.

Implementation parent: `cdc19a5312f4708d064428adbc28203fcf282326`.
Canonical configuration:
`configs/joint_dependency_v2/jdv2_stage_b_v1_eth.yaml`.

## 1. Frozen boundary and provenance

Stage B initializes from the protected strict-no-z Stage-A epoch-13
checkpoint:

```text
outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/
jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt
```

- checkpoint SHA256:
  `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`;
- Stage-A freeze manifest SHA256:
  `ecce9f937717794eadc5e84be486372f01ef9c7727a4e59c36b70e09df06ec79`;
- Stage-A freeze source commit:
  `ad35a440695a4b4eb23ad3e900eb27af9b3b8909`;
- legacy GDTS checkpoint SHA256 remains
  `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950`.

The Stage-A checkpoint, frozen Stage-A config, freeze manifest, and
`dependency_corrector.py` remained byte-identical during implementation.
Their verified SHA256 values are respectively `699336...5bb`,
`bf5d4a...e62`, `ecce9f...f79`, and `0ff28e...522`.

Stage-B checkpoints retain the existing legacy/cache provenance and add:

- `stage_a_parent_checkpoint_sha256`;
- `stage_a_freeze_manifest_sha256`;
- `stage_a_freeze_source_commit`;
- `stage_b_architecture_version=jdv2-stage-b-v1`.

The `joint_goal -> joint_trajectory` load is strict. It verifies the parent
checkpoint bytes and freeze manifest. A cross-stage load transfers weights
only: optimizer, scheduler, epoch and stage progress are not inherited.
Stage-B epoch starts at 1 and its progress/optimizer-step count starts at 0.
A same-stage Stage-B resume additionally checks all four provenance fields.

## 2. Exact architecture

The existing `DependencyCorrector` is used without any architectural or file
change. It has exactly **30,851 trainable parameters**:

```text
edge/time state = relative position (2)
                + relative velocity (2)
                + distance (1)
                + Stage-A relation embedding (16)
                = 21

state encoder:  Linear(21,64) -> SiLU -> Linear(64,64) -> LayerNorm
timestep:       sinusoidal(32) -> Linear(32,64) -> SiLU -> Linear(64,128)
FiLM:           split 128 into gamma/beta, then (1+gamma)h + beta
gate:           Linear(64,32) -> SiLU -> Linear(32,1) -> Sigmoid
value:          Linear(64,64) -> SiLU -> Linear(64,64)
aggregation:    sparse normalized gated sum at both edge endpoints
output:         Linear(64,64) -> SiLU -> Linear(64,2)
```

The last output layer remains zero-initialized. The untrained residual is
therefore tensor-exact zero and
`epsilon_joint = epsilon_base + delta_epsilon` initially equals the frozen
Stage-A/GDTS result under paired randomness. No temporal Transformer, GNN or
other Stage-B network was added.

## 3. Frozen Stage-A conditioning path

For every Stage-B training batch, frozen Stage A runs under `torch.no_grad()`
and in evaluation mode. The actual production policy is used:

```text
strict_no_z
exact_lexicographic_persistent_tie
K=21, P=20, M=4, rank=8, two synchronous refinement rounds
```

The trainer supplies the exact sampler with `(seed, batch_serial)`, where:

```text
batch_serial = (epoch - 1) * batches_per_epoch + batch_index
```

This integer is stable across epoch-boundary resume, does not use
`hash()`, and only seeds the sampler's existing local generators. Tests show
that setting and executing exact Stage-A sampling does not change Python,
NumPy or global Torch RNG state; diffusion noise remains a separate stream.

The sampler returns only its 20 deployed worlds for training:

- goals `[N,20,2]`;
- relation embedding `[E,20,16]`.

The internal 21st GDTS trunk goal is never used as a Stage-B training world.

## 4. Scene-level oracle branch

For each compact scene `c` and deployed world `b`, the implementation forms:

```text
D[c,b] = mean_{i in c} ||g[i,b] - y_final[i]||^2
b_star[c] = argmin_b D[c,b]
```

The same `b_star[c]` is broadcast to every agent in the scene. It gathers:

- selected goal `[N,1,2]`;
- selected edge relation embedding `[E,16]`.

Every edge is checked to have endpoints in the same compact scene. Its
relation is gathered from that scene's selected world, never from an
independently selected agent branch. The existing frozen `_jdv2_contexts()`
is then called once on the selected goal map, producing `[N,1,256]` in the
canonical model. All 21 contexts are not computed in Stage-B training.

This hard oracle assignment is training-only. It does not alter Stage-A
sampling or inference output ordering.

## 5. Diffusion-state training contract

The corrector-active timestep set is derived by the same `np.linspace`
schedule and branch-loop indices used by `ts_sample()`. For the frozen
`100/20/trunk=30` tree it is exactly:

```text
[30, 25, 20, 15, 10, 5]
```

One member is sampled uniformly per scene and broadcast to all its agents.
The implementation validates that every edge has equal endpoint timesteps
and rejects supplied timesteps outside the inference-active set.

For target map-space future velocity `v0 [N,12,2]` and independent Gaussian
noise `epsilon`:

```text
x_t = sqrt(alpha_bar[t]) * v0
    + sqrt(1-alpha_bar[t]) * epsilon
```

The frozen denoiser predicts `epsilon_base` under `no_grad`. Only `x_t` used
for corrector geometry is converted through the existing world-coordinate
adapter. The corrector receives the selected branch relation embedding and
produces `delta_epsilon [N,12,2]`.

Inference remains unchanged: the shared trunk uses only frozen GDTS and each
of the 20 branches uses its same-slot Stage-A relation embedding at the six
active correction steps.

## 6. Objective and zero-edge behavior

Stage-B V1 optimizes only:

```text
L_stageB = 1.0 * L_diff + 0.05 * L_relative
```

`L_diff` is the noise MSE over agents with sparse degree greater than zero.
Degree-zero agents are excluded from its denominator. `L_relative` reuses
`sparse_relative_motion_loss` after recovering the clean velocity estimate
and converting it to world positions.

For `E=0`, both components are exact differentiable zeros connected to every
corrector parameter. A direct backward is valid and produces only exact-zero
gradients. The trainer recognizes that the batch has no dependency signal
and skips its optimizer step, so no E=0 item updates Stage-B parameters or
advances the Stage-B optimizer-step counter.

No goal, collision, entropy, balance, gate, endpoint, smoothness, KL,
marginal-preservation, or teacher loss was added.

## 7. Freeze and gradient behavior

With `training_stage=joint_trajectory`:

- only `jdv2_corrector.*` has `requires_grad=True`;
- Goal U-Net, history encoder and base diffusion denoiser are frozen/eval;
- SocialMotionEncoder, base relation, dynamic relation, unary, joint energy,
  future teacher and sampler-side Stage-A modules are frozen/eval;
- an E>0 backward produces finite nonzero corrector gradients;
- all frozen parameters retain `grad is None`;
- an Adam smoke step changes only corrector tensors.

The parser accepts `strict_no_z + joint_goal` and
`strict_no_z + joint_trajectory`. It rejects `strict_no_z + joint_finetune`.
The global refinement-policy default remains `categorical`; the canonical
Stage-B config explicitly selects the frozen exact policy.

## 8. Compact diagnostics

Training exposes scalar-only diagnostics:

- trainable corrector parameter count;
- oracle branch squared goal error;
- active-timestep histogram;
- degree-positive agent count;
- E=0 skipped count;
- residual RMS, base-epsilon RMS, and residual/base RMS ratio;
- `L_diff` and `L_relative`.

No dense diagnostic tensor is persisted. The gate is not currently exposed
by `DependencyCorrector`; it was therefore not added solely for logging.

## 9. Verification evidence

Validation was implementation-only; no training run was launched.

| Check | Result |
|---|---:|
| `python -m compileall -q .` | PASS |
| full `pytest -q` | 390 passed, 13 skipped |
| Stage-B + Stage-A focused CPU suite | 38 passed, 4 skipped |
| real CUDA/BF16 Stage-B backward + Stage-A freeze suite | 7 passed |
| real epoch-13 `joint_goal -> joint_trajectory` strict load | PASS |
| `git diff --check` | PASS |
| DependencyCorrector trainable parameters | 30,851 |
| canonical active timesteps | 30,25,20,15,10,5 |
| actual parent checkpoint final output-layer nonzeros | 0 weight, 0 bias |
| Stage-A checkpoint SHA256 after implementation | unchanged |

The targeted tests cover architecture and parameter count, exact initial and
E=0 residuals, parser authorization, immutable freeze references, shared
scene oracle selection, multi-scene and cross-scene checks, same-world
relation gathering, timestep derivation/consistency/rejection, parameter
freezing and eval state, deterministic sampler context and RNG isolation,
E>0 finite backward, E=0 differentiable zero, optimizer isolation, paired
untrained inference no-harm, and CUDA/BF16 execution. Existing Stage-A freeze
and all-off/legacy regressions remain green in the full suite.

## 10. Preflight answers

- Exact Stage-B architecture: the unchanged sparse FiLM
  `DependencyCorrector` added only on the six branch diffusion steps.
- DependencyCorrector unchanged: **yes**.
- Exactly 30,851 trainable parameters: **yes**.
- Untrained Stage B reproduces frozen Stage A under paired RNG: **yes,
  tensor-exact in the paired forward test**.
- `strict_no_z + joint_trajectory` safely executable: **yes**.
- Actual frozen Stage-A 20-world outputs used: **yes**.
- One common oracle branch per scene: **yes**.
- Same-slot relation embedding: **yes**.
- Scene-consistent, inference-active timesteps only: **yes**.
- Degree-zero agents excluded from `L_diff`: **yes**.
- E=0 exact-zero safe and optimizer-skipped: **yes**.
- Stage-A/GDTS gradients absent: **yes**.
- Stage-A checkpoint unchanged: **yes**.
- CUDA/BF16 forward/backward: **pass**.
- Existing Stage-A freeze regressions: **pass**.

Final handoff: **`eligible_for_stage_b_v1_training_review`**.
