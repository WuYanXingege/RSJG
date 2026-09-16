# Codex One-Shot Implementation Prompt — RSJG Joint Dependency V2

Repository: `https://github.com/WuYanXingege/RSJG`

Target branch: `research/joint-dependency-v2-clean`

Required base commit: `f9d4fefce5738f4e85942ce4fe8d9dc754d1f637`

## 0. Clean-workspace rule

This branch must be created from the base commit above, independently of V5 / Selector / BASR work. Do not copy dirty source-tree state into this branch.

Preferred server setup:

```bash
git clone https://github.com/WuYanXingege/RSJG.git RSJG_joint_dependency_v2
cd RSJG_joint_dependency_v2
git checkout -b research/joint-dependency-v2-clean f9d4fefce5738f4e85942ce4fe8d9dc754d1f637
```

If using an existing local repository, create a new git worktree instead of cleaning the old one:

```bash
git worktree add ../RSJG_joint_dependency_v2 \
  -b research/joint-dependency-v2-clean \
  f9d4fefce5738f4e85942ce4fe8d9dc754d1f637
```

Never use `git reset --hard` or `git clean -fdx` in the old research workspace.

## 1. Read the design first

Implement exactly according to:

```text
docs/joint_dependency_v2/JOINT_DEPENDENCY_V2_DESIGN.md
```

The fixed method chain is:

```text
GDTS candidate support
    -> scene latent z
    -> hypothesis-conditioned relation r_ij
    -> sparse low-rank joint goal distribution
    -> structured joint goal sampling
    -> frozen GDTS denoiser + zero-init dependency corrector
```

Do not add a new main module unless mathematically necessary.

## 2. New namespace

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
    tensor_ops.py

src/losses_joint_dependency_v2.py
src/joint_dependency_v2_protocol.py
src/cache/joint_dependency_v2_cache.py

tools/build_joint_dependency_v2_cache.py
tools/diagnose_joint_dependency_v2.py
tools/benchmark_joint_dependency_v2.py
```

Add:

```text
--goal_model_type joint_dependency_v2
--training_stage joint_dependency_v2_goal
--training_stage joint_dependency_v2_trajectory
--training_stage joint_dependency_v2_finetune
```

When disabled, all old code paths and checkpoints must behave exactly as before.

## 3. Scene latent

Implement a permutation-invariant attention pool plus log crowd count.

Pseudocode:

```python
class SceneSetPool(nn.Module):
    def forward(self, agent_feat, scene_index):
        logits = self.att_mlp(agent_feat).squeeze(-1)
        weight = segment_softmax(logits, scene_index)
        pooled = segment_sum(weight[:, None] * agent_feat, scene_index)
        count = torch.bincount(scene_index, minlength=num_scenes).float()
        density = torch.log1p(count)[:, None]
        return self.out(torch.cat([pooled, density], dim=-1))
```

Prior:

```python
p_z = softmax(prior_head(scene_pool(agent_feat)), -1)
```

Training posterior:

```python
future_feat = future_encoder(future_world - last_pos[:,None])
post_agent = post_mlp(cat([agent_feat, future_feat], -1))
q_z = softmax(post_head(scene_pool(post_agent)), -1)
```

Loss:

```python
L_z = KL(q_z || p_z)
```

The posterior is train-only.

## 4. Hypothesis-conditioned relation

Required deployable prior:

\[
p(r_{ij}|X,z,g_i^k,g_j^l).
\]

Observation context:

```python
src, dst = edge_index
pair_obs = torch.cat([
    agent_feat[src] + agent_feat[dst],
    (agent_feat[src] - agent_feat[dst]).abs(),
    edge_feat,
], -1)
pair_ctx = pair_encoder(pair_obs)
```

Candidate geometry:

```python
disp = goals - last_pos[:,None,:]
di = disp[src][:,:,None,:]
dj = disp[dst][:,None,:,:]
rel_disp = dj - di
sep = norm(goals[dst][:,None,:,:] - goals[src][:,:,None,:], -1)
cos = safe_cosine(di, dj)
geom = cat([rel_disp, sep[...,None], cos[...,None]], -1)
```

Implement both:

```text
full_pair_relation()
selected_neighbor_relation()
```

The selected-neighbor path is mandatory for efficient sampling.

Training relation teacher descriptor:

```text
minimum future distance
normalized time of minimum distance
final relative displacement
formation change
```

Train:

```python
L_r = KL(q_relation_teacher || p_relation_prior_at_gt_hypothesis)
```

Do not give latent relation modes semantic names.

## 5. Unary goal

Use:

\[
u_i^z(k)=\log p_0(g_i^k|X)+\Delta u_i^z(k).
\]

The learned component is a residual over the frozen GDTS candidate prior.

## 6. Joint pair energy

Use scene-mode and relation-conditioned low-rank factors:

```text
left_factor  [E,R,M,K,d]
right_factor [E,R,M,K,d]
```

```python
energy_rel = -einsum(
    'ermkd,ermld->ermkl',
    left_factor,
    right_factor
) / sqrt(d)
```

Correct relation mixture:

```python
effective_energy = -torch.logsumexp(
    torch.log(relation_prob.clamp_min(eps)) - energy_rel,
    dim=relation_mode_dim
)
```

Use float32 for energy accumulation/logsumexp.

## 7. Conditional pseudo-likelihood

Use soft GT endpoint target.

Reference:

```python
src, dst = edge_index

to_src = torch.einsum(
    'erkl,el->erk',
    pair_energy,
    soft_target[dst]
)

to_dst = torch.einsum(
    'erkl,ek->erl',
    pair_energy,
    soft_target[src]
)

social = torch.zeros_like(unary_score)
social.index_add_(0, src, to_src)
social.index_add_(0, dst, to_dst)

score = unary_score - energy_weight * social
log_p_k_z = F.log_softmax(score, -1)
qz_agent = q_z[scene_index]

log_p_k = torch.logsumexp(
    torch.log(qz_agent.clamp_min(1e-8))[:,:,None] + log_p_k_z,
    dim=1
)

per_agent = -(soft_target * log_p_k).sum(-1)
loss = scene_balanced_mean(per_agent, scene_index)
```

Never call this exact joint likelihood.

## 8. Joint sampler

Output:

```text
joint_goal_index [N,S]
joint_goal_world [N,S,2]
```

Each column is one coherent scene-level world.

Use scene-mode sampling/stratification plus 1–2 rounds of parallel conditional refinement.

Do not call synchronous updates exact Gibbs.

## 9. Dependency corrector

Base:

```python
eps_base = frozen_gdts_denoiser(...)
```

New:

```python
eps_joint = eps_base + delta_eps_dependency
```

The corrector must NOT directly receive:

```text
h_i, h_j, raw goals, scene mode z
```

It only receives:

```text
relative noisy future position
relative noisy future velocity
soft relation embedding
diffusion-step FiLM modulation
```

Pseudocode:

```python
rel_pos = noisy_future[dst] - noisy_future[src]
rel_vel = finite_difference(rel_pos)
distance = norm(rel_pos, -1)

pair_state = cat([
    rel_pos,
    rel_vel,
    distance[...,None],
    relation_embedding_expanded,
], -1)

h = pair_state_encoder(pair_state)

gamma, beta = time_film(time_embedding(t_diff)).chunk(2, -1)
h = gamma * h + beta

alpha = sigmoid(gate_head(h))
value = value_head(h)

# build both directed messages from canonical edge
# aggregate normalized weighted mean per destination agent
message = normalized_sparse_aggregate(alpha, value, edge_index)

delta_eps = zero_init_output(activation(message))
eps_joint = eps_base + delta_eps
```

Required exact properties:

```text
zero-init -> eps_joint == eps_base
zero-edge -> delta_eps == 0
```

Do not add another temporal Transformer.

## 10. Losses

Stage A:

```text
L_joint_goal = L_goal_PL + beta_z * L_z + beta_r * L_r
```

Stage B:

```text
L_joint_trajectory = L_diffusion + lambda_rel * L_relative_motion
```

Do not add entropy/balance regularizers unless actual collapse is measured.

## 11. Cache

Cache only deterministic/frozen-stage data.

Allowed:

```text
synchronized scene tensors
observed geometry / sparse edge features
frozen Goal-U-Net candidate goals and prior
train/val future teacher descriptors
```

Never cache:

```text
p(z)
q(z)
relation probabilities
energy factors
unary residual
sampled joint worlds
dependency hidden states
```

Every cache root must have a versioned manifest with dataset/split/config/checkpoint hashes.

Teacher-descriptor cache must contain:

```json
{"contains_future_supervision": true}
```

Inference/test loaders must not expose future supervision fields to deployable modules.

## 12. Numerical/efficiency constraints

Mandatory:

```text
sparse edges only
no N*N*K*K tensor
variable-N scene packing
no Python per-agent attention loops
bf16 where safe
float32 KL/logsumexp/energy accumulation
all-invalid mask finite fallback
zero-edge scene safe
scene-balanced losses
```

## 13. Tests

Create tests for:

```text
scene permutation invariance
scene packing isolation
future-teacher inference isolation
candidate-hypothesis-dependent relation
relation edge-reversal equivariance
full-vs-selected relation agreement
relation logsumexp mixture vs brute force
no-edge pseudo-likelihood -> unary likelihood
joint-sample column coherence
zero-init dependency identity
a zero-edge agent receives zero dependency residual
old checkpoint compatibility
cache invalidation and leakage prevention
```

Run:

```bash
python -m compileall .
git diff --check
pytest
```

Fix all failures before expensive training.

## 14. Training sequence

Stage A:

```text
joint_dependency_v2_goal
```

Primary gate:

```text
JFDE / joint-goal endpoint error improves
marginal Goal minFDE does not materially degrade
```

Stage B only after Stage A passes:

```text
joint_dependency_v2_trajectory
```

Primary gate:

```text
JADE or relative-motion error improves
marginal minADE/minFDE remains stable
```

Stage C optional joint fine-tune.

Held-out test is not used for hyperparameter selection.

## 15. Required ablations

```text
GDTS
current public JointGoal
V4 only
current JointGoal + V4
new Scene Teacher + unary
+ hypothesis-conditioned relation
+ joint energy
+ structured sampler
+ dependency corrector
full Joint Dependency V2
```

Dependency corrector:

```text
state only
state + observation
state + relation   <-- preferred compact design
state + relation + observation
```

If observation adds negligible improvement after relation bottleneck, remove it.

## 16. Final report

Generate:

```text
docs/joint_dependency_v2/IMPLEMENTATION_REPORT.md
```

Include:

```text
git HEAD
new/modified files
parameter counts
cache manifests
all test results
compileall/git diff check
minADE/minFDE
JADE/JFDE
joint goal error
relative-motion error
latent usage/KL
throughput
peak VRAM
pipeline state
```

Do not wait for intermediate user confirmation. A failed validation gate may stop expensive downstream training, but all modules and tests must already be implemented.
