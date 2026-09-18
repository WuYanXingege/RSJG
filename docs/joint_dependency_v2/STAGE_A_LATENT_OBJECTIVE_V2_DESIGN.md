# Stage-A Latent Objective V2 Design

## 1. Scope and decision

This document proposes a narrow correction to the Stage-A scene-latent
objective. It does not redesign Joint Dependency V2 and does not authorize
training or Stage B.

The proposal is:

1. expose the already-computed per-scene, per-mode composite log-scores;
2. replace trainable expected-PL mode weighting with exact enumeration and
   log-marginalization over the four scene modes;
3. derive a detached responsibility from that marginalized composite model;
4. train the future posterior only as an amortized approximation to that
   responsibility;
5. remove the direct history shortcut from the posterior evidence head;
6. use the detached responsibility, not q, to weight relation distillation.

This is theoretically self-consistent for the defined pseudo/composite model
and is implementable with small reductions and one small posterior-head input
change. It does not make z mathematically identifiable in all cases. Mixture
component starvation, label permutation, weak per-mode score separation, and
E=0 scenes remain explicit risks and must be controlled by acceptance gates,
not hidden with entropy or balance regularizers.

Frozen elements remain unchanged:

- Goal U-Net and the K=21 candidate bank;
- SocialMotionEncoder;
- dynamic-relation architecture, M=4;
- low-rank joint energy, rank=8;
- scene-mode count Z=4;
- joint sampler and P=20 deployed samples;
- GDTS diffusion and evaluation protocol.

## 2. Current implementation and available per-mode score

### 2.1 Existing tensors

For a batch containing C scenes, N agents, E canonical sparse edges, Z=4
scene modes, K=21 goal candidates, and M=4 relation modes, the current
Stage-A path constructs:

| Tensor | Shape | Meaning |
|---|---:|---|
| `target` | `[N,K]` | normalized soft GT endpoint assignment t_i(k) |
| `unary` | `[N,K]` | z-independent candidate score u_i(k) |
| `prior_effective` | `[E,Z,K,K]` | deployable relation-marginalized pair energy |
| `post_effective` | `[E,Z,K,K]` | future-relation-teacher pair energy |
| `local_prior` | `[N,Z,K]` | summed expected neighbour energy, deployable path |
| `local_post` | `[N,Z,K]` | summed expected neighbour energy, teacher path |
| `p_z`, `q_z` | `[C,Z]` | history prior and current future posterior |
| `compact` | `[N]` | compact scene index in `[0,C)` |

For path a in `{prior, post}`, the current conditional distribution is

```text
log pi^a_i,z(k)
  = log_softmax_k(
        u_i(k) - local_energy^a_i,z(k)
    ).
```

The soft-target per-agent, per-mode cross entropy already exists as

```text
PL^a_i,z
  = - sum_k t_i(k) log pi^a_i,z(k)                 [N,Z]
```

inside `_jdv2_pl_from_local()`. The current function then immediately
computes

```text
sum_z q_c(z) PL^post_i,z
sum_z p_c(z) PL^prior_i,z
```

before averaging agents within scenes and averaging scenes. Thus the needed
mode-conditioned score is present, but is not currently returned as `[C,Z]`.

### 2.2 Canonical scene score

To preserve the current scene-balanced semantics, define

```text
ell^a_i,z = sum_k t_i(k) log pi^a_i,z(k) = -PL^a_i,z

log_score^a_c,z
  = (1 / N_c) sum_{i in scene c} ell^a_i,z          [C,Z]

PL^a_c,z = -log_score^a_c,z.                        [C,Z]
```

The agent mean is intentional. A literal product of all agent conditionals
would use a sum rather than a mean. That would make responsibilities
systematically sharper in large scenes and would change the existing
scene-balanced optimization semantics. The selected definition is the log of
a geometric-mean composite score, so every scene contributes one equally
weighted example regardless of N_c.

This score is precisely a mode-conditioned GT joint-goal composite or
pseudo-likelihood score. It is not an exact normalized joint likelihood over
all K^N goal configurations: its local conditionals use soft GT-neighbour
expectations on sparse edges.

For E=0, both local-energy tensors are zero and the canonical unary is
z-independent. Consequently all four `log_score_c,z` values are equal. This
is correct neutral behavior, but it means an isolated scene contains no
objective evidence that can identify z under the frozen architecture.

## 3. Failure mechanism of the current objective

For one scene, suppressing path and scene subscripts, the current expected PL
has the form

```text
L_expected(w) = sum_z w_z PL_z,
```

where w is either trainable q or trainable p. If a is the corresponding
softmax logit,

```text
d L_expected / d a_z
  = w_z (PL_z - sum_j w_j PL_j).
```

The loss is linear in w on the probability simplex. Its optimum for fixed
unequal PL values is a vertex: put all mass on the currently lowest-PL mode.
Small random differences at initialization therefore receive immediate
positive feedback. As a mode loses probability, its score parameters receive
less useful training, reinforcing the early winner.

The current posterior creates a second shortcut:

```text
q logits = posterior_head(concat(history_scene, future_scene)).
```

It can reproduce p from `history_scene`, ignore the future GRU, select the
same low-PL mode, and drive KL(q||p) to zero. Relation KL also uses q as a
trainable mode weight, so q can reduce that loss by selecting the easiest
relation mode context rather than by representing future evidence.

The observed behavior is exactly this fixed point:

- prior and posterior hard usage are `[0,1,0,0]`;
- their entropies are approximately zero;
- mean `abs(q-p)` is `1.856e-9`;
- future-shuffle posterior L1 is `7.695e-11`.

The collapse occurred while beta_z was still very small. Increasing or
decreasing the KL warm-up alone does not correct the linear expected-loss
geometry.

## 4. Corrected probabilistic/composite formulation

### 4.1 One shared z for both Stage-A score paths

The teacher-relation and deployable-relation scores are two equal-weight views
of the same scene mode, not two independently sampled latents. Preserve the
original 0.5/0.5 balance inside one mode score:

```text
log_score_c,z
  = 0.5 * log_score^post_c,z
  + 0.5 * log_score^prior_c,z.                     [C,Z]
```

Equivalently, the composite mode score is the geometric mean of the two path
scores. Marginalizing once after this combination respects the fact that the
same z indexes the teacher and deployable energy branches. Averaging two
separate `logsumexp` losses would implicitly allow the two paths to use
different responsibilities and is therefore not the canonical proposal.

### 4.2 Mixture marginal objective

Let

```text
log_p_c,z = log p_theta(z | X_c)                   [C,Z].
```

The corrected Stage-A scene-mode term is

```text
L_mix
  = -(1/C) sum_c logsumexp_z(
        log_p_c,z + log_score_c,z
    ).
```

All score, log-softmax, logsumexp, and reduction operations remain FP32
islands under AMP/BF16.

This enumerates only Z=4 alternatives. It introduces neither a K^N tensor nor
a dense N-by-N candidate object. Existing edge chunking remains unchanged.

### 4.3 Responsibility

The composite responsibility is

```text
gamma_c,z
  = softmax_z(log_p_c,z + log_score_c,z)            [C,Z].
```

For teacher use:

```text
gamma_teacher = stop_gradient(gamma).
```

The primary mixture loss must be evaluated directly from non-detached
`log_p + log_score`, so it trains the prior and every scored Stage-A module.
Only the copy used as a target or a relation-KL weight is detached.

The relevant gradients are

```text
d L_mix / d PL_c,z       = gamma_c,z / C
d L_mix / d prior_logit  = (p_c,z - gamma_c,z) / C.
```

Thus p is not a free PL weight. It is trained by the likelihood-consistent
difference between prior probability and evidence-updated responsibility.
Mode-score parameters are trained according to responsibility.

### 4.4 Relation to a variational objective

For any normalized q and `log_score_z = -PL_z`, the exact identity is

```text
-log sum_z p_z exp(log_score_z)
  = sum_z q_z PL_z
    + KL(q || p)
    - KL(q || gamma).
```

This shows why the old weighted PL plus a small separately weighted KL is not
the exact latent marginal objective. Because Z=4 can be enumerated exactly,
there is no reason to optimize a loose q-dependent bound. q can instead be
trained as an auxiliary amortized posterior without controlling the
likelihood weights.

## 5. Redefined future-posterior role

### 5.1 Recommended posterior loss direction

q becomes a posterior approximation/future teacher:

```text
L_q
  = (1/C) sum_c KL(
        stop_gradient(gamma_c) || q_phi(z|X_c,Y*_c)
    ).
```

The user-proposed reverse direction

```text
KL(q_phi || stop_gradient(gamma))
```

has the same global minimizer when gamma has full support, so it is not
mathematically invalid. It is nevertheless mode-seeking in its optimization
geometry: q is strongly discouraged from covering low-probability target
modes. Since the failure being corrected is winner-take-all behavior, the
forward teacher-to-student KL is the safer canonical choice. It is ordinary
categorical distillation/cross-entropy and does not impose a uniform target.

No symmetric second q/gamma KL is needed.

### 5.2 Prior learning and double counting

Do not add `KL(stop_gradient(q)||p)` to the canonical objective. The direct
mixture gradient with respect to prior logits is `p-gamma`. A detached
cross-entropy or `KL(stop_gradient(gamma)||p)` has exactly the same prior-logit
gradient. Adding it to `L_mix` would count the same update twice.

Distilling p from detached q is also unnecessary because q is itself trained
to approximate gamma, which already contains p. That extra loop can amplify
early self-confirming assignments. If a future experiment chooses a pure
variational/EM formulation instead of direct `L_mix`, q-to-p distillation can
replace the direct prior update; it must not be silently added on top of it.

Keep `KL(q||p)` as a read-only diagnostic. It is no longer a Stage-A loss.

### 5.3 Relation distillation weight

The existing relation loss is otherwise preserved, but q must no longer be a
trainable weight within it. Replace its `[C,Z]` weighting input by
`stop_gradient(gamma)`. This ensures:

- the relation teacher and deployable dynamic relation still receive the same
  soft-target KL supervision;
- q cannot reduce relation loss by selecting an easy z;
- relation loss cannot train q away from its posterior target;
- no relation architecture or M=4 tensor contract changes.

## 6. Minimal future-sensitive posterior

### 6.1 Selected option: future evidence plus detached prior context

Remove the direct `history_scene -> posterior_head` shortcut. Reuse the
existing future encoder and scene mean exactly:

```text
future_input [N,T,4]
  -> existing Linear(4,64) + SiLU + GRU(64,128)
  -> permutation-invariant scene mean
  -> future_scene [C,128].
```

Change only the small posterior evidence head:

```text
future_scene [C,128]
  -> Linear(128,128) -> SiLU -> LayerNorm(128) -> Linear(128,4)
  -> evidence_logits [C,Z]

evidence_logits
  <- evidence_logits - mean_z(evidence_logits)

q_logits
  = stop_gradient(prior_logits) + evidence_logits

q = softmax(q_logits).
```

This is a minimal form of Bayes-style prior plus future evidence:

- X enters q through the detached prior logits;
- Y* enters through the existing relative-position/velocity GRU;
- q's deviation from p can only be produced by future evidence;
- `L_q` cannot update the prior through the q branch;
- there is no new Transformer, GNN, attention block, or increased hidden
  dimension.

Centering evidence logits removes the softmax-invariant common offset. It is
not a balance loss and does not prefer any mode.

Changing the first posterior-head linear layer from 256 to 128 is the only
parameter-shape change. A shape-preserving fallback could concatenate zeros in
place of `history_scene`, but retaining permanently dead parameters is less
clear and offers no scientific benefit for a required fresh Stage-A run.

### 6.2 What this does and does not guarantee

This architecture eliminates the observed history-only computational bypass.
It does not mathematically force a non-constant evidence head: any neural head
can learn zero weights or constant mode biases. Actual future sensitivity must
therefore be an acceptance criterion. It cannot be guaranteed by architecture
alone without imposing an additional information/diversity constraint, which
is outside the permitted fix.

## 7. Exact Stage-A V2 objective

The recommended objective is

```text
L_JG_v2
  = L_mix
    + beta_q(t) * L_q
    + beta_r(t) * L_relation(gamma_teacher).
```

Use the existing schedule values without inventing a new coefficient:

```text
beta_q(t) = beta_r(t)
          = 0 -> 0.1 linearly over the first 20% of completed
            Stage-A optimizer steps, then 0.1.
```

`beta_q` replaces the old `beta_z * KL(q||p)` slot. q no longer contributes
to `L_mix`, so a zero initial beta cannot let q manipulate the primary score.
The warm-up avoids forcing the posterior to chase unstable responsibilities
at the first updates.

For clarity, the gradient ownership is:

| Term | Prior p | Future q | Unary/social | Dynamic relation | Energy | Relation teacher |
|---|---:|---:|---:|---:|---:|---:|
| `L_mix` | yes | no | yes | yes | yes | yes through post score |
| `L_q` | no | yes | no | no | no | no |
| `L_relation(sg gamma)` | no through gamma | no | as currently required | yes | no | yes |

The post path is still future-supervised through the existing relation
teacher. The prior path remains exactly deployable. The candidate target,
edge accumulation, relation marginalization, energy factors, and agent/scene
normalization do not change.

Useful diagnostics may retain the names `PL_post` and `PL_prior`, but they
must be computed as detached responsibility-weighted summaries:

```text
PL_post_diag  = mean_c sum_z gamma_teacher_c,z PL^post_c,z
PL_prior_diag = mean_c sum_z gamma_teacher_c,z PL^prior_c,z.
```

They are diagnostics, not additional optimized terms.

## 8. Circular optimization audit

There is no within-backward circular gradient path:

```text
p, scores -> L_mix                       (gradients enabled)
p, scores -> gamma -> stop_gradient
                        |
                        +-> L_q -> q only
                        +-> L_relation -> relation teacher/prior only
```

gamma changes between optimizer steps as p and component scores improve. This
is the normal moving target of amortized posterior learning, analogous to an
enumerated E-step. It is not a computational gradient cycle.

The following variants are rejected:

- allowing `L_q` gradients through gamma: q could change the target-producing
  prior/scores rather than approximate it;
- allowing q-loss gradients through the prior logits embedded in q: this
  would reintroduce an indirect q-to-p shortcut;
- keeping q as a PL or relation-KL weight: this preserves the original
  winner-selection mechanism;
- adding both direct `L_mix` prior learning and q-to-p KL: duplicated and
  potentially self-reinforcing prior updates.

## 9. Remaining collapse and identifiability risks

The design removes the specific free-weight failure, but finite-mixture
marginal likelihood alone cannot guarantee use of all components.

### 9.1 Component starvation

If p_z and gamma_z become extremely small, the component receives very little
score gradient. A one-component solution remains a possible stationary/local
optimum when one flexible component explains all training examples. The new
objective makes assignments evidence-derived and auditable; it does not add a
uniform-use constraint.

### 9.2 Weak score separation

The unary is z-independent. z affects the composite score through sparse
dynamic relations and joint energy. If those mode-conditioned energies learn
nearly identical conditionals, gamma will equal p and q has no future-varying
target.

### 9.3 E=0 scenes

Under the frozen architecture, E=0 gives identical scores for every z.
Therefore gamma=p exactly and no future-specific scene mode is identifiable
from the Stage-A objective. The audit found 47/139 E=0 validation windows and
the earlier training diagnosis found 630/4110 E=0 training windows. Latent
health must be reported separately for E>0 and E=0 scenes. E=0 neutrality is
not by itself collapse.

### 9.4 Label switching

Mode indices have no semantic labels. Any permutation of the four modes is an
equivalent solution. This is ordinary mixture non-identifiability and is not a
failure as long as predictions and responsibilities transform consistently.

### 9.5 One future per observed history

The dataset generally provides one realized Y* for each X. It cannot directly
prove that p represents all counterfactual futures for an identical history.
Generalization across similar histories supplies the learning signal.

These risks are why a fresh run can still be rejected for latent collapse.
They are not a justification for silently adding entropy, balance, semantic
labels, larger R, larger networks, or arbitrary diversity losses.

## 10. Required code changes for a later implementation round

No code is changed by this document. A later approved implementation should
be limited to:

### `src/joint_goal_loss.py`

- add an FP32 scene reduction returning mode composite log-score `[C,Z]`;
- add the combined-score mixture NLL and responsibility calculation;
- add forward categorical distillation
  `KL(stopgrad(gamma)||q)`;
- allow relation KL to receive detached responsibility `[C,Z]` instead of q;
- keep the old objective helper available only for explicit legacy/checkpoint
  compatibility tests, not the new objective.

### `src/models/model.py`

- preserve the existing chunked creation of `local_prior/local_post`;
- obtain `log_score_prior/log_score_post [C,Z]` before mode reduction;
- combine them 0.5/0.5 and compute one `L_mix` and gamma;
- remove q from both PL weighting and relation-KL weighting;
- expose gamma/score diagnostics;
- rename loss keys unambiguously, for example
  `jdv2_mixture_pl`, `jdv2_posterior_distill`, and
  `jdv2_relation_kl`.

### `src/models/joint_dependency_v2/future_teacher.py`

- keep the existing future projection, GRU, and invariant scene mean;
- change the posterior head input from 256 history+future channels to the
  128-channel future scene representation;
- form q logits by adding detached prior logits to centered future-evidence
  logits;
- return evidence logits for diagnostics.

### Parser/checkpoint/diagnostics

- add an explicit objective version such as
  `jdv2_latent_objective=v2_marginal_responsibility`;
- save it in checkpoint architecture/training metadata and reject incompatible
  same-stage resume;
- record per-scene score, gamma, p, q, entropy, hard/soft usage, q/gamma KL,
  future-shuffle sensitivity, and E>0/E=0 stratification;
- do not change cache schema: candidates, graphs, coordinates, and teacher
  sidecars remain valid.

No changes are required in the sampler, diffusion denoiser, goal candidate
cache, dynamic-relation network, joint-energy network, or evaluation metrics.

## 11. Test plan

### Mathematical and tensor tests

1. Verify `log_score_prior/post` shape `[C,4]`, FP32 dtype, and finite values
   for variable N, C, and E.
2. Verify scene reduction equals a manual mean over agents for packed scenes.
3. Recover the old expected PL exactly from the new per-mode PL tensor and an
   arbitrary supplied mode distribution; this validates score extraction.
4. Compare mixture NLL and gamma with a brute-force four-mode calculation.
5. Verify `sum_z gamma=1` and the analytic gradients:
   prior-logit gradient `p-gamma`, PL gradient `gamma`.
6. Verify the combined 0.5/0.5 score uses one shared responsibility.
7. For E=0, verify all mode scores are equal, gamma equals p, outputs are
   finite, and no relation/energy tensor is densified.

### Stop-gradient tests

8. Backpropagating `L_q` changes only future-posterior parameters; it must not
   give gradients to p, unary, relation, energy, or gamma-producing tensors.
9. Backpropagating `L_mix` changes p and appropriate score modules but not the
   posterior head.
10. Relation KL must not give gradients to q or the scene prior through its
    detached responsibility weight.
11. Assert q is absent from every PL reduction.

### Future sensitivity and invariance tests

12. Holding p/X fixed and changing a synthetic Y* must change future evidence
    and q for a non-degenerate initialized test fixture.
13. Holding p and future representation fixed while changing
    `history_scene` must not change q; this proves removal of the direct
    shortcut.
14. Agent permutation, edge reversal, and packed-scene permutation tests must
    preserve scores, gamma, q, and total loss after inverse reindexing.
15. No mode is assigned a semantic label; mode-permutation equivariance must
    hold when all z-indexed parameters are permuted consistently.

### Integration/regression tests

16. Full Stage-A forward/backward is finite under FP32 and BF16 FP32 islands.
17. Peak tensor contracts remain O(E Z K^2 M) chunked, with no K^N object.
18. All-off V2 remains bitwise equal to legacy GDTS under the existing test.
19. Old Stage-A checkpoints cannot silently resume under the new objective
    version.
20. Fixed-seed and five-seed validation protocols remain unchanged.

## 12. Migration from the current Stage A

The protected epoch-11 checkpoint remains a read-only scientific reference.
Do not resume it for objective V2 training:

- its q/p heads and mode-specific branches were optimized under the collapsed
  objective;
- the posterior-head input shape changes;
- optimizer moments and best-selection state belong to the old objective.

Start a fresh Stage-A run from the same frozen legacy GDTS source checkpoint
and freshly initialized JDV2 Stage-A modules. The existing full ETH JDV2 cache
may be reused after validating manifest/source hashes because this proposal
does not change candidates, sparse graphs, coordinates, or future teacher
inputs. Assign a new run name, objective version, and checkpoint namespace.

Use seed 2035 first. Do not start Stage B, multi-seed training, or
hyperparameter search until the corrected Stage-A run passes the gates below.

## 13. Retraining protocol

Keep the prior formal Stage-A configuration except for the approved objective
version:

- same ETH train/validation split and full cache;
- same frozen Goal U-Net, history encoder, and GDTS denoiser;
- K=21, Z=4, M=4, energy rank=8, P=20;
- same optimizer, LR, scheduler, precision, seed 2035, and stage-local
  progress semantics;
- same beta 0-to-0.1 warm-up over the first 20% of completed Stage-A steps;
- fixed-seed validation for checkpoint selection plus five-seed repeated
  validation for final comparison;
- primary checkpoint criterion remains mean/reproducible validation JFDE, not
  minFDE alone.

Log, stratified by E>0/E=0 where meaningful:

- `L_mix`, `L_q`, and relation KL;
- `log_score_prior/post` distributions and between-z variance;
- p, q, gamma entropy and hard/soft usage;
- `KL(gamma||q)`, diagnostic `KL(q||p)`, and mean `abs(q-p)`;
- future-shuffle q and gamma sensitivity;
- every existing joint, marginal, energy, relation, runtime, and memory
  diagnostic.

Do not add marginal regularization in this retraining. Marginal preservation
is an acceptance decision, not a reason to alter the objective mid-run.

## 14. Acceptance and rejection criteria

All comparisons use the exact same validation split, 20 samples, fixed seed
set `{2035,2036,2037,2038,2039}`, metric implementation, and world conversion
as the engineering audit.

### 14.1 Engineering gate

Reject/stop for NaN, Inf, OOM, cache/checkpoint mismatch, non-finite gradients,
broken deterministic validation, failed checkpoint reload, or a failed
all-off regression.

### 14.2 Latent-health gate

Accept only if all of the following hold on E>0 validation scenes:

- q changes under compatible future shuffling by substantially more than the
  identical-input repeat/noise control; an audit-level value near the current
  `7.695e-11` is rejection;
- gamma has nonzero between-scene/between-future variation and q tracks it;
- p/q/gamma are not jointly near-delta on the same single mode for essentially
  every scene;
- fixed-z interventions remain capable of changing joint goal allocations.

A high-probability mode or a rarely used mode alone is not rejection. Reject
the specific combined collapse signature: maximum aggregate soft usage above
0.99 together with near-zero entropy, near-zero future sensitivity, and
single-mode hard usage on at least 95% of E>0 scenes. Do not require uniform
usage.

E=0 scenes are reported separately and are not required to identify z.

### 14.3 Joint-prediction gate

Relative to exact same-protocol GDTS, the five-seed means must retain
improvement in:

- JFDE;
- JADE;
- Relative Motion Error;
- Joint Goal Endpoint Error.

Report Joint Goal Compatibility as an additional diagnostic. Do not accept a
latent-health improvement that removes the established joint-modeling gain.

### 14.4 Marginal-preservation gate

Report exactly:

```text
relative minADE degradation
  = (JDV2 minADE - GDTS minADE) / GDTS minADE

relative minFDE degradation
  = (JDV2 minFDE - GDTS minFDE) / GDTS minFDE.
```

Both should be no worse than +2% at final five-seed evaluation. A result above
+2% is a method-risk rejection/pause requiring review, not permission to add a
new loss. The current epoch-11 reference is `+0.91%` minADE and `+13.09%`
minFDE, so its marginal endpoint behavior does not pass this gate.

## 15. Explicitly excluded primary fixes

The following are not part of objective V2:

- entropy bonus;
- uniform usage or balance loss;
- manual mode quotas;
- hard semantic mode labels;
- larger Z/R or M;
- larger hidden dimensions;
- Transformer or additional GNN;
- arbitrary diversity/separation regularizer;
- new marginal-preservation loss.

If mixture component starvation persists after a correctly implemented and
measured run, the result should be reported as a remaining identifiability
failure. Any later fallback, including an information constraint or
anti-starvation schedule, requires a separate theoretical decision and cannot
be smuggled into this implementation.

## 16. Final answers

### Is the proposal theoretically self-consistent?

Yes, for the explicitly defined scene-balanced composite score. It performs
exact finite marginalization over Z=4, derives its responsibility from the
same score, trains q by detached posterior distillation, and uses p alone at
inference. It must be described as a latent mixture of pseudo/composite
conditionals, not as an exact K^N joint likelihood.

### Can it be implemented with minimal changes?

Yes. The expensive `[N,Z,K]` conditionals and chunked edge energies already
exist. Implementation requires exposing a `[C,Z]` reduction, adding a
four-mode `logsumexp`, changing gradient routing, and replacing the
history+future posterior head input with the existing future representation.
No large module or sampler/cache/evaluation change is needed.

### Does latent-collapse risk remain?

Yes. The correction removes q/p as arbitrary expected-loss selectors and
removes the history shortcut, but ordinary mixture marginal likelihood can
still starve components. Risk comes from weak mode-score separation,
z-independent unary scores, sparse/E=0 graphs, flexible components that can
explain all examples with one mode, a single realized future per history, and
ordinary label non-identifiability. The retraining acceptance gates are
therefore mandatory.
