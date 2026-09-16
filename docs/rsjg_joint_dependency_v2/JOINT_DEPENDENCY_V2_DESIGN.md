# RSJG Joint Dependency V2 — Clean Research Design

Repository: `https://github.com/WuYanXingege/RSJG`

Target clean branch: `research/joint-dependency-v2-clean`

Base commit: `f9d4fefce5738f4e85942ce4fe8d9dc754d1f637`

## 1. Research target

The new branch is deliberately separated from the V5 / Selector / BASR line. The core question is:

> How can multimodal marginal pedestrian futures be organized into a tractable explicit joint distribution while preserving a strong marginal predictor and modeling global scene uncertainty, local hypothesis-dependent relations, and continuous joint realization in one coherent chain?

The network should contain only three necessary new mechanisms:

1. **Global scene latent variable** `z` for scene-level uncertainty.
2. **Hypothesis-conditioned latent relation** `r_ij` for local dependency under a concrete future hypothesis.
3. **Marginal-to-joint dependency correction** on top of the frozen GDTS denoiser.

The method should not add modules only for architectural novelty.

## 2. Final probabilistic structure

For a synchronized scene window:

\[
X=\{X_i\}_{i=1}^{N},\qquad Y=\{Y_i\}_{i=1}^{N}.
\]

GDTS provides per-agent goal candidates:

\[
C_i=\{g_i^1,\ldots,g_i^K\}.
\]

The proposed model is

\[
\boxed{
p(Y,G,R,z\mid X,S)
=
p(z\mid X,S)
\,p(G,R\mid z,X,S)
\,p(Y\mid G,R,X,S)
}
\]

where:

- `z`: global scene-level uncertainty;
- `R={r_ij}`: local sparse-edge latent dependencies;
- `G`: joint goal assignment;
- `Y`: full joint future trajectories.

The logical chain is

```text
Observed scene X
    |
    +--> GDTS Goal U-Net --> candidate support C_i + heatmap prior p0
    |
    +--> Sparse social encoder --> h_i
                                  |
                                  +--> scene latent p(z|X)
                                  |
                                  +--> hypothesis-conditioned relation
                                       p(r_ij|X,z,g_i,g_j)
                                                   |
                                                   v
                                         sparse low-rank joint goal energy
                                                   |
                                                   v
                                          joint goal distribution
                                                   |
                                                   v
                                         structured joint sampler
                                                   |
                                                   v
                                          K coherent joint worlds
                                                   |
                         +-------------------------+-------------------------+
                         |                                                   |
                  frozen GDTS denoiser                              dependency corrector
                     epsilon_base                                    Delta epsilon_dep
                         |                                                   |
                         +-------------------------+-------------------------+
                                                   |
                                      epsilon_joint = epsilon_base
                                                   + Delta epsilon_dep
                                                   |
                                                   v
                                         joint future trajectories
```

Training-only teachers:

\[
q_\phi(z\mid X,Y^*),\qquad q_\phi(r_{ij}\mid X,Y_i^*,Y_j^*)
\]

are removed at inference.

## 3. Keep GDTS as candidate support and marginal backbone

Do not replace the Goal U-Net or base diffusion in the first version.

For agent `i`, keep:

\[
C_i=\{g_i^1,\ldots,g_i^K\},
\]

and base log prior

\[
\ell_{0,i}(k)=\log p_0(g_i^k\mid X,S).
\]

The new method learns **joint dependency over this support**. Candidate support quality and joint dependency quality must be evaluated separately.

## 4. Global scene latent variable

### 4.1 Prior

Let `h_i` be the existing sparse-social agent feature.

Use minimal permutation-invariant pooling:

\[
a_i=w_a^\top\tanh(W_a h_i),
\]

\[
\alpha_i=\frac{\exp a_i}{\sum_{j\in scene}\exp a_j},
\]

\[
h_{att}=\sum_i\alpha_i h_i,
\]

and explicitly encode crowd density:

\[
n_s=\log(1+N_s).
\]

Then

\[
h_s=f_s([h_{att},n_s]),
\]

\[
\boxed{p_\theta(z\mid X)=\operatorname{softmax}(f_p(h_s))}.
\]

No extra scene Transformer, hypergraph, or multi-order GNN is needed in the main model.

### 4.2 Future-informed posterior

During training only, encode the GT future to produce

\[
q_\phi(z\mid X,Y^*).
\]

The future descriptor may contain only compact deterministic quantities:

- future relative displacement sequence;
- future velocity;
- endpoint displacement;
- path length;
- cumulative heading change.

Train with

\[
\boxed{
\mathcal L_z
=
D_{KL}\big[q_\phi(z\mid X,Y^*)\,\|\,p_\theta(z\mid X)\big]
}.
\]

Do not invent a semantic label for `z`.

## 5. Hypothesis-conditioned latent relation

The public RSJG relation is observation conditioned. Upgrade it to

\[
\boxed{
p_\theta(r_{ij}\mid X,z,g_i^k,g_j^l)
}.
\]

The same physical pair may have different relations under different future goal hypotheses.

### 5.1 Observation pair context

Reuse existing sparse graph features instead of duplicating them:

\[
c_{ij}^{obs}
=
\phi_{obs}
\big(h_i+h_j,\,|h_i-h_j|,\,e_{ij}^{obs}\big).
\]

### 5.2 Compact goal-pair geometry

Let

\[
d_i^k=g_i^k-x_i^{last}.
\]

Use only

\[
\phi_{ij}^{kl}
=
\left[
d_j^l-d_i^k,\,
\|g_j^l-g_i^k\|,\,
\cos(d_i^k,d_j^l)
\right].
\]

Relation logits:

\[
\ell_{ij,r}^{zkl}
=
f_r(c_{ij}^{obs},e_z,\phi_{ij}^{kl}),
\]

\[
\boxed{
p(r_{ij}=r\mid X,z,k,l)
=
\operatorname{softmax}_r\ell_{ij,r}^{zkl}
}.
\]

Relation modes remain unnamed latent categories.

### 5.3 Relation future teacher

Training only:

\[
q_\phi(r_{ij}\mid X,Y_i^*,Y_j^*).
\]

Use a compact GT pair descriptor:

\[
\phi_{ij}^{future}
=
[d_{min},\,t_{min},\,\Delta p_{final},\,\Delta formation].
\]

Train the deployable prior by

\[
\boxed{
\mathcal L_r
=
\sum_{(i,j)\in E}
D_{KL}
\left[
q_\phi(r_{ij}\mid X,Y^*)
\|\
p_\theta(r_{ij}\mid X,z,g_i^*,g_j^*)
\right]
}.
\]

## 6. Sparse joint goal distribution

### 6.1 Unary potential

Keep the GDTS heatmap prior and learn only a residual:

\[
\boxed{
u_i^z(k)=\ell_{0,i}(k)+\Delta u_\theta(h_i,e_z,g_i^k)}.
\]

### 6.2 Low-rank relation-specific pair energy

For sparse edge `(i,j)`:

\[
\boxed{
E_{ij}^{z,r}(k,l)
=
-
\frac{
A_{ij}^{z,r}(k)^\top B_{ij}^{z,r}(l)
}{\sqrt d}
}.
\]

Use shared factor networks so that swapping agent indices transposes the candidate-pair matrix rather than changing physical semantics.

### 6.3 Correct latent-relation marginalization

\[
\boxed{
E_{ij}^{z}(k,l)
=
-\log
\sum_r
p(r_{ij}=r\mid X,z,k,l)
\exp[-E_{ij}^{z,r}(k,l)]
}.
\]

Do not replace this with `sum_r p(r)E_r`.

### 6.4 Joint intention distribution

For assignment

\[
G=(g_1^{k_1},\ldots,g_N^{k_N}),
\]

define

\[
\boxed{
\log\tilde p(G\mid z,X)
=
\sum_i u_i^z(k_i)
-
\lambda_E\sum_{(i,j)\in E}E_{ij}^{z}(k_i,k_j)
}.
\]

Then

\[
p(G\mid X)=\sum_z p(z\mid X)p(G\mid z,X).
\]

This is the principal explicit joint-probability object of the method.

## 7. Tractable training: conditional pseudo-likelihood

Do not compute or claim an exact partition function over `K^N` assignments.

Construct a soft GT candidate target

\[
q_i^*(k)
\propto
\exp\left(
-\frac{\|g_i^k-g_i^*\|^2}{2\sigma_g^2}
\right).
\]

For neighbor `j`, use the soft expected pair energy

\[
\bar E_{ij}^{z}(k)
=
\sum_l q_j^*(l)E_{ij}^{z}(k,l).
\]

Conditional candidate score:

\[
s_i^z(k)
=
u_i^z(k)
-
\lambda_E\sum_{j\in\mathcal N_i}\bar E_{ij}^{z}(k).
\]

Conditional distribution:

\[
p(k_i=k\mid G_{-i},z,X)
=
\operatorname{softmax}_k s_i^z(k).
\]

Marginalize the training scene posterior:

\[
p(k_i=k\mid X,Y^*)
=
\sum_z
q_\phi(z\mid X,Y^*)
 p(k_i=k\mid G_{-i},z,X).
\]

Loss:

\[
\boxed{
\mathcal L_{goal}
=
-
\sum_i\sum_k
q_i^*(k)
\log p(k_i=k\mid X,Y^*)
}.
\]

Use scene-balanced reduction: average agents inside a scene, then average scenes.

## 8. Structured joint goal sampling

Never enumerate `K^N`.

For each requested joint sample `s`:

1. sample or stratify scene mode
   \[
   z^{(s)}\sim p(z\mid X)
   \]
2. initialize candidates using unary score;
3. apply 1–2 rounds of **Parallel Conditional Refinement**.

At refinement step `l`:

\[
score_i^{(l)}(k)
=
u_i^{z}(k)
-
\lambda_E
\sum_{j\in\mathcal N_i}
E_{ij}^{z}
(k,k_j^{(l-1)}).
\]

Then

\[
k_i^{(l)}\sim Cat(softmax(score_i^{(l)})).
\]

Do not call this exact Gibbs sampling because the update is synchronous.

Output:

\[
G^{(1)},\ldots,G^{(S)},
\]

where every sample column is a coherent joint world.

## 9. Continuous joint realization

### 9.1 Frozen marginal denoiser

Keep the original GDTS denoiser:

\[
\epsilon_i^{base}
=
\epsilon_{GDTS}(Y_{i,t_d},t_d,X_i,g_i).
\]

### 9.2 Learn only cross-agent dependency residual

\[
\boxed{
\epsilon_i^{joint}
=
\epsilon_i^{base}
+
\Delta\epsilon_i^{dep}
}.
\]

Do not claim that an arbitrary residual vector field is an exact score decomposition. Call it a **joint denoising dependency residual**.

### 9.3 Relation bottleneck

Do not feed `h_i,h_j,g_i,g_j,z` directly into the corrector again.

Use the soft relation embedding

\[
e_{ij}^{rel}=\sum_r p(r_{ij}=r)e_r.
\]

At future timestep `tau`, use only

\[
\Delta Y_{ij,\tau}=Y_{j,\tau}-Y_{i,\tau},
\]

\[
\Delta V_{ij,\tau}=V_{j,\tau}-V_{i,\tau}.
\]

Construct

\[
c_{ij,\tau}
=
\phi_c(
\Delta Y_{ij,\tau},
\Delta V_{ij,\tau},
\|\Delta Y_{ij,\tau}\|,
e_{ij}^{rel}
).
\]

Encode the diffusion noise step separately:

\[
e_d=TimeEmbed(t_d).
\]

FiLM modulation:

\[
\tilde c_{ij,\tau}
=
\gamma(e_d)\odot c_{ij,\tau}+\beta(e_d).
\]

Dynamic interaction gate:

\[
\boxed{
\alpha_{ij,\tau}
=
\sigma(w_\alpha^\top\tilde c_{ij,\tau})
}.
\]

Message value:

\[
v_{ij,\tau}=W_v\tilde c_{ij,\tau}.
\]

Normalized sparse aggregation:

\[
\boxed{
m_{i,\tau}
=
\frac{
\sum_{j\in\mathcal N_i}\alpha_{ij,\tau}v_{ij,\tau}
}{
\epsilon+\sum_j\alpha_{ij,\tau}
}
}.
\]

Residual:

\[
\Delta\epsilon_{i,\tau}^{dep}
=
W_0\rho(m_{i,\tau}).
\]

Zero initialize

\[
\boxed{W_0=0}.
\]

Therefore at initialization:

\[
\boxed{\epsilon^{joint}=\epsilon^{GDTS}}.
\]

For zero-edge agents:

\[
\boxed{\Delta\epsilon^{dep}=0}.
\]

Do not add a second temporal transformer. GDTS already models temporal structure; the new corrector is only responsible for sparse cross-agent dependency.

## 10. Training losses

Joint-intention stage:

\[
\boxed{
\mathcal L_{JG}
=
\mathcal L_{goal}
+
\beta_z\mathcal L_z
+
\beta_r\mathcal L_r
}.
\]

Do not add entropy/balance regularizers by default. Add them only if collapse is observed.

Joint-trajectory stage:

\[
\mathcal L_{diff}=\|\epsilon-\epsilon^{joint}\|^2.
\]

Optional small relative-motion auxiliary:

\[
\Delta \hat Y_{ij}=\hat Y_i-\hat Y_j,
\qquad
\Delta Y_{ij}^{*}=Y_i^*-Y_j^*,
\]

\[
\boxed{
\mathcal L_{rel}
=
SmoothL1(\Delta\hat Y_{ij},\Delta Y_{ij}^*)
}.
\]

Final:

\[
\boxed{
\mathcal L_{JT}
=
\mathcal L_{diff}
+
\lambda_{rel}\mathcal L_{rel}
}
\]

with small initial `lambda_rel`, e.g. 0.05–0.1.

## 11. Three paper-level contributions

### Contribution 1 — Global–local latent dependency decomposition

Scene latent `z` represents global future uncertainty. Hypothesis-conditioned relation `r_ij` represents local future dependency. Training-only future posteriors teach both latent spaces without manual relation labels.

### Contribution 2 — Tractable explicit joint intention distribution

A sparse low-rank energy model defines an explicit joint goal distribution over the GDTS multimodal candidate support. Conditional pseudo-likelihood and structured sampling avoid the `K^N` state space.

### Contribution 3 — Marginal-to-joint dependency correction

A zero-initialized sparse time-varying cross-agent denoising residual converts a strong marginal GDTS decoder into a joint trajectory generator without rebuilding the temporal backbone.

These three contributions are one logical chain, not three unrelated tricks.

## 12. Expected metric logic

The architecture is not primarily designed to produce a dramatic marginal minADE/minFDE gain.

Expected validation pattern:

- `minADE/minFDE`: stable or slightly improved;
- `JFDE`: primarily improved by joint intention modeling;
- `JADE`: further improved by the dependency corrector;
- relative-motion error: improved by the dependency corrector;
- diversity: preserved;
- collision rate: auxiliary diagnostic only.

The important empirical claim is:

> preserve strong marginal quality while improving joint consistency.

## 13. V4 positioning

Keep V4 unchanged as a clean post-hoc coupling baseline:

```text
GDTS                        no joint intention, no joint realization
Current JointGoal           joint intention only
V4                          post-hoc fixed-marginal coupling
Current JointGoal + V4      intention + post-hoc coupling
Joint Dependency V2         joint intention + joint continuous realization
```

This comparison directly tests where joint dependence should be introduced.
