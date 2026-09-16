# RSJG Joint Dependency V2 Final Implementation Specification

## Goal

Implement a clean joint dependency prediction framework on top of GDTS.

Pipeline:

GDTS backbone -\> Scene latent z -\> Hypothesis-conditioned relation
r_ij -\> Sparse low-rank joint goal energy -\> Structured joint sampling
-\> Frozen GDTS denoiser + dependency residual

## Fixed architecture

-   agent feature dimension: 128
-   scene latent modes: 4
-   relation modes: 4
-   goal candidates: 21
-   joint samples: 20
-   energy rank: 8
-   dependency hidden dimension: 64

## Existing backbone

Keep SocialMotionEncoder unchanged.

Output: h_i in R\^128

It already provides: - temporal encoding - sparse social message
passing - permutation equivariance

## Scene latent module

Attention pooling:

score: 128 -\> 64 -\> 1

alpha_i = softmax(score_i)

h_scene=sum(alpha_i\*h_i)

append: log(1+number_of_agents)

Scene network:

Linear(129,128) SiLU Linear(128,128) LayerNorm

Output: p(z\|X)

Training teacher:

Future trajectory encoder:

Linear(4,64) GRU(64,128)

Posterior:

q(z\|X,Y\*)

Loss:

KL(q\|\|p)

## Unary goal residual

Candidate feature:

\[goal_x-last_x, goal_y-last_y, distance, log_prior\]

Goal encoder:

4 -\> 64 -\> 64

Residual head:

192 -\> 64 -\> 1

Final:

u(k)=log(p0(k))+delta_u(k)

Initialize last layer to zero.

## Dynamic relation

Reuse existing relation encoder.

Additional goal geometry:

-   relative endpoint distance
-   direction cosine
-   formation difference
-   displacement difference

Geometry encoder:

4 -\> 32 -\> 16

Relation score:

base_relation + scene_mode_bias + geometry_query

p(r\|X,z,g_i,g_j)=softmax(score)

## Joint goal energy

Pair context:

agent_i + agent_j + edge feature

MLP:

270 -\> 128 -\> 64

Goal factor:

64 -\> rank 8

Energy:

E=-A\^T B/sqrt(8)

Use logsumexp for relation marginalization.

## Joint sampler

No neural network.

Steps:

1.  sample scene mode
2.  unary initialization
3.  two rounds parallel conditional refinement

Never enumerate K\^N.

## Dependency corrector

Freeze GDTS denoiser.

epsilon_joint = epsilon_base + Delta_epsilon

Input:

relative position(2) relative velocity(2) distance(1) relation
embedding(16)

Total 21 dimensions.

Network:

21 -\> 64 -\> 64

Use diffusion timestep FiLM.

Gate/value message passing:

gate: 64 -\> 32 -\> 1

value: 64 -\> 64 -\> 64

Aggregation:

sum(alpha\*v)/(sum(alpha)+eps)

Output:

64 -\> 64 -\> 2

Initialize final layer with zeros.

Therefore initial state:

epsilon_joint = epsilon_base

## Ablation switches

Implement:

use_scene_latent use_dynamic_relation use_joint_energy
use_dependency_corrector

## Cache

Precompute:

-   social features
-   graph edges
-   goal candidates
-   heatmap priors
-   teacher descriptors

Avoid repeated expensive computation.
