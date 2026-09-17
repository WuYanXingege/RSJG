# RSJG Joint Dependency V2

# Theory and Design Rationale

## 1. Overview

RSJG Joint Dependency V2 extends GDTS from marginal trajectory
prediction to joint multi-agent prediction.

Original marginal prediction:

\[ P(Y_i\|X_i) \]

Target joint prediction:

\[ P(Y_1,Y_2,...,Y_N\|X) \]

GDTS remains responsible for individual trajectory generation and
diffusion denoising. The proposed modules model cross-agent dependency.

Final diffusion formulation:

\[
`\epsilon`{=tex}*{joint}=`\epsilon`{=tex}*{GDTS}+`\Delta`{=tex}`\epsilon`{=tex}\_{dependency}
\]

------------------------------------------------------------------------

# 2. Motivation

Independent prediction can generate individually reasonable trajectories
but inconsistent joint futures.

Therefore the model should capture:

\[ P(Y_i,Y_j\|X) \]

instead of only:

\[ P(Y_i\|X_i) \]

------------------------------------------------------------------------

# 3. Core Principles

## 3.1 Preserve GDTS

GDTS already provides:

-   multimodal goal prediction
-   diffusion trajectory generation

The new framework adds dependency reasoning without replacing the
generator.

## 3.2 Future-conditioned interaction

Traditional relation:

\[ P(r\_{ij}\|X) \]

only uses history.

RSJG models:

\[ P(r\_{ij}\|X,z,g_i,g_j) \]

where:

-   z is global scene future mode
-   g_i,g_j are future goals
-   r_ij is latent interaction state

## 3.3 Avoid combinatorial explosion

The full joint goal space:

\[ K\^N \]

is intractable.

The framework uses:

-   sparse interaction graph
-   low-rank energy
-   structured sampling

------------------------------------------------------------------------

# 4. Architecture

Pipeline:

    History
      |
    GDTS encoder
      |
    Agent feature h_i
      |
    +----------------+
    |                |
    Scene latent z   Dynamic relation r_ij
    |                |
    +----------------+
            |
    Joint goal energy
            |
    Structured sampling
            |
    GDTS diffusion
            |
    Dependency correction

Three main components:

1.  Future-supervised scene latent.
2.  Hypothesis-conditioned dynamic relation.
3.  Dependency-aware diffusion correction.

------------------------------------------------------------------------

# 5. Scene Latent

Scene latent:

\[ z \]

represents global future evolution.

Prior:

\[ p(z\|X) \]

Teacher posterior:

\[ q(z\|X,Y) \]

Training objective:

\[ L_z=KL(q(z\|X,Y)\|\|p(z\|X)) \]

------------------------------------------------------------------------

# 6. Dynamic Relation

The relation module models:

\[ p(r\_{ij}\|X,z,g_i,g_j) \]

It combines:

-   historical interaction
-   future goal geometry
-   scene mode

Future geometry includes:

-   endpoint distance
-   direction similarity
-   formation change
-   displacement difference

Teacher:

\[ q(r\_{ij}\|X,Y) \]

------------------------------------------------------------------------

# 7. Joint Goal Energy

Independent goal selection may create conflicts.

Joint score:

\[ S(G)=`\sum`{=tex}*i u_i(g_i)-`\sum`{=tex}*{i,j}E\_{ij}(g_i,g_j) \]

where:

-   u_i represents individual goal confidence
-   E_ij represents compatibility cost

Low-rank approximation:

\[ E\_{ij}(k,l)=-A_i(k)\^TB_j(l) \]

------------------------------------------------------------------------

# 8. Structured Sampling

The model avoids enumerating:

\[ K\^N \]

Procedure:

1.  Sample scene mode.
2.  Initialize with unary score.
3.  Perform conditional refinement.

Output:

\[ G^1,G^2,...,G\^S \]

------------------------------------------------------------------------

# 9. Dependency Corrected Diffusion

Keep GDTS diffusion:

\[ `\epsilon`{=tex}\_{base} \]

Add:

\[
`\epsilon`{=tex}*{joint}=`\epsilon`{=tex}*{base}+`\Delta`{=tex}`\epsilon`{=tex}
\]

The corrector uses:

-   relative position
-   relative velocity
-   distance
-   relation embedding

The output layer is zero initialized:

\[ `\Delta`{=tex}`\epsilon=0`{=tex} \]

at initialization.

Therefore the model starts from GDTS and gradually learns interaction
correction.

------------------------------------------------------------------------

# 10. Design Logic

The framework forms a hierarchical dependency chain:

Global:

\[ z \]

answers:

"What future mode does the scene follow?"

Local:

\[ r\_{ij} \]

answers:

"How do two agents interact under this future?"

Continuous:

\[ `\Delta`{=tex}`\epsilon`{=tex} \]

answers:

"How should interaction modify trajectories?"

Therefore:

\[ Scene `\rightarrow `{=tex}Relation `\rightarrow `{=tex}Joint Goal
`\rightarrow `{=tex}Trajectory Correction \]

------------------------------------------------------------------------

# 11. Constraints

Must keep:

-   GDTS backbone
-   diffusion process
-   sparse graph
-   permutation invariance
-   zero initialization

Avoid:

-   unnecessary Transformer
-   dense K\^N search
-   redundant feature design
-   replacing GDTS

The objective is principled dependency modeling rather than simply
increasing network complexity.
