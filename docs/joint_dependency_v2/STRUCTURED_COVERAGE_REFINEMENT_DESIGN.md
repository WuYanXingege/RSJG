# RSJG Joint Dependency V2 — Structured Coverage-Constrained Refinement Design

Date: 2026-09-20

Architecture: `strict_no_z`

Task: `STRUCTURED_COVERAGE_CONSTRAINED_REFINEMENT_DESIGN`

Status: **design only; no source change, training, production sampler change, or Stage B**

## 1. Decision summary

The canonical production candidate is **Coverage-Preserving Structured
Refinement (CPSR)**, implemented as a stochastic Gumbel-perturbed
maximum-weight injective assignment at each refinement round.

For each agent and refinement round, CPSR keeps the existing learned
conditional score matrix, applies the already configured sampling temperature,
adds independent unit-scale Gumbel perturbations, and solves one rectangular
assignment over all `P` slots:

```text
effective_logits = conditional_score / existing_temperature
perturbed_logits = effective_logits + unit_Gumbel_noise
candidate_ids = maximum_weight_injective_assignment(perturbed_logits)
```

The feasible set, rather than a new score term or loss, enforces finite-sample
coverage. With the current `P=20`, `K=21`, every agent receives 20 distinct
candidate IDs per round. All agents compute their score matrices from the old
round's world configuration and are then updated synchronously. Round 0 remains
the already validated Gumbel-Top-P weighted sampling without replacement;
rounds 1 and 2 use CPSR.

This is an inference-only policy with no trainable parameters. Code inspection
confirms that `ParallelConditionalSampler` runs only in the `if_test=True`
JDV2 path, while Stage-A losses call the score-producing modules without
sampling. The protected strict no-z epoch-13 checkpoint can therefore be used
directly. No Stage-A retraining is required or justified before the proposed
inference-only validation.

The stochastic construction must be called **stochastic perturb-and-MAP
structured assignment** or **Gumbel-perturbed maximum-weight matching**. It is
not claimed to be exact Gibbs sampling over assignments.

## 2. Frozen evidence and design objective

The design is based on the protected strict no-z epoch-13 checkpoint:

| Item | Value |
|---|---|
| Checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| K / P / M / energy rank | 21 / 20 / 4 / 8 |
| Split used by audits | ETH validation, 139 windows |
| Audit seeds | 2035, 2036, 2037, 2038, 2039 |
| Architecture | `strict_no_z` |

Weighted-without-replacement initialization provides a clean 20-candidate
round-0 set, but current independent refinement changes E>0 mean unique counts
as follows:

```text
20.000 -> 12.064 -> 11.749
```

Round 1 accounts for 96.2% of the unique-count loss and 89.4% of the oracle
degradation. This is not a one-hot conditional-score failure: round-1 entropy
is 2.615 nats versus `log(21)=3.045`, and mean maximum probability is only
0.163. Instead, broad distributions across slots share preferred candidates:
only 6.916 unique conditional argmax IDs exist before round-1 sampling.

The mechanism audit isolates the update rule:

- independent MAP intensifies collapse and gives minFDE 0.745838;
- current stochastic categorical refinement retains some exploration but
  destroys coverage and gives minFDE 0.432770;
- deterministic injective ASSIGN uses the same scores, retains 20 candidates
  in both rounds, and gives minFDE 0.383394 and JFDE 0.699226;
- ASSIGN is 2.79% better than same-protocol GDTS in minFDE and improves JFDE,
  endpoint, compatibility, and relative-motion error relative to current R2,
  with only a 0.57% JADE increase.

The objective is therefore not generic diversity maximization. It is:

> perform sample-set-level structured allocation of the existing
> neighbor-conditioned conditional compatibility scores subject to a
> finite-sample coverage constraint.

This targets how the finite `P=20` sample budget is allocated. It does not
change the learned social compatibility model.

## 3. Verified current implementation semantics

### 3.1 Call chain

The current JDV2 path is:

```text
GDTS._jdv2_encode(if_test=True)
  -> GDTS._jdv2_goal_outputs(sample=True)
  -> ParallelConditionalSampler.forward(...)
  -> _categorical(...)
```

`GDTS._jdv2_encode` passes `sample=if_test`. During Stage-A loss construction,
the sampler is not invoked. The sampler has no parameters and its choices do
not enter the Stage-A pseudo/composite-likelihood objective. A sampler policy
change is therefore inference-only for the frozen checkpoint.

### 3.2 Actual conditional score

For agent `i`, slot `s`, and candidate `k`, the source constructs the FP32
score

```text
S_i[s,k]
  = unary_score_i[k]
    - sum of selected effective pair-energy contributions
      from edges incident to i under old-round world slot s.
```

In source tensor form:

```text
accumulated.index_add_(0, src, source_energy)
accumulated.index_add_(0, dst, destination_energy)
conditional_score = unary_score[:, None, :] - accumulated
```

Shapes for strict no-z are:

| Object | Shape |
|---|---|
| unary score | `[N,K]` |
| current candidate IDs | `[N,P]` |
| conditional score | `[N,P,K]` |
| candidate mask | `[N,K]`, expanded to `[N,P,K]` |
| structured result | `[N,P]` |

The current categorical primitive samples from
`softmax(S_i[s,:] / T)`, where `T` is the existing
`joint_sampling_temperature`. CPSR must preserve exactly this score and
temperature meaning.

### 3.3 Synchronous update contract

Within a round, both direction-specific relation distributions and pair
energies are computed from the same old `selected [N,P]`. The new IDs are
written only after every agent's score matrix has been computed. CPSR must
retain this Jacobi/parallel contract; it must not update one agent and expose
that result while scoring a later agent.

## 4. Mathematical formulation

### 4.1 Feasible assignment set

For one agent `i`, let `S_i in R^(P x K)` be its current conditional score and
let `v_i[k] in {0,1}` denote the agent-level candidate validity. CPSR V1
requires this mask to be slot-invariant: the expanded mask must equal
`v_i[None,:]` for every slot. Define

```text
F_i = { A in {0,1}^{P x K} :
        sum_k A[s,k] = 1            for every slot s,
        sum_s A[s,k] <= 1           for every candidate k,
        A[s,k] <= v_i[k]             for every (s,k) }.
```

CPSR V1 deliberately does not support slot-varying validity. A full expanded
mask may be passed at the low-level boundary only to verify that all rows are
identical; otherwise the operation fails explicitly. A feasible assignment
requires at least `P` valid candidate columns for each agent. `P<=K` alone is
necessary but not sufficient when masking exists.

### 4.2 Deterministic ASSIGN diagnostic

The successful audit intervention solves

```text
A_i* = argmax_{A in F_i} sum_{s,k} A[s,k] S_i[s,k].
```

This exactly describes the audited rectangular Hungarian ASSIGN operation:
one candidate per slot, capacity one per candidate, invalid entries forbidden,
and the original conditional score unchanged. The audit's CPU implementation
uses `cost=-score` with `scipy.optimize.linear_sum_assignment`.

Multiplying all valid scores by the same positive `1/T` does not change this
deterministic argmax. Temperature matters once perturbations are added.

### 4.3 Canonical stochastic production candidate

Let

```text
L_i[s,k] = S_i[s,k] / T
```

be the same temperature-scaled logits used by current categorical sampling.
For every valid `(s,k)`, draw independent

```text
G_i[s,k] ~ Gumbel(0,1).
```

Then solve

```text
A_i* = argmax_{A in F_i}
       sum_{s,k} A[s,k] (L_i[s,k] + G_i[s,k]).
```

Invalid entries remain infeasible; they are not made merely unlikely. There
is no new perturbation temperature, diversity coefficient, coverage bonus, or
energy normalization. Unit Gumbel noise is added after dividing by the
existing temperature exactly once.

Equivalently, multiplying the entire objective by positive `T` gives
`S + T*G`, but the implementation should use `S/T + G` to mirror the existing
categorical semantics and avoid ambiguity.

### 4.4 Relation to categorical sampling

For a single slot (`P=1`), Gumbel-max on `S/T` is exactly a categorical draw
with probabilities `softmax(S/T)`. If the column-capacity constraints are
removed, row-wise Gumbel-max produces the current family of independent
categorical decisions. CPSR retains those row-level logits and stochastic
perturbations, but couples the row decisions through a shared capacity-one
feasible set.

This does **not** make the result an exact Gibbs sample over complete
assignments with probability proportional to
`exp(sum A*L)`. Exact Gumbel-max over that discrete space would require an
independent Gumbel variable for every complete assignment. Edge-wise Gumbels
are shared by many assignments, so the induced distribution is a tractable
perturb-and-MAP approximation, not exact Gibbs sampling over matchings.

## 5. Candidate-design comparison

| Design | Stochastic | Hard feasibility | New training | New knobs | Assessment |
|---|---:|---:|---:|---:|---|
| Deterministic Hungarian | No | Yes | No | None | Strong diagnostic, but collapses deployed goal sampling to one deterministic set and does not measure policy variability. Not canonical production. |
| Gumbel-Hungarian perturb-and-MAP | Yes | Yes | No | None beyond existing `T` | **Recommended.** Directly extends categorical Gumbel-max semantics, preserves scores, is simple at 20x21, and is exactly testable. |
| Gumbel-Sinkhorn | Yes | Only after projection | Usually benefits from differentiable/training use | Sinkhorn temperature, iterations, projection rules | Unnecessary complexity here; approximate feasibility and extra hyperparameters provide no demonstrated benefit. |
| Soft diversity penalty + independent sampling | Yes | No | No | Penalty coefficient/form | Reject. It changes scores heuristically, has no hard coverage guarantee, and conflicts with the requirement that coverage reside in the feasible set. |
| General capacity-constrained assignment | Yes or no | Yes | No | Capacity rule | Useful for `P>K`, but unnecessary for the authorized `P=20<K=21` setting and introduces an unvalidated allocation policy. |

Canonical choice: **Gumbel-Hungarian / stochastic perturb-and-MAP injective
assignment**, exposed as `structured_gumbel_assignment` and described at the
method level as CPSR.

Deterministic Hungarian remains a validation reference/upper-bound diagnostic,
not an automatic deployment substitute. If stochastic CPSR is materially
worse than deterministic ASSIGN, the discrepancy must be investigated; the
system must not silently switch to deterministic deployment.

## 6. Canonical inference algorithm

### 6.1 Algorithm 1 — Weighted Coverage-Preserving Initialization

The validated round-0 policy is retained rather than reformulated.

```text
Input:
    unary score U [N,K]
    valid mask V [N,K]
    sample count P
    existing temperature T
    explicit initialization RNG stream

Require:
    T > 0
    every agent has at least P valid candidates

for each agent i:
    L_i[k] = U_i[k] / T
    G_i[k] = unit_gumbel(generator_init)
    perturbed_i[k] = L_i[k] + G_i[k]
    perturbed_i[not V_i] = -infinity
    ids_i[0:P] = topk(perturbed_i, k=P, largest=True)
    assert ids_i are valid and pairwise distinct

return ids [N,P]
```

The descending Top-P order supplies the initial world-slot ordering. It is a
weighted Plackett-Luce-style draw without replacement, not uniform sampling.

### 6.2 Algorithm 2 — Structured Stochastic Refinement

```text
Input:
    round-0 IDs C^(0) [N,P]
    unary scores, candidate bank, sparse graph, relation and energy modules
    valid mask V
    existing temperature T
    two explicit round RNG streams

for refinement round l in {1,2}:
    old_ids = C^(l-1)

    # All agents use the same old-round joint worlds.
    compute relation and pair-energy terms from old_ids
    accumulate incident edge energy for every agent/slot/candidate
    S [N,P,K] = unary[:,None,:] - accumulated_pair_energy

    for each agent i:
        require at least P valid candidate columns
        G_i [P,K] = unit_gumbel(generator_refinement[l])
        L_tilde_i = S_i.float() / T + G_i
        forbid invalid entries
        A_i = exact maximum-weight injective assignment(L_tilde_i)
        new_ids_i[s] = the unique k such that A_i[s,k] = 1

    # Synchronous commit after all A_i have been solved.
    C^(l) = new_ids

return C^(2) [N,P]
```

No dense `K^N` joint state is formed. Assignment couples only the `P` slots
for one agent at a time; learned social coupling still enters through each
slot's neighbor-conditioned score.

### 6.3 Why two rounds remain canonical

The old second round was not the root cause: 96.2% of coverage loss already
occurred in round 1. Under deterministic structured ASSIGN, both rounds retain
20 unique IDs, and the E>0 oracle changes only
`0.348167 -> 0.349889 -> 0.350198` while joint metrics remain strong. The
coverage constraint changes the safety of the second round by preventing its
independent updates from consuming support. Therefore CPSR retains the frozen
`L=2` design. A one-round change would confound the policy test and is not
supported as the primary repair.

## 7. `P<=K`, masks, and future `P>K`

The canonical CPSR contract is deliberately limited to injective assignment:

```text
P <= number_of_valid_candidates_for_every_agent.
```

For the current experiment this is `20<=21`. Parser/preflight validation must
reject `P>K`, and runtime validation must reject an agent with fewer than `P`
valid candidates. There is no silent fallback to categorical sampling,
replacement, uniform sampling, or deterministic Top-P.

If a future experiment requires `P>K`, it needs a separately reviewed
capacity model. Balanced integer capacities are the simplest extension, while
probability-informed capacities would introduce a new allocation semantics.
Neither is part of CPSR V1. Explicitly requiring `P<=K` is preferable to
adding unvalidated complexity now.

## 8. Permutation and isolation contracts

### 8.1 Slot permutation

The score, mask, and feasible set contain no positional slot feature. Let
`Pi` permute the `P` rows. Conditional on perturbations being permuted with
their rows,

```text
Assign(Pi S, Pi V, Pi G) = Pi Assign(S,V,G).
```

Thus the stochastic operator is pathwise slot-permutation equivariant when
the exogenous noise is coupled, and its distribution is equivariant when
fresh iid noise is drawn after permutation. Tests must inject/capture the
Gumbel tensor and permute it with the score; merely resetting a sequential
generator after changing tensor order is not a valid pathwise equivariance
test because it attaches different random variables to semantic rows.

### 8.2 Agent permutation

Assignments are solved independently per agent after the existing equivariant
sparse relation/energy aggregation. Conditional on permuting the per-agent
noise tensors with agents, the output IDs permute identically. With fresh iid
noise, the induced distribution remains agent-permutation equivariant. No
agent index enters the score or constraint.

### 8.3 Ties

Deterministic Hungarian can have multiple optimal assignments, and an
implementation's row-order tie breaking can violate strict pathwise slot
equivariance. Continuous independent Gumbel perturbations make exact objective
ties probability zero in exact arithmetic. Production should:

- generate perturbations in FP32 and solve using FP64 CPU costs in the
  reference backend;
- never add a slot-index epsilon or lexicographic slot preference;
- expose perturbations to tests so they can be permuted with slots/agents;
- detect non-finite values and infeasibility;
- document that finite-precision exact ties are an exceptional numerical
  case, and test tied deterministic matrices only as solver behavior, not as
  a mathematically possible strict-equivariance guarantee for indistinguishable
  rows.

There is a fundamental symmetry issue for a deterministic injective map on
identical rows: different candidates must be assigned to otherwise
indistinguishable slots. Stochastic perturbations provide the appropriate
exchangeable symmetry breaking without giving slot indices semantic meaning.

### 8.4 Scene isolation and edge contracts

Current cache records are individual synchronized windows, and CPSR never
combines agents across windows. With packed scenes in any future call, the
operation must be grouped by scene before RNG derivation and assignment.
Scores retain the current canonical-edge source/destination accumulation, so
edge reversal, canonical edge validity, and sparse scene isolation contracts
are unchanged. The assignment itself does not inspect edges.

## 9. RNG specification

CPSR must not depend on an uncontrolled global RNG. Each `(evaluation seed,
window)` obtains three explicit device-compatible streams:

```text
generator_init
generator_refinement_round_1
generator_refinement_round_2
```

Subseeds must be derived by a documented stable integer mixing function from
the evaluation seed, stable window/cache ID, policy tag, and round tag; Python's
process-randomized `hash()` must not be used. Each stream is instantiated as a
`torch.Generator` on the score tensor's device. This makes one stage's random
consumption unable to shift another stage or diffusion.

Unit Gumbels are generated as:

```text
u = torch.rand(shape, generator=generator, device=device, dtype=float32)
u = clamp(u, eps, 1-eps)
g = -log(-log(u))
```

Diffusion retains its separate existing validation stream. For a fixed seed,
window, checkpoint, policy, and backend, structured samples must reproduce.
Different seeds must be able to produce different legal assignments.

The implementation should also allow a precomputed perturbation tensor in
low-level tests. This separates assignment correctness/permutation tests from
generator ordering and makes the equivariance contract exact and auditable.

## 10. Complexity and solver backend

Rectangular Hungarian matching for `P<=K` has approximately
`O(P^2 K)` complexity (or cubic scale in a padded implementation). At
`P=20`, `K=21`, this is roughly `20^2*21 = 8,400` primitive-scale matrix
operations per agent per round, about 16,800 across two rounds, excluding
constant factors and data movement. It does not introduce `K^N` complexity.

The compute count alone does not justify calling the overhead negligible.
`scipy.optimize.linear_sum_assignment` requires transferring each 20x21
matrix from GPU to CPU and transferring 20 IDs back. Synchronization and the
per-agent Python loop may cost more than the small matching computation.

The minimal correctness-first backend should be:

1. FP32 logits and Gumbels on the model device;
2. detached valid perturbed scores copied to CPU FP64;
3. exact SciPy rectangular assignment (`scipy>=1.9` is already declared);
4. assigned IDs copied back to the original device.

This backend is acceptable for the first inference-only validation because it
matches the audited solver semantics and requires no custom CUDA code. Formal
adoption must benchmark:

- sampler-only wall time per window and per agent;
- total end-to-end inference wall time;
- host/device synchronization time;
- peak CPU and CUDA memory;
- scaling over the observed ETH agent-count distribution.

It must be compared with current categorical R2 and deterministic ASSIGN. If
CPU synchronization is material, a batched Torch/GPU-compatible exact solver
may be proposed later, but only with parity tests against the reference solver.
No large CUDA kernel is part of this design.

## 11. Minimal future implementation plan

This section is a proposal for a separately approved implementation turn. No
listed change is made by this document.

### 11.1 `src/parser.py`

Add one explicit option:

```text
--jdv2_refinement_policy
    categorical                    # default; current tensor-exact behavior
    structured_gumbel_assignment   # CPSR
```

For `structured_gumbel_assignment`, preflight requires strict no-z inference,
`P<=K`, two rounds, and stochastic sampling mode. It must reject invalid
combinations rather than silently falling back. It adds no temperature;
`joint_sampling_temperature` remains the sole temperature.

### 11.2 `src/models/joint_dependency_v2/joint_sampler.py`

Add small, isolated helpers:

- weighted Gumbel-Top-P initialization;
- masked unit-Gumbel generation in FP32;
- maximum-weight injective assignment;
- structured Gumbel assignment over `[N,P,K]`;
- feasibility checks and optional trace fields.

Branch only at initial selection/refinement selection. The default
`categorical` branch must call the unchanged `_categorical` path and remain
tensor-exact. Conditional score construction, relation/energy calls, and
synchronous two-round control flow remain unchanged.

The structured branch accepts separate explicit initialization/round
generators (or precomputed perturbations in tests). It returns the same public
tensor/output keys as the current sampler, so diffusion and metrics require no
semantic change.

### 11.3 `src/models/model.py` and evaluation runner

Thread the configured policy and explicit per-window/per-round generators into
`ParallelConditionalSampler`. The validation/evaluation driver, which knows
the stable evaluation seed and window/cache ID, should construct the streams.
Training loss calls remain sampler-free.

Evaluation output must record:

- policy name;
- `P`, `K`, round count, and existing temperature;
- solver/backend and version;
- RNG derivation/version;
- checkpoint and cache hashes.

### 11.4 Checkpoint and config metadata

`sampler_policy` is inference behavior, not a weight-shape architecture field.
It must not be inserted into the strict weight `architecture_config` equality
in a way that prevents the protected epoch-13 checkpoint from loading.

Recommended compatibility rule:

- old checkpoints with no sampler-policy metadata mean their historical
  inference default was `categorical`;
- newly saved checkpoints/configs record an `inference_sampler_config` block;
- evaluation may explicitly override that block to
  `structured_gumbel_assignment`, and records the override in its result;
- resume/training semantics remain categorical unless explicitly authorized;
- weight compatibility continues to be decided by architecture/state dict,
  while inference-policy provenance is checked and reported separately.

This meets both requirements: future artifacts state which policy produced
their predictions, and the existing no-z checkpoint remains directly usable.

### 11.5 No changes elsewhere

Do not change the relation module, low-rank energy, unary network, Goal U-Net,
candidate bank, diffusion, loss, optimizer, checkpoint weights, K/P/M/rank,
or number of refinement rounds. No new neural module is introduced.

## 12. Required tests for a future implementation

1. For every agent with `P<=valid_K`, structured assignments contain `P`
   unique valid candidate IDs.
2. Every assignment row contains exactly one selected candidate.
3. Every candidate column has capacity at most one.
4. Invalid/masked candidates are never selected, including adversarially high
   invalid scores.
5. A fixed seed/window/round produces exactly repeatable Gumbels and IDs.
6. Different seeds can produce different valid assignments on a nondegenerate
   score matrix.
7. A tiny case such as `P=3,K=4` matches brute-force maximum assignment for a
   fixed perturbation matrix.
8. Effective logits are exactly `conditional_score / existing_temperature`;
   no second temperature or score modification appears.
9. Slot permutation plus the same permuted perturbation gives the identically
   permuted output.
10. Agent permutation plus the same permuted perturbations gives the
    identically permuted output.
11. Separate scenes/windows cannot affect each other's result or RNG stream.
12. Edge reversal/canonical-edge tests remain unchanged because score
    construction is unchanged.
13. E=0 and single-agent windows use coverage-preserving initialization and
    skip refinement safely.
14. `P>K` and `valid_K<P` fail explicitly before solving.
15. With no diagnostic/structured flag, categorical output is tensor-exact
    against the current production sampler for fixed RNG state.
16. Config, checkpoint metadata, and evaluation result record the policy while
    an old epoch-13 checkpoint still loads.
17. BF16 model execution keeps score, Gumbel generation, mask handling, and
    assignment cost conversion in the specified FP32/FP64 islands with no
    NaN/Inf or invalid IDs.
18. Round-1 and round-2 updates are synchronous: scores for all agents must be
    computed from old-round IDs.
19. Separate init/round/diffusion streams prove that changing assignment RNG
    consumption cannot shift diffusion noise.
20. The legacy/all-off GDTS and strict no-z default categorical regressions
    remain unchanged.

## 13. Proposed inference-only validation

The first implementation experiment must use the frozen epoch-13 checkpoint,
ETH validation's 139 windows, and seeds 2035–2039. It compares:

1. original strict no-z iid/categorical sampler;
2. deterministic ASSIGN diagnostic;
3. CPSR `structured_gumbel_assignment`.

Use the same candidate cache, model weights, conditional scores, relation,
energy, two rounds, diffusion, coordinate conversion, metric code, and paired
diffusion RNG. Report mean, population standard deviation, and per-seed:

- minADE and minFDE;
- JADE and JFDE;
- Joint Goal Endpoint Error;
- Joint Goal Compatibility;
- Relative Motion Error;
- round-0/1/2 unique counts and geometric oracle;
- joint-world diversity and conditional-score statistics;
- sampler and end-to-end runtime.

Continue E=0/E>0, degree, and agent-count stratification. E=0 verifies
initialization/diffusion; the primary structured-refinement conclusion comes
from E>0.

### 13.1 Production adoption gate

CPSR may be proposed for formal adoption only if:

- minFDE degradation versus exact same-protocol GDTS is at most +2%;
- JFDE, JADE, endpoint, and relative-motion error remain better than GDTS;
- no principal joint metric degrades systematically by more than 2% versus
  the original strict no-z sampler;
- all feasibility, RNG, permutation, metadata, and legacy regressions pass;
- measured runtime overhead is reported and accepted.

If stochastic CPSR is materially worse than deterministic ASSIGN, do not
quietly weaken the coverage constraint, tune an undocumented perturbation
scale, or deploy deterministic ASSIGN. First attribute whether the gap comes
from stochastic perturbation, solver/RNG semantics, or downstream sampling.

Passing this inference-only gate would justify formally freezing Stage A and
requesting approval to enter Stage B. It would not itself start Stage B; that
remains a separate human decision.

## 14. Interpretation and boundaries

### 14.1 Why independent refinement fails

The current operator makes `P` decisions independently even though they are
the finite representation of one sample set. The learned score distributions
are broad but aligned across slots, so independent MAP concentrates severely
and independent categorical draws repeatedly spend slots on shared preferred
candidates. The operator optimizes local slot compatibility without accounting
for the opportunity cost of consuming finite sample support.

### 14.2 Why the assignment constraint addresses the observed mechanism

The constraint forces slots to compete for candidates. A candidate can be
allocated only once within an agent's sample set, while other slots retain
different candidates. This performs structured allocation of the complete
matrix of neighbor-conditioned conditional compatibility scores rather than
adding a heuristic bonus. It does not globally maximize the full multi-agent
joint compatibility objective. The deterministic audit demonstrates that the
existing scores contain enough slot-specific information: structured
allocation preserves R0-like coverage and R2-like social coherence without
retraining.

### 14.3 Marginal versus joint interpretation

- R0 shows that complete support protects marginal coverage but lacks some
  learned social coherence.
- R2 shows that social conditional scores improve joint coherence but that
  independent updates destroy marginal support.
- ASSIGN shows that these are not necessarily an inherent trade-off: the same
  scores can achieve both when the sample set is updated jointly.

CPSR therefore means **coverage-constrained joint hypothesis refinement**, not
diversity regularization. It changes finite-budget allocation, not the learned
energy or relation semantics.

## 15. Known residual risks

1. **Perturb-and-MAP bias.** Edge-wise Gumbels do not sample the exact Gibbs
   distribution over complete assignments. The induced policy may differ from
   both independent categorical sampling and deterministic ASSIGN.
2. **Perturbation sensitivity.** Unit Gumbel noise on `S/T` is canonical but
   may weaken matching quality. The existing temperature is the only allowed
   control; no new hidden scale should be introduced.
3. **Over-strong uniqueness.** Capacity one spends one slot on almost every
   candidate at `P=20,K=21`. This suits best-of-P coverage but may underrepresent
   high-probability modes or harm calibration on other datasets/budgets.
4. **Mask infeasibility.** Fewer than P valid columns must stop explicitly.
5. **CPU synchronization.** Reference SciPy matching may add meaningful
   latency even though the matrices are small.
6. **Finite-precision ties.** Gumbels make ties probability-zero theoretically,
   but implementation precision and solver behavior still require tests.
7. **Two-round dynamics.** Injectivity prevents candidate duplication but does
   not prove convergence or eliminate oscillation between round assignments.
8. **Metric dependence on one trained checkpoint.** Five inference seeds do
   not establish cross-training-seed or cross-dataset generality.
9. **Sample calibration.** Best-of-P metrics reward coverage; a structured
   support set should not be described as calibrated iid samples from the
   learned marginal.

These risks are reasons for the prescribed inference-only validation, not for
adding losses, retraining, or expanding the model.

## 16. Naming candidates

1. **Coverage-Preserving Structured Refinement (CPSR)** — recommended;
   emphasizes the verified mechanism and sample-set update without naming a
   particular solver.
2. Coverage-Constrained Parallel Refinement (CCPR) — accurately preserves the
   synchronous update contract but is less explicit about the sample set.
3. Structured Joint Hypothesis Refinement (SJHR) — broad and paper-friendly,
   but does not state the coverage constraint.
4. Structured Coverage-Aware Refinement (SCAR) — concise, but “aware” can
   imply a soft heuristic rather than hard feasibility.
5. Coverage-Constrained Joint Hypothesis Refinement (CCJHR) — precise but less
   readable.

Recommended paper-level name: **Coverage-Preserving Structured Refinement
(CPSR)**. Recommended configuration value:
`structured_gumbel_assignment`.

## 17. Direct answers

**Q1. Why does independent refinement fail on the current data?**  Broad
slot-conditional distributions have highly aligned rankings. Independent
updates ignore finite-set opportunity cost, so many slots choose the same few
candidates. Round 1 therefore destroys most coverage even though scores are
not one-hot.

**Q2. Why can the assignment constraint solve the observed mechanism?**  It
jointly allocates the unchanged score matrix under capacity one, forcing slots
to compete for candidates. The audited ASSIGN result proves that those scores
support high coverage and social coherence when used structurally.

**Q3. Why is deterministic Hungarian not automatically the final production
policy?**  It produces one fixed maximum-weight set, removes deployed
goal-sampling stochasticity, and does not characterize uncertainty across
valid high-scoring assignments. It is a mechanism diagnostic and reference,
not a stochastic policy.

**Q4. What is the recommended stochastic production refinement?**  Weighted
Gumbel-Top-P initialization followed by two synchronous rounds of unit-Gumbel
perturbation on the existing temperature-scaled conditional scores and exact
maximum-weight injective assignment: CPSR / `structured_gumbel_assignment`.

**Q5. Does it require Stage-A retraining?**  No. The sampler has no trainable
parameters and is absent from the Stage-A loss path. The protected epoch-13
weights can be evaluated directly.

**Q6. How much complexity does it add?**  About `O(P^2 K)` per agent per
round—roughly 8,400 primitive-scale operations for a 20x21 assignment, or
16,800 for two rounds—plus potentially important GPU/CPU synchronization.
Runtime must be benchmarked rather than assumed negligible.

**Q7. Does it preserve permutation and scene-isolation contracts?**  Yes in
distribution, and pathwise when perturbations are permuted with slots/agents.
The score, constraint, and solver objective use no slot/agent positional
semantics. Scene grouping, canonical edges, and synchronous old-round score
construction remain unchanged. Continuous stochastic perturbations resolve
the ordinary deterministic tie-breaking problem almost surely.

**Q8. How is `P>K` handled?**  CPSR V1 rejects it explicitly, as it also
rejects `valid_K<P`. Capacity greater than one requires a separately reviewed
extension; there is no silent fallback.

**Q9. What new failure modes remain?**  Approximate perturb-and-MAP sampling,
noise sensitivity, overly strong uniqueness/calibration effects, mask
infeasibility, CPU synchronization, finite-precision ties, and possible
two-round oscillation. These must be measured without changing the model.

**Q10. If inference-only validation approaches deterministic ASSIGN, may Stage
A be frozen and Stage B begin?**  Passing the stated marginal, joint,
regression, reproducibility, and runtime gates would support freezing Stage A
and submitting a separate request to enter Stage B. It does not authorize an
automatic Stage-B start.

## 18. Final design status

`DESIGN_COMPLETE_AWAITING_REVIEW`

No source code, test, training run, checkpoint, production sampler, or Stage-B
state is changed by this document.
