# Lexicographic Continuity Semantics Design Review

Date: 2026-09-21

Architecture: `strict_no_z`

Task: `LEXICOGRAPHIC_CONTINUITY_SEMANTICS_DESIGN_REVIEW`

Status: **design only; no source change, resolver implementation, training,
production sampler change, Stage-A freeze, or Stage B**

## 1. Decision summary

The unique recommended design is **Lexicographic Continuity-Preserving
Structured Refinement (LCPSR)** with an explicit no-interaction base case.

For each refinement round:

1. compute the existing conditional score once from the old joint worlds;
2. for `degree_i == 0`, keep `previous_ids_i` tensor-exact;
3. for `degree_i > 0`, maximize, in strict lexicographic order,

   ```text
   (J, C_stay, C_geom)
   ```

   over the unchanged injective feasible set;
4. synchronously commit all agents.

The objectives are:

```text
J(A) = sum_{s,k} A[s,k] S_i[s,k]

C_stay(A) = sum_{s,k} A[s,k] 1[k == c_old[s]]

C_geom(A) = -sum_{s,k} A[s,k]
                         ||g_i^k - g_i^(c_old[s])||_2^2.
```

This means that continuity can choose only among primary-optimal assignments.
It cannot trade any nonzero amount of learned conditional score for a better
stay count or a smaller geometric displacement. There is no coefficient,
epsilon, temperature, learned quantity, or candidate/slot-index tie breaker.

The correctness-first implementation candidate is a native exact
lexicographic assignment solver over tuple-valued edge weights. The solver
uses one canonical detached FP32 conditional-score snapshot, losslessly
interprets its IEEE-754 values as exact binary rationals (or equivalent
arbitrary-precision integers), and performs all lexicographic comparisons
exactly. The same treatment is applied to the tertiary geometry computed from
a canonical world-metre candidate snapshot.

`C_stay` and `C_geom` may still leave genuinely indistinguishable optima. In
that case no deterministic single-assignment selector can, in general, remain
equivariant to every symmetry of the input. The resolver must report an
**unresolved optimal equivalence class** rather than silently select by row or
column enumeration. A future real-data audit must measure this condition and
must stop before metric interpretation if a deployment-required matrix is
unresolved.

This proposal is not claimed to have fixed the observed pathology. It is the
canonical design for the next read-only intervention.

## 2. Evidence status and boundaries

The following labels are used throughout this review.

- **Verified experimental fact:** established by the accepted ETH audits.
- **Mathematical consequence:** follows from the stated optimization and
  permutation actions.
- **Design choice:** proposed here and not yet implemented or validated.
- **Unresolved risk:** cannot be removed by definition alone and must be
  tested.

### 2.1 Verified experimental facts

- Weighted Gumbel-Top-P Round 0 retains 20/20 candidates and repairs initial
  finite-slot coverage.
- Fresh refinement Gumbels preserve coverage but materially harm E>0 joint
  coherence relative to deterministic structured refinement.
- Shared-Round-0 deterministic refinement has strong marginal and joint
  metrics, but a common slot permutation changes approximately 44% of real
  unordered scene world sets.
- Degree-zero conditional rows are exactly slot-invariant. Policy D gives
  these agents 0% pathwise invariance; D0 identity raises it to 100%.
- D0 preserves Policy-D metrics and 20/20 coverage, but scene-world-set
  invariance remains only 62.68% / 62.17%.
- Non-singleton components remain unstable, especially size-two components
  at 44.53% / 42.27% exact component-world-set invariance.
- Existing objective differences under permutation were at FP32
  numerical-noise scale. The previous audits did not establish which cases
  are exact ties under a canonical exact representation.

### 2.2 Scope boundary

This review defines tie semantics only. It does not change:

- the learned conditional score;
- relation, energy, unary, Goal U-Net, candidate bank, or diffusion;
- `K=21`, `P=20`, `M=4`, rank 8, temperature, or two-round synchronization;
- Stage-A training or loss;
- the production categorical path or CPSR V1.

## 3. Why the current deterministic Hungarian is not a semantic resolver

For an agent, the current deterministic diagnostic asks SciPy for one element
of

```text
argmax_{A in F_i} J(A),
```

where `F_i` is the injective assignment set. When this set of maximizers has
more than one element, `linear_sum_assignment` is permitted to return any one
of them. Its choice is an implementation result, not part of the mathematical
model.

If rows are permuted, the solver can visit or order tied augmenting paths
differently. Per-agent candidate support can remain exactly the same while
candidate-to-slot pairing changes. Since slot `s` denotes a complete old
joint-world hypothesis, independently changing one agent's row-to-candidate
pairing changes the set of complete scene worlds. This is why support
invariance does not imply unordered joint-world-set invariance.

The problem is therefore not that deterministic optimization is inherently
invalid. The problem is that `argmax J` is set-valued while the current API
silently exposes one enumeration-dependent representative as if it were a
semantic choice.

## 4. Frozen primary problem

For interacting agent `i`, let:

- `S_i in R^(P x K)` be the unchanged conditional score;
- `V_i[k] in {0,1}` be the existing agent-level, slot-invariant mask;
- `c_old[s]` be the previous candidate assigned to slot `s`;
- `g_i^k in R^2` be candidate `k` in world metres.

The feasible set remains:

```text
F_i = { A in {0,1}^{P x K} :
        sum_k A[s,k] = 1                 for every slot s,
        sum_s A[s,k] <= 1                for every candidate k,
        A[s,k] <= V_i[k]                 for every (s,k) }.
```

The existing `P <= valid_K` requirement is unchanged.

The primary objective is:

```text
J(A) = sum_{s,k} A[s,k] S_i[s,k].
```

Define:

```text
J* = max_{A in F_i} J(A)
F_i^J = {A in F_i : J(A) = J*}.
```

**Design invariant:** every interacting-agent output must belong to `F_i^J`.
Neither continuity objective may select a non-primary-optimal assignment.

Temperature does not enter this deterministic semantics. Multiplying all
primary scores by the same positive `1/T` preserves the argmax, but the
reference design should use the raw canonical conditional score so its
primary objective is directly auditable against `J`.

## 5. Secondary objective: exact slot identity continuity

Define:

```text
C_stay(A) = sum_{s,k} A[s,k] 1[k == c_old[s]].

C_stay* = max_{A in F_i^J} C_stay(A)

F_i^(J,stay) = {
    A in F_i^J : C_stay(A) = C_stay*
}.
```

### 5.1 Why it is semantically appropriate

Each slot already represents one old joint-world hypothesis. A retained
candidate preserves that agent's contribution to the same world column. The
stay count therefore measures hypothesis continuity, not arbitrary array
continuity.

It has four useful properties.

1. **Primary preservation.** It is optimized only inside `F_i^J`.
2. **No learned-score modification.** `S` is not changed.
3. **No hyperparameter.** Stay count is an ordinal second objective, not a
   weighted bonus.
4. **Permutation semantics.** Its definition follows the paired semantic
   objects `(slot, previous candidate)` and uses neither slot number nor
   candidate number.

### 5.2 What it does not guarantee

`C_stay` is not sufficient to uniqueify all assignments. Two primary-optimal
assignments may retain the same number of previous candidates but retain
different slots or move the remaining slots differently. If neither can keep
any old assignment, both can have stay count zero.

Thus the fact that a solver returns one result after adding a stay-count stage
would not prove that the result is semantic or permutation-equivariant. The
remaining optimum set must be tested for uniqueness.

## 6. Tertiary objective: goal-geometry continuity

For assignments still tied after `C_stay`, define:

```text
C_geom(A) =
    -sum_{s,k} A[s,k]
                 ||g_i^k - g_i^(c_old[s])||_2^2.

C_geom* = max_{A in F_i^(J,stay)} C_geom(A)

F_i^lex = {
    A in F_i^(J,stay) : C_geom(A) = C_geom*
}.
```

The full interacting-agent semantics is:

```text
lexicographically maximize (J, C_stay, C_geom).
```

### 6.1 Coordinate choice

Use the existing `goal_candidates_world [N,K,2]` in world metres. Do not use
heatmap pixel coordinates or a newly normalized representation.

Squared Euclidean displacement is:

- translation invariant;
- invariant to rotations and reflections under an exact orthogonal transform;
- unchanged in ordering under a uniform positive unit conversion;
- directly interpretable as endpoint-hypothesis movement.

The computation must use one detached canonical coordinate snapshot. It must
not recompute coordinates separately after a slot or candidate permutation,
because separately rounded transformations can create a different numerical
problem.

Nonuniform axis scaling is not an invariance of Euclidean geometry. The
existing world-metre convention is therefore part of the semantic contract.

### 6.2 Role and limits

`C_geom` adds no learned quantity and cannot affect the primary or secondary
optimum. It provides a real hypothesis-continuity meaning for otherwise tied
slot moves: minimize total endpoint displacement from each old world column.

It still need not be unique. Duplicate candidates, equidistant candidates,
regular geometric symmetries, or graph/score symmetries can produce multiple
members of `F_i^lex` with identical tuples.

## 7. Slot- and candidate-permutation contracts

### 7.1 Slot permutation

Let `pi` be a common permutation of the P world slots. Apply it consistently
to:

- rows of `S`;
- rows of the expanded mask;
- `c_old`;
- the current joint-world columns.

For an assignment `A`, let `pi A` permute its rows. Then:

```text
J(pi A; pi S) = J(A; S)
C_stay(pi A; pi c_old) = C_stay(A; c_old)
C_geom(pi A; pi c_old, g) = C_geom(A; c_old, g).
```

Therefore the entire maximizer set is equivariant:

```text
F_lex(pi S, pi c_old, g) = pi F_lex(S, c_old, g).
```

If `F_i^lex` is a singleton, the selected assignment is pathwise
slot-equivariant. If it contains several elements, the set is equivariant but
an arbitrary single representative need not be.

### 7.2 Candidate enumeration permutation

Let `sigma` relabel candidate columns. Apply it consistently to:

- score columns;
- mask columns;
- candidate goal coordinates;
- every value in `c_old`;
- output candidate IDs.

The three objectives and feasibility constraints are preserved under this
bijection. Hence:

```text
F_lex(sigma S, sigma c_old, sigma g)
    = sigma F_lex(S, c_old, g).
```

Candidate numeric value, column position, topological sort order, and slot
index are not semantic data. Sorting rows or columns by candidate numeric ID
is therefore forbidden as a formal tie resolver.

### 7.3 Four distinct invariance notions

The future audit must continue to report these separately:

1. **Objective-level invariance:** optimal lexicographic tuple unchanged.
2. **Support invariance:** each agent retains the same unordered candidate
   set.
3. **Pathwise assignment invariance:** after undoing the permutation, every
   slot has the same candidate.
4. **Unordered joint-world-set invariance:** the complete P scene worlds are
   equal up to a permutation of whole columns.

Only the third property proves a unique per-agent semantic path; only the
fourth directly protects the downstream finite scene-hypothesis set.

## 8. Residual symmetry and the impossibility boundary

Even `(J, C_stay, C_geom)` can have several exact maximizers. Consider two
new candidates with identical score columns and symmetric/equal distances to
two old hypotheses. Swapping those candidates between the two slots changes
neither objective. More generally, an automorphism may simultaneously swap
indistinguishable rows or candidates while leaving every input semantic
quantity unchanged.

In such a case, assume a deterministic selector returns one assignment `A`.
Equivariance under the automorphism requires it to return the transformed
assignment `tau A`. But because the transformed input is semantically
identical, determinism also requires it to return `A`. If `tau A != A`, these
requirements contradict each other. No deterministic single-output selector
can resolve this symmetry without adding information.

The correct contract is therefore:

- require a unique pathwise output when `F_i^lex` is a singleton;
- preserve the previous assignment whenever it is lexicographically optimal
  (with injective previous IDs, `C_stay=P` already makes it the unique
  secondary optimum);
- when `|F_i^lex|>1`, report `AMBIGUOUS_LEXICOGRAPHIC_OPTIMUM` and expose an
  invariant certificate of the optimum tuple/equivalence class;
- never hide this state behind SciPy's row/column order.

For a deployment-required single-output API, unresolved real matrices are a
design failure, not permission to use an index tie breaker. A separately
reviewed source of semantic information or an explicitly stochastic,
exchangeable symmetry treatment would then be required. Neither is approved
here.

The recommended invariant object in a true symmetry is the maximizer
equivalence class, not an arbitrary pathwise representative. Unordered
joint-world-set equality remains the downstream contract whenever a unique
assignment is emitted.

## 9. Interaction-aware identity is a structural base case

The final design should retain:

```text
if degree_i == 0:
    return previous_ids_i
```

rather than force degree-zero agents through the lexicographic solver.

This is necessary because identical score rows do not guarantee that the
previous candidate support is primary-optimal. For example, weighted Round 0
may exclude a candidate that belongs to the unary top P. A unified
`(J,C_stay,C_geom)` resolver must first maximize `J` and could replace the
old support even though no interaction provides slot-specific evidence.

The structural branch says that social refinement is undefined/unnecessary
without incident social evidence, so the refinement operator is identity. It
does not redefine or approximate `J`; the primary assignment problem is
invoked only for the interacting branch. This exactly matches the accepted
D0 intervention and the existing E=0 scene behavior.

This distinction must be explicit in documentation and tests. It must not be
described as a diversity heuristic or as a primary-optimal assignment for an
agent whose previous support is not primary-optimal.

## 10. Exact numerical semantics

### 10.1 Canonical primary representation

For each round, compute the current conditional score once on the existing
path:

```text
S_fp32 = (unary.float()[:,None,:] - accumulated.float()).detach()
```

For the resolver:

1. copy this exact snapshot to CPU without recomputing relation, energy, or
   `index_add_`;
2. reject non-finite valid entries;
3. interpret every FP32 value by its exact IEEE-754 binary value;
4. convert the valid values to exact integers using a common power-of-two
   denominator for that matrix, or retain exact rational pairs;
5. sum and compare primary objectives exactly.

All finite binary floating-point numbers are rational with power-of-two
denominators. A common denominator therefore produces exact integer primary
weights. Python arbitrary-precision integers avoid overflow. The denominator
is derived from the multiset of values and is unchanged by row or column
permutation.

The same process applies to `C_geom`: read one FP32 world-coordinate snapshot,
represent coordinates exactly, compute squared differences exactly, and use
a common power-of-two denominator for the tertiary edge weights.

Under this contract, two primary totals are equal if and only if their exact
integer/rational sums are equal. No `1e-5`, ULP window, `isclose`, or solver
feasibility tolerance defines the optimum.

### 10.2 What this does and does not solve

This defines deterministic semantics for one score snapshot and makes
permutation tests valid by permuting that same snapshot. It does not make two
independently recomputed GPU score matrices identical. CUDA accumulation can
produce slightly different FP32 snapshots if operation ordering changes.

Therefore future tests must distinguish:

- **resolver equivariance:** permute one canonical snapshot;
- **upstream score reproducibility:** independently recompute the score under
  graph/order transformations.

The latter is an upstream numerical risk and must not be disguised as an
assignment tolerance. Canonical edge ordering/deterministic aggregation may
be considered only in a separate review if it is empirically required.

## 11. Exact implementation alternatives

### 11.1 A — multi-stage constrained optimization

The mathematical form is clean:

```text
Stage 1: find J*
Stage 2: maximize C_stay subject to J=J*
Stage 3: maximize C_geom subject to J=J*, C_stay=C_stay*
```

This is acceptable only with exact rational/integer equality constraints.
Repeated calls to SciPy Hungarian cannot express `J=J*`; an LP/MILP backend
with floating feasibility tolerances would reintroduce the ambiguity being
removed. A true exact integer optimizer could implement this form, but no
such existing dependency has been established in the repository.

Assessment: mathematically authoritative, but not the preferred first
engineering backend.

### 11.2 B — native exact lexicographic assignment

Associate each valid edge with the tuple:

```text
w[s,k] = (
    exact_primary_integer[s,k],
    1[k == c_old[s]],
    exact_negative_squared_distance_integer[s,k]
).
```

Tuple addition is componentwise and comparison is lexicographic. A
correctness-first rectangular assignment implementation can use an exact
min-cost-flow/augmenting-path algorithm over this totally ordered additive
group. Capacities remain integral and unchanged. At `P=20,K=21`, an
unoptimized reference implementation is sufficiently small for an
inference-only audit.

The solver must return:

- one assignment and the exact optimal tuple;
- a uniqueness certificate;
- `AMBIGUOUS_LEXICOGRAPHIC_OPTIMUM` when another assignment has the same
  complete tuple.

Uniqueness can be certified without enumerating all assignments: after
obtaining one optimum, forbid each of its selected edges in turn and resolve.
Any distinct assignment omits at least one selected edge. If any re-solve
attains the same exact tuple, the optimum is non-unique.

Assessment: **recommended reference implementation**. It implements the
formal lexicographic objective directly and needs no scalar epsilon.

### 11.3 C — composite scalarization

The expression

```text
J + epsilon_1 C_stay + epsilon_2 C_geom
```

is rejected as the canonical definition. A fixed floating epsilon can exceed
an unknown nonzero primary gap, disappear under rounding, or behave
differently after rescaling. “Small” is not a proof.

An exact arbitrary-precision integer radix encoding could theoretically be
derived after proving finite bounds for all lower-priority objective spans
and using an exact integer assignment solver. That construction would be a
representation of lexicographic tuples, not a heuristic epsilon. It offers no
initial advantage over native tuple comparison and is not recommended for the
first audit.

## 12. Canonical algorithm

### Algorithm: Lexicographic Continuity-Preserving Structured Refinement

```text
Inputs:
    Round-0 candidate IDs C^(0) [N,P]
    existing candidate bank g_world [N,K,2]
    existing unary, graph, relation, energy, and valid mask
    P=20, K=21, two refinement rounds

Require:
    previous IDs are valid and injective per agent
    valid_K(i) >= P
    mask is agent-level and slot-invariant

for round l in {1,2}:
    old_ids = C^(l-1)

    # Existing synchronous score construction; unchanged.
    compute relation and pair energy from old_ids
    accumulate incident energy
    S_fp32 = unary.float()[:,None,:] - accumulated.float()
    freeze one detached canonical S_fp32 snapshot

    for each agent i:
        if degree_i == 0:
            new_ids_i = old_ids_i              # structural identity
            continue

        convert valid S_i entries losslessly to exact primary integers
        compute exact stay indicators from old_ids_i
        compute exact world-metre squared-displacement integers

        solve over injective assignments, exactly and lexicographically:
            first maximize J
            among primary optima maximize C_stay
            among remaining optima maximize C_geom

        certify uniqueness of the complete lexicographic optimum

        if the optimum is not unique:
            return AMBIGUOUS_LEXICOGRAPHIC_OPTIMUM with certificate

        new_ids_i = the unique optimum assignment

    # Existing Jacobi contract.
    C^(l) = synchronously_commit(new_ids)

return C^(2)
```

“Approximately lexicographic”, epsilon perturbations, and solver-order
fallbacks are not part of this algorithm.

## 13. Why per-agent resolution remains the correct scope

The current score for each agent and slot already contains neighbor-conditioned
relation and pair-energy information from the old complete joint world. A
common slot is therefore the semantic continuity anchor. If every per-agent
resolver is unique and slot-equivariant, their synchronous product is also
slot-equivariant, and the complete scene world columns are preserved under a
common relabeling.

The accepted audit shows within-component per-agent assignment degeneracy; it
does not prove that a new component-level combinatorial solver is required.
No global or component assignment is approved. Component-local and scene-level
world-set checks remain diagnostics for whether the per-agent design actually
propagates as predicted.

Disconnected non-singleton components were absent from the ETH validation
stratum, so no claim about their relative semantics is made here.

## 14. Proposed implementation surface after approval

This section is a future proposal only.

Prefer an audit-only module, for example:

```text
tools/lexicographic_continuity_audit.py
tests/test_lexicographic_continuity_audit.py
```

The first intervention should monkey-patch or callback through the existing
diagnostic path. It should not add a production policy or modify `src/`.

Small helpers would cover:

- exact FP32/coordinate-to-integer conversion;
- tuple addition/comparison;
- exact rectangular lexicographic assignment;
- optimum uniqueness certification;
- the degree-zero structural base case;
- slot/candidate permutation audits;
- compact ambiguity and objective diagnostics.

The default categorical and CPSR V1 paths must remain tensor-exact. No
checkpoint or architecture metadata changes are needed for an audit-only
intervention.

## 15. Required tests for a future implementation

1. Tiny `P<=4,K<=5` cases match exhaustive enumeration of the full
   `(J,C_stay,C_geom)` tuple.
2. A better secondary/tertiary value can never override a strictly worse
   primary value, including a one-ULP FP32 primary gap.
3. A better geometry value can never override a worse stay count.
4. The result contains P unique valid candidates and respects masks.
5. `P>K`, `valid_K<P`, non-finite scores, invalid old IDs, and non-injective
   old IDs fail explicitly.
6. Degree-zero output equals previous IDs tensor-exact and invokes no solver.
7. A common slot permutation of score, mask, old IDs, and world columns gives
   the correspondingly permuted unique result.
8. A candidate permutation of score, mask, old IDs, coordinates, and output
   gives the correspondingly permuted unique result.
9. Combined slot/agent/candidate permutations preserve the exact optimum
   tuple and unique semantic assignment.
10. Identical/symmetric matrices with a residual full-tuple tie return
    `AMBIGUOUS_LEXICOGRAPHIC_OPTIMUM`, not an index-selected assignment.
11. If the previous assignment is primary-optimal, `C_stay=P` returns it
    uniquely.
12. Exact conversion is invariant to tensor enumeration and round-trips every
    finite FP32 bit pattern used by the solver.
13. The same canonical score snapshot gives identical results on repeated
    CPU runs.
14. Independently recomputed GPU scores are compared separately and any bit
    differences are reported rather than tolerance-collapsed.
15. Two rounds remain synchronous; no agent sees another agent's new IDs
    while scores are constructed.
16. Candidate geometry uses `goal_candidates_world`, not map coordinates.
17. Primary objective equals the exact brute-force optimum in tiny tests and
    is never worse than Policy D/D0 on the same canonical real matrix.
18. Uniqueness certification finds every brute-force alternative optimum in
    tiny tests.
19. Common slot and candidate permutations use the same frozen score/goal
    snapshot, not recomputed inputs.
20. No audit flag changes default production behavior, checkpoint loading,
    training, categorical sampling, or CPSR V1 reproduction.

## 16. Minimal future read-only validation

Use the same strict-no-z epoch-13 checkpoint, ETH validation 139 windows,
seeds 2035–2039, stochastic production Round 0, two rounds, conditional score,
candidate bank, relation, energy, temperature, diffusion, and paired RNG as
D0.

### Phase 0 — reproduction

Reproduce D0 exactly:

- prediction metrics;
- 20/20 coverage;
- degree-zero identity;
- slot-permutation statistics;
- all RNG boundaries.

Stop if reproduction fails.

### Phase 1 — semantic matrix audit

Run LCPSR on the frozen real score snapshots before trajectory evaluation.
For every interacting agent/round report:

- exact primary optimum and equality to the LCPSR output;
- Policy-D/D0 primary objective gap;
- `C_stay` before/after and gain;
- `C_geom` before/after and gain;
- number/rate of unresolved full-tuple ties;
- uniqueness certificate outcome;
- slot-permutation pathwise invariance;
- candidate-enumeration pathwise invariance;
- support invariance;
- component and whole-scene world-set invariance.

If any deployment-required real matrix remains
`AMBIGUOUS_LEXICOGRAPHIC_OPTIMUM`, stop and report. Do not obtain a metric table
by choosing an arbitrary representative.

### Phase 2 — paired inference validation

Only after Phase 1 passes, compare:

```text
D0
vs
LCPSR
```

with the same Round 0 and paired diffusion RNG. Report five-seed mean/std and
per-seed:

- minADE/minFDE;
- JADE/JFDE;
- Joint Goal Endpoint Error;
- Joint Goal Compatibility;
- Relative Motion Error;
- Round-0/1/2 unique count and geometric oracle;
- E=0/E>0, degree, component-size, and agent-count strata;
- churn and two-cycle rate;
- sampler/end-to-end runtime and memory.

The scientific goal is semantic invariance without losing D0's strong metric
profile, not metric improvement.

## 17. Future acceptance gate

The future audit passes only if all of the following hold.

1. Every interacting assignment is exactly primary-optimal on its canonical
   matrix; no lower objective is traded for continuity.
2. Every agent retains 20/20 valid unique candidates.
3. No deployment-required real matrix has an unresolved full-tuple optimum.
4. Under common slot permutations, whole-scene unordered world-set invariance
   is at least 95% in both rounds, with the residual cases explicitly
   attributed upstream rather than solver order.
5. Candidate-enumeration permutation invariance is 100% for every resolved
   assignment.
6. Objective, support, pathwise, component, and scene invariance are reported
   separately.
7. D0 minADE/minFDE, JADE/JFDE, endpoint, and relative-motion metrics remain
   within the existing 2% band. Compatibility is reported separately.
8. No new training, network, temperature, heuristic weight, or score change
   is introduced.
9. Default categorical and CPSR V1 reproduce exactly; all unit, CUDA/BF16,
   RNG, and all-off regressions pass.

Passing this gate would justify a separate production-adoption validation. It
would not itself productionize LCPSR, freeze Stage A, or authorize Stage B.

## 18. Risks and open questions

### 18.1 Unresolved risks

- **Residual exact symmetry.** The full tuple may not be unique. The proposed
  fail-closed contract is semantically correct but may expose real matrices
  that need another reviewed source of information.
- **Upstream GPU reproducibility.** Exact semantics cannot make independently
  recomputed FP32 score snapshots bit-identical.
- **Reference-solver cost.** Exact arbitrary-precision tuple assignment plus
  uniqueness certification can require up to `P+1` solves per agent/round.
  This is acceptable for an audit but must be benchmarked before adoption.
- **Geometry sensitivity.** Although weight-free and tertiary, world-metre
  geometry can reflect numerical coordinate symmetries or duplicates.
- **Observed-tie interpretation.** The prior `1e-5` audit grouped numerical
  near-equality; the exact audit may show that some pathology comes from tiny
  nonzero FP32 gaps plus solver numerical behavior rather than exact primary
  ties.

### 18.2 Explicit non-goals

No epsilon, candidate-ID ordering, slot-index ordering, Gumbel scale,
temperature search, Sinkhorn, component solver, learned resolver, loss,
retraining, or Stage-B behavior is proposed.

## 19. Answers to the required questions

1. **Why does multiple-optimum assignment cause slot pathology?** `argmax J`
   is set-valued. Solver-specific row traversal can choose a different
   candidate-to-slot bijection after a common row permutation. Per-agent
   support stays fixed while complete joint-world columns change.
2. **Why is deterministic Hungarian insufficient?** It optimizes `J` but does
   not define which primary optimum is semantically correct; its internal tie
   behavior is not model semantics.
3. **Why is `J + epsilon*C` not the strict definition?** No fixed floating
   epsilon is proven smaller than every nonzero primary gap, and rounding or
   score scaling can reverse priorities.
4. **Is `C_stay` mathematically reasonable?** Yes. It preserves old
   slot-candidate identity only within the primary-optimal set, has no
   hyperparameter, and is equivariant when old IDs are permuted consistently.
5. **Is `C_stay` sufficient?** No. Several assignments can share the same
   primary score and stay count.
6. **Is `C_geom` needed?** Yes as the recommended tertiary discriminator. It
   gives remaining moves a hypothesis-continuity meaning without changing
   higher-priority objectives.
7. **How is geometry defined?** Negative total squared Euclidean displacement
   between the assigned world-metre endpoint and that slot's previous
   endpoint, computed exactly from one canonical coordinate snapshot.
8. **Does the lexicographic semantics satisfy both permutation contracts?**
   The maximizer set always does. A single output does when the full optimum
   is unique; irreducible symmetric ties must be reported rather than resolved
   by enumeration.
9. **What is the most reliable implementation?** A native exact
   lexicographic rectangular-assignment reference solver over exact
   tuple-valued integer/rational weights, with explicit uniqueness
   certification.
10. **How is primary equality defined under FP32/FP64?** By exact equality of
    sums of the canonical FP32 bit values after lossless rational/integer
    lifting. FP64 is a transport/inspection format, not the equality
    tolerance.
11. **Is degree-zero identity structural or implicit?** Structural. A unified
    primary optimizer can change Round-0 support even with no social evidence;
    D0's identity meaning must be explicit.
12. **What is the contract under perfect symmetry?** Return an invariant
    optimal equivalence-class/ambiguity certificate and do not emit an
    arbitrary single assignment. Pathwise uniqueness is required only for a
    certified singleton optimum.
13. **What is the minimum next intervention?** An audit-only exact LCPSR
    resolver with degree-zero bypass, full-tuple uniqueness detection, and
    paired D0 slot/candidate-permutation validation; run prediction metrics
    only if the semantic matrix gate passes.
14. **Would passing permutation and metric gates justify production adoption
    validation?** Yes. It would justify entering a separately authorized
    production-adoption validation, but would not itself adopt the resolver,
    freeze Stage A, or start Stage B.

## 20. Final design status

`DESIGN_COMPLETE_AWAITING_REVIEW`

Canonical recommendation:

```text
degree == 0:
    identity refinement

degree > 0:
    exact lexicographic injective assignment
    (primary conditional score,
     previous-slot stay count,
     world-goal geometric continuity)

residual exact symmetry:
    explicit ambiguity/equivalence-class result; no index-based fallback
```

No implementation or experiment is authorized by this document.
