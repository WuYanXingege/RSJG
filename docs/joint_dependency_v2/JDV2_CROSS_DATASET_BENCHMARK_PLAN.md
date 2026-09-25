# JDV2 cross-dataset benchmark plan

## Scope

This is a plan only. It authorizes no training or evaluation. The target
datasets are HOTEL, UNIV, ZARA1, and ZARA2, and the primary rule is to follow
the original GDTS protocol for each target.

## Dataset-by-dataset preflight

Before training any target, create a separate preflight that records and
verifies:

1. the exact raw train/validation/test files and their hashes;
2. whether validation and test are byte/content-identical;
3. the original GDTS baseline protocol for that target;
4. source-batch and JDV2-cache completeness, schema, candidate order, units,
   source commit, and checkpoint hash;
5. availability and hash of the target's frozen GDTS source checkpoint;
6. the resolved configuration and absence of dataset-specific scientific
   hyperparameter changes;
7. GPU/BF16 and Stage-A/Stage-B identity smoke tests;
8. output-directory and checkpoint provenance isolation.

Any mirrored validation/test split must be reported as the original GDTS
protocol and never as an independent held-out test. An optional strict clean
protocol, if later requested, must be reported separately.

## Frozen across datasets

The following are invariant:

- strict-no-z Stage A architecture;
- K=21, P=20, M=4, energy rank 8;
- radius+TTC graph semantics (6 m, TTC 8 s, dt 0.4 s);
- frozen Goal U-Net candidate/world interface;
- weighted Gumbel-Top-P Round 0;
- two synchronous `exact_lexicographic_persistent_tie` refinement rounds;
- DependencyCorrector architecture (30,851 parameters);
- post-corrector/pre-epsilon-add `component_zero_mean` projection;
- active timesteps `[30,25,20,15,10,5]`;
- same-slot relation and one scene-level oracle training branch;
- `lambda_diff=1.0`, `lambda_relative=0.05`;
- Adam/1e-4/ExponentialLR semantics and BF16 FP32-island behavior;
- E=0 and mixed degree-zero Stage-A numerical identity;
- the metric set, coordinate conversion, P=20 deployed evaluation, and fixed
  predeclared inference seeds.

Dataset-specific frozen GDTS, Stage-A, and Stage-B checkpoints are permitted.
Output paths, manifests, and checkpoint hashes must be isolated per dataset.

## Execution order

For each target independently:

1. complete and review its protocol/cache/source-checkpoint preflight;
2. reproduce/freeze target-specific Stage A without altering the method;
3. initialize a fresh V2-A corrector from that target's frozen Stage A;
4. train/select using the target's original GDTS protocol;
5. run the predeclared same-protocol five-seed evaluation;
6. publish metrics, identity/coverage/runtime contracts, and hashes;
7. stop for review before advancing to another target.

No cross-dataset result may be used to return to ETH and tune architecture,
sampler, projection, loss weights, timesteps, optimizer, or precision. V2-B
remains unauthorized.

## Required comparison and reporting

Each dataset report must compare same-protocol GDTS, Stage A, and V2-A and
include minADE, minFDE, JADE, JFDE, endpoint, compatibility, and relative
motion; five-seed mean/std and per-seed values; E=0/E>0 identity strata;
20/20 coverage; runtime/memory; checkpoint/config/source hashes; and an
explicit fairness versus strict-heldout statement.

The only eligible next activity is
**`eligible_for_jdv2_cross_dataset_benchmark_preflight`**.
