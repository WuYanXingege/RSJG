# JDV2 clean split reproduction plan

## Status

**Design-only remediation plan. No training is authorized by this document.**

This plan replaces the blocked Stage-B adoption freeze. Its sole purpose is
to establish a leakage-free ETH checkpoint-selection and final-evaluation
protocol while preserving the frozen scientific method.

## 1. Required protocol

Use the existing deterministic training-source partition:

```yaml
model_selection_split: internal_train
internal_validation_strategy: source_block
final_test_split: heldout_test
```

A read-only dry run of the current loader yields:

| Logical split | Physical source cache | Windows | Raw-source role |
|---|---|---:|---|
| train | train | 3,790 | all training sources except final block |
| internal valid | train | 320 | `train/uni_examples.txt` |
| final test | test | 139 | untouched ETH `biwi_eth.txt` |

The source-block boundary is deterministic and holds out the complete final
raw source rather than randomly interleaving overlapping windows. Before any
training, a new machine audit must prove zero identity/content overlap among
the 3,790 training windows, 320 internal-validation windows, and 139 final
test windows.

## 2. Preflight requirements

Before launching reproduction:

1. materialize and hash the exact member IDs for all three logical splits;
2. verify raw-source and window-content disjointness;
3. persist the member lists in the run protocol artifact;
4. ensure logical internal validation disables augmentation and does not load
   or expose train-only future-teacher sidecars to the deployed prediction
   path;
5. verify candidate-cache indices remain aligned when a `Subset` of the train
   cache is used for internal validation;
6. run all Stage-A freeze, Stage-B identity, CUDA/BF16, sampler, and resume
   regressions;
7. predeclare seeds, checkpoint criteria, stopping rules, and the single final
   test evaluation before execution.

The current `Subset` implementation uses a physical train dataset for both
logical partitions. A future preflight must explicitly audit the distinction
between physical cache split and logical train/validation role, especially
the train-only teacher-sidecar condition. This is a protocol/infrastructure
check, not permission to change a model or loss.

## 3. Reproduction sequence

### Stage A

Train strict-no-z Stage A from the same frozen legacy GDTS checkpoint using
only the 3,790 logical training windows. Select the checkpoint only on the 320
internal source-block validation windows. Preserve the frozen architecture,
K/P/M/rank, exact persistent-tie sampler, relation, energy, loss, optimizer,
scheduler, precision, and predeclared inference seeds.

After selection, freeze and hash the new clean-protocol Stage-A checkpoint.
Do not inspect final-test metrics while choosing the epoch or changing the
method.

### Stage B V2-A

Initialize a fresh V2-A corrector from the new clean Stage-A parent. Train on
the same logical training partition and select only on the same internal
source-block validation partition. Preserve component-zero-mean projection,
30,851-parameter corrector, loss coefficients, active timesteps, optimizer,
scheduler, precision, sampler, and all identity contracts.

Do not initialize from the current contaminated-selection epoch-5 checkpoint.
The present V2-A checkpoint remains an immutable development artifact.

### Final test

Only after Stage A and Stage B are frozen without test access, evaluate the
untouched 139-window ETH test split once under the predeclared five inference
seeds. Report Stage A and V2-A with paired worlds/noise and the existing metric
implementation. Do not return to ETH for tuning after observing final-test
results.

## 4. Required comparisons and labels

The clean report must separate:

- internal-validation checkpoint-selection metrics;
- final held-out-test metrics;
- historical mirrored-split development evidence.

It must never relabel the current 139-window validation results as held-out
test results. Historical V1 and post-hoc centering may remain explanatory
development controls, but any paper-level claim should be based on clean
Stage-A/V2-A reproduction.

## 5. What remains frozen

This remediation does not authorize:

- Stage-A or Stage-B architecture changes;
- loss, lambda, temperature, sampler, graph, relation, energy, or diffusion
  changes;
- V2-B;
- hyperparameter search;
- other datasets;
- reuse of test metrics for model selection.

The resume-state infrastructure fix should be present before reproduction so
an interruption is semantically equivalent to uninterrupted execution.

## 6. Exit gate

Clean adoption becomes reviewable only if:

1. the persisted logical split fingerprints are disjoint;
2. both stages were selected without test access;
3. protected scientific contracts and checkpoint provenance pass;
4. same-stage resume equivalence passes;
5. the final test was evaluated only after freeze;
6. all results are labeled by split and selection role.

Eligible next activity:
**`eligible_for_clean_split_reproduction_design`**.
