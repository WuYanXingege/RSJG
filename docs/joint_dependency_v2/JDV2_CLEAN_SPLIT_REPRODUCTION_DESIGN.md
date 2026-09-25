# JDV2 Clean-Split Reproduction Design

## 1. Scope and frozen scientific method

This document locks the execution protocol for a fresh ETH reproduction. It
does not change strict-no-z Stage A, exact persistent refinement, the frozen
GDTS generator, DependencyCorrector, component-zero-mean V2-A, any loss, or
any optimizer/scheduler setting. Historical results retain the label **ETH
method-development / mirrored-validation evidence**.

The clean protocol has exactly three logical roles:

| Logical role | Physical cache | Physical indices | Windows | Raw source |
|---|---:|---:|---:|---|
| train | train | 0–3789 | 3790 | six complete source blocks |
| internal_valid | train | 3790–4109 | 320 | `train/uni_examples.txt` |
| final_test | test | 0–138 | 139 | `val/biwi_eth.txt` |

Checkpoint selection is `internal_train` with `source_block`; the held-out
test is never a model-selection split.

## 2. Immutable split manifest

The canonical manifest is
`outputs/joint_dependency_v2/eth/joint_dependency_v2/clean_split_protocol/manifest.json`.
Its embedded stable JSON hash is
`0a6deb6ff9026b1784d728ef70f6e7dd1020c73819d4f71557dd7578772cc2b4`.

Every member records physical split/index, cache filename and SHA256, JDV2
cache filename, canonical raw source path and hash, scene, frame IDs, agent
IDs, identity fingerprint, and trajectory-content fingerprint. Runtime loading
re-authenticates the embedded/configured hash, exact counts, exact index
ranges, physical roles, internal duplicates, and all cross-role fingerprint
intersections. It never silently regenerates membership.

## 3. Physical address versus logical permission

`dataset_set_name` now exposes two independent properties:

- `physical_set_name` selects the batch/JDV2 cache directory and the original
  underlying index;
- `logical_set_name` controls augmentation and future-teacher permission.

Thus internal-valid member `k` is a `Subset` view of original physical train
index `i=3790+k` and reads `train/<i>.pt`. It is logically `valid`, so
augmentation is disabled and `jdv2_teacher_cache` is never attached. Logical
train retains its teacher sidecar in the training phase. No cache is copied,
renumbered, or modified.

Ground-truth future remains available to the metric implementation. It does
not enter deployed prediction. Optional validation loss/diagnostic computation
occurs only after prediction and has no optimizer, scheduler, checkpoint
selection, early-stopping, or sampler-state side effect; the train-only
teacher sidecar is structurally absent.

## 4. Final-test access state machine

Clean training uses `final_test_access=blocked`. The trainer constructs only
`train` and `valid`; a test loader does not exist. Direct test-loader access
authenticates the final lock before constructing the physical test dataset.
Clean `train_test` is rejected.

Final evaluation requires all of the following:

1. explicit `phase=test`, `load_checkpoint=best`, and
   `final_test_access=authorized`;
2. target `stage_a` or `stage_b`;
3. a `jdv2-clean-final-test-lock-v1` artifact;
4. exact split-manifest hash and current source commit;
5. exact hashes and paths for both canonical configs and both selected
   checkpoints, plus selection epoch/metric;
6. the fixed seed list `[2035,2036,2037,2038,2039]`;
7. target checkpoint path equal to the selected run's `best_model.pt`.

The lock is checked before test-loader construction and again immediately
before evaluation. Artifact drift fails closed. Clean checkpoint payloads also
persist the split-manifest hash; resume and cross-stage loads reject a mismatch.

## 5. Sequential reproduction lock

### Phase A — clean Stage A

Use `configs/joint_dependency_v2/jdv2_stage_a_clean_eth.yaml`. Train on the
3790 logical-train windows, select only on the 320 internal-valid windows,
then freeze/hash the selected checkpoint and write its selection manifest.
Do not access final test.

### Phase B — clean Stage-B V2-A

Populate the predeclared Stage-A parent SHA256 in
`configs/joint_dependency_v2/jdv2_stage_b_v2a_clean_eth.yaml` only after Phase
A is frozen. Initialize a fresh V2-A from that exact clean Stage-A checkpoint,
train on 3790, select on 320, and freeze/hash its checkpoint. Historical V1 or
V2-A development checkpoints are not training parents. Do not access final
test.

### Phase C — one final held-out event

After both selections and config hashes are locked, create the final-test lock
and evaluate clean Stage A and clean Stage-B V2-A once on the 139 held-out ETH
windows with the five predeclared seeds. No training or tuning follows those
metrics.

## 6. Configuration invariants

Stage A preserves strict-no-z, K=21, P=20, M=4, rank=8, radius/TTC graph,
two exact-lexicographic-persistent-tie refinement rounds, frozen GDTS,
optimizer Adam, LR 1e-4, ExponentialLR, BF16, seed 2035, validation seed 2035,
JFDE-resolving `best_metric=auto`, and patience 12.

Stage-B V2-A preserves the 30,851-parameter DependencyCorrector,
`component_zero_mean`, `lambda_diff=1.0`, `lambda_relative=0.05`, Adam 1e-4,
BF16, active diffusion timesteps, scene-level oracle training branch,
same-slot relation, JADE primary/JFDE tie-break, and all frozen Stage-A
semantics. Its clean parent hash is deliberately unset during preflight, so
premature Stage-B initialization fails closed.

## 7. Interpretation boundary

Protocol success means split integrity, method invariance, provenance,
checkpoint-selection validity, and final-test isolation. It does not require
clean checkpoints to reproduce historical mirrored-validation numbers. The
future clean held-out comparison is a new protocol and must be reported as
such.
