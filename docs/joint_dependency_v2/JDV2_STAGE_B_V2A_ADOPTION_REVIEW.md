# JDV2 Stage-B V2-A adoption and protocol review

## Final state

**`ADOPTION_BLOCKED_BY_SPLIT_PROTOCOL`**

Next state: **`eligible_for_clean_split_reproduction_design`**.

No model was trained, no checkpoint was modified, no scientific semantics
changed, and no other dataset or V2-B work was started. Because the canonical
ETH validation and test caches are fully identical, this review does not
create a Stage-B freeze document, freeze manifest, or freeze-contract test.

## 1. Candidate reviewed

| Item | Value |
|---|---|
| Audit base commit | `817cc7551bd7f3422422230821acf8b3dd5cdc6c` |
| Architecture | `jdv2-stage-b-v2a` |
| Residual semantics | `component_zero_mean` |
| Selected checkpoint | epoch 5 |
| V2-A SHA256 | `e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73` |
| Stage-A parent SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| V1 SHA256 | `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a` |
| DependencyCorrector parameters | 30,851 |
| DependencyCorrector source SHA256 | `0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522` |
| Projection source SHA256 | `355de2a9ace3265efe1d5fc1ed4b5a6c054a621fc452a7ad95d70b6a2c3b5630` |

All protected hashes were reproduced after the audit and engineering fix.

## 2. Split-integrity gate

The exact current loaders produced train/valid/test counts of
4,110/139/139. Content-level fingerprinting found:

| Pair | Exact ID overlap | Exact trajectory-content overlap |
|---|---:|---:|
| train / valid | 0 | 0 |
| train / test | 0 | 0 |
| valid / test | **139/139 (100%)** | **139/139 (100%)** |

The valid and test caches are also ordered-identical. Both cache sets point
to `data/eth5/eth/val/biwi_eth.txt`; the local `val` and `test` raw files have
the same SHA256. Full evidence and fingerprint definitions are in
`JDV2_EVALUATION_SPLIT_AUDIT.md` and the machine-readable split audit.

This triggers the mandatory blocking case. The labels
`model_selection_split=dataset_valid` and `final_test_split=heldout_test` do
not create separation in the actual local data. The current results must not
be called clean held-out-test results.

## 3. Scientific evidence that remains valid

The protocol defect changes the adoption claim, not the internal paired
method-development comparison. On the same predeclared ETH validation
windows/seeds, V2-A still provides the documented mechanism evidence:

- versus Stage A: minADE +0.636%, minFDE +0.259%, JADE +0.429%, JFDE
  -0.533%, Relative Motion -0.156%;
- versus V1: minADE -0.936%, minFDE -0.986%, JADE -0.267%, JFDE -0.783%,
  Relative Motion -0.146%;
- V1-gap recovery: 59.92% minADE and 79.38% minFDE;
- mean removed common-residual energy fraction: 33.33%;
- E=0 and mixed-scene degree-zero agents retain tensor-exact Stage-A
  execution;
- candidate coverage remains 20/20;
- the projection has zero parameters and 3.82% measured runtime overhead.

The supported interpretation remains narrow: the component-relative
zero-mean constraint is the principal beneficial mechanism, and V2-A is its
training/inference-consistent implementation. C versus D is practical parity,
not evidence of a large retraining-specific gain. These are ETH
method-development findings pending clean reproduction.

## 4. Historical Stage-A and Stage-B selection

Stage-A epoch 13 was selected by JFDE on `dataset_valid`, and its five-seed
reference metrics use the same 139 validation windows. Stage-B V1 and V2-A
also used that split for checkpoint selection; V2-A epoch 5 was selected by
JADE with JFDE tie-break. Since those windows equal the declared test set,
neither the current Stage-A parent nor the dependent Stage-B checkpoints can
serve as clean benchmark checkpoints.

Clean reproduction must therefore retrain/reselect Stage A under an internal
training-only validation split, then initialize and reselect Stage B from
that clean Stage-A parent. Reusing the current epoch-13/epoch-5 selection and
only rerunning evaluation would not repair leakage.

## 5. Resume-state reproducibility audit

Source inspection confirmed the historical defect:

- `validations_without_improvement` was initialized to zero inside every
  `_train_loop` call;
- checkpoints preserved model, optimizer, scheduler, scaler, best selection,
  best metric tables, epoch, stage progress, and optimizer-step count;
- checkpoints did not preserve validation patience;
- the Stage-A V2 early-collapse consecutive-epoch counter was also
  process-local and affected a stopping decision.

The V2-A interruption resumed from completed epoch 12 and left epoch 5 as the
global JADE/JFDE optimum. Resetting patience extended computation but did not
change the selected checkpoint. It is therefore an infrastructure
reproducibility defect, not a scientific invalidation of the existing paired
development result.

## 6. Minimal resume fix

`src/trainer.py` now:

1. owns both stopping counters as trainer state;
2. stores `validations_without_improvement` and
   `collapse_signature_epochs` in JDV2 checkpoints;
3. restores them only for exact same-stage resume;
4. resets them to zero for cross-stage weight initialization;
5. uses a documented zero fallback for legacy checkpoints without fields;
6. updates validation patience before saving a newly selected best
   checkpoint, so that saved state matches the completed validation event;
7. preserves all model, loss, optimizer, scheduler, sampler, projection,
   metric, and checkpoint-provenance semantics.

The legacy fallback is deliberately conservative: missing historical state
may extend training, but it cannot cause a premature stop based on unknown
history.

The deterministic lightweight regression interrupts a synthetic run after
epoch 3 and resumes it. Compared with uninterrupted execution, it verifies
identical next epoch (4), optimizer state, scheduler state, best selection,
best tables, patience count, model state, and early-stop epoch (5). It also
tests legacy fallback, cross-stage reset, and rejection of negative state.

## 7. Validation evidence

```text
python -m compileall -q .
# PASS

pytest -q
# 434 passed, 12 warnings

pytest -q \
  tests/test_jdv2_stage_b_v2a.py::test_cuda_bf16_projection_and_zero_init_identity \
  tests/test_jdv2_stage_b_mixed_precision_identity.py::test_cuda_bf16_e0_and_mixed_identity
# 2 passed on the real RTX 5070 Ti CUDA/BF16 path

git diff --check
# PASS
```

The warnings are existing dependency deprecations and the existing PyTorch
Transformer layout warning. No test failed.

## 8. Adoption decision

V2-A passed its original implementation and paired development gates, and no
V2-B evidence or authorization is created here. Nevertheless, the primary
adoption gate is data protocol integrity. Because valid and test are fully
identical, paper-level Stage-B freezing is blocked.

No file named `JDV2_STAGE_B_V2A_ADOPTION_FREEZE.md`, no
`stage_b_v2a_freeze/manifest.json`, and no freeze-contract test is created by
this review. The replacement deliverable is the clean split reproduction
plan in `JDV2_CLEAN_SPLIT_REPRODUCTION_PLAN.md`.

Primary final state: **`ADOPTION_BLOCKED_BY_SPLIT_PROTOCOL`**.

Next state: **`eligible_for_clean_split_reproduction_design`**.
