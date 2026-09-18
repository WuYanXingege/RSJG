# Stage-A Audit and Engineering Fix Report

## 1. Scope and decision

This round changed engineering and measurement paths only. It did not change
the scene-latent, relation, energy, or dependency-corrector architecture; it
did not change any loss or coefficient; and it did not start Stage-A training
or Stage B.

The engineering fixes pass the repository test gate, and repeated validation
now gives a trustworthy comparison. The scientific gate remains closed:

- epoch 11 remains the preferred Stage-A reference over epoch 20, but its
  original single-pass JFDE of `0.70912` was an unusually favorable stochastic
  draw; its five-seed mean is `0.75099 +/- 0.01600`;
- relative to an exact same-protocol GDTS evaluation, epoch 11 improves joint
  metrics but degrades minFDE by `13.09%`;
- the scene posterior/prior is conclusively collapsed to scene mode 1 and is
  insensitive to future shuffling;
- the relation teacher and dynamic relation retain multi-mode, future, goal
  pair, and z sensitivity, but a one-seed neutralization intervention does not
  show a positive deployed-metric contribution.

Therefore, do not enter Stage B and do not resume the current Stage-A run.
The evidence still supports a minimal, separately reviewed change to the
Stage-A latent objective before retraining. This report does not implement or
select that theoretical change.

The checkpoint's training source commit is
`4b75110a571e6fd200938ff951a3a218ca4eb1e8`. The fixes in this report are
uncommitted working-tree changes at the time of writing.

## 2. Modified files and artifacts

Source and tests:

- `src/utils.py`: full RNG snapshot/seed/restore context;
- `src/parser.py`: explicit `--validation_seed`, defaulting to the training
  seed;
- `src/trainer.py`: deterministic validation wrapper, custom audit metrics,
  same-split external graph injection for the baseline, historical best-state
  restoration, `last_model.pt`, cross-stage reset, and stage-local completed
  optimizer-step state;
- `src/jdv2_audit.py`: repeated-validation aggregation and scene/relation
  read-only audits;
- `tools/repeated_jdv2_validation.py`: generic one-checkpoint repeated
  validation CLI, with five default seeds;
- `tools/audit_jdv2_stage_a.py`: epoch 11/20, exact GDTS baseline, latent, and
  relation audit driver;
- `tests/test_joint_dependency_v2_integration.py`: RNG, aggregation,
  checkpoint, old-checkpoint compatibility, explicit last-checkpoint, and
  cross-stage reset tests.

Protected/audit artifacts:

- `outputs/joint_dependency_v2/eth/joint_dependency_v2/reference_checkpoints/jdv2_stage_a_full_seed2035/epoch_011_best_model.pt`;
- the adjacent `epoch_011_best_model.metadata.json` and `SHA256SUMS`;
- `outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_engineering_fix/stage_a_audit_results.json`;
- `audit_rerun.log` in the same audit directory.

The existing run directory was not deleted. Its log contains a complete epoch
31 and an interrupted epoch 32 from an external continuation before this audit
finished. No Stage-A or audit process remained active at report time. These
later records do not alter the protected epoch-11 reference.

## 3. Protected epoch-11 reference

| Field | Value |
|---|---|
| Epoch/stage | 11 / `joint_goal` |
| Selection stored by the original run | JFDE `0.7091219624` |
| Checkpoint bytes | 37,538,552 |
| Checkpoint SHA256 | `be8ae2207308a674d809e570fdeafe07a2e276f088a8d5ce30bae8d9b6252db3` |
| Metadata SHA256 | `f4a3934be7fa9d6d851a2475a4b5b8b9e268ee9e1f69f8e01923573e38038180` |
| Source GDTS SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Cache manifest hash | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| Protection | checkpoint, metadata, and checksum file mode `0444`; outside the run's `saved_models` directory |

`sha256sum -c SHA256SUMS` passes. Resume/checkpoint rotation cannot target the
reference path.

## 4. Deterministic and repeated validation protocol

Primary training validation now defaults to `validation_seed = seed`. Before
validation, Python, NumPy, CPU Torch, every CUDA-device RNG, and cuDNN
deterministic/benchmark flags are snapshotted. A fixed seed initializes the
isolated stream. The original stochastic deployed policy still runs:

- joint-goal initialization and both refinement rounds still call categorical
  multinomial sampling;
- diffusion still draws its trunk and ETH branch noise;
- validation is not replaced by MAP inference.

The complete RNG and cuDNN state is restored in a `finally` block. Therefore,
validation no longer consumes or perturbs the training RNG stream. Test-mode
behavior remains backward-compatible unless an explicit evaluation seed is
passed.

The generic repeated command is:

```bash
python tools/repeated_jdv2_validation.py \
  --config <run-config.yaml> \
  --checkpoint <checkpoint.pt> \
  --output <results.json> \
  --device cuda:0
```

It defaults to seeds `2035 2036 2037 2038 2039` and reports every seed plus
population mean/std. The formal audit used the same seeds, the 139-window ETH
validation split, 20 samples, `GDTS.compute_model_metrics`, and
`scene.make_world_coord_torch` for every compared model.

## 5. Epoch 11 versus epoch 20: five-seed results

All metrics are errors; lower is better.

| Checkpoint | Metric | Mean | Std |
|---|---|---:|---:|
| epoch 11 | minADE@K | 0.288946 | 0.004816 |
| epoch 11 | minFDE@K | 0.446046 | 0.009029 |
| epoch 11 | JADE | 0.424094 | 0.010590 |
| epoch 11 | JFDE | 0.750989 | 0.015997 |
| epoch 11 | Joint Goal Endpoint Error | 0.759258 | 0.020128 |
| epoch 11 | Joint Goal Compatibility | 0.492833 | 0.027062 |
| epoch 11 | Relative Motion Error | 0.300921 | 0.007298 |
| epoch 20 | minADE@K | 0.292240 | 0.002975 |
| epoch 20 | minFDE@K | 0.455062 | 0.011543 |
| epoch 20 | JADE | 0.428986 | 0.005756 |
| epoch 20 | JFDE | 0.762553 | 0.013708 |
| epoch 20 | Joint Goal Endpoint Error | 0.768707 | 0.007448 |
| epoch 20 | Joint Goal Compatibility | 0.460879 | 0.014387 |
| epoch 20 | Relative Motion Error | 0.283770 | 0.002044 |

Per-seed primary/joint values:

| Checkpoint | Seed | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|
| epoch 11 | 2035 | 0.441342 | 0.768557 | 0.783938 | 0.516638 | 0.310497 |
| epoch 11 | 2036 | 0.417862 | 0.747359 | 0.753778 | 0.486331 | 0.296145 |
| epoch 11 | 2037 | 0.415976 | 0.732506 | 0.754765 | 0.465329 | 0.301404 |
| epoch 11 | 2038 | 0.413860 | 0.735859 | 0.726823 | 0.464434 | 0.289985 |
| epoch 11 | 2039 | 0.431432 | 0.770666 | 0.776988 | 0.531433 | 0.306573 |
| epoch 20 | 2035 | 0.431954 | 0.766368 | 0.778576 | 0.464766 | 0.283351 |
| epoch 20 | 2036 | 0.418783 | 0.753926 | 0.759254 | 0.461340 | 0.281523 |
| epoch 20 | 2037 | 0.426617 | 0.744833 | 0.761153 | 0.442271 | 0.286861 |
| epoch 20 | 2038 | 0.433074 | 0.785708 | 0.770266 | 0.451103 | 0.285295 |
| epoch 20 | 2039 | 0.434501 | 0.761931 | 0.774286 | 0.484915 | 0.281822 |

Epoch 11 is better in mean JFDE, JADE, both marginal metrics, and joint goal
endpoint error. Epoch 20 is better in compatibility and relative motion. The
JFDE mean difference is `0.01156`, smaller than the per-checkpoint stochastic
standard deviation. Epoch 11 is retained as the reference because it remains
best under the frozen primary metric and has the broader metric compromise,
not because the original single draw was definitive.

## 6. Exact same-protocol GDTS baseline

The baseline did not reuse a legacy test artifact. It strict-loaded the legacy
GDTS checkpoint into the already tested JDV2 all-off passthrough and evaluated
the same synchronized validation records. JDV2 graph records were injected
only to compute Relative Motion Error; prediction generation remained exact
GDTS. Sample count, seeds, metric functions, and world-coordinate conversion
were identical.

| Metric | GDTS mean | GDTS std | Epoch-11 relative change |
|---|---:|---:|---:|
| minADE@K | 0.286332 | 0.005157 | +0.91% |
| minFDE@K | 0.394404 | 0.008789 | +13.09% |
| JADE | 0.467982 | 0.002848 | -9.38% |
| JFDE | 0.815469 | 0.015009 | -7.91% |
| Joint Goal Endpoint Error | 0.799670 | 0.010164 | -5.05% |
| Joint Goal Compatibility | 0.674531 | 0.005805 | -26.94% |
| Relative Motion Error | 0.380473 | 0.005892 | -20.91% |

The exact GDTS per-seed JFDE values are `0.809972`, `0.790851`, `0.814494`,
`0.829298`, and `0.832731`. The complete per-seed values for every metric are
in `stage_a_audit_results.json`.

Stage A therefore answers the first scientific question asymmetrically: it is
better on the requested joint-consistency metrics, approximately neutral on
minADE, and materially worse on marginal endpoint minFDE. The minFDE change is
well outside the 1--2% diagnostic line.

## 7. Scene-latent audit

The audit covers all 139 validation windows.

| Quantity | Result |
|---|---|
| Prior entropy mean / max | `1.635e-7` / `1.362e-6` (maximum possible is `log(4)=1.386`) |
| Posterior entropy mean / max | `4.689e-8` / `2.780e-7` |
| Prior soft usage | `[3.14e-9, 0.9999999911, 2.54e-9, 3.23e-9]` |
| Posterior soft usage | `[7.63e-10, 0.9999999977, 7.26e-10, 7.99e-10]` |
| Prior hard usage | `[0, 1, 0, 0]` |
| Posterior hard usage | `[0, 1, 0, 0]` |
| Mean `abs(q-p)` | `1.856e-9` |
| KL(q||p) | numerical zero; raw FP32 residual is on the order of `1e-9` |
| Future-shuffle posterior L1 mean / max | `7.695e-11` / `4.477e-10` |

Fixing X and cyclically shuffling compatible agent futures within each scene
does not measurably change q. This establishes posterior collapse and loss of
future sensitivity, rather than merely a well-matched rich prior/posterior.

The fixed-z intervention adds an important qualification. Against the normal
mode-1 allocation, forcing modes 0/2/3 changes mean candidate assignments by
`15.39%/14.66%/14.40%` and mean selected goals by
`0.404/0.387/0.374 m`. Thus the downstream z-conditioned parameters are not
identical; the failure is that learned inference allocates zero practical mass
to those paths. Fixing z=1 is exactly identical to deployed inference.

## 8. Dynamic-relation audit

There are 92 E>0 windows, 47 E=0 windows, and 431 canonical validation edges.
Statistics exclude E=0 windows where appropriate.

| Quantity | Result |
|---|---|
| Dynamic soft usage | `[0.1771, 0.0811, 0.3520, 0.3899]` |
| Teacher soft usage | `[0.2162, 0.0704, 0.3326, 0.3808]` |
| Dynamic entropy mean | `1.0794` |
| Teacher entropy mean | `1.0872` |
| Variance across candidate pairs, mean | `3.027e-4` |
| Variance across z, mean | `2.035e-3` |
| Dynamic versus history-only KL, mean | `0.01636` |
| Dynamic versus history-only TV, mean | `0.06182` |
| Future-shuffle teacher TV, mean | `0.28911` |

The relation teacher is future-sensitive and not mode-collapsed. The dynamic
relation is not history-only: it varies across candidate pairs and z and
differs measurably from the base relation. However, actual deployed z is
collapsed, so the observed across-z capacity is not used by the current scene
inference.

At fixed seed 2035, neutralizing only the dynamic residual gave:

| Metric | Normal | Dynamic residual neutralized |
|---|---:|---:|
| minADE@K | 0.294112 | 0.292993 |
| minFDE@K | 0.451103 | 0.443408 |
| JADE | 0.441342 | 0.435473 |
| JFDE | 0.768557 | 0.763611 |
| Goal endpoint | 0.783938 | 0.779741 |
| Compatibility | 0.516638 | 0.506782 |
| Relative motion | 0.310497 | 0.310010 |

This single controlled intervention does not establish a benefit from the
trained dynamic residual; every reported error is slightly lower when it is
neutralized. It is an inference intervention on a jointly trained checkpoint,
not a replacement for a retrained relation ablation, so the correct conclusion
is “contribution unconfirmed,” not “relation architecture failed.”

## 9. Checkpoint/resume repair

New JDV2 checkpoints persist:

- primary/tie-break `_best_selection`;
- the complete best-metric table and best epochs;
- stage progress and explicit completed stage optimizer steps;
- optimizer, scheduler, scaler, architecture, ablations, and cache/source
  hashes as before.

Same-stage resume restores all historical selection state before the next
validation. For old checkpoints such as epoch 11/20, the comparator is restored
from `best_metric`, and per-metric minima/epochs are reconstructed from
`log_curve.txt` only through the loaded epoch. A worse first validation after
resume therefore cannot overwrite the historical best.

Every newly completed epoch writes `saved_models/last_model.pt` after its
optimizer, validation, scheduler, and diagnostics work. No `last_model.pt` was
fabricated for the frozen historical run because this round did not train.

## 10. Stage-A to Stage-B reset repair

Cross-stage loading is now explicitly weight initialization:

- target-stage start epoch is 1;
- `stage_progress` is 0;
- completed target-stage optimizer steps are 0;
- optimizer, scheduler, and scaler state load only for same-stage resume;
- Stage-B teacher curriculum reads a stage-local counter incremented only
  after successful optimizer steps.

Same-stage resume keeps epoch, progress, optimizer/scheduler/scaler, and the
completed-step counter. Unit tests cover the cross-stage epoch/progress reset,
same-stage continuation, persisted step state, and old-checkpoint best-state
compatibility.

## 11. Verification

Using `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg`:

- `python -m compileall .`: passed;
- `pytest`: `231 passed, 4 skipped`;
- `git diff --check`: passed.

The skips are existing CUDA-conditional tests in the sandboxed test process;
the formal audit itself ran successfully on the host RTX 5070 Ti. No loss,
network structure, model architecture, or training objective was modified.

## 12. Recommendation on the Stage-A latent objective

Yes: after making the measurement reliable, a Stage-A latent-objective change
is still recommended before retraining. The reason is now direct rather than
inferred from epoch logs: both p and q use one mode, q is effectively invariant
to compatible future shuffling, and forced unused z modes demonstrably lead to
different goal allocations that inference never uses.

The next change should be the smallest theory-approved objective correction
that makes q future-responsive and prevents the dual PL expectation from
immediately selecting one mode. It must be reviewed separately against the
frozen theory. This report does not recommend adding an ad hoc entropy/balance
regularizer, changing the scene network, or changing the relation/energy
architecture. Stage B remains blocked until that decision and a new Stage-A
run are approved.
