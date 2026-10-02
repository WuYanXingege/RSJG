# GDTS SDD Baseline Protocol

## Scope

This launcher preserves the repository parser-default GDTS baseline on
Stanford Drone Dataset (SDD). It is a `1e-4`, 300-epoch diagnostic and is not
the paper Table-II training protocol. The paper-aligned two-stage launcher is
documented in `GDTS_SDD_PAPER_ALIGNED_PROTOCOL.md`. Neither path enables Joint
Dependency V2 or changes the GDTS model, loss, sampler, or evaluator.

## Frozen configuration

- dataset / test set: `sdd`
- split protocol: 30 training scenes / 17 test scenes
- validation strategy: `validate_on_test`
- model: `goal_model_type=independent`, `training_stage=baseline`
- coordinate system: original SDD pixels
- epochs: 300
- optimizer: Adam, learning rate `1e-4`
- scheduler: ExponentialLR, gamma `0.995`
- batch size: 64
- trajectory-window stride: 1
- map downsample factor: 8
- data augmentation: enabled
- trajectory samples: 20
- numerical precision: FP32
- seed / validation seed: 2035
- validation: epoch 5 onward, every 20 epochs

The SDD protocol uses the same 17 scenes for validation and test. Consequently,
the best-checkpoint result is not an independently selected held-out test
estimate. Baseline ADE and FDE are reported in pixels.

## Commands

Build the losslessly compressed source-batch cache and then train/test:

```bash
PYTHON_BIN=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python \
  bash tools/run_sdd_baseline.sh all
```

The phases can also be run separately:

```bash
bash tools/run_sdd_baseline.sh pre-process
bash tools/run_sdd_baseline.sh train_test
```

Formal outputs are written beneath:

```text
output/sdd/runs/gdts_baseline_sdd_seed2035/
```

The shared compressed cache is written beneath:

```text
outputs/joint_dependency_v2/cache/baseline_source_batches/
  sdd/sdd/data_batches_zstd_v1/
```

The legacy GDTS baseline cache mixes independent trajectory windows; it is
suitable for the original marginal ADE/FDE protocol but not for scene-grouped
JADE/JFDE or collision metrics. Those require a separate synchronized
evaluation protocol and must not be inferred from this training run.
