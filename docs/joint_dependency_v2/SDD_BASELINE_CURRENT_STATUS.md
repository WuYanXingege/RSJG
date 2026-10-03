# SDD Baseline Current Status and Handoff

## 1. Snapshot

This document records the complete SDD baseline state as of
**2026-10-03 13:54:43 +08:00**. It is a handoff/status document, not a final
benchmark report.

Current primary state:

- branch: `research/joint-dependency-v2-clean`
- training source commit:
  `04fcaa4114a7fd695ede82fabd87adab3ff44781`
- protocol: paper-aligned legacy SDD 30/17 protocol
- active stage: Goal U-Net BCE pretraining
- active epoch: 1 / 150
- snapshot progress: 909 / 2440 train batches (37%)
- downstream joint stage: not started
- active detached process group: PID/PGID `962146`
- status at snapshot: running, finite, no post-fix traceback

The active log is:

```text
output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/
resume_after_loss_mask_fix.log
```

The IDE tabs under `gdts_baseline_sdd_seed2035` belong to the older
diagnostic experiment and do not reflect the current run.

## 2. Why the original SDD run was not paper-aligned

The first local SDD baseline run was:

```text
output/sdd/runs/gdts_baseline_sdd_seed2035/
```

It used the repository parser-style setup:

- joint training from scratch;
- learning rate `1e-4`;
- planned 300 epochs;
- no separately trained 150-epoch Goal U-Net initialization.

It was therefore a useful diagnostic but not a reproduction of the GDTS paper
training schedule. Recorded validation rows currently extend through epoch 80:

| Epoch | ADE | FDE |
| ---: | ---: | ---: |
| 20 | 8.409531 | 14.380629 |
| 40 | 8.760548 | 15.207258 |
| 60 | 8.805721 | 15.192688 |
| 80 | 9.209265 | 16.020972 |

Among these rows, epoch 20 is best. These values are in SDD pixels under the
repository's legacy 30/17 validation protocol. They must not be presented as
the final paper-aligned baseline.

The old process was deliberately stopped. Its run directory, logs, and
checkpoints were preserved.

## 3. Paper-aligned two-stage protocol

The current experiment implements:

1. Goal U-Net BCE pretraining for 150 epochs.
2. Selection of the best Goal U-Net checkpoint by validation BCE.
3. Fresh construction of the full GDTS model.
4. Strict loading of only `goal_module.*` weights.
5. Joint GDTS training for 250 fresh stage-local epochs.
6. Five stochastic evaluation runs from the selected joint checkpoint.

Frozen settings:

| Setting | Value |
| --- | --- |
| Dataset/target | SDD |
| Split | legacy 30 train scenes / 17 validation-test scenes |
| Observation/prediction | 8 / 12 frames |
| Goal epochs | 150 |
| Joint epochs | 250 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Goal scheduler | ExponentialLR, gamma 0.99 |
| Joint scheduler | ExponentialLR, gamma 0.995 |
| Batch size | 64 |
| Samples | 20 |
| Data augmentation | enabled |
| Precision | FP32 |
| Seed | 2025 |
| Validation seed | 2025 |
| Data workers | 1 |

This SDD split uses the same 17 scenes for model selection and final
evaluation. It reproduces the legacy published protocol boundary but is not a
clean, independently held-out test estimate. ADE/FDE are reported in pixels.

The authoritative launcher and protocol are:

```text
tools/run_sdd_paper_baseline.sh
docs/joint_dependency_v2/GDTS_SDD_PAPER_ALIGNED_PROTOCOL.md
```

## 4. Cache state

The complete source-batch cache has been built and is reusable:

```text
outputs/joint_dependency_v2/cache/baseline_source_batches/
sdd/sdd/data_batches_zstd_v1/
```

Counts:

| Split | Batches |
| --- | ---: |
| train | 2440 |
| valid | 814 |
| test | 814 |

Additional facts:

- all three `finished_*_batches.txt` markers exist;
- `cache_manifest.json` exists;
- cache size at snapshot: approximately 40 GB;
- manifest SHA256:
  `f7104967843b52e61543d6aaffb083cfd92d058d6d9b05913b0ab1cd54fc9c7d`.

The cache does not need to be rebuilt for a normal Goal or joint restart as
long as preprocessing settings and cache schema remain unchanged.

## 5. Engineering changes and incidents

### 5.1 Two-stage integration

Commit:

```text
9c7f98cdc2501225ae63489b66754f3bc07b18bb
Implement paper-aligned SDD two-stage GDTS training
```

This commit introduced:

- typed Goal-pretrain checkpoints;
- strict Goal-only initialization for a fresh GDTS model;
- checkpoint provenance and SHA256 recording;
- BCE-only deterministic Goal validation;
- raw-logit BCE checkpoint selection;
- exact `last` checkpoint support for both stages;
- the paper-aligned launcher and documentation;
- regression tests.

The joint stage never imports Goal-stage optimizer, scheduler, epoch, or
best-selection state. The history encoder and diffusion denoiser begin from
fresh initialization.

### 5.2 Cache rebuild incident

During an early fast-debug smoke check, the shared SDD cache path was
invalidated and removed by the preprocessor because the debug manifest did not
match the formal manifest. The interrupted partial cache was not reused.

The formal full cache was subsequently rebuilt from the raw source with
`fast_debug=False`. The counts and manifest hash in Section 4 describe the
current complete cache.

### 5.3 Goal loss-mask startup failure

The first paper-aligned launch completed cache construction and entered Goal
epoch 1, but failed before the first optimizer step:

```text
AttributeError:
'Goal_Pretrain' object has no attribute 'compute_loss_mask'
```

Failure log:

```text
output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/pipeline.log
```

No Goal checkpoint was produced and no joint training was started by that
failed attempt.

### 5.4 Loss-mask fix

Commit:

```text
04fcaa4114a7fd695ede82fabd87adab3ff44781
Fix Goal pretraining loss-mask contract
```

The fix added the same cumulative-presence loss-mask contract already used by
the full GDTS model and added a `get_loss -> backward` regression test.

Validation after the fix:

- `python -m compileall -q .`: PASS
- focused Goal protocol tests: 13 PASS
- full `pytest -q`: 492 PASS, 12 warnings
- `git diff --check`: PASS

The fix was pushed to GitHub before training was restarted.

## 6. Current live run

Run directory:

```text
output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/
```

Current process hierarchy at the snapshot:

- pipeline shell PID/PGID: `962146`;
- Goal parent process: `962176`;
- data-loader worker observed: `962194`.

The process was started with:

```bash
nohup setsid nice -n 10 ionice -c2 -n7 taskset -c 0-3 \
  env PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 \
      OPENBLAS_NUM_THREADS=4 \
      NUMEXPR_NUM_THREADS=4 \
      PYTHON_BIN=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python \
      CACHE_ROOT=outputs/joint_dependency_v2/cache/baseline_source_batches \
  bash tools/run_sdd_paper_baseline.sh all \
  > output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/resume_after_loss_mask_fix.log \
  2>&1 < /dev/null &
```

Runtime snapshot:

- Goal epoch 1: 909 / 2440 batches;
- GPU: NVIDIA GeForce RTX 5070 Ti;
- model process GPU memory: approximately 8.2 GB;
- total GPU memory observed: approximately 8.9 / 16.3 GB;
- host memory available: approximately 24 GB;
- no NaN, Inf, OOM, or traceback after the fix.

No Goal checkpoint exists yet because the first epoch has not completed.
Consequently, the current partial epoch cannot be resumed after interruption;
once epoch 1 completes, `goal_pretrain_last_model.pt` will provide a
post-scheduler exact resume point after every epoch.

The first-epoch throughput suggests roughly tens of minutes per Goal epoch.
Any full-stage ETA inferred now is provisional because validation and later
I/O behavior are not yet represented.

## 7. Expected automatic handoff

If the current process remains healthy, the launcher will:

1. finish Goal epochs 1-150;
2. retain `goal_pretrain_best_model.pt` and
   `goal_pretrain_last_model.pt`;
3. run Goal-stage evaluation;
4. verify the typed best Goal checkpoint;
5. create `gdts_paper_sdd_joint_seed2025`;
6. initialize only its Goal U-Net from the selected Goal checkpoint;
7. start 250 fresh joint epochs;
8. select the joint best checkpoint and run the configured evaluation.

The joint output directory is:

```text
output/sdd/runs/gdts_paper_sdd_joint_seed2025/
```

At this snapshot it has not been created/populated because Goal pretraining is
still in epoch 1.

## 8. Monitoring and recovery commands

Live status:

```bash
tail -f output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/\
resume_after_loss_mask_fix.log
```

Process group:

```bash
PID=$(cat output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/pipeline.pid)
ps -o pid,ppid,pgid,sid,ni,%cpu,%mem,rss,stat,etime,cmd --forest -g "$PID"
```

Goal recovery after at least one completed epoch:

```bash
bash tools/run_sdd_paper_baseline.sh goal-resume
```

Joint recovery after at least one completed joint epoch:

```bash
bash tools/run_sdd_paper_baseline.sh joint-resume
```

These recovery commands default to the post-scheduler `last` checkpoint.
They should not be run while the active pipeline is alive.

## 9. Artifact integrity

Relevant hashes at the snapshot:

| Artifact | SHA256 |
| --- | --- |
| launcher | `0fe603fe3b290bc5c520f5650e6fe486142ccba36ab643f5ea818980e946f93d` |
| Goal run config | `07841039a5dfb945ecd2ae8741902e39aa09254b7daf6762756b9e06b715dd3b` |
| cache manifest | `f7104967843b52e61543d6aaffb083cfd92d058d6d9b05913b0ab1cd54fc9c7d` |

Runtime outputs and caches are intentionally not committed to Git. The
protocol, implementation, tests, and this status handoff are committed.

## 10. Next analysis checkpoints

Do not interpret method quality before the following milestones:

1. first Goal epoch completes and `last` checkpoint reload is verified;
2. first validation BCE is recorded;
3. Goal best epoch and learning curve are available;
4. strict Goal-only initialization metadata appears in the joint protocol;
5. joint training produces stable finite ADE/FDE;
6. final paper-aligned five-run SDD metrics are available.

The final comparison should keep the following separate:

- old `1e-4`, 300-epoch joint-only diagnostic;
- current paper-aligned 150+250 two-stage GDTS run;
- any later JDV2 SDD experiment.
