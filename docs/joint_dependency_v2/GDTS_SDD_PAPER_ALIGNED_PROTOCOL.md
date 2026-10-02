# GDTS SDD Paper-Aligned Two-Stage Protocol

## Purpose

This protocol targets the training schedule stated in the GDTS paper for SDD:

1. train the Goal U-Net with BCE for 150 epochs;
2. initialize only the Goal U-Net of a fresh GDTS model from the selected
   typed checkpoint;
3. jointly train Goal U-Net, history encoder, and diffusion denoiser for
   250 stage-local epochs.

It does not resume the earlier `1e-4`, 300-epoch diagnostic run.

## Frozen settings

- SDD TrajNet split: 30 train scenes / 17 validation-test scenes
- observation / prediction: 8 / 12 frames
- coordinate unit: SDD pixels
- optimizer: Adam
- initial learning rate: `1e-3`
- Goal scheduler: ExponentialLR, gamma `0.99` (upstream goal-pretrainer)
- joint scheduler: ExponentialLR, gamma `0.995` (upstream GDTS trainer)
- Goal stage: 150 epochs
- joint stage: 250 fresh epochs
- batch size: 64
- K: 20
- DDPM / DDIM / trunk steps: 100 / 20 / 30
- TTST: enabled
- FP32
- seed / fixed validation seed: 2025
- validation: every 10 epochs
- final evaluation: five stochastic runs from the selected joint checkpoint

The paper states exponential annealing but does not publish separate gamma
values. The values above retain the corresponding released-code
implementations.

## Initialization contract

`--goal_pretrain_checkpoint` is accepted only for a fresh independent
baseline training run. The loader requires:

- checkpoint type `gdts_goal_pretrain`, format version 1;
- matching dataset, test target, and Goal U-Net architecture;
- a state dictionary containing only `goal_module.*` parameters;
- strict parameter-name and shape equality.

The joint stage never inherits the Goal-stage optimizer, scheduler, epoch, or
best-selection state. The history encoder and diffusion network are freshly
initialized. The parent path, SHA256, epoch, and planned Goal-stage duration
are recorded in the joint evaluation protocol and checkpoints.

## Commands

From the repository root:

```bash
export PYTHON_BIN=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python
export CACHE_ROOT=outputs/joint_dependency_v2/cache/baseline_source_batches
```

Run the complete pipeline:

```bash
bash tools/run_sdd_paper_baseline.sh all
```

Run stages separately:

```bash
bash tools/run_sdd_paper_baseline.sh pre-process
bash tools/run_sdd_paper_baseline.sh goal-pretrain
bash tools/run_sdd_paper_baseline.sh joint
bash tools/run_sdd_paper_baseline.sh test
```

Resume from each stage's post-scheduler last checkpoint (the default),
or explicitly select a numbered checkpoint:

```bash
bash tools/run_sdd_paper_baseline.sh goal-resume
bash tools/run_sdd_paper_baseline.sh joint-resume
bash tools/run_sdd_paper_baseline.sh goal-resume 100
bash tools/run_sdd_paper_baseline.sh joint-resume 100
```

Formal outputs:

```text
output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/
output/sdd/runs/gdts_paper_sdd_joint_seed2025/
```

Recommended low-resource detached launch:

```bash
mkdir -p output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025
nohup setsid nice -n 10 ionice -c2 -n7 taskset -c 0-3 \
  env PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 \
      OPENBLAS_NUM_THREADS=4 \
      NUMEXPR_NUM_THREADS=4 \
      PYTHON_BIN="$PYTHON_BIN" \
      CACHE_ROOT="$CACHE_ROOT" \
  bash tools/run_sdd_paper_baseline.sh all \
  > output/sdd/runs/gdts_paper_sdd_goal_pretrain_seed2025/pipeline.log 2>&1 \
  < /dev/null &
```

## Evaluation boundary

The released 30/17 protocol uses the same 17 scenes for validation and test.
Consequently, best-checkpoint selection is not a clean independent-test
estimate. Results must be labelled as the published legacy SDD protocol, and
ADE/FDE are reported in pixels.
