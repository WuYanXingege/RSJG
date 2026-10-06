# Experiment preregistration

- Parent: official GDTS HOTEL epoch 110, fixed SHA256 in `OFFICIAL_PROTOCOL_MANIFEST.json`.
- Architecture: canonical strict-no-z Stage-A; GDTS and dependency corrector frozen;
  4 relation modes, energy rank 8, K=21 candidates, P=20 joint worlds, two exact
  lexicographic persistent-tie refinement rounds.
- A0: `mean_energy`. A1: `expected_conditional_mc`, S=4, CPU ObjectiveRNG IDs and
  CUDA FP32 effective loss under BF16 autocast. No parameter-count difference.
- Seeds: train 3101/3102/3103; inference 2035–2039. Each training-seed pair shares
  one authenticated zero-update full model state.
- Optimizer: Adam, lr 1e-4, clip 1; ExponentialLR gamma 0.995 once per complete
  epoch; 40 epochs; no early stop; one whole-scene window per update; no augmentation;
  num_workers=0.
- Selection: lowest complete-validation scene-mean JADE, then JFDE, then earlier epoch.
- Precision/resource limits: CUDA BF16; 12 GiB peak reserved; 6 hours per arm,
  36 hours training, 48 hours full queue.
- Primary comparison: per training seed average the five inference seeds, then report
  paired A1−A0 effects. Improvement requires JADE −0.01 m and −2%, same direction
  for at least 2/3 seeds, with the registered ADE/FDE/JFDE/collision no-harm limits.
- Mechanism checkpoint: A1 seed 3101 selected epoch. Interventions are pair cost zero,
  fixed straight-path closest-approach cost `max(0, 1-d_min/0.4m)^2`, and removal of
  the uniform-reference two-way interaction while retaining additive row/column terms.
- No Stage-B, other fold, hyperparameter search, structural change, or SDD action.

If any source, scene isolation, finite, frozen-family, checkpoint, memory, or deadline
contract fails, the queue stops and retains the failure. It does not skip windows or
change precision, epochs, seeds, or budgets.
