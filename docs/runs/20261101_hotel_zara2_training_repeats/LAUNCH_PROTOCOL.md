# HOTEL/ZARA2 independent training-seed repeats

- Asset root: `RSJG_EBJD_assets/experiments/hotel_zara2_training_repeats_20261101_113517`
- Fixed order: HOTEL/3102 → ZARA2/3102 → HOTEL/3103 → ZARA2/3103 → CPU summary.
- Provider seed: 3101 for every run.
- New training seeds: 3102 and 3103; each run constructs a fresh unified model and optimizer.
- Training contract: 192/4/768, 3+3 FJAB, FULL/Raw, 30 epochs, rollout/projection OFF.
- Inference contract: 20 worlds, 20 DDIM steps, selection seed 2035, final seeds 2035–2039.
- Total cumulative GPU wall-clock limit: 48 hours.
- A schema-rejected prelaunch attempt consumed 1.418282421 seconds and is
  conservatively carried into the cumulative budget; it performed zero
  optimizer updates.
- The two existing cleanup directories are intentionally excluded from this commit.

The durable asset-side `RUN_MANIFEST.json`, `GPU_BUDGET.json`, effective
configs, checkpoints, raw logs, and per-run reports are authoritative. This
small tracked file records the launch contract only; it does not claim that
the background pipeline has completed.
