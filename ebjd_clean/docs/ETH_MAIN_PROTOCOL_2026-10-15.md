# EBJD ETH main protocol (seed 3101)

This run is the fold-matched continuation of the completed HOTEL and UNIV
experiments. It changes the held-out ETH/UCY fold and fold-specific fitted
state only; model, objective, optimizer, schedule, selection rule, inference
budget, and execution semantics remain unchanged.

- Dataset/protocol: ETH/UCY leave-one-scene-out, held-out `eth`; 8 observed
  frames, 12 predicted frames, 0.4 s interval, metric coordinates.
- Scene policy: complete synchronized scenes; all agents retained; validation
  and test are the mirrored official package split and are not independent.
- Training seed: 3101; one training seed is executed and the three-training-seed
  minimum remains disclosed rather than silently relaxed.
- Initialization: fresh fold-specific 150-epoch FP32 GDTS goal pretraining;
  only the goal U-Net tensors are imported and remain trainable.
- EBJD: 100 epochs; logical batch 4 scenes; gradient accumulation 4; BF16
  training with FP32 geometry/loss/optimizer arithmetic; AdamW projected on
  actual optimizer steps; save every epoch.
- Learning rates: new modules `1e-4`, goal U-Net `1e-5`; five-epoch warmup,
  cosine decay to 1% of the initial rate.
- Rollout objective: epochs 21--40 ramp from 0.01 to 0.20, every four optimizer
  updates with 4x frequency compensation; 4 worlds and 20 DDIM steps.
- Selection: seed 2035, strict marginal tolerance 0, preregistered constrained
  selector; if no eligible checkpoint exists, report the least-violation
  checkpoint without relabelling it as a pass.
- Final evaluation: inference seeds 2035--2039, 20 worlds, 20 DDIM steps,
  `minADE`, `minFDE`, `JADE`, and `JFDE`; no per-agent reordering.

The held-out population is the JMM/AgentFormer ETH source pinned at commit
`f4c29aa...`: 253 synchronized scenes and 364 agent occurrences, or 1.438735
agents per scene. The matched GDTS comparator is checkpoint `126acf2a...`,
evaluated directly on that exact validation manifest. Seed 2035 is frozen in
`docs/validation/EBJD_ETH_MATCHED_GDTS_SELECTION_2026-10-15.json` and is the
only baseline used by the preregistered selector. Seeds 2036--2039 are computed
after EBJD training and combined with seed 2035 only for the final five-seed
method comparison. The older 139-window audit and the preliminary 904-window
raw-GDTS export remain historical provenance and are not used by this run.

The persistent supervisor executes goal pretraining, materialization, CUDA
preflight, EBJD training, checkpoint freeze, five-seed evaluation, report
generation, Git commit, and GitHub push attempt in that order. Any failed
contract check stops the pipeline and records `FAILED` in `PIPELINE_STATE.json`.
