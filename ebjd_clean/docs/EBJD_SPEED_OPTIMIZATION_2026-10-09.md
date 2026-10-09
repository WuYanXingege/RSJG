# EBJD HOTEL scene-staged speed optimization

Date: 2026-10-09 (Asia/Shanghai)

Execution-code commit: `9ecde66aa86f7a577fa29bd94b0168ae735a1f16`

Formal strategy: `compact_grouped_v1`

## Outcome

The OOM-safe one-scene execution remains available as the fallback.  The
validated production strategy additionally compacts invalid agents, uses
deterministic contiguous scene groups bounded by 4 scenes, 96 padded agent
slots and a 1.5 padding ratio, and disables the six nested denoiser block
checkpoints only inside the already checkpointed differentiable DDIM rollout.
It keeps U-Net checkpointing and base-diffusion denoiser block checkpointing.

The model, objectives and logical training protocol are unchanged: 3,612,764
parameters; logical B4 × accumulation 4; complete scenes; S4 × 20-step
differentiable rollout every four optimizer updates; three independent VJPs;
FP64 two-halfspace projection; one optimizer commit; and P20 × 20 validation.

All 32 repository tests pass.  Fixed-draw FP32 and configured BF16 comparisons,
single-factor timing, lightweight profiling, 48 continuous updates and strict
checkpoint migration all pass.  The formal serial queue is running.

## Implementation and semantic preservation

Random time, diffusion noise and rollout initial noise are still drawn once in
the original logical `[B,Npad]` layout, including invalid padding RNG
consumption.  Only afterward are valid agents selected in their original order.
Non-prefix masks are tested.  Agent IDs are kept in valid-agent order, while
scene IDs, frame IDs, timestamps, source sequences and metadata remain
scene-level fields.

Contiguous groups never cross a logical microbatch and never reorder scenes.
For a physical group with `A_g` valid agents and `B_g` scenes inside an original
logical microbatch with `A_m` agents and `B_m` scenes, agent-reduced losses are
weighted by `A_g/A_m` and scene-reduced losses and protected MA/MF gradients by
`B_g/B_m`.  The original K-microbatch average, including the `[4,4,1]` tail, is
then applied before one clip, AdamW proposal, QP and commit.

The denoiser receives an immutable per-call `checkpoint_blocks` flag.  Rollout
closures capture `False`; no global module flag is toggled between forward and
backward.  Base diffusion calls retain the configured block checkpoint policy.

Strict config, run identity and migration receipts now record:

- `execution_strategy=compact_grouped_v1`;
- `scene_compaction=true`;
- maximum scenes `4`, agent slots `96`, padding ratio `1.5`;
- `rollout_block_checkpointing=false`.

These fields are individually semantic-hash exclusions for controlled execution
migration.  The migration entry point accepts only the exact versioned profile,
not arbitrary group budgets.

## Work reduction on the reconstructed failure update

For valid counts `[[3,6,30,1], [57,1,7,8], [3,2,4,8], [15,3,4,5]]`, the
certified scene-at-a-time path retained each logical microbatch's full Npad for
every scene.  The optimized deterministic groups are:

| Logical valid counts | Optimized contiguous groups | Agent slots, old → new | N² slots, old → new |
| --- | --- | ---: | ---: |
| 3, 6, 30, 1 | `[3,6]`, `[30]`, `[1]` | 120 → 43 | 3,600 → 973 |
| 57, 1, 7, 8 | `[57]`, `[1,7,8]` | 228 → 81 | 12,996 → 3,441 |
| 3, 2, 4, 8 | `[3,2,4]`, `[8]` | 32 → 20 | 256 → 112 |
| 15, 3, 4, 5 | `[15]`, `[3,4,5]` | 60 → 30 | 900 → 300 |
| Total | — | **440 → 174** | **17,752 → 4,826** |

These are tensor-slot counts, not FLOP or latency claims.

## Numerical validation

The comparison uses the certified epoch-21 checkpoint SHA256
`9a7a312b44aca95976f727c854b6a9fbd83891dc5cb212733355acffdee133d3`
and scenes with 1, 2, 5 and 3 agents.  CUDA RNG endpoints are exact in every
comparison, and repeated reference runs pass their fixed numerical bounds.

| Precision/path | Max semantic-loss abs diff | Total-gradient relative diff | Adam candidate relative diff | Projected decrement relative diff | Valid rollout trajectory relative diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 rollout off | 1.31e-6 | 1.31e-4 | 7.49e-5 | 7.49e-5 | — |
| FP32 rollout on | 2.41e-6 | 1.28e-4 | 6.59e-5 | 6.60e-5 | 2.81e-7 |
| BF16 rollout off | 2.44e-3 | 4.90e-3 | 1.285e-2 | 1.285e-2 | — |
| BF16 rollout on | 3.05e-4 | 7.21e-3 | 1.909e-2 | 1.911e-2 | 0 |

Projected MA/MF constraints pass the `-1e-6` feasibility gate.  Explicit
category norms verify gradient connections from base and rollout losses to the
trainable U-Net, denoiser, future geometry and map-sampling injection.

An initial FP32 artifact is retained as a validation-script failure: it included
expected execution counters such as physical group count in the semantic-loss
comparison and required non-deterministic CUDA kernels to repeat bit-for-bit.
The implementation already satisfied the pre-registered numerical tolerances.
The corrected script narrows the comparison domain without relaxing those
tolerances.

## Profiling and paired performance

On the same single-agent full 20-step rollout, both paths invoke 20 DDIM-step
checkpoints.  The reference records 486 denoiser block-checkpoint markers
(base plus rollout/recomputation), while the final path records only the 6 base
diffusion markers.  Diagnostic profile wall time falls from 14.16 s to 8.39 s.
Small, mixed and N57 normal profiles also pass.  A mixed-rollout profile attempt
was deliberately aborted because its profiler event table reached about 84% of
host memory; it is not used as evidence.  Formal timing was performed without a
profiler.

The reconstructed failure-group paired benchmark uses one fixed run per
single-factor strategy after warmup.  It includes H2D inside physical groups,
all losses, three VJPs, AdamW, QP and parameter application; loader I/O and JSONL
logging are excluded.

| Strategy | Normal update (s) | Rollout update (s) | 3:1 weighted (s) |
| --- | ---: | ---: | ---: |
| Certified reference | 3.36 | 90.18 | 25.06 |
| Compaction only | 2.57 | 88.07 | 23.95 |
| Compaction + rollout step checkpoint only | 2.46 | 66.30 | 18.42 |
| Above + max group 2 | 1.93 | 45.89 | 12.92 |
| Final max group 4 | **1.75** | **38.32** | **10.90** |

The paired final speedup is `2.3005×`.  This single-factor table is not treated
as a latency distribution.

The final strategy then completed 48 continuous mixed updates with 12 rollouts:

- normal p50/p90: 1.94/2.22 s;
- rollout p50/p90: 44.58/52.53 s;
- actual-frequency mean: 12.99 s/update;
- maximum allocated/reserved: 5,689,860,608 / 6,180,306,944 bytes;
- minimum conservative driver-free estimate: 9,191,030,784 bytes;
- reconstructed failure rollout: 39.63 s;
- dense N57 × logical B4 × accumulation 4 rollout: 72.57 s;
- `[4,4,1]` K3 tail: passed;
- every record and updated state: finite, no OOM.

At 266 optimizer updates per full epoch, the update-only estimate is about
0.96 h/epoch.  The actual complete-epoch wall time will also include data work
and P20 × 20 validation and must be measured from the first completed epoch.
The remaining queue has 79 seed-3101 epochs plus 100 epochs each for seeds 3102
and 3103; update-only time is roughly 11.2 days, before validation overhead.

## Migration and formal queue

The user explicitly chose to restart from epoch 21 rather than wait for the
slow epoch 22 to finish.  The old queue stopped after logging update 5688; 102
uncheckpointed epoch-22 updates were intentionally discarded.  The immutable
origin remains epoch 21/update 5586.

Migration-only validation created a new epoch-21 boundary checkpoint SHA256
`c9d994a0a4c49ac09527290d182d8144248db0c3fd514ae6d282af33cf358ea3`.
Both restoration passes are exact for model, optimizer, update index, Python,
NumPy, CPU/CUDA and loader RNG.  All 21 selector candidates and the epoch-14
least-violation pointer are preserved.

Formal queue `hotel_main_speed_9ecde66_20261009_115417` started at
2026-10-09 11:56:44 +08:00:

- supervisor PID `1263005`;
- seed-3101 PID `1263016`, RUNNING from epoch 22/update 5586;
- seeds 3102 and 3103, QUEUED and serial;
- manifest SHA256 `30a1d292d0d233983c12d409543d42823ca6a72a15faa91f41b2ed3aedf68463`;
- first-update receipt SHA256
  `84f81a20ea396a8fa30b54102ca8fe0a90ea0c10aa4cd97bbc248dd5bd4c22e2`.

The first formal normal updates completed in 1.63–3.28 s.  The first complete
rollout update (5589) completed in 45.27 s with peak allocated/reserved
5,074,056,704 / 5,563,744,256 bytes; losses, protected dots, optimizer and
parameters are finite.  The execution worktree remains clean and fixed at
`9ecde66`, so later documentation commits cannot invalidate queued seeds.

## Evidence boundary

This work validates execution equivalence, memory safety and throughput.  It
does not establish EBJD method effectiveness.  The 445 mirrored validation/test
windows remain internal-development evidence, and final multi-seed training and
test evaluation are incomplete.
