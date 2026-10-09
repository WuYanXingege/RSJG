# EBJD HOTEL epoch-22 OOM recovery

Date: 2026-10-09 (Asia/Shanghai)  
Incident: 2026-10-08 21:47:36 +08:00  
Branch: `research/ebjd-clean-v1`  
Execution-code commit: `709d2d93b9cb701e5a496642e7c3539e4f992f9d`

## Decision

The old serial queue remains an immutable `FAILED` record. Seed 3101 is
recoverable only from the complete epoch-21 boundary; the 46 unsaved epoch-22
updates are discarded. The recovery keeps the original logical B4 × gradient
accumulation 4, S=4 × 20-step differentiable rollout, BF16/FP32 policy, loss,
optimizer, scheduler, data order and validation contract. It changes the
physical graph lifetime to one scene at a time and stages pending batches on
CPU.

## Bound source and failure evidence

- Source `last.pt`: SHA256
  `9a7a312b44aca95976f727c854b6a9fbd83891dc5cb212733355acffdee133d3`.
  It is `ebjd-checkpoint-v2`, `epoch_boundary_only=true`, epoch 21,
  `update_index=5586`.
- A read-only copy is stored under
  `outputs/recovery_sources/epoch22_oom_20261008/` with the same SHA256.
- Model/optimizer audit: 901 tensors, 10,840,097 elements, zero non-finite
  elements. The constructed model has 3,612,764 parameters.
- Restored optimizer LR values are `9.323119580739992e-05` and
  `9.323119580739993e-06`; the old loader, Python, NumPy, CPU and CUDA RNG
  states are present.
- Selector history has 21 candidates, `best_eligible=null`, and the inherited
  least-violation candidate is epoch 14. The corresponding old artifact SHA256
  is `f8a741567b798674c75b57b7de2c11358d07fb4035410732c11512895a22d281`.
- Old queue-state SHA256:
  `784defdb80df39c5b77dbcfee7eaf797fc3df8995b2c9584c695937880329ace`;
  old seed-3101 log SHA256:
  `17680a8f1ba2bbf0235e7c2eaafe884841b4d02986365d4f4c383850cb7a1fcd`.
  The traceback reports an attempted 1.78-GiB allocation with 986.06 MiB
  free, 11.40 GiB allocated and 2.04 GiB reserved but unallocated.

The exact failing C++ allocation site is not present in the traceback. The
request size exactly matches a B4 × N57 × 32 × 256 × 256 FP32 map-feature
tensor (`1.78125 GiB`) produced around `_sample_map`/`grid_sample`; this remains
strong localization evidence, not proof that the forward cast rather than its
backward/recomputation made the failed request.

The recovered epoch-22 loader order reconstructs the failing optimizer update
47 as logical padded-N values `[30, 57, 8, 15]`, with valid-agent counts
`[[3,6,30,1], [57,1,7,8], [3,2,4,8], [15,3,4,5]]`.

## Semantics retained

Random time, diffusion noise and rollout initial noise are drawn once in the
original `[B,Npad]` layout, then sliced by scene. Padding RNG consumption is
unchanged. Agent-reduced diffusion/map terms use `n_s/A`; scene-reduced
geometry, rollout, MA and MF terms use `1/B`. The actual number `K` of logical
microbatches is retained, including the final `[4,4,1]` tail. Only after all K
logical microbatches does the trainer perform one global clip, one AdamW
proposal, one two-halfspace projection and one optimizer-state commit.

The projection routine was also made genuinely FP64 through normalization,
active-set solution and vector combination. Validation exposed that the old
routine cast normalized rows/multipliers back to FP32 and could select its zero
fallback at a tangent boundary even though the intended FP64 QP had a non-zero
solution. This is a numerical implementation repair of the existing QP, not a
new objective.

## Validation

All 26 repository tests pass, including unequal-agent reduction, original-N
scene slicing, tail-K behavior, rollout-on/off update equivalence, optimizer
LR/state/name restoration, archived-config compatibility and a tangent-QP
regression.

The final CUDA comparison uses scenes with 1, 2, 5 and 3 valid agents.

- FP32 localization passes: rollout-on total-gradient relative error
  `1.2406e-4`, Adam candidate `6.5400e-5`, and projected decrement
  `6.5497e-5`.
- Configured BF16 passes its machine-precision bounds: rollout-on total/MA/MF
  gradient relative errors are `0.006995`, `0.007609`, `0.007160`; projected
  decrement relative error is `0.018447` with direction cosine `0.99982984`.
  CUDA RNG is exact and the projected constraint dots are `0.00881797` and
  `-2.50e-10` (above the `-1e-6` feasibility gate).
- The first, stricter 0.005 BF16 attempt is retained rather than relabelled. It
  failed because a raw-gradient bound had also been applied to the nonlinear
  Adam candidate. The final script separates raw gradients (`2 × BF16 eps`)
  from Adam/QP decrements (`3 × BF16 eps`) and records both. FP32 localization
  rules out a reduction-weight or wiring discrepancy at this scale.
- Dense B4×4 rollout with Npad=57 passes in 99.29 s: peak allocated
  3,591,031,808 bytes, peak reserved 4,305,453,056 bytes, capacity margin
  12,312,248,320 bytes.
- The exact reconstructed failure group passes in 94.72 s: peak allocated
  3,591,031,808 bytes, peak reserved 4,263,510,016 bytes, capacity margin
  12,354,191,360 bytes. All losses, parameters and optimizer state are finite.
- N57 P20×20 inference passes in 0.49 s with finite `[1,20,57,12,2]` output.
- A fresh diagnostic process completed 48/48 mixed-size updates, including 12
  rollout updates. The original failing padded-N group was reproduced at
  ordinal 47. Maximum allocated/reserved memory was 3,591,031,808 /
  4,292,870,144 bytes, the minimum capacity margin was 12,324,831,232 bytes,
  and every recorded scalar, parameter and optimizer tensor remained finite.
- A GPU migration-only preflight then restored model, optimizer, update index,
  Python/NumPy/CPU/CUDA/loader RNG exactly, saved an epoch-21 checkpoint under
  the new execution identity, and restored that checkpoint a second time with
  every exact-state check passing. It preserved all 21 selector candidates,
  `best_eligible=null` and the epoch-14 least-violation pointer. The preflight
  migration receipt SHA256 is
  `defa68e2a43ee7f9a05503a64a1154c8d21945b635457b13ad8c4c023a6c2858`.
- The formal queue start is a separate post-validation action; it is not
  inferred from the diagnostic copy or migration-only preflight.

Large validation artifacts remain under ignored
`outputs/oom_recovery_validation_20261008/`. Their hashes are listed in the
structured summary.

## Evidence boundary

This repair establishes bounded execution and controlled state continuity. It
does not establish method effectiveness. The 445 validation/test windows are
mirrored official-package windows and remain internal-development evidence.
Epoch 21 improved over epoch 20 but still violates both bound baseline marginal
thresholds; multi-seed training is incomplete.
