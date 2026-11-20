# Optional performance candidates

Status: `CPU_READY_GPU_PROFILE_PENDING`. This package is never imported by the
formal controller. Existing architecture, losses, default callbacks, source
identity checks and training authorization remain unchanged.

| Candidate | Explicit opt-in | Default |
|---|---|---|
| P0 | Instantiate `BatchedModuleTelemetry` instead of `ModuleTelemetry` | Original observer |
| P1 | `perf.stages.joint_callbacks(..., defer_diagnostics=True)` | `False` |
| P2 | `perf.stages.joint_callbacks(..., cpu_mask_check=True)` | `False` |
| P3 | `checkpoint_arm(model, map_checkpoint=False, bridge_checkpoint=False)` | Original checkpoint flags retained/restored |
| P4 | Diagnostic-only algebra in `scripts/perf_audit/mechanisms.py` | Original geometry unchanged |
| P5 | Read-only CPU cacheability probe in the same script | No inference cache |

`baseline`, `p0`, `p01`, and `p012` are incremental benchmark arms. CPU full-state
acceptance compares the baseline, P0 alone, and the P0/P1/P2 combination; the
P0/P1 intermediate arm and every GPU arm still require the pending benchmark.
Checkpoint and telemetry-off arms are optional diagnostics, not recommended
production settings.

The CPU tests include raw outputs, gradients, every-step pre/post state hashes,
AdamW, RNG, data order, native recovery/Top5 and journal replay. Their synthetic
maps are 32 pixels; they do not certify 256-pixel GPU memory capacity or GPU
bitwise equality. Statistics may differ by FP64 reduction rounding independently
of the exact model/optimizer comparison.

The new namespace changes the candidate source manifest. It does **not** make
the current formal frozen SHA valid for modified code. A future formal adoption
requires separate GPU acceptance, a new reviewed/frozen configuration and user
authorization. Do not copy these files into the running experiment.

Manual commands and evidence:
[Chinese report](../../docs/research/bcep_v3/performance_optimization_20261120/REPORT_ZH.md).
