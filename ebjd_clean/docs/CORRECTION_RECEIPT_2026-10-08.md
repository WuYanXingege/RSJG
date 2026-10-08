# EBJD correction, data and real-smoke receipt — 2026-10-08

## Outcome

All implementation blockers identified against `01faa633` were corrected
without adding an architecture module. The fixed executable implementation is
`b8605264d18250577da8b89ec506e732235d5aa4`; this document only records the
evidence produced from that code.

Formal training was deliberately not started in this correction run. The
status is **implementation/data ready, effectiveness unknown**.

## Corrections

1. Metrics now expose per-agent and per-scene values plus exact sum/counts.
   Formal marginal values aggregate valid agents over the split; joint and
   coordination-gap values aggregate scenes. Results are invariant to batch
   size, final short batches, padding and shard boundaries.
2. Softmin is normalized as
   `-tau * (logsumexp(-e/tau) - log(P))`; all-zero error is exactly zero.
3. Checkpoint selection first requires both minADE and minFDE to be no worse
   than the bound GDTS validation thresholds, then minimizes
   `JADE + 0.5 * JFDE`. If no candidate is eligible, only
   `least_violation.pt` is written and `simultaneous_improvement=false` remains
   explicit. Pareto and full candidate histories survive resume.
4. Geometry supervision returns one value per scene and aggregates
   `mean(alpha_b² * Lgeo_b)`. A t=1 scene cannot inherit another scene's gate.
5. Bridge encode/decode, DDIM state updates and geometry calculations disable
   outer autocast and execute in FP32. Scale statistics, Gram/dot products and
   optimizer constraint accumulation use FP64.
6. The future-time embedding is fixed; total trainable parameters are exactly
   3,612,764.
7. Config fields are either wired into the trainer/model or rejected. Atomic
   checkpoints contain full resolved config/hash, run/data/source identities,
   optimizer/update/scales, selector state and Python/NumPy/Torch CPU/CUDA plus
   loader-generator RNG. Resume is supported at epoch boundaries only.
8. The exporter writes complete synchronized scene shards with identities,
   frames/timestamps and categorical semantic crops. The loader validates
   hashes and content, caches only one decompressed shard, pads in the collator
   and never truncates agents.

## Complete HOTEL export

Command:

```bash
PYTHONPATH=. ../../.conda/rsjg/bin/python -m ebjd.export_ethucy \
  --source-root ../../GDTS_official_297d508_HOTEL/data/eth5 \
  --fold hotel --output-root data/hotel_official --shard-scenes 16
```

| split | scenes | agent-occurrences | min/max N | manifest SHA256 |
|---|---:|---:|---:|---|
| train | 4,249 | 37,702 | 1 / 57 | `f4ebb40ad8a20f851880b87c953affaa5c612ae80a3041d46dcc51fd6b91b7e8` |
| validation | 445 | 1,197 | 1 / 8 | `90fb42545ad043410f7c12b9ca91a1a0b218895b19252650e62f7fbefdb77d6d` |
| test | 445 | 1,197 | 1 / 8 | `84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78` |

Train–validation and train–test overlap are zero. Validation–test overlap is
445/445. This mirror is preserved and disclosed; it cannot support an
independent final-test claim.

## Parity and loader evidence

Full validation parity passed:

- 445/445 synchronized grouped scene identities matched after mapping by agent
  ID (agent array order is not treated as identity);
- 1,197 agent trajectories / 23,940 positions matched official fragments;
- maximum world-coordinate absolute error: `4.577636740776825e-7 m`;
- maximum pixel-coordinate absolute error: `2.3203079152267492e-5 px`;
- semantic crop controls: 10,773/10,773 agreement, including 9,400
  out-of-bounds zero-fill controls;
- legacy downsampled semantic source agreement: 1.0.

The full train loader read all 4,249 scenes and 37,702 agent-occurrences,
including N=57. Peak process RSS was 1,314,811,904 bytes; maximum observed RSS
increase was 654,155,776 bytes. Validation peak RSS was 607,444,992 bytes.

## Executed validation

| layer | command/evidence | outcome |
|---|---|---|
| Static | `compileall`, `git diff --check` | PASS |
| CPU | `pytest -q` | 20 passed |
| CPU CLI | run id `correction_cpu_smoke_20261008` | scale fit, update, validation and atomic checkpoints PASS |
| CUDA normal update | real N=57, full trainer path | 281 parameter tensors changed; peak 2,107,094,528 B; PASS |
| CUDA rollout update | real N=57, S4, 20 differentiable steps | both applied constraint dots positive; peak 3,524,430,848 B; PASS |
| CUDA inference | real N=57, P20×20, world chunk 2 | output `[1,20,57,12,2]`; peak 990,078,464 B; PASS |
| CUDA BF16 boundary | outer BF16, bridge decode | FP32 outputs, zero difference from FP32, exact endpoint; PASS |
| Formal training | multi-epoch/multi-seed | NOT RUN |

Real-scene smoke metrics are intentionally not interpreted as trained model
performance because the new modules are untrained. The checks test execution,
gradient flow, parameter mutation, constraints and memory only.

## Baseline binding and selection boundary

The validation baseline is the existing synchronized `E_joint` result from
official GDTS epoch 110, P20×20, seed 2035, 445 scenes and 1,197
agent-occurrences. Formal startup recomputes:

- validation manifest SHA256
  `90fb42545ad043410f7c12b9ca91a1a0b218895b19252650e62f7fbefdb77d6d`;
- comparator checkpoint SHA256
  `5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b`;
- comparator result SHA256
  `df9aa35cd26296558c864a2bcda792a8411bbc83a530376c45d3429a768eae6c`.

The EBJD goal-U-Net initializer is separately fixed to epoch 96 with SHA256
`f6a52cf228d733bd684aef043d843ba7d942e40495d5eec354ce1e4e3507ec13`.

## Local evidence paths and hashes

Large data and outputs remain ignored by Git. Their paths and hashes are
preserved in `validation/CORRECTION_VALIDATION_SUMMARY.json`. The primary files
are:

```text
data/hotel_official/DATASET_RECEIPT.json
outputs/audits/hotel_official_export_parity.json
outputs/audits/hotel_official_train_load_resources.json
outputs/audits/hotel_real_smoke_v2_normal_maxN57_b860526.json
outputs/audits/hotel_real_smoke_v2_rollout_maxN57_b860526.json
outputs/audits/hotel_real_smoke_v2_inference_P20x20_maxN57_b860526.json
outputs/audits/bf16_fp32_boundary_b860526.json
```

## Next command (not executed here)

```bash
PYTHONPATH=. ../../.conda/rsjg/bin/python -m ebjd.train \
  --config configs/endpoint_bridge_joint.yaml \
  --device cuda --run-id hotel_seed3101_main
```

Before launching, confirm the GPU is reserved for this run. A formal run will
refuse a dirty `ebjd_clean/` tree, reused run directory or any bound hash
mismatch. Because validation and test are mirrored in this package, later
publication claims still require an independent standard-protocol test.
