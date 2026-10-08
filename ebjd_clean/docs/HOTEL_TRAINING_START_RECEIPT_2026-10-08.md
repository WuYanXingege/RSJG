# EBJD HOTEL formal training start receipt — 2026-10-08

## Status at 2026-10-08 17:04:34 +08:00

The fixed HOTEL main-model queue is **RUNNING**. Training seed 3101 completed
its first real optimizer update and continued to update 29 at this snapshot.
Seeds 3102 and 3103 are **QUEUED** and will start serially only after the prior
run exits successfully.

This is a launch receipt, not a completion or performance report. No validation
epoch has completed and no effectiveness claim is made.

## Fixed execution identity

- Source commit: `e7a1948e63f3f39cfc2f09243a97235928dabd09`
- Branch at launch: `research/ebjd-clean-v1`
- Runtime worktree:
  `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_clean`
- GPU: physical GPU 0, NVIDIA GeForce RTX 5070 Ti, 16,303 MiB
- Supervisor PID: `1208390`
- Active seed3101 PID: `1208393`
- Queue ID: `hotel_main_e7a1948_reshard1_20261008_170006`
- Queue manifest SHA256:
  `f96f064761abe3102994a5a8d7c2785df99ff83bf43d9b380304d22bc12c2eef`

The reporting files were created in a separate detached worktree. The runtime
worktree remains clean at `e7a1948`, so future epoch-boundary resume identity is
not changed by this receipt commit.

## Queue

| order | seed | run ID | config hash | launch status |
|---:|---:|---|---|---|
| 1 | 3101 | `hotel_main_seed3101_e7a1948_reshard1_20261008` | `8974a66034bdedbf46c868fd87cab1b79222f4fd64627ef9d12099aa9e5ee3fd` | RUNNING |
| 2 | 3102 | `hotel_main_seed3102_e7a1948_reshard1_20261008` | `0b2faeb3e3b30775dea99643287426dc7ad5a91afc471eb8bd6423f4b8709a8a` | QUEUED |
| 3 | 3103 | `hotel_main_seed3103_e7a1948_reshard1_20261008` | `8cc545f20e0103a9512614d080e4958e959bb773d5d94176b37ccefe682e2a75` | QUEUED |

All three configs retain 100 epochs, batch 4 scenes, accumulation 4, 100 noise
bins, S4×20 rollout, P20×20 validation and the original BF16/FP32 precision
boundary. Only training seed and run identity differ.

Output directories:

```text
ebjd_clean/outputs/formal_hotel_e7a1948_reshard1_20261008/hotel/3101/none/hotel_main_seed3101_e7a1948_reshard1_20261008/
ebjd_clean/outputs/formal_hotel_e7a1948_reshard1_20261008/hotel/3102/none/hotel_main_seed3102_e7a1948_reshard1_20261008/
ebjd_clean/outputs/formal_hotel_e7a1948_reshard1_20261008/hotel/3103/none/hotel_main_seed3103_e7a1948_reshard1_20261008/
```

Queue logs and state:

```text
ebjd_clean/outputs/queue_configs/hotel_main_e7a1948_reshard1_20261008_170006/QUEUE_STATE.json
ebjd_clean/outputs/queue_configs/hotel_main_e7a1948_reshard1_20261008_170006/logs/seed3101.log
```

## Data and comparator binding

- Original complete HOTEL receipt SHA256:
  `ed3fe7365328b698d717275471bb087e6674a1bfa5bd7d077549d7d4cee21e80`
- Original train manifest SHA256:
  `f4ebb40ad8a20f851880b87c953affaa5c612ae80a3041d46dcc51fd6b91b7e8`
- Lossless one-scene-per-NPZ train manifest SHA256:
  `2a5f1909d928e5277d426ba04e3488940e34a5f71fad9da6331e5e673ec1966c`
- Reshard receipt SHA256:
  `cae88c8b53d3df5053570b75b93147e76d842176f8b1e4cc9ce6373049398ef8`
- Validation manifest SHA256:
  `90fb42545ad043410f7c12b9ca91a1a0b218895b19252650e62f7fbefdb77d6d`
- Goal-U-Net epoch-96 initialization SHA256:
  `f6a52cf228d733bd684aef043d843ba7d942e40495d5eec354ce1e4e3507ec13`
- GDTS epoch-110 comparator checkpoint SHA256:
  `5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b`
- Comparator result SHA256:
  `df9aa35cd26296558c864a2bcda792a8411bbc83a530376c45d3429a768eae6c`

The reshard is a storage-only transformation. It preserved all 4,249 scenes,
37,702 agent-occurrences, original scene order and maximum N=57. Every written
scene array was reopened and checked for exact equality. This retained global
shuffle semantics while avoiding repeated decompression of a multi-scene NPZ.
Dataset initialization/hash verification took 1.47 s and 128 random scene
accesses took 2.49 s in the preflight measurement.

## Preserved failed prelaunch attempt

The initial queue `hotel_main_e7a1948_20261008_164457` was stopped before any
optimizer update. Global shuffle over 266 compressed multi-scene shards caused
repeated full-shard decompression; after approximately three minutes it still
had not completed scale fitting and used only 284 MiB GPU memory. The queue is
retained as `FAILED`, exit 143, with its empty training log and run identity.
No checkpoint or result from that attempt is reused.

## First real optimizer update

Receipt SHA256:
`1f8a8baa00462363c9013b5f5c87442100dfd8e539a10a452816df428f274344`.

The first update was epoch 1/update 1 and used four microbatches of four scenes
each (16 effective scenes). It contained 96 valid agent-occurrences; padded N
was `[5, 6, 33, 8]`, with `[6, 6, 90, 10]` padding slots. Key observations:

- update elapsed: 2.571 s;
- CUDA peak allocated: 4,927,405,056 bytes;
- CUDA peak reserved: 8,260,681,728 bytes;
- total loss: 1.963073, diffusion: 1.958646;
- geometry: 0.026166, map: 0.031183;
- applied decrement norm: 0.032596;
- finite real parameter update completed.

Rollout is disabled by protocol in epochs 1–20, so the first update correctly
records `rollout_update=0` and zero constraint dots. The first required rollout
evidence will occur in epoch 21, not during this launch receipt.

By update 29 the run remained active. The largest observed padded N was 51;
maximum per-update peak allocated memory was 7,850,637,312 bytes and peak
reserved was 15,128,854,528 bytes. This is close enough to total device memory
that OOM monitoring must remain active when the N=57 scene appears. No agent,
world or sampling step has been removed.

## Interpretation boundary

The fixed marginal constraints remain minADE ≤ 0.1352196876519829 m and
minFDE ≤ 0.19338157261354552 m at validation seed 2035. No candidate has yet
been evaluated. `simultaneous_improvement` will only mean marginal feasibility;
joint improvement and all-four improvement must be calculated separately.

Validation and test are the same 445 windows. Results from this queue remain
internal HOTEL development evidence and cannot be described as an independent
final test or a five-fold benchmark result.
