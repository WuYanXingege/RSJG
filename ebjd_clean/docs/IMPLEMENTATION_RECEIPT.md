# EBJD clean v1 implementation receipt

## Identity and status

- Date: 2026-10-08 (Asia/Shanghai)
- Branch: `research/ebjd-clean-v1`
- Review baseline: `01faa633162a2b73794a40184e8d64e4d44964bc`
- Corrected implementation commit used by final CUDA checks:
  `b8605264d18250577da8b89ec506e732235d5aa4`
- Frozen design SHA256:
  `9296937cc81f0e64a94b7c774a76e04d489fffe76e8ee5b64ca9bda3ff3af1af`
- Change boundary: `ebjd_clean/` only
- Static/CPU tests: **PASS**
- CUDA synthetic and real-scene bounded checks: **PASS**
- Complete HOTEL scene export/parity/load audit: **PASS**
- Formal multi-epoch/multi-seed training: **NOT RUN**
- Metric improvement claim: **NONE**

This receipt supersedes the implementation receipt at `01faa633`. Details and
reproducible evidence are in `CORRECTION_RECEIPT_2026-10-08.md` and
`validation/CORRECTION_VALIDATION_SUMMARY.json`.

## Implemented contract

The package contains an invertible endpoint plus whitened Brownian-bridge
representation, trainable 23-convolution goal U-Net, history GRU and two
history social blocks, 72-token history/map memory, six full factorized future
blocks, four independent v-heads, differentiable clean-future geometry/map
injection and shared deterministic DDIM training/inference sampling. Padding
queries are zeroed, social attention is scene/world-local and no future target
is an inference condition.

The final instantiated graph has **3,612,764 trainable parameters**. The
`[12,128]` future-time table is a fixed buffer, not a trainable parameter.

The corrected implementation provides:

- normalized softmin and split-wide sum/count metric aggregation;
- agent-weighted formal minADE/minFDE and scene-weighted JADE/JFDE/gaps;
- marginal-constrained selection followed by `JADE + 0.5 × JFDE`;
- per-scene `alpha²` geometry gating;
- explicit FP32 bridge/DDIM/geometry regions and FP64 scale/constraint sums;
- YAML wiring or strict rejection of unsupported fields;
- atomic epoch-boundary checkpoints with resolved config, identity, selector,
  optimizer, scales and Python/NumPy/Torch/loader RNG state;
- complete-scene sharded manifests and one-shard bounded loading.

## Initialization and comparator are separate

EBJD initializes only its trainable goal U-Net from goal-only epoch 96:

```text
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean/output/hotel/runs/grouped_fresh_goal_official_prior_seed3101_20261005_114637/epoch_096.pt
SHA256 f6a52cf228d733bd684aef043d843ba7d942e40495d5eec354ce1e4e3507ec13
```

Validation constraints use the official full GDTS epoch-110 checkpoint and its
synchronized P20/20-step seed-2035 result:

```text
checkpoint SHA256 5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b
result SHA256     df9aa35cd26296558c864a2bcda792a8411bbc83a530376c45d3429a768eae6c
minADE            0.1352196876519829 m
minFDE            0.19338157261354552 m
```

These artefacts have different roles and are not represented as the same
parent. Formal startup recomputes the comparator checkpoint/result and
validation-manifest hashes before constructing a run.

## Dataset boundary

The complete HOTEL export is local and ignored by Git:

```text
ebjd_clean/data/hotel_official/
DATASET_RECEIPT.json SHA256 ed3fe7365328b698d717275471bb087e6674a1bfa5bd7d077549d7d4cee21e80
```

It contains 4,249 train scenes / 37,702 agent-occurrences (maximum N=57), and
445 validation scenes / 1,197 agent-occurrences (maximum N=8). Test has the
same counts. Train has zero overlap with validation and test. Validation and
test are exact mirrors (445 windows), so their evaluation is internal evidence
only and is not an independent final test.

The exporter independently implements the coordinate operations and does not
import legacy code at runtime. The audited reference files at official GDTS
commit `297d508` are:

```text
data_utils.py  be6f647ecd7e3819cb0f3c32edb83079af475727bd944107242d0d9ac6c56be2
scene_eth5.py  b772037f9ddc7fcca3a4c06c3daf2ec85fad0c008e8ac153feb4e4fab587274e
scene_base.py  3de849142b83db9430b1361c4282f53acebee25191951eefe00ccfc4654310f6
```

## Empirical scope

The final checks establish executable semantics, data identity and capacity on
the observed maximum N. They do not establish generalization or improvement.
A formal result still requires the preregistered multi-seed training, validation
selection and an independent standard-protocol final test. All failed seeds and
negative ablations must be retained.
