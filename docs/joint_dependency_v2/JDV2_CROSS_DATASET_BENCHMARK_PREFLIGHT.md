# JDV2 cross-dataset benchmark preflight

## Status and scope

**`CROSS_DATASET_BENCHMARK_BLOCKED`**

This was a preflight-only audit at starting source commit
`480bda5f694010395750e122171f05c837e4e535`. It did not train Stage A or
Stage B, did not evaluate cross-dataset benchmark metrics, and did not change
the frozen scientific method. The machine-readable evidence is
`outputs/joint_dependency_v2/cross_dataset_preflight/results.json`.

No target is currently ready for Stage-A reproduction. Therefore there is no
`eligible_for_<target>_stage_a_reproduction` next state yet.

| Target | Protocol | GDTS checkpoint | Source cache | JDV2 cache | Config | CUDA smoke | Ready? |
|---|---|---|---|---|---|---|---|
| HOTEL | Original GDTS mirrored valid/test | Missing | Canonical cache missing | Missing | A/B templates generated; B unresolved | Generic V2-A BF16 contract passed; target A not runnable | No — `TARGET_BLOCKED_MISSING_GDTS_CHECKPOINT` |
| UNIV | Original GDTS mirrored valid/test | Available, authenticated, epoch 100 | Canonical cache missing; legacy cache rejected | Missing | A/B templates generated; B unresolved | Generic V2-A BF16 contract passed; target A missing cache | No — `TARGET_BLOCKED_MISSING_CACHE` |
| ZARA1 | Original GDTS mirrored valid/test | Missing | Canonical cache missing | Missing | A/B templates generated; B unresolved | Generic V2-A BF16 contract passed; target A not runnable | No — `TARGET_BLOCKED_MISSING_GDTS_CHECKPOINT` |
| ZARA2 | Original GDTS mirrored valid/test | Missing | Canonical cache missing | Missing | A/B templates generated; B unresolved | Generic V2-A BF16 contract passed; target A not runnable | No — `TARGET_BLOCKED_MISSING_GDTS_CHECKPOINT` |

## 1. Dataset protocol audit

The audit instantiated/inspected the repository's original ETH5 experiment
logic in `experiment_eth5.py`. It is leave-one-out: each target's own scene is
validation/test, while the remaining scenes form training. In all four
targets, the ordered validation and test raw files are byte-identical.
Consequently every target is classified
`ORIGINAL_GDTS_MIRRORED_VAL_TEST`. These future results must be described as
following the original GDTS ETH/UCY protocol, not as an independent held-out
test after model selection.

Observed ordered raw inputs were:

- HOTEL train: `crowds_zara02`, `crowds_zara03`, `students001`,
  `students003`, `crowds_zara01`, `biwi_eth`; valid/test: `biwi_hotel`.
- UNIV train: `biwi_hotel`, `crowds_zara02`, `crowds_zara03`,
  `crowds_zara01`, `biwi_eth`, `uni_examples`; valid/test: `students001`,
  `students003`.
- ZARA1 train: `biwi_hotel`, `crowds_zara02`, `crowds_zara03`,
  `students001`, `students003`, `biwi_eth`, `uni_examples`; valid/test:
  `crowds_zara01`.
- ZARA2 train: `biwi_hotel`, `crowds_zara03`, `students001`,
  `students003`, `crowds_zara01`, `biwi_eth`, `uni_examples`; valid/test:
  `crowds_zara02`.

The distinct raw-file identities are:

| File | Bytes | SHA256 |
|---|---:|---|
| `biwi_eth.txt` | 224263 | `cd75b1008b82b7f442b2e03967b0f4aac36da2e73d440df1f605bb197d23fb33` |
| `biwi_hotel.txt` | 143799 | `9caa771bb9153d6b809dd0916b6f86761b641e6bbb15e766c1de3133fbbb7fcf` |
| `crowds_zara01.txt` | 205711 | `1147a1962a09abfb86f28c6cddcac862e095a0cf129b3016385b69eacdd09d85` |
| `crowds_zara02.txt` | 392227 | `8a649d0f8c9ae75c87c4d23a85f892786b0aa30266e996c7be03e69dafff22ff` |
| `crowds_zara03.txt` | 199834 | `16b3e899932c4baacd07f45013d5b921f90bc5a29eb2b0fe42f4d7c904ac3108` |
| `students001.txt` | 879714 | `a6d87f278d94136fe39b8be91555487a29ac77259ae403b9dba2d5c18caf7b5b` |
| `students003.txt` | 724279 | `e25798b660634330aa89f8bb259425de720e84d0873902726c1d1f4ccff21d6c` |
| `uni_examples.txt` | 109307 | `61f432c0ab3070ed0ef150fbeabcd7baf839cab5495a46e6105bd747f0a092a7` |

Full per-target relative paths, ordering, size, scene label, and hash are in
`results.json`. Canonical materialized caches do not exist, so content-level
valid/test identity could not be established for HOTEL/ZARA1/ZARA2. A legacy
UNIV cache has 947/947 ordered valid/test hard-link identities, but it is not
accepted as current JDV2 provenance.

## 2. GDTS checkpoint audit

Only UNIV has a located original-protocol checkpoint:

- path: `../GDTS/output/univ/saved_models/best_model.pt`;
- SHA256: `ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e`;
- stored epoch: 100;
- config: `../GDTS/output/univ/config.yaml`, SHA256
  `07452f9693ea2f19c802be73b0c0a78ec8f1a881deabb7a5994cab9b41cc1c09`;
- protocol phase: `train_test`;
- architecture: its 112 state-dict keys and tensor shapes match the ETH GDTS
  source architecture exactly.

HOTEL, ZARA1, and ZARA2 have no located original/frozen GDTS checkpoint and
are marked `BASELINE_CHECKPOINT_MISSING`. This preflight did not train or
fabricate them. The ETH checkpoint is explicitly forbidden as a substitute.

## 3. Cache audit

The canonical synchronized source-cache roots are expected at:

```text
outputs/joint_dependency_v2/cache/source_batches/eth5/<target>/
data_batches_jdv2_v2
```

None of the four exists. Consequently train/valid/test completeness,
canonical `scene_index`/`scene_ptr`, synchronized frames, units, ordering, and
cross-scene isolation cannot be certified for current JDV2 inputs.

UNIV has a legacy candidate at
`../GDTS/output/univ/joint/data_batches_joint_v2`, manifest SHA256
`c27f4488a8f963cacf7f91026dab183e6f40db004a44ab71aa3d1160db87c50b`.
It contains deterministically numbered 3302/947/947 train/valid/test windows
and valid/test are 947/947 hard-link identical. It is rejected because its
manifest declares `goal_model_type=joint` and its root/schema is not the
current `data_batches_jdv2_v2` / `goal_model_type=joint_dependency_v2`
contract.

Target Stage-A JDV2 caches are also absent at
`outputs/joint_dependency_v2/cache/<target>_full_stage_a`. They must later be
built from the same target's authenticated GDTS checkpoint and certify
`jdv2-cache-v1`, K=21, frozen candidate order, map/world candidate shapes,
canonical sparse edges, `world_m`/`world_m_per_s`, source commit, source
checkpoint hash, and manifest hash. No ETH cache may be reused.

## 4. Frozen method and generated configurations

Eight canonical templates were generated:

| Stage | HOTEL | UNIV | ZARA1 | ZARA2 |
|---|---|---|---|---|
| Stage A SHA256 | `957d094cd322e51fd732ea286cef75eb99dbdba4d726e0509c955d89b7213def` | `efa60733e20153e11798181d2d52ab9bb3a84b7cd240ca1edfd72ff48de40560` | `0a3d22d316055eae3d0d9efe2b6a34707490347d5fddedc7698b6d8806959e9f` | `bc5042b01a9c4686262d3776138add3a2a23b236c894dc1656dd487089e7f759` |
| Stage-B V2-A SHA256 | `bfb1975e88c211dece77623fb9e36cdfbd88fdf1c358417c77b5f29bf0433a1f` | `59c3f55b68bf97b9c97e28c88a7afdd9ad374e29b8ccded1b34974a8aeeb3cf0` | `5427e0a041437fb9c656ed944dfe795b114fbd44082b15e6e73edd7f72b0d1d2` | `4bf634709794d1a349efb51e3e56c73469d6ec8ca924b91c0115866c0acc43a2` |

All templates preserve strict-no-z, K=21, P=20, M=4, rank 8, radius/TTC
6 m/8 s at dt=0.4 s, weighted Gumbel-Top-P Round 0, two synchronous exact
persistent refinement rounds, Adam/1e-4/ExponentialLR, BF16 with the existing
FP32 islands, and seeds 2035–2039. V2-A preserves the 30,851-parameter
DependencyCorrector, `component_zero_mean`, active timesteps
`[30,25,20,15,10,5]`, same-slot relation, one scene-level oracle branch,
`lambda_diff=1.0`, and `lambda_relative=0.05`.

Only target identity, run/output roots, cache roots, and target provenance are
allowed to differ. Each target writes below
`outputs/joint_dependency_v2/<target>/...` and has an isolated cache root.

Every Stage-B template intentionally leaves the target Stage-A parent hash,
freeze-manifest hash, and freeze-source commit null. A target-aware parser
gate now rejects such a template with an explicit unresolved-provenance error;
after those fields are populated, target-token checks reject cross-target
cache, checkpoint, run, or freeze-manifest paths. Trainer checkpoint loading
accepts a target-specific freeze-manifest path and verifies its hash, parent
checkpoint hash, dataset identity, and source commit. The historical ETH
fallback remains unchanged.

## 5. Fairness and smoke status

For every target, planned GDTS, Stage A, and V2-A use the same target's
original mirrored validation/test semantics, metric implementation, world
conversion, P=20 deployment, and fixed five seeds. V1 cross-dataset training
is not part of the plan.

Target Stage-A forward/coverage smokes were not run because no target has
both an authenticated checkpoint and canonical JDV2 caches. Target Stage-B
smokes were not fabricated because no target Stage-A freeze exists. The
generic production architecture/identity checks ran on a real NVIDIA GeForce
RTX 5070 Ti in CUDA/BF16 and passed 2/2 tests, including zero-initialized V2-A
projection identity and E=0/mixed degree-zero identity.

## 6. Validation and protected artifacts

- `python -m compileall -q .`: PASS.
- `pytest -q`: **467 passed**.
- `git diff --check`: PASS.
- focused cross-dataset preflight tests: **15 passed**.
- affected parser/Stage-B regression set: **59 passed**.
- real CUDA/BF16 targeted tests: **2 passed**.

The protected ETH artifacts remain unchanged:

- Stage A: `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`;
- Stage-B V1: `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a`;
- Stage-B V2-A: `e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73`;
- DependencyCorrector source: `0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522`;
- component projection source: `355de2a9ace3265efe1d5fc1ed4b5a6c054a621fc452a7ad95d70b6a2c3b5630`.

## 7. Blockers and execution recommendation

The immediate prerequisites are:

1. locate or reproduce the original-protocol HOTEL, ZARA1, and ZARA2 GDTS
   source checkpoints and freeze their hashes;
2. build and audit canonical synchronized source caches for all targets;
3. build each target's JDV2 Stage-A cache from that same target's checkpoint;
4. rerun target-specific Stage-A CUDA/BF16, exact-sampler, 20/20 coverage,
   finite-output, and provenance smokes;
5. only then mark one target `TARGET_READY_FOR_STAGE_A`.

UNIV is the recommended first target **after its two caches are built and
audited**, solely because it is the only target with an authenticated,
architecture-compatible original GDTS checkpoint and therefore has the
lowest remaining engineering uncertainty. This is not based on expected
metrics. The planned one-at-a-time order is UNIV, then HOTEL, ZARA1, ZARA2,
with a freeze/hash/report review between Stage A, V2-A, and each next target.

Current next state: none. Training remains unauthorized.
