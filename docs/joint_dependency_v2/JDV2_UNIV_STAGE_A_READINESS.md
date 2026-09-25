# JDV2 UNIV Stage-A Readiness

## Final state

`UNIV_READY_FOR_STAGE_A`

Next state: `eligible_for_univ_stage_a_reproduction`.

This work built and audited the two missing UNIV caches and ran a real
CUDA/BF16 no-optimizer Stage-A smoke. It did **not** train Stage A or Stage B,
run benchmark evaluation, change scientific hyperparameters, or populate a
UNIV Stage-B parent checkpoint.

## Provenance

- Branch: `research/joint-dependency-v2-clean`
- Cache-build source commit:
  `db47b913820371cca25876d08c624709830ae1e8`
- Target/protocol: `eth5/univ`, original GDTS mirrored-valid/test protocol
- Authenticated GDTS checkpoint:
  `../GDTS/output/univ/saved_models/best_model.pt`
- Checkpoint epoch: `100`
- Checkpoint SHA256:
  `ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e`
- GDTS protocol config SHA256:
  `07452f9693ea2f19c802be73b0c0a78ec8f1a881deabb7a5994cab9b41cc1c09`

All eight raw-file hashes were rechecked against the cross-dataset preflight.
The UNIV validation and test files remain byte-identical: `students001.txt`
and `students003.txt` have the same hashes in both splits. This is a faithful
reproduction of the repository's original mirrored-valid/test protocol, not
an independent held-out-test protocol.

## Canonical synchronized source cache

The cache was rebuilt from raw data with the current JDV2 preprocessing path;
the legacy `data_batches_joint_v2` directory was neither relabeled nor used as
the authoritative source.

- Root:
  `outputs/joint_dependency_v2/cache/source_batches/eth5/univ/data_batches_jdv2_v2`
- Format: synchronized-window batch format version 2
- Manifest SHA256:
  `e1f7ae3897762a97dd8f0939e0bb8bd18e3646bab83d0a853ace4b40ea5df9aa`
- Stable manifest hash:
  `12b9909dcf8a4e346cb1bcd179c58faef7a2688bf3c8d3d0c3956909486e8943`

| Split | Windows | Agent N min/mean/max | Source composition |
|---|---:|---:|---|
| train | 3,302 | 1 / 3.918837 / 14 | ETH 139; HOTEL 445; ZARA1 705; ZARA2 998+695; UNIV examples 320 |
| valid | 947 | 3 / 25.695882 / 57 | students001 425; students003 522 |
| test | 947 | 3 / 25.695882 / 57 | students001 425; students003 522 |

The full read-only scan passed deterministic contiguous naming, finite tensor,
frame synchronization, `scene_index`/`scene_ptr`, source provenance, and
no-cross-scene-mixing checks. Validation and test were materialized as 947/947
ordered hard-link-identical records, matching the raw protocol while avoiding
duplicate storage.

## Canonical UNIV JDV2 cache

- Root: `outputs/joint_dependency_v2/cache/univ_full_stage_a`
- Schema: `jdv2-cache-v1`
- Manifest file SHA256:
  `a4252f64a9733532fa1f443998532dc8e60974c960ba7e05018b16b01906b69f`
- Stable manifest hash:
  `7d4983630a2665f6856afeba7265a822e5bd87cc3f1c856485d644ca6a718008`
- Dataset hash:
  `8ae93ab50f597ada6888c1333f7418fa202906b83581720eec73288b50ef9d2d`
- Split hashes:
  - train: `1a282deb8b99a5c54c32814aa5350ea55f87eeac484bef2260168e1bddc713aa`
  - valid: `5d992fd509bee8a1c611d77b81beb898f86b76d5b57d37d0348772b870b103ec`
  - test: `3017ae304a0ded8f9d4f22d7f6a80e46926d884958929bea8016c9f9d7f58878`

The frozen cache contract is `K=21`, candidate order
`generate_goal_candidates-v1`, `dt=0.4`, position `world_m`, velocity
`world_m_per_s`, and parameter-free `radius_ttc` graph with radius 6 m,
TTC 8 s, and no adaptive graph. The recorded goal-checkpoint hash is exactly
the authenticated UNIV GDTS hash above.

| Split | Records | Teacher sidecars | E=0 | Edge E min/mean/max |
|---|---:|---:|---:|---:|
| train | 3,302 | 3,302 | 677 | 0 / 8.673228 / 88 |
| valid | 947 | 947 | 0 | 2 / 244.442450 / 909 |
| test | 947 | 947 | 0 | 2 / 244.442450 / 909 |

Every record was scanned. Counts exactly match the source cache; cache IDs and
indices align; map/world candidates are `[N,21,2]`; priors are `[N,21]`;
edges are canonical and stay within scenes; values are finite; and source
metadata hashes align. Deployment records contain neither future-supervision
fields nor forbidden learned-cache tensors. Future pair descriptors exist only
in separate one-to-one teacher sidecars.

## Frozen config provenance

Only the two previously unresolved cache-provenance fields were populated in
the UNIV Stage-A and Stage-B V2-A configs:

```yaml
jdv2_cache_source_commit: db47b913820371cca25876d08c624709830ae1e8
jdv2_cache_manifest_hash: 7d4983630a2665f6856afeba7265a822e5bd87cc3f1c856485d644ca6a718008
```

Resulting config hashes are:

- Stage A: `1a48f32b52349e84d4673437668e4a657c664403c9057faab403c9d7e232348d`
- Stage B V2-A:
  `546804a817b957ee84aeffbb7d9d8761d90b933b0a5ec2284e8cfda52a3d18c4`

The Stage-B fields `stage_a_parent_checkpoint_sha256`,
`stage_a_freeze_manifest_sha256`, and `stage_a_freeze_source_commit` remain
`null`. UNIV Stage B therefore continues to fail closed.

## CUDA/BF16 no-optimizer smoke

The exact resolved Stage-A UNIV config was instantiated on an NVIDIA GeForce
RTX 5070 Ti with BF16 autocast and the existing frozen FP32 islands. No
optimizer was constructed and no parameter update occurred.

| Path | N | E | Prediction | Candidate IDs | Goal bank | Relation | Coverage |
|---|---:|---:|---|---|---|---|---|
| E=0 | 1 | 0 | `[20,20,1,2]` | `[1,21]` | `[1,21,2]` | `[0,20,16]` | 20/20 |
| E>0 | 5 | 6 | `[20,20,5,2]` | `[5,21]` | `[5,21,2]` | `[6,20,16]` | 20/20 per agent |

Both paths were finite and free of cross-scene edges. `strict_no_z` and
`exact_lexicographic_persistent_tie` executed successfully. The source cache,
JDV2 manifest, and UNIV checkpoint all authenticated. Hashes of the frozen
GDTS modules and inactive DependencyCorrector state were identical before and
after the smoke. Peak CUDA memory was 104,801,280 bytes allocated and
136,314,880 bytes reserved.

## Protected ETH artifacts

The protected hashes were reverified unchanged:

- Stage A: `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`
- Stage-B V1: `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a`
- Stage-B V2-A: `e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73`
- DependencyCorrector source:
  `0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522`
- Component projection source:
  `355de2a9ace3265efe1d5fc1ed4b5a6c054a621fc452a7ad95d70b6a2c3b5630`

No ETH checkpoint or cache was used as a UNIV substitute.

## Validation and repository scope

The targeted readiness regression passed (`20 passed`). Repository-wide
validation also passed:

```text
python -m compileall -q .
pytest -q
git diff --check
```

- `compileall`: PASS
- `pytest`: `472 passed, 12 warnings in 211.11s`
- `git diff --check`: PASS

Large cache records and runtime logs remain local and untracked. The committed
machine-readable evidence is
`outputs/joint_dependency_v2/univ/joint_dependency_v2/stage_a_readiness/results.json`;
it contains the aggregate source-cache audit, JDV2-cache audit, smoke result,
config hashes, checkpoint provenance, raw hashes, and protected ETH hashes.

## Readiness decision

All mandatory source-cache, JDV2-cache, provenance, frozen-interface, and
CUDA/BF16 smoke gates passed. UNIV alone is promoted to
`TARGET_READY_FOR_STAGE_A`; HOTEL, ZARA1, and ZARA2 retain their existing
missing-checkpoint blockers. The cross-dataset state is therefore
`CROSS_DATASET_BENCHMARK_PARTIALLY_READY`.

The only authorized next state is
`eligible_for_univ_stage_a_reproduction`. This report does not authorize
Stage-B training or any other dataset.
