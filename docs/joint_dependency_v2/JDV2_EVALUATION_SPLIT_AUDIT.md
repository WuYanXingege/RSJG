# JDV2 evaluation split integrity audit

## Status

**`VALID_TEST_FULLY_IDENTICAL`**

The canonical ETH JDV2 loader configuration does not provide a clean
model-selection/test separation. The 139 `valid` windows and the 139 `test`
windows are identical in order, cache identity, agent/frame identity, and
trajectory content. Consequently, the existing ETH results are valid
method-development/validation evidence, but they are not clean held-out-test
results.

Machine-readable evidence:

```text
outputs/joint_dependency_v2/eth/joint_dependency_v2/
stage_b_v2a/adoption_protocol/split_audit.json
```

SHA256:
`7fa942ef516b9805e8205345ae4679302891dce70e01671cc5ca08f5c80d1570`.

## 1. Provenance and loader behavior

The audit started from source commit
`817cc7551bd7f3422422230821acf8b3dd5cdc6c` and loaded the saved canonical
V2-A configuration:

```text
outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/
jdv2_stage_b_v2a_eth_seed2035/config.yaml
```

Its relevant fields are:

```yaml
model_selection_split: dataset_valid
final_test_split: heldout_test
internal_validation_strategy: source_block
```

`src/data_loader.py` uses the physical `valid` and `test` caches directly
when `model_selection_split=dataset_valid`. The deterministic `source_block`
partition is only activated when `model_selection_split=internal_train`; it
was therefore inactive in the historical Stage-A and Stage-B runs.

The audit instantiated `get_dataloader(args, split)` for all three canonical
splits. It then read the exact cache members selected by those loaders. It did
not infer split membership from filenames.

## 2. Fingerprint definition

Each window has two independent stable fingerprints:

- exact ID SHA256 over canonical raw-source path, scene, starting frames,
  all frame IDs, all agent IDs, synchronized-window flag, and cache-format
  version;
- exact content SHA256 over the ID payload plus dtype, shape, and C-order
  bytes for `frame_ids`, `seq_list`, `abs_pixel_coord`, `scene_index`, and
  `scene_ptr`.

The machine artifact additionally records every cache ID/path, cache-file
SHA256, scene, frame IDs, agent IDs, canonical raw-source path, and source-file
SHA256. No heuristic spatial/temporal matching is used.

## 3. Split inventory

| Split | Windows | Scenes and counts | Raw sources |
|---|---:|---|---:|
| train | 4,110 | hotel 445; univ 1,267; zara1 705; zara2 1,693 | 7 |
| valid | 139 | eth 139 | 1 |
| test | 139 | eth 139 | 1 |

The valid and test cache records both identify their source as:

```text
data/eth5/eth/val/biwi_eth.txt
```

with SHA256
`cd75b1008b82b7f442b2e03967b0f4aac36da2e73d440df1f605bb197d23fb33`.
The separately present raw file `data/eth5/eth/test/biwi_eth.txt` has the same
byte size and the same SHA256. Thus the duplication exists both at raw-file
level and in the materialized cache.

## 4. Pairwise overlap

Percentages are reported relative to each side of the pair.

| Pair | Exact ID overlap | ID overlap % | Exact content overlap | Content overlap % |
|---|---:|---:|---:|---:|
| train / valid | 0 | 0% / 0% | 0 | 0% / 0% |
| train / test | 0 | 0% / 0% | 0 | 0% / 0% |
| valid / test | **139** | **100% / 100%** | **139** | **100% / 100%** |

All splits have zero internal duplicate ID fingerprints and zero internal
duplicate content fingerprints. The valid and test fingerprint sequences are
also ordered-identical, not merely equal as unordered sets.

Classification: **fully identical valid/test splits**.

## 5. Historical Stage-A protocol

The strict-no-z Stage-A run recorded:

- `model_selection_split=dataset_valid`;
- 139 validation windows and 139 test windows;
- epoch 13 selected by validation JFDE (`0.7434224215`, seed 2035);
- five-seed reference metrics reported on the ETH validation split.

The protected epoch-13 checkpoint metadata confirms the selection epoch and
JFDE criterion. `JDV2_STAGE_A_FREEZE.md` explicitly labels the published
reference table as ETH validation, 139 windows. Because those 139 validation
windows are byte-identical to the declared held-out test cache, epoch 13 was
selected on the same examples that would be used by `heldout_test`. The
five-seed Stage-A reference evaluations reuse those same windows.

This does not invalidate the Stage-A mechanism-development comparisons, but
the checkpoint and its metrics cannot be relabeled as a clean held-out-test
benchmark.

## 6. Historical Stage-B protocol

Stage-B V1 and V2-A inherited the same direct `dataset_valid` loader. V2-A
epoch 5 was selected by validation JADE with a JFDE tie-break, again on the
same 139 ETH windows as the declared test cache. The paired A/B/C/D
comparisons remain internally comparable method-development evidence because
they use the same worlds, seeds, noise, metrics, and windows. They do not
establish paper-level held-out adoption.

## 7. Integrity decision

The split-integrity gate fails Case B. No Stage-B adoption freeze, freeze
manifest, or freeze-contract test is created. Existing checkpoint bytes and
reports are retained unchanged and must be described as ETH
method-development/validation evidence.

Clean benchmark reproduction requires a new protocol with model selection
strictly inside the training sources and exactly one untouched ETH test
evaluation after selection. Because both the Stage-A parent and Stage-B V2-A
checkpoint were selected on the mirrored ETH validation/test windows, both
stages require clean-protocol reproduction; merely reevaluating the current
checkpoints cannot remove selection leakage.
