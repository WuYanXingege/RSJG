# JDV2 evaluation protocol decision

## Decision

The primary paper-development and reproduction protocol is
**`ORIGINAL_GDTS_PROTOCOL`**. For ETH/UCY this is the original GDTS mirrored
validation/test protocol: validation and test may contain the same held-out
scene/file.

The optional robustness protocol is **`STRICT_CLEAN_HELDOUT_PROTOCOL`**. Its
3790/320/139 train/internal-validation/final-test split, fail-closed access
controls, configurations, manifest, tests, and documentation remain
maintained. It is inactive by default and does not block adoption under the
primary original protocol.

## Source evidence

This decision follows the repository's original data and execution semantics,
not a JDV2-specific exception:

- `src/data_pre_process.py::_eval_source_files_are_identical()` compares the
  ordered validation and test raw sources byte-for-byte.
- `src/data_pre_process.py::_link_identical_eval_batches()` documents that
  ETH/UCY distributes identical validation and test files and hard-links the
  resulting cache batches when they are identical.
- the original `train.sh` invokes `main.py --phase train_test` for ETH, HOTEL,
  UNIV, ZARA1, and ZARA2.
- the content audit found all 139 ETH validation windows ordered-identical to
  all 139 test windows.

Therefore, mirrored validation/test is a property of the original GDTS
ETH/UCY protocol. It was not introduced by Stage A, Stage-B V1, or Stage-B
V2-A.

## Reporting contract

Results from the primary protocol must be described as either:

- "following the original GDTS ETH/UCY evaluation protocol"; or
- "under the original GDTS mirrored validation/test protocol".

They must not be described as an independent held-out test performed after
model selection. The following two statements must remain separate:

1. **Fair relative comparison:** GDTS, Stage A, V1, and V2-A used the same
   windows, seeds, coordinate conversion, sampler/world protocol, and metric
   implementation under the original GDTS protocol.
2. **Strict held-out generalization:** not established by the primary ETH
   protocol.

## Relationship to the historical split audit

`JDV2_STAGE_B_V2A_ADOPTION_REVIEW.md` remains immutable. Its
`ADOPTION_BLOCKED_BY_SPLIT_PROTOCOL` status was correct under the strict clean
held-out adoption criterion. This decision does not relabel that audit as a
pass.

The protocol-scoped outcomes are now:

| Scope | Outcome |
|---|---|
| Original GDTS mirrored validation/test | V2-A is eligible for and receives a separate protocol-scoped freeze |
| Strict clean held-out | Not established; optional future robustness work |

The interrupted clean Stage-A run
`jdv2_stage_a_clean_eth_seed2035` is classified
**`ABORTED_NON_AUTHORITATIVE`**. It completed epoch 1 and was stopped during
epoch 2 (last logged progress 1938/3790). Its checkpoints and metrics are not
scientific evidence and may not be used as a Stage-B parent. Runtime files are
retained for provenance and are not part of the freeze commit.

## Operational contract

- Canonical original-protocol V2-A configuration:
  `configs/joint_dependency_v2/jdv2_stage_b_v2a_eth.yaml`.
- Its `model_selection_split: dataset_valid` is intentionally retained and
  corresponds to the original GDTS protocol.
- Optional clean configuration:
  `configs/joint_dependency_v2/jdv2_stage_b_v2a_clean_eth.yaml`.
- Clean-split source, manifests, tests, and documentation must not be deleted.
- No historical checkpoint, scientific source, loss, sampler, or config is
  changed by this protocol decision.
