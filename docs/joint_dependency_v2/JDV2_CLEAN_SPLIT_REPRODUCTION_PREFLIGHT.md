# JDV2 Clean-Split Reproduction Preflight

## Outcome

**Primary state: `CLEAN_SPLIT_REPRODUCTION_READY`**

**Next state only: `eligible_for_clean_stage_a_reproduction`**

No training, optimizer step, checkpoint creation, final-test metric, Stage B,
or method modification was performed in this task.

## Provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Preflight base commit | `6765e1adefb48fd20689501954f124aadcc438da` |
| Manifest stable hash | `0a6deb6ff9026b1784d728ef70f6e7dd1020c73819d4f71557dd7578772cc2b4` |
| Manifest file SHA256 | `c8136967954bac5477b422da5d7f5b05fa979856a1318c4b323836ffff9f57ad` |
| Full source-audit SHA256 | `7fa942ef516b9805e8205345ae4679302891dce70e01671cc5ca08f5c80d1570` |

The full content-level source audit authenticated all 4,110 physical train and
139 physical test records. The clean preflight then reconstructed the source
boundary twice, required identical manifests, and verified all 320 JDV2
internal-valid alignments.

## Split audit

| Logical split | Physical split/index | Windows | Source |
|---|---|---:|---|
| train | train/0–3789 | 3790 | six complete contiguous files |
| internal valid | train/3790–4109 | 320 | `data/eth5/eth/train/uni_examples.txt` |
| final test | test/0–138 | 139 | `data/eth5/eth/val/biwi_eth.txt` |

All six pairwise overlap counts (three split pairs × ID/content fingerprint)
are zero. Each role has zero internal duplicate ID fingerprints and zero
internal duplicate content fingerprints. Every source occurs in exactly one
contiguous block. Repeated construction was byte-for-byte logically identical.

## Cache and permission audit

- 320/320 internal-valid members map to original train indices 3790–4109.
- Each member reads `train/<original-index>.pt`, with exact `cache_id`, frame
  tensor, K=21 candidate bank, and valid edge tensor.
- Logical internal valid uses physical train addressing but contains no
  `jdv2_teacher_cache` and has augmentation disabled.
- Logical train still loads its matching teacher sidecar.
- No cache was moved, copied, renumbered, regenerated, or edited.

An explicit no-optimizer dry-run confirmed train teacher `True`, valid teacher
`False`, valid augmentation `False`, and pre-lock test-loader construction
raising `clean final test is locked`.

## Final-test isolation

Clean Stage-A/Stage-B training creates only train and internal-valid loaders;
the test loader is structurally absent. The final test requires an authorized,
hash-locked, two-checkpoint/two-config artifact and fixed seeds 2035–2039.
Lock validation happens before test dataset construction and immediately before
metrics. The combined `train_test` phase is forbidden.

## Canonical configurations

- `configs/joint_dependency_v2/jdv2_stage_a_clean_eth.yaml`
- `configs/joint_dependency_v2/jdv2_stage_b_v2a_clean_eth.yaml`

Stage A recovers historical optimizer, scheduler, LR, BF16, seed, validation,
early-stop, loss, graph, relation, energy, candidate, and sampling settings;
its production sampler is the subsequently frozen exact persistent policy.
Stage-B V2-A differs from its approved development config only in clean split,
output, final-test guard, and expected clean parent provenance. Its parent hash
is intentionally null until Phase-A selection is frozen.

## Protected-artifact verification

| Artifact | Verified SHA256 |
|---|---|
| Stage-A development best | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Stage-B V1 best | `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a` |
| Stage-B V2-A development best | `e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73` |
| `dependency_corrector.py` | `0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522` |

The DependencyCorrector remains exactly 30,851 parameters.

## Verification

The focused clean protocol suite passed **11/11** tests and covers exact membership, overlap,
determinism, source isolation, all 320 cache addresses, logical permissions,
historical loader behavior, phase-scoped loader construction, final-lock
authentication/tamper rejection, parser guards, scientific config invariants,
and the 30,851-parameter contract. Existing resume-state, Stage-A freeze,
Stage-B V2-A, mixed-precision identity, and CUDA/BF16 contracts are retained
and are part of the full-suite gate.

Final validation commands:

```bash
python -m compileall -q .
pytest -q
git diff --check
```

Preflight validation completed with **446 passed, 0 skipped** for the full
suite and **8 passed** in the explicit Stage-A freeze plus CUDA/BF16 identity
subset. `compileall` and `git diff --check` passed. The pushed commit is
recorded in the task handoff.

## Stop

Preflight is complete. The only permitted next action after review is a fresh
clean Stage-A reproduction. This report does not authorize training by itself,
does not authorize final-test access, and does not authorize Stage B.
