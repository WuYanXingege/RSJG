# Experiment registry

## Formal identity

The default paper model is the original FULL architecture with Raw output.
All checkpoints listed below strict-load the same 307-tensor,
3,612,764-parameter model state. Full hashes are authoritative.

| Fold / role | Epoch | Checkpoint SHA256 | Status |
|---|---:|---|---|
| HOTEL FULL | 45 | `425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1` | Known reference; source `3b13adb8…` |
| UNIV selected FULL | 73 | `f526506055f346513b7c57994f9935d40e624162d74e110104073312cfc4bbe9` | Formal identity retained |
| UNIV Control FULL candidate | 81 | `e9fd76be585fef0d9cf6ab82422e5a5775392030c59283054a5247a6f47eafdd` | Performance reference only; source `b338611b…` |

UNIV epoch81 is architecturally FULL, but epochs 74–81 were a 2,040-update
`rollout_mechanism_off` continuation using the
`small_group_uncheckpoint_v1` execution profile. It therefore must not be
described as the same training protocol as HOTEL, and it does not overwrite
epoch73's historical selection identity.

## Existing five-inference-seed results

| Fold / checkpoint | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| HOTEL epoch45 FULL | 0.138007 ± 0.001258 | 0.202037 ± 0.002553 | 0.168566 ± 0.001605 | 0.276314 ± 0.003906 |
| UNIV epoch73 selected FULL | 0.274416 ± 0.000742 | 0.516526 ± 0.001583 | 0.532303 ± 0.000711 | 1.134360 ± 0.002454 |
| UNIV epoch81 Control FULL candidate | 0.269949 ± 0.000780 | 0.507857 ± 0.001533 | 0.526733 ± 0.000308 | 1.123216 ± 0.001315 |

These are five stochastic inference repeats (2035–2039) of one training seed
3101, not five independently trained models. Validation/test windows are
mirrored in the supplied HOTEL and UNIV packages, so these are internal
held-out-scene results rather than independent final-test generalization.

The HOTEL original contract remains **FAIL**: compared with matched GDTS,
joint metrics improved but minADE/minFDE regressed. Moving the code does not
change that conclusion. Exact source values are in
`results/FORMAL_RESULTS.json`.

## Historical variants and negative results

| Variant | Fixed source | Registry decision |
|---|---|---|
| Future Social SELF_ONLY | `b848a76b13419e2eb90de137dace98fd70c559df` | Frozen inference intervention; not the default model |
| UNIV Future Social HALF | `ff0aa9a8b36018ac56ae66b79a4de779c9135697` | `NO_CLEAR_GAIN`; not the default model |
| HOTEL Future Social HALF | `cf59f0f14efd19bfcfe279e4a99291e54a63b24a` | Cross-fold `NOT_SUPPORTED`, HOTEL contract `FAIL` |
| ETH native-goal conditioning | `5c9df1d73026be26dd108a179211aa12684392af` | Historical variant; excluded from the unified FULL table |
| NO_TOKEN Fixed-anchor path-loss pilot | `d689e3a13757451c4447cbef48faf64917c7fe17` | Failed historical path; excluded from FULL |

Native-goal candidates, `JointGoalAssembler`, anchor residuals, Fixed-anchor
outputs, HALF/SELF_ONLY, motion-summary/map-token V2 changes, grouped gates and
the extra path loss are not reachable from the default configuration or
evaluation CLI. Reproduction pointers live under `experiments/ablations/`;
their implementation is intentionally not copied into the paper core.

## Fold readiness

- **HOTEL — READY:** stable training/validation/test manifests, goal-U-Net,
  matched comparator assets and epoch45 FULL checkpoint are bound by hash.
- **UNIV — READY WITH HISTORY NOTE:** stable manifests and FULL epoch73/epoch81
  checkpoints are bound. Exact reproduction of the reported five-seed rows
  uses the mirrored validation manifest SHA256 `46c52540…`.
- **ETH — NOT READY FOR UNIFIED FULL:** existing EBJD results use historical
  native-goal/anchor/Fixed-anchor variants. No result is promoted.
- **ZARA1/ZARA2 — NOT PREPARED:** no paper-workspace assets or verified FULL
  run configuration are asserted.
