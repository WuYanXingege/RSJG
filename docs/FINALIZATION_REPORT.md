# EBJD paper-workspace finalization

## Decision

The retained method is **Endpoint–Bridge Joint Diffusion (EBJD)**, Chinese
**终点—桥联合扩散**, using the FULL architecture and Raw decoding. The current
training profile is 192/4/768; the original reference checkpoints below use
the separately supported legacy 128/4/512 profile.
The paper-facing modules are MSCE, EBD, GCJD, CTF and FJAB. Registered module
attributes and state-dict paths remain unchanged.

The workspace was branched from the HOTEL result commit `3b13adb8…`, before
native-goal, V2 and Future-Social interpolation entered the core. Historical
JDV2 code, large review archives, old logs and experimental launch scripts were
not copied. HALF/SELF_ONLY and failed migrations are represented only by
fixed commit pointers and small result records.

## Compatibility and validation

- The package imports from this workspace without an external `PYTHONPATH`.
- HOTEL epoch45, UNIV epoch73 and UNIV Control epoch81 each strict-load all
  307 state tensors into the same legacy 3,612,764-parameter FULL model.
- The paper-facing class renames do not change registered state keys.
- Existing tests cover reversible EBD encoding/decoding, FULL forward and
  required gradients, mixed-size padding, permutation/scene isolation,
  deterministic sampling, metrics, optimizer projection and certified
  execution grouping.
- The finalization test additionally exercises 20-world, 20-step Raw sampling.
- No formal checkpoint is updated and no formal training is launched.

Machine-readable receipts are `results/FINALIZATION_VALIDATION.json`,
`results/ASSET_MANIFEST.json` and `results/TEST_RECEIPT.json`.

## Asset state

The stable sibling store `RSJG_EBJD_assets` contains copies of the HOTEL and
UNIV datasets, reference checkpoints, goal-U-Net initialization weights and
small matched-GDTS records. Originals were preserved. The local `assets/`
links are ignored, and all expected hashes are tracked.

HOTEL and UNIV are operational. ETH and ZARA folds remain explicitly
unprepared for unified FULL experiments; no path or metric is fabricated.

## Limitations

Current formal tables contain one training seed per checkpoint. Five
inference seeds quantify sampling variability only. The distributed
validation/test manifests mirror the same held-out-scene windows. HOTEL's
original no-marginal-regression performance contract remains failed. UNIV
epoch81 has a different continuation objective identity and remains a
reference candidate, not a retroactive replacement for epoch73. The cleanup
does not constitute a new structure ablation, performance improvement or
statistical significance claim.
