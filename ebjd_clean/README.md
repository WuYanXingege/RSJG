# Endpoint–Bridge Joint Diffusion (EBJD)

This directory is the isolated EBJD-clean-v1 implementation. Runtime training
and evaluation do not import the legacy GDTS/JDV2 model, trainer, sampler or
metrics. External GDTS artefacts are read only as explicitly hashed data,
initialization and comparator inputs.

All commands below run from `ebjd_clean/`.

## Environment and tests

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
PYTHONPATH=. pytest -q
```

The HOTEL config distinguishes two external roles:

- goal-only epoch 96 initializes EBJD's trainable U-Net;
- official GDTS epoch 110 and its synchronized P20 result define the validation
  marginal thresholds.

All files are bound by SHA256. Formal startup recomputes the validation
manifest, comparator checkpoint and comparator result hashes and refuses any
mismatch. These two checkpoints are not claimed to be the same parent.

## Scene export and data contract

Create the complete synchronized HOTEL-fold scene manifests directly from the
read-only ETH/UCY text, homography and semantic-map files:

```bash
PYTHONPATH=. python -m ebjd.export_ethucy \
  --source-root ../GDTS_official_297d508_HOTEL/data/eth5 \
  --fold hotel --output-root data/hotel_official --shard-scenes 16
```

Each NPZ shard contains complete scenes, never independent-agent fragments:

- `observed [N,8,2]` and `future [N,12,2]` in metres;
- `semantic_maps [N,6,256,256]`, categorical one-hot uint8;
- scene/source/window identity, 20 source frames/timestamps and all agent IDs.

The 32 m crop is centred only on the final observed position. The loader checks
shard hashes, metadata, shapes, finite values and synchronized identity. It
loads one shard at a time and pads agents only in the collator; no agent is
truncated.

Reproduce the source parity and load-resource audits with:

```bash
PYTHONPATH=. python scripts/audit_export_parity.py \
  --manifest data/hotel_official/validation/validation_manifest.json \
  --legacy-batches ../GDTS_official_297d508_HOTEL/output/hotel/data_batches/valid_batches \
  --grouped-batches ../RSJG_JDV2_clean/outputs/joint_dependency_v2/hotel_a0_a1_5b6e4e4/cache/source_batches/eth5/hotel/data_batches_jdv2_v2/valid_batches \
  --homography ../GDTS_official_297d508_HOTEL/data/eth5/hotel/H.txt \
  --semantic-map ../GDTS_official_297d508_HOTEL/data/eth5/hotel/pred_mask.png \
  --source-scene hotel --output outputs/audits/hotel_official_export_parity.json

PYTHONPATH=. python scripts/audit_manifest_resources.py \
  --manifest data/hotel_official/train/train_manifest.json \
  --output outputs/audits/hotel_official_train_load_resources.json
```

## Training and evaluation

Synthetic wiring smoke (not a benchmark):

```bash
PYTHONPATH=. python -m ebjd.train \
  --config configs/endpoint_bridge_joint.yaml --device cpu --smoke
```

Formal HOTEL training after verifying the receipt and GPU availability:

```bash
PYTHONPATH=. python -m ebjd.train \
  --config configs/endpoint_bridge_joint.yaml \
  --device cuda --run-id hotel_seed3101_main
```

Runs are isolated as `outputs/<fold>/<seed>/<ablation>/<run_id>/`. Existing
directories are refused; only explicit `--resume CHECKPOINT` resumes an epoch
boundary, with run identity, selection history and RNG state restored.

Evaluation never selects a checkpoint on test:

```bash
PYTHONPATH=. python -m ebjd.evaluate \
  --config configs/endpoint_bridge_joint.yaml \
  --checkpoint outputs/hotel/3101/none/hotel_seed3101_main/best.pt \
  --split test --device cuda \
  --output outputs/hotel/3101/none/hotel_seed3101_main/test.json
```

The packaged HOTEL validation and test manifests are mirrors. Their results are
internal development evidence, not an independent final test. A publishable
final claim still requires a genuinely independent standard-protocol test.

Choose one preregistered ablation with `--ablation`: `future_social_off`,
`noisy_geometry`, `cartesian_velocity`, `geometry_loss_off`,
`rollout_loss_off`, or `actual_step_constraint_off`.

See `docs/DESIGN.md`, `docs/IMPLEMENTATION_RECEIPT.md` and
`docs/CORRECTION_RECEIPT_2026-10-08.md` for the frozen design and evidence.
