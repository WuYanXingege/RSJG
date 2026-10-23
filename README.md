# Endpoint–Bridge Joint Diffusion (EBJD)

**Endpoint–Bridge Joint Diffusion for Joint Pedestrian Trajectory Prediction**

中文：**终点—桥联合扩散**

This is the only paper-development workspace for EBJD. The default path is
the **FULL** architecture with Raw decoding. New formal training uses the
192-wide, four-head, FFN-768 profile. Historical 128/4/512 checkpoints remain
registered references and are loaded only with their legacy profile.
It does not import code from another RSJG worktree and does not include
native-goal, Fixed-anchor, SELF_ONLY, HALF, V2 token/anchor, or extra path-loss
logic in the default model.

## Layout

- `ebjd/`: model, data, training, deterministic DDIM sampling and metrics.
- `configs/architecture.yaml`: the one authoritative network definition.
- `configs/training.yaml`: default HOTEL-certified training contract.
- `configs/folds/`: fold-specific data and asset bindings.
- `docs/METHOD.md`: equations, dimensions, data flow and module map.
- `docs/EXPERIMENT_REGISTRY.md`: checkpoints, results and negative evidence.
- `results/`: small tracked result and validation receipts only.
- `assets/`: tracked instructions plus ignored links to the persistent asset store.
- `experiments/ablations/`: history index only; not imported by FULL.

## Environment and assets

The verified environment is Python 3.10.21, PyTorch 2.8.0+cu128, CUDA 12.8,
NumPy 1.26.4, PyYAML 6.0.3, Pillow 12.3.0 and pytest 9.1.1.

```bash
cd /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_Paper
conda env create -f environment.yml
conda activate ebjd-paper
python -m pip install -e .
./scripts/link_assets.sh
python scripts/verify_workspace.py
```

`EBJD_ASSET_ROOT=/stable/path/RSJG_EBJD_assets` overrides the default sibling
asset store. Data, checkpoints, logs, prediction banks and generated outputs
are ignored by Git. Expected hashes are in `results/ASSET_MANIFEST.json`.

## Resolve the configuration

The runnable JSON is deterministically materialized from the unique
architecture, default training contract and one fold binding:

```bash
python scripts/resolve_config.py --fold hotel
python scripts/resolve_config.py --fold univ
python scripts/resolve_config.py --fold zara1
```

Generated files appear under `configs/resolved/` and record the current source
commit. Run commands from this repository root; no external `PYTHONPATH` is
needed or supported.

## Train FULL

No training was launched during paper-workspace finalization. The normal
entry point is:

```bash
python -m ebjd.train \
  --config configs/resolved/hotel.json \
  --ablation none --device cuda --run-id hotel_full_seed3101
```

The UNIV epoch81 Control checkpoint is only a performance-reference candidate:
its epochs 74–81 continuation used `rollout_mechanism_off`. It is not evidence
that the UNIV history used the default HOTEL training contract. Fresh paper
training should use `configs/training.yaml`; historical continuation identity
is recorded separately in `configs/reference_runs/univ_control_epoch81.yaml`.

## Evaluate Raw P20

```bash
python -m ebjd.evaluate \
  --config configs/resolved/hotel.json \
  --checkpoint assets/checkpoints/hotel/epoch_045_full.pt \
  --checkpoint-sha256 425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1 \
  --split test --device cuda --output outputs/eval/hotel_epoch45.json

python -m ebjd.evaluate \
  --config configs/resolved/univ.json \
  --checkpoint assets/checkpoints/univ/epoch_073_selected_formal.pt \
  --checkpoint-sha256 f526506055f346513b7c57994f9935d40e624162d74e110104073312cfc4bbe9 \
  --split validation --device cuda --output outputs/eval/univ_epoch73.json
```

The evaluator always uses FULL, Raw decoding, 20 worlds, 20 deterministic
DDIM steps and the five configured inference seeds. It requires the expected
checkpoint hash and strict-loads every state entry. Summarize generated files
with:

```bash
python scripts/summarize_results.py outputs/eval/*.json
```

## Scope

HOTEL and UNIV assets are locally ready. ETH is deliberately **not ready** in
this paper workspace because existing ETH EBJD results used historical
native-goal/anchor variants and cannot be placed in the unified FULL table.
ZARA1 synchronized manifests are prepared (705 held-out windows, 2,356 agent
occurrences, mean 3.342); its goal U-Net and matched-GDTS validation assets
are hash-bound. The fresh 192/4/768 formal run is separate from the preserved
three-epoch 128/4/512 attempt.
ZARA2 remains unprepared. Missing fold assets are reported, not guessed.

See `docs/FINALIZATION_REPORT.md` for the validation receipt and limitations.

## Training-performance status

The 2026-10-22 rollout-backward audit accepted the bounded
`small_group_uncheckpoint_v1` execution profile. Physical groups with at most
32 padded agent slots retain the complete 20-step differentiable DDIM graph
for the three required VJPs; denser groups keep DDIM-step checkpointing. A
paired 16-update HOTEL window was value-identical and measured 1.465x mixed
compute / 1.424x end-to-end speedup. A dense real UNIV update exercised the
checkpointed fallback. See
`docs/performance/20261022_rollout_backward_v2/REPORT_ZH.md` for the exact
scope, memory tradeoff and receipts. No model, objective, optimizer, sampling,
validation or checkpoint schema changed.

For rollout-OFF ZARA1 training, `prefetch_runtime_v1` adds four ordered loader
workers (prefetch factor two) and removes unused constraint buffers/scalar
synchronizations. A paired real-data 192/4/768 gate was update-identical and
measured 1.720x end-to-end speedup; see
`docs/performance/20261023_zara1_wide_prefetch_v1/REPORT_ZH.md`.
