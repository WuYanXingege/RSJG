# Endpoint–Bridge Joint Diffusion (EBJD)

This directory is a standalone implementation of the frozen EBJD-clean-v1
design. It does not import the legacy Stage-A/JDV2 trainer, sampler, model or
metrics. All trainable components are active from epoch 1.

## Environment

From the repository/worktree root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e 'ebjd_clean[test]'
pytest ebjd_clean/tests
```

The accepted HOTEL U-Net source is bound by absolute path and SHA256 in
`configs/endpoint_bridge_joint.yaml`. Change both fields together for another
fold, and retain the source receipt. Loading aborts on a hash mismatch.

## Input contract

Each NPZ split contains `observed`, `future`, `semantic_maps`, and optionally
`scene_ids`. Dense or NumPy object arrays are accepted. One entry is a complete
synchronized scene:

- `observed`: `[N,8,2]` metres;
- `future`: `[N,12,2]` metres;
- `semantic_maps`: `[N,6,H,W]` semantic crops or `[N,14,H,W]` complete map input;
- no scene agent may be truncated. The collator pads agents and supplies a mask.

The current config paths are explicit placeholders for the fold-exported NPZ
files. This implementation does not silently reinterpret legacy pickle batches.

## Run

Synthetic wiring smoke (not a benchmark):

```bash
python -m ebjd.train \
  --config ebjd_clean/configs/endpoint_bridge_joint.yaml \
  --device cpu --smoke
```

Formal training after binding the three fold NPZ files:

```bash
python -m ebjd.train \
  --config ebjd_clean/configs/endpoint_bridge_joint.yaml \
  --device cuda
```

Evaluation never selects a checkpoint on the test split:

```bash
python -m ebjd.evaluate \
  --config ebjd_clean/configs/endpoint_bridge_joint.yaml \
  --checkpoint ebjd_clean/outputs/endpoint_bridge_joint_seed3101/best.pt \
  --split test --device cuda \
  --output ebjd_clean/outputs/endpoint_bridge_joint_seed3101/test.json
```

Choose one preregistered ablation with `--ablation`: `future_social_off`,
`noisy_geometry`, `cartesian_velocity`, `geometry_loss_off`,
`rollout_loss_off`, or `actual_step_constraint_off`.

Profile the exact differentiable sampler without reading a dataset:

```bash
python ebjd_clean/scripts/profile_runtime.py \
  --device cuda --pixels 256 --agents 2 --worlds 4 --steps 20 --backward
```

Outputs are written only below `ebjd_clean/outputs/` and ignored by Git. See
`docs/DESIGN.md` for the full frozen specification and
`docs/IMPLEMENTATION_RECEIPT.md` for verified scope and limitations.
