# EBJD clean v1 implementation receipt

## Identity and isolation

- Date: 2026-10-08 (Asia/Shanghai)
- Branch: `research/ebjd-clean-v1`
- Base: `092ed91f1f4117f50aa835ec8e47cd2c9ce74a98`
- Frozen design SHA256: `9296937cc81f0e64a94b7c774a76e04d489fffe76e8ee5b64ca9bda3ff3af1af`
- Version-controlled change boundary: `ebjd_clean/` only
- Legacy Stage-A/JDV2 runtime imports: none

## Implemented contract

The package contains an exact endpoint-plus-whitened-Brownian-bridge
representation, trainable 23-convolution U-Net, history GRU and two history
social blocks, 72-token history/map memory, six full factorized future blocks,
four independent v-heads, differentiable clean-future geometry/map injection,
and shared deterministic DDIM training/inference sampling. Padding queries are
zeroed, social attention is scene/world-local, and no future target is an
inference condition.

Losses include balanced final/coarse goal and bridge v-losses, history-defined
relative-motion supervision, future-map BCE, and S4 soft marginal/joint free
rollout risk. `ActualStepAdamW` constructs the AdamW decrement including group
learning rates, moments, preconditioning and decoupled decay, projects that
actual decrement against the two marginal halfspaces in FP64, advances state
once, and applies parameters once.

All six preregistered ablations are exposed through one CLI switch. Training
uses synchronized scene/map rotation and reflection, training-fold-only RMS
scales, atomic checkpoints, validation-only selection, and fold-explicit NPZ
inputs.

## Initialization binding

The HOTEL config binds the accepted goal U-Net at epoch 96:

```text
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean/output/hotel/runs/grouped_fresh_goal_official_prior_seed3101_20261005_114637/epoch_096.pt
SHA256 f6a52cf228d733bd684aef043d843ba7d942e40495d5eec354ce1e4e3507ec13
```

Exactly 46 U-Net tensors are translated and loaded with `strict=True`; the hash
is checked before deserialization. The U-Net remains trainable. Other folds
must bind their own accepted fold-matched source and hash.

## Actual parameter count

PyTorch `named_parameters()` reports 3,614,300 trainable scalars:

| component | parameters |
|---|---:|
| U-Net | 613,772 |
| other history/map context | 510,980 |
| six future blocks | 2,380,800 |
| other denoiser components/heads | 108,748 |

The frozen design's hand estimate was 3,612,764. The actual instantiated graph
is 1,536 parameters larger; the U-Net exactly matches its 613,772 estimate.
The PyTorch count is authoritative and no block/head was removed to force the
estimate.

## Verified environment and results

- Python 3.10.21
- PyTorch 2.8.0+cu128
- NumPy 1.26.4
- PyYAML 6.0.3
- pytest 9.1.1
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB

`PYTHONPATH=. python -m pytest -q` from `ebjd_clean/` passed 11 tests. These
tests cover representation round-trip/literal endpoint, exact t=1 and near-zero
coefficients, N=1/N=2/multiple scenes/padding, finite output, agent permutation,
scene isolation, required module gradients, differentiable sampler, batched vs
world-chunked equivalence, hand-computed metrics/legacy packing, source hash
loading, all ablation constructors, and the actual-step projection including
duplicate/opposite constraints and a finite-difference protected risk check.

Real CUDA synthetic resource checks used 256×256 map crops:

| check | shape/budget | elapsed | peak allocated | peak reserved | outcome |
|---|---|---:|---:|---:|---|
| differentiable rollout + backward | B1, N2, S4, 20 steps, BF16 | 4.425 s | 198,537,728 B | 255,852,544 B | finite; 296 parameter tensors with gradients |
| inference | B1, N2, P20, 20 steps, BF16 | 0.785 s | 117,461,504 B | 155,189,248 B | finite |

The standalone training CLI also completed its CPU synthetic smoke, including
scale fitting, a real AdamW update, validation, and atomic last/best checkpoint
writes. Synthetic losses/metrics are deliberately not reported as research
results.

## Scope and unresolved empirical work

No formal ETH/UCY fold training or test evaluation has been run in this
implementation receipt. No improvement in minADE, minFDE, JADE or JFDE is
claimed. The measured GPU profile is a small N=2 capacity check, not a worst-case
scene guarantee. Formal work still requires exporting audited complete-scene
fold NPZ files, binding one accepted U-Net per fold, at least three training
seeds, validation-only checkpoint selection, and the five fixed P20 inference
seeds. Negative ablation results must be retained.

The metric adapter assumes input coordinates are already metres and never
applies a second conversion. The input contract intentionally refuses legacy
pickle semantics rather than guessing coordinate transforms or scene packing.
