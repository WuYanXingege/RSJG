# Frozen self-trained Provider / random six-layer joint pilot

This namespace reuses the restored EBJD six-layer architecture, representation,
loss, coupling, sampler and metrics. It does not weaken the historical config
validator or load a GDTS / old EBJD checkpoint. The new builder is intentionally
separate from `ebjd.config.build_model` and the original runner's main entry.

- `data.py`: purpose/member checked, observed-only export and candidate adapter.
- `model.py`: explicit 192/4/768/6 random downstream; all-group base LR 1e-4.
- `runner.py`: original scene/agent weighting and accumulation, full cursor,
  pending-gradient/RNG recovery, bounded training and selection-dev H_JADE.
- `inference.py`: fixed endpoint; keyed candidate/noise gather; offline shuffle.
- `checks.py`: synthetic gradient/recovery plus real identity/augmentation checks.
- `evaluation.py`: freeze both selected models and all outputs before scoring.

Default training remains off. `scripts/run_bcep_joint.py` requires an explicit
`--execute-authorized-two-runs` flag and rejects a previously claimed execution.
A failed or exhausted run is not automatically restarted. Complete best/last
states remain in the task asset directory; no large artifacts are Git inputs.

CPU exact recovery is tested on a synthetic runner instance including pending
accumulation. CUDA adaptive pooling / grid sampling gradients have no strict
PyTorch deterministic implementation; FP32 `warn_only` preserves original
operators and RNG/state rather than claiming bitwise CUDA training replay.
