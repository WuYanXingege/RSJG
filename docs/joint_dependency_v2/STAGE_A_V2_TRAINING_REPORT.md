# RSJG Joint Dependency V2 — Stage-A Latent Objective V2 Training Report

Date: 2026-09-18  
Dataset/run: ETH, `jdv2_stage_a_v2_full_seed2035`  
Decision: **Stage-A V2 rejected by the latent-health and marginal-preservation gates; Stage B was not started.**

## 1. Executive summary

The approved latent-objective V2 was implemented without changing the Goal
U-Net candidate bank, SocialMotionEncoder, dynamic-relation architecture,
low-rank energy architecture, joint sampler, K/Z/M/rank/P contracts, or
evaluation metrics. The implementation uses one finite Z=4 mixture
marginalization, derives detached responsibilities from that mixture, trains
the future posterior only as an amortized approximation, and removes q from
mixture weighting, relation weighting, and prior training.

The formal ETH seed-2035 run was initialized from the same frozen legacy GDTS
checkpoint with all JDV2 Stage-A modules freshly initialized. It was stopped
normally by the approved early diagnostic gate after epoch 4. On E>0 training
scenes, p, q, and gamma all assigned more than 99.999997% aggregate mass to
mode 1, their entropies fell to approximately 1e-6, future-shuffle sensitivity
fell to 5.94e-9, and 100% of hard assignments selected mode 1. The signature
persisted for epochs 3 and 4.

Thus, V2 removed the original free q/p expected-loss selector, but ordinary
mixture component starvation still occurred. `L_q` approaching zero is not a
success signal here: q accurately tracks a collapsed gamma and has learned to
ignore future evidence. The run was stopped before a validation plateau, as
required by the stronger early-collapse gate.

The epoch-4 checkpoint retains improvements over exact same-protocol GDTS in
the five-seed means for JADE, JFDE, Joint Goal Endpoint Error, Joint Goal
Compatibility, and Relative Motion Error. However, it fails both the latent
health gate and the marginal gate: minADE degrades 1.93% and minFDE degrades
12.27% versus GDTS. It is also worse than the protected V1 epoch-11 reference
on every reported joint metric. It is therefore not approved for Stage B.

## 2. Source and data provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Formal-run source commit | `294ebaef33225602fa390fea45f1fb9ffabc644b` |
| Objective implementation commit | `9b9b4986274805c8fb976303aa47d04b5b1abe23` |
| Cache-compatibility commit | `294ebaef33225602fa390fea45f1fb9ffabc644b` |
| Latent objective | `v2_marginal_responsibility` |
| Legacy GDTS checkpoint SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Cache manifest content hash stored in checkpoint | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| Cache `manifest.json` file SHA256 | `eadae55391b45224a3d7920e6128764db9a56298ffe70d723030ec00193786dd` |
| Cache-building source commit | `4b75110a571e6fd200938ff951a3a218ca4eb1e8` |
| Dataset hash | `cfe8cabd980d32da8071b0be573967c413a5e486418ecfb950ce4fad8f8ef0c2` |
| Train/valid/test windows | 4110 / 139 / 139 |
| Candidate/scene/relation modes | K=21 / Z=4 / M=4 |
| Energy rank / deployed samples | 8 / 20 |
| Coordinates and time step | world metres, world m/s, dt=0.4 s |

The cache was not rebuilt because the approved design explicitly permits
reuse: V2 changes neither candidates, sparse graphs, coordinates, nor future
teacher inputs. The first launch failed before any optimizer step because the
strict loader compared the cache-building commit with the newer objective
commit. A narrow compatibility fix added the explicit
`--jdv2_cache_source_commit` pin. It changes only the expected source-commit
field, still validates every other manifest field, and never edits a cache
record. The fix was committed and the formal run then began from epoch 1.

No source code changed after the formal run began. No V1 checkpoint or
optimizer state was resumed.

## 3. Implemented objective and gradient ownership

For each packed scene c and mode z, the implementation first forms FP32
scene-balanced composite log scores for the prior-relation and
future-relation paths:

```text
log_score[c,z] = 0.5 * log_score_post[c,z]
               + 0.5 * log_score_prior[c,z]

L_mix = mean_c -logsumexp_z(log p(z|X_c) + log_score[c,z])

gamma[c,z] = softmax_z(log p(z|X_c) + log_score[c,z])
gamma_teacher = stop_gradient(gamma)

L_q = mean_c KL(gamma_teacher || q_phi(z|X_c,Y*_c))
L_JG = L_mix + beta_q(t) L_q
                 + beta_r(t) L_relation(gamma_teacher)
```

The implementation enforces the approved semantics:

- q is absent from mixture PL weighting, relation-KL weighting, and prior
  training;
- `L_q` cannot update p because q uses detached prior logits and gamma is
  detached;
- relation KL detaches gamma internally;
- p receives latent-mixture gradient only through `L_mix`;
- the posterior head receives the existing 128-dimensional future scene
  representation, with no trainable direct history-scene shortcut;
- inference uses p only;
- the objective remains a latent mixture of pseudo/composite likelihoods and
  is not an exact K^N joint likelihood.

The historical `L_PL_post` and `L_PL_prior` names remain diagnostics only.

## 4. Modified files and scope audit

Latent-objective V2 changes are confined to the approved loss/model/posterior
and parser/checkpoint/diagnostic areas:

- `src/joint_goal_loss.py`: FP32 per-mode scene reduction, mixture NLL,
  gamma, forward posterior distillation, and per-mode relation KL;
- `src/models/model.py`: shared prior/post score, q-free gradient routing,
  detached gamma relation weights, and E>0/E=0 diagnostics;
- `src/models/joint_dependency_v2/future_teacher.py`: future-evidence-only
  posterior head plus detached prior context;
- `src/parser.py`: objective version, early gate, and explicit cache source
  pin;
- `src/trainer.py`: checkpoint objective metadata, diagnostic aggregation,
  deterministic selection, and two-consecutive-epoch collapse gate;
- `src/data_loader.py`: exact, explicit cache source-commit compatibility pin;
- `tests/test_joint_dependency_v2.py`,
  `tests/test_joint_dependency_v2_integration.py`,
  `tests/test_joint_dependency_v2_cache.py`, and `tests/test_parser_joint.py`:
  mathematical, gradient-routing, checkpoint, cache, deterministic-RNG,
  AMP, and all-off regression coverage.

The already approved engineering-audit utilities and reports were included in
the source commit (`src/jdv2_audit.py`, `src/utils.py`, and `tools/` audit and
repeated-validation entry points). No sampler, diffusion denoiser, candidate
generator, dynamic-relation architecture, energy architecture, or evaluation
metric was redesigned.

No forbidden entropy, balance, diversity, marginal-preservation, or semantic
mode loss was added. No Transformer/GNN or hidden-size expansion was added.

## 5. Verification before training

Using `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python`:

- full test suite: **237 passed, 4 skipped, 12 warnings**;
- non-sandbox CUDA checks: BF16 and FP16 full Stage-A autocast plus CUDA RNG
  restoration, **3 passed**;
- `python -m compileall .`: passed;
- `git diff --check`: passed;
- `test_all_off_is_exact_legacy_passthrough`: passed with tensor-exact legacy
  output equality;
- objective gradient-ownership and incompatible-resume tests: passed;
- best and last checkpoints were reloadable by the five-seed evaluator.

The four skipped tests in the CPU-suite process were CUDA-gated tests, which
were then exercised separately with GPU access.

## 6. Resolved training configuration

| Setting | Resolved value |
|---|---|
| Stage | `joint_goal` only |
| Initialization | frozen legacy GDTS; fresh JDV2 Stage-A modules |
| Seed | 2035 |
| Maximum epochs | 300, subject to early collapse/plateau stopping |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 0 (PyTorch Adam default; not explicitly overridden) |
| Scheduler | ExponentialLR, gamma=0.995 per epoch |
| Precision | BF16 autocast with frozen FP32 islands |
| Batch/window semantics | one cached window per loader item; batch size 64 argument; 4110 train windows |
| Workers / CPU threads | 1 worker; OMP=4, MKL=4 |
| Edge chunk size | 256 |
| Validation | every epoch, isolated deterministic seed 2035 |
| Final evaluation | stochastic deployed-policy evaluation, seeds 2035–2039 |
| Checkpoint selection | deterministic validation JFDE |
| Save interval / patience | 10 epochs / 12 validations |
| Beta schedule | 0 to 0.1 over first 20% of Stage-A optimizer progress, then 0.1 |

Exact command:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python main.py --dataset eth5 --test_set eth --phase train --run_name jdv2_stage_a_v2_full_seed2035 --goal_model_type joint_dependency_v2 --training_stage joint_goal --jdv2_latent_objective v2_marginal_responsibility --jdv2_early_collapse_gate True --jdv2_source_checkpoint /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/GDTS/output/eth/saved_models/best_model.pt --jdv2_cache_root /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean/outputs/joint_dependency_v2/cache/eth_full_stage_a --jdv2_cache_source_commit 4b75110a571e6fd200938ff951a3a218ca4eb1e8 --batch_cache_root /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean/outputs/joint_dependency_v2/cache/source_batches --num_epochs 300 --optimizer Adam --learning_rate 1e-4 --scheduler ExponentialLR --batch_size 64 --num_workers 1 --start_validation 1 --validate_every 1 --validation_seed 2035 --save_every 10 --early_stopping_patience 12 --jdv2_edge_chunk_size 256 --amp_enabled True --amp_dtype bf16 --seed 2035 --device cuda:0 --use_wandb False --data_augmentation False --shuffle_train_batches True --shuffle_test_batches False --num_goal_candidates 21 --num_samples 20 --trajectory_dt 0.4 --graph_type radius_ttc --graph_radius 6.0 --ttc_threshold 8.0 --adaptive_graph False --joint_diagnostics True
```

The saved command and resolved configuration are also present in the run
directory as `training_command.txt` and `config.yaml`.

## 7. Epoch-level training trend

The deterministic seed-2035 validation metrics and training losses were:

| Epoch | LR | L_mix | L_q | Relation KL | L_JG | minADE | minFDE | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000e-4 | 3.033755 | 1.138e-3 | 0.211610 | 3.033979 | 0.307589 | 0.474905 | 0.454938 | 0.814584 | 0.827529 | 0.631583 | 0.357197 |
| 2 | 9.950e-5 | 3.007764 | 5.692e-5 | 0.316742 | 3.008557 | 0.303573 | 0.464848 | 0.454273 | 0.805763 | 0.823296 | 0.597497 | 0.340588 |
| 3 | 9.900e-5 | 2.989845 | 2.016e-8 | 0.290273 | 2.991045 | 0.299531 | 0.451113 | 0.447748 | 0.790438 | 0.806792 | 0.584983 | 0.335135 |
| 4 | 9.851e-5 | 2.977868 | -2.125e-9 | 0.234339 | 2.979228 | 0.298425 | 0.450438 | 0.437697 | 0.764493 | 0.780659 | 0.563193 | 0.329005 |

The tiny negative KL values at epoch 4 are FP32 round-off near zero, not a
negative theoretical divergence. Losses, gradients, optimizer steps,
validation, BF16 execution, and checkpoints remained finite. Goal minFDE was
constant at 0.361863, as expected from the frozen candidate bank.

Validation was still improving at epoch 4, so no plateau claim is made. The
run stopped because continuing a scientifically invalid collapsed latent run
would violate the explicit early gate.

## 8. Latent diagnostics and early gate

### 8.1 E>0 training scenes

| Epoch | L_mix | L_q | Var_z(log score) | H(p) | H(q) | H(gamma) | Future-shuffle L1 | KL(q||p) diagnostic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.032805 | 1.007e-3 | 1.643e-4 | 1.531e-1 | 1.521e-1 | 1.530e-1 | 4.718e-3 | 1.020e-3 |
| 2 | 3.005641 | 6.474e-5 | 5.701e-4 | 4.809e-4 | 3.848e-4 | 4.781e-4 | 1.924e-6 | 3.571e-5 |
| 3 | 2.985766 | 7.244e-9 | 1.072e-3 | 1.162e-5 | 1.191e-5 | 1.134e-5 | 8.873e-8 | 7.227e-9 |
| 4 | 2.972926 | approximately 0 | 1.569e-3 | 1.011e-6 | 1.015e-6 | 9.791e-7 | 5.944e-9 | approximately 0 |

At epoch 4, E>0 aggregate soft usage was:

```text
p:     [1.858e-8, 0.999999970814, 1.231e-8, 2.722e-8]
q:     [1.893e-8, 0.999999970146, 1.233e-8, 2.721e-8]
gamma: [1.804e-8, 0.999999971602, 1.187e-8, 2.634e-8]
```

Hard usage was `[0, 1, 0, 0]` for p, q, and gamma. The dominant mode is an
uninterpreted index; no semantic label is assigned.

The increasing between-z score variance shows that the mode-conditioned score
path is not tensor-identical. It does not prevent collapse because the mixture
prior/responsibility gives the losing components negligible gradient. The
future evidence signal was initially measurable during epoch 1 and then shut
down as q distilled the increasingly degenerate gamma.

### 8.2 E=0 training scenes

E=0 scenes comprised 15.33% of training windows in the aggregated epoch
diagnostics. As designed, their between-z composite-score variance was exactly
zero and relation KL is not defined because there are no edges. At epoch 4:

| Diagnostic | E>0 | E=0 |
|---|---:|---:|
| L_mix | 2.972926 | 3.005168 |
| L_q | approximately 0 | 3.810e-9 |
| H(p) | 1.011e-6 | 1.781e-5 |
| H(q) | 1.015e-6 | 1.799e-5 |
| H(gamma) | 9.791e-7 | 1.781e-5 |
| Future-shuffle L1 | 5.944e-9 | 3.476e-9 |
| Var_z(log score) | 1.569e-3 | 0 |

E=0 non-identifiability is reported but was not used to trigger rejection.
The gate is based on E>0 scenes.

### 8.3 Validation confirmation

Epoch-4 E>0 validation independently showed H(p)=6.13e-7,
H(q)=5.78e-7, H(gamma)=5.26e-7, future-shuffle L1=4.39e-9, and
100% mode-1 hard usage. Its between-z score variance was 9.85e-3. This rules
out the possibility that the gate was merely a noisy minibatch artifact.

The gate hit once at epoch 3 and again at epoch 4, then wrote
`early_collapse_gate.json` and ended normally. There was no NaN, Inf, OOM,
cache mismatch, checkpoint mismatch, or tensor-contract failure in the formal
run.

## 9. Relation diagnostics

Relation KL did not monotonically track the improving validation metrics:
it changed 0.2116 -> 0.3167 -> 0.2903 -> 0.2343. At epoch 4, the normalized
E>0 predicted-relation usage was approximately
`[0.3794, 0.2067, 0.1236, 0.2903]`, and teacher usage was
`[0.3792, 0.1944, 0.1605, 0.2659]`. All four relation modes remained used;
their normalized entropies were approximately 1.0755 (predicted) and 0.8666
(teacher), below but not near zero relative to log(4)=1.3863.

Therefore the immediate early-stop cause is the scene latent, not a complete
relation-mode collapse. Nevertheless, because detached gamma is effectively
one-hot on scene mode 1, relation learning receives useful weight for only
that scene mode. This run cannot establish healthy relation behavior across
all z values.

## 10. Checkpoints and resource use

Checkpoint selection used isolated deterministic seed-2035 validation JFDE.
Every completed epoch improved this criterion, so epoch 4 is both the best and
last epoch:

| Checkpoint | Path | SHA256 |
|---|---|---|
| Best JFDE, epoch 4 | `outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_v2_full_seed2035/saved_models/best_model.pt` | `d4d36b5c057993b0fb90e414890344b0b2a4ff3ec1454cf5b8e3d7c34d38a0d5` |
| Last, epoch 4 | `outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_v2_full_seed2035/saved_models/last_model.pt` | `e63c0961633fec2da2da1cf4b8060c461c74e264c6fb82f1a631d12cb877b378` |

The best checkpoint records the objective version, epoch=4,
stage-progress=0.0133333, 16,440 completed Stage-A optimizer steps, source
checkpoint hash, cache manifest hash, architecture/ablation contracts, and
historical best-selection state.

- Total wall-clock training time: 23 min 50 s.
- Epoch runtime: approximately 342–350 s.
- Peak CUDA allocated: 866,006,528 bytes (825.9 MiB).
- Peak CUDA reserved: 979,369,984 bytes (934.0 MiB).
- Trainable/frozen parameters: 474,906 / 7,857,439.

These checkpoints are retained for audit, but neither is an accepted Stage-A
handoff checkpoint.

## 11. Five-seed same-protocol evaluation

All comparisons use the same ETH validation split, P=20 samples, stochastic
deployed-policy sampling (not MAP), seeds `{2035,2036,2037,2038,2039}`, metric
implementation, and `scene.make_world_coord_torch` coordinate conversion.

### 11.1 Stage-A V2 per seed

| Seed | minADE | minFDE | JADE | JFDE | Goal endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.298425 | 0.450438 | 0.437697 | 0.764493 | 0.780659 | 0.563193 | 0.329005 |
| 2036 | 0.288908 | 0.449846 | 0.435735 | 0.795634 | 0.794922 | 0.563744 | 0.326952 |
| 2037 | 0.290983 | 0.433362 | 0.430283 | 0.755389 | 0.773965 | 0.532129 | 0.329365 |
| 2038 | 0.289940 | 0.449526 | 0.428258 | 0.776990 | 0.776765 | 0.536278 | 0.304929 |
| 2039 | 0.290988 | 0.430872 | 0.433832 | 0.756634 | 0.764993 | 0.582703 | 0.323821 |

### 11.2 Five-seed means and reference comparison

Lower is better for every metric in this table. Relative change is
`(V2 - reference) / reference`; negative values are improvements.

| Metric | V2 epoch 4 mean +/- std | Exact GDTS mean +/- std | V2 vs GDTS | V1 epoch 11 mean +/- std | V2 vs V1 |
|---|---:|---:|---:|---:|---:|
| minADE | 0.291849 +/- 0.003377 | 0.286332 +/- 0.005157 | +1.93% | 0.288946 +/- 0.004816 | +1.00% |
| minFDE | 0.442809 +/- 0.008770 | 0.394404 +/- 0.008789 | +12.27% | 0.446046 +/- 0.009029 | -0.73% |
| JADE | 0.433161 +/- 0.003463 | 0.467982 +/- 0.002848 | -7.44% | 0.424094 +/- 0.010590 | +2.14% |
| JFDE | 0.769828 +/- 0.015022 | 0.815469 +/- 0.015009 | -5.60% | 0.750989 +/- 0.015997 | +2.51% |
| Joint Goal Endpoint Error | 0.778261 +/- 0.009799 | 0.799670 +/- 0.010164 | -2.68% | 0.759258 +/- 0.020128 | +2.50% |
| Joint Goal Compatibility | 0.555609 +/- 0.018883 | 0.674531 +/- 0.005805 | -17.63% | 0.492833 +/- 0.027062 | +12.74% |
| Relative Motion Error | 0.322814 +/- 0.009157 | 0.380473 +/- 0.005892 | -15.15% | 0.300921 +/- 0.007298 | +7.28% |

The repeated results are stored in
`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_v2/best_epoch004_five_seed.json`.
The exact GDTS and protected V1 epoch-11 values come from
`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_engineering_fix/stage_a_audit_results.json`.

## 12. Acceptance-gate decision

| Gate | Result | Evidence |
|---|---|---|
| Engineering health | Pass | finite BF16 run, stable memory, deterministic validation, reloadable checkpoints, no cache/tensor failure |
| Latent health | **Fail** | p/q/gamma jointly delta on mode 1; future sensitivity at noise floor; 100% E>0 hard usage |
| Joint gain vs GDTS | Pass | JADE, JFDE, endpoint, compatibility, and relative-motion means improve |
| Marginal preservation | **Fail** | minADE +1.93% is borderline-pass; minFDE +12.27% exceeds +2% limit |
| Improvement vs protected V1 | **Fail** | V2 is worse on all reported joint metrics; only minFDE improves slightly |

Overall decision: **do not accept this Stage-A V2 checkpoint and do not enter
Stage B**. The result is not an engineering implementation failure: the
specified objective executed as designed, passed its unit/regression tests,
and produced internally consistent responsibilities. It is a remaining
identifiability/component-starvation failure anticipated by the design
document.

## 13. Interpretation and next decision

The experiment answers the immediate scientific question: exact
marginalization plus detached posterior distillation is theoretically
self-consistent but is not sufficient, in this architecture/data regime, to
prevent a single flexible component from capturing essentially all mixture
mass. Once gamma collapses, q correctly matching gamma removes its future
signal, and the other z-conditioned components are starved of gradient.

No additional training should be launched from this checkpoint. The next
step requires a separate, explicit theoretical review of mixture
anti-starvation/identifiability mechanisms and of the persistent marginal FDE
risk. This report does not authorize or silently recommend adding entropy,
uniform balance, arbitrary diversity, marginal-preservation loss, larger
networks, or semantic labels. Any revised objective must be approved before a
new Stage-A run.

Stage B, UNIV/HOTEL, multi-seed training, and hyperparameter search were not
started.
