# RSJG Joint Dependency V2 Stage-A Training Diagnosis

## 1. Review scope and snapshot

This is a read-only diagnosis of the ETH `joint_goal` run. No model source,
loss, cache, checkpoint, or running process was changed by this review.

- Source commit: `4b75110a571e6fd200938ff951a3a218ca4eb1e8`
- Run: `jdv2_stage_a_full_seed2035`
- Seed: `2035`
- Cache manifest: `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949`
- Frozen GDTS checkpoint SHA256:
  `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950`
- Snapshot boundary: epoch 27, the last complete epoch at review time. Any
  partially written later epoch is excluded.
- Configuration: BF16, Adam, learning rate `1e-4`, ExponentialLR with
  `gamma=0.995`, 300 configured epochs, validation every epoch, periodic
  checkpoint every 10 epochs, no early stopping.

The review used the training curve, per-epoch diagnostics, validation logs,
saved checkpoints, the frozen design documents, and the implementation of the
Stage-A loss, sampler, diffusion validation path, checkpoint transition logic,
scene prior/posterior, and relation teacher/prior.

## 2. Executive decision

The run is numerically healthy, but the full Stage-A design is not healthy
enough to hand directly to Stage B.

1. Validation performance reached an effective plateau around epochs 10--12.
   The training objective is still decreasing, so this is not optimizer
   convergence; it is validation saturation/decoupling.
2. The scene latent has unambiguously collapsed to scene mode 1. Both prior and
   posterior assign essentially all probability to that mode, entropy is near
   zero, and validation allocates every joint sample to it.
3. The relation module is not fully collapsed. It remains multi-mode and the
   implementation is candidate-pair conditioned, but its usage is becoming
   strongly concentrated and the falling relation KL no longer tracks
   validation improvement after the early phase.
4. Validation is stochastic. A single validation pass is not sufficient to
   resolve close checkpoints, and current checkpoint selection is therefore
   noisy.
5. The current best checkpoint is epoch 11 under the frozen primary criterion
   JFDE. It also wins the principal marginal, joint-trajectory, and endpoint
   metrics among completed epochs.
6. Do not start Stage B yet. First stop/hold the current Stage-A run, preserve
   epoch 11, repeat validation under controlled seeds, run the minimal
   baseline/ablation checks below, and correct the Stage-B progress reset issue.

The recommended choice is **B, followed by a narrowly scoped C**: stop the
current Stage-A continuation, diagnose/approve the minimum change needed for
latent identifiability and deterministic validation, then rerun Stage A. It is
not A (blind continuation) and not D (immediate Stage B).

## 3. Current training trend

### 3.1 Loss and learning rate

| Epoch | LR | L_PL_post | L_PL_prior | L_r | L_z | L_JG |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000e-4 | 3.03327 | 3.03869 | 0.14633 | 6.076e-2 | 3.03615 |
| 2 | 9.9500e-5 | 2.99271 | 3.02446 | 0.36519 | 7.407e-6 | 3.00951 |
| 10 | 9.5589e-5 | 2.92411 | 2.95640 | 0.09356 | 2.013e-9 | 2.94173 |
| 11 | 9.5111e-5 | 2.91912 | 2.95028 | 0.08344 | 1.608e-9 | 2.93616 |
| 18 | 9.1832e-5 | 2.89665 | 2.91779 | 0.04408 | 6.800e-10 | 2.90851 |
| 20 | 9.0916e-5 | 2.89222 | 2.91096 | 0.03743 | 2.080e-9 | 2.90281 |
| 26 | 8.8222e-5 | 2.88160 | 2.89506 | 0.02469 | 9.184e-10 | 2.88938 |
| 27 | 8.7781e-5 | 2.88045 | 2.89293 | 0.02286 | 5.414e-10 | 2.88770 |

From epoch 11 to 26, `L_JG` improves by 1.59% and relation KL improves by
70.4%, while JFDE is essentially unchanged (+0.15%) and JADE, minADE, and
minFDE are worse by 3.62%, 6.07%, and 3.29%, respectively. The learning rate
at epoch 27 is still 87.8% of its initial value, so the plateau is not caused
by an exhausted learning rate.

The KL warm-up is also incomplete: beta is 0.045 at epoch 27 and reaches 0.1
only at 20% of 300 epochs, approximately epoch 60. The scene collapse occurred
before beta became appreciable, so simply completing the warm-up is unlikely
to restore the unused modes.

### 3.2 Validation trend

The early phase shows genuine within-run improvement. Relative to epoch 1,
epoch 11 improves JFDE by 15.24%, JADE by 12.89%, joint-goal endpoint error by
15.76%, and the marginal minADE/minFDE by 12.55%/14.36%. The best observed
compatibility and relative-motion values occur later, at epochs 20 and 21.

However, after epoch 10 the metrics oscillate rather than improve
persistently. Over epochs 10--18:

| Metric | Mean | Standard deviation | Range |
|---|---:|---:|---:|
| JFDE | 0.7480 | 0.0256 | 0.7091--0.7852 |
| JADE | 0.4275 | 0.0156 | 0.4043--0.4482 |
| minADE@20 | 0.2882 | 0.0073 | 0.2742--0.2951 |
| minFDE@20 | 0.4380 | 0.0155 | 0.4092--0.4564 |
| Joint Goal Endpoint Error | 0.7593 | 0.0280 | 0.7157--0.7996 |
| Joint Goal Compatibility | 0.5047 | 0.0204 | 0.4803--0.5458 |
| Relative Motion Error | 0.3020 | 0.0064 | 0.2924--0.3121 |

Epoch 25 is a particularly poor stochastic pass (JFDE 0.8916), followed by
epoch 26 at 0.7102, almost equal to the epoch-11 best. Such a one-epoch rebound
is not consistent with the smooth training-loss curve.

There is no train-set JFDE/JADE or validation PL loss in the current logs, so a
formal train-minus-validation generalization gap cannot be computed. The
available evidence instead shows a proxy gap: training loss continues to fall
while held-out predictive metrics stop improving.

### 3.3 Saturation judgement

Stage A has **validation-level saturation but not training-loss saturation**.
Continuing the unchanged run may further reduce PL/KL losses, but the available
evidence does not support expecting a better deployable joint predictor. The
negative late-phase association is instructive: over epochs 10--26, relation
KL and JFDE have Pearson correlation about -0.36, meaning relation KL keeps
falling while JFDE tends to worsen. This is not proof of overfitting by itself,
because validation is stochastic, but it rules out treating lower KL as a
surrogate for better joint prediction.

## 4. Validation stochasticity

### 4.1 Randomness audit

| Component | Current behavior | Deterministic per epoch? |
|---|---|---|
| Cached 21 goal candidates | Frozen/cache-backed | Yes |
| Validation order | `shuffle_test_batches=False` | Yes |
| Scene-mode allocation | Floor plus largest remainder from `p(z|X)` | Yes for fixed probabilities |
| Unary joint-goal initialization | `torch.multinomial` in `sample` mode | No |
| Two conditional-refinement rounds | `torch.multinomial` each round | No |
| Continuous goal refinement | Disabled | Not applicable |
| Shared diffusion trunk | Random Gaussian `x_T` | No |
| ETH diffusion branches | Random noise with `eta=1` | No |
| Validation seed | Seeded once at process start, not reset per validation | No |
| RNG restoration | CPU torch RNG only | Incomplete; CUDA/Python/NumPy state is not restored |

No explicit generator is passed to the JDV2 sampler. The validation path calls
the full diffusion sampler, so Stage-A validation includes both categorical
goal noise and trajectory diffusion noise. `joint_sampling_mode=map` would
remove categorical sampling but would not make diffusion deterministic.

### 4.2 How much of the fluctuation is sampling noise?

It cannot be identified from this curve alone because model parameters also
change each epoch. The epoch-10--18 standard deviations above are an upper
bound on sampling variability plus checkpoint drift, not a clean estimate of
sampling variance. The epoch-25/26 reversal and the absence of matching
training-loss shocks strongly suggest that sampling accounts for a material
part of the observed fluctuations.

### 4.3 Required validation protocol

Add two evaluation views before any checkpoint is promoted:

1. A fixed-seed validation pass that snapshots and restores CPU, CUDA, NumPy,
   and Python RNG state and uses the same fixed random streams for every
   checkpoint. Keep stochastic sampling if it is the deployed policy; fixed
   random numbers are preferable to changing the policy to MAP for primary
   selection.
2. Repeated stochastic validation of the same checkpoint, minimally five
   seeds, reporting mean, standard deviation, and the per-seed metrics. Repeat
   epoch 11 and epoch 20 first. Epochs 16/24 cannot currently be recovered
   unless a checkpoint exists outside the reviewed directory.

A MAP-goal plus fixed-diffusion-noise evaluation is useful as a secondary
deterministic diagnostic, not as a replacement for the deployed stochastic
metric.

## 5. Scene-latent health

### 5.1 Observed collapse

The scene latent has clear posterior, prior, and usage collapse:

| Epoch | Mean p(z), rounded | Mean q(z), rounded | H[p] | H[q] | KL(q||p) |
|---:|---|---|---:|---:|---:|
| 1 | [0.020, 0.906, 0.034, 0.040] | [0.033, 0.935, 0.012, 0.019] | 0.248 | 0.154 | 6.076e-2 |
| 2 | [1.2e-5, 0.999925, 2.5e-5, 3.8e-5] | [1.7e-5, 0.999935, 2.1e-5, 2.7e-5] | 7.89e-4 | 7.09e-4 | 7.41e-6 |
| 11 | [8.3e-10, ~1, 8.2e-10, 1.2e-9] | [6.1e-10, ~1, 7.0e-10, 1.0e-9] | 5.35e-8 | 4.54e-8 | 1.61e-9 |
| 27 | [3.4e-10, ~1, 3.6e-10, 5.0e-10] | [2.5e-10, ~1, 2.6e-10, 3.9e-10] | 2.19e-8 | 1.75e-8 | 5.41e-10 |

The maximum entropy for four modes is `log(4)=1.386`. Validation sampled-mode
usage is exactly `[0,1,0,0]` from the first logged epoch onward. This is not a
case where KL is small because a rich posterior has been successfully matched
by a rich prior. Both distributions are effectively deterministic and use the
same single mode.

### 5.2 Why the current objective permits this

The implementation matches the documented dual-PL objective, but its mode
weighting is an expectation of per-mode loss:

```text
sum_z q(z) * PL_z    and    sum_z p(z) * PL_z
```

This gives both prior and posterior an immediate incentive to put all mass on
whichever randomly initialized mode currently has the lowest PL. The future
posterior also receives `history_scene` as input and is free to ignore its
future GRU features. Once both heads select the same mode, KL becomes nearly
zero and does not preserve future sensitivity. There is no entropy or usage
regularizer by frozen design. The collapse happens while beta is still tiny,
so it is primarily a PL/identifiability degeneracy rather than excessive KL
pressure late in warm-up.

### 5.3 Does z participate in prediction?

Structurally, yes: `p(z|X)` chooses the scene modes used by the sampler, and z
indexes the dynamic-relation query/bias and joint-energy scene embedding. The
posterior `q(z|X,Y*)` weights both posterior PL and relation KL during training.

Functionally in this run, only mode 1 participates. The other three
mode-specific relation and energy branches are effectively unused at
deployment. Thus the scene latent does not provide the intended multimodal
global future representation.

### 5.4 Required latent diagnostics

The existing entropy/usage logs are necessary and were decisive here. Add the
following read-only diagnostics before choosing a remedy:

- per-scene prior and posterior entropy distributions, not only means;
- hard mode counts and soft usage for p and q;
- `mean |q-p|` and KL distributions;
- future sensitivity: keep X fixed and permute/shuffle Y between compatible
  scenes, then measure change in q with JS/KL distance;
- contribution sensitivity: evaluate the same checkpoint with z fixed versus
  its predicted allocation, without retraining, clearly labelled as an
  interventional audit.

Do not interpret mode indices semantically. Do not add balance/entropy losses
without a separate theoretical decision, because that would change the frozen
objective.

## 6. Relation-module health

### 6.1 Usage and entropy

Relation diagnostics are averaged per window; an `E=0` window contributes an
all-zero usage vector and zero entropy. There are 630 zero-edge training
windows, so raw usage vectors sum to `3480/4110 = 0.8467`, not one. Normalizing
the logged vectors over non-empty windows gives:

| Epoch | Deployable relation usage | Teacher relation usage |
|---:|---|---|
| 1 | [0.209, 0.224, 0.334, 0.234] | [0.210, 0.225, 0.331, 0.234] |
| 11 | [0.189, 0.076, 0.387, 0.348] | [0.218, 0.068, 0.413, 0.302] |
| 20 | [0.116, 0.028, 0.429, 0.427] | [0.131, 0.021, 0.473, 0.375] |
| 27 | [0.071, 0.014, 0.498, 0.416] | [0.075, 0.010, 0.545, 0.369] |

After correcting for empty windows, approximate deployable/teacher entropies
at epoch 27 are 0.846/0.808 nats. They are below the four-mode maximum but not
zero. The relation module therefore has increasing concentration, especially
loss of mode 1, but has not collapsed to a single mode.

### 6.2 Relation KL versus validation

Relation KL falls from 0.146 at epoch 1 to 0.0229 at epoch 27. This initially
coincides with better validation, but after epoch 10 the relationship breaks:
KL continues to fall while JFDE/JADE oscillate or worsen. Relation KL shows
teacher-prior agreement on the soft GT-weighted candidate pairs; it does not by
itself show that the learned modes improve deployable joint samples.

### 6.3 Candidate-pair conditioning

The code path is genuinely candidate-pair conditioned: it builds four
goal-geometry scalars for every `(k,l)`, encodes them, and contracts the result
with the scene/relation query. A read-only real-candidate probe on one six-edge
ETH training window, with the saved dynamic-relation module and neutral base
logits, found at epoch 11:

- mean probability standard deviation across candidate pairs: 0.00652;
- mean L1 deviation from the pair-averaged relation: 0.0206;
- mean maximum pair total-variation distance: 0.0390.

At epoch 20 these values were 0.00552, 0.0174, and 0.0337. This confirms a
non-constant candidate-geometry path, but its effect is modest and slightly
smaller at epoch 20. Because the probe neutralizes the learned history logits,
it is not a full validation-set effect-size estimate. Current aggregate logs
cannot prove that deployed relations differ materially across candidate pairs.

Required additions are full-validation distributions of:

- KL/TV between `p(r|X,z,k,l)` and the history-only relation;
- variance across `(k,l)` and across z;
- teacher response after future permutation;
- usage and entropy conditioned on `E>0` and weighted both per edge and per
  scene.

The relation module is provisionally healthy at the numerical level, but its
scientific contribution remains unproven until the dynamic-relation ablation.

## 7. Joint energy and system health

Energy magnitude grows smoothly: mean/std change from -0.142/0.061 at epoch 1
to -1.251/0.695 at epoch 27; the epoch-27 range is approximately
[-4.980, 0.022]. There is no NaN/Inf, but increasing scale without a
corresponding validation trend should be monitored for sampler overconfidence.

The run is operationally stable over epochs 1--26:

- epoch runtime: 301--331 s, mean 319 s;
- peak CUDA allocated: 826.15 MiB, constant;
- peak CUDA reserved: 934 MiB, constant;
- BF16, gradients, optimizer steps, validation, and checkpoint saves show no
  logged numerical failure.

## 8. Best checkpoint

### 8.1 Metric-wise best completed epochs

| Metric, lower is better | Best epoch | Value |
|---|---:|---:|
| JFDE | 11 | 0.709122 |
| JADE | 11 | 0.404284 |
| minADE@20 | 11 | 0.274197 |
| minFDE@20 | 11 | 0.409224 |
| Joint Goal Endpoint Error | 11 | 0.715697 |
| Joint Goal Compatibility | 20 | 0.468559 |
| Relative Motion Error | 21 | 0.288539 |

The frozen Stage-A comparator is JFDE. Epoch 11 also simultaneously gives the
best JADE, minADE, minFDE, and joint-goal endpoint error, so it is the most
defensible current checkpoint. The later compatibility/relative-motion wins
do not outweigh this broad dominance, especially because they come from
single stochastic passes.

Current selection:

```text
outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/
jdv2_stage_a_full_seed2035/saved_models/best_model.pt
```

The checkpoint metadata correctly records epoch 11 and primary JFDE
0.7091219624. It also records the architecture, ablation switches, cache hash,
source-checkpoint hash, optimizer, scheduler, and Stage-A progress.

This is the best **observed single-pass** checkpoint, not yet the best expected
checkpoint under stochastic evaluation. Re-evaluate epoch 11 and epoch 20
with the same five seeds before final promotion.

### 8.2 Checkpoint risks found in code

1. Same-stage resume does not restore `_best_selection` or the historical
   best-metric tables. They are initialized to infinity. The first validation
   after resuming Stage A can therefore overwrite `best_model.pt` even when it
   is worse than epoch 11.
2. There is no explicit `last_model.pt`. Only periodic epoch checkpoints and
   the best checkpoint are saved. At the snapshot, epoch 20 is the latest
   periodic state despite complete logs through epoch 27.
3. Periodic checkpoints are written before that epoch's validation. This is
   not wrong for weights, but their embedded best-selection state reflects the
   preceding validation history.

Preserve the current epoch-11 file before any resume. Correct the resume-state
logic before relying on automatic checkpoint selection after interruption.

## 9. Goal metric and baseline comparison

`Goal_minFDE` is exactly 0.3618628 for every epoch because the implementation
computes it from the frozen 21-candidate Goal U-Net bank, not from the learned
joint candidate selection. Its constancy is expected and is evidence that the
candidate generator stayed frozen; it is not evidence that Stage A stopped
learning.

The existing legacy artifact under
`GDTS/output/eth/joint/final_results/eth_stage1_epoch010_seed2035` is not a
valid direct baseline for the current curve: it is a held-out test result for a
different `training_stage=joint` model and uses a selected best-of-20 seed
protocol, while the current numbers are one-pass validation results. Therefore
the present logs establish improvement over early Stage-A epochs, but do not
yet establish superiority over the marginal GDTS baseline.

## 10. Stage-B interface audit

### 10.1 Correctly implemented interfaces

- Stage A inference produces prior scene-mode allocations, complete joint goal
  assignments `[N,P,2]`, selected relation probabilities, and relation
  embeddings `[E,P,16]`.
- Stage B receives joint goal-conditioned frozen GDTS contexts, the canonical
  sparse graph and weights, last world/map positions, and branch-specific
  relation embeddings.
- The common diffusion trunk remains unchanged. The corrector is applied only
  after the branch split as `epsilon_base + Delta_epsilon`.
- In `joint_trajectory`, GDTS goal U-Net/history encoder/denoiser and the whole
  Stage-A stack are set to `requires_grad=False` and `eval()`. The optimizer
  family contains only the corrector.
- The frozen base denoiser runs under `no_grad`. The Stage-A checkpoint's
  corrector output layer is still exactly zero, so Stage-A validation did not
  silently learn or apply a nonzero trajectory correction.
- Strict architecture, ablation, cache-hash, source-hash, and allowed-stage
  transition checks are present.

### 10.2 Blocking Stage-B issue

Cross-stage checkpoint loading returns `start_epoch = source_epoch + 1` and
also inherits the saved `stage_progress`. Per-batch progress is then recomputed
from the absolute epoch number divided by the new stage's configured total.
Starting Stage B from epoch-11 Stage A therefore starts Stage B at epoch 12 and
about 3.67% progress rather than epoch 1 and 0% completed Stage-B steps. It
also executes 289 rather than 300 Stage-B epochs and shortens/shifts the
teacher curriculum.

This violates the frozen requirement that the Stage-B teacher probability be
based on completed Stage-B optimizer steps. Stage B must not start until the
stage-local epoch/progress reset is corrected and tested. Optimizer/scheduler
state is correctly not inherited across stages, but that does not fix the
epoch/progress issue.

Even after that implementation correction, entering Stage B immediately is
not recommended because the selected Stage-A scene latent is functionally
single-mode.

## 11. Decision among A/B/C/D

- **A. Continue unchanged Stage A:** not recommended. The validation curve has
  saturated, scene modes collapsed by epoch 2, and continued PL/KL improvement
  has not translated to better held-out prediction.
- **B. Stop Stage A early:** recommended for the current run. Preserve epoch
  11 and do not spend the remaining 273 epochs merely driving KL lower.
- **C. Adjust Stage A and retrain:** recommended only after the minimal audits
  below and explicit approval of any objective change. Deterministic validation
  and checkpoint/progress fixes are engineering corrections; a latent-collapse
  remedy changes the frozen training design and must not be introduced
  silently.
- **D. Freeze current checkpoint and enter Stage B:** not recommended now.
  Epoch 11 should be frozen as a reference checkpoint, but Stage B is gated by
  the progress-reset bug, stochastic checkpoint confirmation, baseline
  comparison, and scene-latent collapse decision.

## 12. Minimum necessary experiments

Run these in order, without a broad hyperparameter search.

1. **Checkpoint-noise audit, no training.** Evaluate epoch 11 and epoch 20 on
   the same validation windows with five fixed seeds. Report mean/std for all
   seven Stage-A metrics. Also run one fixed-random-stream validation for each.
2. **Exact marginal baseline, no training.** Evaluate the legacy/all-off GDTS
   checkpoint on the identical validation and held-out test records, sample
   count, seed set, and metric code. This answers whether Stage A beats the
   marginal baseline; do not compare validation with the existing legacy test
   artifact.
3. **Latent functionality audit, no training.** Measure p/q entropy and hard
   usage per scene, future-shuffle sensitivity of q, and prediction sensitivity
   when z is fixed. This confirms whether any future information survives
   beyond the already observed single-mode output.
4. **Relation functionality audit, no training.** Measure candidate-pair and z
   variance of `p(r)`, teacher future-shuffle sensitivity, and the change in
   sampled goals/metrics when the dynamic residual is neutralized. Label this
   as inference intervention, not a retrained ablation.
5. **Three controlled Stage-A ablations, only if a retraining decision is
   approved.** One seed each, same cache and schedule: `use_scene_latent=False`,
   `use_dynamic_relation=False`, and `use_joint_energy=False`. Retraining is
   required for a fair contribution claim; evaluation-time disabling measures
   reliance, not counterfactual trained performance.
6. **One Stage-B run only after the gates pass.** Fix/test stage-local progress,
   initialize from the confirmed Stage-A checkpoint, train only the corrector,
   and compare corrector-on against corrector-off with the same evaluation
   seeds. This answers whether dependency correction improves joint metrics on
   top of Stage A without damaging marginals.

## 13. Risk register

| Risk | Severity | Evidence / consequence |
|---|---|---|
| Scene prior/posterior single-mode collapse | Blocking scientific risk | p/q and sampled usage are essentially `[0,1,0,0]` |
| Stochastic one-pass checkpoint selection | High | Goal multinomial plus diffusion noise; no per-validation seed reset |
| Stage-B progress inherited from Stage A | Blocking implementation risk | Incorrect teacher-curriculum clock and stage duration |
| Best checkpoint overwritten after resume | High | Historical comparator state is not restored |
| No explicit last checkpoint | Medium | Interruption loses all work after the latest periodic save |
| Relation concentration | Medium | Two modes dominate; one mode falls to about 1% usage |
| Candidate-conditioned relation effect may be weak | Medium | Nonzero but modest probe variance; no full-validation effect log |
| Energy scale growth | Medium | Magnitude rises while validation is flat; monitor sampler sharpness |
| Invalid baseline comparison | High scientific risk | Existing legacy artifact is a different model/split/seed-selection protocol |
| Marginal degradation cannot yet be judged | High | No same-protocol GDTS baseline evaluation is available |

## 14. Final status

Stage A is operationally stable and its epoch-11 checkpoint is the correct
current reference. Nevertheless, scene-latent collapse and validation
stochasticity prevent a claim that the full JDV2 Stage-A mechanism is working
as designed. The Stage-B progress-reset issue independently prevents a safe
stage transition.

**Action:** hold Stage B, preserve epoch 11, perform the fixed-seed/repeated
validation and exact marginal-baseline evaluation, audit latent/relation
sensitivity, then request approval for the minimum Stage-A correction and
rerun decision.
