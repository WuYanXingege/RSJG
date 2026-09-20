# Stage-A Failure Decomposition Audit

Date: 2026-09-18

Branch: `research/joint-dependency-v2-clean`

Source commit: `7beabd8d56deecd071dadcb3b36478258552deea`

Scope: read-only ETH validation audit; no training, Stage B execution, model-source change, loss change, or architecture change.

## 1. Executive decision

The audit supports two separate conclusions.

1. **There is no evidence that the collapsed categorical variable `z` is acting as a useful scene-dependent latent.** For both protected checkpoints, deployed `predicted_z` and `fixed_z=1` produce exactly identical values for all seven recorded metrics under every one of the five validation seeds. Other fixed branches are worse, but this demonstrates specialization of the permanently selected branch, not useful scene-dependent routing.
2. **The persistent marginal minFDE loss is introduced primarily by the initial finite-slot joint sampler, specifically iid categorical allocation with replacement of `P=20` slots over `K=21` candidates.** Pair-energy refinement and goal-conditioned diffusion recover part of that loss rather than create it. The decisive control is `E=0`: without any pair energy, V1 and V2 still degrade minFDE by 29.00% and 26.42%, respectively, relative to same-protocol GDTS.

Therefore:

- a fresh `use_scene_latent=False` retrained ablation is scientifically justified;
- the next controlled **training** experiment should be that no-`z` ablation with the sampler held unchanged, so it answers only whether `z` is necessary;
- the next method component indicated by the marginal-erosion evidence is the **sampler/sample-slot allocation**, not joint energy or trajectory generation;
- no V3 design and no Stage B run should begin from this audit.

## 2. Protocol and provenance

### 2.1 Checkpoints

| Model | Checkpoint | Epoch | SHA256 |
|---|---|---:|---|
| Exact same-protocol GDTS | `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/GDTS/output/eth/saved_models/best_model.pt` | legacy best | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Stage-A V1 protected reference | `outputs/joint_dependency_v2/eth/joint_dependency_v2/reference_checkpoints/jdv2_stage_a_full_seed2035/epoch_011_best_model.pt` | 11 | `be8ae2207308a674d809e570fdeafe07a2e276f088a8d5ce30bae8d9b6252db3` |
| Stage-A V2 rejected reference | `outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_v2_full_seed2035/saved_models/best_model.pt` | 4 | `d4d36b5c057993b0fb90e414890344b0b2a4ff3ec1454cf5b8e3d7c34d38a0d5` |

V1 predates the future-posterior input correction. Loading it under the current code skips exactly one shape-incompatible, inference-only key: `jdv2_future_teacher.posterior_head.0.weight`. The future posterior is not on the deployed validation path; all deployed V1 inference weights are loaded from the protected checkpoint.

### 2.2 Evaluation controls

- Split: ETH `valid`.
- Seeds: `2035, 2036, 2037, 2038, 2039`.
- Samples: `P=20`; stochastic deployed-policy evaluation, not MAP.
- Candidate bank: frozen `K=21` Goal U-Net bank from the established cache.
- Randomness: each run snapshots and isolates Python, NumPy, CPU Torch, and all CUDA RNG streams.
- Coordinates: the same `scene.make_world_coord_torch` conversion and existing metric implementation are used for GDTS, V1, and V2.
- Population: 368 valid agents per seed, or 1,840 agent observations per model across five seeds. Repeated occurrences across seeds are stochastic repeats of the same validation agents, not 1,840 independent agents.
- All interventions are read-only runtime interventions. No optimizer is constructed or stepped.

Machine-readable output:

`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_failure_decomposition/results.json`

SHA256: `5f5e4207a920dfd4c34c91c1fb00531f08850487aa781be13f9d3ff8d0df8da1`.

The audit runner is retained beside it as an experiment artifact:

`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/stage_a_failure_decomposition/run_audit.py`.

## 3. Scene-latent necessity audit

### 3.1 Interventions

- `predicted_z`: unchanged deployed stochastic policy.
- `fixed_z0` ... `fixed_z3`: force every scene and sample slot to the named mode while leaving all other sampling stochasticity unchanged.
- `mean_z_parameters`: replace the four dynamic-relation query/bias branches and four joint-energy scene embeddings by their parameter mean. The canonical unary path is already `z`-independent.
- `zero_energy_resample`: set conditional pair energy to zero but retain both refinement rounds and consume their normal random draws.
- `freeze_initial_slots`: consume the normal refinement draws but preserve the initial selected candidate IDs.

`mean_z_parameters` is only a diagnostic. Because relation softmax and effective energy are nonlinear, averaging parameters is not identical to averaging output distributions or energies. None of these interventions is a fair substitute for retraining.

### 3.2 Same-protocol reference

All values are five-seed mean ± population standard deviation; lower is better.

| Model | minADE | minFDE | JADE | JFDE | Joint goal endpoint | Compatibility | Relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| GDTS | 0.286332 ± 0.005157 | 0.394404 ± 0.008789 | 0.467982 ± 0.002848 | 0.815469 ± 0.015009 | 0.799670 ± 0.010164 | 0.674531 ± 0.005805 | 0.380473 ± 0.005892 |
| V1 predicted `z` | 0.288946 ± 0.004816 | 0.446046 ± 0.009029 | 0.424094 ± 0.010590 | 0.750989 ± 0.015997 | 0.759258 ± 0.020128 | 0.492833 ± 0.027062 | 0.300921 ± 0.007298 |
| V2 predicted `z` | 0.291849 ± 0.003377 | 0.442809 ± 0.008770 | 0.433161 ± 0.003463 | 0.769828 ± 0.015022 | 0.778261 ± 0.009799 | 0.555609 ± 0.018883 | 0.322814 ± 0.009157 |

This reproduces the known trade-off. Relative to GDTS, minFDE degrades by 13.09% for V1 and 12.27% for V2, while the joint metrics improve.

### 3.3 V1 interventions

| Intervention | minADE | minFDE | JADE | JFDE | Joint goal endpoint | Relative motion |
|---|---:|---:|---:|---:|---:|---:|
| predicted `z` | 0.288946 | 0.446046 | 0.424094 | 0.750989 | 0.759258 | 0.300921 |
| fixed `z=0` | 0.307492 | 0.479583 | 0.448456 | 0.804004 | 0.823201 | 0.342684 |
| fixed `z=1` | 0.288946 | 0.446046 | 0.424094 | 0.750989 | 0.759258 | 0.300921 |
| fixed `z=2` | 0.306220 | 0.477598 | 0.447884 | 0.804566 | 0.822837 | 0.340162 |
| fixed `z=3` | 0.302066 | 0.468982 | 0.444689 | 0.793643 | 0.808393 | 0.339325 |
| mean `z` parameters | 0.304277 | 0.471775 | 0.446543 | 0.797952 | 0.816514 | 0.340258 |

Relative to predicted/fixed-1, alternate fixed modes worsen V1 JFDE by 5.68–7.13% and minFDE by 5.14–7.52%. The mean-parameter intervention worsens JFDE by 6.25% and minFDE by 5.77%.

### 3.4 V2 interventions

| Intervention | minADE | minFDE | JADE | JFDE | Joint goal endpoint | Relative motion |
|---|---:|---:|---:|---:|---:|---:|
| predicted `z` | 0.291849 | 0.442809 | 0.433161 | 0.769828 | 0.778261 | 0.322814 |
| fixed `z=0` | 0.303671 | 0.472357 | 0.453621 | 0.813863 | 0.825218 | 0.350049 |
| fixed `z=1` | 0.291849 | 0.442809 | 0.433161 | 0.769828 | 0.778261 | 0.322814 |
| fixed `z=2` | 0.303199 | 0.469393 | 0.452369 | 0.808562 | 0.816343 | 0.348140 |
| fixed `z=3` | 0.301923 | 0.468580 | 0.450774 | 0.804418 | 0.814215 | 0.345207 |
| mean `z` parameters | 0.304403 | 0.472307 | 0.453519 | 0.814038 | 0.826068 | 0.345977 |

Alternate fixed modes worsen V2 JFDE by 4.49–5.72% and minFDE by 5.82–6.67%. The mean-parameter intervention worsens JFDE by 5.74% and minFDE by 6.66%.

### 3.5 Interpretation

For V1 and V2, `predicted_z` and `fixed_z=1` agree exactly—not merely within standard deviation—for every seed and every recorded metric. Thus:

- the learned deployed policy does not select modes as a function of the scene;
- intervention on categorical routing cannot change predictions because the prior has already collapsed to mode 1;
- the quality difference between branch 1 and branches 0/2/3 shows that branch 1 absorbed useful relation/energy parameters while the unused branches remained inferior;
- this branch specialization does **not** demonstrate that a global categorical scene latent is necessary;
- mean-parameter degradation also does not rescue that claim because nonlinear parameter averaging creates an out-of-distribution network.

The strongest conclusion supported by these checkpoints is: **the useful computation is a single specialized relation/energy branch, not scene-dependent categorical allocation.** A fresh no-`z` model is required to determine whether the same computation can be learned without redundant latent branches.

## 4. Marginal erosion audit

### 4.1 Definitions

For each valid agent and seed:

- **candidate-bank oracle**: minimum GT endpoint distance over all frozen `K=21` candidates;
- **initial goal oracle**: minimum endpoint distance among the `P=20` iid categorical samples before pair refinement;
- **selected goal oracle**: minimum endpoint distance among the final joint-goal slots after two refinement rounds;
- **trajectory minFDE**: minimum endpoint distance among the final `P=20` trajectories;
- **finite-slot loss** = initial goal oracle − bank oracle;
- **pair-refinement delta** = selected goal oracle − initial goal oracle; negative is improvement;
- **diffusion gap** = trajectory minFDE − selected goal oracle; negative means trajectory generation recovers endpoint error relative to its selected conditioning goals.

Candidate rank is computed from the frozen candidate log-prior for bank-rank coverage. Unary rank uses the trained Stage-A unary score. Candidate IDs and oracle endpoint errors are retained separately, so rank and geometric coverage are not conflated.

### 4.2 End-to-end decomposition

| Quantity | GDTS | V1 | V2 |
|---|---:|---:|---:|
| Frozen bank oracle endpoint | 0.361863 | 0.361863 | 0.361863 |
| Initial sampled-goal oracle | — | 0.499591 | 0.506813 |
| Final selected-goal oracle | 0.384980 | 0.480680 | 0.478121 |
| Final trajectory minFDE | 0.394404 | 0.446046 | 0.442809 |
| Bank → initial finite-slot loss | — | +0.137729 | +0.144950 |
| Initial → selected pair-refinement delta | — | −0.018912 | −0.028692 |
| Selected goal → trajectory diffusion gap | +0.009424 | −0.034634 | −0.035312 |

GDTS's deployed 20-goal set is only `+0.023117` above the frozen candidate-bank oracle. By contrast, the V1/V2 initial iid slots are `+0.137729/+0.144950` above it.

The relative decomposition of the final minFDE gap is exact at the mean level:

| Decomposition versus GDTS | V1 | V2 |
|---|---:|---:|
| Extra initial selection/slot loss | +0.114611 | +0.121833 |
| Pair refinement recovered | −0.018912 | −0.028692 |
| Relative diffusion recovery | −0.044058 | −0.044736 |
| Residual final minFDE gap | **+0.051642** | **+0.048404** |
| Relative final degradation | **+13.09%** | **+12.27%** |

Thus the final degradation is a residual after pair refinement and diffusion compensate for a substantially larger initial coverage deficit.

### 4.3 Slot coverage and uniqueness

| Diagnostic | V1 initial | V1 final | V2 initial | V2 final |
|---|---:|---:|---:|---:|
| Mean unique candidate IDs among 20 slots | 12.294 | 11.320 | 12.573 | 12.235 |
| Frozen-prior rank-1 coverage | 69.78% | 72.61% | 69.24% | 73.26% |
| Top-3 coverage | 97.39% | 95.60% | 97.77% | 97.45% |
| Top-5 coverage | 99.73% | 99.57% | 99.78% | 99.95% |
| Mean selected unary rank | 8.636 | 8.958 | 9.136 | 9.179 |

With `P=20` and `K=21`, repeated categorical draws cover only about 12–13 unique candidates. Approximately 30% of agent evaluations miss frozen-prior rank 1 before refinement. The mechanism is not lack of candidate-bank quality: the bank oracle is shared and remains 0.361863.

The trained unary residual also changes ordering: the GT-oracle candidate's mean rank is 8.133 under the frozen candidate prior, 9.554 under V1 unary, and 10.546 under V2 unary. This can contribute to which candidates receive slots, but the directly observed failure point remains finite stochastic allocation rather than the existence of the bank itself.

### 4.4 Pair-energy controls

| Policy | V1 minFDE | V2 minFDE |
|---|---:|---:|
| Normal predicted-`z` policy | 0.446046 | 0.442809 |
| Zero pair energy, retain refinement resampling | 0.460244 | 0.469477 |
| Freeze initial slots | 0.465148 | 0.476865 |

Normal pair energy improves minFDE over zero-energy resampling by 0.014198 for V1 and 0.026668 for V2. Even zero-energy resampling improves over frozen initial slots by 0.004904 and 0.007388. Therefore the pair-energy mechanism is a net recovery mechanism in aggregate.

The relationship between energy strength and selected-goal change is weak but points in the same direction:

- V1 Pearson/Spearman correlations between energy rank range and pair-refinement goal delta are −0.069/−0.069;
- V2 correlations are −0.073/−0.081;
- the strongest-energy quartile improves selected-goal oracle by 0.0495 for V1 and 0.0555 for V2.

There are local strata where refinement hurts, so energy is not uniformly beneficial for every agent. It is nevertheless not the primary cause of the aggregate 12–13% marginal degradation.

### 4.5 E=0/E>0 stratification

| Stratum | Model | Count | Bank oracle | Initial oracle | Selected oracle | Trajectory minFDE | Relative to GDTS |
|---|---|---:|---:|---:|---:|---:|---:|
| E=0 | GDTS | 235 | 0.486705 | — | 0.503111 | 0.516320 | — |
| E=0 | V1 | 235 | 0.486705 | 0.690792 | 0.690792 | 0.666071 | +29.00% |
| E=0 | V2 | 235 | 0.486705 | 0.678460 | 0.678460 | 0.652709 | +26.42% |
| E>0 | GDTS | 1605 | 0.343584 | — | 0.367684 | 0.376554 | — |
| E>0 | V1 | 1605 | 0.343584 | 0.471596 | 0.449916 | 0.413830 | +9.90% |
| E>0 | V2 | 1605 | 0.343584 | 0.481681 | 0.448787 | 0.412076 | +9.43% |

Every E=0 window here is a single-agent window. Pair energy and relation edges do not exist, pair-refinement delta is exactly zero, yet the largest degradation occurs in this stratum. Diffusion reduces rather than increases its selected-goal error. This rules out pair energy as the primary origin.

### 4.6 Agent-count and graph-degree stratification

Final trajectory minFDE and relative change from the matched GDTS stratum:

| Agent-count bin | Count | GDTS | V1 | V1 rel. | V2 | V2 rel. |
|---|---:|---:|---:|---:|---:|---:|
| N=1 | 235 | 0.516320 | 0.666071 | +29.00% | 0.652709 | +26.42% |
| N=2 | 200 | 0.465674 | 0.563109 | +20.92% | 0.506312 | +8.73% |
| N=3–4 | 900 | 0.314829 | 0.346230 | +9.97% | 0.351872 | +11.77% |
| N=5–8 | 415 | 0.442518 | 0.472104 | +6.69% | 0.475337 | +7.42% |
| N≥9 | 90 | 0.491583 | 0.489401 | −0.44% | 0.512993 | +4.36% |

| Degree bin | Count | GDTS | V1 | V1 rel. | V2 | V2 rel. |
|---|---:|---:|---:|---:|---:|---:|
| degree=0 | 385 | 0.380478 | 0.473173 | +24.36% | 0.468951 | +23.25% |
| degree=1 | 500 | 0.309503 | 0.392723 | +26.89% | 0.368389 | +19.03% |
| degree=2–3 | 510 | 0.429825 | 0.445312 | +3.60% | 0.463923 | +7.93% |
| degree≥4 | 445 | 0.461252 | 0.483331 | +4.79% | 0.479610 | +3.98% |

For degree-zero agents inside an otherwise E>0 window, the current two-round sampler still redraws all agents even though those agents have zero accumulated pair energy. This explains why degree-zero candidate IDs can change despite zero local energy and identifies a sampler-mechanics issue, not a hidden pair interaction.

### 4.7 Attribution verdict

| Candidate cause | Verdict | Evidence |
|---|---|---|
| A. Loss of candidate coverage | **Primary** | JDV2 initial goal oracle is 0.4996/0.5068 versus bank oracle 0.3619; rank-1 is missed about 30% of the time. |
| B. Joint-energy reranking | Not primary; net beneficial | Pair refinement improves endpoint oracle by 0.0189/0.0287; zero-energy and frozen-slot controls are worse. |
| C. Diffusion conditioned on selected goals | Not primary; net beneficial | Diffusion gap is −0.0346/−0.0353 for JDV2, compared with +0.0094 for GDTS. |
| D. Sample-slot allocation | **Mechanism underlying A** | 20 draws with replacement represent only 12.3/12.6 unique candidates initially and can resample locally unconnected agents. |
| E. Other | Secondary unary-ordering risk | The GT-oracle candidate is demoted under the learned unary score, especially in V2; this changes slot probabilities but does not overturn the directly measured allocation bottleneck. |

## 5. Exact plan for a future retrained no-`z` Stage-A ablation

This section is an implementation plan only. It is not a V3 proposal and has not been implemented.

### 5.1 Scientific contract

Train the existing Stage-A model with

```text
p(r_ij | X, g_i, g_j)
```

and no global categorical scene latent. Preserve:

- frozen Goal U-Net candidate bank, `K=21`;
- `SocialMotionEncoder`;
- `M=4` latent relations;
- candidate-pair-conditioned relation geometry;
- low-rank joint energy with rank 8;
- `P=20` joint samples and the current two-round sampler;
- the current validation protocol and all frozen GDTS components.

Remove rather than neutralize:

- `p(z|X)` and the scene-prior module;
- `q(z|X,Y*)`, future scene encoder, and scene posterior;
- `gamma` and every scene-latent loss/diagnostic;
- scene-mode sampling/allocation;
- `z` embeddings;
- `z`-indexed relation query/bias and energy branches.

The current `use_scene_latent=False` path is a useful partial-ablation compatibility path, but it is not the requested strict retrained ablation: it retains four stored relation branches and averages them, and it retains scene-indexed parameters that are merely neutralized. The strict experiment must instantiate a genuinely no-`z` parameterization.

### 5.2 Tensor contracts

Let `N` be agents, `E` canonical sparse edges, `K=21`, `M=4`, rank `D_r=8`, and `P=20`.

- Unary score remains `[N,K]` and is already canonically `z`-independent.
- Dynamic relation query becomes `[M,16]`, bias `[M]`, and full relation becomes `[E,K,K,M]`.
- Relation teacher remains `[E,M]`.
- Energy factors become `[E,M,K,8]`; relation-specific energy becomes `[E,K,K,M]`; effective energy becomes `[E,K,K]`.
- Selected-neighbor relation remains `[E,P,K,M]`.
- Selected effective energy remains `[E,P,K]`.
- Sampler candidate IDs remain `[N,P]`; there is no scene-mode, agent-mode, or edge-mode tensor.

A singleton compatibility axis may be used transiently only at a clearly marked boundary if it materially reduces implementation risk. It must not correspond to trainable `z` parameters or latent sampling.

### 5.3 Training objective

Use the existing no-scene-mode composite pseudo-likelihood for the relation/energy model:

```text
L_no_z = 0.5 L_PL_post + 0.5 L_PL_prior + beta_r L_r
```

- `L_PL_post` uses the future relation teacher, as in the approved Stage-A relation design.
- `L_PL_prior` uses hypothesis-conditioned relation inference.
- `L_r` trains the relation prior against the future relation teacher.
- There is no `L_mix`, `L_q`, `KL_z`, `gamma`, or scene-mode weighting.
- No entropy, balance, diversity, marginal-preservation, or new auxiliary loss is added.

The result remains composite/pseudo-likelihood; it must not be described as an exact `K^N` joint likelihood.

### 5.4 File-level implementation plan

| File | Required future change |
|---|---|
| `src/parser.py` | Add an explicit strict no-`z` ablation/config identity and reject incompatible scene-latent objective options. Do not silently map it to four-branch parameter averaging. |
| `src/models/model.py` | Conditionally avoid constructing/running scene prior and scene posterior; route no-`z` relation, energy, sampler, loss, trainable-module, diagnostics, and checkpoint contracts. |
| `src/models/joint_dependency_v2/future_teacher.py` | In strict no-`z` construction, retain only the relation teacher; do not instantiate future scene GRU/posterior parameters. |
| `src/models/joint_dependency_v2/dynamic_relation.py` | Add the true `[M,16]`/`[M]` no-`z` parameterization and remove scene-mode indexing from that path. Do not initialize it by averaging a trained four-branch checkpoint. |
| `src/models/joint_dependency_v2/joint_energy.py` | Add a true no-scene-embedding factor path with `[E,M,K,8]` factors and no `z`-indexed branch. |
| `src/models/joint_dependency_v2/unary_goal.py` | In strict no-`z` construction, omit the unused scene embedding; preserve the existing canonical `[N,K]` computation. |
| `src/models/joint_dependency_v2/joint_sampler.py` | Remove scene-mode allocation/indexing from the strict path while preserving `P=20`, stochastic categorical sampling, two refinement rounds, relation conditioning, and current RNG semantics. |
| `src/joint_goal_loss.py` and the JDV2 loss path in `src/models/model.py` | Compute scalar no-`z` prior/posterior composite terms and unweighted relation KL; remove all `gamma/q/p` logic from this path. |
| `src/trainer.py` | Train only SocialMotionEncoder, unary residual, relation teacher/prior, dynamic relation, and joint energy; preserve frozen GDTS and stage-local optimizer/scheduler behavior. Encode the architecture identity in checkpoint metadata. |
| Existing JDV2 test files | Add the contracts below without weakening all-off/legacy tests. |

Cache records do not contain a required sampled `z`, so the existing full ETH cache can be reused if its source/checkpoint/manifest hashes continue to pass strict preflight. No cache schema change is expected.

### 5.5 Required tests

1. Strict no-`z` state dict contains no scene-prior, scene-posterior, `scene_embedding`, or four-branch relation-query parameters.
2. Full relation `[E,K,K,M]`, energy `[E,K,K,M]`, factor `[E,M,K,8]`, sampler `[N,P]`, and E=0 shapes are exact.
3. Relation/energy gradients are finite and nonzero where expected; no gradient reaches frozen Goal U-Net, history encoder, or diffusion denoiser.
4. Agent permutation equivariance/invariance and packed-scene isolation remain valid.
5. Dynamic relations change across candidate pairs and retain future-teacher sensitivity during training.
6. E=0 and single-agent forward/loss/validation remain finite.
7. All-off GDTS output remains bitwise unchanged under the existing regression contract.
8. Strict architecture metadata prevents accidental resume from V1/V2 scene-latent checkpoints.
9. Checkpoint save/reload and deterministic five-seed repeated validation agree.
10. Complexity remains sparse and does not materialize `K^N` combinations.

### 5.6 Migration and controlled retraining protocol

- Start from the same frozen legacy GDTS checkpoint.
- Freshly initialize every no-`z` Stage-A module; do not resume V1/V2 weights.
- Use the same cache, optimizer, LR, scheduler, BF16/FP32 islands, seed 2035, validation seed, stopping policy, `K=21`, `M=4`, rank 8, and `P=20`.
- Keep the current sampler unchanged for this experiment. Combining no-`z` retraining with a coverage-aware sampler would make the scene-latent question unidentifiable.
- Select the checkpoint with deterministic validation JFDE and perform the established five-seed final evaluation.
- Compare against both exact same-protocol GDTS and protected V1 epoch 11.

Acceptance reporting must include JFDE, JADE, joint goal endpoint, compatibility, relative motion, minADE, minFDE, relation usage/entropy, candidate coverage, and E=0/E>0 strata. Marginal changes remain explicit diagnostic gates; in particular, a joint-metric gain must not hide a large minFDE regression.

## 6. Answers to the required scientific questions

### 6.1 Does `z` contribute useful prediction behavior despite collapse?

**No evidence of useful scene-dependent latent behavior is present.** Predicted routing is exactly equivalent to forcing the dominant mode. The dominant branch is useful relative to unused branches, but that is branch specialization after collapse, not evidence that categorical scene allocation is necessary.

### 6.2 Is a fresh no-`z` retrained ablation scientifically justified?

**Yes.** Read-only intervention cannot answer how a single no-`z` branch would reorganize under training. A fresh strict no-`z` ablation is the smallest controlled experiment that tests necessity without introducing a new architecture family or objective.

### 6.3 Exactly where is marginal minFDE lost?

It is lost primarily between the complete frozen candidate bank and the **initial 20 stochastic joint-sampler slots**. The bank itself is unchanged and strong. Sampling with replacement loses endpoint coverage before pair energy acts. Pair refinement and diffusion compensate for this loss, but not enough to reach the GDTS allocation baseline.

### 6.4 Which component should the next method change target?

For marginal preservation, the evidence points to the **sampler/sample-slot allocation**. It does not support targeting joint energy or trajectory generation first. Scene latent should be tested by removal, not redesigned yet. Any future sampler change must preserve stochastic deployed-policy evaluation and should be tested separately from the no-`z` ablation.

### 6.5 What is the single minimum next training experiment?

Run **one fresh ETH seed-2035 strict no-`z` Stage-A ablation with the existing sampler unchanged**, followed by the same deterministic checkpoint selection and five-seed final evaluation. This isolates question A. Do not combine it with a sampler fix, do not resume a collapsed checkpoint, and do not start Stage B.

After that ablation is reviewed, the audit supports a separate coverage-aware sampler experiment as the minimum experiment for question B; it should not be conflated with the no-`z` run.

## 7. Risks and limits

- Fixed-mode interventions test existing trained branches, not the capacity or optimization of a retrained no-`z` model.
- Mean-parameter intervention is nonlinear and out of distribution; it is qualitative only.
- The five seed repeats quantify stochastic inference variation on one validation split, not training-seed uncertainty.
- Agent records repeated over seeds are paired stochastic observations, not independent dataset samples.
- Pair energy can hurt individual agents or strata even though its aggregate effect is beneficial.
- Unary residual ordering is a secondary risk and should remain measured in the no-`z` run, but modifying its loss now would confound the requested experiment.
- A future coverage-aware sampler must be designed and reviewed separately; this report does not authorize such a method change.

## 8. Status

`AUDIT_COMPLETE_NO_TRAINING_LAUNCHED`

Stage B remains blocked pending review of this report and any subsequently approved Stage-A experiment.
