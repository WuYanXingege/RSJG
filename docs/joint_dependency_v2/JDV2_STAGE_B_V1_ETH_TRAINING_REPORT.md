# JDV2 Stage-B V1 ETH Training Report

## 1. Outcome

Final state:

`STAGE_B_V1_NO_CLEAR_GAIN`

The authorized ETH Stage-B V1 run completed normally. Training was numerically
healthy and only the 30,851-parameter `DependencyCorrector` changed, but the
automatically selected checkpoint did not improve JADE or JFDE over frozen
Stage A. It improved Relative Motion Error relative to the frozen production
reference by 1.30%, while degrading minADE/minFDE by 2.58%/2.20% and
JADE/JFDE by 0.69%/0.22%. Stage B is therefore not promoted or frozen.

No hyperparameter sweep, additional training seed, dataset, Stage-A change, or
Stage-B follow-up training was run.

## 2. Provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Authorized starting commit | `187bc0328af71fe6b12a945cb2868283525c27f7` |
| Formal protocol commit | `5b599443c0c6943af135ea9f1b62dfae07a7a2d0` |
| Stage-A checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Stage-A freeze manifest SHA256 | `ecce9f937717794eadc5e84be486372f01ef9c7727a4e59c36b70e09df06ec79` |
| Cache manifest SHA256 | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| Legacy GDTS checkpoint SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Best Stage-B checkpoint SHA256 | `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a` |
| Dataset/split | ETH validation, 139 windows |

The exact successful command is saved in
`outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_b_v1_eth_seed2035/training_command.txt`.
The first launch was rejected before model construction by the cache provenance
gate because its manually transcribed cache commit was wrong. A subsequent
monitoring-only restart was interrupted during epoch 1 before any checkpoint
was written. The recorded formal run then started cleanly from epoch 1 with
the exact hashes above; no resume state was used. The stale provenance fields
written by the rejected launch were corrected in the resolved config to match
the successful command, manifest, training log, and checkpoint metadata.

## 3. Formal protocol and hard gates

| Setting | Resolved value |
|---|---:|
| Training stage | `joint_trajectory` |
| Architecture version | `jdv2-stage-b-v1` |
| Training seed / validation seed | 2035 / 2035 |
| Epoch limit | 100 |
| Validation | epoch 5 onward, every 5 epochs |
| Early stopping | patience 6 validation checks |
| Selection | minimum JADE; JFDE tie-break |
| Optimizer / LR | Adam / 1e-4 |
| Scheduler | ExponentialLR |
| Loss | `L_diff + 0.05 * L_relative` |
| Gradient clip | 1 |
| Precision | BF16 with existing FP32 islands |
| K/P/M/rank | 21/20/4/8 |
| Refinement | two rounds, `exact_lexicographic_persistent_tie` |
| Trainable parameters | 30,851 |
| Frozen parameters | 8,158,433 |

The optimizer contained only the `jdv2_corrector` group. The active timestep
set remained `[30,25,20,15,10,5]`; all six timesteps were exercised. CUDA/BF16
forward/backward, scene-level oracle selection, same-world relation gathering,
deterministic training sampling, frozen-module eval state, and gradient freeze
gates passed before training.

## 4. Epoch-0 no-harm reference

The zero-initialized corrector and the frozen Stage-A path were evaluated with
paired seed 2035 and were exactly equal in all seven reported metrics:

| Metric | Stage A | Untrained Stage B | Difference |
|---|---:|---:|---:|
| minADE | 0.279318 | 0.279318 | 0 |
| minFDE | 0.388424 | 0.388424 | 0 |
| JADE | 0.421182 | 0.421182 | 0 |
| JFDE | 0.693689 | 0.693689 | 0 |
| Joint Goal Endpoint | 0.692777 | 0.692777 | 0 |
| Compatibility | 0.478072 | 0.478072 | 0 |
| Relative Motion | 0.296992 | 0.296992 | 0 |

The exact structured sampler preserves 20/20 candidates by contract; the
post-training audit independently observed 20/20 for every evaluated agent.

## 5. Training and validation trends

Training stopped at epoch 40 after six validation checks without a new JADE
best. Total wall time was 5:53:19. Training loss continued falling after the
validation optimum, establishing a generalization plateau rather than an
optimization failure.

| Epoch | total loss | L_diff | L_relative | minADE | minFDE | JADE | JFDE | Rel. motion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.78476 | 0.77844 | 0.12625 | 0.28734 | 0.39980 | 0.42516 | 0.69855 | 0.29966 |
| **10** | **0.78500** | **0.77895** | **0.12086** | **0.28516** | **0.39614** | **0.42382** | **0.69462** | **0.29952** |
| 15 | 0.78088 | 0.77492 | 0.11914 | 0.28723 | 0.39312 | 0.42446 | 0.69443 | 0.30258 |
| 20 | 0.77179 | 0.76579 | 0.11999 | 0.29037 | 0.39753 | 0.42641 | 0.69757 | 0.30544 |
| 25 | 0.76991 | 0.76393 | 0.11953 | 0.29219 | 0.40858 | 0.42757 | 0.70245 | 0.30668 |
| 30 | 0.76752 | 0.76164 | 0.11765 | 0.29060 | 0.39821 | 0.42643 | 0.69602 | 0.30562 |
| 35 | 0.76018 | 0.75433 | 0.11701 | 0.29295 | 0.40822 | 0.42866 | 0.70052 | 0.30557 |
| 40 | 0.75937 | 0.75341 | 0.11920 | 0.29008 | 0.40184 | 0.42588 | 0.69585 | 0.29962 |

Goal endpoint and compatibility stayed constant across validation epochs, as
expected because Stage A and its sampler were frozen. The automatically chosen
checkpoint is epoch 10 with selection tuple `(JADE=0.423815538,
JFDE=0.694622956)`; the later epoch-15 JFDE is slightly lower, but its primary
JADE is worse and therefore cannot replace epoch 10.

## 6. Best-checkpoint integrity

The best checkpoint records `training_stage=joint_trajectory`,
`stage_b_architecture_version=jdv2-stage-b-v1`, the exact Stage-A parent/freeze
hashes, 34,800 completed Stage-B optimizer steps, and the correct best-selection
tuple.

- State-dict key sets are identical between Stage A and Stage B.
- Non-corrector tensors changed: 0.
- Corrector tensors changed: 22/22.
- Corrector parameter count: 30,851.
- Corrector parameter-delta L2 norm: 5.285087.
- Maximum absolute parameter delta: 0.358767.
- All inference outputs and audit scalars were finite.

Thus Stage-A/GDTS hash integrity and parameter freezing held throughout the
run, and the learned corrector is not trivially zero.

## 7. Five-seed deployed-policy result

The fixed epoch-10 checkpoint was evaluated once for each inference seed
2035-2039. Lower is better.

| Seed | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Rel. motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.285165 | 0.396137 | 0.423816 | 0.694623 | 0.692777 | 0.478072 | 0.299519 |
| 2036 | 0.283450 | 0.390158 | 0.409785 | 0.691952 | 0.675271 | 0.492697 | 0.293215 |
| 2037 | 0.282397 | 0.392965 | 0.415400 | 0.717468 | 0.697011 | 0.509598 | 0.302381 |
| 2038 | 0.282929 | 0.392294 | 0.415893 | 0.711349 | 0.688420 | 0.492245 | 0.300870 |
| 2039 | 0.279139 | 0.383781 | 0.415443 | 0.702956 | 0.695656 | 0.463932 | 0.293398 |
| **Mean** | **0.282616** | **0.391067** | **0.416067** | **0.703670** | **0.689827** | **0.487309** | **0.297877** |
| **Std** | **0.001971** | **0.004116** | **0.004480** | **0.009685** | **0.007851** | **0.015375** | **0.003840** |

### Relative to frozen Stage-A production reference

| Metric | Frozen Stage A | Stage B V1 | Relative change |
|---|---:|---:|---:|
| minADE | 0.275516 | 0.282616 | **+2.577%** |
| minFDE | 0.382648 | 0.391067 | **+2.200%** |
| JADE | 0.413199 | 0.416067 | **+0.694%** |
| JFDE | 0.702125 | 0.703670 | **+0.220%** |
| Joint Goal Endpoint | 0.689827 | 0.689827 | 0.000% |
| Compatibility | 0.487309 | 0.487309 | 0.000% |
| Relative Motion | 0.301813 | 0.297877 | **-1.304%** |

The run therefore does not have the desired signature: JADE did not improve,
JFDE was only approximately preserved, and both marginal metrics crossed the
2% diagnostic band. The relative-motion gain is real but isolated.

A supplementary standard-evaluator Stage-A run was recorded to separate
corrector effects from differences between the historical adoption-audit
runner and trainer diagnostics. Relative to that paired reference, Stage B was
also worse in minADE (+1.60%), minFDE (+1.27%), JADE (+0.66%), and JFDE
(+0.23%), with effectively unchanged Relative Motion (+0.02%). This secondary
comparison does not replace the authoritative frozen production reference.

## 8. E=0/E>0, coverage, and residual behavior

| Stratum | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Rel. motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| E=0 | 0.359898 | 0.520694 | 0.359898 | 0.520694 | 0.495247 | 0.000000 | 0.000000 |
| E>0 | 0.271300 | 0.372087 | 0.444762 | 0.797146 | 0.789232 | 0.736260 | 0.450053 |

There were 47 E=0 and 92 E>0 windows per seed. On every E=0 corrector call,
`delta_epsilon` was tensor-exact zero (28,200/28,200), so small stochastic
metric differences between separately executed E=0 runs cannot be attributed
to the corrector. The E>0 comparison against frozen Stage A is the relevant
mechanism result: approximately +3.0% minADE/minFDE, +0.8% JADE/JFDE, and
-1.3% Relative Motion.

Across five seeds and 1,840 evaluated agent-seed records, the initial/final
candidate support remained exactly 20/20: mean=min=max=20 and the 20/20 rate
was 100%. Stage-B trajectory correction therefore did not damage Stage-A
candidate/world coverage.

Residual magnitude evolved as follows:

- Training mean delta/base RMS: 0.0801 at epoch 1, 0.1425 at the selected
  epoch 10, and 0.1719 at epoch 40 (observed maximum epoch mean 0.1777).
- Five-seed inference E>0 mean delta RMS: 0.1731.
- Five-seed inference E>0 mean base RMS: 0.8975.
- Five-seed inference E>0 mean delta/base RMS: 0.1948.
- E=0 delta/base RMS: exactly 0.

The residual is non-negligible, but its learned corrections trade a small
relative-motion improvement for worse marginal and aggregate joint trajectory
metrics. This is evidence of ineffective generalization, not a dead zero
corrector and not a Stage-A sampler failure.

## 9. Runtime and memory

- Formal training wall time: 5:53:19 (40 epochs).
- Mean epoch runtime from compact diagnostics: 528.85 s.
- Training peak CUDA allocated: 351,913,984 bytes (335.61 MiB).
- Training peak CUDA reserved: 585,105,408 bytes (558.00 MiB).
- Five-seed stratified inference audit: 471.21 s total, 94.24 s/seed.
- Inference peak CUDA allocated/reserved: 104,762,880 / 130,023,424 bytes.

## 10. Validation

- `python -m compileall -q .`: passed.
- `pytest`: 403 passed, 12 warnings.
- Targeted real-CUDA Stage-A freeze/Stage-B V1 tests: 22 passed.
- `git diff --check`: passed.
- Epoch-0 paired no-harm gate: exact pass.
- Best-checkpoint frozen-parameter audit: exact pass.
- Five-seed 20/20 coverage and E=0 exact-zero audit: exact pass.

## 11. Interpretation and recommendation

The improvement is not a general trajectory-level dependency improvement.
Only Relative Motion improved against the frozen production reference; JADE,
JFDE, minADE, and minFDE did not. Marginal quality is therefore not considered
preserved under the requested frozen-reference comparison, although the final
model still retains the large inherited Stage-A gains over GDTS.

Recommendation: stop at `STAGE_B_V1_NO_CLEAR_GAIN`. Do not adopt/freeze Stage
B V1 and do not tune `lambda_relative`, learning rate, architecture, timestep
sampling, oracle selection, or losses inside this experiment. Any diagnosis or
modified Stage-B proposal requires a separate review and authorization.

## 12. Artifacts

- Run: `outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_b_v1_eth_seed2035/`
- Best checkpoint: `saved_models/best_model.pt`
- Last checkpoint: `saved_models/last_model.pt`
- Machine-readable result: `outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v1/final_results.json`
- Post-training audit: `outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v1/posttraining_audit.json`
- Paired diagnostic Stage-A reference: `outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v1/stage_a_paired_reference.json`

No Stage B adoption, Stage B follow-up, Stage A modification, or additional
dataset/training run was started.
