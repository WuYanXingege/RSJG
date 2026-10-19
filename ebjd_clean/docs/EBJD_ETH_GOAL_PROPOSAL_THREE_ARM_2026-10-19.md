# EBJD ETH goal-proposal three-arm audit

Status: **COMPLETE**  
Date: 2026-10-19  
Scope: read-only forward inference; **0 optimizer updates; no training; no gamma scan**.

## Decision

The next single direction is to **migrate the native GDTS Route-A goal-proposal path into EBJD as the proposal base**, retaining the endpoint-bridge/joint model only downstream of those proposals. Do not spend the next experiment on more diffusion-weight or goal-loss tuning.

The evidence is decisive on this frozen ETH mirror set:

- Route A gives endpoint minFDE@20 **0.575 m**.
- Direct EBJD epoch-5 heatmap proposals (B) give **1.865 m**, a paired mean degradation of **+1.290 m** versus A; B beats A for only **12.1%** of agent occurrences.
- Actual EBJD rollout endpoints (C) recover to **1.015 m**, but remain **+0.441 m** behind A and beat A on only **39.6%** of occurrences.
- Conditional B0 is worse still at **2.523 m**. Epoch-5 training improves B over B0 by **0.659 m** on average, so the result is not evidence that training merely destroyed an otherwise good EBJD-crop goal head. The EBJD-crop proposal route is weak at initialization and remains weak after five epochs.

This supports a proposal-source repair, not a claim that the endpoint-bridge representation or joint interaction idea is intrinsically invalid. C is materially better than B, demonstrating that the EBJD diffusion path does not simply reproduce the auxiliary heatmap proposals.

## Frozen protocol and identities

All routes use the same JMM ETH validation/test mirror: **253 windows, 364 agent occurrences, 8 observed frames, 12 future frames, 0.4 s interval, metres, seed 2035, K=20**. `independent_test=false`; these are diagnostic validation results, not an independent final test claim.

| Asset | SHA256 |
|---|---|
| ETH manifest | `03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b` |
| EBJD epoch 5 | `a5f45606329b80754567cf86381ceff0d19e68c8eaddf999c74f71ef1bdde525` |
| matched GDTS checkpoint | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| B0 parent goal-only checkpoint | `d130f47a3e13f457920a724465cc02d55bdb22e4827f7889bc49331d6c3bfec6` |

The formal EBJD checkpoint was trained from source commit `d266681`; the native GDTS execution tree is `092ed91`; the frozen probe implementation is `c0da9c6`.

Route definitions were enforced as follows:

- **A:** matched GDTS goal U-Net, native full-scene semantic/trajectory preprocessing, sigmoid of channel 12, native TTST, and the **20 k-means centres actually used as diffusion branches**.
- **B:** EBJD epoch-5 map U-Net with the exact EBJD 6-channel 256x256 semantic crop plus eight observed heatmaps, sigmoid of channel 12, then the same native TTST.
- **C:** final endpoints of the previously certified epoch-5 P20, gamma=1 trajectory bank.
- **B0:** only because B failed against A: the bound parent goal-only U-Net translated exactly into EBJD's U-Net layout, evaluated with the same EBJD crops and TTST as B.

Native TTST samples 1,000 pixels, clusters them into 20 centres, then appends one argmax. The appended argmax is reported only as an auxiliary diagnostic and is excluded from every main K=20 value. This prevents the historical 20-versus-21 ambiguity.

## Main endpoint results

Lower is better except recall, pairwise spread and unique-count diagnostics.

| Route | mean minFDE@20 | median | P90 | P95 | recall@0.5 m | recall@1.0 m |
|---|---:|---:|---:|---:|---:|---:|
| A native GDTS proposals | **0.575** | **0.372** | **1.163** | **1.910** | **66.8%** | **87.4%** |
| B EBJD e5 heatmap proposals | 1.865 | 1.391 | 4.362 | 5.052 | 17.3% | 43.7% |
| C EBJD e5 rollout endpoints | 1.015 | 0.587 | 2.580 | 3.412 | 46.2% | 65.7% |
| B0 parent goal-only on EBJD crop | 2.523 | 1.825 | 5.910 | 6.446 | 12.1% | 36.0% |

Paired deltas use `right - left`, so positive values mean the right route is worse:

| Pair | mean delta | median delta | right better |
|---|---:|---:|---:|
| A -> B | +1.290 m | +0.606 m | 12.1% |
| A -> C | +0.441 m | +0.171 m | 39.6% |
| B -> C | -0.849 m | -0.508 m | 71.4% |
| B0 -> B | -0.659 m | -0.154 m | 62.1% |

## Proposal geometry diagnostics

| Route | RMS radius from crop centre | mean pairwise distance | mean single-candidate error | effective unique K |
|---|---:|---:|---:|---:|
| A | 4.391 m | 2.205 m | 2.250 m | 19.978 |
| B | 14.749 m | 18.112 m | 13.763 m | 20.000 |
| C | 5.100 m | 1.806 m | 2.513 m | 20.000 |
| B0 | 16.184 m | 19.395 m | 15.791 m | 20.000 |

B is not failing because TTST collapsed to duplicate centres: it produces 20 unique candidates. It fails because its probability mass sends TTST across almost the entire 32 m crop. The auxiliary B argmax error is also **5.849 m** (A: **1.833 m**), so excessive diversity is not the only issue; the heatmap mode itself is poorly located.

C has much more compact endpoint samples than B and recovers 0.849 m in paired minFDE, consistent with the diffusion denoiser using learned map features and history/social conditioning rather than treating the auxiliary endpoint heatmap as its proposal bank. However, C's compact set remains less covering than A's native goal proposals.

## Strata and crop boundary

| Route | N=1 mean minFDE@20 | N>=2 mean minFDE@20 |
|---|---:|---:|
| A | 0.680 | **0.468** |
| B | 1.510 | 2.223 |
| C | 0.994 | 1.037 |
| B0 | 2.217 | 2.833 |

Weighted reconstruction from both crowd strata exactly recovers each overall mean. All **364/364** ground-truth endpoints, and also all complete 12-step future trajectories, lie inside the EBJD crop (`|x|<=16 m`, `|y|<=16 m` relative to the last observation). Consequently:

- the in-crop stratum is the full dataset;
- the out-of-crop stratum count is zero;
- the preregistered out-of-crop example does not exist and is correctly reported as unavailable rather than fabricated;
- crop truncation of ground truth cannot explain the ETH deficit, although crop construction/input statistics may still differ from native GDTS preprocessing.

## Dataflow audit

The actual EBJD encoder constructs eight observed heatmaps in crop-normalized world coordinates and concatenates them with six semantic channels (`encoders.py:105-123`). Its U-Net returns both 32-channel features and 12 future logits (`encoders.py:158-173`). The features enter 64 map-memory tokens (`encoders.py:172-182`) and are also sampled along every denoised trajectory with zero padding and `align_corners=True` (`denoiser.py:166-175`). The logits receive BCE supervision for all 12 future channels; targets are Gaussian maps centred on world-coordinate offsets and zeroed only if a target centre is outside the crop (`objectives.py:121-142`). Map BCE is added at weight 0.1 to diffusion loss (`trainer.py:367-372`). There is no special final-channel weight.

Thus the last heatmap is trained as an auxiliary future-position channel, but EBJD inference never calls TTST on it. Route B is a diagnostic readout, not a reconstruction of Route C's sampler.

Native GDTS instead concatenates the scene semantic image with the first eight trajectory maps, runs its goal U-Net, applies sigmoid to the final channel, and directly calls TTST (`RSJG_JDV2_clean/src/models/model.py:1895-1903`). Native TTST thresholds at 5% of the per-map maximum, draws 1,000 samples with replacement, forms 20 k-means centres, and appends argmax (`sampling_2D_map.py:115-174`). Route A follows that path and converts endpoint pixels through the native ETH homography.

The B0 translation is the exact formal-training mapping from `goal_module.*` to `encoder.map_unet.*` and is loaded strict (`config.py:393-421`). B0 therefore tests the parent weights on EBJD crops; it does not claim equivalence to A, whose checkpoint and native preprocessing are both different.

## Engineering validation

- FP32 CUDA inference for A, B and B0; stored FP32 certified bank for C.
- Actual forward calls: A 253, B 253, B0 253, C 0; optimizer updates: 0.
- Representative run times: A 9.18 s, B 16.85 s, B0 18.52 s; peak allocated CUDA memory 454 MB.
- Independent full replay produced byte-identical agent metrics and raw candidate banks. The committed CSV SHA256 is `251c555d9c8ed97479a7124ef74d199944bb0c0630f967b7437c34e868240579`.
- Coordinate unit tests cover `0 -> -16 m`, `255 -> +16 m`, and the half-pixel crop centre; TTST tests enforce exactly 20 centres plus one excluded argmax.
- Raw candidate banks remain local under `RSJG_EBJD_diagnostic_assets/eth_goal_proposal_three_arm_20261019/`; only small metrics, figures and receipts are versioned.

## Case evidence and boundaries

`case_panels.png` includes maximum B-vs-C improvement, maximum regression, B/C-bad-A-good, and a fixed-seed random example. The plots visibly show B's border-wide proposal spread and C's recovery in some cases. There is no out-of-crop panel because the stratum is empty.

This audit diagnoses endpoint proposal quality only. It does not establish an independent-test score, does not test a migrated A-to-EBJD architecture, and does not show that stronger proposals will automatically improve JADE/JFDE. The next bounded experiment should wire frozen/native A proposals into EBJD with the rest of the pipeline held fixed, then test whether endpoint and joint trajectory metrics improve together.

## Artifacts

- `docs/validation/eth_goal_proposal_three_arm_20261019/summary.json`: complete machine-readable result.
- `docs/validation/eth_goal_proposal_three_arm_20261019/agent_metrics.csv`: all 1,456 route-agent rows (A/B/C/B0).
- `docs/validation/eth_goal_proposal_three_arm_20261019/case_panels.png`: fixed qualitative cases.
- `docs/validation/eth_goal_proposal_three_arm_20261019/REPRODUCIBILITY.json`: exact replay hashes.
- `scripts/run_eth_goal_proposal_probe.py`: frozen execution entry point.
- `tests/test_goal_proposal.py`: coordinate, 20/21 and metric-contract tests.
