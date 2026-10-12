---
experiment_id: EBJD-B-261012-002
date: 2026-10-13
system: EBJD
type: rollout_weight005_paired_pilot
status: COMPLETED
independent_test: false
---

# EBJD epoch45 rollout-weight 0.05 paired pilot

## Frozen epoch50 results

| model | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| GDTS | 0.134384 ± 0.001351 | 0.191589 ± 0.002872 | 0.208469 ± 0.002341 | 0.357055 ± 0.004002 |
| EBJD_epoch45 | 0.138007 ± 0.001258 | 0.202037 ± 0.002553 | 0.168566 ± 0.001605 | 0.276314 ± 0.003906 |
| control_epoch50 | 0.146598 ± 0.000948 | 0.219778 ± 0.003929 | 0.176315 ± 0.001862 | 0.295207 ± 0.004384 |
| candidate_epoch50 | 0.144859 ± 0.001128 | 0.214739 ± 0.003692 | 0.177966 ± 0.002007 | 0.296955 ± 0.004582 |

Five inference-seed means ± sample SD in metres; training seed is 3101.

## Frozen conclusions

- Engineering contract: **PASSED**; reused control: **PASSED**.
- Candidate−control: ΔminADE `-0.001739`, ΔminFDE `-0.005039`, ΔJADE `+0.001651`, ΔJFDE `+0.001749` m.
- Marginal effect: **PASSED**; joint preservation: **NOT_PASSED**; combined short target: **NOT_PASSED**.
- Coverage: **MIXED_OR_NOT_IMPROVED**; Δmedian-candidate-FDE `+0.033655`, Δp90-candidate-FDE `+0.147848`, Δspread-RMS `+0.090713` m.
- Candidate−epoch45: ΔminADE `+0.006851`, ΔminFDE `+0.012702`, ΔJADE `+0.009400`, ΔJFDE `+0.020641` m; all-four verdict: **NOT_PASSED**.
- Candidate−GDTS: ΔminADE `+0.010475`, ΔminFDE `+0.023150`, ΔJADE `-0.030503`, ΔJFDE `-0.060099` m; original contract: **NOT_PASSED**.
- Further rollout-weight search: **NOT_SUPPORTED**.

Spread is reported together with centre error and fixed-radius hit rates; greater dispersion alone is not treated as improved coverage.
The HOTEL test mirror is not an independent held-out test (`independent_test=false`). No cross-training-seed significance claim is made.

## Bound artifacts

Detailed per-seed, prefix-K, shared-random-subset, strata, event, track and world-permutation tables are stored beside the structured JSON.
