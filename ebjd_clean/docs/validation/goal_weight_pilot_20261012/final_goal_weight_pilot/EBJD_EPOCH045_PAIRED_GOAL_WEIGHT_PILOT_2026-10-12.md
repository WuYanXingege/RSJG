---
experiment_id: EBJD-B-261012-001
date: 2026-10-12
system: EBJD
type: goal_weight_paired_pilot
status: COMPLETED
independent_test: false
---

# EBJD epoch 45 goal-weight paired pilot

## Frozen epoch-50 results

| model | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| GDTS | 0.134384 ± 0.001351 | 0.191589 ± 0.002872 | 0.208469 ± 0.002341 | 0.357055 ± 0.004002 |
| EBJD_epoch45 | 0.138007 ± 0.001258 | 0.202037 ± 0.002553 | 0.168566 ± 0.001605 | 0.276314 ± 0.003906 |
| control_epoch50 | 0.146598 ± 0.000948 | 0.219778 ± 0.003929 | 0.176315 ± 0.001862 | 0.295207 ± 0.004384 |
| candidate_epoch50 | 0.147188 ± 0.001193 | 0.222277 ± 0.003629 | 0.176208 ± 0.002227 | 0.296438 ± 0.004502 |

Values are five inference-seed means ± sample SD in metres; training seed is 3101.

## Preregistered conclusions

- Engineering contract: **PASSED**.
- Candidate−control ΔminADE: `+0.000590 m`; ΔminFDE: `+0.002499 m`.
- Candidate−control ΔJADE: `-0.000108 m`; ΔJFDE: `+0.001231 m`.
- Short target-effect verdict: **NOT_PASSED**.
- Candidate relative to GDTS contract: **NOT_PASSED** (ΔminADE `+0.012804`, ΔminFDE `+0.030687`, ΔJADE `-0.032262`, ΔJFDE `-0.060617` m).
- Candidate−control heavy-tail mean ΔminFDE: `+0.002499 m`.

The HOTEL test mirror is not an independent held-out test (`independent_test=false`). No cross-training-seed significance claim is made.

## Bound artifacts

- Result JSON SHA256: `see FINAL_RESULT_RECEIPT.json` (replaced in receipt after write).
- Detailed CSV tables and bank receipts are stored beside the structured result.
