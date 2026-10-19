# ETH real-bank endpoint-coverage verification

## Decision

**PARTIAL_RECOVERY.** Fixed-bank endpoint spreading improves all four P20
metrics at epochs 5, 22, and 23 for every tested non-unit factor, but no factor
reaches the matched-GDTS reference. The operation therefore confirms that
insufficient endpoint spread explains a material part of the ETH deficit; it
does not establish that ETH is fixed.

No training or frozen model inference was needed because complete real banks
already existed. Optimizer updates were exactly zero. The original banks,
checkpoints, HOTEL/UNIV configurations, world order, and agent order were not
modified.

## Provenance and scope

The prompt-referenced commit `5fea468a14c36dda0fa81bd02b83cf0ec08ace07`
and patch `RSJG_EBJD_ETH_coverage_repair_2026-10-19.patch` were not present in
the supplied server filesystem or local Git object database. An independent
branch was therefore created from the required base
`a69c3190fb364b3d8fa2bfdef3b03c21c4a2e1eb`; the already-tested frozen-bank
reader was applied from local commit `bd8843b`, and the exact formula from the
prompt was implemented as a read-only bank operation.

All three bank manifests bind to the same real contract:

- ETH JMM validation mirror: 253 scenes and 364 agent occurrences;
- P20, 20 sampler steps, inference seed 2035, FP32 metre coordinates;
- source manifest SHA256
  `03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b`;
- config SHA256
  `936924d8b773f61e9fc5e419fdc0e939bd369beb155c4e74474924ae727214a9`;
- checkpoints epoch 5/22/23 SHA256
  `a5f45606329b80754567cf86381ceff0d19e68c8eaddf999c74f71ef1bdde525`,
  `5d580d560857be3658f74f21a4f60f3d34f1214c0c517f0d29d791852b433151`,
  and `b55e4c7c06e2f41686d5cca665ac1fc17c5b2e7cb7ccb7c4715a17959cb96909`.

The audit re-read the actual manifest, config, checkpoint, and every NPZ scene
and verified each recorded SHA256. Gamma 1 reproduced all four archived metrics
with maximum absolute difference exactly `0.0 m` for all three epochs.

## Full P20 results

Matched GDTS is `0.417053 / 0.580032 / 0.489907 / 0.743412` for
minADE/minFDE/JADE/JFDE.

| Epoch | gamma | minADE | minFDE | JADE | JFDE | Four improve | Reach GDTS |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 5 | 1.00 | 0.582031 | 1.015402 | 0.647612 | 1.121817 | baseline | no |
| 5 | 1.25 | 0.539881 | 0.939630 | 0.613262 | 1.070135 | yes | no |
| 5 | 1.50 | 0.515129 | 0.888931 | 0.593089 | 1.036817 | yes | no |
| 5 | 2.00 | 0.502460 | 0.863858 | 0.586174 | 1.036832 | yes | no |
| 22 | 1.00 | 0.590316 | 1.067265 | 0.639615 | 1.149861 | baseline | no |
| 22 | 1.25 | 0.556671 | 1.005016 | 0.613058 | 1.101359 | yes | no |
| 22 | 1.50 | 0.539828 | 0.966236 | 0.600659 | 1.071139 | yes | no |
| 22 | 2.00 | 0.531983 | 0.951508 | 0.597224 | 1.075846 | yes | no |
| 23 | 1.00 | 0.651152 | 1.262008 | 0.693644 | 1.324343 | baseline | no |
| 23 | 1.25 | 0.604445 | 1.159682 | 0.650604 | 1.230500 | yes | no |
| 23 | 1.50 | 0.574487 | 1.083787 | 0.623409 | 1.163529 | yes | no |
| 23 | 2.00 | 0.542657 | 1.001041 | 0.596369 | 1.099359 | yes | no |

At epoch 5, gamma 1.5 improves the four metrics by 11.49%, 12.46%, 8.42%,
and 7.58%, respectively. It nevertheless remains 23.52%, 53.26%, 21.06%,
and 39.47% above matched GDTS. Gamma 2 gives slightly stronger marginal and
JADE gains, but its JFDE is fractionally worse than gamma 1.5 and its
multi-agent behavior is less acceptable.

## Coverage and invariants

Epoch 5 illustrates the mechanism and its cost:

| gamma | Center bias mean | RMS radius | Pairwise endpoint distance | Mean single-candidate FDE |
|---:|---:|---:|---:|---:|
| 1.00 | 2.126838 | 1.476871 | 1.805575 | 2.513357 |
| 1.25 | 2.126838 | 1.846089 | 2.256969 | 2.692268 |
| 1.50 | 2.126838 | 2.215307 | 2.708363 | 2.890813 |
| 2.00 | 2.126838 | 2.953742 | 3.611151 | 3.333784 |

Candidate-mean drift was at most `1.07e-14 m`, bridge-residual drift at most
`7.11e-15 m`, and the RMS-radius scale error at most `6.67e-16`. Thus the
observed metric changes are from endpoint spread, not center movement, bridge
changes, reordering, or fresh sampling. The unchanged 2.13 m center bias is a
hard limit on what this operation can recover. Gamma 2 increases mean
single-candidate FDE by 32.64% even while improving P20 oracle metrics.

## N=1 versus N>=2

The epoch-5 split contains 183 single-agent scenes and 70 multi-agent scenes
(181 agent occurrences). The multi-agent results expose the main safety
boundary:

| gamma | N>=2 minADE | N>=2 minFDE | N>=2 JADE | N>=2 JFDE |
|---:|---:|---:|---:|---:|
| 1.00 | 0.547322 | 1.037209 | 0.729313 | 1.456400 |
| 1.25 | 0.506146 | 0.946484 | 0.717875 | 1.429036 |
| 1.50 | 0.487727 | 0.890578 | 0.726045 | 1.427690 |
| 2.00 | 0.490198 | 0.870305 | 0.773322 | 1.505709 |

Gamma 2 improves marginal oracle metrics but worsens both joint metrics versus
the original bank. Gamma 1.5 is the most balanced single tested setting: all
four global metrics improve, and both N>=2 joint metrics remain better than
gamma 1. It is still an exploratory inference variant, not a selected formal
model.

## Candidate-budget and case checks

The nested K curves were computed after transforming the same 20 worlds, with
no resampling or per-agent reordering. At epoch 5, gamma 1.5 is worse than the
original at K=1 (`minFDE 2.8861` versus `2.5086`) and remains slightly worse on
the marginal metrics at K=4; its advantage appears only with a larger candidate
budget. This is coverage redistribution, not a uniformly better predictive
distribution. Complete curves for every epoch and gamma are in `K_CURVES.csv`.

Fixed epoch-5 cases include the largest improvement, largest regression,
largest center bias, and largest expansion. The largest improvement reduces
minFDE by 2.709 m. The largest regression increases it by 3.401 m; visual
inspection shows the formerly near-GT candidate being pushed away. The largest
expansion reaches a 9.768 m maximum endpoint radius at gamma 2. These failures
rule out treating gamma 2 as a generally safe repair.

## Artifacts and verification

- [summary and asset bindings](validation/eth_endpoint_coverage_repair_20261019/SUMMARY.json)
- [complete P20 metrics](validation/eth_endpoint_coverage_repair_20261019/FULL_METRICS.csv)
- [N=1/N>=2 strata](validation/eth_endpoint_coverage_repair_20261019/STRATA.csv)
- [all nested K curves](validation/eth_endpoint_coverage_repair_20261019/K_CURVES.csv)
- epoch-specific full JSON files and four PNG/PDF case pairs in the same tree
- fixed-case PNGs: [largest improvement](validation/eth_endpoint_coverage_repair_20261019/cases_epoch005/case_00_largest_improvement.png),
  [largest regression](validation/eth_endpoint_coverage_repair_20261019/cases_epoch005/case_01_largest_regression.png),
  [largest center bias](validation/eth_endpoint_coverage_repair_20261019/cases_epoch005/case_02_largest_center_bias.png),
  and [largest expansion](validation/eth_endpoint_coverage_repair_20261019/cases_epoch005/case_03_largest_expansion.png)
- CPU tests: 46 passed in 8.21 seconds; real bank audits required no GPU and no
  optimizer updates

Raw banks and checkpoints remain only in the host persistent asset/output
directories and are not part of Git.

## One next action

Freeze **gamma 1.5** before looking at further metrics and export/evaluate the
same epoch-5 bank for inference seeds 2035--2039, with one shared setting and no
additional gamma scan. This tests whether the balanced partial recovery is RNG
robust; it remains mirror-validation evidence and must not be described as an
independent test or five training seeds.
