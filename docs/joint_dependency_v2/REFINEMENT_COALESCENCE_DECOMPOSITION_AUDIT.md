# RSJG Joint Dependency V2 — Refinement Coalescence Decomposition Audit

Date: 2026-09-20  
Dataset/split: ETH validation, 139 synchronized windows  
Scope: inference-only/read-only refinement tracing and intervention  
Training and Stage B status: **not started**

## 1. Executive conclusion

The observed `20 -> 12.8` per-agent candidate collapse is created almost
entirely by the first refinement round, and its primary mechanism is
cross-slot alignment of the conditional score geometry rather than random
categorical collision alone.

- In E>0, R2 changes mean unique candidates `20.000 -> 12.064 -> 11.749`.
  Round 1 accounts for 96.2% of the total unique-count loss and 89.4% of the
  total refinement oracle damage.
- Before sampling, the conditional argmax already has only `6.916` unique
  candidates in round 1 and `5.733` in round 2. Categorical sampling produces
  *more* diversity (`12.064/11.749`) than MAP. MAP collapses to
  `6.916 -> 5.186` sampled candidates and catastrophically worsens minFDE to
  0.745838. Stochastic categorical sampling is therefore not the primary
  cause; it partially mitigates an already shared score geometry.
- Conditional distributions are not individually low-entropy: round-1 mean
  entropy is 2.615 nats versus `log(21)=3.045`, and mean maximum probability
  is only 0.163. The failure is better described as similar broad
  distributions sharing the same preferred candidates, not sharp one-hot
  score collapse.
- Exact joint-world collapse is small under R2: unique world signatures are
  `20.000 -> 19.837 -> 19.828`. The synchronous second round creates some
  additional local clustering but is not the main event.
- R1 improves minFDE over R2 by 2.11%, but remains 7.41% worse than GDTS and
  weakens compatibility/relative-motion by about 4%. Removing the second
  round is insufficient.
- The diagnostic Hungarian ASSIGN intervention maintains 20 unique
  candidates in both rounds, keeps the E>0 goal oracle nearly unchanged
  (`0.348167 -> 0.349889 -> 0.350198`), and obtains minFDE 0.383394, 2.79%
  better than same-protocol GDTS. It also improves JFDE, endpoint,
  compatibility, and relative-motion relative to R2; JADE changes by only
  +0.57%.

ASSIGN meets the predefined `STRONG_REFINEMENT_CANDIDATE` diagnostic target,
but it remains a **diagnostic intervention only**. The unique next
recommendation is **C: design and separately review a structured
coverage-constrained refinement**. This report does not authorize changing
the production sampler, retraining, or entering Stage B.

## 2. Scope and invariants

The canonical round-0 initialization is the previously validated
Gumbel-Top-P weighted sampling without replacement. All variants use the
same strict no-z epoch-13 checkpoint, K=21 candidate bank, P=20 slots, M=4
relations, rank-eight energy, unary scores, relation and energy parameters,
frozen GDTS diffusion, cache, world conversion, metrics, validation split,
and five seeds.

Only audit-time control flow differs:

- **R0:** weighted round 0, no refinement;
- **R1:** weighted round 0, current round 1 only;
- **R2:** weighted round 0, both current categorical rounds;
- **MAP:** weighted round 0, independent per-slot argmax in both rounds;
- **ASSIGN:** weighted round 0, deterministic maximum-weight injective
  assignment in both rounds.

ASSIGN solves, independently for each agent, the P-by-K linear assignment
with every slot assigned once and every candidate used at most once. Invalid
candidates receive prohibitive cost. It changes no score, parameter, or
network and is not a production proposal.

No `src/` file was changed. The audit temporarily patches the categorical
primitive inside a context manager and restores both that primitive and the
sampler's canonical two-round setting on exit.

## 3. Provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Strict no-z training commit | `84bab030baed3044826b9b0d89a783a86e324978` |
| Audit implementation/final execution commit | `f552aae` |
| Architecture | `strict_no_z` |
| Checkpoint epoch | 13 |
| Checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Legacy GDTS source SHA256 | `126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950` |
| Cache manifest SHA256 | `2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949` |
| Sampler reference SHA256 | `d64670ac83e62fc17b658819c4505c423732c4a33dab308e0bb04becb0190abd` |
| K / P / M / rank | 21 / 20 / 4 / 8 |
| Split / windows | ETH validation / 139 |
| Evaluation seeds | 2035, 2036, 2037, 2038, 2039 |

Machine-readable result:

`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/refinement_coalescence/results.json`

SHA256:
`08bff37b941113823797377319feb814642c63d0d701b746dfd9cf9a4b2ead3e`.

The 1.1 MiB result contains only aggregated statistics and per-seed metrics;
no dense `[scene,agent,P,K]` tensor was persisted. All recorded values are
finite.

## 4. Baseline reproduction gate

Before R0/R1/MAP/ASSIGN ran, R2 was compared with the saved weighted
two-round result from the sampler-coverage audit. All 35 per-seed prediction
metrics and five core candidate/decomposition quantities were tensor-protocol
identical. The maximum absolute error across 40 checks was `0.0` under a
`1e-10` tolerance.

| Quantity | Previous reference | Reproduced R2 |
|---|---:|---:|
| Initial unique | 20.000000 | 20.000000 |
| Final unique | 12.802717 | 12.802717 |
| Initial oracle | 0.366551 | 0.366551 |
| Final goal oracle | 0.461294 | 0.461294 |
| minFDE | 0.432770 | 0.432770 |
| JADE | 0.412915 | 0.412915 |
| JFDE | 0.704535 | 0.704535 |

An initial gate implementation incorrectly compared the E>0-only round-2
summary with the all-agent final summary. It stopped before running any other
variant, as required. The aggregation was corrected to use each agent's last
valid round; the rerun then passed all 40 checks exactly.

## 5. Exact RNG control

An unchanged iid-with-replacement control run captures Python, NumPy, CPU
Torch, every CUDA-device RNG state, and cuDNN flags at these per-window
boundaries:

```text
pre-initial -> pre-round1 -> pre-round2 -> pre-diffusion -> window-end
```

Every diagnostic starts from the control's pre-initial state. Weighted round
0 uses an independent explicit per-window CUDA generator. R1/R2 categorical
rounds start from their paired control boundary. MAP/ASSIGN are deterministic
but still restore the same round boundary. All variants restore the identical
pre-diffusion state, and the complete window-end state must match the control.

| Variant | pre-round1 | pre-round2 | pre-diffusion | window-end |
|---|---:|---:|---:|---:|
| R0 | N/A | N/A | 695/695 | 695/695 |
| R1 | 460/460 | N/A | 695/695 | 695/695 |
| R2 | 460/460 | 460/460 | 695/695 | 695/695 |
| MAP | 460/460 | 460/460 | 695/695 | 695/695 |
| ASSIGN | 460/460 | 460/460 | 695/695 | 695/695 |

There are 92 E>0 windows per seed, hence 460 refinement-boundary checks, and
139 total windows per seed, hence 695 diffusion/window checks. Different
random-number consumption cannot explain the metric differences.

## 6. Round-by-round candidate coverage

The main refinement conclusion must come from E>0. Values below aggregate
1,605 agent observations across five seeds.

| R2 round | unique | duplicate | frozen top-1/3/5 | unary top-1/3/5 | GT-oracle present | mean frozen/unary rank |
|---|---:|---:|---|---|---:|---|
| Round 0 | 20.000 | 0.000 | 98.44/100/100% | 98.82/100/100% | 97.45% | 10.782 / 10.696 |
| Round 1 | 12.064 | 7.936 | 73.77/96.95/99.88% | 64.55/96.14/99.56% | 70.22% | 9.539 / 9.751 |
| Round 2 | 11.749 | 8.251 | 73.77/97.07/99.88% | 63.05/94.39/98.88% | 67.79% | 9.325 / 9.814 |

Round 1 shifts the average candidate rank upward while destroying set
coverage. This is the same rank-versus-oracle conflict observed previously:
many slots move toward high-score candidates, but the finite sample set loses
the geometrically useful tail.

Round deltas in E>0:

| Transition | unique delta | oracle delta | frozen rank-1 delta | GT-oracle-present delta |
|---|---:|---:|---:|---:|
| Round 0 -> 1 | -7.936 | +0.097125 | -24.67 pp | -27.23 pp |
| Round 1 -> 2 | -0.315 | +0.011491 | 0.00 pp | -2.43 pp |
| Round 0 -> 2 | -8.251 | +0.108615 | -24.67 pp | -29.66 pp |

Round 1 therefore causes 96.2% of the unique loss and 89.4% of the oracle
damage. Round 2 is a smaller continuation, not the root cause.

## 7. Geometric oracle and complete error decomposition

### 7.1 Overall

E=0 agents have no refinement; their final goal equals round 0.

| Variant | bank | round 0 | round 1 | round 2 | final goal | trajectory minFDE | diffusion gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.361863 | 0.366551 | N/A | N/A | 0.366551 | **0.378607** | +0.012056 |
| R1 | 0.361863 | 0.366551 | 0.445291 | N/A | 0.451271 | 0.423620 | -0.027651 |
| R2 | 0.361863 | 0.366551 | 0.445291 | 0.456782 | 0.461294 | 0.432770 | -0.028524 |
| MAP | 0.361863 | 0.366551 | 0.774193 | 0.889034 | 0.838341 | 0.745838 | -0.092503 |
| ASSIGN | 0.361863 | 0.366551 | 0.349889 | 0.350198 | **0.368323** | 0.383394 | +0.015071 |

### 7.2 E>0

| Variant | bank | round 0 | round 1 | round 2/final | trajectory minFDE | round-1 delta | round-2 delta | diffusion gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.343584 | 0.348167 | N/A | 0.348167 | **0.356069** | N/A | N/A | +0.007903 |
| R1 | 0.343584 | 0.348167 | 0.445291 | 0.445291 | 0.407673 | +0.097125 | N/A | -0.037618 |
| R2 | 0.343584 | 0.348167 | 0.445291 | 0.456782 | 0.418163 | +0.097125 | +0.011491 | -0.038619 |
| MAP | 0.343584 | 0.348167 | 0.774193 | 0.889034 | 0.777069 | +0.426027 | +0.114841 | -0.111965 |
| ASSIGN | 0.343584 | 0.348167 | 0.349889 | **0.350198** | 0.361557 | +0.001722 | +0.000309 | +0.011359 |

ASSIGN removes almost all refinement-induced oracle damage. Its positive
diffusion gap is visible because the selected goals are now strong enough
that diffusion no longer compensates for a poor goal set; nevertheless its
trajectory minFDE remains better than GDTS.

### 7.3 E=0 control

All five variants are exactly identical in E=0 because no refinement call is
made:

| bank | round 0/final goal | trajectory minFDE | diffusion gap |
|---:|---:|---:|---:|
| 0.486705 | 0.492113 | 0.532534 | +0.040421 |

All cross-variant differences therefore come from E>0 refinement, not
diffusion or initial allocation.

## 8. Conditional score geometry

R2's actual source equation is:

```text
conditional_score[i,s,k]
  = unary_score[i,k]
    - sum(pair_energy contributions from incident edges at world slot s)
```

No energy coefficient or score was altered.

### 8.1 Distribution statistics

| R2 statistic | Round 1 | Round 2 |
|---|---:|---:|
| entropy mean +/- std | 2.615 +/- 0.489 | 2.552 +/- 0.567 |
| entropy p10 / p90 / p95 | 1.996 / 3.040 / 3.042 | 1.816 / 3.040 / 3.041 |
| max probability mean | 0.163 | 0.178 |
| max probability median | 0.120 | 0.126 |
| max probability p90 / p95 | 0.320 / 0.429 | 0.366 / 0.491 |
| unique conditional argmax | **6.916** | **5.733** |
| unique categorical sample | 12.064 | 11.749 |

With a maximum entropy of 3.045 nats, these are broad distributions. Their
top choices nevertheless overlap strongly across slots. The categorical
draw adds diversity relative to argmax; it does not create the underlying
argmax collision.

### 8.2 Cross-slot similarity

| R2 statistic | Round 1 | Round 2 | Change |
|---|---:|---:|---:|
| pairwise JS mean | 0.11552 | 0.11612 | +0.5% |
| pairwise JS median | 0.02182 | 0.01570 | -28.0% |
| pairwise JS p10 | 0.000019 | 0 | lower |
| top-3 Jaccard | 0.38169 | 0.42356 | +11.0% |
| top-5 Jaccard | 0.46345 | 0.50059 | +8.0% |

The typical/central pair becomes more similar and top-k candidate sets overlap
more in round 2. The mean JS does not fall because a minority of slot pairs
remain well separated. Thus there is local positive feedback, not uniform
distribution collapse.

## 9. Unary versus pair-energy decomposition

Unary is identical across slots for one agent, so unary-only top-1 diversity
is exactly 1. Pair energy raises full conditional top-1 diversity to 6.916 in
round 1 and 5.733 in round 2. It therefore creates meaningful slot
differentiation rather than simply forcing every slot to the unary mode.

| R2 statistic | Round 1 | Round 2 |
|---|---:|---:|
| pair-energy variance across slots | 1.866 | 2.638 |
| unary/conditional rank Spearman | 0.545 | 0.542 |
| conditional top-1 differs from unary | 79.30% | 79.76% |

Pair-energy variation increases in round 2, while rank correlation stays
moderate. This rules out the simplistic claim that pair energy merely erases
all slot context. The failure is that the resulting slot-specific scores are
not diverse enough to support independent updates without repeated IDs.

## 10. Argmax collision versus sampled collision

| Variant | Round-1 unique | Round-2 unique | Final E>0 oracle | Overall minFDE |
|---|---:|---:|---:|---:|
| categorical R2 | 12.064 | 11.749 | 0.456782 | 0.432770 |
| independent MAP | 6.916 | 5.186 | 0.889034 | 0.745838 |
| injective ASSIGN | 20.000 | 20.000 | 0.350198 | 0.383394 |

If categorical collision were primary, MAP would have retained slot-specific
argmax diversity. Instead MAP is much worse. The evidence supports:

1. **Primary:** similar conditional score ordering across slots (Mechanism B);
2. **Secondary:** independent with-replacement draws still create avoidable
   repeated IDs (Mechanism A);
3. **Minor continuation:** round-2 context feedback increases local overlap
   (Mechanism C), but it accounts for little of the total damage.

## 11. World-slot diversity

| Variant/round | unique joint worlds | duplicate worlds | mean Hamming | min Hamming | identical-pair fraction |
|---|---:|---:|---:|---:|---:|
| R2 round 0 | 20.000 | 0 | 2.647 | 2.647 | 0 |
| R2 round 1 | 19.837 | 0.163 | 3.274 | 1.489 | 0.00089 |
| R2 round 2 | 19.828 | 0.172 | 3.251 | 1.376 | 0.00092 |
| MAP round 1 | 16.085 | 3.915 | 2.421 | 0.435 | 0.05915 |
| MAP round 2 | 12.433 | 7.567 | 2.057 | 0.100 | 0.17398 |
| ASSIGN round 1 | 20.000 | 0 | 3.489 | 3.489 | 0 |
| ASSIGN round 2 | 20.000 | 0 | 3.489 | 3.489 | 0 |

R2's per-agent coalescence does not become wholesale exact-world collapse:
different agents' repeated IDs combine into almost 20 unique joint worlds.
The declining minimum Hamming distance shows local world clusters, while the
mean Hamming remains high. MAP exposes what genuine world-level convergence
would look like and is clearly pathological.

## 12. Slot transition multiplicity

For each target candidate, convergence multiplicity counts distinct source
candidates whose slots map into that target.

| R2 transition | maximum multiplicity | mean multiplicity | targets >=2 | >=3 | >=5 |
|---|---:|---:|---:|---:|---:|
| Round 0 -> 1 | 3.626 | 1.692 | 45.04% | 16.60% | 1.57% |
| Round 1 -> 2 | 3.477 | 1.663 | 44.01% | 15.66% | 1.44% |
| Round 0 -> 2 | 3.824 | 1.749 | 45.84% | 18.37% | 2.47% |

The first round already merges several distinct initial hypotheses into each
surviving candidate. Round 2 continues the same behavior but contributes much
less additional set-level loss.

## 13. Assignment score quality

Mean conditional score per assigned slot:

| Variant | Round 1 selected | Round 1 independent MAP | regret | Round 2 selected | Round 2 independent MAP | regret |
|---|---:|---:|---:|---:|---:|---:|
| categorical R2 | 4.1370 | 4.6715 | 0.5345 | 4.7631 | 5.2874 | 0.5243 |
| MAP | 4.6715 | 4.6715 | 0 | 5.6563 | 5.6563 | 0 |
| ASSIGN | **4.1672** | 4.6715 | 0.5042 | 4.6105 | 5.0811 | 0.4706 |

Round 1 has exactly the same score matrix for R2, MAP, and ASSIGN because all
start from the same round-0 worlds. ASSIGN's injective solution has a 10.8%
lower mean score than independent MAP, but a 0.73% *higher* realized score
than categorical R2 while retaining all 20 candidates. Thus the current
categorical update sacrifices coverage without obtaining a realized
conditional-score gain over the structured assignment. Round-2 score
matrices differ because their round-1 worlds differ, so cross-variant round-2
scores are descriptive rather than directly paired objectives.

## 14. R0/R1/R2/MAP/ASSIGN prediction results

All metrics are lower-is-better; values are five-seed mean +/- population
standard deviation.

| Variant | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| R0 | **0.275695 +/- 0.001813** | **0.378607 +/- 0.004341** | 0.436121 +/- 0.007826 | 0.745299 +/- 0.012407 |
| R1 | 0.286351 +/- 0.002936 | 0.423620 +/- 0.005082 | 0.415186 +/- 0.002395 | 0.707106 +/- 0.003445 |
| R2 | 0.286761 +/- 0.004787 | 0.432770 +/- 0.011473 | **0.412915 +/- 0.002887** | 0.704535 +/- 0.008531 |
| MAP | 0.396791 +/- 0.006024 | 0.745838 +/- 0.011157 | 0.461243 +/- 0.002235 | 0.836305 +/- 0.010807 |
| ASSIGN | 0.277473 +/- 0.002062 | 0.383394 +/- 0.004226 | 0.415280 +/- 0.007885 | **0.699226 +/- 0.009338** |

| Variant | endpoint | compatibility | relative motion |
|---|---:|---:|---:|
| R0 | 0.731292 +/- 0.012671 | 0.606515 +/- 0.015397 | 0.350859 +/- 0.003254 |
| R1 | 0.694801 +/- 0.003337 | 0.522073 +/- 0.007882 | 0.316751 +/- 0.005067 |
| R2 | 0.693684 +/- 0.011990 | 0.498550 +/- 0.012808 | 0.304229 +/- 0.003892 |
| MAP | 0.849622 +/- 0.010617 | 0.671167 +/- 0.009217 | 0.347620 +/- 0.003767 |
| ASSIGN | **0.682405 +/- 0.011575** | **0.481323 +/- 0.014784** | **0.298115 +/- 0.003290** |

R0 has the best marginal values but loses substantial joint coherence versus
R2. ASSIGN nearly retains R0's marginal recovery and matches or improves the
R2 joint metrics.

## 15. Per-seed prediction metrics

| Variant | Seed | minADE | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 2035 | 0.276967 | 0.385808 | 0.423746 | 0.724427 | 0.712366 | 0.605823 | 0.347458 |
| R0 | 2036 | 0.276269 | 0.375838 | 0.446264 | 0.753618 | 0.741975 | 0.626963 | 0.354653 |
| R0 | 2037 | 0.278023 | 0.374455 | 0.442139 | 0.757189 | 0.743724 | 0.593933 | 0.351883 |
| R0 | 2038 | 0.273191 | 0.375515 | 0.436078 | 0.753481 | 0.738340 | 0.585859 | 0.353681 |
| R0 | 2039 | 0.274023 | 0.381419 | 0.432377 | 0.737779 | 0.720056 | 0.620000 | 0.346621 |
| R1 | 2035 | 0.286642 | 0.424343 | 0.416942 | 0.701789 | 0.691213 | 0.512351 | 0.319984 |
| R1 | 2036 | 0.290909 | 0.429041 | 0.417078 | 0.712552 | 0.700452 | 0.524617 | 0.307035 |
| R1 | 2037 | 0.281830 | 0.414551 | 0.414727 | 0.707761 | 0.691625 | 0.529755 | 0.320545 |
| R1 | 2038 | 0.285253 | 0.422565 | 0.410700 | 0.706128 | 0.695060 | 0.530495 | 0.319756 |
| R1 | 2039 | 0.287119 | 0.427601 | 0.416482 | 0.707302 | 0.695655 | 0.513145 | 0.316435 |
| R2 | 2035 | 0.277644 | 0.413022 | 0.410207 | 0.711320 | 0.703014 | 0.502733 | 0.303111 |
| R2 | 2036 | 0.287931 | 0.445093 | 0.417578 | 0.692262 | 0.684657 | 0.488958 | 0.302966 |
| R2 | 2037 | 0.291128 | 0.437114 | 0.414978 | 0.714596 | 0.709328 | 0.481415 | 0.300462 |
| R2 | 2038 | 0.287035 | 0.427505 | 0.411267 | 0.696979 | 0.676268 | 0.500795 | 0.302844 |
| R2 | 2039 | 0.290070 | 0.441119 | 0.410546 | 0.707515 | 0.695153 | 0.518851 | 0.311764 |
| MAP | 2035 | 0.405329 | 0.759806 | 0.460044 | 0.846929 | 0.862217 | 0.686260 | 0.348519 |
| MAP | 2036 | 0.396346 | 0.733369 | 0.464147 | 0.818373 | 0.835558 | 0.671849 | 0.347714 |
| MAP | 2037 | 0.400688 | 0.757029 | 0.463496 | 0.840976 | 0.855357 | 0.673896 | 0.341680 |
| MAP | 2038 | 0.387531 | 0.733748 | 0.458215 | 0.829741 | 0.838401 | 0.658926 | 0.353457 |
| MAP | 2039 | 0.394065 | 0.745239 | 0.460312 | 0.845505 | 0.856577 | 0.664906 | 0.346731 |
| ASSIGN | 2035 | 0.275171 | 0.385707 | 0.406853 | 0.695738 | 0.674951 | 0.478297 | 0.292438 |
| ASSIGN | 2036 | 0.277328 | 0.375410 | 0.423400 | 0.703824 | 0.695315 | 0.493069 | 0.299906 |
| ASSIGN | 2037 | 0.281098 | 0.387147 | 0.424214 | 0.713960 | 0.693715 | 0.492077 | 0.301867 |
| ASSIGN | 2038 | 0.277910 | 0.382890 | 0.416299 | 0.696785 | 0.683487 | 0.453692 | 0.299699 |
| ASSIGN | 2039 | 0.275858 | 0.385815 | 0.405634 | 0.685822 | 0.664556 | 0.489482 | 0.296663 |

These are five stochastic inference repeats of one trained checkpoint, not
five independent training seeds.

## 16. E=0 and E>0 prediction metrics

| Variant / stratum | minADE | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 / E=0 | 0.364403 | 0.532534 | 0.364403 | 0.532534 | 0.500359 | 0 | 0 |
| R1/R2/MAP/ASSIGN / E=0 | 0.364403 | 0.532534 | 0.364403 | 0.532534 | 0.500359 | 0 | 0 |
| R0 / E>0 | **0.262706** | **0.356069** | 0.472759 | 0.853994 | 0.849269 | 0.916366 | 0.530103 |
| R1 / E>0 | 0.274922 | 0.407673 | 0.441129 | 0.796290 | 0.794136 | 0.788784 | 0.478570 |
| R2 / E>0 | 0.275393 | 0.418163 | **0.437699** | 0.792404 | 0.792448 | 0.753244 | 0.459651 |
| MAP / E>0 | 0.401534 | 0.777069 | 0.510715 | 0.991492 | 1.028050 | 1.014046 | 0.525209 |
| ASSIGN / E>0 | 0.264745 | 0.361557 | 0.441272 | **0.784384** | **0.775407** | **0.727217** | **0.450412** |

E=0 equality is an additional paired-diffusion control. ASSIGN's improvement
comes entirely from E>0 refinement and is not an aggregate artifact driven by
single-agent windows.

## 17. Degree and agent-count stratification

### 17.1 R2 coalescence by degree

| Degree | unique R0 -> R1 -> R2 | oracle R0 -> R1 -> R2 |
|---|---|---|
| degree=0 | 20 -> 12.953 -> 12.907 | 0.357552 -> 0.200800 -> 0.186915 |
| degree=1 | 20 -> 12.646 -> 12.564 | 0.293864 -> 0.414655 -> 0.410032 |
| degree=2-3 | 20 -> 11.735 -> 11.271 | 0.394726 -> 0.494529 -> 0.510400 |
| degree>=4 | 20 -> 11.485 -> 10.991 | 0.423717 -> 0.505697 -> 0.538827 |

Degree-zero agents are still resampled by the global synchronous round and
can improve by chance/unary resampling. For degree>=1, especially high
degree, coverage loss and oracle damage increase with coupling complexity.
Round-0-to-2 maximum convergence multiplicity rises from 3.20 at degree zero
to 4.11 at degree>=4.

### 17.2 R2 coalescence by agent count

| Agent count | unique R0 -> R1 -> R2 | oracle R0 -> R1 -> R2 |
|---|---|---|
| N=1 | 20 -> N/A -> N/A | 0.492113 -> N/A -> N/A |
| N=2 | 20 -> 12.250 -> 11.875 | 0.435329 -> 0.554859 -> 0.574600 |
| N=3-4 | 20 -> 12.250 -> 12.029 | 0.287404 -> 0.391638 -> 0.390098 |
| N=5-8 | 20 -> 11.829 -> 11.361 | 0.401423 -> 0.483404 -> 0.521756 |
| N>=9 | 20 -> 10.867 -> 10.456 | 0.516527 -> 0.562593 -> 0.562198 |

Larger scenes show stronger unique-count collapse. This supports a social
conditional coupling contribution without implying that the learned pair
energy is intrinsically invalid.

### 17.3 Final trajectory minFDE by degree

| Degree | R0 | R1 | R2 | ASSIGN |
|---|---:|---:|---:|---:|
| degree=0 | 0.392301 | 0.400789 | 0.399798 | **0.390368** |
| degree=1 | **0.306406** | 0.389210 | 0.383412 | 0.319963 |
| degree=2-3 | **0.403239** | 0.457587 | 0.469220 | 0.411847 |
| degree>=4 | 0.419653 | 0.443106 | 0.474983 | **0.416021** |

ASSIGN recovers most of R0's marginal advantage while preserving joint
refinement behavior. The largest R2-to-ASSIGN recovery is not restricted to
degree zero.

## 18. Relative comparisons and acceptance target

Negative is lower/better.

| Variant vs GDTS | minADE | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | -3.72% | -4.01% | -6.81% | -8.60% | -8.55% | -10.08% | -7.78% |
| R1 | +0.01% | +7.41% | -11.28% | -13.29% | -13.11% | -22.60% | -16.75% |
| R2 | +0.15% | +9.73% | -11.77% | -13.60% | -13.25% | -26.09% | -20.04% |
| MAP | +38.58% | +89.10% | -1.44% | +2.56% | +6.25% | -0.50% | -8.63% |
| **ASSIGN** | **-3.09%** | **-2.79%** | **-11.26%** | **-14.25%** | **-14.66%** | **-28.64%** | **-21.65%** |

| Variant vs R2 | minADE | minFDE | JADE | JFDE | endpoint | compatibility | relative motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | -3.86% | -12.52% | +5.62% | +5.79% | +5.42% | +21.66% | +15.33% |
| R1 | -0.14% | -2.11% | +0.55% | +0.37% | +0.16% | +4.72% | +4.12% |
| MAP | +38.37% | +72.34% | +11.70% | +18.70% | +22.48% | +34.62% | +14.26% |
| **ASSIGN** | **-3.24%** | **-11.41%** | **+0.57%** | **-0.75%** | **-1.63%** | **-3.46%** | **-2.01%** |

Against the formal strict no-z iid baseline, ASSIGN improves minFDE by
12.08%, JADE by 0.46%, JFDE by 4.09%, endpoint by 7.66%, compatibility by
2.86%, and relative motion by 1.18%. It recovers 126.43% of the original
iid-versus-GDTS minFDE gap and no principal joint metric worsens by 2%.
Therefore it qualifies as `STRONG_REFINEMENT_CANDIDATE` under the predefined
diagnostic rule. This label supports the mechanism conclusion, not automatic
production adoption.

## 19. Marginal-versus-joint trade-off

R0 proves that preserving all initial candidates restores marginal quality,
but its joint coherence is materially weaker than R2. R2 proves that social
refinement improves joint coherence, but its independent slot updates erase
marginal coverage. R1 lies between them but does not solve the trade-off.

ASSIGN is the first diagnostic in this chain to retain R0-like marginal
coverage and R2-like joint coherence simultaneously. It does so without
retraining or changing conditional scores, identifying the update constraint,
not model capacity, as the actionable mechanism.

## 20. Mechanism attribution

| Mechanism | Attribution | Evidence |
|---|---|---|
| A. Independent sampling collision | Secondary | Categorical draws still produce about eight duplicate slots, but they are substantially more diverse than MAP. |
| B. Cross-slot score alignment | **Primary** | Only 6.916/5.733 unique argmax candidates exist before sampling; median JS is low and top-k overlap grows. |
| C. Synchronous context feedback | Present but secondary | Round 2 lowers unique argmax and minimum Hamming, but causes only 3.8% of total unique loss and 10.6% of oracle damage; exact worlds stay almost fully unique. |
| Pair energy as a whole | Not condemned | It changes about 79% of conditional top-1 choices away from unary and creates 5–7 distinct argmax modes; structured use of the same scores performs strongly. |
| Cross-slot independent update | **Key actionable mechanism** | ASSIGN with unchanged scores keeps 20 candidates and improves both marginal and joint metrics. |

## 21. Decision logic

- **Case 1:** partly true—R1 is better than R2 marginally, but round 1 already
  creates almost all collapse, so reducing only the second round is not enough.
- **Case 2:** strongly true—MAP collapses much more than R2; categorical
  randomness is not the primary problem.
- **Case 3:** strongly true—MAP collapses, while ASSIGN restores marginal and
  joint metrics using identical scores and no retraining.
- **Case 4:** false for this diagnostic—ASSIGN does not expose a harmful
  marginal-versus-joint trade-off in the reported metrics.
- **Case 5:** partly true—R0 is marginally excellent and still beats GDTS on
  joint metrics, but it loses substantial coherence relative to R2.
- **Case 6:** true—refinement adds joint value while independent update harms
  marginal coverage; a sample-set-level constraint is indicated.

Unique recommendation: **C — design structured coverage-constrained
refinement**. This should be a separate, explicitly reviewed method-design
step. It should preserve the learned conditional scores and investigate a
stochastic/deployment-compatible constrained update; this audit does not
authorize Hungarian as the final production policy.

## 22. Answers to the required questions

**Q1. Which round causes the 20 -> 12.8 collapse?**  Round 1. It accounts
for 96.2% of unique loss and 89.4% of oracle damage. Round 2 is secondary.

**Q2. Are conditional argmax candidates already repeated before sampling?**
Yes. Only 6.916 unique argmax candidates exist in round 1 and 5.733 in round
2, despite P=20.

**Q3. Sampling collision or score concentration?**  Primarily cross-slot
score alignment, with independent sampling collision secondary. The scores
are broad rather than one-hot, but share preferred candidates across slots.

**Q4. Do distributions become more similar from round 1 to round 2?**
Typically yes: median JS falls 28%, top-3 overlap rises 11%, and top-5 overlap
rises 8%. Mean JS is flat/slightly higher, so the effect is local rather than
uniform.

**Q5. Is there positive-feedback synchronous context collapse?**  A weak
local form exists, but not global world collapse. Round 2 reduces argmax
diversity and minimum Hamming, while exact unique worlds remain about 19.83.

**Q6. Does one round retain joint gain while reducing marginal loss?**  It
retains JADE/JFDE/endpoint within 0.55% of R2 and improves minFDE 2.11%, but
remains 7.41% worse than GDTS and weakens compatibility/relative motion by
about 4%. It is not sufficient.

**Q7. What does MAP show?**  Removing categorical randomness makes collapse
far worse, proving stochastic categorical sampling is not the main cause and
currently provides useful exploration around aligned score distributions.

**Q8. Can ASSIGN preserve coverage and joint metrics?**  Yes. It keeps 20
unique candidates in both rounds, reaches minFDE 0.383394, and matches or
improves R2 on every principal joint metric except a negligible +0.57% JADE.

**Q9. Is ASSIGN's gain explained by coverage with only modest score cost?**
Yes. Against independent MAP it pays about 0.50 conditional-score units per
slot in round 1 (10.8%), but it scores 0.73% higher than the actual
categorical realization while retaining full coverage. No retraining or score
change contributes to the gain.

**Q10. Does current refinement trade large coverage for a small score gain?**
It is worse than that at round 1: it loses about eight unique candidates and
does not realize a score gain over ASSIGN. Independent MAP can raise score,
but its prediction metrics collapse, demonstrating score/coverage
misalignment at the sample-set level.

**Q11. What is the one next direction?**  **C: design structured
coverage-constrained refinement.** Do not merely remove round 2, replace
categorical with MAP, cancel refinement, retrain, or enter Stage B.

## 23. Tests and engineering validation

The audit tests cover deterministic MAP, injective/mask-valid ASSIGN,
P>K rejection, non-finite rejection, exact tiny brute-force assignment
optimality, agent isolation, R0/R1/R2 call counts and shapes, paired
pre-round/pre-diffusion RNG, E=0/single-agent behavior, world-scene isolation,
primitive restoration, and real-CUDA assignment.

Validation before the audit:

- real-GPU full suite: **272 passed**, 12 warnings;
- targeted CPU suite: **11 passed, 1 CUDA-only skipped**;
- `python -m compileall -q .`: passed;
- `git diff --check`: passed;
- existing legacy/all-off and sampler regression tests remained green.

Final validation was repeated after generating this report; results are
recorded in the completion handoff.

## 24. Status

`AUDIT_COMPLETE_NO_TRAINING_NO_STAGE_B`

The production sampler, model, loss, checkpoint, and training path remain
unchanged. Work stops here pending human review.

