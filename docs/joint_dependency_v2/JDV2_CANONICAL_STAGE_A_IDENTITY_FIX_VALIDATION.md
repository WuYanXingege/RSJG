# JDV2 Canonical Stage-A Identity Fix Validation

## 1. Final result

**Primary state: `CANONICAL_STAGE_A_LITERAL_IDENTITY_CONFIRMED`**

The mixed-precision identity defect documented at source commit `4abc1a0`
has been fixed by source commit
`d1a1382ae324b9096ac3a6d8de812ffbbb8d9780`.

Under paired ETH-validation execution, canonical Stage-A inference now equals
the literal base diffusion route tensor-exactly in both CUDA/BF16 and
CUDA/FP32:

```text
A canonical Stage-A == B literal Stage-A == C Stage-B base reference
```

This equality holds over all 139 windows and all 176,640 trajectory values,
including E=0, mixed active/inactive, and all-active windows.  The fix did not
modify either checkpoint, the corrector architecture, the sampler, the loss,
or any training configuration.  No retraining was performed.

## 2. Critical semantic boundary

The route names used in this report have deliberately narrow meanings:

| Path | Checkpoint | Correction arithmetic | Meaning |
|---|---|---|---|
| A | Stage-A epoch 13 | canonical config, with the new exact-zero inference bypass | canonical Stage-A |
| B | same Stage-A checkpoint | explicitly off | literal Stage-A base route |
| C | V2-A epoch 5 | explicitly off | Stage-B base reference only |
| D | same V2-A checkpoint | on | actual deployed V2-A correction path |

In particular, **C is not V2-A with correction enabled**.  A=B=C certifies
the Stage-B base reference used for paired comparisons; only D represents the
learned V2-A trajectory correction.

## 3. Confirmed pre-fix root cause

The historical report
`JDV2_CANONICAL_STAGE_A_ROUTE_CERTIFICATION.md` remains unchanged.  Its direct
operator trace established:

1. first dtype divergence at `epsilon_base + delta_epsilon`;
2. `epsilon_base` was BF16 and the tensor-exact-zero residual was FP32;
3. the addition promoted the result to FP32 even though values were unchanged;
4. the first numerical divergence appeared in the subsequent DDIM
   `second_term` (`48/48` values, maximum absolute difference
   `0.0004484504461288452`);
5. the drift propagated to 139,675 of 176,640 final trajectory values across
   all 92 E>0 windows.

Therefore P02 is **CONFIRMED**, not merely a likely downstream numerical
effect.  The small aggregate metric differences in the pre-fix audit neither
weaken the identity contract nor explain the previously observed full A/B
performance gap.

## 4. Minimal production change

The fix is confined to `src/models/model.py`:

- `_jdv2_output_layer_is_exact_zero` checks the corrector's final weight and
  bias with exact `count_nonzero == 0` tests; no tolerance is used.
- During evaluation only, and only when `training_stage == joint_goal`, an
  exact-zero final output layer bypasses dependency-correction arithmetic and
  executes the literal base DDIM route.
- The bypass is independent of residual-projection mode.
- A nonzero corrector continues through the existing active-agent correction
  route.
- Stage-B `joint_trajectory` training does not use the new Stage-A bypass, so
  zero-initialized output-layer gradients remain connected.

The implementation is at `src/models/model.py:97` and
`src/models/model.py:3791-3808`.  Diff comparison from `4abc1a0` to `d1a1382`
confirms no change to:

- `dependency_corrector.py`;
- any JDV2 config;
- `joint_sampler.py`.

No additional RNG draw, denoiser forward, or corrector forward was added.

## 5. Unit and repository validation

The new contracts in
`tests/test_jdv2_canonical_stage_a_identity_fix.py` cover:

- CPU/FP32 mixed and all-active exact identity;
- CUDA/BF16 mixed and all-active exact identity;
- exact-zero checkpoint save/reload;
- preservation of nonzero active-agent correction and inactive-agent
  identity;
- preservation of the Stage-B training route and RNG state.

Validation results at `d1a1382`:

```text
python -m compileall -q .        PASS
pytest -q                       504 passed, 12 warnings
git diff --check                PASS
```

The checkpoint-reload test verifies that an exact-zero output head remains
eligible for the literal bypass after serialization and reload.

## 6. Full A/B/C recertification

Protocol:

- ETH validation: 139 windows;
- seed: 2035;
- shared candidate IDs, goals, contexts, relation state, `x_T`, and DDIM RNG;
- 47 E=0 windows, 30 mixed windows, 62 all-active windows;
- 20/20 candidate coverage.

### CUDA/BF16

| Comparison | Scope | Different / compared elements | Nonidentical windows | Max abs diff |
|---|---|---:|---:|---:|
| A vs B | all | 0 / 176,640 | 0 | 0 |
| A vs B | active agents | 0 / 139,680 | 0 | 0 |
| A vs B | inactive agents | 0 / 36,960 | 0 | 0 |
| A vs B | E=0 | 0 / 22,560 | 0 | 0 |
| A vs B | mixed | 0 / 43,200 | 0 | 0 |
| A vs B | all-active | 0 / 110,880 | 0 | 0 |
| B vs C | all | 0 / 176,640 | 0 | 0 |

### CUDA/FP32

Every entry in the table above is also exactly zero in FP32.  Both runs report
null first-dtype-divergence and null first-value-divergence fields.

All pairing contracts passed in both precisions: batch/input identity,
candidate IDs, joint goals, contexts, shared relation state, post-encode RNG,
post-diffusion RNG, and 20/20 coverage.

## 7. Immutable provenance

| Artifact | SHA256 |
|---|---|
| Stage-A epoch-13 checkpoint | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Stage-B V2-A epoch-5 checkpoint | `e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73` |
| Stage-A frozen config | `bf5d4a692b0a9e841de8523f2780013147ec3e9bcb78877cc3c352571bc63e62` |
| Stage-B V2-A config | `20ae4ba7fee93c6c61e9969c5828b880d78e600439ba0e20fd3b5dc2c78f9395` |
| Post-fix BF16 result | `3df9d0b8592c8fc448594b05cf7dee2ee534de66e8b4bf9b27ecebeda4625bb9` |
| Post-fix FP32 result | `8796e60061e5c913763e0f3aedbbab94cbb992240eadc99f59337d2d0c1f0aec` |
| Post-fix five-seed result | `4c1103cfd2bd923c93368d881e16de83c308cd9693087a25b1f5cc85014c39f6` |

The new machine artifacts record source commit `d1a1382`; all historical
results remain preserved under their original source commits.

## 8. Explicit five-seed paired evaluation

The post-fix comparison uses seeds 2035-2039, ETH validation (139 windows),
CUDA/BF16, one shared V2-A checkpoint, identical goals/context/relation state,
and paired diffusion noise:

- C: correction off, Stage-B base reference;
- D: correction on, actual V2-A.

All metrics are lower-is-better.

### Mean and standard deviation

| Metric | C reference off | D actual V2-A | D-C | Relative D-C |
|---|---:|---:|---:|---:|
| minADE@K | 0.278167 +/- 0.001332 | 0.279971 +/- 0.001605 | +0.001804 | +0.649% |
| minFDE@K | 0.386226 +/- 0.002358 | 0.387217 +/- 0.003534 | +0.000992 | +0.257% |
| JADE | 0.413328 +/- 0.004507 | 0.414958 +/- 0.004441 | +0.001631 | +0.394% |
| JFDE | 0.702051 +/- 0.008413 | 0.698169 +/- 0.008011 | -0.003882 | -0.553% |
| Joint Goal Endpoint | 0.689933 +/- 0.007654 | 0.689933 +/- 0.007654 | 0 | 0% |
| Compatibility | 0.487513 +/- 0.015452 | 0.487513 +/- 0.015452 | 0 | 0% |
| Relative Motion | 0.297807 +/- 0.004667 | 0.297441 +/- 0.004300 | -0.000366 | -0.123% |

### Per-seed path C

| Seed | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.279302 | 0.388470 | 0.421198 | 0.693709 | 0.692777 | 0.478072 | 0.296951 |
| 2036 | 0.279874 | 0.386044 | 0.407132 | 0.691778 | 0.675802 | 0.493716 | 0.289917 |
| 2037 | 0.277629 | 0.388015 | 0.412326 | 0.713992 | 0.697011 | 0.509598 | 0.303760 |
| 2038 | 0.277942 | 0.386759 | 0.412961 | 0.708028 | 0.688420 | 0.492245 | 0.301016 |
| 2039 | 0.276088 | 0.381840 | 0.413021 | 0.702751 | 0.695656 | 0.463932 | 0.297392 |

### Per-seed path D

| Seed | minADE | minFDE | JADE | JFDE | Endpoint | Compatibility | Relative motion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2035 | 0.282328 | 0.389425 | 0.422988 | 0.691250 | 0.692777 | 0.478072 | 0.298416 |
| 2036 | 0.280427 | 0.386951 | 0.409467 | 0.689806 | 0.675802 | 0.493716 | 0.292028 |
| 2037 | 0.279399 | 0.390181 | 0.413452 | 0.711465 | 0.697011 | 0.509598 | 0.303023 |
| 2038 | 0.280312 | 0.389048 | 0.415213 | 0.702648 | 0.688420 | 0.492245 | 0.300766 |
| 2039 | 0.277388 | 0.380482 | 0.413670 | 0.695677 | 0.695656 | 0.463932 | 0.292972 |

### Paired routing contracts

For every one of the five seeds:

- C and D candidate IDs and joint goals are identical;
- diffusion RNG post-state is identical;
- E=0 trajectories are tensor-exact;
- inactive agents in mixed scenes are tensor-exact;
- candidate coverage remains 20/20;
- every one of the 139,680 active-agent trajectory values differs between C
  and D, confirming that the nonzero V2-A corrector remains active rather than
  being captured by the Stage-A bypass.

Goal endpoint and compatibility are identical by construction because the
paired experiment fixes the joint goals before trajectory diffusion.

## 9. Interpretation

The identity fix succeeds without redefining the learned active path.  It
removes the unwanted exact-zero mixed-precision drift while retaining a
material nonzero V2-A correction on degree-positive agents.

The corrected paired evaluation shows a mixed trajectory-level effect:

- JFDE and relative-motion error improve slightly under D;
- minADE, minFDE, and JADE degrade slightly;
- goal-only metrics remain unchanged as required.

These small paired changes do not constitute a uniform new V2-A gain, but
they also do not reproduce or explain the earlier full Stage-A/V2-A gap.
They isolate only the effect of enabling the trained corrector while holding
the Stage-B base reference, goals, and diffusion randomness fixed.  Historical
model-selection and final-result artifacts must therefore remain separately
identified by their original source commit and protocol.

## 10. Decision

The numerical identity blocker is closed:

```text
CANONICAL_STAGE_A_LITERAL_IDENTITY_CONFIRMED
```

No architecture change, loss change, retraining, or checkpoint replacement is
justified by this fix.  Any later adoption decision for V2-A must use the
explicit correction-on path D and must not cite path C as deployed V2-A.
