# JDV2 Stage-B V1 Failure-Mechanism Audit

## Status

**`AUDIT_IMPLEMENTATION_BLOCKED`**

Interpretation state: **`INCONCLUSIVE`**.

Stage-B V2 status: **`stage_b_v2_design_not_yet_justified`**.

The mandatory seed-2035, 139-window parity gate found a production-path
violation before the scientific counterfactuals were admissible.  The audit
therefore stopped.  No training, finetuning, parameter update, Stage-A
change, DependencyCorrector change, or Stage-B V2 implementation was made.

## 1. Scope and provenance

| Item | Value |
|---|---|
| Branch | `research/joint-dependency-v2-clean` |
| Audit starting HEAD | `4ccc6c86f8296eac9a8ab24622329a39d2ff0ad7` |
| Dataset / split | ETH validation |
| Parity seed | 2035 |
| Windows | 139 (47 E=0, 92 E>0) |
| Stage-A checkpoint | strict-no-z epoch 13 |
| Stage-A SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| Stage-B checkpoint | V1 epoch 10 |
| Stage-B SHA256 | `e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a` |
| Precision | canonical BF16 with existing FP32 islands |
| Candidate coverage | minimum 20/20 |

Machine-readable evidence is in
`outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_b_v1/failure_mechanism_audit/results.json`.

## 2. Exact NoiseTape replay

The audit-local sampler reproduces production RNG consumption explicitly:

- `x_T [N,12,2]` is drawn on CPU and transferred to the model device,
  matching `ts_sample()`;
- branch noise is retained as 20 branches by six active steps at
  timesteps `30,25,20,15,10,5`;
- FULL and NONE use the same Stage-A worlds, candidate IDs, contexts,
  window ordering and tape;
- Python, NumPy, CPU Torch and all-CUDA RNG states are snapshotted and
  restored around the production controls.

Across all 139 windows:

| Gate | Maximum absolute difference |
|---|---:|
| audit-local FULL vs production Stage-B FULL | **0.0** |
| audit-local NONE vs production corrector-disabled Stage A | **0.0** |

Thus the local replay itself is tensor-exact.  Candidate coverage remained
20/20 and all sampled goals were finite.

## 3. Blocking E=0 finding

The required contract says every E=0 variant must be tensor-exact Stage A.
Production Stage-B V1 does not satisfy that contract under canonical BF16:

| Quantity over 47 E=0 windows | Result |
|---|---:|
| windows with FULL != NONE | **47 / 47** |
| learned residual maximum absolute value | **0.0** |
| FULL-vs-NONE trajectory maximum absolute difference | **0.009775877** |
| mean of per-window trajectory mean absolute differences | **0.000891427** |

The corrector correctly returns a FP32 exact-zero tensor when `E=0`.
However, production `ts_sample()` still executes:

```text
eps = eps + delta
```

The denoiser output is produced in the BF16 autocast path while the exact-zero
residual is FP32.  Adding the FP32 zero promotes the expression and changes
the arithmetic used by the subsequent DDIM update.  Therefore the residual
contains no learned dependency signal, yet the trajectory changes.  The
audit-local FULL replay matches this behavior exactly; the audit-local NONE
replay likewise matches the corrector-disabled production path exactly.

This is not ordinary unpaired stochastic variation: FULL and NONE use the
same explicit NoiseTape.  It is also not common/component drift, branch
deployment mismatch, objective-gradient mismatch, or relation under-use,
because E=0 has no edges, no active relation embedding, and an exact-zero
corrector output.

## 4. Why the scientific audit stopped

Two required conditions cannot simultaneously hold with the immutable
production source:

1. FULL must reproduce production Stage-B V1 tensor-exactly.
2. E=0 FULL must reproduce paired Stage A tensor-exactly.

The first condition passes.  The second fails in all 47 E=0 windows.  Making
the audit silently skip `eps + 0` would satisfy the second condition only by
ceasing to represent production FULL, invalidating causal comparison.  The
task explicitly prohibited changing production `src/`, so the audit stopped
instead of masking the discrepancy or continuing with an inconsistent
baseline.

Consequently, the following were deliberately **not run after the failed
gate**:

- five-seed common/component residual counterfactuals;
- branch-rank and ORACLE_ONLY analysis;
- INIT/BEST gradient comparison;
- relation zero/mean/shuffle sensitivity;
- any Stage-B V2 design or training.

No conclusions about hypotheses H1-H4 are claimed.

## 5. Narrowest required engineering action

Before a Stage-B V2 design review is scientifically justified, a separately
authorized production engineering fix must make the degree-zero path a true
no-op without altering E>0 semantics.  The narrow contract is:

```text
if E == 0:
    do not execute a dtype-promoting epsilon addition
    preserve the base-denoiser branch update tensor-exactly
```

That change is not implemented here.  After such a fix, the affected
all-off/Stage-A/Stage-B CUDA-BF16 regressions and the entire failure-mechanism
audit must be rerun from the parity gate.  It would be premature to change
Stage-B architecture, objective, relation conditioning, or branch policy
before this paired baseline is valid.

## 6. Requested questions

1. Component-common versus centered energy: **not estimable after failed gate**.
2. Marginal recovery under component centering: **not estimable**.
3. JADE/JFDE under component centering: **not estimable**.
4. COMMON_ONLY damage attribution: **not estimable**.
5. Goal-oracle versus other-branch utility: **not estimable**.
6. Goal-oracle/trajectory-oracle agreement: **not estimable**.
7. ORACLE_ONLY versus FULL: **not estimable**.
8. Branch-rank Spearman relationship: **not estimable**.
9. INIT/BEST gradient ratio and cosine: **not estimable**.
10. Weighted relative-gradient role: **not estimable**.
11. Branch-specific relation use: **not estimable**.
12. Primary localization: **none of H1-H4 can yet be selected; a prior
    E=0 mixed-precision no-op violation is identified**.
13. Narrowest justified next action: **engineering repair and re-audit, not a
    Stage-B V2 method change**.

## 7. Integrity and stop condition

- Stage-A checkpoint hash: unchanged.
- Stage-B checkpoint hash: unchanged.
- `src/` diff: empty.
- No checkpoint was created.
- No optimizer step was executed.
- No Stage B V2, Stage A, HOTEL, UNIV or ZARA run was started.

Final mechanism finding: **E=0 mixed-precision exact-zero addition drift**.

Final eligibility: **`stage_b_v2_design_not_yet_justified`**.
