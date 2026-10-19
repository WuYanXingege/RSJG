# EBJD ETH endpoint-coverage calibration

This branch adds a read-only calibration and audit for an existing P20
trajectory bank. It does not train a model or update optimizer state.

For world `k`, agent `i`, and future step `t=1..12`, the calibrated trajectory
is

```text
Y_new[k,i,t] = Y[k,i,t] + (t/12) * (gamma-1)
                * (g[k,i] - mean_worlds(g[:,i]))
```

where `g` is the original predicted endpoint. The operation preserves the
world/agent pairing, world order, candidate mean trajectory, and the nonlinear
bridge residual relative to each candidate's straight endpoint bridge. The
endpoint RMS radius and pairwise endpoint distances scale by `gamma`.
`gamma=1` is the literal original bank. The operation does not read ground
truth, add NFE, or repair endpoint-center bias.

Run from `ebjd_clean`:

```bash
python -m ebjd.audit_endpoint_calibration \
  --bank /path/to/seed_2035/manifest.json \
  --factors 1 1.25 1.5 2 \
  --output /path/to/calibration.json \
  --case-output-dir /path/to/cases
```

The audit verifies every bank scene SHA256 before reading it and reports the
four formal metrics, N=1/N>=2 strata, coverage statistics, nested-prefix K
curves, exact replay difference, and invariants. It uses float64 for the
calibration and audit calculations while retaining the original float32 bank
unchanged.
