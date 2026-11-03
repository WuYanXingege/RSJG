# Figure QA

- Core conclusion: H reduces both best-world and all-world collision rates relative to G and matched GDTS across all five fold means.
- Evidence: `fivefold_cr_methods` shows absolute mean ± inference-seed SD; `fivefold_cr_deltas` shows paired fold-level mean differences.
- Archetype: quantitative grid; Python/Matplotlib backend.
- Replicate unit: five inference seeds from one selected checkpoint per fold. Error bars are sample SD (`ddof=1`).
- No observations or folds were excluded. Source data are `FIVEFOLD_COLLISION_SUMMARY.csv` and `PER_SEED_COLLISION_METRICS.csv`.
- Both two-panel figures passed the 1.5 pt rendered alignment gate. PDF glyph minima were 7 pt and 6 pt. Final PDF collision audits returned zero FAIL and zero WARN.
- Visual review: all five fold labels and uncertainty bars are legible at 182.9 mm width; UNIV's higher absolute scale does not clip or hide the lower-rate folds; delta zero lines and negative-improvement direction are unambiguous.
