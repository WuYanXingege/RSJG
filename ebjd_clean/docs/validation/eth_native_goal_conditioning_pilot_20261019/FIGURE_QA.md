# Figure QA

## Figure contract

- Core conclusion: native-goal conditioning consistently improves frozen-checkpoint EBJD metrics, but does not close the gap to matched GDTS.
- Results question: does the frozen GDTS endpoint prior improve both marginal and joint ETH prediction under a paired short run?
- Archetype: quantitative grid plus a separate image-plate-style fixed-case figure.
- Backend: Python/Matplotlib only.
- Final size: 7.2 inch double-column width; editable SVG/PDF; PNG preview. A 600 dpi TIFF is reproducible by the script but intentionally excluded from Git because each file is about 60 MB.
- Replicate unit: five inference seeds on one frozen training-seed-3101 checkpoint; error bars are mean ± sample SD.
- Exclusions: none; all five inference seeds, 253 scenes and 364 agent occurrences are retained.

## Panel audit

| Figure/panel | Unique role | Center/spread | Replicate/sample | Alignment | Collision | Visual inspection |
|---|---|---|---|---|---|---|
| Summary a | absolute method comparison and GDTS boundary | mean ± seed SD | 5 inference seeds | PASS | PASS | PASS |
| Summary b | paired Candidate−Control consistency | raw paired seed deltas | 5 inference seeds | PASS | PASS | PASS |
| Summary c | fixed-prefix candidate-budget response | mean ± seed SD | same stored P20 banks | PASS | PASS | PASS |
| Summary d | N=1 versus N≥2 boundary | mean ± seed SD | 183/70 scenes | PASS | PASS | PASS |
| Cases a | maximum parent→Candidate JFDE improvement | raw scene | fixed seed 2035 rule | PASS | PASS | PASS |
| Cases b | maximum degradation | raw scene | fixed seed 2035 rule | PASS | PASS | PASS |
| Cases c | A good but Candidate poor boundary | raw scene | fixed seed 2035 rule | PASS | PASS | PASS |
| Cases d | SHA256-fixed random scene | raw scene | fixed seed 2035 rule | PASS | PASS | PASS |

Static source preflight: 21 pass, 0 warn, 0 fail. PDF glyph audit: minimum rendered text 6 pt in both figures. Rendered collision audit: 0 fail, 0 warn for both figures. The stored alignment JSON receipts pass the 1.5 pt multi-panel contract. Case panels use identical world-coordinate semantics within each scene, without normalization or crop clipping. Full candidate banks remain local; committed CSV files are the source data for quantitative panels.
