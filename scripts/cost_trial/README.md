# ZARA2 candidate-cost proxy trial

The experiment freezes the original source at `468c5eddde48bdeec47c4bb3c39418352be16bb9` and changes only the final pair-loss backward proxy. `bcep_v2/cost_trial.py` uses the original stage writer, optimizer, seed 3201, update order and selection rules. Candidate assets are read-only upstream inputs. The original five-fold controller remains paused.

Set `SOURCE`, `PYTHON`, `UPSTREAM`, and `RUN` to the local trial checkout, Python environment, frozen upstream run, and independent trial output directory. Do not point `RUN` at `UPSTREAM`.

```bash
cd "$SOURCE"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1
"$PYTHON" -m bcep_v2.cost_acceptance --upstream "$UPSTREAM" --out "$RUN"
"$PYTHON" -m bcep_v2.cost_trial prepare --upstream "$UPSTREAM" --out "$RUN"
"$PYTHON" -m bcep_v2.cost_trial train --upstream "$UPSTREAM" --out "$RUN"
```

The first two commands must finish before training. Only this trial identity may resume with `train --resume`; never use the five-fold controller to launch it. Do not run the commands again for a second experiment. Checkpoints retain every completed epoch plus distinct H-JADE Top5, best and last. Maximum 16800 committed updates; minimum 30 completed epochs and patience 15, at most 100 epochs.

After `STATUS.json` reports `COMPLETE`:

```bash
"$PYTHON" scripts/cost_trial/report.py --run "$RUN" --upstream "$UPSTREAM" --output "$RESULTS"
"$PYTHON" scripts/cost_trial/plot.py --results "$RESULTS" --audit-library "$FIGURE_AUDIT_LIBRARY"
```

The report script refuses unfinished training. It reuses existing dev histories and computes only the missing historical high-count endpoint curves from frozen weights, without DDIM. Curves use a Python quantitative grid: all-window behavior and high-count robustness, with each model's dev H-JADE rank1 marked. One training seed supports a descriptive within-fold comparison, not confidence intervals. SVG/PDF preserve text; all plotted data are exported CSV. Final visual inspection and PDF text/collision audits precede publication.

Primary comparison: each method's dev H-JADE rank1, using all-scene H-JADE/H-JFDE and N>=9 H-JFDE. Compare also with the old epoch 5 tradeoff; the old epoch 5/26 JADE difference is contextual, not a significance threshold. No test selection or automatic next fold.
