# Epoch 45 paired five-epoch follow-up commands — not launched

The materialization command must be run from the archived diagnostic commit on
a clean tree. It binds both generated configs to that exact commit and changes
no training hyperparameter other than the candidate's `goal_v_weight`.

```bash
rsjg_root=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG
repo="$rsjg_root/RSJG_EBJD_epoch45_diagnostic_worktree"
py="$rsjg_root/.conda/rsjg/bin/python"
origin="$rsjg_root/RSJG_EBJD_clean/ebjd_clean/outputs/formal_hotel_speed_9ecde66_20261009/hotel/3101/none/hotel_main_seed3101_9ecde66_speed_20261009"
pair_root="$rsjg_root/RSJG_EBJD_clean/ebjd_clean/outputs/epoch45_goal_weight_pair"
config_root="$pair_root/materialized_configs"

cd "$repo/ebjd_clean" || exit 1
"$py" scripts/materialize_epoch45_short_experiment.py \
  --source-config "$origin/RESOLVED_CONFIG.json" \
  --output-dir "$config_root" \
  --experiment-output-root "$pair_root/runs"
```

Control arm (exact full-state restore, original objective, epochs 46–50):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$py" -u -m ebjd.train \
  --config "$config_root/CONTROL_GOAL_WEIGHT_1.yaml" \
  --resume-from "$origin/epoch_045.pt" \
  --resume-from-sha256 425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1 \
  --max-epochs-this-run 5 \
  --run-id hotel_epoch45_control_goalw1_e46_50
```

Candidate arm (same full state and RNG; only `goal_v_weight: 1→2`):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$py" -u -m ebjd.train \
  --config "$config_root/CANDIDATE_GOAL_WEIGHT_2.yaml" \
  --objective-fork-from "$origin/epoch_045.pt" \
  --objective-fork-from-sha256 425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1 \
  --objective-fork-contract goal_v_weight_2 \
  --max-epochs-this-run 5 \
  --run-id hotel_epoch45_candidate_goalw2_e46_50
```

These commands are recorded for the next round only. This diagnostic round did
not execute either command and performed zero optimizer updates.
