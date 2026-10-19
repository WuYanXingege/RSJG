# ETH native-goal conditioning pilot: exact command record

These commands correspond to the archived v4 protocol. Existing output directories are intentionally refused by the CLIs; this is an execution record, not an instruction to overwrite the completed run.

```bash
repo=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_eth_native_goal_pilot
project="$repo/ebjd_clean"
python=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python
assets=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_diagnostic_assets/eth_native_goal_conditioning_20261019
data=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_clean/ebjd_clean/data/eth_jmm_official_scene_shards_20261015
parent=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_clean/ebjd_clean/outputs/formal_eth_main_20261015_seed3101/formal_training/eth/3101/none/eth_main_seed3101_d266681_20261015/epoch_005.pt
gdts=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/GDTS
gdts_ckpt="$gdts/output/eth/saved_models/best_model.pt"

cd "$project"
git rev-parse HEAD
git status --short
sha256sum "$data/validation/validation_manifest.json" "$data/train/train_manifest.json" "$parent" "$gdts_ckpt"
```

Expected critical identities:

```text
implementation 6f7a05f718accf322dc0dedb6d024111a115e6f8
validation     03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b
train          8824bbad3b1b6c964bd828b1bee3ea5870e614b300b6d14090644a72818c54de
parent         a5f45606329b80754567cf86381ceff0d19e68c8eaddf999c74f71ef1bdde525
GDTS/provider  126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950
```

## Export frozen provider banks

```bash
PYTHONPATH=. "$python" scripts/export_native_gdts_goal_bank.py \
  --manifest "$data/train/train_manifest.json" \
  --expected-manifest-sha256 8824bbad3b1b6c964bd828b1bee3ea5870e614b300b6d14090644a72818c54de \
  --gdts-root "$gdts" --gdts-checkpoint "$gdts_ckpt" \
  --expected-gdts-sha256 126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950 \
  --output "$assets/train_seed3101_provider092_rngfix" \
  --seed 3101 --device cuda:0

for seed in 2035 2036 2037 2038 2039; do
  PYTHONPATH=. "$python" scripts/export_native_gdts_goal_bank.py \
    --manifest "$data/validation/validation_manifest.json" \
    --expected-manifest-sha256 03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b \
    --gdts-root "$gdts" --gdts-checkpoint "$gdts_ckpt" \
    --expected-gdts-sha256 126acf2a34f52986c536c397fe3acb04c769a3cde95077461d7971a1b0792950 \
    --output "$assets/validation_seed${seed}_provider092_rngfix" \
    --seed "$seed" --device cuda:0
done
```

The seed-2035 replay was verified against the certified A route: 20 centres, no appended argmax, 0.5747143882 native oracle FDE.

## Freeze v4 plan and configs

```bash
PYTHONPATH=. "$python" scripts/prepare_native_goal_pilot.py \
  --parent "$parent" \
  --expected-parent-sha256 a5f45606329b80754567cf86381ceff0d19e68c8eaddf999c74f71ef1bdde525 \
  --cache-root "$assets" \
  --output "$assets/pilot_protocol_v4" \
  --run-output-root "$assets/runs_v4"
```

## Migration and compatibility gates

```bash
PYTHONPATH=. "$python" -m ebjd.train \
  --config "$assets/pilot_protocol_v4/CANDIDATE_CONFIG.json" \
  --device cuda:0 --run-id native_goal_gate_migration_v4 \
  --native-goal-fork-from "$parent" \
  --native-goal-fork-from-sha256 a5f45606329b80754567cf86381ceff0d19e68c8eaddf999c74f71ef1bdde525 \
  --absolute-stop-epoch 10 --migration-only

PYTHONPATH=. "$python" scripts/check_native_goal_compatibility.py \
  --device cuda \
  --output docs/validation/eth_native_goal_conditioning_pilot_20261019/COMPATIBILITY_RECEIPT.json

PYTHONPATH=. "$python" -m pytest -q
```

## Paired formal training

```bash
PYTHONPATH=. "$python" -m ebjd.train \
  --config "$assets/pilot_protocol_v4/CONTROL_CONFIG.json" \
  --device cuda:0 --run-id eth_native_goal_control_6f7a05f_20261019 \
  --resume-from "$parent" \
  --resume-from-sha256 a5f45606329b80754567cf86381ceff0d19e68c8eaddf999c74f71ef1bdde525 \
  --absolute-stop-epoch 10 --max-epochs-this-run 5

PYTHONPATH=. "$python" -m ebjd.train \
  --config "$assets/pilot_protocol_v4/CANDIDATE_CONFIG.json" \
  --device cuda:0 --run-id eth_native_goal_candidate_6f7a05f_20261019 \
  --native-goal-fork-from "$parent" \
  --native-goal-fork-from-sha256 a5f45606329b80754567cf86381ceff0d19e68c8eaddf999c74f71ef1bdde525 \
  --absolute-stop-epoch 10 --max-epochs-this-run 5
```

## Frozen five-seed evaluation

```bash
control="$assets/runs_v4/eth/3101/none/eth_native_goal_control_6f7a05f_20261019/epoch_010.pt"
candidate="$assets/runs_v4/eth/3101/none/eth_native_goal_candidate_6f7a05f_20261019/epoch_010.pt"
evalroot="$assets/evaluation_v4"

PYTHONPATH=. "$python" -m ebjd.evaluate \
  --config "$assets/pilot_protocol_v4/CONTROL_CONFIG.json" \
  --checkpoint "$parent" --split validation --device cuda:0 \
  --prediction-bank-root "$assets/evaluation/banks_epoch005" \
  --output "$assets/evaluation/epoch005_seeds2035_2039.json"

PYTHONPATH=. "$python" -m ebjd.evaluate \
  --config "$assets/pilot_protocol_v4/CONTROL_CONFIG.json" \
  --checkpoint "$control" --split validation --device cuda:0 \
  --prediction-bank-root "$evalroot/banks_control" \
  --output "$evalroot/control_epoch010_seeds2035_2039.json"

PYTHONPATH=. "$python" -m ebjd.evaluate \
  --config "$assets/pilot_protocol_v4/CANDIDATE_CONFIG.json" \
  --checkpoint "$candidate" --split validation --device cuda:0 \
  --prediction-bank-root "$evalroot/banks_candidate" \
  --output "$evalroot/candidate_epoch010_seeds2035_2039.json"

PYTHONPATH=. "$python" -m ebjd.evaluate \
  --config "$assets/pilot_protocol_v4/CANDIDATE_CONFIG.json" \
  --checkpoint "$candidate" --split validation --device cuda:0 \
  --disable-native-goal-conditioner \
  --prediction-bank-root "$evalroot/banks_candidate_disabled" \
  --output "$evalroot/candidate_disabled_epoch010_seeds2035_2039.json"

PYTHONPATH=. "$python" scripts/evaluate_gdts_scene_manifest.py \
  --jdv2-root "$gdts" --config "$gdts/output/eth/config.yaml" \
  --checkpoint "$gdts_ckpt" \
  --manifest "$data/validation/validation_manifest.json" \
  --seeds 2035 2036 2037 2038 2039 --device cuda:0 \
  --output "$evalroot/matched_gdts_seeds2035_2039.json"
```

## Diagnostics, figures and QA

```bash
PYTHONPATH=. "$python" scripts/analyze_native_goal_pilot.py \
  --asset-root "$assets" \
  --output docs/validation/eth_native_goal_conditioning_pilot_20261019

python /home/ee615/.codex/skills/nature-figure/scripts/validate_figure.py \
  scripts/analyze_native_goal_pilot.py --strict

python /home/ee615/.codex/skills/nature-figure/scripts/audit_pdf_text.py \
  docs/validation/eth_native_goal_conditioning_pilot_20261019/native_goal_pilot_summary.pdf \
  --min-pt 5 --json

"$python" /home/ee615/.codex/skills/nature-figure/scripts/audit_figure_collisions.py \
  docs/validation/eth_native_goal_conditioning_pilot_20261019/native_goal_pilot_summary.pdf \
  --json-out docs/validation/eth_native_goal_conditioning_pilot_20261019/native_goal_pilot_summary.collision-audit.json

"$python" /home/ee615/.codex/skills/nature-figure/scripts/audit_figure_collisions.py \
  docs/validation/eth_native_goal_conditioning_pilot_20261019/native_goal_pilot_fixed_cases.pdf \
  --json-out docs/validation/eth_native_goal_conditioning_pilot_20261019/native_goal_pilot_fixed_cases.collision-audit.json
```
