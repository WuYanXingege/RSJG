#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$project_root/../.." && pwd)"
python_bin="$workspace_root/.conda/rsjg/bin/python"
assets="$workspace_root/RSJG_EBJD_diagnostic_assets/univ_no_token_transfer_20261021"
banks="$assets/native_banks"
protocol="$assets/protocol"
run_root="$assets/runs"
evaluation="$assets/evaluation"
archive="$project_root/docs/validation/ebjd_univ_no_token_transfer_20261021"
parent="$workspace_root/RSJG_EBJD_clean/ebjd_clean/outputs/formal_univ_main_20261013_seed3101/formal_training/univ/3101/none/univ_main_seed3101_1ef0c76_20261013/selected_formal.pt"
parent_sha="f526506055f346513b7c57994f9935d40e624162d74e110104073312cfc4bbe9"
provider="$workspace_root/GDTS/output/univ/saved_models/best_model.pt"
provider_sha="ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e"
train_manifest="$workspace_root/RSJG_EBJD_clean/ebjd_clean/data/univ_official/train/train_manifest.json"
heldout_manifest="$workspace_root/RSJG_EBJD_clean/ebjd_clean/data/univ_official/validation/validation_manifest.json"
train_sha="6f6fca362ec1e591bf9a79c9b1a0204123947a0c7b754992f131432af40dfd5d"
heldout_sha="46c52540aac6f0313865f01f53d7c735d91cb8116fdbee39cb65f0cf545c7d8e"
control_id="univ_transfer_control_e74_81_20261021"
candidate_id="univ_transfer_no_token_e74_81_20261021"
control_run="$run_root/univ/3101/rollout_mechanism_off/$control_id"
candidate_run="$run_root/univ/3101/v2_no_token/$candidate_id"

cd "$project_root"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$assets" "$banks" "$run_root" "$evaluation"

export_bank() {
  local manifest="$1" manifest_sha="$2" seed="$3" output="$4"
  if [[ ! -f "$output/manifest.json" ]]; then
    "$python_bin" -u -B scripts/export_native_gdts_goal_bank.py \
      --manifest "$manifest" --expected-manifest-sha256 "$manifest_sha" \
      --gdts-root "$workspace_root/GDTS" --gdts-checkpoint "$provider" \
      --expected-gdts-sha256 "$provider_sha" --output "$output" \
      --seed "$seed" --device cuda:0
  fi
}

export_bank "$train_manifest" "$train_sha" 3101 "$banks/train_seed3101"
for seed in 2035 2036 2037 2038 2039; do
  export_bank "$heldout_manifest" "$heldout_sha" "$seed" "$banks/validation_seed${seed}"
done

if [[ ! -f "$protocol/PLAN.json" ]]; then
  "$python_bin" -u -B scripts/prepare_univ_no_token_transfer.py \
    --parent "$parent" --bank-root "$banks" --run-root "$run_root" \
    --output "$protocol"
fi
control_config="$protocol/CONTROL_CONFIG.json"
candidate_config="$protocol/CANDIDATE_CONFIG.json"

if [[ ! -f "$control_run/last.pt" ]]; then
  "$python_bin" -u -B -m ebjd.train \
    --config "$control_config" --device cuda:0 \
    --ablation rollout_mechanism_off --run-id "$control_id" \
    --objective-fork-from "$parent" --objective-fork-from-sha256 "$parent_sha" \
    --objective-fork-contract rollout_mechanism_off \
    --absolute-stop-epoch 81 --migration-only
fi
if [[ ! -f "$candidate_run/last.pt" ]]; then
  "$python_bin" -u -B -m ebjd.train \
    --config "$candidate_config" --device cuda:0 \
    --ablation v2_no_token --run-id "$candidate_id" \
    --v2-fork-from "$parent" --v2-fork-from-sha256 "$parent_sha" \
    --v2-fork-contract univ_epoch73_legacy_v1 \
    --absolute-stop-epoch 81 --migration-only
fi

if [[ ! -f "$assets/CUDA_PREFLIGHT_RECEIPT.json" ]]; then
  "$python_bin" -u -B scripts/check_univ_no_token_transfer.py \
    --control-config "$control_config" --candidate-config "$candidate_config" \
    --parent "$parent" --expected-sha256 "$parent_sha" \
    --output "$assets/CUDA_PREFLIGHT_RECEIPT.json" --device cuda:0
fi

run_arm() {
  local config="$1" ablation="$2" run_dir="$3" elapsed="$4"
  if [[ ! -f "$run_dir/epoch_081.pt" ]]; then
    /usr/bin/time -f %e -o "$elapsed" \
      "$python_bin" -u -B -m ebjd.train \
      --config "$config" --device cuda:0 --ablation "$ablation" \
      --run-id "$(basename "$run_dir")" --resume "$run_dir/last.pt" \
      --absolute-stop-epoch 81
  fi
}

run_arm "$control_config" rollout_mechanism_off "$control_run" "$assets/control_training_seconds.txt"
run_arm "$candidate_config" v2_no_token "$candidate_run" "$assets/candidate_training_seconds.txt"

control_result="$evaluation/control_epoch081_raw_seeds2035_2039.json"
candidate_raw="$evaluation/candidate_epoch081_raw_seeds2035_2039.json"
candidate_fixed="$evaluation/candidate_epoch081_fixed_anchor_seeds2035_2039.json"
if [[ ! -f "$control_result" ]]; then
  /usr/bin/time -f %e -o "$assets/control_evaluation_seconds.txt" \
    "$python_bin" -u -B -m ebjd.evaluate \
    --config "$control_config" --checkpoint "$control_run/epoch_081.pt" \
    --ablation rollout_mechanism_off --split validation --device cuda:0 \
    --output "$control_result"
fi
if [[ ! -f "$candidate_fixed" ]]; then
  /usr/bin/time -f %e -o "$assets/candidate_evaluation_seconds.txt" \
    "$python_bin" -u -B -m ebjd.evaluate \
    --config "$candidate_config" --checkpoint "$candidate_run/epoch_081.pt" \
    --ablation v2_no_token --split validation --device cuda:0 \
    --output "$candidate_raw" --fixed-anchor-output "$candidate_fixed"
fi

"$python_bin" -u -B scripts/finalize_univ_no_token_transfer.py \
  --protocol "$protocol" --assets "$assets" \
  --control-run "$control_run" --candidate-run "$candidate_run" \
  --control-result "$control_result" \
  --candidate-raw-result "$candidate_raw" \
  --candidate-fixed-result "$candidate_fixed" \
  --parent-result "$project_root/docs/experiments/2026-10-13_univ_main_seed3101/EBJD_UNIV_SEEDS2035_2039.json" \
  --gdts-comparison "$project_root/docs/experiments/2026-10-13_univ_main_seed3101/UNIV_FINAL_COMPARISON.json" \
  --archive "$archive"

printf 'COMPLETE\narchive=%s\n' "$archive"
