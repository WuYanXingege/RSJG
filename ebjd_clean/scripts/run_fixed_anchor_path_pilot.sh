#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$project_root/../.." && pwd)"
python_bin="$workspace_root/.conda/rsjg/bin/python"
assets="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_no_token_fixed_anchor_path_pilot_20261020"
protocol="${EBJD_PATH_PILOT_PROTOCOL:-$project_root/docs/validation/ebjd_no_token_fixed_anchor_path_pilot_20261020/protocol}"
run_root="$assets/runs_v1"
evaluation="$assets/evaluation_v1"
parent="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_v2_integrated_20261020/runs_v1/eth/3101/v2_no_token/eth_v2_no_token_20261020/epoch_025.pt"
parent_sha="cb201fe03d661bb7687b0592ad5ac45481b79cb06503fa73bdfb4b453c6043ff"
control_id="eth_no_token_path_control_20261020"
candidate_id="eth_no_token_path_candidate_20261020"
control_run="$run_root/eth/3101/v2_no_token/$control_id"
candidate_run="$run_root/eth/3101/v2_no_token/$candidate_id"

cd "$project_root"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source_commit="$(git rev-parse HEAD)"
mkdir -p "$assets" "$evaluation"

if [[ ! -f "$protocol/PLAN.json" ]]; then
  "$python_bin" scripts/prepare_fixed_anchor_path_pilot.py \
    --parent "$parent" --output "$protocol" --run-root "$run_root" \
    --source-commit "$source_commit"
fi

control_config="$protocol/CONTROL_CONFIG.json"
candidate_config="$protocol/CANDIDATE_CONFIG.json"

migrate_arm() {
  local config="$1" run_id="$2" run_dir="$3"
  if [[ ! -f "$run_dir/last.pt" ]]; then
    "$python_bin" -u -B -m ebjd.train \
      --config "$config" --device cuda:0 --ablation v2_no_token \
      --run-id "$run_id" --objective-fork-from "$parent" \
      --objective-fork-from-sha256 "$parent_sha" \
      --objective-fork-contract fixed_anchor_path_weight \
      --absolute-stop-epoch 28 --migration-only
  fi
}

migrate_arm "$control_config" "$control_id" "$control_run"
migrate_arm "$candidate_config" "$candidate_id" "$candidate_run"

if [[ ! -f "$assets/CUDA_PREFLIGHT_RECEIPT.json" ]]; then
  "$python_bin" -u -B scripts/check_fixed_anchor_path_pilot.py \
    --control-config "$control_config" --candidate-config "$candidate_config" \
    --parent "$parent" --expected-sha256 "$parent_sha" --device cuda:0 \
    --output "$assets/CUDA_PREFLIGHT_RECEIPT.json"
fi

run_arm() {
  local config="$1" run_dir="$2" time_file="$3"
  if [[ ! -f "$run_dir/epoch_028.pt" ]]; then
    /usr/bin/time -f %e -o "$time_file" \
      "$python_bin" -u -B -m ebjd.train \
      --config "$config" --device cuda:0 --ablation v2_no_token \
      --run-id "$(basename "$run_dir")" --resume "$run_dir/last.pt" \
      --absolute-stop-epoch 28
  fi
}

run_arm "$control_config" "$control_run" "$assets/control_training_seconds.txt"
run_arm "$candidate_config" "$candidate_run" "$assets/candidate_training_seconds.txt"

evaluate_arm() {
  local arm="$1" config="$2" checkpoint="$3" output="$4"
  if [[ ! -f "$output" ]]; then
    /usr/bin/time -f %e -o "$assets/${arm}_evaluation_seconds.txt" \
      "$python_bin" -u -B -m ebjd.evaluate \
      --config "$config" --checkpoint "$checkpoint" --ablation v2_no_token \
      --split validation --device cuda:0 --endpoint-projection-radius 0 \
      --output "$output"
  fi
}

control_result="$evaluation/control_epoch028_fixed_anchor_seeds2035_2039.json"
candidate_result="$evaluation/candidate_epoch028_fixed_anchor_seeds2035_2039.json"
evaluate_arm control "$control_config" "$control_run/epoch_028.pt" "$control_result"
evaluate_arm candidate "$candidate_config" "$candidate_run/epoch_028.pt" "$candidate_result"

"$python_bin" -u -B scripts/finalize_fixed_anchor_path_pilot.py \
  --protocol "$protocol" --assets "$assets" \
  --control-run "$control_run" --candidate-run "$candidate_run" \
  --control-result "$control_result" --candidate-result "$candidate_result" \
  --parent-results "$project_root/docs/validation/ebjd_no_token_fixed_anchor_20261020/RESULTS.json" \
  --output "$project_root/docs/validation/ebjd_no_token_fixed_anchor_path_pilot_20261020/RESULTS.json" \
  --report "$project_root/docs/EBJD_NO_TOKEN_FIXED_ANCHOR_PATH_PILOT_2026-10-20.md"

printf 'COMPLETE\nreport=%s\n' \
  "$project_root/docs/EBJD_NO_TOKEN_FIXED_ANCHOR_PATH_PILOT_2026-10-20.md"
