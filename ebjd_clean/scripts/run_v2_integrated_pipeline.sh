#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$project_root/../.." && pwd)"
python_bin="$workspace_root/.conda/rsjg/bin/python"
assets="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_v2_integrated_20261020"
protocol="$assets/protocol_v1"
run_root="$assets/runs_v1"
evaluation="$assets/evaluation_v1"
parent="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_rollout_boundary_20261019/runs_v1/shared_runs/eth/3101/none/eth_rollout_shared_20261019/epoch_020.pt"
parent_sha="54bec78024e356857b7c93866921521a88a87e9f9ef4572d2d23d5a17648ad0e"
legacy_protocol="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_rollout_boundary_20261019/protocol_v1"
control_root="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_rollout_boundary_20261019/runs_v1/off_runs/eth/3101/rollout_mechanism_off/eth_rollout_off_20261019"
control_checkpoint="$control_root/epoch_025.pt"
control_result="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_rollout_boundary_20261019/evaluation_v1/off_epoch025_seeds2035_2039.json"
full_id="eth_v2_full_20261020"
no_token_id="eth_v2_no_token_20261020"
no_anchor_id="eth_v2_no_anchor_20261020"
full_run="$run_root/eth/3101/v2_full/$full_id"
no_token_run="$run_root/eth/3101/v2_no_token/$no_token_id"
no_anchor_run="$run_root/eth/3101/v2_no_anchor/$no_anchor_id"

cd "$project_root"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "$protocol/PLAN.json" ]]; then
  "$python_bin" scripts/prepare_v2_integrated_experiment.py \
    --parent "$parent" --output "$protocol" --run-root "$run_root"
fi

if [[ ! -f "$assets/CPU_PREFLIGHT_RECEIPT.json" ]]; then
  "$python_bin" scripts/preflight_v2_integrated.py \
    --protocol-dir "$protocol" --parent "$parent" \
    --expected-sha256 "$parent_sha" \
    --output "$assets/CPU_PREFLIGHT_RECEIPT.json"
fi

# The prior formal queue owns CUDA until it has produced and evaluated C.
if systemctl --user is-active --quiet rsjg-ebjd-eth-rollout-boundary-v2.service; then
  printf 'GPU_PENDING: protected service is still active; no CUDA work started.\n' >&2
  exit 75
fi
if [[ ! -f "$control_checkpoint" || ! -f "$control_result" ]]; then
  printf 'ASSET_PENDING: completed rollout-OFF control checkpoint/result missing.\n' >&2
  exit 76
fi

mkdir -p "$evaluation"
if [[ ! -f "$assets/CONTROL_REUSE_RECEIPT.json" ]]; then
  "$python_bin" scripts/certify_v2_control_reuse.py \
    --parent "$parent" --parent-sha256 "$parent_sha" \
    --control-checkpoint "$control_checkpoint" --control-result "$control_result" \
    --control-receipt "$control_root/OBJECTIVE_FORK_RECEIPT.json" \
    --output "$assets/CONTROL_REUSE_RECEIPT.json"
fi

if [[ ! -f "$assets/CUDA_PREFLIGHT_RECEIPT.json" ]]; then
  "$python_bin" scripts/check_v2_integrated_cuda.py \
    --config "$protocol/V2_FULL_CONFIG.json" --parent "$parent" \
    --expected-sha256 "$parent_sha" --device cuda:0 \
    --output "$assets/CUDA_PREFLIGHT_RECEIPT.json"
fi

if [[ ! -f "$assets/SPEED_AUDIT.json" ]]; then
  timeout --signal=TERM 2700 "$python_bin" scripts/benchmark_v2_speed.py \
    --legacy-config "$legacy_protocol/OFF_CONFIG.json" \
    --v2-config "$protocol/V2_FULL_CONFIG.json" \
    --parent "$parent" --expected-sha256 "$parent_sha" \
    --device cuda:0 --warmup 2 --updates 16 \
    --output "$assets/SPEED_AUDIT.json"
fi

run_arm() {
  local config="$1"
  local ablation="$2"
  local run_id="$3"
  local run_dir="$4"
  if [[ -f "$run_dir/epoch_025.pt" ]]; then
    return
  fi
  if [[ -f "$run_dir/last.pt" ]]; then
    "$python_bin" -u -B -m ebjd.train \
      --config "$config" --device cuda:0 --ablation "$ablation" \
      --run-id "$run_id" --resume "$run_dir/last.pt" \
      --absolute-stop-epoch 25
  else
    "$python_bin" -u -B -m ebjd.train \
      --config "$config" --device cuda:0 --ablation "$ablation" \
      --run-id "$run_id" --v2-fork-from "$parent" \
      --v2-fork-from-sha256 "$parent_sha" \
      --absolute-stop-epoch 25 --max-epochs-this-run 5
  fi
}

run_arm "$protocol/V2_FULL_CONFIG.json" v2_full "$full_id" "$full_run"
run_arm "$protocol/V2_NO_TOKEN_CONFIG.json" v2_no_token "$no_token_id" "$no_token_run"
run_arm "$protocol/V2_NO_ANCHOR_CONFIG.json" v2_no_anchor "$no_anchor_id" "$no_anchor_run"

evaluate_arm() {
  local config="$1"
  local checkpoint="$2"
  local ablation="$3"
  local output="$4"
  local bank_root="$5"
  if [[ ! -f "$output" ]]; then
    "$python_bin" -u -B -m ebjd.evaluate \
      --config "$config" --checkpoint "$checkpoint" --ablation "$ablation" \
      --split validation --device cuda:0 --prediction-bank-root "$bank_root" \
      --output "$output"
  fi
}

evaluate_arm "$protocol/V2_FULL_CONFIG.json" "$full_run/epoch_025.pt" v2_full \
  "$evaluation/v2_full_epoch025_seeds2035_2039.json" "$evaluation/banks_v2_full"
evaluate_arm "$protocol/V2_NO_TOKEN_CONFIG.json" "$no_token_run/epoch_025.pt" v2_no_token \
  "$evaluation/v2_no_token_epoch025_seeds2035_2039.json" "$evaluation/banks_v2_no_token"
evaluate_arm "$protocol/V2_NO_ANCHOR_CONFIG.json" "$no_anchor_run/epoch_025.pt" v2_no_anchor \
  "$evaluation/v2_no_anchor_epoch025_seeds2035_2039.json" "$evaluation/banks_v2_no_anchor"

if [[ ! -f "$evaluation/v2_full_initial_pairing_seeds2035_2039.json" ]]; then
  "$python_bin" -u -B -m ebjd.evaluate \
    --config "$protocol/V2_FULL_CONFIG.json" --checkpoint "$full_run/epoch_025.pt" \
    --ablation v2_full --split validation --device cuda:0 --pairing-rounds 0 \
    --output "$evaluation/v2_full_initial_pairing_seeds2035_2039.json"
fi
if [[ ! -f "$evaluation/v2_full_direct_geometry_seeds2035_2039.json" ]]; then
  "$python_bin" -u -B -m ebjd.evaluate \
    --config "$protocol/V2_FULL_CONFIG.json" --checkpoint "$full_run/epoch_025.pt" \
    --ablation v2_full --split validation --device cuda:0 \
    --geometry-mapping direct_index \
    --output "$evaluation/v2_full_direct_geometry_seeds2035_2039.json"
fi
if [[ ! -f "$evaluation/v2_full_projection_rho0_seeds2035_2039.json" ]]; then
  "$python_bin" -u -B -m ebjd.evaluate \
    --config "$protocol/V2_FULL_CONFIG.json" --checkpoint "$full_run/epoch_025.pt" \
    --ablation v2_full --split validation --device cuda:0 \
    --endpoint-projection-radius 0 \
    --output "$evaluation/v2_full_projection_rho0_seeds2035_2039.json"
fi
p95="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["assets"]["train_nearest_anchor_endpoint_error_p95_m"])' "$protocol/PLAN.json")"
if [[ ! -f "$evaluation/v2_full_projection_train_p95_seeds2035_2039.json" ]]; then
  "$python_bin" -u -B -m ebjd.evaluate \
    --config "$protocol/V2_FULL_CONFIG.json" --checkpoint "$full_run/epoch_025.pt" \
    --ablation v2_full --split validation --device cuda:0 \
    --endpoint-projection-radius "$p95" \
    --output "$evaluation/v2_full_projection_train_p95_seeds2035_2039.json"
fi

"$python_bin" scripts/finalize_v2_integrated.py \
  --assets "$assets" --protocol "$protocol" \
  --control-run "$control_root" --full-run "$full_run" \
  --no-token-run "$no_token_run" --no-anchor-run "$no_anchor_run" \
  --output-dir "$project_root/docs/validation/ebjd_v2_integrated_20261020" \
  --report "$project_root/docs/EBJD_V2_INTEGRATED_NETWORK_2026-10-20.md"

printf 'COMPLETE\nreport=%s\n' \
  "$project_root/docs/EBJD_V2_INTEGRATED_NETWORK_2026-10-20.md"
