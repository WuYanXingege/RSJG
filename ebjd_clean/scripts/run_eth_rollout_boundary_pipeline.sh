#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$project_root/../.." && pwd)"
python_bin="$workspace_root/.conda/rsjg/bin/python"
previous_assets="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_native_goal_conditioning_20261019"
assets="$workspace_root/RSJG_EBJD_diagnostic_assets/eth_rollout_boundary_20261019"
protocol="$assets/protocol_v1"
run_root="$assets/runs_v1"
evaluation="$assets/evaluation_v1"
origin="$previous_assets/runs_v4/eth/3101/none/eth_native_goal_candidate_6f7a05f_20261019/epoch_010.pt"
origin_sha="b339ab96c288fe0757d2bce75f6408fa80dfde3b030a1109af68b4a760aa4f12"
shared_id="eth_rollout_shared_20261019"
on_id="eth_rollout_on_20261019"
off_id="eth_rollout_off_20261019"
shared_run="$run_root/shared_runs/eth/3101/none/$shared_id"
on_run="$run_root/on_runs/eth/3101/none/$on_id"
off_run="$run_root/off_runs/eth/3101/rollout_mechanism_off/$off_id"
shared_checkpoint="$shared_run/epoch_020.pt"
on_checkpoint="$on_run/epoch_025.pt"
off_checkpoint="$off_run/epoch_025.pt"

cd "$project_root"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "$protocol/PLAN.json" ]]; then
  "$python_bin" scripts/prepare_rollout_boundary_pilot.py \
    --candidate-epoch10 "$origin" \
    --expected-sha256 "$origin_sha" \
    --output "$protocol" --run-root "$run_root"
fi

if [[ ! -f "$assets/PREFLIGHT_RECEIPT.json" ]]; then
  "$python_bin" scripts/check_rollout_boundary_pairing.py \
    --config "$protocol/SHARED_CONFIG.json" \
    --checkpoint "$origin" --expected-sha256 "$origin_sha" \
    --device cuda:0 --output "$assets/PREFLIGHT_RECEIPT.json"
fi

if [[ ! -f "$shared_checkpoint" ]]; then
  if [[ -f "$shared_run/last.pt" ]]; then
    "$python_bin" -u -B -m ebjd.train \
      --config "$protocol/SHARED_CONFIG.json" --device cuda:0 \
      --run-id "$shared_id" --resume "$shared_run/last.pt" \
      --absolute-stop-epoch 20
  else
    "$python_bin" -u -B -m ebjd.train \
      --config "$protocol/SHARED_CONFIG.json" --device cuda:0 \
      --run-id "$shared_id" --resume-from "$origin" \
      --resume-from-sha256 "$origin_sha" \
      --absolute-stop-epoch 20 --max-epochs-this-run 10
  fi
fi
shared_sha="$(sha256sum "$shared_checkpoint" | awk '{print $1}')"

if [[ ! -f "$on_checkpoint" ]]; then
  if [[ -f "$on_run/last.pt" ]]; then
    "$python_bin" -u -B -m ebjd.train \
      --config "$protocol/ON_CONFIG.json" --device cuda:0 \
      --run-id "$on_id" --resume "$on_run/last.pt" \
      --absolute-stop-epoch 25
  else
    "$python_bin" -u -B -m ebjd.train \
      --config "$protocol/ON_CONFIG.json" --device cuda:0 \
      --run-id "$on_id" --resume-from "$shared_checkpoint" \
      --resume-from-sha256 "$shared_sha" \
      --absolute-stop-epoch 25 --max-epochs-this-run 5
  fi
fi

if [[ ! -f "$off_checkpoint" ]]; then
  if [[ -f "$off_run/last.pt" ]]; then
    "$python_bin" -u -B -m ebjd.train \
      --config "$protocol/OFF_CONFIG.json" --device cuda:0 \
      --ablation rollout_mechanism_off --run-id "$off_id" \
      --resume "$off_run/last.pt" --absolute-stop-epoch 25
  else
    "$python_bin" -u -B -m ebjd.train \
      --config "$protocol/OFF_CONFIG.json" --device cuda:0 \
      --ablation rollout_mechanism_off --run-id "$off_id" \
      --objective-fork-from "$shared_checkpoint" \
      --objective-fork-from-sha256 "$shared_sha" \
      --objective-fork-contract rollout_mechanism_off \
      --absolute-stop-epoch 25 --max-epochs-this-run 5
  fi
fi

mkdir -p "$evaluation"
if [[ ! -f "$evaluation/epoch020_seeds2035_2039.json" ]]; then
  "$python_bin" -u -B -m ebjd.evaluate \
    --config "$protocol/SHARED_CONFIG.json" --checkpoint "$shared_checkpoint" \
    --split validation --device cuda:0 \
    --prediction-bank-root "$evaluation/banks_epoch020" \
    --output "$evaluation/epoch020_seeds2035_2039.json"
fi
if [[ ! -f "$evaluation/off_epoch025_seeds2035_2039.json" ]]; then
  "$python_bin" -u -B -m ebjd.evaluate \
    --config "$protocol/OFF_CONFIG.json" --checkpoint "$off_checkpoint" \
    --ablation rollout_mechanism_off --split validation --device cuda:0 \
    --prediction-bank-root "$evaluation/banks_off_epoch025" \
    --output "$evaluation/off_epoch025_seeds2035_2039.json"
fi
if [[ ! -f "$evaluation/on_epoch025_seeds2035_2039.json" ]]; then
  "$python_bin" -u -B -m ebjd.evaluate \
    --config "$protocol/ON_CONFIG.json" --checkpoint "$on_checkpoint" \
    --split validation --device cuda:0 \
    --prediction-bank-root "$evaluation/banks_on_epoch025" \
    --output "$evaluation/on_epoch025_seeds2035_2039.json"
fi

docs_validation="$project_root/docs/validation/eth_rollout_boundary_pilot_20261019"
report="$project_root/docs/EBJD_ETH_ROLLOUT_BOUNDARY_PILOT_2026-10-19.md"
if [[ ! -f "$docs_validation/RESULTS_AND_DIAGNOSTICS.json" ]]; then
  "$python_bin" scripts/finalize_rollout_boundary_pilot.py \
    --protocol-dir "$protocol" \
    --shared-run "$shared_run" --on-run "$on_run" --off-run "$off_run" \
    --epoch10-result "$previous_assets/evaluation_v4/candidate_epoch010_seeds2035_2039.json" \
    --gdts-result "$previous_assets/evaluation_v4/matched_gdts_seeds2035_2039.json" \
    --epoch20-result "$evaluation/epoch020_seeds2035_2039.json" \
    --off-result "$evaluation/off_epoch025_seeds2035_2039.json" \
    --on-result "$evaluation/on_epoch025_seeds2035_2039.json" \
    --epoch20-banks "$evaluation/banks_epoch020" \
    --off-banks "$evaluation/banks_off_epoch025" \
    --on-banks "$evaluation/banks_on_epoch025" \
    --preflight "$assets/PREFLIGHT_RECEIPT.json" \
    --output-dir "$docs_validation" --report "$report"
fi

printf 'COMPLETED\nreport=%s\nresults=%s\n' \
  "$report" "$docs_validation/RESULTS_AND_DIAGNOSTICS.json"
