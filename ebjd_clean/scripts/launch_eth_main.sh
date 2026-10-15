#!/usr/bin/env bash
set -euo pipefail

action="${1:-launch}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ebjd_root="$repo/ebjd_clean"
workspace="$(dirname "$repo")"
python_bin="$workspace/.conda/rsjg/bin/python"
runtime="$workspace/RSJG_EBJD_clean/ebjd_clean/outputs/formal_eth_main_20261015_seed3101"
pid_file="$runtime/pipeline.pid"
launch_log="$runtime/pipeline_supervisor.log"
state_file="$runtime/PIPELINE_STATE.json"

mkdir -p "$runtime"

if [[ "$action" == "status" ]]; then
  if [[ -f "$state_file" ]]; then
    "$python_bin" -m json.tool "$state_file"
  else
    printf 'No state file yet: %s\n' "$state_file"
  fi
  exit 0
fi

if [[ "$action" != "launch" && "$action" != "resume" ]]; then
  printf 'Usage: %s [launch|resume|status]\n' "$0" >&2
  exit 2
fi

if [[ -f "$pid_file" ]]; then
  existing_pid="$(tr -dc '0-9' < "$pid_file")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    printf 'ETH pipeline is already running with PID %s\n' "$existing_pid"
    exit 0
  fi
fi

cd "$ebjd_root"
PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid -f nohup "$python_bin" -u -B scripts/run_univ_main_pipeline.py \
    --fold eth \
    --base-config "$ebjd_root/configs/endpoint_bridge_joint_eth_formal.yaml" \
    --date-tag 20261015 \
    --git-branch research/ebjd-eth-main-seed3101-v1 \
    --python "$python_bin" \
    --runtime-dir "$runtime" \
    --goal-run-id eth_goal_pretrain_seed3101_20261015 \
    --hotel-checkpoint "$workspace/RSJG_EBJD_clean/ebjd_clean/outputs/formal_hotel_speed_9ecde66_20261009/hotel/3101/none/hotel_main_seed3101_9ecde66_speed_20261009/epoch_045.pt" \
    --gdts-five-seed-result "$runtime/baseline/GDTS_ETH_SEEDS2035_2039.json" \
    --gdts-evaluator-script "$ebjd_root/scripts/evaluate_gdts_scene_manifest.py" \
    --gdts-root "$workspace/RSJG_JDV2_clean" \
    --gdts-config "$workspace/GDTS/output/eth/config.yaml" \
    --gdts-checkpoint "$workspace/GDTS/output/eth/saved_models/best_model.pt" \
    --gdts-manifest "$workspace/RSJG_EBJD_clean/ebjd_clean/data/eth_jmm_official_scene_shards_20261015/validation/validation_manifest.json" \
    >"$launch_log" 2>&1 < /dev/null

pipeline_pid=""
for _ in 1 2 3 4 5; do
  if [[ -f "$state_file" ]]; then
    pipeline_pid="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["supervisor_pid"])' "$state_file")"
    if kill -0 "$pipeline_pid" 2>/dev/null; then
      break
    fi
  fi
  sleep 1
done
if [[ -z "$pipeline_pid" ]] || ! kill -0 "$pipeline_pid" 2>/dev/null; then
  printf 'Detached supervisor did not remain alive; inspect %s\n' "$launch_log" >&2
  exit 1
fi
printf '%s\n' "$pipeline_pid" > "$pid_file"
printf 'ETH pipeline PID: %s\nstate: %s\nlog: %s\n' \
  "$pipeline_pid" "$state_file" "$launch_log"
