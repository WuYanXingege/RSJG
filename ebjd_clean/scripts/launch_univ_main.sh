#!/usr/bin/env bash
set -euo pipefail

action="${1:-launch}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ebjd_root="$repo/ebjd_clean"
workspace="$(dirname "$repo")"
python_bin="$workspace/.conda/rsjg/bin/python"
runtime="$workspace/RSJG_EBJD_clean/ebjd_clean/outputs/formal_univ_main_20261013_seed3101"
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
    printf 'UNIV pipeline is already running with PID %s\n' "$existing_pid"
    exit 0
  fi
fi

cd "$ebjd_root"
PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup "$python_bin" -u -B scripts/run_univ_main_pipeline.py \
    --python "$python_bin" \
    --runtime-dir "$runtime" \
    --goal-run-id univ_goal_pretrain_seed3101_20261013 \
    --hotel-checkpoint "$workspace/RSJG_EBJD_clean/ebjd_clean/outputs/formal_hotel_speed_9ecde66_20261009/hotel/3101/none/hotel_main_seed3101_9ecde66_speed_20261009/epoch_045.pt" \
    --gdts-five-seed-result "$workspace/RSJG_JDV2_clean/outputs/joint_dependency_v2/univ/joint_dependency_v2/audits/gdts_baseline_five_seed/results.json" \
    >"$launch_log" 2>&1 < /dev/null &
pipeline_pid=$!
printf '%s\n' "$pipeline_pid" > "$pid_file"
printf 'UNIV pipeline PID: %s\nstate: %s\nlog: %s\n' \
  "$pipeline_pid" "$state_file" "$launch_log"
