#!/usr/bin/env bash
# Explicit invocation only. No resource polling, cron, or competing supervisor.
set -euo pipefail
if [[ $# != 1 ]]; then echo 'Usage: bash scripts/start_bcep_jmm.sh /absolute/RUN_PLAN.json' >&2; exit 2; fi
cd "$(dirname "$0")/.."
PYTHON="../.conda/rsjg/bin/python"
PLAN="$1"
ASSETS="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["assets"])' "$PLAN")"
# Existing formal execution uses the same frozen configs and complete checkpoints.
if [[ ! -f "$ASSETS/identity/EXECUTION_IDENTITY.json" ]]; then
  "$PYTHON" -B scripts/check_bcep_jmm.py --plan "$PLAN" --gpu
  "$PYTHON" -B scripts/finalize_bcep_jmm_runtime.py --plan "$PLAN"
fi
nohup setsid "$PYTHON" -B -u scripts/run_bcep_jmm.py --plan "$PLAN" --execute-authorized-fivefold </dev/null >>"$ASSETS/logs/pipeline.log" 2>&1 &
PID=$!
echo "Launch requested, PID=$PID; inspect $ASSETS/PIPELINE_STATE.json and logs/pipeline.log. Launch is not evidence of an optimizer update."
