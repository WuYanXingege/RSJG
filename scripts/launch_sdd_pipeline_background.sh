#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT="/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_Paper"
EXPERIMENT_ROOT="/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/sdd_jmm_unified_coupling_bridge_v1_20261103_155713"
PYTHON="/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python"

cd "${PAPER_ROOT}"
SOURCE_COMMIT="${1:-$(git rev-parse HEAD)}"
if [[ "$(git rev-parse HEAD)" != "${SOURCE_COMMIT}" ]]; then
  echo "Refusing launch: HEAD does not match source commit ${SOURCE_COMMIT}" >&2
  exit 2
fi
if pgrep -f "scripts/run_sdd_unified_pipeline.py --root ${EXPERIMENT_ROOT}" >/dev/null; then
  echo "Refusing launch: an SDD supervisor is already running" >&2
  exit 3
fi
if pgrep -f "scripts/run_legacy_gdts_stage.py.*${EXPERIMENT_ROOT}" >/dev/null; then
  echo "Refusing launch: an SDD matched-GDTS trainer is already running" >&2
  exit 4
fi
"${PYTHON}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'

LAST_CHECKPOINT="${EXPERIMENT_ROOT}/04_matched_gdts/runtime/output/sdd/runs/sdd_jmm_matched_gdts_seed3101_v1/saved_models/last_model.pt"
if [[ ! -f "${LAST_CHECKPOINT}" ]]; then
  echo "Refusing launch: complete matched-GDTS last checkpoint is absent" >&2
  exit 5
fi
PERSISTED_UPDATES="$(${PYTHON} -c "import torch; p=torch.load('${LAST_CHECKPOINT}',map_location='cpu',weights_only=False); print(int(p['epoch'])*2544)")"
LOG="${EXPERIMENT_ROOT}/logs/supervisor.log"
mkdir -p "$(dirname "${LOG}")"
setsid nohup "${PYTHON}" -u scripts/run_sdd_unified_pipeline.py \
  --root "${EXPERIMENT_ROOT}" \
  --source-commit "${SOURCE_COMMIT}" \
  --migration-reason "execution-only SDD input pipeline optimization" \
  --migration-optimizer-updates "${PERSISTED_UPDATES}" \
  >> "${LOG}" 2>&1 < /dev/null &
PID=$!
TEMPORARY="${EXPERIMENT_ROOT}/SUPERVISOR.pid.tmp"
printf '%s\n' "${PID}" > "${TEMPORARY}"
mv "${TEMPORARY}" "${EXPERIMENT_ROOT}/SUPERVISOR.pid"
echo "Started SDD supervisor PID=${PID}, commit=${SOURCE_COMMIT}, persisted_updates=${PERSISTED_UPDATES}"
