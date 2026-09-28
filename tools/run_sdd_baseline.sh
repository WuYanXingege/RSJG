#!/usr/bin/env bash
set -euo pipefail

# Reproduce the upstream GDTS SDD 30/17 baseline protocol. SDD uses pixel
# coordinates and validates on the same 17 scenes used for the final test.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python}"
RUN_NAME="${RUN_NAME:-gdts_baseline_sdd_seed2035}"
CACHE_ROOT="${CACHE_ROOT:-outputs/joint_dependency_v2/cache/baseline_source_batches}"
MODE="${1:-all}"

case "${MODE}" in
  pre-process|train|train_test|all) ;;
  *)
    echo "Usage: $0 [pre-process|train|train_test|all]" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"

common_args=(
  --dataset sdd
  --test_set sdd
  --goal_model_type independent
  --training_stage baseline
  --run_name "${RUN_NAME}"
  --reproducibility True
  --seed 2035
  --validation_seed 2035
  --num_epochs 300
  --batch_size 64
  --start_validation 5
  --validate_every 20
  --learning_rate 0.0001
  --optimizer Adam
  --scheduler ExponentialLR
  --skip_ts_window 1
  --down_factor 8
  --num_workers 1
  --data_augmentation True
  --num_samples 20
  --num_test_runs 5
  --amp_enabled False
  --amp_dtype fp32
  --compress_batch_cache True
  --batch_cache_root "${CACHE_ROOT}"
  --use_wandb False
)

run_phase() {
  local phase="$1"
  "${PYTHON_BIN}" main.py --phase "${phase}" "${common_args[@]}"
}

if [[ "${MODE}" == "all" ]]; then
  run_phase pre-process
  run_phase train_test
else
  run_phase "${MODE}"
fi
