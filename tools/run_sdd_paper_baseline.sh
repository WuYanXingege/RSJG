#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned SDD protocol:
#   1) Goal U-Net BCE pre-training for 150 epochs.
#   2) Fresh GDTS joint training for 250 epochs, initialized only from the
#      selected Goal U-Net checkpoint.
#
# The 30/17 TrajNet split validates on the published 17 test scenes. Metrics
# are marginal best-of-20 ADE/FDE in SDD pixels.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-outputs/joint_dependency_v2/cache/baseline_source_batches}"
GOAL_RUN_NAME="${GOAL_RUN_NAME:-gdts_paper_sdd_goal_pretrain_seed2025}"
JOINT_RUN_NAME="${JOINT_RUN_NAME:-gdts_paper_sdd_joint_seed2025}"
MODE="${1:-all}"
CHECKPOINT="${2:-last}"

GOAL_RUN_DIR="output/sdd/runs/${GOAL_RUN_NAME}"
JOINT_RUN_DIR="output/sdd/runs/${JOINT_RUN_NAME}"
GOAL_CHECKPOINT="${ROOT_DIR}/${GOAL_RUN_DIR}/saved_models/goal_pretrain_best_model.pt"

case "${MODE}" in
  pre-process|goal-pretrain|goal-resume|joint|joint-resume|test|all) ;;
  *)
    echo "Usage: $0 [pre-process|goal-pretrain|goal-resume|joint|joint-resume|test|all] [epoch|last]" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "goal-resume" || "${MODE}" == "joint-resume" ]]; then
  if [[ "${CHECKPOINT}" != "last" && ! "${CHECKPOINT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${MODE} requires 'last' or a positive numbered checkpoint epoch" >&2
    exit 2
  fi
fi

cd "${ROOT_DIR}"

common_args=(
  --dataset sdd
  --test_set sdd
  --goal_model_type independent
  --training_stage baseline
  --reproducibility True
  --seed 2025
  --validation_seed 2025
  --batch_size 64
  --start_validation 1
  --validate_every 10
  --learning_rate 0.001
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

write_provenance() {
  local run_dir="$1"
  local stage="$2"
  mkdir -p "${run_dir}"
  {
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse HEAD)"
    echo "dataset=sdd"
    echo "protocol=paper_aligned_30_17_validate_on_test"
    echo "stage=${stage}"
    echo "seed=2025"
    echo "learning_rate=0.001"
  } > "${run_dir}/source_provenance.txt"
}

run_preprocess() {
  "${PYTHON_BIN}" main.py --phase pre-process --run_name "${GOAL_RUN_NAME}"     --num_epochs 150 "${common_args[@]}"
}

run_goal_pretrain() {
  write_provenance "${GOAL_RUN_DIR}" "goal_pretrain_150"
  "${PYTHON_BIN}" main.py --phase goal_pretrain     --run_name "${GOAL_RUN_NAME}" --num_epochs 150     "${common_args[@]}"
}

run_goal_resume() {
  write_provenance "${GOAL_RUN_DIR}" "goal_pretrain_150_resume"
  "${PYTHON_BIN}" main.py --phase goal_pretrain     --run_name "${GOAL_RUN_NAME}" --num_epochs 150     --load_checkpoint "${CHECKPOINT}" "${common_args[@]}"
}

require_goal_checkpoint() {
  if [[ ! -f "${GOAL_CHECKPOINT}" ]]; then
    echo "Missing typed Goal U-Net checkpoint: ${GOAL_CHECKPOINT}" >&2
    exit 3
  fi
}

run_joint() {
  require_goal_checkpoint
  write_provenance "${JOINT_RUN_DIR}" "joint_training_250"
  "${PYTHON_BIN}" main.py --phase train_test     --run_name "${JOINT_RUN_NAME}" --num_epochs 250     --goal_pretrain_checkpoint "${GOAL_CHECKPOINT}"     "${common_args[@]}"
}

run_joint_resume() {
  require_goal_checkpoint
  write_provenance "${JOINT_RUN_DIR}" "joint_training_250_resume"
  "${PYTHON_BIN}" main.py --phase train     --run_name "${JOINT_RUN_NAME}" --num_epochs 250     --goal_pretrain_checkpoint "${GOAL_CHECKPOINT}"     --load_checkpoint "${CHECKPOINT}" "${common_args[@]}"
}

run_test() {
  require_goal_checkpoint
  "${PYTHON_BIN}" main.py --phase test     --run_name "${JOINT_RUN_NAME}" --num_epochs 250     --goal_pretrain_checkpoint "${GOAL_CHECKPOINT}"     --load_checkpoint best "${common_args[@]}"
}

case "${MODE}" in
  pre-process) run_preprocess ;;
  goal-pretrain) run_goal_pretrain ;;
  goal-resume) run_goal_resume ;;
  joint) run_joint ;;
  joint-resume) run_joint_resume ;;
  test) run_test ;;
  all)
    run_preprocess
    run_goal_pretrain
    run_joint
    ;;
esac
