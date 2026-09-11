#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PROJECT_DIR}/../.conda/rsjg/bin/python"
TEST_SET="${1:?Usage: train_packed_multiway_v4.sh <eth|univ> [--worker]}"
if [[ "${TEST_SET}" != "eth" && "${TEST_SET}" != "univ" ]]; then
    echo "This formal launcher currently supports eth or univ." >&2
    exit 2
fi

RUN_NAME="multiway_coupling_v4_packed_cache_v2"
RUN_DIR="${PROJECT_DIR}/output/${TEST_SET}/joint/runs/${RUN_NAME}"
CACHE_ROOT="${PROJECT_DIR}/output/${TEST_SET}/joint/trajectory_bank_cache_v1/stage0_epoch100_s4_k20"
CHECKPOINT="${PROJECT_DIR}/output/${TEST_SET}/saved_models/epoch_100.pt"
PIPELINE_LOG="${RUN_DIR}/packed_v4_pipeline.log"
CACHE_LOG="${RUN_DIR}/trajectory_cache_build.log"
TRAIN_LOG="${RUN_DIR}/packed_v4_train.log"
TEST_LOG="${RUN_DIR}/packed_v4_test.log"
PID_FILE="${RUN_DIR}/packed_v4_pipeline.pid"
STATE_FILE="${RUN_DIR}/pipeline_state.txt"

COMMON_ARGS=(
    --dataset eth5
    --test_set "${TEST_SET}"
    --goal_model_type joint
    --run_name "${RUN_NAME}"
    --training_stage multiway_coupling
    --trajectory_coupling multiway_v4
    --upstream_generator stage0_independent
    --pretrain_path "${CHECKPOINT}"
    --freeze_upstream_generator True
    --continuous_refinement False
    --social_mode_scope off
    --num_samples 20
    --num_joint_samples 20
    --trajectory_dim 128
    --num_relation_modes 4
    --trajectory_pair_rank 8
    --trajectory_energy_type lowrank
    --trajectory_conditioned_relation True
    --pair_specific_gate True
    --graph_type radius_ttc
    --graph_radius 6.0
    --ttc_threshold 8.0
    --synchronizer multiway
    --sync_iterations 4
    --sinkhorn_iterations 8
    --sync_temperature 0.2
    --sync_inference_top_k_edges 0
    --hard_projection hungarian
    --use_identity_bypass True
    --keep_threshold 0.6
    --lambda_alignment 1.0
    --lambda_pair_score 0.5
    --lambda_pair_assignment 0.5
    --lambda_no_harm 2.0
    --lambda_perm_entropy 0.02
    --lambda_relation_prior 0.02
    --lambda_gate_reg 0.0
    --model_selection_split internal_train
    --internal_validation_strategy source_block
    --internal_validation_fraction 0.1
    --internal_validation_seed 2025
    --final_test_split heldout_test
    --learning_rate 3e-4
    --optimizer AdamW
    --scheduler CosineAnnealingLR
    --num_epochs 30
    --start_validation 1
    --validate_every 1
    --early_stopping_patience 5
    --save_every 5
    --coupling_grad_accum_steps 1
    --num_workers 2
    --data_augmentation False
    --best_metric JADE
    --use_wandb False
    --reproducibility True
    --seed 2025
    --use_trajectory_bank_cache True
    --use_multi_scene_packing True
    --scene_balanced_loss True
    --num_cached_seeds_per_window 4
    --trajectory_bank_seed_base 42
    --force_rebuild_trajectory_bank False
    --trajectory_bank_cache_root "${CACHE_ROOT}"
    --max_agents_per_pack 32
    --max_edges_per_pack 256
    --max_scenes_per_pack 8
    --fast_debug False
)

run_worker() {
    cd "${PROJECT_DIR}"
    printf 'cache_building %s\n' "$(date --iso-8601=seconds)" > "${STATE_FILE}"
    "${PYTHON_BIN}" -u main.py --phase trajectory_cache \
        "${COMMON_ARGS[@]}" > "${CACHE_LOG}" 2>&1

    printf 'training %s\n' "$(date --iso-8601=seconds)" > "${STATE_FILE}"
    "${PYTHON_BIN}" -u main.py --phase train \
        "${COMMON_ARGS[@]}" > "${TRAIN_LOG}" 2>&1

    printf 'testing %s\n' "$(date --iso-8601=seconds)" > "${STATE_FILE}"
    "${PYTHON_BIN}" -u main.py --phase test --load_checkpoint best \
        "${COMMON_ARGS[@]}" > "${TEST_LOG}" 2>&1

    "${PYTHON_BIN}" -u tools/generate_v4_report.py \
        --run_dir "${RUN_DIR}" \
        --output "${PROJECT_DIR}/docs/${TEST_SET^^}_MULTIWAY_COUPLING_V4_PACKED_RESULTS.md"
    printf 'complete %s\n' "$(date --iso-8601=seconds)" > "${STATE_FILE}"
}

if [[ "${2:-}" == "--worker" ]]; then
    run_worker
    exit 0
fi

mkdir -p "${RUN_DIR}"
if [[ -f "${PID_FILE}" ]]; then
    EXISTING_PID="$(tr -dc '0-9' < "${PID_FILE}")"
    if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "Packed V4 ${TEST_SET} pipeline is already running with PID ${EXISTING_PID}."
        exit 0
    fi
fi

nohup setsid "${PROJECT_DIR}/train_packed_multiway_v4.sh" \
    "${TEST_SET}" --worker >> "${PIPELINE_LOG}" 2>&1 < /dev/null &
PIPELINE_PID=$!
echo "${PIPELINE_PID}" > "${PID_FILE}"
echo "Started packed V4 ${TEST_SET} pipeline: PID=${PIPELINE_PID}"
echo "State: ${STATE_FILE}"
echo "Cache log: ${CACHE_LOG}"
echo "Train log: ${TRAIN_LOG}"
