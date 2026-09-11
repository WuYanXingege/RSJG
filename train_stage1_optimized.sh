#!/usr/bin/env bash
set -euo pipefail

# Stage-1 v2: retain GDTS candidate coverage, stabilize crowded-scene energy,
# normalize the variable-agent mode objective, and prevent latent-mode collapse.
# The named run writes only to output/eth/joint/runs/stage1_optimized_v1 and
# reuses output/eth/joint/data_batches_joint_v2.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/../.conda/rsjg/bin/python"
cd "$PROJECT_DIR"

exec "$PYTHON_BIN" main.py \
    --dataset eth5 \
    --test_set eth \
    --run_name stage1_optimized_v1 \
    --phase train_test \
    --goal_model_type joint \
    --training_stage joint \
    --pretrain_path output/eth/saved_models/best_model.pt \
    --device cuda:0 \
    --num_epochs 80 \
    --learning_rate 0.0001 \
    --scheduler ExponentialLR \
    --start_validation 5 \
    --validate_every 5 \
    --early_stopping_patience 4 \
    --save_every 5 \
    --best_metric ADE_world \
    --num_samples 20 \
    --num_goal_candidates 21 \
    --num_social_modes 8 \
    --num_relation_modes 4 \
    --goal_pair_rank 8 \
    --num_refinement_steps 2 \
    --joint_sampling_strategy coverage \
    --goal_candidate_prior heatmap \
    --pair_energy_normalization mean \
    --normalize_mode_loss_by_agents True \
    --lambda_goal 20 \
    --lambda_diff 1 \
    --lambda_mode 1 \
    --lambda_PL 1 \
    --lambda_relation_entropy 0.01 \
    --lambda_relation_balance 0.05 \
    --lambda_mode_balance 0.05 \
    --data_augmentation False \
    --use_wandb False \
    --num_workers 0
