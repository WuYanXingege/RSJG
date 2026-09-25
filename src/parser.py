import os
import sys
import argparse
import warnings
import math
import re

import torch
import yaml

sys.path.append('.')  # needed lo launch from .
sys.path.append('..')  # needed lo launch from ./scripts

from src.utils import str2bool


GOAL_MODEL_ALIASES = {
    'independent': 'independent',
    'gdts-base': 'independent',
    'base': 'independent',
    'social': 'social',
    'social-gdts': 'social',
    'lowrank': 'lowrank',
    'lr-goal': 'lowrank',
    'energy': 'energy',
    'sparse-energy': 'energy',
    'joint': 'joint',
    'full-joint': 'joint',
    'joint-dependency-v2': 'joint_dependency_v2',
    'jdv2': 'joint_dependency_v2',
    'rsjg-v2': 'joint_dependency_v2',
}

# Execution controls must come from the current command, not leak from a
# preprocessing/training invocation through the shared experiment config.
TRANSIENT_CONFIG_DEFAULTS = {
    'phase': 'train_test',
    'load_checkpoint': None,
    'pretrain_path': None,
    'force_reprocess': False,
    'jdv2_active': None,
}


def normalize_goal_model_type(value):
    """Normalize paper-facing ablation names to stable internal names."""
    key = str(value).strip().lower().replace('_', '-')
    if key not in GOAL_MODEL_ALIASES:
        allowed = ', '.join(sorted(GOAL_MODEL_ALIASES))
        raise argparse.ArgumentTypeError(
            f"Unknown goal model type '{value}'. Choose one of: {allowed}")
    return GOAL_MODEL_ALIASES[key]


def get_parser():
    parser = argparse.ArgumentParser(
        description='GDTS: Goal-Guided Diffusion Model with Tree Sampling for Multi-Modal Pedestrian Trajectory Prediction')

    ##############################################################
    # Dataset Split and Preprocessing
    ##############################################################
    parser.add_argument('--dataset', '-d', default='sdd', type=str,choices=['eth5', 'sdd', 'ind'])
    parser.add_argument('--test_set', '-ts', default='sdd', type=str,
        choices=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd', 'ind'])
    parser.add_argument(
        '--run_name', default=None,
        help=('Optional experiment subdirectory. Checkpoints/logs are isolated '
              'under runs/<name> while the preprocessed cache is shared.'))
    parser.add_argument('--skip_ts_window', '-skip', default=1, type=int,
        help="When extracting trajectory fragments, skip skip_ts_window time-steps between two consecutive starting_frames. "
             "If skip_ts_window >= seq_length there is no overlapping. If skip_ts_window == 1 make full use of the data.")
    parser.add_argument('--down_factor', default=8, type=int, help="Image down scale factor for CNN")
    ##############################################################
    # Training/testing parameters
    ##############################################################
    parser.add_argument('--phase', '-ph', default='train_test', type=str,
        choices=['pre-process', 'trajectory_cache', 'build-jdv2-cache',
                 'train', 'test', 'train_test', 'goal_pretrain'],
        help='Phase selection. During test phase you need to load a pre-trained model')
    parser.add_argument('--load_checkpoint', '-lc', default=None, type=str,
        help="Load pre-trained model for testing or resume training. Specify "
             "the epoch to load or 'best' to load the best model. Default=None means do not load any model.")
    parser.add_argument('--num_epochs', '-ne', default=300, type=int)
    parser.add_argument('--batch_size', '-bs', default=64, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='Adam', type=str,
                        choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--scheduler', default='ExponentialLR', type=str,
                        choices=['ExponentialLR', 'CosineAnnealingLR',
                                 'ReduceLROnPlateau', 'None'])
    parser.add_argument('--device', default="cuda:0", type=str, help='What device to use')
    parser.add_argument('--num_workers', '-nm', default=1, type=int)
    parser.add_argument('--clip', default=1, type=float, help="Gradient clip")
    parser.add_argument('--data_augmentation', default=True, type=str2bool, const=True,
        nargs='?', help="Apply data augmentation to the train set.")
    parser.add_argument('--save_every', '-se', default=None, type=int,
        help="Save model weights and outputs every save_every epochs. If None save every num_epochs//5 epochs.")
    parser.add_argument(
        '--batch_cache_root', default=None,
        help='Optional scratch root for generated batch caches; checkpoints '
             'and logs remain under output/.')
    parser.add_argument(
        '--compress_batch_cache', default=False, type=str2bool, const=True,
        nargs='?', help='Store generated batch pickles as lossless zstd '
                        'streams (recommended for SDD).')
    parser.add_argument('--start_validation', default=5, type=int, help="Validate the model starting from this epoch")
    parser.add_argument('--validate_every', '-ve', default=20, type=int, help="Validate model every validate_every epochs")
    parser.add_argument(
        '--early_stopping_patience', default=0, type=int,
        help=('Stop after this many validation checks without improvement in '
              'best_metric; 0 disables early stopping.'))
    parser.add_argument('--shuffle_train_batches', default=True, type=str2bool, const=True,
        nargs='?', help="Shuffle train batches. Set to False for deterministic behavior.")
    parser.add_argument('--shuffle_test_batches', default=False, type=str2bool, const=True,
        nargs='?', help="Shuffle valid and test batches. Set to False for deterministic behavior.")
    parser.add_argument(
        '--num_samples', default=20, type=int, help="Number of samples/modalities. Set to 1 for deterministic model")
    
    ##############################################################
    # Network parameters
    ##############################################################
    parser.add_argument(
        '--use_ttst', default=True, type=str2bool, const=True, nargs='?', help="Use Test Time Sampling Trick")
    parser.add_argument(
        '--e_dim', default=256, type=int, help="embedding dimension")
    parser.add_argument(
        '--ddpm_step', default=100, type=int, help="ddpm diffusion steps")
    parser.add_argument(
        '--ddim_step', default=20, type=int, help="ddim sampling steps")
    parser.add_argument(
        '--trunk_stage_step', default=30, type=int, help="trunk stage steps")

    ##############################################################
    # Relation-aware socially-coupled joint goal model
    ##############################################################
    parser.add_argument(
        '--goal_model_type', default='independent',
        type=normalize_goal_model_type,
        choices=['independent', 'social', 'lowrank', 'energy', 'joint',
                 'joint_dependency_v2'],
        help=("Goal ablation: independent (GDTS-Base), social "
              "(Social-GDTS), lowrank (LR-Goal), energy "
              "(Sparse-Energy), or joint (Full-Joint)."))
    parser.add_argument('--use_scene_latent', default=True, type=str2bool,
                        const=True, nargs='?')
    parser.add_argument('--use_dynamic_relation', default=True, type=str2bool,
                        const=True, nargs='?')
    parser.add_argument('--use_joint_energy', default=True, type=str2bool,
                        const=True, nargs='?')
    parser.add_argument('--use_dependency_corrector', default=True,
                        type=str2bool, const=True, nargs='?')
    parser.add_argument('--jdv2_active', default=None, type=str2bool,
                        help=argparse.SUPPRESS)
    parser.add_argument('--jdv2_scene_modes', default=4, type=int)
    parser.add_argument('--jdv2_relation_modes', default=4, type=int)
    parser.add_argument('--jdv2_energy_rank', default=8, type=int)
    parser.add_argument('--jdv2_edge_chunk_size', default=256, type=int)
    parser.add_argument('--jdv2_minimum_active_mode', default=False,
                        type=str2bool, const=True, nargs='?')
    parser.add_argument('--jdv2_cache_root', default=None)
    parser.add_argument('--jdv2_cache_schema', default='jdv2-cache-v1')
    parser.add_argument('--jdv2_source_checkpoint', default=None)
    parser.add_argument('--jdv2_cache_manifest_hash', default=None)
    parser.add_argument('--jdv2_source_checkpoint_hash', default=None)
    parser.add_argument('--stage_a_parent_checkpoint_sha256', default=None)
    parser.add_argument('--stage_a_freeze_manifest_sha256', default=None)
    parser.add_argument('--stage_a_freeze_manifest_path', default=None)
    parser.add_argument(
        '--cross_dataset_protocol_target', default=None,
        choices=['hotel', 'univ', 'zara1', 'zara2'],
        help=('Fail-closed target identity for JDV2 cross-dataset '
              'benchmark configurations.'))
    parser.add_argument('--stage_a_freeze_source_commit', default=None)
    parser.add_argument('--stage_b_architecture_version', default=None)
    parser.add_argument(
        '--jdv2_residual_projection', default='none',
        choices=['none', 'component_zero_mean'],
        help=('Stage-B residual routing. The default preserves the frozen '
              'V1 execution tensor-exactly.'))
    parser.add_argument(
        '--jdv2_cache_source_commit', default=None,
        help=('Explicitly pin the commit that built a reusable JDV2 cache. '
              'All data/checkpoint/graph fields remain strictly validated.'))
    parser.add_argument('--jdv2_stage_progress', default=0.0, type=float)
    parser.add_argument(
        '--jdv2_latent_objective',
        default='v2_marginal_responsibility',
        choices=['v2_marginal_responsibility', 'strict_no_z'],
        help=('Frozen Stage-A latent objective version stored in JDV2 '
              'checkpoints.'))
    parser.add_argument(
        '--jdv2_refinement_policy', default='categorical',
        choices=[
            'categorical',
            'structured_gumbel_assignment',
            'exact_lexicographic_persistent_tie',
        ],
        help=('Inference-only JDV2 refinement policy. The default preserves '
              'the historical categorical sampler tensor-exactly.'))
    parser.add_argument(
        '--jdv2_early_collapse_gate', default=True, type=str2bool,
        const=True, nargs='?',
        help=('Stop Stage-A V2 after two consecutive early E>0 collapse '
              'diagnostics; this never changes the objective.'))
    parser.add_argument('--lambda_JG', default=1.0, type=float)
    parser.add_argument('--lambda_relative', default=0.05, type=float)
    parser.add_argument('--amp_enabled', default=False, type=str2bool,
                        const=True, nargs='?')
    parser.add_argument('--amp_dtype', default='fp32',
                        choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument(
        '--use_social_encoder', default=True, type=str2bool, const=True,
        nargs='?', help='Use sparse social message passing after temporal encoding.')
    parser.add_argument(
        '--joint_goal_enabled', default=None, type=str2bool, const=True,
        nargs='?', help='Optional explicit joint sampler switch; normally derived from goal_model_type.')
    parser.add_argument('--social_feature_dim', default=128, type=int)
    parser.add_argument('--social_attention_layers', default=1, type=int)
    parser.add_argument('--num_social_modes', default=8, type=int)
    parser.add_argument('--num_relation_modes', default=4, type=int)
    parser.add_argument('--num_goal_candidates', default=21, type=int)
    parser.add_argument('--goal_pair_rank', default=8, type=int)
    parser.add_argument(
        '--graph_type', default='radius_ttc', type=str,
        choices=['full', 'radius', 'radius_only', 'radius_ttc'])
    parser.add_argument(
        '--graph_radius', default=6.0, type=float,
        help='Interaction radius in world-coordinate metres.')
    parser.add_argument(
        '--ttc_threshold', default=8.0, type=float,
        help='Approximate time-to-collision gate in seconds.')
    parser.add_argument('--num_joint_samples', default=None, type=int)
    parser.add_argument('--num_refinement_steps', default=2, type=int)
    parser.add_argument('--joint_sampling_temperature', default=1.0, type=float)
    parser.add_argument(
        '--joint_sampling_mode', default='sample', type=str,
        choices=['sample', 'map'])
    parser.add_argument(
        '--joint_sampling_strategy', default='coverage', type=str,
        choices=['iid', 'coverage'],
        help=('coverage preserves TTST candidate diversity across the returned '
              'sample set; iid reproduces categorical sampling with replacement.'))
    parser.add_argument(
        '--goal_candidate_prior', default='heatmap', type=str,
        choices=['uniform', 'heatmap'],
        help=('Prior over TTST representatives. uniform avoids double-counting '
              'heatmap density already encoded by TTST cluster locations.'))
    parser.add_argument('--goal_candidate_temperature', default=1.0, type=float)
    parser.add_argument('--energy_weight', default=1.0, type=float)
    parser.add_argument(
        '--pair_energy_normalization', default='mean', type=str,
        choices=['sum', 'mean'],
        help=('Normalize accumulated pair energy by weighted node degree so '
              'its scale is stable across sparse and crowded scenes.'))
    parser.add_argument(
        '--adaptive_graph', default=False, type=str2bool, const=True,
        nargs='?',
        help=('Learn a symmetric soft gate on radius/TTC candidate edges. '
              'Disabled by default for checkpoint compatibility.'))
    parser.add_argument(
        '--adaptive_graph_hidden_dim', default=64, type=int,
        help='Hidden width of the symmetric edge-gating MLP.')
    parser.add_argument(
        '--adaptive_graph_top_k', default=0, type=int,
        help=('Optional per-agent top-k edge proposals, symmetrized by union; '
              '0 keeps every radius/TTC candidate edge.'))
    parser.add_argument(
        '--adaptive_graph_gate_floor', default=0.05, type=float,
        help='Minimum learned weight for an edge retained by adaptive gating.')
    parser.add_argument(
        '--adaptive_graph_target_density', default=0.5, type=float,
        help='Target mean soft-gate value used by graph_density_loss.')
    parser.add_argument(
        '--lambda_graph_density', default=0.0, type=float,
        help='Weight of adaptive soft-gate density calibration; 0 disables it.')
    parser.add_argument(
        '--goal_soft_sigma', default=1.0, type=float,
        help='Soft endpoint target sigma in world-coordinate metres.')
    parser.add_argument('--lambda_goal', default=20.0, type=float)
    parser.add_argument('--lambda_mode', default=1.0, type=float)
    parser.add_argument('--lambda_PL', default=1.0, type=float)
    parser.add_argument(
        '--lambda_joint_rank', default=0.0, type=float,
        help=('Weight of scene-level listwise ranking over latent joint-goal '
              'proposals; 0 preserves the previous objective.'))
    parser.add_argument(
        '--joint_rank_target_temperature', default=0.5, type=float,
        help='Temperature (metres) for oracle joint-proposal ranking targets.')
    parser.add_argument(
        '--joint_rank_score_temperature', default=1.0, type=float,
        help='Temperature applied to predicted joint-proposal scores.')
    parser.add_argument('--lambda_diff', default=1.0, type=float)
    parser.add_argument('--lambda_relation_entropy', default=0.01, type=float)
    parser.add_argument('--lambda_relation_balance', default=0.05, type=float)
    parser.add_argument('--lambda_mode_balance', default=0.05, type=float)
    parser.add_argument(
        '--normalize_mode_loss_by_agents', default=True, type=str2bool,
        const=True, nargs='?',
        help=('Use a per-agent scene log-likelihood so large crowds do not '
              'dominate every gradient update.'))
    parser.add_argument(
        '--hard_relation', default=False, type=str2bool, const=True, nargs='?')
    parser.add_argument(
        '--use_group_relative_feature', default=True, type=str2bool,
        const=True, nargs='?')
    parser.add_argument(
        '--pairwise_neighbor_target', default='soft', type=str,
        choices=['soft', 'nearest'])
    parser.add_argument(
        '--training_stage', default='finetune', type=str,
        choices=['baseline', 'joint', 'finetune', 'alignment',
                 'multiway_coupling', 'joint_goal', 'joint_trajectory',
                 'joint_finetune'],
        help=('baseline reproduces GDTS; joint freezes GDTS; finetune trains '
              'all modules; alignment freezes the trajectory generator and '
              'trains the reference-centric V3 aligner; multiway_coupling '
              'freezes the selected upstream generator by default and trains '
              'the graph-wide V4 coupling network.'))
    parser.add_argument(
        '--baseline_lr_scale', default=1.0, type=float,
        help=('Learning-rate multiplier for the original GDTS U-Net, history '
              'encoder and denoiser during finetuning.'))
    parser.add_argument(
        '--structured_lr_scale', default=1.0, type=float,
        help='Learning-rate multiplier for the social/joint modules.')
    parser.add_argument(
        '--baseline_unfreeze_epoch', default=1, type=int,
        help=('During finetuning, keep original GDTS modules frozen before '
              'this run-local epoch; 1 disables delayed unfreezing.'))
    parser.add_argument(
        '--continuous_refinement', default=False, type=str2bool, const=True,
        nargs='?',
        help=('After discrete joint candidate selection, apply a learned '
              'bounded continuous endpoint residual.'))
    parser.add_argument(
        '--continuous_refinement_steps', default=1, type=int,
        help='Number of recurrent continuous endpoint correction steps.')
    parser.add_argument(
        '--continuous_refinement_max_delta', default=0.5, type=float,
        help='Maximum total continuous endpoint displacement in world metres.')
    parser.add_argument(
        '--lambda_continuous_refinement', default=0.0, type=float,
        help='Weight of best-joint-proposal endpoint refinement loss.')
    parser.add_argument(
        '--lambda_refinement_delta', default=0.0, type=float,
        help='Weight of the continuous residual magnitude regularizer.')
    parser.add_argument(
        '--trajectory_alignment', default=False, type=str2bool, const=True,
        nargs='?',
        help=('Relation-aware post-diffusion alignment of complete trajectory '
              'samples. Hard inference only permutes each agent\'s K samples, '
              'so marginal minADE/minFDE are exactly preserved.'))
    parser.add_argument('--trajectory_alignment_hidden_dim', default=128,
                        type=int)
    parser.add_argument('--trajectory_alignment_sinkhorn_iters', default=8,
                        type=int)
    parser.add_argument('--trajectory_alignment_steps', default=2, type=int)
    parser.add_argument('--trajectory_alignment_temperature', default=0.15,
                        type=float)
    parser.add_argument('--trajectory_alignment_target_temperature',
                        default=0.5, type=float)
    parser.add_argument('--trajectory_alignment_fde_weight', default=1.0,
                        type=float)
    parser.add_argument('--trajectory_alignment_keep_threshold', default=0.6,
                        type=float)
    parser.add_argument('--trajectory_alignment_top_k_edges', default=4,
                        type=int)
    parser.add_argument('--lambda_trajectory_alignment', default=0.0,
                        type=float,
                        help='Weight of the soft JMM-shaped trajectory loss.')
    parser.add_argument('--lambda_trajectory_pair', default=0.0, type=float,
                        help='Weight of relation-aware pair-ranking supervision.')
    parser.add_argument('--lambda_alignment_no_harm', default=0.0, type=float,
                        help='Penalty when soft alignment is worse than identity.')
    parser.add_argument('--lambda_alignment_entropy', default=0.0, type=float,
                        help='Weight making Sinkhorn assignments near-discrete.')

    ##############################################################
    # Graph-wide multiway trajectory coupling (V4)
    ##############################################################
    parser.add_argument(
        '--trajectory_coupling', default='none', type=str,
        choices=['none', 'reference_v3', 'multiway_v4'],
        help=('Post-diffusion coupling implementation. reference_v3 preserves '
              'the historical anchor-based aligner; multiway_v4 jointly '
              'synchronizes permutations over every graph edge.'))
    parser.add_argument(
        '--upstream_generator', default='stage0_independent', type=str,
        choices=['stage0_independent', 'stage1_joint'],
        help='Frozen marginal trajectory generator used by V4.')
    parser.add_argument('--trajectory_conditioned_relation', default=True,
                        type=str2bool, const=True, nargs='?')
    parser.add_argument('--pair_specific_gate', default=True, type=str2bool,
                        const=True, nargs='?')
    parser.add_argument('--trajectory_energy_type', default='lowrank',
                        choices=['mlp', 'lowrank'])
    parser.add_argument('--trajectory_dim', default=128, type=int)
    parser.add_argument('--trajectory_pair_rank', default=8, type=int,
                        choices=[4, 8, 16])
    parser.add_argument('--synchronizer', default='multiway',
                        choices=['reference', 'multiway'])
    parser.add_argument('--sync_iterations', default=4, type=int,
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument('--sinkhorn_iterations', default=8, type=int,
                        choices=[4, 8, 16])
    parser.add_argument('--sync_temperature', default=0.2, type=float)
    parser.add_argument(
        '--sync_inference_top_k_edges', default=0, type=int,
        help=('Optional per-agent edge pruning at hard inference only; 0 '
              'keeps the full graph. Training always uses every edge.'))
    parser.add_argument('--hard_projection', default='hungarian',
                        choices=['greedy', 'hungarian'])
    parser.add_argument('--use_identity_bypass', default=True, type=str2bool,
                        const=True, nargs='?')
    parser.add_argument('--keep_threshold', default=0.6, type=float)
    parser.add_argument('--use_pair_assignment_loss', default=True,
                        type=str2bool, const=True, nargs='?')
    parser.add_argument('--lambda_alignment', default=1.0, type=float)
    parser.add_argument('--lambda_pair_score', default=0.5, type=float)
    parser.add_argument('--lambda_pair_assignment', default=0.5, type=float)
    parser.add_argument('--lambda_no_harm', default=2.0, type=float)
    parser.add_argument('--lambda_perm_entropy', default=0.02, type=float)
    parser.add_argument('--lambda_relation_prior', default=0.02, type=float)
    parser.add_argument('--lambda_gate_reg', default=0.0, type=float)
    parser.add_argument('--coupling_fde_weight', default=1.0, type=float)
    parser.add_argument('--coupling_rel_geom_weight', default=1.0, type=float)
    parser.add_argument('--coupling_rel_endpoint_weight', default=1.0,
                        type=float)
    parser.add_argument('--coupling_joint_temperature', default=0.5,
                        type=float)
    parser.add_argument('--coupling_pair_target_temperature', default=0.5,
                        type=float)
    parser.add_argument('--coupling_pair_pred_temperature', default=0.2,
                        type=float)
    parser.add_argument('--coupling_no_harm_margin', default=0.0, type=float)
    parser.add_argument('--freeze_upstream_generator', default=True,
                        type=str2bool, const=True, nargs='?')
    parser.add_argument('--use_component_mode', default=False, type=str2bool,
                        const=True, nargs='?')
    parser.add_argument('--social_mode_scope', default='off',
                        choices=['off', 'scene', 'component'])
    parser.add_argument('--coupling_grad_accum_steps', default=1, type=int)
    parser.add_argument('--cache_frozen_trajectory_banks', default=False,
                        type=str2bool, const=True, nargs='?')
    parser.add_argument(
        '--use_trajectory_bank_cache', default=False, type=str2bool,
        const=True, nargs='?',
        help=('Read frozen upstream K-trajectory banks instead of running '
              'diffusion during V4 training/evaluation.'))
    parser.add_argument(
        '--use_multi_scene_packing', default=False, type=str2bool,
        const=True, nargs='?',
        help=('Pack multiple variable-N synchronized windows along the agent '
              'axis. Requires use_trajectory_bank_cache.'))
    parser.add_argument(
        '--scene_balanced_loss', default=False, type=str2bool,
        const=True, nargs='?',
        help=('Reduce every V4 loss within each scene, then average scenes. '
              'Required for formal packed V4 training.'))
    parser.add_argument('--num_cached_seeds_per_window', default=4, type=int)
    parser.add_argument('--trajectory_bank_seed_base', default=42, type=int)
    parser.add_argument(
        '--force_rebuild_trajectory_bank', default=False, type=str2bool,
        const=True, nargs='?',
        help=('Move an incompatible trajectory-bank cache aside and rebuild '
              'it. Never silently accepts a mismatched manifest.'))
    parser.add_argument(
        '--trajectory_bank_cache_root', default=None,
        help=('Optional explicit root for frozen trajectory-bank records. '
              'The default is shared by compatible runs under output/.'))
    parser.add_argument('--max_agents_per_pack', default=32, type=int)
    parser.add_argument('--max_edges_per_pack', default=256, type=int)
    parser.add_argument('--max_scenes_per_pack', default=8, type=int)
    parser.add_argument('--trajectory_dt', default=0.4, type=float)
    parser.add_argument(
        '--model_selection_split', default='auto',
        choices=['auto', 'dataset_valid', 'internal_train'],
        help=('Checkpoint-selection split. V4 auto resolves to a deterministic '
              'held-out block of the training cache, never the final test '
              'scene.'))
    parser.add_argument('--internal_validation_fraction', default=0.1,
                        type=float)
    parser.add_argument('--internal_validation_seed', default=2025, type=int)
    parser.add_argument(
        '--internal_validation_strategy', default='source_block',
        choices=['source_block', 'window_random'],
        help=('source_block holds out the final complete raw training file '
              '(preferred, prevents overlapping-window leakage); '
              'window_random is a deterministic fallback.'))
    parser.add_argument('--final_test_split', default='heldout_test',
                        choices=['heldout_test'])
    parser.add_argument(
        '--clean_split_protocol', default=False, type=str2bool, const=True,
        nargs='?', help='Enable the immutable JDV2 clean-split protocol.')
    parser.add_argument('--clean_split_manifest_path', default=None)
    parser.add_argument('--clean_split_manifest_hash', default=None)
    parser.add_argument(
        '--final_test_access', default='legacy',
        choices=['legacy', 'blocked', 'authorized'],
        help=('Clean runs keep final-test access blocked until an explicit '
              'checkpoint/config lock authorizes the one final evaluation.'))
    parser.add_argument('--final_test_lock_path', default=None)
    parser.add_argument('--clean_evaluation_target', default=None,
                        choices=['stage_a', 'stage_b'])
    parser.add_argument(
        '--best_metric', default='auto', type=str,
        choices=['auto', 'ADE', 'ADE_world', 'JADE', 'JFDE'],
        help=('Validation metric used for best_model.pt. auto keeps ADE for '
              'GDTS-Base and uses world-coordinate JADE for social models.'))
    parser.add_argument(
        '--collision_threshold_meter', default=None, type=float,
        help='World-space collision threshold. No unverified dataset default is assumed.')
    parser.add_argument('--collision_interpolation_steps', default=2, type=int)
    parser.add_argument(
        '--goal_recall_threshold_meter', default=None, type=float,
        help='World-space endpoint threshold used for Goal Recall@K.')
    parser.add_argument(
        '--force_reprocess', default=False, type=str2bool, const=True,
        nargs='?', help='Rebuild cached trajectory batches for the selected model path.')
    parser.add_argument(
        '--joint_diagnostics', default=True, type=str2bool, const=True,
        nargs='?', help='Log compact graph/mode/relation/energy diagnostics.')
    

    ##############################################################
    # Debug parameters
    ##############################################################
    parser.add_argument(
        '--use_wandb', default=True, type=str2bool, const=True, nargs='?')
    parser.add_argument(
        '--num_test_runs', default=5, type=int, help="Number of test run to average")
    parser.add_argument(
        '--fast_debug', '-fd', default=False, type=str2bool, const=True, nargs='?', help="Set to True for fast debug mode")
    parser.add_argument(
        '--fast_debug_num', default=3, type=int, help="Number of batches in fast debug mode")
    parser.add_argument(
        '--reproducibility', '-r', default=True, type=str2bool, const=True,
        nargs='?', help="Set to True to set the seed for reproducibility")
    parser.add_argument(
        '--seed', default=2025, type=int, help="Random seed for reproducibility")
    parser.add_argument(
        '--validation_seed', default=None, type=int,
        help=('Fixed seed for an isolated stochastic validation stream. '
              'Defaults to --seed and does not alter deployed-policy sampling.'))
    parser.add_argument(
        '--pretrain_path', default=None, help="Path to the pre-trained model checkpoint")
    # parser.add_argument(
    #     '--if_plot', default=False, type=str2bool, const=True, nargs='?', help="Set to True to plot the results")
    return parser


def load_args(cur_parser, parsed_args):
    """
    Load args from saved config file and confront them with parsed args.
    parsed_args are the entered parsed arguments for this run, while saved_args
    are the previously saved config arguments.

    The priority is:
    command line > saved configuration files > default values in script.
    """
    with open(parsed_args.config, 'r') as f:
        saved_args = yaml.full_load(f)
    if not isinstance(saved_args, dict):
        raise ValueError(f'Configuration {parsed_args.config} is not a mapping')
    saved_args = dict(saved_args)
    # These two switches are derived from primary CLI arguments. Persisting
    # their computed values makes changing --num_samples or --goal_model_type
    # alone conflict with an old config even though no real choice changed.
    saved_args['num_joint_samples'] = None
    saved_args['joint_goal_enabled'] = None
    saved_args.update(TRANSIENT_CONFIG_DEFAULTS)
    current_keys = set(vars(parsed_args))
    unknown = set(saved_args) - current_keys
    if unknown:
        raise KeyError('WRONG ARG(S): {}'.format(', '.join(sorted(unknown))))
    # Older baseline YAML files legitimately lack newly introduced joint-goal
    # options.  Missing keys retain argparse defaults instead of invalidating
    # an otherwise loadable GDTS checkpoint.
    cur_parser.set_defaults(**saved_args)
    return cur_parser.parse_args()


def save_args(args):
    """
    Save args to config file
    """
    args_dict = dict(vars(args))
    args_dict['num_joint_samples'] = None
    args_dict['joint_goal_enabled'] = None
    args_dict.update(TRANSIENT_CONFIG_DEFAULTS)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.config, 'w') as f:
        yaml.dump(args_dict, f)


def check_and_add_additional_args(args):
    """
    Add default paths, device and other additional args to parsed args
    """
    args.goal_model_type = normalize_goal_model_type(args.goal_model_type)
    if args.validation_seed is None:
        args.validation_seed = args.seed
    args.jdv2_active = bool(
        args.use_scene_latent or args.use_dynamic_relation or
        args.use_joint_energy or args.use_dependency_corrector)
    if (args.run_name is not None and
            re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name) is None):
        raise ValueError(
            'run_name must start with a letter/digit and contain only '
            'letters, digits, underscore, hyphen, or dot')
    integer_minimums = {
        'skip_ts_window': (args.skip_ts_window, 1),
        'down_factor': (args.down_factor, 1),
        'batch_size': (args.batch_size, 1),
        'num_workers': (args.num_workers, 0),
        'num_epochs': (args.num_epochs, 1),
        'validate_every': (args.validate_every, 1),
        'start_validation': (args.start_validation, 0),
        'early_stopping_patience': (args.early_stopping_patience, 0),
        'num_test_runs': (args.num_test_runs, 1),
        'validation_seed': (args.validation_seed, 0),
        'ddpm_step': (args.ddpm_step, 1),
        'ddim_step': (args.ddim_step, 1),
        'trunk_stage_step': (args.trunk_stage_step, 0),
        'adaptive_graph_hidden_dim': (args.adaptive_graph_hidden_dim, 1),
        'adaptive_graph_top_k': (args.adaptive_graph_top_k, 0),
        'baseline_unfreeze_epoch': (args.baseline_unfreeze_epoch, 1),
        'continuous_refinement_steps': (
            args.continuous_refinement_steps, 1),
        'trajectory_alignment_hidden_dim': (
            args.trajectory_alignment_hidden_dim, 1),
        'trajectory_alignment_sinkhorn_iters': (
            args.trajectory_alignment_sinkhorn_iters, 1),
        'trajectory_alignment_steps': (args.trajectory_alignment_steps, 1),
        'trajectory_alignment_top_k_edges': (
            args.trajectory_alignment_top_k_edges, 0),
        'trajectory_dim': (args.trajectory_dim, 1),
        'sync_iterations': (args.sync_iterations, 1),
        'sinkhorn_iterations': (args.sinkhorn_iterations, 1),
        'sync_inference_top_k_edges': (
            args.sync_inference_top_k_edges, 0),
        'coupling_grad_accum_steps': (args.coupling_grad_accum_steps, 1),
        'num_cached_seeds_per_window': (
            args.num_cached_seeds_per_window, 1),
        'max_agents_per_pack': (args.max_agents_per_pack, 1),
        'max_edges_per_pack': (args.max_edges_per_pack, 0),
        'max_scenes_per_pack': (args.max_scenes_per_pack, 1),
        'jdv2_scene_modes': (args.jdv2_scene_modes, 1),
        'jdv2_relation_modes': (args.jdv2_relation_modes, 1),
        'jdv2_energy_rank': (args.jdv2_energy_rank, 1),
        'jdv2_edge_chunk_size': (args.jdv2_edge_chunk_size, 1),
    }
    invalid_integers = [
        name for name, (value, minimum) in integer_minimums.items()
        if value < minimum]
    if invalid_integers:
        raise ValueError(
            'Invalid integer range for: ' + ', '.join(invalid_integers))
    if args.ddim_step > args.ddpm_step or \
            args.trunk_stage_step > args.ddpm_step:
        raise ValueError(
            'ddim_step and trunk_stage_step cannot exceed ddpm_step')
    if args.e_dim < 2 or args.e_dim % 2:
        raise ValueError('e_dim must be a positive even integer >= 2')
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not math.isfinite(args.clip) or args.clip < 0):
        raise ValueError('learning_rate must be positive and clip non-negative')
    if args.save_every is not None and args.save_every <= 0:
        raise ValueError('save_every must be positive when specified')
    needs_synchronized_scene = (
        args.goal_model_type != 'independent' and not (
            args.goal_model_type == 'joint_dependency_v2' and
            not args.jdv2_active))
    if needs_synchronized_scene and args.data_augmentation:
        # Legacy pixel-space augmentation does not update a scene's metric
        # homography. Using it for TTC/energy/metric losses would silently mix
        # transformed pixels with the original world-coordinate calibration.
        warnings.warn(
            'Disabling legacy data augmentation for social goal models: '
            'its pixel transform is not homography-aware.', UserWarning)
        args.data_augmentation = False
    if args.num_goal_candidates < 1:
        raise ValueError('num_goal_candidates must be >= 1')
    if args.num_samples < 1:
        raise ValueError('num_samples must be >= 1')
    if args.num_social_modes < 1 or args.num_relation_modes < 1:
        raise ValueError('num_social_modes and num_relation_modes must be >= 1')
    if (args.social_feature_dim < 1 or args.social_attention_layers < 0 or
            args.goal_pair_rank < 1 or args.num_refinement_steps < 0):
        raise ValueError('goal_pair_rank must be >= 1 and refinement steps >= 0')
    if (not math.isfinite(args.graph_radius) or
            not math.isfinite(args.ttc_threshold) or
            args.graph_radius <= 0 or args.ttc_threshold <= 0):
        raise ValueError('graph_radius and ttc_threshold must be positive')
    positive_values = (
        args.goal_soft_sigma, args.joint_sampling_temperature,
        args.goal_candidate_temperature, args.joint_rank_target_temperature,
        args.joint_rank_score_temperature,
        args.continuous_refinement_max_delta,
        args.trajectory_alignment_temperature,
        args.trajectory_alignment_target_temperature,
        args.sync_temperature,
        args.coupling_joint_temperature,
        args.coupling_pair_target_temperature,
        args.coupling_pair_pred_temperature,
        args.trajectory_dt)
    if any(not math.isfinite(value) or value <= 0
           for value in positive_values):
        raise ValueError(
            'goal sigma and candidate/sampling temperatures must be positive')
    nonnegative = {
        'energy_weight': args.energy_weight,
        'lambda_goal': args.lambda_goal,
        'lambda_mode': args.lambda_mode,
        'lambda_PL': args.lambda_PL,
        'lambda_joint_rank': args.lambda_joint_rank,
        'lambda_diff': args.lambda_diff,
        'lambda_JG': args.lambda_JG,
        'lambda_relative': args.lambda_relative,
        'lambda_relation_entropy': args.lambda_relation_entropy,
        'lambda_relation_balance': args.lambda_relation_balance,
        'lambda_mode_balance': args.lambda_mode_balance,
        'lambda_graph_density': args.lambda_graph_density,
        'lambda_continuous_refinement': (
            args.lambda_continuous_refinement),
        'lambda_refinement_delta': args.lambda_refinement_delta,
        'trajectory_alignment_fde_weight': (
            args.trajectory_alignment_fde_weight),
        'lambda_trajectory_alignment': args.lambda_trajectory_alignment,
        'lambda_trajectory_pair': args.lambda_trajectory_pair,
        'lambda_alignment_no_harm': args.lambda_alignment_no_harm,
        'lambda_alignment_entropy': args.lambda_alignment_entropy,
        'lambda_alignment': args.lambda_alignment,
        'lambda_pair_score': args.lambda_pair_score,
        'lambda_pair_assignment': args.lambda_pair_assignment,
        'lambda_no_harm': args.lambda_no_harm,
        'lambda_perm_entropy': args.lambda_perm_entropy,
        'lambda_relation_prior': args.lambda_relation_prior,
        'lambda_gate_reg': args.lambda_gate_reg,
        'coupling_fde_weight': args.coupling_fde_weight,
        'coupling_rel_geom_weight': args.coupling_rel_geom_weight,
        'coupling_rel_endpoint_weight': (
            args.coupling_rel_endpoint_weight),
        'coupling_no_harm_margin': args.coupling_no_harm_margin,
    }
    invalid_nonnegative = [
        name for name, value in nonnegative.items()
        if not math.isfinite(value) or value < 0]
    if invalid_nonnegative:
        raise ValueError(
            'These weights must be non-negative: ' +
            ', '.join(invalid_nonnegative))
    if not 0.0 <= args.jdv2_stage_progress <= 1.0:
        raise ValueError('jdv2_stage_progress must lie in [0,1]')
    if args.goal_model_type == 'joint_dependency_v2':
        if args.jdv2_cache_schema != 'jdv2-cache-v1':
            raise ValueError('Unsupported jdv2_cache_schema')
        if args.jdv2_latent_objective == 'strict_no_z':
            if args.use_scene_latent:
                raise ValueError(
                    'strict_no_z requires use_scene_latent=False')
            if not args.use_dynamic_relation or not args.use_joint_energy:
                raise ValueError(
                    'strict_no_z requires dynamic relation and joint energy')
            if args.training_stage not in {'joint_goal', 'joint_trajectory'}:
                raise ValueError(
                    'strict_no_z authorizes only frozen Stage-A joint_goal '
                    'or Stage-B joint_trajectory')
        if args.jdv2_refinement_policy in {
                'structured_gumbel_assignment',
                'exact_lexicographic_persistent_tie'}:
            if args.jdv2_latent_objective != 'strict_no_z':
                raise ValueError('structured refinement requires strict_no_z')
            if args.num_samples > args.num_goal_candidates:
                raise ValueError('structured refinement requires P<=K')
            if args.num_refinement_steps != 2:
                raise ValueError(
                    'structured refinement requires two refinement rounds')
            if args.joint_sampling_mode != 'sample':
                raise ValueError(
                    'structured refinement requires stochastic '
                    'joint_sampling_mode=sample')
        if (args.jdv2_cache_source_commit is not None and
                re.fullmatch(r'[0-9a-f]{40}',
                             args.jdv2_cache_source_commit) is None):
            raise ValueError(
                'jdv2_cache_source_commit must be a full lowercase git SHA')
        for name in (
                'stage_a_parent_checkpoint_sha256',
                'stage_a_freeze_manifest_sha256'):
            value = getattr(args, name)
            if value is not None and re.fullmatch(r'[0-9a-f]{64}', value) is None:
                raise ValueError(f'{name} must be a lowercase SHA256')
        if (args.stage_a_freeze_source_commit is not None and
                re.fullmatch(r'[0-9a-f]{40}',
                             args.stage_a_freeze_source_commit) is None):
            raise ValueError(
                'stage_a_freeze_source_commit must be a full lowercase git SHA')
        if (args.stage_b_architecture_version is not None and
                args.stage_b_architecture_version not in {
                    'jdv2-stage-b-v1', 'jdv2-stage-b-v2a'}):
            raise ValueError('Unsupported stage_b_architecture_version')
        if args.training_stage == 'joint_trajectory' and \
                args.stage_b_architecture_version is not None:
            if args.jdv2_latent_objective != 'strict_no_z':
                raise ValueError('Stage-B requires strict_no_z')
            if args.jdv2_refinement_policy != \
                    'exact_lexicographic_persistent_tie':
                raise ValueError(
                    'Stage-B requires the frozen exact Stage-A sampler')
            expected_projection = (
                'component_zero_mean'
                if args.stage_b_architecture_version == 'jdv2-stage-b-v2a'
                else 'none')
            if args.jdv2_residual_projection != expected_projection:
                raise ValueError(
                    f'{args.stage_b_architecture_version} requires '
                    f'jdv2_residual_projection={expected_projection}')
        elif args.jdv2_residual_projection != 'none':
            raise ValueError(
                'component residual projection is Stage-B-only')
        if (args.phase == 'build-jdv2-cache' and
                args.training_stage not in {
                    'joint_goal', 'joint_trajectory', 'joint_finetune'}):
            args.training_stage = 'joint_goal'
        frozen = {
            'jdv2_scene_modes': (args.jdv2_scene_modes, 4),
            'jdv2_relation_modes': (args.jdv2_relation_modes, 4),
            'jdv2_energy_rank': (args.jdv2_energy_rank, 8),
            'num_goal_candidates': (args.num_goal_candidates, 21),
            'num_samples': (args.num_samples, 20),
            'num_refinement_steps': (args.num_refinement_steps, 2),
            'social_feature_dim': (args.social_feature_dim, 128),
        }
        invalid = [name for name, (value, expected) in frozen.items()
                   if value != expected]
        if invalid:
            details = ', '.join(
                f'{name}={frozen[name][0]} (required {frozen[name][1]})'
                for name in invalid)
            raise ValueError('Invalid frozen JDV2 dimensions: ' + details)
        v2_stages = {'joint_goal', 'joint_trajectory', 'joint_finetune'}
        if args.jdv2_active and args.training_stage not in v2_stages:
            raise ValueError(
                'Active joint_dependency_v2 requires training_stage in '
                'joint_goal, joint_trajectory, joint_finetune')
        if not args.jdv2_active and args.training_stage in v2_stages:
            raise ValueError(
                'JDV2 training stages are invalid when all four modules are '
                'disabled; use legacy baseline semantics')
        if args.phase == 'build-jdv2-cache' and not args.jdv2_active:
            raise ValueError('build-jdv2-cache requires active JDV2 modules')
        if args.adaptive_graph:
            raise ValueError(
                'Canonical JDV2 uses the parameter-free sparse proposal graph')
        if (args.training_stage == 'joint_trajectory' and
                not args.use_dependency_corrector):
            raise ValueError(
                'joint_trajectory requires use_dependency_corrector=True')
    elif args.training_stage in {
            'joint_goal', 'joint_trajectory', 'joint_finetune'}:
        raise ValueError(
            'JDV2 training stages require goal_model_type=joint_dependency_v2')
    if args.amp_enabled and args.amp_dtype == 'fp32':
        raise ValueError('amp_enabled requires amp_dtype=fp16 or bf16')
    if not args.amp_enabled and args.amp_dtype != 'fp32':
        raise ValueError('amp_dtype requires amp_enabled=True')
    positive_lr_scales = {
        'baseline_lr_scale': args.baseline_lr_scale,
        'structured_lr_scale': args.structured_lr_scale,
    }
    invalid_lr_scales = [
        name for name, value in positive_lr_scales.items()
        if not math.isfinite(value) or value <= 0]
    if invalid_lr_scales:
        raise ValueError(
            'Learning-rate scales must be positive: ' +
            ', '.join(invalid_lr_scales))
    unit_interval_values = {
        'adaptive_graph_gate_floor': args.adaptive_graph_gate_floor,
        'adaptive_graph_target_density': (
            args.adaptive_graph_target_density),
    }
    invalid_unit_interval = [
        name for name, value in unit_interval_values.items()
        if not math.isfinite(value) or value < 0 or value >= 1]
    if invalid_unit_interval:
        raise ValueError(
            'Adaptive graph gate values must lie in [0, 1): ' +
            ', '.join(invalid_unit_interval))
    if args.continuous_refinement and args.goal_model_type == 'independent':
        raise ValueError(
            'continuous_refinement requires a social goal model')
    if args.adaptive_graph and args.goal_model_type == 'independent':
        raise ValueError('adaptive_graph requires a social goal model')
    if args.trajectory_alignment and args.goal_model_type == 'independent':
        raise ValueError(
            'trajectory_alignment requires a social goal model')
    # Preserve the old boolean CLI while making the coupling implementation
    # explicit for all new configs.
    if args.trajectory_coupling == 'reference_v3':
        args.trajectory_alignment = True
    elif args.trajectory_alignment and args.trajectory_coupling == 'none':
        args.trajectory_coupling = 'reference_v3'
    if args.training_stage == 'multiway_coupling':
        if args.trajectory_coupling != 'multiway_v4':
            raise ValueError(
                'training_stage=multiway_coupling requires '
                'trajectory_coupling=multiway_v4')
        if args.synchronizer != 'multiway':
            raise ValueError(
                'The formal V4 stage requires synchronizer=multiway; use '
                'the reference_v3 path for the anchor ablation.')
        if args.continuous_refinement:
            raise ValueError(
                'V4 must not move trajectory coordinates; keep '
                'continuous_refinement=False.')
        if args.use_multi_scene_packing and not \
                args.use_trajectory_bank_cache:
            raise ValueError(
                'use_multi_scene_packing requires '
                'use_trajectory_bank_cache=True')
        if args.use_multi_scene_packing and not args.scene_balanced_loss:
            raise ValueError(
                'Formal multi-scene packing requires '
                'scene_balanced_loss=True')
    elif args.trajectory_coupling == 'multiway_v4':
        raise ValueError(
            'trajectory_coupling=multiway_v4 requires '
            'training_stage=multiway_coupling')
    if args.training_stage == 'alignment' and not args.trajectory_alignment:
        raise ValueError(
            'training_stage=alignment requires trajectory_alignment=True')
    if args.training_stage == 'alignment' and not any((
            args.lambda_trajectory_alignment,
            args.lambda_trajectory_pair,
            args.lambda_alignment_no_harm,
            args.lambda_alignment_entropy)):
        raise ValueError(
            'training_stage=alignment requires at least one alignment loss')
    if (not 0 <= args.trajectory_alignment_keep_threshold <= 1 or
            not math.isfinite(args.trajectory_alignment_keep_threshold)):
        raise ValueError(
            'trajectory_alignment_keep_threshold must lie in [0,1]')
    if (not 0 <= args.keep_threshold <= 1 or
            not math.isfinite(args.keep_threshold)):
        raise ValueError('keep_threshold must lie in [0,1]')
    if (not 0 < args.internal_validation_fraction < 1 or
            not math.isfinite(args.internal_validation_fraction)):
        raise ValueError('internal_validation_fraction must lie in (0,1)')
    alignment_weights_enabled = any((
        args.lambda_trajectory_alignment,
        args.lambda_trajectory_pair,
        args.lambda_alignment_no_harm,
        args.lambda_alignment_entropy,
    ))
    if alignment_weights_enabled and not args.trajectory_alignment:
        raise ValueError(
            'Trajectory-alignment losses require trajectory_alignment=True')
    if (args.lambda_continuous_refinement > 0 or
            args.lambda_refinement_delta > 0) and \
            not args.continuous_refinement:
        raise ValueError(
            'Continuous-refinement losses require '
            'continuous_refinement=True')
    if args.lambda_graph_density > 0 and not args.adaptive_graph:
        raise ValueError(
            'lambda_graph_density > 0 requires adaptive_graph=True')
    if (args.adaptive_graph and
            args.adaptive_graph_target_density <
            args.adaptive_graph_gate_floor):
        raise ValueError(
            'adaptive_graph_target_density cannot be below gate_floor')
    if (args.lambda_joint_rank > 0 and
            args.goal_model_type not in {'lowrank', 'joint'}):
        raise ValueError(
            'lambda_joint_rank > 0 requires lowrank or joint goal model')
    if (args.collision_threshold_meter is not None and
            (not math.isfinite(args.collision_threshold_meter) or
             args.collision_threshold_meter < 0)):
        raise ValueError('collision_threshold_meter must be non-negative')
    if (args.goal_recall_threshold_meter is not None and
            (not math.isfinite(args.goal_recall_threshold_meter) or
             args.goal_recall_threshold_meter < 0)):
        raise ValueError('goal_recall_threshold_meter must be non-negative')
    if args.collision_interpolation_steps < 1:
        raise ValueError('collision_interpolation_steps must be >= 1')

    if args.joint_goal_enabled is None:
        args.joint_goal_enabled = args.goal_model_type in {
            'lowrank', 'energy', 'joint'} or (
                args.goal_model_type == 'joint_dependency_v2' and
                args.jdv2_active)
    if args.goal_model_type == 'independent' and args.joint_goal_enabled:
        raise ValueError(
            'joint_goal_enabled=True requires a non-independent goal model')
    if (args.goal_model_type == 'independent' and
            args.training_stage == 'joint'):
        raise ValueError(
            'training_stage=joint requires a social goal_model_type')
    if (args.goal_model_type == 'independent' and
            args.training_stage == 'alignment'):
        raise ValueError(
            'training_stage=alignment requires a social goal_model_type')
    if (args.goal_model_type != 'independent' and
            not (args.goal_model_type == 'joint_dependency_v2' and
                 not args.jdv2_active) and
            args.training_stage == 'baseline'):
        raise ValueError(
            'training_stage=baseline is reserved for goal_model_type='
            'independent; use joint or finetune for social goal models')
    if (args.goal_model_type == 'independent' and
            args.best_metric in {'JADE', 'JFDE'}):
        raise ValueError(
            'JADE/JFDE require synchronized social scene-window batches')
    if args.num_joint_samples is None:
        args.num_joint_samples = args.num_samples
    if args.num_joint_samples != args.num_samples:
        raise ValueError(
            'num_joint_samples must equal num_samples so each tree branch has '
            'one coherent scene-level goal configuration')

    if args.model_selection_split == 'auto':
        args.model_selection_split = (
            'internal_train'
            if args.training_stage == 'multiway_coupling'
            else 'dataset_valid')
    if (args.training_stage == 'multiway_coupling' and
            args.model_selection_split != 'internal_train'):
        raise ValueError(
            'Formal V4 training selects checkpoints only on internal_train; '
            'the held-out UNIV test scene is reserved for final evaluation.')

    if args.clean_split_protocol:
        if args.model_selection_split != 'internal_train':
            raise ValueError(
                'Clean protocol requires model_selection_split=internal_train')
        if args.internal_validation_strategy != 'source_block':
            raise ValueError(
                'Clean protocol requires internal_validation_strategy='
                'source_block')
        if args.final_test_split != 'heldout_test':
            raise ValueError(
                'Clean protocol requires final_test_split=heldout_test')
        if not args.clean_split_manifest_path or not re.fullmatch(
                r'[0-9a-f]{64}', str(args.clean_split_manifest_hash or '')):
            raise ValueError(
                'Clean protocol requires a manifest path and SHA256 hash')
        if args.phase not in {'train', 'test'}:
            raise ValueError(
                'Clean protocol supports only separate train or locked '
                'test phases')
        if args.phase == 'train' and args.final_test_access != 'blocked':
            raise ValueError(
                'Clean training requires final_test_access=blocked')
        if args.phase == 'test':
            if args.final_test_access != 'authorized':
                raise ValueError(
                    'Clean final evaluation requires authorized test access')
            if not args.final_test_lock_path:
                raise ValueError(
                    'Clean final evaluation requires a lock artifact')
            if args.load_checkpoint != 'best':
                raise ValueError(
                    'Clean final evaluation requires load_checkpoint=best')
            if args.clean_evaluation_target not in {'stage_a', 'stage_b'}:
                raise ValueError(
                    'Clean final evaluation requires a locked target')
    elif args.final_test_access != 'legacy':
        raise ValueError(
            'Non-clean runs must retain legacy final-test access semantics')

    # set current device
    if args.device.startswith('cuda') and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.device = 'cpu'
        args.use_cuda = False
    # dataset and test_set checks
    if args.dataset == 'eth5':
        assert args.test_set in ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    else:
        # hard assignation of test set
        args.test_set = args.dataset
    if args.cross_dataset_protocol_target is not None:
        target = args.cross_dataset_protocol_target
        if args.dataset != 'eth5' or args.test_set != target:
            raise ValueError(
                'cross_dataset_protocol_target must equal the ETH5 test_set')
        if args.goal_model_type != 'joint_dependency_v2':
            raise ValueError(
                'cross-dataset protocol is defined only for JDV2')

        def _contains_target(path):
            if path is None:
                return True
            tokens = re.split(r'[^a-z0-9]+', str(path).lower())
            return target in tokens

        target_paths = {
            'run_name': args.run_name,
            'jdv2_cache_root': args.jdv2_cache_root,
            'jdv2_source_checkpoint': args.jdv2_source_checkpoint,
        }
        if args.training_stage == 'joint_trajectory':
            target_paths.update({
                'pretrain_path': args.pretrain_path,
                'stage_a_freeze_manifest_path':
                    args.stage_a_freeze_manifest_path,
            })
        mismatched = [
            name for name, path in target_paths.items()
            if path is not None and not _contains_target(path)]
        if mismatched:
            raise ValueError(
                'Cross-dataset paths must be target-isolated: ' +
                ', '.join(mismatched))
        if args.training_stage == 'joint_trajectory':
            unresolved = [
                name for name in (
                    'stage_a_parent_checkpoint_sha256',
                    'stage_a_freeze_manifest_sha256',
                    'stage_a_freeze_source_commit')
                if getattr(args, name) is None]
            if unresolved:
                raise ValueError(
                    'Cross-dataset Stage-B remains unresolved until the '
                    'target Stage-A freeze exists: ' + ', '.join(unresolved))
    if not args.save_every:
        args.save_every = 10 # int(args.num_epochs // 5)
    # set parameters for trajectories
    args = compute_term(args)
    # change a few things in fast debug mode
    if args.fast_debug:
        args.num_test_runs = 1
        args.start_validation = 0
    # set directories
    args.base_dir = '.'  # base directory
    args.save_base_dir = (
        os.path.join('outputs', 'joint_dependency_v2')
        if args.goal_model_type == 'joint_dependency_v2' and args.jdv2_active
        else 'output')
    args.save_dir = os.path.join(
        args.base_dir, args.save_base_dir, str(args.test_set))
    # Preserve the exact legacy output path for GDTS-Base while isolating all
    # ablation checkpoints/configs from one another.
    if args.goal_model_type != 'independent':
        args.save_dir = os.path.join(args.save_dir, args.goal_model_type)
    if args.run_name is not None:
        args.save_dir = os.path.join(args.save_dir, 'runs', args.run_name)
    args.model_dir = args.save_dir 
    # One model configuration is shared by preprocessing, training and test.
    # Phase-specific files silently reset K/graph/refinement at evaluation.
    args.config = os.path.join(args.save_dir, 'config.yaml')
    args.branch_stage_step = int((args.ddpm_step - args.trunk_stage_step) // (args.ddpm_step/args.ddim_step))
    return args

def compute_term(args):
    """
    Set parameters for trajectories
    """
    args.obs_length = 8
    args.pred_length = 12
    args.seq_length = args.obs_length + args.pred_length
    return args


def print_args(args):
    """
    Print parsed args to screen
    """
    print("-"*62)
    print("|" + " "*25 + "PARAMETERS" + " "*25 + "|")
    print("-" * 62)
    for k, v in vars(args).items():
        print(f"| {k:25s}: {v}")
    print("-"*62 + "\n")


def main_parser():
    """
    Pipeline from_parsed args to args
    """
    # Parse input parameters
    parser = get_parser()
    parsed_args = parser.parse_args()
    parsed_args = check_and_add_additional_args(parsed_args)

    # Prefer the common cross-phase config. On first use of an older checkout,
    # import its phase-specific YAML and immediately migrate it.
    common_config = parsed_args.config
    legacy_config = os.path.join(
        parsed_args.save_dir, f'config_{parsed_args.phase}.yaml')
    if os.path.exists(common_config):
        args = load_args(parser, parsed_args)
        args = check_and_add_additional_args(args)
    elif os.path.exists(legacy_config):
        parsed_args.config = legacy_config
        args = load_args(parser, parsed_args)
        args = check_and_add_additional_args(args)
        args.config = common_config
        save_args(args)
    else:
        args = parsed_args
        save_args(args)
    # print args given to the model
    print_args(args)
    return args


if __name__ == '__main__':
    # Parse input parameters
    parser = get_parser()
    pars_args = parser.parse_args()
    pars_args = check_and_add_additional_args(pars_args)
    print_args(pars_args)
