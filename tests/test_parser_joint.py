import os
import sys

import yaml

import pytest

from src.parser import (
    check_and_add_additional_args,
    get_parser,
    load_args,
    save_args,
)
from src.data_grouping import batch_cache_manifest, batch_cache_path


def _parse(*arguments):
    return get_parser().parse_args(list(arguments))


def test_paper_alias_derives_joint_switch_samples_and_isolated_output_path():
    args = _parse(
        '--device', 'cpu', '--goal_model_type', 'Full-Joint',
        '--data_augmentation', 'False', '--num_samples', '7')
    args = check_and_add_additional_args(args)

    assert args.goal_model_type == 'joint'
    assert args.joint_goal_enabled is True
    assert args.num_joint_samples == 7
    assert os.path.normpath(args.save_dir) == 'output/sdd/joint'


def test_independent_keeps_legacy_output_and_disallows_joint_only_stage():
    args = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'independent'))
    assert os.path.normpath(args.save_dir) == 'output/sdd'
    assert args.joint_goal_enabled is False

    with pytest.raises(ValueError, match='training_stage=joint'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'independent',
            '--training_stage', 'joint'))


def test_social_model_disables_non_calibrated_pixel_augmentation():
    with pytest.warns(UserWarning, match='homography-aware'):
        args = check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'social',
            '--data_augmentation', 'True'))
    assert args.data_augmentation is False


def test_structured_model_disallows_legacy_baseline_stage():
    with pytest.raises(ValueError, match='reserved'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'joint',
            '--training_stage', 'baseline', '--data_augmentation', 'False'))


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        ('--energy_weight', '-1'),
        ('--lambda_PL', 'nan'),
        ('--goal_soft_sigma', '0'),
        ('--collision_interpolation_steps', '0'),
    ],
)
def test_invalid_structured_hyperparameters_are_rejected(option, value):
    with pytest.raises(ValueError):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--data_augmentation', 'False',
            option, value))


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        ('--skip_ts_window', '0'),
        ('--down_factor', '0'),
        ('--validate_every', '0'),
        ('--num_workers', '-1'),
        ('--ddim_step', '101'),
    ],
)
def test_invalid_runtime_integer_ranges_fail_during_parsing(option, value):
    with pytest.raises(ValueError):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--data_augmentation', 'False', option, value))


def test_saved_derived_sample_count_does_not_block_num_samples_override(
        tmp_path, monkeypatch):
    parser = get_parser()
    original = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'joint',
        '--data_augmentation', 'False', '--num_samples', '20'))
    original.config = str(tmp_path / 'config.yaml')
    save_args(original)
    saved = yaml.safe_load((tmp_path / 'config.yaml').read_text())
    assert saved['num_joint_samples'] is None

    parsed = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'joint',
        '--data_augmentation', 'False', '--num_samples', '7'))
    parsed.config = original.config
    monkeypatch.setattr(sys, 'argv', [
        'program', '--device', 'cpu', '--goal_model_type', 'joint',
        '--data_augmentation', 'False', '--num_samples', '7'])
    loaded = check_and_add_additional_args(load_args(parser, parsed))
    assert loaded.num_samples == 7
    assert loaded.num_joint_samples == 7


def test_new_joint_optimization_flags_validate_dependencies():
    args = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'joint',
        '--data_augmentation', 'False',
        '--adaptive_graph', 'True', '--lambda_graph_density', '0.01',
        '--continuous_refinement', 'True',
        '--lambda_continuous_refinement', '0.2',
        '--lambda_joint_rank', '0.1',
        '--baseline_lr_scale', '0.2',
        '--baseline_unfreeze_epoch', '2'))
    assert args.adaptive_graph is True
    assert args.continuous_refinement is True
    assert args.baseline_lr_scale == pytest.approx(0.2)

    with pytest.raises(ValueError, match='adaptive_graph requires'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'independent',
            '--adaptive_graph', 'True'))
    with pytest.raises(ValueError, match='Continuous-refinement losses'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'joint',
            '--data_augmentation', 'False',
            '--lambda_refinement_delta', '0.1'))


def test_trajectory_alignment_stage_has_closed_configuration_contract():
    args = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'joint',
        '--data_augmentation', 'False',
        '--training_stage', 'alignment',
        '--trajectory_alignment', 'True',
        '--lambda_trajectory_alignment', '1',
        '--lambda_trajectory_pair', '0.5'))
    assert args.training_stage == 'alignment'
    assert args.trajectory_alignment is True

    with pytest.raises(ValueError, match='requires trajectory_alignment'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'joint',
            '--data_augmentation', 'False',
            '--training_stage', 'alignment',
            '--lambda_trajectory_alignment', '1'))


def test_multiway_v4_contract_and_leak_free_validation_defaults():
    args = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'joint',
        '--data_augmentation', 'False',
        '--training_stage', 'multiway_coupling',
        '--trajectory_coupling', 'multiway_v4'))
    assert args.upstream_generator == 'stage0_independent'
    assert args.freeze_upstream_generator is True
    assert args.model_selection_split == 'internal_train'
    assert args.final_test_split == 'heldout_test'
    assert args.social_mode_scope == 'off'
    assert args.hard_projection == 'hungarian'

    with pytest.raises(ValueError, match='requires trajectory_coupling'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'joint',
            '--data_augmentation', 'False',
            '--training_stage', 'multiway_coupling'))
    with pytest.raises(ValueError, match='must not move trajectory'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'joint',
            '--data_augmentation', 'False',
            '--training_stage', 'multiway_coupling',
            '--trajectory_coupling', 'multiway_v4',
            '--continuous_refinement', 'True'))


def test_jdv2_active_and_all_off_configuration_contracts():
    active = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'jdv2',
        '--training_stage', 'joint_goal', '--data_augmentation', 'False'))
    assert active.goal_model_type == 'joint_dependency_v2'
    assert active.jdv2_active is True
    assert active.num_samples == 20
    assert active.num_goal_candidates == 21
    assert os.path.normpath(active.save_dir).startswith(
        os.path.normpath('outputs/joint_dependency_v2'))

    all_off = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'joint_dependency_v2',
        '--training_stage', 'baseline',
        '--use_scene_latent', 'False',
        '--use_dynamic_relation', 'False',
        '--use_joint_energy', 'False',
        '--use_dependency_corrector', 'False'))
    assert all_off.jdv2_active is False
    assert all_off.data_augmentation is True
    legacy = check_and_add_additional_args(_parse(
        '--device', 'cpu', '--goal_model_type', 'independent'))
    assert batch_cache_path(all_off) == batch_cache_path(legacy)
    assert batch_cache_manifest(all_off) == batch_cache_manifest(legacy)

    with pytest.raises(ValueError, match='invalid when all four'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'jdv2',
            '--training_stage', 'joint_goal',
            '--use_scene_latent', 'False',
            '--use_dynamic_relation', 'False',
            '--use_joint_energy', 'False',
            '--use_dependency_corrector', 'False'))


def test_jdv2_rejects_noncanonical_dimensions_and_wrong_stage():
    with pytest.raises(ValueError, match='full lowercase git SHA'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'jdv2',
            '--training_stage', 'joint_goal', '--data_augmentation', 'False',
            '--jdv2_cache_source_commit', 'not-a-commit'))
    with pytest.raises(ValueError, match='frozen JDV2 dimensions'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'jdv2',
            '--training_stage', 'joint_goal', '--num_samples', '19',
            '--data_augmentation', 'False'))
    with pytest.raises(ValueError, match='requires training_stage'):
        check_and_add_additional_args(_parse(
            '--device', 'cpu', '--goal_model_type', 'jdv2',
            '--training_stage', 'finetune', '--data_augmentation', 'False'))
