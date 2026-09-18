#!/usr/bin/env python3
"""Run deterministic/repeated Stage-A validation and read-only audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from src.jdv2_audit import (
    AUDIT_METRICS,
    audit_scene_and_relation,
    repeated_validation,
)
from src.parser import check_and_add_additional_args, get_parser
from src.models.model import GDTS
from src.trainer import trainer


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_experiment_args(config_path, output_dir, device):
    with open(config_path) as handle:
        saved = yaml.safe_load(handle)
    parser = get_parser()
    bootstrap = check_and_add_additional_args(parser.parse_args([]))
    known = set(vars(bootstrap))
    unknown = set(saved) - known
    if unknown:
        raise KeyError('Unknown saved config keys: ' + ', '.join(sorted(unknown)))
    parser.set_defaults(**saved)
    args = parser.parse_args([])
    args.phase = 'test'
    args.load_checkpoint = None
    args.pretrain_path = None
    args.use_wandb = False
    args.shuffle_test_batches = False
    args.num_workers = 0
    args.device = device
    args = check_and_add_additional_args(args)
    args.model_dir = str(output_dir)
    args.save_dir = str(output_dir)
    os.makedirs(args.model_dir, exist_ok=True)
    return args


def _active_evaluator(config, checkpoint, output_dir, device):
    args = _load_experiment_args(config, output_dir, device)
    evaluator = trainer(args)
    epoch = evaluator._load_state_file(str(checkpoint))
    evaluator.net.eval()
    return evaluator, epoch


def _baseline_evaluator(config, checkpoint, output_dir, device):
    # Build the loader in active mode so it consumes the exact same
    # synchronized scene-window split as JDV2. Then replace only the network
    # with the exact all-off/legacy namespace. The already-created dataset
    # retains its validated JDV2 manifest and graph records.
    args = _load_experiment_args(config, output_dir, device)
    evaluator = trainer(args)
    args.training_stage = 'baseline'
    args.use_scene_latent = False
    args.use_dynamic_relation = False
    args.use_joint_energy = False
    args.use_dependency_corrector = False
    args.jdv2_active = False
    args.joint_goal_enabled = False
    args.best_metric = 'auto'
    args.pretrain_path = str(checkpoint)
    args.model_dir = str(output_dir)
    args.save_dir = str(output_dir)
    evaluator.args = args
    evaluator.net = GDTS(args, evaluator.device).to(evaluator.device)
    evaluator._pending_training_state = None
    evaluator._load_state_file(str(checkpoint), baseline_initialization=False)
    evaluator.net.eval()
    return evaluator


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--epoch11-checkpoint', required=True)
    parser.add_argument('--epoch20-checkpoint', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[2035, 2036, 2037, 2038, 2039])
    return parser.parse_args()


def main():
    cli = _parse_cli()
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    edge_cache_root = str(Path(yaml.safe_load(
        Path(cli.config).read_text())['jdv2_cache_root']).resolve())

    target = output / 'stage_a_audit_results.json'
    result = {
        'protocol': {
            'split': 'valid',
            'seeds': cli.seeds,
            'sample_count': 20,
            'metric_implementation': 'GDTS.compute_model_metrics',
            'coordinate_conversion': 'scene.make_world_coord_torch',
            'sampling_policy': 'stochastic deployed policy (not MAP)',
            'rng_isolation': 'Python/NumPy/CPU Torch/all CUDA snapshot+restore',
        },
        'checkpoints': {
            'epoch11': {
                'path': str(Path(cli.epoch11_checkpoint).resolve()),
                'sha256': _sha256(cli.epoch11_checkpoint),
            },
            'epoch20': {
                'path': str(Path(cli.epoch20_checkpoint).resolve()),
                'sha256': _sha256(cli.epoch20_checkpoint),
            },
            'gdts_baseline': {
                'path': str(Path(cli.baseline_checkpoint).resolve()),
                'sha256': _sha256(cli.baseline_checkpoint),
            },
        },
    }

    def persist():
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')

    epoch11_eval, epoch11 = _active_evaluator(
        cli.config, cli.epoch11_checkpoint, output / 'epoch11', cli.device)
    result['checkpoints']['epoch11']['epoch'] = epoch11
    result['epoch11'] = repeated_validation(
        epoch11_eval, epoch11, cli.seeds)
    persist()
    audits = audit_scene_and_relation(epoch11_eval, seed=cli.seeds[0])
    result['scene_audit_epoch11'] = audits['scene']
    result['relation_audit_epoch11'] = audits['relation']
    persist()

    original_dynamic = epoch11_eval.args.use_dynamic_relation
    epoch11_eval.args.use_dynamic_relation = False
    try:
        neutral_raw = epoch11_eval._evaluate_epoch(
            epoch11, mode='valid', evaluation_seed=cli.seeds[0],
            metric_names=AUDIT_METRICS)
    finally:
        epoch11_eval.args.use_dynamic_relation = original_dynamic
    result['relation_audit_epoch11'].update({
        'dynamic_residual_neutralized_seed': cli.seeds[0],
        'dynamic_residual_neutralized_metrics': {
            name: float(neutral_raw['valid_' + name])
            for name in AUDIT_METRICS},
        'normal_same_seed_metrics': {
            name: result['epoch11']['runs'][0][name]
            for name in AUDIT_METRICS},
    })
    persist()

    epoch20_eval, epoch20 = _active_evaluator(
        cli.config, cli.epoch20_checkpoint, output / 'epoch20', cli.device)
    result['checkpoints']['epoch20']['epoch'] = epoch20
    result['epoch20'] = repeated_validation(
        epoch20_eval, epoch20, cli.seeds)
    persist()

    baseline_eval = _baseline_evaluator(
        cli.config, cli.baseline_checkpoint, output / 'gdts_baseline',
        cli.device)
    result['gdts_baseline'] = repeated_validation(
        baseline_eval, 0, cli.seeds,
        external_edge_cache_root=edge_cache_root)
    persist()
    print(f'Wrote {target}')


if __name__ == '__main__':
    main()
