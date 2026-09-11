#!/usr/bin/env python3
"""One-real-batch smoke test for the formal UNIV Multiway Coupling V4 path.

The smoke test intentionally reuses the immutable synchronized cache and the
real Stage-0 checkpoint.  It does not run preprocessing or an epoch of
training.  It verifies that the V4 loss back-propagates only into the coupling
modules and that hard inference preserves every agent's trajectory bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.metrics import compute_metric_mask
from src.parser import check_and_add_additional_args, get_parser
from src.trainer import trainer


def _formal_args(device: str, run_name: str):
    cli = [
        '--phase', 'train',
        '--dataset', 'eth5',
        '--test_set', 'univ',
        '--goal_model_type', 'joint',
        '--run_name', run_name,
        '--training_stage', 'multiway_coupling',
        '--trajectory_coupling', 'multiway_v4',
        '--upstream_generator', 'stage0_independent',
        '--pretrain_path', 'output/univ/saved_models/epoch_100.pt',
        '--freeze_upstream_generator', 'True',
        '--continuous_refinement', 'False',
        '--social_mode_scope', 'off',
        '--num_samples', '20',
        '--num_joint_samples', '20',
        '--trajectory_dim', '128',
        '--num_relation_modes', '4',
        '--trajectory_pair_rank', '8',
        '--trajectory_energy_type', 'lowrank',
        '--trajectory_conditioned_relation', 'True',
        '--pair_specific_gate', 'True',
        '--graph_type', 'radius_ttc',
        '--graph_radius', '6.0',
        '--ttc_threshold', '8.0',
        '--synchronizer', 'multiway',
        '--sync_iterations', '4',
        '--sinkhorn_iterations', '8',
        '--sync_temperature', '0.2',
        '--sync_inference_top_k_edges', '0',
        '--hard_projection', 'hungarian',
        '--use_identity_bypass', 'True',
        '--keep_threshold', '0.6',
        '--lambda_alignment', '1.0',
        '--lambda_pair_score', '0.5',
        '--lambda_pair_assignment', '0.5',
        '--lambda_no_harm', '2.0',
        '--lambda_perm_entropy', '0.02',
        '--lambda_relation_prior', '0.02',
        '--lambda_gate_reg', '0.0',
        '--model_selection_split', 'internal_train',
        '--internal_validation_strategy', 'source_block',
        '--internal_validation_fraction', '0.1',
        '--internal_validation_seed', '2025',
        '--final_test_split', 'heldout_test',
        '--data_augmentation', 'False',
        '--num_workers', '0',
        '--device', device,
        '--use_wandb', 'False',
        '--reproducibility', 'True',
        '--seed', '2025',
    ]
    return check_and_add_additional_args(get_parser().parse_args(cli))


def _gradient_norms(model):
    groups = {
        'trajectory_encoder': model.multiway_coupler.trajectory_encoder,
        'trajectory_relation': model.multiway_coupler.trajectory_relation,
        'trajectory_energy': model.multiway_coupler.trajectory_energy,
        'permutation_synchronizer': (
            model.multiway_coupler.permutation_synchronizer),
    }
    norms = {}
    for name, module in groups.items():
        norms[name] = float(sum(
            parameter.grad.detach().norm().cpu()
            for parameter in module.parameters()
            if parameter.grad is not None))
    return norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--run_name', default='multiway_coupling_v4_smoke')
    args = parser.parse_args()

    formal = _formal_args(args.device, args.run_name)
    processor = trainer(formal)
    model = processor.net
    model.train()

    train_batch, train_id = next(iter(processor.data_loaders['train']))
    train_inputs, train_seq = model.prepare_inputs(train_batch, train_id)
    losses = model.get_loss(train_inputs, train_seq)
    coefficients = model.set_losses_coeffs()
    total = sum(coefficients[name] * value for name, value in losses.items())
    if not torch.isfinite(total):
        raise FloatingPointError(f'non-finite V4 smoke loss: {losses}')
    total.backward()

    gradient_norms = _gradient_norms(model)
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0
           for value in gradient_norms.values()):
        raise AssertionError(
            f'Every V4 module must receive a finite non-zero gradient: '
            f'{gradient_norms}')
    frozen_gradient_count = sum(
        parameter.grad is not None
        for module in model._baseline_modules()
        for parameter in module.parameters())
    if frozen_gradient_count:
        raise AssertionError(
            'The frozen Stage-0 generator unexpectedly received gradients')

    model.eval()
    valid_batch, valid_id = next(iter(processor.data_loaders['valid']))
    valid_inputs, valid_seq = model.prepare_inputs(valid_batch, valid_id)
    with torch.no_grad():
        predictions, auxiliary = model(valid_inputs, if_test=True)
        metric_mask = compute_metric_mask(valid_seq)
        diagnostics = model.multiway_batch_diagnostics(
            auxiliary, valid_inputs, metric_mask)

    permutation = auxiliary['multiway_coupling_permutation']
    expected = torch.arange(
        permutation.shape[1], device=permutation.device).expand_as(permutation)
    if not torch.equal(permutation.sort(dim=1).values, expected):
        raise AssertionError('Hungarian output is not a per-agent bijection')
    if predictions.shape[0] != formal.num_samples:
        raise AssertionError('Hard inference changed trajectory-bank size')

    result = {
        'status': 'passed',
        'device': str(processor.device),
        'train_agents': int(train_inputs['obs_traj_world'].shape[0]),
        'valid_agents': int(valid_inputs['obs_traj_world'].shape[0]),
        'num_samples': int(predictions.shape[0]),
        'loss_total': float(total.detach().cpu()),
        'losses': {
            name: float(value.detach().cpu())
            for name, value in losses.items()},
        'v4_gradient_norms': gradient_norms,
        'frozen_upstream_gradient_count': frozen_gradient_count,
        'hard_permutation_is_bijective': True,
        'marginal_ADE_max_abs_error': float(
            diagnostics['scalar']['marginal_ADE_max_abs_error']),
        'marginal_FDE_max_abs_error': float(
            diagnostics['scalar']['marginal_FDE_max_abs_error']),
        'changed_fraction': float(
            diagnostics['scalar']['changed_fraction']),
    }
    output = Path(formal.model_dir) / 'smoke_test_result.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f'Wrote smoke-test result: {output}')


if __name__ == '__main__':
    main()
