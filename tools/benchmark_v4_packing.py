#!/usr/bin/env python3
"""Verify online/cache equivalence and benchmark V4 data paths A/B/C."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from src.data_loader import get_dataloader
from src.models.model import GDTS
from src.multiway_coupling_loss import (
    compute_scene_balanced_multiway_coupling_loss,
)
from src.parser import check_and_add_additional_args, get_parser
from src.trajectory_bank_builder import _set_all_rng
from src.trajectory_bank_cache import (
    deterministic_trajectory_seed,
    get_trajectory_bank_dataloader,
)


class GpuMonitor:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        while not self.stop_event.wait(0.1):
            try:
                output = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'], text=True,
                    stderr=subprocess.DEVNULL)
                util, memory = output.strip().splitlines()[0].split(',')
                self.samples.append((float(util), float(memory)))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                return

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def summary(self):
        if not self.samples:
            return {'gpu_utilization_percent': None,
                    'nvidia_smi_memory_used_mib': None}
        values = np.asarray(self.samples)
        return {
            'gpu_utilization_percent': float(values[:, 0].mean()),
            'nvidia_smi_memory_used_mib': float(values[:, 1].mean()),
        }


def make_args(cli):
    parser = get_parser()
    arguments = parser.parse_args([
        '--dataset', 'eth5', '--test_set', cli.test_set,
        '--goal_model_type', 'joint', '--run_name', 'packing_benchmark_v2',
        '--phase', 'train', '--training_stage', 'multiway_coupling',
        '--trajectory_coupling', 'multiway_v4',
        '--upstream_generator', 'stage0_independent',
        '--pretrain_path', cli.checkpoint,
        '--freeze_upstream_generator', 'True',
        '--continuous_refinement', 'False', '--social_mode_scope', 'off',
        '--num_samples', '20', '--num_joint_samples', '20',
        '--model_selection_split', 'internal_train',
        '--internal_validation_strategy', 'source_block',
        '--data_augmentation', 'False', '--num_workers', '0',
        '--shuffle_train_batches', 'False', '--use_wandb', 'False',
        '--num_cached_seeds_per_window', '4',
        '--trajectory_bank_seed_base', '42',
        '--trajectory_bank_cache_root', cli.cache_root,
        '--max_agents_per_pack', str(cli.max_agents),
        '--max_edges_per_pack', str(cli.max_edges),
        '--max_scenes_per_pack', str(cli.max_scenes),
    ])
    return check_and_add_additional_args(arguments)


def load_model(args):
    device = torch.device(args.device)
    model = GDTS(args, device).to(device)
    checkpoint = torch.load(
        args.pretrain_path, map_location=device, weights_only=False)
    model.load_state_dict(
        checkpoint.get('model_state_dict', checkpoint), strict=False)
    model.configure_training_epoch(1)
    return model, device


def verify_equivalence(args, model, device):
    original_shuffle = args.shuffle_train_batches
    args.shuffle_train_batches = False
    raw_loader = get_dataloader(args, set_name='train')
    args.shuffle_train_batches = original_shuffle
    cached_loader = get_trajectory_bank_dataloader(args, 'train')
    cached_loader.dataset.set_seed_index(0)
    batch_data, batch_id = next(iter(raw_loader))
    inputs, _ = model.prepare_inputs(batch_data, batch_id)
    scene_name = batch_id['scene_name'][0]
    frame_start = int(inputs['frame_ids'][0, 0])
    seed = deterministic_trajectory_seed(
        args.trajectory_bank_seed_base, scene_name, frame_start, 0)
    model.eval()
    _set_all_rng(seed, args.use_cuda)
    with torch.inference_mode():
        prediction, _ = model._v4_upstream_predictions(inputs)
        online = model._future_predictions_world(prediction, inputs)
    cached_record = cached_loader.dataset[0]
    cached = cached_record['raw_future_world'].to(device)
    max_error = float((online - cached).abs().max().cpu())
    equivalent = torch.allclose(online, cached, atol=1e-6, rtol=1e-6)
    if not equivalent:
        raise AssertionError(
            f'Online/cache equivalence failed: max_abs_error={max_error}')
    return {
        'passed': True, 'atol': 1e-6, 'rtol': 1e-6,
        'max_abs_error': max_error, 'scene_name': scene_name,
        'frame_start': frame_start, 'seed_index': 0, 'seed': seed,
        'K': int(online.shape[1]),
    }


def coupling_loss(args, inputs, output):
    details = compute_scene_balanced_multiway_coupling_loss(
        permutations=output['soft_permutation'],
        pair_scores=output['pair_score'],
        raw_future=output['raw_trajectories'],
        ground_truth=inputs['gt_future_world'],
        edge_index=inputs['edge_index'],
        future_mask=inputs['future_mask'], agent_mask=inputs['agent_mask'],
        scene_index=inputs['scene_index'],
        relation_posterior=output['relation_posterior'],
        relation_prior=output['relation_prior'],
        pair_gate=output['pair_gate'], edge_weight=inputs['edge_weight'],
        lambda_fde=args.coupling_fde_weight,
        lambda_rel_geom=args.coupling_rel_geom_weight,
        lambda_rel_end=args.coupling_rel_endpoint_weight,
        tau_joint=args.coupling_joint_temperature,
        tau_pair_target=args.coupling_pair_target_temperature,
        tau_pair_pred=args.coupling_pair_pred_temperature,
        no_harm_margin=args.coupling_no_harm_margin,
        weight_alignment=args.lambda_alignment,
        weight_pair_score=args.lambda_pair_score,
        weight_pair_assignment=args.lambda_pair_assignment,
        weight_no_harm=args.lambda_no_harm,
        weight_perm_entropy=args.lambda_perm_entropy,
        weight_relation_prior=args.lambda_relation_prior,
        weight_gate_reg=args.lambda_gate_reg,
        gate_regularization=('l1' if args.lambda_gate_reg > 0 else 'none'))
    return details['loss_total']


def benchmark_cached(args, model, device, packed, max_windows):
    args.use_trajectory_bank_cache = True
    args.use_multi_scene_packing = packed
    args.scene_balanced_loss = True
    loader = get_trajectory_bank_dataloader(args, 'train')
    loader.dataset.set_epoch(1)
    loader.dataset.set_seed_index(None)
    iterator = iter(loader)
    counters = dict(windows=0, packs=0, agents=0, edges=0)
    timings = {name: 0.0 for name in (
        'data_loading_ms', 'graph_build_ms', 'trajectory_encoder_ms',
        'pair_module_ms', 'synchronizer_ms', 'backward_ms')}
    model.train()
    for module in model._baseline_modules():
        module.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_total = time.perf_counter()
    with GpuMonitor() as monitor:
        while counters['windows'] < max_windows:
            load_started = time.perf_counter()
            try:
                pack = next(iterator)
            except StopIteration:
                break
            timings['data_loading_ms'] += 1000 * (
                time.perf_counter() - load_started)
            graph_started = time.perf_counter()
            inputs = model.prepare_cached_trajectory_pack(pack)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timings['graph_build_ms'] += 1000 * (
                time.perf_counter() - graph_started)
            output = model.multiway_coupler(
                inputs['raw_future_world'], inputs['obs_world'][:, -1],
                inputs['edge_index'], inputs['edge_feat'],
                inputs['edge_weight'], scene_index=inputs['scene_index'],
                hard=False, profile=True)
            for name, value in output['profile_ms'].items():
                timings[name] += value
            loss = coupling_loss(args, inputs, output)
            backward_started = time.perf_counter()
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timings['backward_ms'] += 1000 * (
                time.perf_counter() - backward_started)
            model.zero_grad(set_to_none=True)
            counters['windows'] += int(inputs['scene_ptr'].numel() - 1)
            counters['packs'] += 1
            counters['agents'] += int(inputs['raw_future_world'].shape[0])
            counters['edges'] += int(inputs['edge_index'].shape[1])
    elapsed = time.perf_counter() - started_total
    denominator = max(counters['packs'], 1)
    result = {
        **counters,
        'seconds': elapsed,
        'windows_per_second': counters['windows'] / max(elapsed, 1e-9),
        'scenes_per_second': counters['windows'] / max(elapsed, 1e-9),
        'agents_per_second': counters['agents'] / max(elapsed, 1e-9),
        'avg_scenes_per_pack': counters['windows'] / denominator,
        'avg_agents_per_pack': counters['agents'] / denominator,
        'avg_edges_per_pack': counters['edges'] / denominator,
        **{name: value / denominator for name, value in timings.items()},
        'peak_allocated_mib': (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == 'cuda' else 0.0),
        'peak_reserved_mib': (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == 'cuda' else 0.0),
        **monitor.summary(),
    }
    return result


def benchmark_online(args, model, device, max_windows):
    args.use_trajectory_bank_cache = False
    original = args.shuffle_train_batches
    args.shuffle_train_batches = False
    loader = get_dataloader(args, set_name='train')
    args.shuffle_train_batches = original
    iterator = iter(loader)
    counters = dict(windows=0, packs=0, agents=0, edges=0)
    timings = {name: 0.0 for name in (
        'data_loading_ms', 'upstream_generation_ms', 'graph_build_ms',
        'trajectory_encoder_ms', 'pair_module_ms', 'synchronizer_ms',
        'backward_ms')}
    model.train()
    for module in model._baseline_modules():
        module.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_total = time.perf_counter()
    with GpuMonitor() as monitor:
        while counters['windows'] < max_windows:
            load_started = time.perf_counter()
            try:
                batch_data, batch_id = next(iterator)
            except StopIteration:
                break
            timings['data_loading_ms'] += 1000 * (
                time.perf_counter() - load_started)
            inputs, seq_list = model.prepare_inputs(batch_data, batch_id)
            scene_name = batch_id['scene_name'][0]
            frame_start = int(inputs['frame_ids'][0, 0])
            seed = deterministic_trajectory_seed(
                args.trajectory_bank_seed_base, scene_name, frame_start, 0)
            _set_all_rng(seed, args.use_cuda)
            upstream_started = time.perf_counter()
            prediction, auxiliary = model._v4_upstream_predictions(inputs)
            raw = model._future_predictions_world(prediction, inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timings['upstream_generation_ms'] += 1000 * (
                time.perf_counter() - upstream_started)
            graph_started = time.perf_counter()
            edge_index, edge_feat, edge_weight, _ = model._v4_graph_inputs(
                inputs, auxiliary)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timings['graph_build_ms'] += 1000 * (
                time.perf_counter() - graph_started)
            output = model.multiway_coupler(
                raw, inputs['obs_traj_world'][:, -1], edge_index,
                edge_feat, edge_weight, scene_index=inputs['scene_index'],
                hard=False, profile=True)
            for name, value in output['profile_ms'].items():
                timings[name] += value
            gt = inputs['world_coord'][args.obs_length:].permute(1, 0, 2)
            packed_inputs = {
                'gt_future_world': gt,
                'future_mask': seq_list[args.obs_length:].permute(1, 0).bool(),
                'agent_mask': seq_list.cumprod(0)[-1].bool(),
                'scene_index': inputs['scene_index'], 'edge_index': edge_index,
                'edge_weight': edge_weight,
            }
            loss = coupling_loss(args, packed_inputs, output)
            backward_started = time.perf_counter()
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timings['backward_ms'] += 1000 * (
                time.perf_counter() - backward_started)
            model.zero_grad(set_to_none=True)
            counters['windows'] += 1
            counters['packs'] += 1
            counters['agents'] += int(raw.shape[0])
            counters['edges'] += int(edge_index.shape[1])
    elapsed = time.perf_counter() - started_total
    denominator = max(counters['packs'], 1)
    return {
        **counters, 'seconds': elapsed,
        'windows_per_second': counters['windows'] / max(elapsed, 1e-9),
        'scenes_per_second': counters['windows'] / max(elapsed, 1e-9),
        'agents_per_second': counters['agents'] / max(elapsed, 1e-9),
        'avg_scenes_per_pack': 1.0,
        'avg_agents_per_pack': counters['agents'] / denominator,
        'avg_edges_per_pack': counters['edges'] / denominator,
        **{name: value / denominator for name, value in timings.items()},
        'peak_allocated_mib': (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == 'cuda' else 0.0),
        'peak_reserved_mib': (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == 'cuda' else 0.0),
        **monitor.summary(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', required=True, choices=[
        'eth', 'hotel', 'univ', 'zara1', 'zara2'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache_root', required=True)
    parser.add_argument('--max_windows', type=int, default=12)
    parser.add_argument('--max_agents', type=int, default=32)
    parser.add_argument('--max_edges', type=int, default=256)
    parser.add_argument('--max_scenes', type=int, default=8)
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    args = make_args(cli)
    model, device = load_model(args)
    equivalence = verify_equivalence(args, model, device)
    cases = {
        'A_online_single': benchmark_online(
            args, model, device, cli.max_windows),
        'B_cached_single': benchmark_cached(
            args, model, device, False, cli.max_windows),
        'C_cached_packed': benchmark_cached(
            args, model, device, True, cli.max_windows),
    }
    baseline = cases['A_online_single']['windows_per_second']
    for value in cases.values():
        value['speedup_vs_A'] = value['windows_per_second'] / max(
            baseline, 1e-9)
    payload = {
        'test_set': cli.test_set,
        'checkpoint': str(Path(cli.checkpoint).resolve()),
        'cache_root': str(Path(cli.cache_root).resolve()),
        'K': 20,
        'cached_seeds_merged': False,
        'online_cache_equivalence': equivalence,
        'packing_limits': {
            'max_agents_per_pack': cli.max_agents,
            'max_edges_per_pack': cli.max_edges,
            'max_scenes_per_pack': cli.max_scenes,
        },
        'cases': cases,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
