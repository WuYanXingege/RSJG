"""Read-only Stage-A validation and latent/relation audit helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import compute_metric_mask
from src.models.joint_dependency_v2.future_teacher import (
    future_pair_descriptor,
)
from src.utils import isolated_random_seed


AUDIT_METRICS = (
    'minADE@K', 'minFDE@K', 'JADE', 'JFDE',
    'Joint_Goal_Endpoint_Error', 'Joint_Goal_Compatibility',
    'Relative_Motion_Error',
)


def summarize_repeated_metrics(
        runs: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return population mean/std for a homogeneous list of metric maps."""
    if not runs:
        raise ValueError('At least one validation run is required')
    keys = tuple(runs[0])
    if any(tuple(run) != keys for run in runs):
        raise ValueError('Repeated-validation metric keys/order differ')
    return {
        key: {
            'mean': float(np.mean([float(run[key]) for run in runs])),
            'std': float(np.std([float(run[key]) for run in runs])),
        }
        for key in keys
    }


def repeated_validation(evaluator, epoch: int, seeds: Sequence[int],
                        external_edge_cache_root=None):
    """Evaluate the deployed stochastic policy under fixed independent seeds."""
    runs = []
    for seed in seeds:
        raw = evaluator._evaluate_epoch(
            epoch, mode='valid', evaluation_seed=int(seed),
            metric_names=AUDIT_METRICS,
            external_edge_cache_root=external_edge_cache_root)
        values = {
            name: float(raw['valid_' + name]) for name in AUDIT_METRICS}
        runs.append({'seed': int(seed), **values})
    metric_runs = [
        {name: run[name] for name in AUDIT_METRICS} for run in runs]
    return {'runs': runs, 'summary': summarize_repeated_metrics(metric_runs)}


def _distribution(values: Iterable[float]):
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {'count': 0}
    return {
        'count': int(array.size),
        'mean': float(array.mean()),
        'std': float(array.std()),
        'min': float(array.min()),
        'p10': float(np.quantile(array, 0.10)),
        'median': float(np.median(array)),
        'p90': float(np.quantile(array, 0.90)),
        'max': float(array.max()),
    }


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp_min(1e-12)
    return -(probability * probability.log()).sum(dim=-1)


def _append(store, name, value):
    if torch.is_tensor(value):
        store[name].extend(value.detach().float().cpu().reshape(-1).tolist())
    else:
        store[name].append(float(value))


def _goal_probability(net, inputs):
    count = inputs['x_augmented'].shape[1]
    image = inputs['tensor_image'].unsqueeze(0).repeat(count, 1, 1, 1)
    maps = inputs['input_traj_maps'][:, :net.args.obs_length]
    logits = net.goal_module(torch.cat((image, maps), dim=1))
    return torch.sigmoid(logits[:, -1:])


def _shuffle_future_within_scene(future, scene_index):
    shuffled = future.clone()
    for scene in torch.unique(scene_index, sorted=True):
        index = torch.nonzero(scene_index.eq(scene), as_tuple=False).flatten()
        if index.numel() > 1:
            shuffled[index] = future[index.roll(1)]
    return shuffled


def _run_sampler(net, structured, inputs, probability, seed):
    with isolated_random_seed(seed, use_cuda=inputs['world_coord'].is_cuda):
        return net.jdv2_sampler(
            structured['unary_score'], structured['goal_candidates_world'],
            probability, inputs['scene_index'], structured['edge_index'],
            structured['edge_feat'], structured['agent_feat'],
            inputs['obs_traj_world'][:, -1],
            structured['base_relation_logits'], net.jdv2_dynamic_relation,
            net.jdv2_joint_energy,
            candidate_mask=structured['candidate_mask'],
            sampling_mode=net.args.joint_sampling_mode,
            use_scene_latent=net.args.use_scene_latent,
            use_dynamic_relation=net.args.use_dynamic_relation,
            use_joint_energy=net.args.use_joint_energy)


@torch.no_grad()
def audit_scene_and_relation(evaluator, seed: int = 2035):
    """Audit p/q and dynamic relations over the full validation split.

    The function never updates parameters or optimizer state. Future shuffles
    are performed only within each synchronized scene window.
    """
    net = evaluator.net
    net.eval()
    scene_values = defaultdict(list)
    relation_values = defaultdict(list)
    p_sum = torch.zeros(net.args.jdv2_scene_modes, dtype=torch.float64)
    q_sum = torch.zeros_like(p_sum)
    p_hard = torch.zeros_like(p_sum)
    q_hard = torch.zeros_like(p_sum)
    relation_dynamic_sum = torch.zeros(
        net.args.jdv2_relation_modes, dtype=torch.float64)
    relation_teacher_sum = torch.zeros_like(relation_dynamic_sum)
    relation_dynamic_count = 0
    relation_teacher_count = 0
    edge_positive_windows = 0
    zero_edge_windows = 0
    z_changed = [[] for _ in range(net.args.jdv2_scene_modes)]
    z_goal_delta = [[] for _ in range(net.args.jdv2_scene_modes)]

    with isolated_random_seed(seed, use_cuda=evaluator.device.type == 'cuda'):
        for window_index, (batch_data, batch_id) in enumerate(
                evaluator.data_loaders['valid']):
            inputs, _ = net.prepare_inputs(batch_data, batch_id)
            with evaluator._autocast_context():
                structured = net._jdv2_goal_outputs(
                    inputs, _goal_probability(net, inputs), sample=False,
                    include_teacher=True)
            p = structured['scene_prior']['prob'].float()
            q = structured['scene_posterior']['prob'].float()
            p_log = structured['scene_prior']['log_prob'].float()
            q_log = structured['scene_posterior']['log_prob'].float()
            # Mathematical KL is non-negative. In the fully collapsed regime
            # its FP32 residual is around 1e-8 and can round slightly below
            # zero, so report that numerical residue as zero.
            kl = (q * (q_log - p_log)).sum(dim=-1).clamp_min(0)
            _append(scene_values, 'prior_entropy', _entropy(p))
            _append(scene_values, 'posterior_entropy', _entropy(q))
            _append(scene_values, 'kl_q_p', kl)
            _append(scene_values, 'mean_abs_q_minus_p',
                    (q - p).abs().mean(dim=-1))
            p_sum += p.sum(dim=0).cpu().double()
            q_sum += q.sum(dim=0).cpu().double()
            p_hard += torch.bincount(
                p.argmax(-1).cpu(), minlength=p.shape[-1]).double()
            q_hard += torch.bincount(
                q.argmax(-1).cpu(), minlength=q.shape[-1]).double()

            future = inputs['world_coord'][net.args.obs_length:].permute(
                1, 0, 2).contiguous().float()
            shuffled = _shuffle_future_within_scene(
                future, inputs['scene_index'])
            anchor = inputs['obs_traj_world'][:, -1].float()
            shuffled_velocity = torch.diff(
                torch.cat((anchor[:, None], shuffled), dim=1), dim=1
            ) / float(net.args.trajectory_dt)
            shuffled_q = net.jdv2_future_teacher.scene_posterior(
                shuffled, shuffled_velocity, anchor,
                structured['scene_prior']['logits'],
                inputs['scene_index'])['prob'].float()
            _append(scene_values, 'future_shuffle_l1',
                    (shuffled_q - q).abs().sum(dim=-1))

            predicted = _run_sampler(
                net, structured, inputs, p, seed + window_index)
            for mode in range(net.args.jdv2_scene_modes):
                fixed = torch.zeros_like(p)
                fixed[:, mode] = 1.0
                intervention = _run_sampler(
                    net, structured, inputs, fixed, seed + window_index)
                z_changed[mode].append(float((
                    predicted['candidate_index'] !=
                    intervention['candidate_index']).float().mean().cpu()))
                z_goal_delta[mode].append(float(torch.linalg.vector_norm(
                    predicted['goals'] - intervention['goals'], dim=-1
                ).mean().cpu()))

            edge_index = structured['edge_index']
            edge_count = edge_index.shape[1]
            if edge_count == 0:
                zero_edge_windows += 1
                continue
            edge_positive_windows += 1
            teacher = structured['relation_teacher']['prob'].float()
            relation_teacher_sum += teacher.sum(dim=0).cpu().double()
            relation_teacher_count += teacher.shape[0]
            _append(relation_values, 'teacher_entropy', _entropy(teacher))

            shuffled_descriptor = future_pair_descriptor(
                shuffled, anchor, edge_index)
            shuffled_teacher = net.jdv2_future_teacher.relation_posterior(
                shuffled_descriptor)['prob'].float()
            _append(relation_values, 'future_shuffle_teacher_tv',
                    0.5 * (shuffled_teacher - teacher).abs().sum(dim=-1))

            chunk = net.args.jdv2_edge_chunk_size
            for start in range(0, edge_count, chunk):
                stop = min(start + chunk, edge_count)
                local_edge = edge_index[:, start:stop]
                base_logits = structured['base_relation_logits'][start:stop]
                dynamic = net.jdv2_dynamic_relation.full_pair_relation(
                    base_logits, structured['goal_candidates_world'], anchor,
                    local_edge, mode_enabled=True)['prob'].float()
                relation_dynamic_sum += dynamic.sum(
                    dim=(0, 1, 2, 3)).cpu().double()
                relation_dynamic_count += int(np.prod(dynamic.shape[:-1]))
                _append(relation_values, 'dynamic_entropy',
                        _entropy(dynamic))
                _append(relation_values, 'candidate_pair_variance',
                        dynamic.var(dim=(2, 3), unbiased=False).mean(dim=(1, 2)))
                _append(relation_values, 'z_variance',
                        dynamic.var(dim=1, unbiased=False).mean(dim=(1, 2, 3)))
                history = F.softmax(base_logits.float(), dim=-1)[
                    :, None, None, None, :]
                _append(relation_values, 'dynamic_vs_history_kl', (
                    dynamic * (
                        dynamic.clamp_min(1e-12).log() -
                        history.clamp_min(1e-12).log())).sum(dim=-1))
                _append(relation_values, 'dynamic_vs_history_tv',
                        0.5 * (dynamic - history).abs().sum(dim=-1))

    p_total = float(p_sum.sum())
    q_total = float(q_sum.sum())
    scene_audit = {
        name: _distribution(values) for name, values in scene_values.items()}
    scene_audit.update({
        'prior_soft_usage': (p_sum / max(p_total, 1.0)).tolist(),
        'posterior_soft_usage': (q_sum / max(q_total, 1.0)).tolist(),
        'prior_hard_usage': (p_hard / max(float(p_hard.sum()), 1.0)).tolist(),
        'posterior_hard_usage': (
            q_hard / max(float(q_hard.sum()), 1.0)).tolist(),
        'z_intervention': {
            str(mode): {
                'candidate_changed_fraction': _distribution(z_changed[mode]),
                'goal_delta_world_m': _distribution(z_goal_delta[mode]),
            } for mode in range(net.args.jdv2_scene_modes)},
    })
    relation_audit = {
        name: _distribution(values)
        for name, values in relation_values.items()}
    relation_audit.update({
        'edge_positive_windows': edge_positive_windows,
        'zero_edge_windows': zero_edge_windows,
        'dynamic_soft_usage': (
            relation_dynamic_sum / max(relation_dynamic_count, 1)).tolist(),
        'teacher_soft_usage': (
            relation_teacher_sum / max(relation_teacher_count, 1)).tolist(),
    })
    return {'scene': scene_audit, 'relation': relation_audit}


__all__ = [
    'AUDIT_METRICS', 'audit_scene_and_relation', 'repeated_validation',
    'summarize_repeated_metrics',
]
