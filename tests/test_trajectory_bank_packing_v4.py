from types import SimpleNamespace

import pytest
import torch

from src.metrics import JADE, JFDE, minADE_at_K, minFDE_at_K
from src.models.interaction_graph import EDGE_FEATURE_DIM, build_interaction_graph
from src.models.multiway_trajectory_coupler import (
    RelationAwareMultiwayCoupler,
    assert_same_trajectory_set,
)
from src.multiway_coupling_loss import (
    compute_multiway_coupling_loss,
    compute_scene_balanced_multiway_coupling_loss,
)
from src.trajectory_bank_cache import (
    DynamicSceneBatchSampler,
    deterministic_trajectory_seed,
    pack_trajectory_bank_records,
)


def _scene(n, offset, seed_index=0, seed=100):
    generator = torch.Generator().manual_seed(seed + int(offset * 10))
    obs = torch.randn(n, 8, 2, generator=generator) + offset
    raw = obs[:, -1, None, None] + torch.randn(
        n, 20, 12, 2, generator=generator).cumsum(2) * 0.1
    gt = raw[:, 0] + 0.02
    return {
        'window_id': f'w{offset}',
        'scene_name': f's{offset}',
        'frame_ids': torch.arange(20),
        'agent_ids': torch.arange(n),
        'obs_world': obs,
        'gt_future_world': gt,
        'future_mask': torch.ones(n, 12, dtype=torch.bool),
        'agent_mask': torch.ones(n, dtype=torch.bool),
        'raw_future_world': raw,
        'cached_seed_index': seed_index,
        'cached_seed': seed,
    }


def _graph_for_scenes(scenes):
    indices, features, weights = [], [], []
    offset = 0
    for scene in scenes:
        index, feature, weight = build_interaction_graph(
            scene['obs_world'], graph_type='full')
        indices.append(index + offset)
        features.append(feature)
        weights.append(weight)
        offset += scene['obs_world'].shape[0]
    return (torch.cat(indices, 1), torch.cat(features), torch.cat(weights))


def _coupler():
    torch.manual_seed(11)
    model = RelationAwareMultiwayCoupler(
        trajectory_dim=24,
        edge_dim=EDGE_FEATURE_DIM,
        num_relation_modes=4,
        trajectory_pair_rank=4,
        sync_iterations=2,
        sinkhorn_iterations=4,
        sync_temperature=0.2,
        inference_edge_topk=0,
    )
    return model.eval()


def test_deterministic_cache_seed_is_stable_and_window_specific():
    first = deterministic_trajectory_seed(42, 'univ', 100, 0)
    assert first == deterministic_trajectory_seed(42, 'univ', 100, 0)
    assert len({
        first,
        deterministic_trajectory_seed(42, 'univ', 101, 0),
        deterministic_trajectory_seed(42, 'univ', 100, 1),
        deterministic_trajectory_seed(42, 'eth', 100, 0),
    }) == 4


def test_pack_keeps_seed_axis_out_of_k_and_builds_prefixes():
    packed = pack_trajectory_bank_records([
        _scene(2, 0, seed_index=1), _scene(3, 10, seed_index=3)])
    assert packed['raw_future_world'].shape == (5, 20, 12, 2)
    assert packed['scene_ptr'].tolist() == [0, 2, 5]
    assert packed['scene_index'].tolist() == [0, 0, 1, 1, 1]
    assert packed['cached_seed_indices'].tolist() == [1, 3]
    assert packed['K'] == 20


def test_dynamic_packing_respects_all_limits_without_splitting():
    dataset = SimpleNamespace(records=[
        {'num_agents': 3, 'edge_upper_bound': 3},
        {'num_agents': 2, 'edge_upper_bound': 1},
        {'num_agents': 4, 'edge_upper_bound': 6},
        {'num_agents': 1, 'edge_upper_bound': 0},
    ])
    dataset.__len__ = lambda: len(dataset.records)
    # Special methods are looked up on the type, not the instance.
    class Wrapper:
        records = dataset.records
        def __len__(self):
            return len(self.records)
    sampler = DynamicSceneBatchSampler(
        Wrapper(), shuffle=False, max_agents=5, max_edges=6,
        max_scenes=2, seed=0)
    packs = list(sampler)
    assert sorted(index for pack in packs for index in pack) == [0, 1, 2, 3]
    for pack in packs:
        assert len(pack) <= 2
        assert sum(Wrapper.records[i]['num_agents'] for i in pack) <= 5
        assert sum(Wrapper.records[i]['edge_upper_bound'] for i in pack) <= 6


@torch.no_grad()
def test_packed_forward_equals_independent_for_every_v4_stage():
    scenes = [_scene(3, 0), _scene(2, 20)]
    packed = pack_trajectory_bank_records(scenes)
    edge_index, edge_feat, edge_weight = _graph_for_scenes(scenes)
    model = _coupler()
    packed_result = model(
        packed['raw_future_world'], packed['obs_world'][:, -1],
        edge_index, edge_feat, edge_weight,
        scene_index=packed['scene_index'], hard=True)

    node_start = edge_start = 0
    node_keys = (
        'trajectory_features', 'soft_permutation',
        'synchronized_permutation', 'permutation', 'aligned_trajectories')
    edge_keys = ('relation_posterior', 'pair_score', 'pair_gate')
    for scene in scenes:
        n = scene['obs_world'].shape[0]
        local_index, local_feat, local_weight = build_interaction_graph(
            scene['obs_world'], graph_type='full')
        e = local_index.shape[1]
        single = model(
            scene['raw_future_world'], scene['obs_world'][:, -1],
            local_index, local_feat, local_weight,
            scene_index=torch.zeros(n, dtype=torch.long), hard=True)
        for key in node_keys:
            torch.testing.assert_close(
                packed_result[key][node_start:node_start+n], single[key],
                atol=1e-6, rtol=1e-5)
        for key in edge_keys:
            torch.testing.assert_close(
                packed_result[key][edge_start:edge_start+e], single[key],
                atol=1e-6, rtol=1e-5)
        node_start += n
        edge_start += e


@torch.no_grad()
def test_identical_coordinates_in_different_scenes_never_cross_connect():
    scene_a = _scene(3, 0)
    scene_b = {**_scene(3, 1), 'obs_world': scene_a['obs_world'].clone()}
    packed = pack_trajectory_bank_records([scene_a, scene_b])
    index, _, _ = build_interaction_graph(
        packed['obs_world'], packed['scene_index'], graph_type='full')
    assert index.shape[1] == 6
    assert torch.equal(
        packed['scene_index'][index[0]], packed['scene_index'][index[1]])


def _loss_kwargs():
    return dict(
        lambda_fde=1.0, lambda_rel_geom=1.0, lambda_rel_end=1.0,
        tau_joint=0.5, tau_pair_target=0.5, tau_pair_pred=0.2,
        no_harm_margin=0.0, weight_alignment=1.0,
        weight_pair_score=1.0, weight_pair_assignment=1.0,
        weight_no_harm=1.0, weight_perm_entropy=1.0,
        weight_relation_prior=1.0, weight_gate_reg=1.0,
        gate_regularization='l1')


def test_scene_balanced_loss_is_exact_mean_of_single_scene_components():
    scenes = [_scene(3, 0), _scene(1, 20)]
    packed = pack_trajectory_bank_records(scenes)
    edge_index, edge_feat, edge_weight = _graph_for_scenes(scenes)
    model = _coupler()
    output = model(
        packed['raw_future_world'], packed['obs_world'][:, -1],
        edge_index, edge_feat, edge_weight,
        scene_index=packed['scene_index'], hard=False)
    balanced = compute_scene_balanced_multiway_coupling_loss(
        output['soft_permutation'], output['pair_score'],
        packed['raw_future_world'], packed['gt_future_world'], edge_index,
        future_mask=packed['future_mask'], agent_mask=packed['agent_mask'],
        scene_index=packed['scene_index'],
        relation_posterior=output['relation_posterior'],
        relation_prior=output['relation_prior'], pair_gate=output['pair_gate'],
        edge_weight=edge_weight, **_loss_kwargs())

    singles = []
    node_start = edge_start = 0
    for scene in scenes:
        n = scene['obs_world'].shape[0]
        local_index, _, local_weight = build_interaction_graph(
            scene['obs_world'], graph_type='full')
        e = local_index.shape[1]
        singles.append(compute_multiway_coupling_loss(
            output['soft_permutation'][node_start:node_start+n],
            output['pair_score'][edge_start:edge_start+e],
            scene['raw_future_world'], scene['gt_future_world'], local_index,
            future_mask=scene['future_mask'], agent_mask=scene['agent_mask'],
            relation_posterior=output['relation_posterior'][
                edge_start:edge_start+e],
            relation_prior=output['relation_prior'][edge_start:edge_start+e],
            pair_gate=output['pair_gate'][edge_start:edge_start+e],
            edge_weight=local_weight, **_loss_kwargs()))
        node_start += n
        edge_start += e
    for key in (
            'loss_alignment', 'loss_pair_score', 'loss_pair_assignment',
            'loss_no_harm', 'loss_perm_entropy', 'loss_relation_prior',
            'loss_gate_reg', 'loss_identity', 'loss_total'):
        expected = torch.stack([item[key] for item in singles]).mean()
        torch.testing.assert_close(balanced[key], expected, atol=1e-7, rtol=1e-6)


@torch.no_grad()
def test_singleton_empty_graph_is_finite_bijective_and_metrics_preserved():
    scene = _scene(1, 0)
    index, feature, weight = build_interaction_graph(
        scene['obs_world'], graph_type='full')
    output = _coupler()(
        scene['raw_future_world'], scene['obs_world'][:, -1],
        index, feature, weight, hard=True)
    assert index.shape == (2, 0)
    assert torch.isfinite(output['soft_permutation']).all()
    assert_same_trajectory_set(
        scene['raw_future_world'], output['aligned_trajectories'])
    obs = scene['obs_world'].permute(1, 0, 2).unsqueeze(0).expand(20, -1, -1, -1)
    raw = torch.cat((obs, scene['raw_future_world'].permute(1, 2, 0, 3)), 1)
    aligned = torch.cat((obs, output['aligned_trajectories'].permute(1, 2, 0, 3)), 1)
    gt = torch.cat((scene['obs_world'], scene['gt_future_world']), 1).permute(1, 0, 2)
    for metric in (minADE_at_K, minFDE_at_K, JADE, JFDE):
        raw_value = metric(raw, gt, scene['agent_mask'], None, 8) \
            if metric in (JADE, JFDE) else metric(raw, gt, scene['agent_mask'], 8)
        aligned_value = metric(aligned, gt, scene['agent_mask'], None, 8) \
            if metric in (JADE, JFDE) else metric(aligned, gt, scene['agent_mask'], 8)
        assert raw_value == pytest.approx(aligned_value, abs=1e-7)
