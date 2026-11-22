"""CPU-only synthetic observations; no real split, assets, or future is opened."""
import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from joint_flow_dependency import inference, sampling
from joint_flow_dependency.model import EdgeDependence, undirected_edges


SCENE = 'synthetic-observed-only-scene'
AGENTS = ['person-8', 'person-2', 'person-19']
EDGES = torch.tensor([[0, 1], [0, 2], [1, 2]])
RAW_IDS = [0, 7, 31, 255]


def test_seed_2035_matches_frozen_legacy_bytes():
    # Captured from the unmodified d4fbe6d implementation before seed plumbing.
    expected = (
        '2fd2fbd01e743fa117fe32e2f3ec03785a35a138aaa4aea563a784788b48d3ed',
        'd69d05784039323ba2c3824301359606fe2778470073c938d70084763df296f6',
        '070b7196f74dbf746d07cbe2a704ff84f5e9bbd29f2c0065b5945385994ccd61',
    )
    implicit = (*sampling.base_noise(SCENE, AGENTS, EDGES), sampling.bridge_noise(SCENE, AGENTS, RAW_IDS))
    explicit = (*sampling.base_noise(SCENE, AGENTS, EDGES, seed=2035),
                sampling.bridge_noise(SCENE, AGENTS, RAW_IDS, seed=2035))
    for old, new, digest in zip(implicit, explicit, expected):
        assert torch.equal(old, new)
        assert hashlib.sha256(new.numpy().tobytes()).hexdigest() == digest


def test_all_five_seeds_reproduce_distinct_personal_edge_and_bridge_streams():
    outputs = []
    for seed in range(2035, 2040):
        first = (*sampling.base_noise(SCENE, AGENTS, EDGES, seed=seed),
                 sampling.bridge_noise(SCENE, AGENTS, RAW_IDS, seed=seed))
        second = (*sampling.base_noise(SCENE, AGENTS, EDGES, seed=seed),
                  sampling.bridge_noise(SCENE, AGENTS, RAW_IDS, seed=seed))
        assert all(torch.equal(a, b) for a, b in zip(first, second))
        assert not torch.equal(first[0][:, 0], first[1][:, 0])
        assert not torch.equal(first[0][:12, 0], first[2][0, 0, 0])
        assert all(all(not torch.equal(a, b) for a, b in zip(first, previous)) for previous in outputs)
        outputs.append(first)


@pytest.fixture
def observed_route(monkeypatch):
    class Model(nn.Module):
        def encode_context(self, observed, maps, valid):
            return observed

    class Head(nn.Module):
        def forward(self, base, observed, maps, valid, selected, context, features, geometry):
            return selected, torch.zeros_like(selected)

    provider = nn.Module()
    provider.base = nn.Identity()
    provider.head = Head()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3201)
        dependence = EdgeDependence()
    # Deliberately omit any future/target attribute from the inference batch.
    batch = SimpleNamespace(observed=torch.arange(48).float().reshape(1, 3, 8, 2)/100,
        semantic_maps=torch.zeros(1, 3, 1, 1), valid=torch.ones(1, 3, dtype=torch.bool),
        scene_ids=[SCENE], agent_ids=[AGENTS])
    context = torch.zeros(3, 128)
    monkeypatch.setattr(inference, 'encode_observed', lambda provider, batch: (context, None))
    monkeypatch.setattr(inference, 'flow_world', lambda provider, observed, context, z: z.clone())
    monkeypatch.setattr(inference, 'select_joint_samples',
                        lambda raw: (torch.arange(255, 235, -1), [0.]))
    calls = dict(personal_and_edge=[], bridge=[], decoded_noise=[])

    def base_noise(scene, agents, edges, seed=2035, count=256):
        calls['personal_and_edge'].append(seed)
        return sampling.base_noise(scene, agents, edges, seed=seed, count=count)

    def bridge_noise(scene, agents, raw_indices, seed=2035):
        calls['bridge'].append(seed)
        return sampling.bridge_noise(scene, agents, raw_indices, seed=seed)

    def decode(model, context, corrected, noise):
        calls['decoded_noise'].append(noise.clone())
        return corrected[..., None, :].expand(-1, -1, -1, 12, -1).clone()

    monkeypatch.setattr(inference, 'base_noise', base_noise)
    monkeypatch.setattr(inference, 'bridge_noise', bridge_noise)
    monkeypatch.setattr(inference, 'decode', decode)
    return Model(), provider, dependence, batch, calls


@pytest.mark.parametrize('seed', range(2035, 2040))
@pytest.mark.parametrize('independent', [False, True])
def test_generate_passes_seed_to_every_noise_stream(observed_route, seed, independent):
    model, provider, dependence, batch, calls = observed_route
    _, endpoints, _ = inference.generate(model, provider, dependence, batch,
                                         independent=independent, seed=seed)
    assert calls['personal_and_edge'] == calls['bridge'] == [seed]
    epsilon, xi = sampling.base_noise(SCENE, AGENTS, undirected_edges(batch.observed[0]), seed=seed)
    assert torch.equal(endpoints['epsilon'], epsilon) and torch.equal(endpoints['xi'], xi)
    expected_bridge = sampling.bridge_noise(SCENE, AGENTS, endpoints['raw_indices'], seed=seed)
    assert torch.equal(endpoints['noise'], expected_bridge)
    assert torch.equal(calls['decoded_noise'][0], expected_bridge)
    assert endpoints['raw_indices'].tolist() == list(range(255, 235, -1))
    if independent:
        assert torch.equal(endpoints['raw'], epsilon)


def test_generate_default_remains_2035(observed_route):
    model, provider, dependence, batch, calls = observed_route
    implicit, old_endpoints, _ = inference.generate(model, provider, dependence, batch)
    explicit, new_endpoints, _ = inference.generate(model, provider, dependence, batch, seed=2035)
    assert calls['personal_and_edge'] == calls['bridge'] == [2035, 2035]
    assert torch.equal(implicit, explicit)
    for key in ('epsilon', 'xi', 'raw', 'selected', 'corrected', 'noise', 'raw_indices'):
        assert torch.equal(old_endpoints[key], new_endpoints[key])
