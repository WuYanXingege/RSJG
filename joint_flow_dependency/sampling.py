"""Independent keyed streams, whole-scene greedy medoids, transported bridge noise."""
import numpy as np
import torch
from bcep_joint.common import local_seed
from ebjd.unified import gather_candidates


def keyed_normal(stream, *key, shape):
    gen = torch.Generator().manual_seed(local_seed('joint-flow-dependence-v1', stream, *key))
    return torch.randn(*shape, generator=gen, dtype=torch.float32)


def base_noise(scene, agents, edges, seed=2035, count=256):
    assert seed == 2035
    epsilon = torch.stack([keyed_normal('personal', scene, str(a), seed, shape=(count, 2)) for a in agents], 1)
    edge_values = []
    for i, j in edges.cpu().tolist():
        pair = sorted((str(agents[i]), str(agents[j])))
        edge_values.append(keyed_normal('undirected-edge', scene, *pair, seed, shape=(count, 2)))
    xi = torch.stack(edge_values, 1) if edge_values else torch.empty(count, 0, 2)
    return epsilon, xi


def bridge_noise(scene, agents, raw_indices, seed=2035):
    """Original raw joint IDs, not compressed ranks, identify bridge occurrences."""
    indices = [int(k) for k in raw_indices]
    z = torch.zeros(1, len(indices), len(agents), 12, 2)
    for i, agent in enumerate(agents):
        for k, raw in enumerate(indices):
            z[0, k, i] = keyed_normal('bridge', scene, str(agent), raw, seed, shape=(12, 2))
    return z


def select_joint_samples(endpoints, k=20):
    """Exact greedy medoids on mean per-agent Euclidean distance; stable raw-ID ties.

    Discrete distance accounting uses float64 CPU, as the existing finite-support
    selector does. Model, Flow, likelihood and bridge arithmetic remain FP32.
    """
    x = endpoints.detach().cpu().double().numpy()
    m, n, d = x.shape
    assert d == 2 and n > 0 and 0 < k <= m and np.isfinite(x).all()
    distance = np.zeros((m, m), dtype=np.float64)
    for i in range(n):
        distance += np.linalg.norm(x[:, None, i]-x[None, :, i], axis=-1)
    distance /= n
    first = int(distance.sum(0).argmin())
    chosen = [first]
    nearest = distance[:, first].copy()
    used = np.zeros(m, dtype=bool)
    used[first] = True
    costs = [float(nearest.sum())]
    for _ in range(1, k):
        gain = np.maximum(nearest[:, None]-distance, 0).sum(0)
        gain[used] = -np.inf
        idx = int(gain.argmax())
        chosen.append(idx)
        used[idx] = True
        nearest = np.minimum(nearest, distance[:, idx])
        costs.append(float(nearest.sum()))
    assert len(set(chosen)) == k
    return torch.tensor(chosen, device=endpoints.device), costs


def shuffled_correspondence(goals, noise, scene, agents, repetition, seed=2035):
    assert goals.shape[:3] == noise.shape[:3] == (1, 20, len(agents))
    assert repetition in range(5)
    pi = torch.arange(20)[None, :, None].expand(1, 20, len(agents)).clone()
    if len(agents) > 1:
        for i, agent in enumerate(agents):
            gen = torch.Generator().manual_seed(local_seed('joint-flow-correspondence-shuffle-v1', scene, str(agent), repetition, seed))
            pi[0, :, i] = torch.randperm(20, generator=gen)
    pi = pi.to(goals.device)
    changed_goals, changed_noise = gather_candidates(goals, pi), gather_candidates(noise, pi)
    assert torch.equal(gather_candidates(changed_goals, pi.argsort(1)), goals)
    assert torch.equal(gather_candidates(changed_noise, pi.argsort(1)), noise)
    return changed_goals, changed_noise, pi
