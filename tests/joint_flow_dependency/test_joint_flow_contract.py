import numpy as np
import torch
from joint_flow_dependency.model import EdgeDependence, undirected_edges, delta_nll, sample_latents
from joint_flow_dependency.sampling import select_joint_samples, base_noise, bridge_noise, shuffled_correspondence


def example(n=4):
    torch.manual_seed(3201)
    return torch.randn(n, 8, 2), torch.randn(n, 128)


def test_factor_covariance_and_gradient():
    observed, context = example()
    model = EdgeDependence()
    pack = model(observed, context)
    r = pack['R']; f = pack['factor']
    assert float(torch.linalg.eigvalsh(r.detach()).min()) > 0
    for i in range(4):torch.testing.assert_close(r[2*i:2*i+2, 2*i:2*i+2], torch.eye(2), rtol=0, atol=5e-7)
    for k, (i, j) in enumerate(pack['edges'].tolist()):
        left = torch.linalg.solve_triangular(pack['L'][i], pack['B'][k, 0], upper=False)
        right = torch.linalg.solve_triangular(pack['L'][j], pack['B'][k, 1], upper=False)
        torch.testing.assert_close(r[2*i:2*i+2, 2*j:2*j+2], left@right.T, atol=1e-7, rtol=1e-5)
    truth = torch.randn(4, 2)
    actual = delta_nll(r, truth)
    # Independent full Gaussian library likelihood as a reference.
    gaussian = torch.distributions.MultivariateNormal(torch.zeros(8), covariance_matrix=r)
    iid = torch.distributions.MultivariateNormal(torch.zeros(8), torch.eye(8))
    expected = (iid.log_prob(truth.flatten())-gaussian.log_prob(truth.flatten()))/4
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-7)
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert float(model.net[0].weight.grad.abs().sum()) > 0
    assert float(model.net[-1].weight.grad.abs().sum()) > 0
    # Empirical covariance tests the actual sampler, not just the R expression.
    torch.manual_seed(123)
    eps = torch.randn(60000, 4, 2); xi = torch.randn(60000, len(pack['edges']), 2)
    z = sample_latents(pack, eps, xi).detach().flatten(1)
    empirical = z.T@z/len(z)
    torch.testing.assert_close(empirical, r.detach(), rtol=0, atol=.025)


def test_singleton_isolated_and_undirected_union():
    observed, context = example(12)
    edges = undirected_edges(observed)
    assert (edges[:, 0] < edges[:, 1]).all() and len(edges.unique(dim=0)) == len(edges)
    model = EdgeDependence()
    for n in (1, 4):
        pack = model(observed[:n], context[:n], torch.empty(0, 2, dtype=torch.long))
        assert torch.equal(pack['R'], torch.eye(2*n))
        eps = torch.randn(7, n, 2)
        assert torch.equal(sample_latents(pack, eps, torch.empty(7, 0, 2)), eps)
        loss = delta_nll(pack['R'], torch.randn(n, 2))
        assert loss.requires_grad and float(loss.detach()) == 0.
        model.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_greedy_whole_joint_samples():
    torch.manual_seed(9)
    samples = torch.randn(9, 3, 2)
    selected, costs = select_joint_samples(samples, 4)
    d = (samples[:, None].double()-samples[None].double()).norm(dim=-1).mean(-1).numpy()
    chosen = []
    for idx in selected.tolist():
        remaining = [i for i in range(9) if i not in chosen]
        objective = [d[:, chosen+[i]].min(1).sum() for i in remaining]
        # Risk and gain reductions can differ by FP64 roundoff on a true tie.
        assert abs(objective[remaining.index(idx)]-min(objective)) < 1e-12
        well_separated = [i for i, v in zip(remaining, objective) if v-min(objective) < 1e-12]
        if len(well_separated) == 1:
            assert idx == remaining[int(np.argmin(objective))]
        chosen.append(idx)
    assert np.all(np.diff(costs) <= 0)
    tied, _ = select_joint_samples(torch.zeros(25, 2, 2))
    assert tied.tolist() == list(range(20))


def test_stream_identity_and_shuffling():
    edges = torch.tensor([[0, 1], [1, 2]])
    eps, xi = base_noise('s', ['a','b','c'], edges)
    reordered_eps, reordered_xi = base_noise('s', ['c','b','a'], torch.tensor([[2,1],[1,0]]))
    assert torch.equal(eps, reordered_eps.flip(1)) and torch.equal(xi, reordered_xi)
    assert not torch.equal(eps[:, 0], xi[:, 0])
    ids = list(range(0, 40, 2))
    noise = bridge_noise('s', ['a','b','c'], ids)
    assert torch.equal(noise[:, 0], bridge_noise('s', ['a','b','c'], [0])[:, 0])
    assert not torch.equal(noise[:, 1], bridge_noise('s', ['a','b','c'], [1])[:, 0])
    goals = torch.randn(1, 20, 3, 2)
    for rep in range(5):
        changed, changed_noise, pi = shuffled_correspondence(goals, noise, 's', ['a','b','c'], rep)
        for i in range(3):
            assert torch.equal(changed[0,:,i], goals[0,pi[0,:,i],i])
            assert torch.equal(changed_noise[0,:,i], noise[0,pi[0,:,i],i])
        target = torch.randn(3, 2)
        assert torch.equal((changed-target[None,None]).norm(dim=-1).amin(1), (goals-target[None,None]).norm(dim=-1).amin(1))
