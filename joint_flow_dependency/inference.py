"""Observed-only continuous joint sampling -> scene medoids -> one head -> one bridge."""
import torch
from bcep.motion_frame import frame, to_world
from bcep_v2.provider import spatial_encoding, GEOMETRY
from ebjd.sampling import differentiable_sample
from .model import undirected_edges, sample_latents
from .sampling import base_noise, bridge_noise, select_joint_samples


@torch.no_grad()
def encode_observed(provider, batch):
    assert len(batch.observed) == 1 and batch.valid.all()
    return spatial_encoding(provider.base, batch.observed[0], batch.semantic_maps[0], batch.valid[0])


@torch.no_grad()
def flow_world(provider, observed, context, z):
    count, n, _ = z.shape
    residual, _ = provider.base.flow.transform(z.reshape(count*n, 2),
        context[None].expand(count, -1, -1).reshape(count*n, -1), 12*provider.base.dt)
    origin, rotation, scale = frame(observed, 12, provider.base.dt, provider.base.scales[-1])
    return to_world(residual.reshape(count, n, 2), origin[None], rotation[None], scale)


@torch.no_grad()
def sample_endpoints(provider, dependence, batch, context, map_features, *, independent=False, shared_base=None, seed=2035):
    observed = batch.observed[0]
    edges = undirected_edges(observed)
    scene, agents = batch.scene_ids[0], batch.agent_ids[0]
    epsilon, xi = base_noise(scene, agents, edges, seed=seed) if shared_base is None else shared_base
    epsilon, xi = epsilon.to(observed.device), xi.to(observed.device)
    if independent:
        z = epsilon
        factorization = None
    else:
        factorization = dependence(observed, context, edges)
        z = sample_latents(factorization, epsilon, xi)
    raw = flow_world(provider, observed, context, z)
    indices, distortion = select_joint_samples(raw)
    selected = raw[indices]
    # Exactly one frozen bounded correction, applied after whole-scene compression.
    corrected, shift = provider.head(provider.base, observed, batch.semantic_maps[0], batch.valid[0],
        selected, context, map_features, dict(layout=GEOMETRY))
    assert shift.norm(dim=-1).max() < 0.5
    noise = bridge_noise(scene, agents, indices.cpu().tolist(), seed=seed).to(observed.device)
    return dict(raw=raw, selected=selected, corrected=corrected[None], noise=noise,
                raw_indices=indices, distortion=distortion, R=None if independent else factorization['R'],
                epsilon=epsilon, xi=xi, correction_calls=1)


@torch.no_grad()
def decode(model, context, corrected, noise):
    path, _, _ = differentiable_sample(model.bridge, context, noise, 20,
                                     proposal_goals=corrected, proposal_eta=0.)
    assert torch.isfinite(path).all()
    torch.testing.assert_close(path[..., -1, :], corrected, rtol=1e-6, atol=1e-6)
    return path


@torch.no_grad()
def generate(model, provider, dependence, batch, *, independent=False, seed=2035):
    """The supplied batch must come from the truth=False reader. Future is never read."""
    model.eval(); provider.eval(); dependence.eval()
    context, features = encode_observed(provider, batch)
    endpoints = sample_endpoints(provider, dependence, batch, context, features, independent=independent, seed=seed)
    bridge_context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    path = decode(model, bridge_context, endpoints['corrected'], endpoints['noise'])
    return path, endpoints, dict(ddim_calls=1, ddim_steps=20, correction_calls=1,
                                 flow_context=context, map_features=features, bridge_context=bridge_context)


def endpoint_minfde(endpoints, truth):
    """Per-agent minFDE for one [K,N,2] set, using only endpoint errors."""
    return (endpoints.float()-truth[None].float()).norm(dim=-1).amin(0)
