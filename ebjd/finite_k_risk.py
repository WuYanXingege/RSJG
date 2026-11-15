"""Differentiable finite-budget endpoint coverage, without sampled-coordinate ST.

For a discrete radial distribution, E[min of K IID distances] is exact via
order-statistic survival probabilities. The continuous spatial density is
approximated by deterministic inverse-CDF quadrature within *every* component,
including its exterior sectors. This is a training surrogate, not an exact
continuous minFDE expectation, and never participates in inference selection.
"""
import torch


def discrete_best_of_k(distances, masses, k=20):
    """Return [...]; nonnegative distances, normalized nonnegative masses."""
    if k < 1 or distances.shape != masses.shape:
        raise ValueError('invalid order-statistic distribution')
    distance, order = distances.sort(dim=-1, stable=True)
    probability = masses.gather(-1, order)
    probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-30)
    # Reverse sum avoids cancellation of 1-CDF near probability one.
    survival = probability.flip(-1).cumsum(-1).flip(-1)[..., 1:]
    return distance[..., 0] + ((distance[..., 1:] - distance[..., :-1])
                              * survival.clamp(0, 1).pow(k)).sum(-1)


def finite_k_endpoint_risk(head, context, truth, goal_scale, distribution, *, k=20, order=2):
    """Midpoint quantiles, order^2 nodes per 2-D component, train-only GT.

    All probability logits receive the coverage signal. Exterior scale parameters
    remain differentiable as well. No RNG is consumed and no network feature is
    detached. Padding is discarded before geometry, including NaN padding GT.
    """
    if order not in (2, 3, 4):
        raise ValueError('bounded quadrature orders are 2, 3, 4')
    d = distribution
    with torch.autocast(device_type=truth.device.type, enabled=False):
        valid = context.valid
        relative = truth.float()[valid] - context.map_center.float()[valid]
        half = context.crop_width_m / 2
        q = (torch.arange(order, device=truth.device, dtype=torch.float32) + .5) / order
        u, v = torch.meshgrid(q, q, indexing='ij')
        uv = torch.stack((u.flatten(), v.flatten()), -1)
        nodes = order**2
        lo, hi = head.kernel_cdf_bounds(context)
        p = lo[:, None, :] + (hi-lo)[:, None, :] * uv[None]
        offsets = head.grid.float()[:, None, :] * half + head.kernel_sigma_m(context) * torch.special.ndtri(p)
        local_distance = torch.linalg.vector_norm(
            offsets[None] - relative[:, None, None, :], dim=-1)
        scale = d.tail_scale[valid].float() * goal_scale.float()
        radius = half - scale[..., None] * torch.log1p(-uv[:, 0])
        tangent = 2 * uv[:, 1] - 1
        sign = torch.tensor([1., -1., 1., -1.], device=truth.device)[None, :, None]
        axis = (torch.arange(4, device=truth.device) < 2)[None, :, None]
        x = torch.where(axis, sign * radius, tangent * radius)
        y = torch.where(axis, tangent * radius, sign * radius)
        tail_distance = torch.linalg.vector_norm(
            torch.stack((x, y), -1) - relative[:, None, None, :], dim=-1)
        distance = torch.cat((local_distance, tail_distance), 1).flatten(1)
        probability = (head.log_masses(d).exp()[valid][..., None]
                       .expand(-1, -1, nodes) / nodes).flatten(1)
        return (discrete_best_of_k(distance, probability, k) / goal_scale.float()).mean()
