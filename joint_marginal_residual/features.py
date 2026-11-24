"""Label-free compression coverage summaries in common world-meter axes."""
from __future__ import annotations

import torch

FEATURE_VERSION = 'dual-assignment-world-meters-v1'
BASE_DIM = 154  # frozen Flow context128 + history16 + B geometry6 + J summary4
DIFFERENCE_DIM = 5  # P-J: mass, offset x/y, radius; mean allocation excess a


@torch.no_grad()
def coverage_statistics(raw, raw_ids, valid=None, *, return_assignments=False):
    """Compute only [M,K,N] distances, using pre-correction raw medoids.

    Ties use the smallest original raw ID, independent of current slot order.
    Statistics are float64 internally, then stored as compact FP32 features.
    Invalid rows do not enter means, and their returned statistics are zero.
    """
    raw = torch.as_tensor(raw).detach()
    raw_ids = torch.as_tensor(raw_ids, device=raw.device, dtype=torch.long)
    if raw.ndim != 3 or raw.shape[-1] != 2 or raw_ids.ndim != 1:
        raise ValueError('raw must be [M,N,2], raw_ids [K]')
    m, n, _ = raw.shape; k = raw_ids.numel()
    if not (m > 0 and k > 0 and raw_ids.unique().numel() == k):
        raise ValueError('nonempty samples and distinct representatives required')
    if (raw_ids < 0).any() or (raw_ids >= m).any():
        raise ValueError('representative ID outside original sample range')
    valid = (torch.ones(n, dtype=torch.bool, device=raw.device) if valid is None else
             torch.as_tensor(valid, device=raw.device, dtype=torch.bool))
    if valid.shape != (n,) or not valid.any():
        raise ValueError('at least one valid person required')
    values = raw[:, valid].double()
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite valid endpoints')
    selected = values[raw_ids]
    offset = values[:, None] - selected[None]  # [M,K,Nvalid,2], no MxM matrix
    distances = torch.linalg.vector_norm(offset, dim=-1)
    # Sorting the addends makes the joint sum independent of person ordering,
    # including exact geometric ties; it does not change the defined mean.
    joint_distances = distances.sort(dim=-1).values.sum(-1) / values.shape[1]
    order = raw_ids.argsort()
    j = order[joint_distances[:, order].argmin(1)]
    p = order[distances[:, order].argmin(1)]
    rows = torch.arange(m, device=raw.device)
    people = torch.arange(values.shape[1], device=raw.device)
    excess = distances[rows, j] - distances[rows[:, None], p, people[None]]
    assert (excess >= 0).all()
    slots = torch.arange(k, device=raw.device)
    jm = (j[:, None, None] == slots[None, :, None]).expand_as(distances)
    pm = p[:, None] == slots[None, :, None]

    def summarize(member):
        count = member.sum(0)
        denominator = count.clamp_min(1).double()
        mean_offset = (offset * member[..., None]).sum(0) / denominator[..., None]
        radius = (distances * member).sum(0) / denominator
        return torch.cat((count[..., None].double()/m, mean_offset, radius[..., None]), -1)

    sj, sp = summarize(jm), summarize(pm)
    a_j = (excess[:, None] * jm).sum(0) / jm.sum(0).clamp_min(1)
    out_j = raw.new_zeros(k, n, 4, dtype=torch.float32)
    out_p = torch.zeros_like(out_j)
    out_a = raw.new_zeros(k, n, 1, dtype=torch.float32)
    out_j[:, valid] = sj.float(); out_p[:, valid] = sp.float()
    out_a[:, valid, 0] = a_j.float()
    joint_cost = joint_distances[rows, j].mean()
    personal_cost = distances[rows[:, None], p, people[None]].mean()
    gap = excess.mean()
    identity_error = (gap - (joint_cost-personal_cost)).abs()
    assert float(identity_error) <= 1e-12
    if int(valid.sum()) == 1:
        assert torch.equal(j[:, None], p) and not excess.count_nonzero()
    result = dict(stats_j=out_j, stats_p=out_p, mean_a_j=out_a,
        diagnostics=dict(joint_coverage_m=float(joint_cost), personal_coverage_m=float(personal_cost),
            mean_assignment_excess_m=float(gap), gap_identity_error_m=float(identity_error),
            M=m, K=k, valid_agents=int(valid.sum()), distance_elements=m*k*int(valid.sum()),
            feature_version=FEATURE_VERSION))
    if return_assignments:
        result.update(j_slot=j, p_slot=p, excess=excess)
    return result


@torch.no_grad()
def frozen_features(observed, flow_context, teacher, coverage, valid=None):
    """No labels, IDs or future truth: observations plus frozen B output only."""
    teacher = torch.as_tensor(teacher).detach().float()
    observed = torch.as_tensor(observed, device=teacher.device).detach().float()
    context = torch.as_tensor(flow_context, device=teacher.device).detach().float()
    k, n, t, xy = teacher.shape
    if t != 12 or xy != 2 or observed.shape != (n,8,2) or context.shape != (n,128):
        raise ValueError('expected B[K,N,12,2], observations[N,8,2], context[N,128]')
    valid = (torch.ones(n, device=teacher.device, dtype=torch.bool) if valid is None else
             torch.as_tensor(valid, device=teacher.device, dtype=torch.bool))
    if valid.shape != (n,) or not valid.any():raise ValueError('empty/invalid mask')
    last = observed[:, -1]
    history = (observed-last[:, None]).reshape(n,16)
    geometry = (teacher[:, :, (0,5,11), :] - last[None,:,None,:]).reshape(k,n,6)
    sj = coverage['stats_j'].to(device=teacher.device, dtype=torch.float32).detach()
    sp = coverage['stats_p'].to(device=teacher.device, dtype=torch.float32).detach()
    a = coverage['mean_a_j'].to(device=teacher.device, dtype=torch.float32).detach()
    if sj.shape != (k,n,4) or sp.shape != sj.shape or a.shape != (k,n,1):
        raise ValueError('coverage feature dimensions changed')
    base = torch.cat((context[None].expand(k,-1,-1), history[None].expand(k,-1,-1), geometry, sj), -1)
    difference = torch.cat((sp-sj, a), -1)
    base = torch.where(valid[None,:,None], base, 0.)
    difference = torch.where(valid[None,:,None], difference, 0.)
    if not torch.isfinite(base).all() or not torch.isfinite(difference).all():
        raise ValueError('nonfinite frozen features')
    return dict(base_features=base.contiguous(), difference_features=difference.contiguous(), valid=valid)
