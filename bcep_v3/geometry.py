"""Observed-only physical reference paths, evaluated before assignment mixing."""
import torch


def reference_paths(context, goals):
    # [B,N,A,T,2]; exact candidate endpoint, never a barycentre endpoint.
    ratio = torch.arange(1, 13, device=goals.device, dtype=torch.float32) / 12
    base = context.baseline.float()
    g = goals.transpose(1, 2).float()
    return base[:, :, None] + ratio[None, None, None, :, None] * (
        g - base[:, :, -1, None])[:, :, :, None]


def pair_features(paths, observed, b, i, j, scale):
    """[E,A,C,12], for actual directed neighbour candidate pairs.

    Minimum synchronous distance/time, mean distance, relative endpoint,
    relative observed position/velocity, endpoint displacement mismatch, CPA.
    """
    yi, yj = paths[b, i, :, None], paths[b, j, None, :]
    rel = yj - yi
    distance = rel.square().sum(-1).add(1e-8).sqrt()
    minimum, time_index = distance.min(-1)
    last = observed[:, :, -1].float()
    vel = (last - observed[:, :, -4].float()) / 1.2
    pos = last[b, j] - last[b, i]
    dv = vel[b, j] - vel[b, i]
    k = paths.shape[2]
    end = rel[..., -1, :]
    # CPA uses each actual candidate pair's reference velocity, NOT a mixed
    # endpoint or a candidate-independent observed-velocity proxy.
    reference_dv = (end - pos[:, None, None]) / 4.8
    cpa_t = (-(pos[:, None, None] * reference_dv).sum(-1) /
             reference_dv.square().sum(-1).clamp_min(1e-6)).clamp(0, 4.8)
    cpa_d = (pos[:, None, None] + cpa_t[..., None] * reference_dv).square().sum(-1).add(1e-8).sqrt()
    static = torch.cat((pos / scale, dv), -1)[:, None, None].expand(-1, k, k, -1)
    mismatch = reference_dv.square().sum(-1).add(1e-8).sqrt()
    return torch.cat((minimum[..., None] / scale, (time_index[..., None] + 1) / 12,
                      distance.mean(-1, keepdim=True) / scale, end / scale,
                      mismatch[..., None], static, (cpa_t / 4.8)[..., None],
                      (cpa_d / scale)[..., None]), -1).clamp(-10, 10)
